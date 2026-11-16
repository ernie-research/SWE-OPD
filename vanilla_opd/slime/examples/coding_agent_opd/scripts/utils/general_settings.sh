#!/bin/bash
# ==============================================================================
# SHARED LAYER for the multi-node coding-agent OPD run-* scripts.
#
# BASELINE = run-qwen3.5-9B-coding-opd-fully_async-multinode-16tr-8te-8ro.sh.
# Every default here is that script's current value; the other layouts differ only
# by overriding vars.
#
# USAGE
#   SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
#   source "${SCRIPT_DIR}/utils/general_settings.sh"  # part 1: runs immediately
#   ... layout + tuning (node roles, GPU split, teacher, in-flight sizing, *_ARGS) ...
#   build_debug_args                                 # part 2: call when you need it
#   setup_proxy
#   build_runtime_env_json
#
# SCOPE RULE: this file holds ONLY what is constant across every layout AND never tuned
#   per experiment. Anything whose VALUE depends on the teacher spec, the rollout GPU
#   count, the batch width or the experiment's intent belongs in the run-* script. This
#   file may FORWARD such a value (e.g. list its name in RUNTIME_ENV_JSON) but must never
#   DECIDE it.
#
# PART 1 (executed on source): venv/PATH/RAY_LAUNCH, PDC topology, RUN_DIR + awk log
#   fan-out, NCCL/GLOO, cudnn/NVLink, SWE+RLE sandbox env, HTTP-timing + top-k-b64
#   protocol toggles. All vars are `${VAR:-default}`; override by exporting before launch,
#   or by plain assignment AFTER the source (part-2 functions read them at call time).
#
# PART 2 (functions, nothing runs until called):
#   preclean_ray             kill stale ray/train_async before `ray start`
#   dump_engine_logs         copy this node's ray/sglang engine logs into RUN_DIR
#   metrics_poller           background KV-cache/metrics scrape (needs METRICS_ENGINE_PORTS)
#   build_debug_args         DEBUG_ARGS dump policy (needs EXP_NAME/RUN_TAG)
#   setup_proxy              no_proxy/proxy exports (needs TEACHER_IP)
#   build_runtime_env_json   RUNTIME_ENV_JSON for `ray job submit` (call LAST)
#
# STAYS IN THE run-* SCRIPT (per-layout / per-experiment, deliberately NOT here):
#   * node-role selection (TEACHER_RANK / SPECIAL_RANK), RAY_GPUS / TEACHER_GPUS, the
#     actor/rollout GPU split, whether to pass --placement-group-head-ip
#   * the WHOLE teacher lifecycle: TEACHER_PORT/MODEL/TP/DP/MEM_FRACTION/EXTRA flags,
#     start_teacher + wait_teacher, and the OPD_TEACHER_* HTTP timeouts / concurrency
#     (OPD_TEACHER_CONCURRENCY tracks TEACHER_DP, so it must be exported there)
#   * in-flight sizing: ROLLOUT_GPUS_PER_ENGINE / INFLIGHT_STEPS ->
#     SGLANG_SERVER_CONCURRENCY / RLE container ceiling
#   * METRICS_ENGINE_PORTS (the port stride IS rollout_num_gpus_per_engine)
#   * MODEL/WANDB/CKPT/ROLLOUT/RM/PERF/OPD/OPTIMIZER/SGLANG/MISC *_ARGS, batch sizes
#   * the ray head/worker bring-up + `ray job submit` itself
#
# NOTE: `set -ex` below is inherited by the sourcing script (all run-* set it anyway).
# A fatal here (missing venv / missing RLE secret) `exit`s the caller on purpose.
# ==============================================================================
set -ex

# Dir of THIS file (for sibling tools like rollout_dashboard/). SCRIPT_DIR is left alone if
# the caller already set it; otherwise it is the caller's own dir (BASH_SOURCE[-1] = bottom
# of the source stack = the run-* script).
GENERAL_SETTINGS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_DIR="${SCRIPT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[-1]}")" &>/dev/null && pwd)}"
SLIME_DIR="${SLIME_DIR:-$(cd "${GENERAL_SETTINGS_DIR}/../../../.." && pwd)}"

# ============ python env: run EVERYTHING under slime-venv ============
# ray, sglang (teacher + rollout) and train_async.py must all use the venv interpreter.
# Prepending the venv's bin to PATH makes python3/`python3 -m sglang...` resolve to the venv.
SLIME_VENV="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}"
if [[ ! -x "${SLIME_VENV}/bin/python3" ]]; then
   echo "slime-venv python not found at ${SLIME_VENV}/bin/python3" >&2
   exit 1
fi
export PATH="${SLIME_VENV}/bin:${PATH}"
# Exported because build_runtime_env_json reads it from the ENVIRONMENT to pin the ray
# workers' interpreter (a plain shell var breaks a manual run that did not export it).
export SLIME_VENV

# Ray actors inherit the raylet's sys.executable, NOT $PATH: a bare `ray start` runs
# ${SLIME_VENV}/bin/ray, whose shebang is baked at venv-creation time (for a hardlink clone
# it still points at the ORIGINAL venv python). Always start ray THROUGH the venv python.
RAY_LAUNCH=("${SLIME_VENV}/bin/python3" "${SLIME_VENV}/bin/ray")
echo "Using python: $(command -v python3)  ray-launch: ${RAY_LAUNCH[*]}"

# ============ PDC cluster topology ============
# PADDLE_TRAINERS = comma-separated host/IP list; index 0 is the ray head node.
IFS=',' read -ra HOSTS <<< "${PADDLE_TRAINERS:-127.0.0.1}"
ALL_HOSTS="$(IFS=,; echo "${HOSTS[*]}")"
MASTER_ADDR="${MASTER_ADDR:-${HOSTS[0]}}"
RANK="${PADDLE_TRAINER_ID:-0}"
NNODES="${PADDLE_TRAINERS_NUM:-${#HOSTS[@]}}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
WAIT_WORKER_SECS="${WAIT_WORKER_SECS:-10}"
POD_IP="$(hostname -i | awk '{print $1}')"
export MASTER_ADDR

# ============ per-run output dir (per-rank log files) ============
# `date` differs per node, so for a SHARED log dir across ranks set RUN_TAG. EXP_NAME comes
# from the CALLER's filename (BASH_SOURCE[-1]), never from this file.
EXP_NAME="${EXP_NAME:-$(basename "${BASH_SOURCE[-1]}" .sh)}"
RUN_TAG="${RUN_TAG:-$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S)}" # 东八区
RUN_DIR="${RUN_DIR:-${SLIME_DIR}/logs/${EXP_NAME}_${RUN_TAG}}"
mkdir -p "${RUN_DIR}"

# Convenience: (re)point logs/latest-run at THIS run's dir.
ln -sfn "${RUN_DIR}" "${SLIME_DIR}/logs/latest-run"

# Cross-node NCCL/GLOO interface (the actor->rollout weight-sync group spans nodes).
# Default reuses whatever the platform already exported (bond0 on this cluster).
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME:-bond0}}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-${GLOO_SOCKET_IFNAME}}"
# Debug: one rollout engine can stall at "Init torch distributed begin" (NCCL init) and
# never become healthy, cascading into "Server process terminated unexpectedly". Turn on
# NCCL logging to capture the exact init failure. Override to "" to disable.
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,ENV}"

# Log fan-out (SINGLE awk dispatcher; reads the firehose once, appends per line):
#   run.rank             : the full firehose (everything).
#   rollout_manager.rank : ONLY RolloutManager actor lines.
#   sglang_engine.rank   : ONLY SGLangEngine actor lines.
#   megatron.rank        : ONLY MegatronTrainRayActor actor lines.
# Each Ray actor line carries its "(<ActorName> pid=NNN)" prefix (ANSI color codes precede
# the '('), so the literal "(<ActorName> pid=" is found by index() as a substring. Multi-line
# messages (prompt dumps, tracebacks) only match on their first line, so continuation lines
# land only in run.rank -- acceptable.
#
# WHY awk (not `tee >(grep) >(grep) ...`): single log lines are 10-40KB while a
# process-substitution pipe holds 64KB. When one grep filter fell behind, its pipe backed up
# and GNU tee (--output-error=warn-nopipe) PERMANENTLY dropped that output, so the per-role
# files froze mid-run while run.rank kept growing (seen on ..._24t-2t-6r_20260727_013607).
# One awk reads the stream once and appends to every target: no per-output SIGPIPE drop, and
# the producer backpressures uniformly. fflush() keeps `tail -f` live.
exec > >(awk \
      -v run="${RUN_DIR}/run.rank${RANK}.log" \
      -v rollout="${RUN_DIR}/rollout_manager.rank${RANK}.log" \
      -v sglang="${RUN_DIR}/sglang_engine.rank${RANK}.log" \
      -v megatron="${RUN_DIR}/megatron.rank${RANK}.log"  '
   {
      print $0                                                     # -> terminal (awk stdout = original tty)
      print $0 >> run
      if (index($0, "(RolloutManager pid="))        print $0 >> rollout
      if (index($0, "(SGLangEngine pid="))          print $0 >> sglang
      if (index($0, "(MegatronTrainRayActor pid="))  print $0 >> megatron
      fflush()
   }') 2>&1
echo "RUN_DIR=${RUN_DIR} RANK=${RANK} NNODES=${NNODES} MASTER_ADDR=${MASTER_ADDR} POD_IP=${POD_IP}"

export PYTHONUNBUFFERED=1

# ============ cudnn load-order workaround ============
# transformer_engine dlopens libcudnn_*.so.9 in glob order and can hit
# libcudnn_engines_precompiled.so.9 before libcudnn_graph.so.9 -> undefined symbol. Pointing
# CUDNN_PATH at the wheel's cudnn dir makes TE load the dispatcher, which pulls the sublibs
# in the right DT_NEEDED order.
if [[ -z "${CUDNN_PATH:-}" ]]; then
   CUDNN_PATH="$(python3 -c 'import sysconfig,os;print(os.path.join(sysconfig.get_path("purelib"),"nvidia/cudnn"))' 2>/dev/null)"
   [[ -d "${CUDNN_PATH}" ]] && export CUDNN_PATH || unset CUDNN_PATH
fi

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# ============ SWE + RLE sandbox knobs ============
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-rle}"
# RLE_ENV / RLE_TOOL_TYPE / RLE_API_KEY come from an untracked secret file, never from a
# tracked script. Anything already exported in the env wins over the file.
RLE_SECRET_FILE="${RLE_SECRET_FILE:-${SLIME_DIR}/.secret/.key.sh}"
source "${RLE_SECRET_FILE}" || { echo "FATAL: RLE secret file not found: ${RLE_SECRET_FILE}" >&2; exit 1; }
export RLE_SESSION_IDS_DIR="${RLE_SESSION_IDS_DIR:-${SLIME_DIR}/rle_sessions}"
export RLE_SESSION_TIMEOUT="${RLE_SESSION_TIMEOUT:-3600}"   # 秒
export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-scaleswe}"
export SWE_EVAL_PROTOCOL="${SWE_EVAL_PROTOCOL:-scaleswe}"
export SLIME_AGENT_MAX_STEPS="${SLIME_AGENT_MAX_STEPS:-100}"
export SLIME_AGENT_TOOL_TIMEOUT_SEC="${SLIME_AGENT_TOOL_TIMEOUT_SEC:-90}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-120}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-16}"

# The teacher lifecycle (TEACHER_PORT/MODEL/TP/DP/MEM_FRACTION/EXTRA flags, start_teacher,
# wait_teacher) and the OPD_TEACHER_* HTTP timeouts / concurrency live in the run-* script:
# every one of those values is tuned per teacher spec. setup_proxy and RUNTIME_ENV_JSON
# below only FORWARD TEACHER_IP / OPD_TEACHER_* if the run script set them.

# ============ per-POST phase timing (diagnostic, default OFF) ============
# SLIME_HTTP_TIMING=1 logs a phase breakdown every SLIME_HTTP_TIMING_INTERVAL_SEC
# (slime/utils/http_utils.py). Use it before optimizing the per-turn context re-send.
export SLIME_HTTP_TIMING="${SLIME_HTTP_TIMING:-0}"
export SLIME_HTTP_TIMING_INTERVAL_SEC="${SLIME_HTTP_TIMING_INTERVAL_SEC:-60}"

# ============ rollout engine: compact top-k response (default ON) ============
# Returns the [T, K] student top-k ids as ONE base64 int32 blob instead of a JSON int list
# (generate.py handles both). At OPD_ID_TOPK=1024 the JSON path is ~10x the body + parse cost.
export SGLANG_TOP_LOGPROBS_IDS_B64="${SGLANG_TOP_LOGPROBS_IDS_B64:-true}"

# ============ sglang metrics/observability master switch + KV poller ============
# ENABLE_SGLANG_METRICS gates BOTH the --sglang-enable-metrics* CLI flags (SGLANG_METRICS_ARGS,
# built in the run script) and the poller below.
# ENABLE_TEACHER_METRICS gates metrics flags passed to the TEACHER sglang server (native args,
# no --sglang- prefix). Defaults to ENABLE_SGLANG_METRICS so a single switch controls both.
# WHY the poller: sglang's scheduler stdout carries `#cached-token` but NOT a cache-hit RATE;
# the rate lives only on each engine's in-memory /metrics endpoint, so it vanishes when ray
# stops. kv_dashboard.sh re-discovers the rollout engines each interval (hosts from
# PADDLE_TRAINERS, ports via `ss` + a /metrics confirm) and prints a per-engine table of
# cumulative hit rate / KV usage / new_token_ratio. Running it as an external process also
# keeps `set -x` xtrace from dumping raw Prometheus text into the log.
#   METRICS_POLL_INTERVAL : seconds between scrapes (0 disables the poller)
#   METRICS_ENGINE_PORTS  : ss-discovery FALLBACK list. NOT defaulted here: slime starts at
#                           15000 and steps by rollout_num_gpus_per_engine, so the list is a
#                           function of the rollout layout -> the run-* script sets it.
ENABLE_SGLANG_METRICS="${ENABLE_SGLANG_METRICS:-1}"
ENABLE_TEACHER_METRICS="${ENABLE_TEACHER_METRICS:-1}"
METRICS_POLL_INTERVAL="${METRICS_POLL_INTERVAL:-120}"
METRICS_LOG="${RUN_DIR}/sglang_metrics.rank${RANK}.log"
KV_DASHBOARD="${KV_DASHBOARD:-${GENERAL_SETTINGS_DIR}/rollout_dashboard/kv_dashboard.sh}"

# ==============================================================================
# PART 2: helpers. Nothing below runs until the run script calls it.
# ==============================================================================

metrics_poller() {
   set +x
   PORTS="${METRICS_ENGINE_PORTS:-}" bash "${KV_DASHBOARD}" --watch "${METRICS_POLL_INTERVAL}"
}

# ---- best-effort local cleanup BEFORE `ray start` ----
# A stale ray head/GCS in /tmp/ray from a prior run makes `ray start` fail with a session-name
# mismatch. NOT auto-invoked: sourcing a settings file must never kill a live cluster. Does
# NOT touch an already-serving teacher (the run script's start_teacher reuses it).
preclean_ray() {
   pkill -9 -f "train_async.py" || true
   ray stop --force || true
   pkill -9 ray || true
   sleep 3
}

# ---- on-failure post-mortem: preserve this node's ray/sglang engine logs ----
# The rollout engines' NCCL/torch-dist init output lives in /tmp/ray/session_*/logs, which the
# next `ray stop`/launch wipes. Copy it into RUN_DIR (shared FS) so a failed init can be
# diagnosed after the fact. Never fails the run itself.
dump_engine_logs() {
   set +e
   local dest="${RUN_DIR}/engine_logs/${RANK}_$(hostname)_${POD_IP}"
   mkdir -p "${dest}"
   local s
   s="$(ls -dt /tmp/ray/session_*/logs 2>/dev/null | head -1)"
   echo "[dump] node RANK=${RANK} ip=${POD_IP} -> ${dest} (session=${s})"
   nvidia-smi > "${dest}/nvidia-smi.txt" 2>&1 || true
   if [[ -n "${s}" && -d "${s}" ]]; then
      cp "${s}"/worker-*.err "${s}"/worker-*.out "${dest}/" 2>/dev/null || true
      # job-driver-*.log = train.py 的 driver 输出（ray job 的日志源）；dashboard*/agent 日志里
      # 记录 job 状态迁移和日志流断开的原因。cleanup.sh 会 rm -rf /tmp/ray/*，不先拷出来就永久丢失
      # —— 一次 "submit 返回 0 但 job 还在跑" 的事故就是因此无法定位（run_20260803_140107）。
      cp "${s}"/job-driver-*.log "${dest}/" 2>/dev/null || true
      cp "${s}"/dashboard.log "${s}"/dashboard.err "${s}"/dashboard_agent.log "${dest}/" 2>/dev/null || true
      tail -n 400 "${s}/raylet.out" > "${dest}/raylet.tail.txt" 2>/dev/null || true
   fi
   set -e
}

# ---- dump policy -> DEBUG_ARGS (appends; needs EXP_NAME/RUN_TAG) ----
# DUMP_DETAILS=1 writes under ${TRAJ_ROOT}/${EXP_NAME}_${RUN_TAG}, kept OUT of RUN_DIR so logs
# stay small and trajectories can be swept independently. DUMP_DETAILS_NUM_ROLLOUTS=k caps it
# to the FIRST k rollouts. Inspect with visualize_rollouts.py.
# train_data dumps are OFF by default: with --opd-id-topk they are written once per DP rank
# and filled the local disk. DUMP_TRAIN_DATA=1 re-enables them via --dump-details, which
# also writes dumps/tokenizer for the trace viewer.
build_debug_args() {
   [[ -z "${DEBUG_ARGS+x}" ]] && DEBUG_ARGS=()
   [[ "${DUMP_DETAILS:-1}" -ne 1 ]] && return 0
   DUMP_DETAILS_NUM_ROLLOUTS=${DUMP_DETAILS_NUM_ROLLOUTS:-5}
   TRAJ_ROOT="${TRAJ_ROOT:-$(cd "${SLIME_DIR}/.." && pwd)/traj}"
   DUMP_DETAILS_DIR="${DUMP_DETAILS_DIR:-${TRAJ_ROOT}/${EXP_NAME}_${RUN_TAG}}"
   if [[ "${DUMP_TRAIN_DATA:-0}" -eq 1 ]]; then
      DEBUG_ARGS+=(--dump-details "${DUMP_DETAILS_DIR}")
   else
      DEBUG_ARGS+=(--save-debug-rollout-data "${DUMP_DETAILS_DIR}/rollout_data/{rollout_id}.pt")
   fi
   if [[ -n "${DUMP_DETAILS_NUM_ROLLOUTS}" ]]; then
      DEBUG_ARGS+=(--dump-details-num-rollouts "${DUMP_DETAILS_NUM_ROLLOUTS}")
   fi
}

# ---- proxy (call AFTER TEACHER_IP is known) ----
# swebench's test-spec builder does an UNTIMED requests.get to raw.githubusercontent.com while
# creating the SWE grading spec. github raw is NOT directly reachable from this cluster, so
# that call hangs forever and freezes the whole rollout. Route http(s) through the
# github-reaching proxy and bypass it for everything in-cluster: the teacher, ALL trainer pods
# (so cross-node ray / weight-sync / teacher HTTP go direct), the sglang rollout engines, and
# the RLE prod hosts (.baidu*.com + prod gateway IPs). NO_PROXY_EXTRA adds layout-specific
# hosts (comma-separated, no leading comma needed).
setup_proxy() {
   export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${TEACHER_IP:-},${POD_IP},${ALL_HOSTS},${NO_PROXY_EXTRA:+${NO_PROXY_EXTRA},}10.0.0.18,10.0.0.19,.example.com,.internal.example.com,.baidubce.com,${no_proxy:-}"
   export NO_PROXY="${no_proxy}"
   GITHUB_PROXY="${GITHUB_PROXY:-http://proxy.example.com:8188}"
   export HTTPS_PROXY="${GITHUB_PROXY}"
   export HTTP_PROXY="${GITHUB_PROXY}"
   export https_proxy="${GITHUB_PROXY}"
   export http_proxy="${GITHUB_PROXY}"
}

# ---- RUNTIME_ENV_JSON for `ray job submit` (call LAST, after setup_proxy) ----
# Ray actors do NOT inherit this shell's env: MASTER_ADDR + the socket ifnames must be passed
# so the cross-node NCCL/GLOO groups bind to the right interface, and PATH/SLIME_VENV pin the
# ray workers to the slime-venv interpreter.
# This only FORWARDS whatever is exported: the OPD_TEACHER_* trio (timeouts +
# OPD_TEACHER_CONCURRENCY, which must track the run script's TEACHER_DP) is exported by the
# run-* script, and a key that is unset is simply skipped.
build_runtime_env_json() {
   export SLIME_DIR
   RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "MASTER_ADDR",
    "GLOO_SOCKET_IFNAME", "NCCL_SOCKET_IFNAME",
    "NCCL_DEBUG", "NCCL_DEBUG_SUBSYS",
    "SLIME_AGENT_SANDBOX_BACKEND",
    "RLE_ENV", "RLE_TOOL_TYPE", "RLE_API_KEY", "RLE_SESSION_IDS_DIR", "RLE_SESSION_TIMEOUT",
    "SLIME_AGENT_RLE_IMAGE_PREFIX",
    "SWE_TRAIN_PROTOCOL", "SWE_EVAL_PROTOCOL",
    "SLIME_AGENT_MAX_STEPS", "SLIME_AGENT_TOOL_TIMEOUT_SEC",
    "SWE_EVAL_TIMEOUT_SEC", "SWE_BOOT_CONCURRENCY",
    # Only FORWARDED (unset keys are skipped), decided per layout:
    #   SWE_ROLLOUT_GUARD_SEC        per-sample rollout wall-clock guard (utils/rollout.py)
    #   CODING_OPD_MAX_ABORTED_GROUPS cumulative dropped-group budget of the dynamic-sampling
    #                                 filter (utils/filters.py); only the SYNC rollout path
    #                                 runs that filter, so the fully-async layouts leave it unset.
    "SWE_ROLLOUT_GUARD_SEC", "CODING_OPD_MAX_ABORTED_GROUPS",
    "OPD_TEACHER_TIMEOUT_SEC", "OPD_TEACHER_CONNECT_TIMEOUT_SEC", "OPD_TEACHER_CONCURRENCY",
    "SLIME_HTTP_TIMING", "SLIME_HTTP_TIMING_INTERVAL_SEC",
    "SGLANG_TOP_LOGPROBS_IDS_B64",
    "CUDNN_PATH",
    # Allocator knob (e.g. expandable_segments:True). Only FORWARDED: the layouts that need
    # it export it themselves; unset keys are skipped, so other layouts are unaffected.
    # Must go through here -- the megatron actor runs inside a ray worker, which does NOT
    # inherit the launcher shell's env.
    "PYTORCH_CUDA_ALLOC_CONF",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["TP_SOCKET_IFNAME"] = os.environ["GLOO_SOCKET_IFNAME"]
env["PYTHONPATH"] = f"{os.environ.get('MEGATRON_DIR') or os.environ.get('MEGATRON_LM') or os.environ.get('GOLDEN_OPD_DIR', '/path/to/GOLDEN_OPD_DIR') + '/Megatron-LM'}:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["PATH"] = os.environ["PATH"]
env["SLIME_VENV"] = os.environ["SLIME_VENV"]
print(json.dumps({"env_vars": env}))
PY
)
}
