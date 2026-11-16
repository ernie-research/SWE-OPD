#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"
SLIME_VENV="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}"
export PATH="${SLIME_VENV}/bin:${PATH}"
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_LM_PATH:-/root/Megatron-LM}${PYTHONPATH:+:${PYTHONPATH}}"
# DSV4's sglang attention backend imports flash_mla on every prefill. .slime_venv has
# only sgl_kernel.flash_mla (different API) and flash_mla is source-only, so
# DSV4_FLASH_MLA_SITE holds a local build (cp312 / sm_90a / cudart 12). Added to
# PYTHONPATH here on every rank, since mpirun does not forward PYTHONPATH.
DSV4_FLASH_MLA_SITE="${DSV4_FLASH_MLA_SITE:-/path/to/<user>/<user>/dsv4_glm45air_opd_probe/pysite}"
if [[ -f "${DSV4_FLASH_MLA_SITE}/flash_mla/flash_mla_interface.py" ]]; then
   export PYTHONPATH="${PYTHONPATH}:${DSV4_FLASH_MLA_SITE}"
else
   echo "flash_mla site missing: ${DSV4_FLASH_MLA_SITE}" >&2
   exit 1
fi

RANK="${PADDLE_TRAINER_ID:?}"
WORLD="${PADDLE_TRAINERS_NUM:?}"
TEACHER_NNODES="${TEACHER_NNODES:-2}"
TEACHER_REPLICAS="${TEACHER_REPLICAS:-2}"
TEACHER_TP="${TEACHER_TP:-16}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
if (( TEACHER_NNODES * TEACHER_REPLICAS != WORLD )); then
   echo "teacher groups ${TEACHER_REPLICAS}x${TEACHER_NNODES} do not cover ${WORLD} nodes" >&2
   exit 1
fi
if (( TEACHER_TP % TEACHER_NNODES != 0 || TEACHER_TP / TEACHER_NNODES != 8 )); then
   echo "this launcher requires eight GPUs/node: TP=${TEACHER_TP}, nnodes=${TEACHER_NNODES}" >&2
   exit 1
fi
IFS=',' read -ra TRAINER_HOSTS <<< "${PADDLE_TRAINERS:?}"
if (( ${#TRAINER_HOSTS[@]} != WORLD )); then
   echo "PADDLE_TRAINERS contains ${#TRAINER_HOSTS[@]} hosts, expected ${WORLD}" >&2
   exit 1
fi

GROUP_ID=$(( RANK / TEACHER_NNODES ))
NODE_RANK=$(( RANK % TEACHER_NNODES ))
LEADER_RANK=$(( GROUP_ID * TEACHER_NNODES ))
LEADER_IP="${TRAINER_HOSTS[LEADER_RANK]}"
DIST_PORT=$(( ${TEACHER_NCCL_PORT_BASE:-23000} + GROUP_ID * 100 ))
RUN_DIR="${RUN_DIR:-${RUN_ROOT:?}/logs-phase2}"
# Remote ranks get RUN_DIR but not always RUN_ROOT; recover it so every rank sees the
# same group markers.
if [[ -z "${RUN_ROOT:-}" ]]; then
   case "${RUN_DIR}" in
      */logs-phase2) RUN_ROOT="${RUN_DIR%/logs-phase2}" ;;
      *) echo "cannot recover RUN_ROOT from unexpected RUN_DIR=${RUN_DIR}" >&2; exit 1 ;;
   esac
fi
export RUN_ROOT
mkdir -p "${RUN_DIR}"
TEACHER_LOG_BASE="${RUN_DIR}/phase2.teacher.group${GROUP_ID}.node${NODE_RANK}.rank${RANK}"
TEACHER_LOG="${TEACHER_LOG_BASE}.log"
GROUP_DONE="${RUN_ROOT}/.phase2_teacher_group${GROUP_ID}.done"
GROUP_FAILED="${RUN_ROOT}/.phase2_teacher_group${GROUP_ID}.failed"
# The group leader owns the attempt counter; secondary ranks relaunch their half of the
# teacher whenever it advances, so one crashed scheduler does not abort the job (mpirun
# kills everything on the first non-zero rank). 02-phase2-teacher.sh clears it.
ATTEMPT_FILE="${RUN_ROOT}/.phase2_teacher_group${GROUP_ID}.attempt"
if (( NODE_RANK == 0 )); then
   rm -f "${GROUP_DONE}" "${GROUP_FAILED}"
fi

TEACHER_PID=""
cleanup() {
   if [[ -n "${TEACHER_PID}" ]] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
      kill -TERM -- "-${TEACHER_PID}" 2>/dev/null || true
      for _ in {1..30}; do
         kill -0 "${TEACHER_PID}" 2>/dev/null || break
         sleep 1
      done
      kill -KILL -- "-${TEACHER_PID}" 2>/dev/null || true
      wait "${TEACHER_PID}" 2>/dev/null || true
   fi
}
trap cleanup EXIT INT TERM

export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-1}"
export SGLANG_OPD_TOPK_OVERLAP=1
export SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS="${SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS:-1}"
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE="${TEACHER_LOGITS_CHUNK:-256}"
export no_proxy="127.0.0.1,localhost,${LEADER_IP},${no_proxy:-}"
export NO_PROXY="${no_proxy}"
# Diagnostic: synchronous launches, so an asynchronous illegal-instruction error names
# its kernel.
if [[ "${TEACHER_CUDA_LAUNCH_BLOCKING:-0}" == 1 ]]; then
   export CUDA_LAUNCH_BLOCKING=1
   export TORCH_SHOW_CPP_STACKTRACES=1
fi

# Diagnostic: torch reference paths for DSV4's optimized hash-routing kernels, the
# suspected source of out-of-range expert ids behind an earlier illegal instruction
# (the fused_moe kernel itself passed a standalone probe). MHC_PRE must be off with
# PRENORM off, or mhc_pre reads n_splits before assignment.
if [[ "${TEACHER_DSV4_TORCH_ROUTING:-0}" == 1 ]]; then
   export SGLANG_OPT_USE_FUSED_HASH_TOPK=0
   export SGLANG_OPT_USE_TILELANG_MHC_PRE=0
   export SGLANG_OPT_USE_TILELANG_MHC_POST=0
   export SGLANG_OPT_DEEPGEMM_HC_PRENORM=0
fi

TEACHER_ATTENTION_MODE="${TEACHER_ATTENTION_MODE:-dsa}"
# The job image's CUDA 12.8 cannot compile sglang's DSV4 topk_v2 JIT kernel; v1
# computes the same top-512 selection. Set here because mpirun forwards no SGLANG_*.
export SGLANG_OPT_USE_TOPK_V2="${SGLANG_OPT_USE_TOPK_V2:-0}"
# DP attention with attn_tp = 1 (dp_size == tp_size):
#   * DSV4 splits the attention output projection into o_groups=8 // attn_tp groups, so
#     TP=16 without DP attention gives 0 groups and fails on the first prefill. TP<=8 on
#     one node would need ~70 GiB/GPU for the ~563 GiB checkpoint, too much.
#   * Any attn_tp > 1 makes the attention output an allreduce, which DSV4 refuses to
#     short-circuit for the idle zero-token batches DP attention hands out; that hangs
#     the warmup. At attn_tp=1 the attention weights are replicated (+~8 GiB/GPU).
# sglang clamps chunked prefill to 8192 under DP attention.
TEACHER_ATTN_DP="${TEACHER_ATTN_DP:-${TEACHER_TP}}"
(( TEACHER_ATTN_DP >= 1 )) || TEACHER_ATTN_DP=1
(( TEACHER_TP % TEACHER_ATTN_DP == 0 && 8 % (TEACHER_TP / TEACHER_ATTN_DP) == 0 )) || {
   echo "attn_tp=$(( TEACHER_TP / TEACHER_ATTN_DP )) does not divide o_groups=8: TP=${TEACHER_TP}, dp=${TEACHER_ATTN_DP}" >&2
   exit 1
}
TEACHER_DP_ARGS=(--dp-size "${TEACHER_ATTN_DP}")
if (( TEACHER_ATTN_DP > 1 )); then
   TEACHER_DP_ARGS+=(--enable-dp-attention)
fi
TEACHER_MODE_ARGS=()
TEACHER_LOAD_ARGS=()
if [[ -n "${TEACHER_MODEL_LOADER_THREADS:-}" ]]; then
   [[ "${TEACHER_MODEL_LOADER_THREADS}" =~ ^[1-9][0-9]*$ ]] || {
      echo "TEACHER_MODEL_LOADER_THREADS must be a positive integer: ${TEACHER_MODEL_LOADER_THREADS}" >&2
      exit 1
   }
   TEACHER_LOAD_ARGS+=(
      --model-loader-extra-config
      "{\"num_threads\": ${TEACHER_MODEL_LOADER_THREADS}}"
   )
fi
# 0731's routed experts are MXFP4. On SM90 sglang's "auto" MoE backend picks the Triton
# path, which fails with "Hidden size mismatch" on the packed weights; Marlin is the
# Hopper path for this layout. Defaulted here because mpirun does not forward it.
TEACHER_MOE_RUNNER_BACKEND="${TEACHER_MOE_RUNNER_BACKEND:-marlin}"
if [[ -n "${TEACHER_MOE_RUNNER_BACKEND:-}" ]]; then
   TEACHER_LOAD_ARGS+=(--moe-runner-backend "${TEACHER_MOE_RUNNER_BACKEND}")
fi
case "${TEACHER_ATTENTION_MODE}" in
   dsa) ;;
   dense)
      # Pre-rename NSA names (newer sglang: SGLANG_DSA_*, NSA kept as an alias).
      export SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD="${TEACHER_DENSE_ATTN_KV_LEN_THRESHOLD:-131072}"
      export SGLANG_MAX_KV_CHUNK_CAPACITY="${TEACHER_MAX_KV_CHUNK_CAPACITY:-262144}"
      export SGLANG_NSA_SPARSE_PREFILL_Q_CHUNK="${TEACHER_NSA_SPARSE_PREFILL_Q_CHUNK:-8192}"
      export SGLANG_NSA_MQA_LOGITS_MAX_ROWS="${TEACHER_NSA_MQA_LOGITS_MAX_ROWS:-4096}"
      export SGLANG_MOE_TOKEN_CHUNK_SIZE="${TEACHER_MOE_TOKEN_CHUNK_SIZE:-16384}"
      TEACHER_MODE_ARGS+=(
         --kv-cache-dtype "${TEACHER_KV_CACHE_DTYPE:-bf16}"
         --max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-1}"
      )
      ;;
   *)
      echo "TEACHER_ATTENTION_MODE must be dsa or dense: ${TEACHER_ATTENTION_MODE}" >&2
      exit 1
      ;;
esac

echo "teacher attention mode=${TEACHER_ATTENTION_MODE}, tp=${TEACHER_TP}, attn_dp=${TEACHER_ATTN_DP} (attn_tp=$(( TEACHER_TP / TEACHER_ATTN_DP ))), chunked_prefill=${TEACHER_CHUNKED_PREFILL_SIZE:-16384}, max_prefill=${TEACHER_MAX_PREFILL_TOKENS:-16384}, score_concurrency=${SCORE_CONCURRENCY:-2}, nsa_rows=${TEACHER_NSA_MQA_LOGITS_MAX_ROWS:-upstream}, moe_chunk=${TEACHER_MOE_TOKEN_CHUNK_SIZE:-upstream}"

# Optional: one ordered read per host into the page cache, so the per-rank checkpoint
# reads that follow hit local memory instead of contending on the shared filesystem.
if [[ "${TEACHER_PREWARM_WEIGHTS:-0}" == 1 ]]; then
   [[ -d "${TEACHER_MODEL}" ]] || {
      echo "cannot prewarm non-local teacher model: ${TEACHER_MODEL}" >&2
      exit 1
   }
   mapfile -d '' TEACHER_WEIGHT_FILES < <(
      find "${TEACHER_MODEL}" -maxdepth 1 -type f -name '*.safetensors' -print0 | sort -z
   )
   (( ${#TEACHER_WEIGHT_FILES[@]} > 0 )) || {
      echo "no safetensors files found for prewarm: ${TEACHER_MODEL}" >&2
      exit 1
   }
   prewarm_concurrency="${TEACHER_PREWARM_CONCURRENCY:-1}"
   [[ "${prewarm_concurrency}" =~ ^[1-9][0-9]*$ ]] || {
      echo "TEACHER_PREWARM_CONCURRENCY must be a positive integer: ${prewarm_concurrency}" >&2
      exit 1
   }
   prewarm_start="$(date +%s)"
   echo "prewarming ${#TEACHER_WEIGHT_FILES[@]} teacher shards into local page cache with ${prewarm_concurrency} readers"
   printf '%s\0' "${TEACHER_WEIGHT_FILES[@]}" | \
      xargs -0 -r -n 1 -P "${prewarm_concurrency}" sh -c 'cat -- "$1" > /dev/null' _
   echo "teacher page-cache prewarm complete in $(( $(date +%s) - prewarm_start ))s"
fi

# The teacher runs under setsid, so a crashed attempt can leave a live launch_server
# holding its ports. Match only our venv interpreter plus this model path, so no other
# tenant's or experiment's process is hit.
kill_stale_teachers() {
   local stale_pids=() proc pid cmdline alive
   for proc in /proc/[0-9]*; do
      pid="${proc#/proc/}"
      [[ "${pid}" == "$$" ]] && continue
      cmdline="$(tr '\0' ' ' < "${proc}/cmdline" 2>/dev/null)" || continue
      [[ "${cmdline}" == *"${SLIME_VENV}/bin/python3"* ]] || continue
      [[ "${cmdline}" == *sglang.launch_server* ]] || continue
      [[ "${cmdline}" == *"${TEACHER_MODEL}"* ]] || continue
      stale_pids+=("${pid}")
   done
   if (( ${#stale_pids[@]} > 0 )); then
      echo "killing ${#stale_pids[@]} stale teacher process(es) from a previous attempt: ${stale_pids[*]}"
      kill -TERM "${stale_pids[@]}" 2>/dev/null || true
      for _ in $(seq 30); do
         alive=0
         for pid in "${stale_pids[@]}"; do
            kill -0 "${pid}" 2>/dev/null && alive=1
         done
         (( alive == 0 )) && break
         sleep 2
      done
      for pid in "${stale_pids[@]}"; do
         kill -0 "${pid}" 2>/dev/null && kill -KILL "${pid}" 2>/dev/null
      done
   fi
   # sglang's scheduler/detokenizer are multiprocessing-spawn children whose cmdline
   # lacks sglang.launch_server, so the loop above misses them once the parent is dead,
   # and they keep the rendezvous port. Any process from our venv holding an nvidia fd
   # here is an orphan (this attempt's teacher is not started yet and Phases 1/3 never
   # overlap Phase 2), so this scan is unconditional.
   local orphans=0
   for proc in /proc/[0-9]*; do
      pid="${proc#/proc/}"
      [[ "${pid}" == "$$" ]] && continue
      cmdline="$(tr '\0' ' ' < "${proc}/cmdline" 2>/dev/null)" || continue
      [[ "${cmdline}" == *"${SLIME_VENV}/bin/python3"* ]] || continue
      ls -l "${proc}/fd" 2>/dev/null | grep -q nvidia || continue
      echo "killing orphaned teacher worker pid=${pid}"
      kill -KILL "${pid}" 2>/dev/null || true
      orphans=$(( orphans + 1 ))
   done
   if (( ${#stale_pids[@]} > 0 || orphans > 0 )); then
      sleep 5
   fi
}

# Sockets take a moment to be released; fail here rather than with EADDRINUSE deep
# into a 16-rank rendezvous.
wait_for_ports() {
   local port attempt
   for port in "${DIST_PORT}" "${TEACHER_PORT}"; do
      for attempt in $(seq 30); do
         ss -ltnH "sport = :${port}" 2>/dev/null | grep -q . || break
         if (( attempt == 30 )); then
            echo "port ${port} still in use before launch: rank=${RANK}" >&2
            ss -ltnp "sport = :${port}" >&2 || true
            return 1
         fi
         echo "waiting for port ${port} to be released (${attempt}/30)"
         sleep 2
      done
   done
   return 0
}

start_teacher() {
   local attempt="$1"
   TEACHER_LOG="${TEACHER_LOG_BASE}.a${attempt}.log"
   : > "${TEACHER_LOG}"
   setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      "${SLIME_VENV}/bin/python3" -m sglang.launch_server \
      --model-path "${TEACHER_MODEL:?}" \
      --host 0.0.0.0 \
      --port "${TEACHER_PORT}" \
      --tp "${TEACHER_TP}" \
      "${TEACHER_DP_ARGS[@]}" \
      --nnodes "${TEACHER_NNODES}" \
      --node-rank "${NODE_RANK}" \
      --dist-init-addr "${LEADER_IP}:${DIST_PORT}" \
      --trust-remote-code \
      --disable-cuda-graph \
      --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-16384}" \
      --disable-radix-cache \
      --max-prefill-tokens "${TEACHER_MAX_PREFILL_TOKENS:-16384}" \
      --watchdog-timeout "${TEACHER_WATCHDOG_TIMEOUT:-7200}" \
      --mem-fraction-static "${TEACHER_MEM_FRACTION:-0.77}" \
      "${TEACHER_LOAD_ARGS[@]}" \
      "${TEACHER_MODE_ARGS[@]}" \
      > "${TEACHER_LOG}" 2>&1 &
   TEACHER_PID=$!
   echo "teacher group=${GROUP_ID} node=${NODE_RANK} attempt=${attempt} pid=${TEACHER_PID} log=${TEACHER_LOG}"
}

stop_teacher() {
   cleanup
   TEACHER_PID=""
}

read_attempt() {
   local value
   value="$(cat "${ATTEMPT_FILE}" 2>/dev/null || true)"
   [[ "${value}" =~ ^[0-9]+$ ]] || value=0
   printf '%s' "${value}"
}

wait_for_health() {
   local waited=0 timeout="${TEACHER_WAIT_TIMEOUT_SEC:-7200}"
   until curl --noproxy '*' -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; do
      if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
         echo "teacher exited before health: rank=${RANK}" >&2
         tail -n 120 "${TEACHER_LOG}" >&2 || true
         return 1
      fi
      if (( waited >= timeout )); then
         echo "teacher health timeout=${timeout} rank=${RANK}" >&2
         tail -n 120 "${TEACHER_LOG}" >&2 || true
         return 1
      fi
      sleep 5
      waited=$(( waited + 5 ))
   done
   echo "teacher group ${GROUP_ID} healthy after ${waited}s"
   return 0
}

TEACHER_LAUNCH_ATTEMPTS="${TEACHER_LAUNCH_ATTEMPTS:-3}"
[[ "${TEACHER_LAUNCH_ATTEMPTS}" =~ ^[1-9][0-9]*$ ]] || {
   echo "TEACHER_LAUNCH_ATTEMPTS must be a positive integer: ${TEACHER_LAUNCH_ATTEMPTS}" >&2
   exit 1
}

if (( NODE_RANK == 0 )); then
   rc=1
   for (( attempt=1; attempt<=TEACHER_LAUNCH_ATTEMPTS; attempt++ )); do
      # Announce first, then clean up: the announcement makes secondary ranks (polling
      # every 5 s) tear down their half before the new rendezvous binds DIST_PORT.
      printf '%s\n' "${attempt}" > "${ATTEMPT_FILE}.tmp.${RANK}"
      mv -f "${ATTEMPT_FILE}.tmp.${RANK}" "${ATTEMPT_FILE}"
      stop_teacher
      kill_stale_teachers
      if ! wait_for_ports; then
         echo "group ${GROUP_ID} attempt ${attempt}: ports never released" >&2
         continue
      fi
      start_teacher "${attempt}"
      if ! wait_for_health; then
         echo "group ${GROUP_ID} attempt ${attempt}: teacher never became healthy" >&2
         continue
      fi
      set +e
      SKIP_SERVE=1 TEACHER_IP=127.0.0.1 TEACHER_PORT="${TEACHER_PORT}" \
         SCORE_SHARD_INDEX="${GROUP_ID}" SCORE_NUM_SHARDS="${TEACHER_REPLICAS}" \
         bash "${OFFLINE_OPD_DIR}/run_phase2.sh"
      rc=$?
      set -e
      if (( rc == 0 )); then
         break
      fi
      echo "group ${GROUP_ID} attempt ${attempt}: scorer exited rc=${rc}" >&2
      # Relaunching is safe (scoring resumes from per-sample artifacts), but a scorer
      # that failed while its teacher still serves did not fail for a launch reason;
      # reloading the weights would only repeat the error.
      if kill -0 "${TEACHER_PID}" 2>/dev/null; then
         echo "group ${GROUP_ID}: teacher still alive, not relaunching" >&2
         break
      fi
   done
   if (( rc == 0 )); then
      printf 'completed=%s leader=%s\n' "$(date -Is)" "${RANK}" > "${GROUP_DONE}"
   else
      printf 'rc=%s failed=%s leader=%s\n' "${rc}" "$(date -Is)" "${RANK}" > "${GROUP_FAILED}"
      exit "${rc}"
   fi
else
   total_timeout="${TEACHER_PHASE2_WAIT_TIMEOUT_SEC:-172800}"
   waited=0
   seen=0
   while :; do
      # Wait for the leader to announce a new attempt.
      while :; do
         if [[ -f "${GROUP_DONE}" ]]; then
            stop_teacher
            exit 0
         fi
         if [[ -f "${GROUP_FAILED}" ]]; then
            cat "${GROUP_FAILED}" >&2
            stop_teacher
            exit 1
         fi
         cur="$(read_attempt)"
         if (( cur > seen )); then
            break
         fi
         if (( waited >= total_timeout )); then
            echo "timed out waiting for teacher group ${GROUP_ID} leader" >&2
            stop_teacher
            exit 1
         fi
         sleep 5
         waited=$(( waited + 5 ))
      done
      seen="${cur}"
      stop_teacher
      kill_stale_teachers
      # Secondary ranks wait for their ports too: relaunching while the previous
      # process still holds detokenizer_port leaves the leader on a half rendezvous.
      if ! wait_for_ports; then
         echo "group ${GROUP_ID} node ${NODE_RANK} attempt ${seen}: ports never released" >&2
         stop_teacher
         exit 1
      fi
      start_teacher "${seen}"
      # Hold this attempt until the group finishes, the leader announces the
      # next attempt, or this node's own teacher dies.
      while :; do
         if [[ -f "${GROUP_DONE}" ]]; then
            stop_teacher
            exit 0
         fi
         if [[ -f "${GROUP_FAILED}" ]]; then
            cat "${GROUP_FAILED}" >&2
            stop_teacher
            exit 1
         fi
         cur="$(read_attempt)"
         if (( cur > seen )); then
            break
         fi
         if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
            echo "secondary teacher exited: group=${GROUP_ID}, node=${NODE_RANK}, attempt=${seen}" >&2
            tail -n 120 "${TEACHER_LOG}" >&2 || true
            break
         fi
         if (( waited >= total_timeout )); then
            echo "timed out waiting for teacher group ${GROUP_ID} scorer" >&2
            stop_teacher
            exit 1
         fi
         sleep 10
         waited=$(( waited + 10 ))
      done
   done
fi
