#!/bin/bash
# Serve a Megatron training checkpoint (torch_dist `iter_XXXXXXX`) as an
# OpenAI-compatible sglang endpoint, so run_eval*.sh can score it unchanged.
#
#   # serve the LATEST saved iteration, then run the full fn_calling benchmark
#   RUN_EVAL=1 bash examples/coding_agent_opd/eval/serve_ckpt.sh
#
#   # serve one specific iteration and stop (eval later / by hand)
#   bash examples/coding_agent_opd/eval/serve_ckpt.sh 29
#   bash examples/coding_agent_opd/eval/serve_ckpt.sh /path/to/Qwen3.5-9B_save_ckpt/iter_0000029
#
#   # smoke: 1 instance against the served ckpt
#   RUN_EVAL=1 EVAL_SCRIPT=run_eval.sh LIMIT=1 MAX_STEPS_ABS=8 CONCURRENT=1 \
#     bash examples/coding_agent_opd/eval/serve_ckpt.sh 29
#
# Three stages. Stage 1 is NOT skippable and has no reuse path: a megatron ckpt is
# re-converted by THIS run every time, and only the weights this run wrote are served.
# Pre-converted weights (an `iter_*` dir under `*_hf`, or an engine already serving
# them) are refused, not reused -- nothing on disk proves which iteration wrote them.
#   1. CONVERT  torch_dist iter dir -> HF dir  (tools/convert_torch_dist_to_hf.py)
#      `-a/--add-missing-from-origin-hf` is REQUIRED here: Qwen3.5-9B is a VLM whose
#      megatron ckpt only holds `model.language_model.*` + `lm_head`, while the HF
#      config (Qwen3_5ForConditionalGeneration) also needs `model.visual.*` (333
#      tensors) and the MTP head -- those are copied verbatim from ORIGIN_HF.
#      Uses the "raw"/direct converter, matching the training default
#      (--megatron-to-hf-mode raw) and slime's own qwen3_5 weight mapping.
#   2. SERVE    python -m sglang.launch_server on that HF dir, flags mirroring the
#      rollout/teacher engines (tool-call-parser qwen3_coder + flashinfer, see the
#      Blackwell/head_dim=256 note in the multinode run script). Detached via setsid
#      so it outlives this shell / an SSH drop.
#   3. EVAL     (RUN_EVAL=1) exec the existing eval script with LLM_URL/MODEL_NAME
#      pointed at the fresh server. Nothing about the harness changes.
# Every run ends with a PROVENANCE block (pass or fail): ckpt src + kind + iteration,
# whether this run converted it, the dir sglang was pointed at, and the live engine's
# own /get_model_info model_path -- everything needed to attribute a number later.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${SLIME_DIR}"

MEGATRON_LM="${MEGATRON_LM:-/root/Megatron-LM}"

# ============ python env (same venv the training / eval runs use) ============
SLIME_VENV="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}"
if [[ -x "${SLIME_VENV}/bin/python3" ]]; then
    export PATH="${SLIME_VENV}/bin:${PATH}"
else
    echo "WARN: ${SLIME_VENV}/bin/python3 not found, using $(command -v python3)" >&2
fi
# cudnn load-order workaround (transformer_engine dlopen order); see general_settings.sh.
if [[ -z "${CUDNN_PATH:-}" ]]; then
    CUDNN_PATH="$(python3 -c 'import sysconfig,os;print(os.path.join(sysconfig.get_path("purelib"),"nvidia/cudnn"))' 2>/dev/null || true)"
    [[ -d "${CUDNN_PATH}" ]] && export CUDNN_PATH || unset CUDNN_PATH
fi
export PYTHONPATH="${MEGATRON_LM}:${SLIME_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# ============ which checkpoint ============
# $1 = a megatron iter dir, a bare iteration number, nothing (=> latest iteration),
#      or an ORIGINAL HF model dir (the untrained base model / an SFT release, for a
#      baseline run): that case has nothing to convert and is served as-is.
#      The OUTPUT of an earlier conversion is NOT valid input -- see the iter_*/_hf
#      refusal below.
# Every other input is FATAL. A typo'd path used to fall through to the "bare
# iteration number" branch and silently build a nonsense ckpt path instead of
# stopping, so nothing here guesses any more.
CKPT_BASE="${CKPT_BASE:-/path/to/Qwen3.5-9B_save_ckpt_scaleswe}"
ARG="${1:-${ITER:-}}"

# ---- HF_DIR is DERIVED from the checkpoint, never inherited ----
# An HF_DIR left in the environment is how the 2026-08-02 sweep scored the SFT
# checkpoint while every label (TAG / MODEL_NAME / LOG_DIR / OUT_DIR, and the
# server's --served-model-name) still said "iter199": ${HF_DIR:-default} let one
# exported variable replace the weights without touching a single label, and the
# 500-instance result looked entirely normal. Only $1 selects what gets served.
if [[ -n "${HF_DIR:-}" ]]; then
    echo "WARN: ignoring inherited HF_DIR=${HF_DIR}; it is derived from the ckpt path below" >&2
fi
unset HF_DIR

if [[ -z "${ARG}" ]]; then
    LATEST_FILE="${CKPT_BASE}/latest_checkpointed_iteration.txt"
    [[ -f "${LATEST_FILE}" ]] \
        || { echo "FATAL: no checkpoint given and ${LATEST_FILE} does not exist" >&2; exit 1; }
    ARG="$(tr -dc '0-9' <"${LATEST_FILE}")"
    [[ -n "${ARG}" ]] || { echo "FATAL: ${LATEST_FILE} holds no iteration number" >&2; exit 1; }
fi

if [[ "${ARG}" =~ ^[0-9]+$ ]]; then
    CKPT_DIR="${CKPT_BASE}/iter_$(printf '%07d' "$((10#${ARG}))")"
    [[ -d "${CKPT_DIR}" ]] \
        || { echo "FATAL: iteration ${ARG} has no checkpoint dir at ${CKPT_DIR}" >&2; exit 1; }
elif [[ -d "${ARG%/}" ]]; then
    CKPT_DIR="${ARG%/}"
else
    echo "FATAL: '${ARG}' is neither an iteration number nor an existing directory" >&2
    exit 1
fi

# SRC_DIR is what the user actually pointed at, kept verbatim for the provenance
# block at the end: CKPT_DIR is blanked in the already-HF branch below.
SRC_DIR="${CKPT_DIR}"

if [[ -f "${CKPT_DIR}/common.pt" ]]; then     # megatron torch_dist -> must be converted
    CKPT_BASE="$(dirname "${CKPT_DIR}")"
    TAG="$(basename "${CKPT_DIR}")"           # iter_0000029
    HF_DIR="${CKPT_BASE}_hf/${TAG}"
    CKPT_KIND="megatron torch_dist"
    if [[ "${TAG}" =~ ^iter_0*([0-9]+)$ ]]; then
        ITERATION="${BASH_REMATCH[1]}"
        DEFAULT_MODEL_NAME="Qwen3.5-9B-iter${ITERATION}"
    else
        ITERATION=""
        DEFAULT_MODEL_NAME="Qwen3.5-9B-${TAG}"
    fi
elif [[ -f "${CKPT_DIR}/config.json" ]]; then # already HF -> nothing to convert
    TAG="$(basename "${CKPT_DIR}")"           # Qwen3.5-9B / checkpoint-246
    # ---- refuse the OUTPUT of an earlier conversion ----
    # A megatron ckpt reaches the server through the conversion below and no other
    # way. An `iter_*` dir under `*_hf` is exactly such an output: nothing in it
    # records which iteration or which converter flags produced it, while TAG /
    # MODEL_NAME / LOG_DIR would still read like a genuine iteration -- an
    # unattributable number, the same failure mode as the 2026-08-02 sweep.
    HF_PARENT="$(dirname "${CKPT_DIR}")"
    if [[ "${TAG}" =~ ^iter_[0-9]+$ || "${HF_PARENT}" == *_hf ]]; then
        echo "FATAL: ${CKPT_DIR} holds CONVERTED weights (an iter_* dir under *_hf)." >&2
        echo "       This script serves a megatron ckpt only by converting it itself." >&2
        if [[ -f "${HF_PARENT%_hf}/${TAG}/common.pt" ]]; then
            echo "       Pass the megatron ckpt instead:" >&2
            echo "         ${HF_PARENT%_hf}/${TAG}" >&2
        fi
        exit 1
    fi
    HF_DIR="${CKPT_DIR}"
    CKPT_DIR=""
    CKPT_KIND="huggingface (original)"
    ITERATION=""
    DEFAULT_MODEL_NAME="${TAG}"
else
    echo "FATAL: ${CKPT_DIR} is neither a megatron torch_dist ckpt (no common.pt)" >&2
    echo "       nor a HuggingFace model dir (no config.json)" >&2
    exit 1
fi

ORIGIN_HF="${ORIGIN_HF:-/path/to/Qwen3.5-9B}"

# ============ server knobs ============
# GPUS x TP -> DP: all visible GPUs are used, TP-sharded replicas load-balanced by
# sglang's built-in dp router (8 GPUs / tp2 = 4 replicas), which is what CONCURRENT=64
# agentic decode needs.
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP="${TP:-2}"
NUM_GPUS="$(awk -F, '{print NF}' <<<"${GPUS}")"
DP="${DP:-$(( NUM_GPUS / TP ))}"
(( DP >= 1 )) || DP=1
PORT="${PORT:-9500}"                          # below the ephemeral floor; see the port sanity check
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"         # host used in LLM_URL (proxy-bypassed)
MODEL_NAME="${MODEL_NAME:-${DEFAULT_MODEL_NAME}}"
MEM_FRACTION="${MEM_FRACTION:-0.85}"
ATTN_BACKEND="${ATTN_BACKEND:-flashinfer}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
# ---- token budget: 128K total window, per-turn generation uncapped by default ----
# Three knobs, one inequality:  MAX_MODEL_LEN + obs + MAX_TOKENS <= CONTEXT_LEN
#   CONTEXT_LEN    (-> sglang --context-length) the HARD window: sglang 400s any
#                  request whose prompt + max_tokens exceeds it, and eval_chat
#                  (eval_chat.py:339) treats 400 as deterministic -> instance lost.
#   MAX_TOKENS     (-> eval_chat --max-tokens) per-turn generation cap. **0 (the
#                  default) sends no max_tokens at all**, so the server grants
#                  context_len - input_len - 2 to every turn and the inequality
#                  below is vacuous. A positive value is sent VERBATIM on every
#                  turn and never shrinks (reproduces the old fixed-cap metric).
#   MAX_MODEL_LEN  (-> eval_chat --max-model-len) SOFT early stop: the loop breaks
#                  with status="truncated" once the last turn's usage.total_tokens
#                  reaches it (eval_chat.py:464/488). It bounds how large the prompt
#                  can grow, but the NEXT prompt still adds one observation on top --
#                  execute_bash observations are middle-truncated at 16000 chars
#                  (r2egym/observation.py), so ~8K tokens of slack is kept below.
# 128K - 16K generation - 8K observation slack = 104K prompt budget.
CONTEXT_LEN="${CONTEXT_LEN:-$((1024*128))}"          # 131072
EVAL_MAX_TOKENS="${MAX_TOKENS:-0}"                   # 0 = uncapped, forwarded to the eval
EVAL_MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((1024*128))}" # 106496, forwarded to the eval
if (( EVAL_MAX_TOKENS > 0 && EVAL_MAX_MODEL_LEN + EVAL_MAX_TOKENS > CONTEXT_LEN )); then
    echo "FATAL: MAX_MODEL_LEN(${EVAL_MAX_MODEL_LEN}) + MAX_TOKENS(${EVAL_MAX_TOKENS}) > CONTEXT_LEN(${CONTEXT_LEN})." >&2
    echo "       Late-trajectory requests would be 400'd by the server; lower one of them." >&2
    exit 1
fi
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1800}"      # seconds to wait for /health_generate

LOG_DIR="${LOG_DIR:-${SLIME_DIR}/examples/coding_agent_opd/logs/serve_${TAG}_$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S)}"
mkdir -p "${LOG_DIR}"
SERVER_LOG="${LOG_DIR}/sglang_server.log"
PID_FILE="${LOG_DIR}/sglang_server.pid"

# Never route the local endpoint through the GitHub proxy that run_eval.sh exports.
export no_proxy="127.0.0.1,localhost,${SERVE_HOST},$(hostname -i | awk '{print $1}'),${no_proxy:-}"
export NO_PROXY="${no_proxy}"

LLM_URL="http://${SERVE_HOST}:${PORT}/v1/chat/completions"

echo "=============================================="
echo " ckpt      : ${CKPT_DIR:-<none, serving an existing HF dir>}"
echo " origin hf : ${ORIGIN_HF}"
echo " hf out    : ${HF_DIR}"
echo " serve     : ${LLM_URL}  (model=${MODEL_NAME} gpus=${GPUS} tp=${TP} dp=${DP})"
echo " budget    : context_len=${CONTEXT_LEN}  max_model_len=${EVAL_MAX_MODEL_LEN}  max_tokens=${EVAL_MAX_TOKENS}$( ((EVAL_MAX_TOKENS == 0)) && echo ' (uncapped: server grants context_len-input_len-2/turn)')"
echo " log dir   : ${LOG_DIR}"
echo " run_eval  : ${RUN_EVAL:-0} (${EVAL_SCRIPT:-run_eval.sh})"
echo "=============================================="

# ============ stage 1: torch_dist -> HF (ALWAYS re-converted) ============
# There is no "the dir already has config.json, reuse it" shortcut, and no way to
# skip this stage. Such a dir can be a half-written conversion, a different
# iteration, or a different model entirely, and serving it produces a benchmark
# labelled with the wrong checkpoint. So: convert every time, then serve EXACTLY
# what this conversion wrote (verified below, and again against the live endpoint
# after health).
#
# A dir already loaded by a running engine on ${PORT} cannot be re-converted -- the
# converter would rewrite the safetensors under it. That is FATAL: scoring whatever
# that engine happens to hold is the one thing this script must never do.
if curl -sf "http://${SERVE_HOST}:${PORT}/health_generate" >/dev/null 2>&1; then
    echo "FATAL: a server already answers on ${SERVE_HOST}:${PORT}, so ${HF_DIR} cannot be" >&2
    echo "       re-converted safely (rewriting weights under a live engine corrupts them)." >&2
    echo "       Its weights were not produced by this run and will not be scored here:" >&2
    echo "       stop that server, or pass PORT=<other>." >&2
    exit 1
fi

CONVERT_LOG="${LOG_DIR}/convert.log"
CONVERT_SHARDS=""
CONVERT_SECONDS=""
if [[ -z "${CKPT_DIR}" ]]; then
    DEPLOY_NOTE="the ckpt itself: already HuggingFace, nothing converted"
    echo "[convert] not needed: ${HF_DIR} is an original HuggingFace model dir"
    [[ -f "${HF_DIR}/config.json" ]] \
        || { echo "FATAL: ${HF_DIR}/config.json disappeared" >&2; exit 1; }
else
    DEPLOY_NOTE="converted from the megatron ckpt above by THIS run"
    CONVERT_MARK="${LOG_DIR}/.convert_started"
    : >"${CONVERT_MARK}"
    CONVERT_ARGS=(--input-dir "${CKPT_DIR}" --output-dir "${HF_DIR}" --origin-hf-dir "${ORIGIN_HF}"
                  --add-missing-from-origin-hf)
    [[ -e "${HF_DIR}" ]] && CONVERT_ARGS+=(--force)
    echo "[convert] ${CKPT_DIR} -> ${HF_DIR}"
    CONVERT_T0="${SECONDS}"
    python3 tools/convert_torch_dist_to_hf.py "${CONVERT_ARGS[@]}" 2>&1 | tee "${CONVERT_LOG}"
    CONVERT_SECONDS="$(( SECONDS - CONVERT_T0 ))"

    # ---- verify the conversion, file by file ----
    # copy_assets() writes config.json LAST, so its presence means the run finished;
    # the mtime check is what catches a STALE dir. --force only skips the converter's
    # "output dir exists" guard -- it does NOT wipe the dir, so a previous conversion
    # with a different shard count can leave orphan model-*.safetensors behind. Every
    # shard the index actually references must be newer than this run's marker.
    # Prints the shard count, which the provenance block at the end reports.
    CONVERT_SHARDS="$(python3 - "${HF_DIR}" "${CONVERT_MARK}" <<'PY'
import json, os, sys

hf_dir, mark = sys.argv[1], sys.argv[2]
t0 = os.path.getmtime(mark)
index = os.path.join(hf_dir, "model.safetensors.index.json")
if not os.path.isfile(index):
    sys.exit(f"FATAL: conversion produced no model.safetensors.index.json in {hf_dir}")
with open(index) as fh:
    shards = sorted(set(json.load(fh)["weight_map"].values()))
for name in ["config.json", "model.safetensors.index.json", *shards]:
    path = os.path.join(hf_dir, name)
    if not os.path.isfile(path):
        sys.exit(f"FATAL: {path} is missing after the conversion")
    if os.path.getmtime(path) < t0:
        sys.exit(f"FATAL: {path} predates this conversion -- stale HF dir, refusing to serve it")
print(len(shards))
PY
)"
    rm -f "${CONVERT_MARK}"
    echo "[convert] done in ${CONVERT_SECONDS}s: ${CONVERT_SHARDS} shards + index + config.json," \
         "all written by this run"
fi

# ============ stage 2: sglang server ============
# ---- port sanity ----
# ${PORT} is only the HTTP port. Two constraints on it:
#  * keep it OUT of the kernel's ephemeral range so nothing transient squats on it;
#  * PORT + 10000 <= 65535, because sglang sets SGLANG_GRPC_PORT = PORT + 10000 and
#    validates it ("SGLANG_GRPC_PORT (71500) must be between 1 and 65535" killed an
#    attempt at PORT=61500) -- so hiding ABOVE the ephemeral range is not an option.
# 9500 satisfies both. NOTE: this does NOT protect the engine's INTERNAL torch.dist
# port -- see the retry loop below.
read -r EPH_LO EPH_HI < /proc/sys/net/ipv4/ip_local_port_range
if (( PORT <= EPH_HI && PORT + 63 >= EPH_LO )); then
    echo "WARN: PORT ${PORT} is inside the ephemeral range ${EPH_LO}-${EPH_HI};" \
         "prefer PORT < ${EPH_LO} (default 9500)." >&2
fi
if (( PORT + 10000 > 65535 )); then
    echo "FATAL: PORT ${PORT} + 10000 > 65535; sglang's derived SGLANG_GRPC_PORT is invalid." >&2
    exit 1
fi
if ss -ltn 2>/dev/null | tail -n +2 \
    | awk -v p="${PORT}" '{n=split($4,a,":"); if (a[n]+0==p) found=1} END{exit !found}'; then
    echo "FATAL: something already LISTENs on ${PORT} but is not a healthy sglang;" >&2
    echo "       inspect it, or pick another PORT=..." >&2
    exit 1
fi

# ---- launch with retries: sglang's nccl/torch.dist port is a race ----
# That port has NOTHING to do with ${PORT}: PortArgs.init_new falls back to
# get_free_port() (bind :0 -> read the number -> CLOSE -> rebind later), so it is an
# EPHEMERAL port (this box: 10000-61000, heavily used). Between the close and the
# scheduler's real bind an unrelated outbound connection can take it, and the engine
# dies with "port: 18009 ... EADDRINUSE -> Rank 0 scheduler died during
# initialization" (hit at both 18015 and 18009 on different launches). Pinning
# --nccl-port is NOT a fix at dp>1: data_parallel_controller.py:247-253 calls
# init_new per dp rank and holds each port, so all ranks would collide on one port.
# The race is rare and independent per attempt, so just relaunch.
LAUNCH_TRIES="${LAUNCH_TRIES:-3}"
SERVER_UP=0
for (( attempt=1; attempt<=LAUNCH_TRIES; attempt++ )); do
    (( attempt > 1 )) && SERVER_LOG="${LOG_DIR}/sglang_server.retry${attempt}.log"
    echo "[serve] launching sglang on GPUs ${GPUS} (attempt ${attempt}/${LAUNCH_TRIES}) -> ${SERVER_LOG}"
    CUDA_VISIBLE_DEVICES="${GPUS}" setsid python3 -m sglang.launch_server \
        --model-path "${HF_DIR}" \
        --served-model-name "${MODEL_NAME}" \
        --host 0.0.0.0 \
        --port "${PORT}" \
        --tp "${TP}" \
        --dp "${DP}" \
        --trust-remote-code \
        --tool-call-parser "${TOOL_CALL_PARSER}" \
        --attention-backend "${ATTN_BACKEND}" \
        --context-length "${CONTEXT_LEN}" \
        --mem-fraction-static "${MEM_FRACTION}" \
        --watchdog-timeout 1200 \
        ${SGLANG_EXTRA_ARGS:-} \
        </dev/null >"${SERVER_LOG}" 2>&1 &
    SERVER_PID=$!
    echo "${SERVER_PID}" > "${PID_FILE}"
    echo "[serve] pid=${SERVER_PID} (detached; survives disconnect)"

    DEADLINE=$((SECONDS + HEALTH_TIMEOUT))
    while :; do
        if curl -sf "http://${SERVE_HOST}:${PORT}/health_generate" >/dev/null 2>&1; then
            SERVER_UP=1
            break
        fi
        kill -0 "${SERVER_PID}" 2>/dev/null || break      # died -> handled below
        if (( SECONDS > DEADLINE )); then
            echo "FATAL: server not healthy after ${HEALTH_TIMEOUT}s; see ${SERVER_LOG}" >&2
            exit 1
        fi
        sleep 10
        echo "[serve] waiting for ${SERVE_HOST}:${PORT} ... ($(tail -n 1 "${SERVER_LOG}" 2>/dev/null | cut -c1-120))"
    done
    (( SERVER_UP == 1 )) && break

    if grep -q "EADDRINUSE" "${SERVER_LOG}"; then
        echo "[serve] lost an internal port to the ephemeral range" \
             "($(grep -oE "port: [0-9]+" "${SERVER_LOG}" | head -1)); relaunching"
        kill -TERM -"${SERVER_PID}" 2>/dev/null || true
        sleep 15
        continue
    fi
    echo "FATAL: sglang died during startup, last 40 lines of ${SERVER_LOG}:" >&2
    tail -n 40 "${SERVER_LOG}" >&2
    exit 1
done
if (( SERVER_UP != 1 )); then
    echo "FATAL: sglang lost the internal-port race ${LAUNCH_TRIES} times; retry, or" >&2
    echo "       raise LAUNCH_TRIES=..." >&2
    exit 1
fi
echo "[serve] up. $(curl -s "http://${SERVE_HOST}:${PORT}/get_model_info" || true)"

# ============ verify: the endpoint serves EXACTLY ${HF_DIR} ============
# --served-model-name is just a label the launcher chose, so it is never evidence of
# which weights are loaded -- /get_model_info's model_path is. Skipping this check is
# what let a whole 500-instance sweep be filed under "iter199" while the engine held
# the SFT checkpoint (2026-08-02).
SERVED_PATH="$(curl -s "http://${SERVE_HOST}:${PORT}/get_model_info" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin).get("model_path",""))' 2>/dev/null || true)"
if [[ -z "${SERVED_PATH}" ]]; then
    echo "FATAL: /get_model_info returned no model_path; cannot prove what is served" >&2
    exit 1
fi
if [[ "$(readlink -f "${SERVED_PATH}")" != "$(readlink -f "${HF_DIR}")" ]]; then
    echo "FATAL: the endpoint on ${SERVE_HOST}:${PORT} serves" >&2
    echo "         ${SERVED_PATH}" >&2
    echo "       but this run converted / expects" >&2
    echo "         ${HF_DIR}" >&2
    echo "       Everything downstream would be labelled '${MODEL_NAME}' with the wrong weights." >&2
    exit 1
fi
echo "[verify] endpoint serves ${SERVED_PATH}  (== ${HF_DIR}, model name '${MODEL_NAME}')"

# ============ preflight: the eval's ACTUAL request budget must be accepted ============
# /health_generate only proves the engine decodes; it says nothing about whether a
# request carrying the eval's max_tokens fits the window. One probe here replaces 500
# instances each failing on their first call (and 500 booted RLE pods burnt for nothing).
PREFLIGHT_JSON="${LOG_DIR}/preflight.json"
# With MAX_TOKENS=0 the eval sends no max_tokens, so there is no per-turn cap to
# validate -- probe with a token or two just to confirm the endpoint generates.
PREFLIGHT_MAX_TOKENS=$(( EVAL_MAX_TOKENS > 0 ? EVAL_MAX_TOKENS : 8 ))
PREFLIGHT_CODE="$(curl -s -o "${PREFLIGHT_JSON}" -w '%{http_code}' \
    "http://${SERVE_HOST}:${PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"${MODEL_NAME}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK.\"}],\"max_tokens\":${PREFLIGHT_MAX_TOKENS}}")"
if [[ "${PREFLIGHT_CODE}" != "200" ]]; then
    echo "FATAL: preflight POST with max_tokens=${PREFLIGHT_MAX_TOKENS} returned HTTP ${PREFLIGHT_CODE}:" >&2
    head -c 800 "${PREFLIGHT_JSON}" >&2; echo >&2
    echo "hint: CONTEXT_LEN (${CONTEXT_LEN}) must exceed the eval's MAX_TOKENS (${EVAL_MAX_TOKENS})," >&2
    echo "      since eval_chat sends that max_tokens on every turn. Relaunch with a" >&2
    echo "      larger CONTEXT_LEN, or lower MAX_TOKENS for the eval." >&2
    exit 1
fi
echo "[preflight] OK: max_tokens=${PREFLIGHT_MAX_TOKENS} accepted within context_len=${CONTEXT_LEN}"

# ============ provenance: the last thing printed, pass or fail ============
# The single block that answers "which weights produced these numbers?" -- the ckpt
# that was passed in, whether THIS run converted it, the dir sglang was pointed at,
# and what the live engine reports as its own model_path. MODEL_NAME is printed as
# what it is: a label chosen by the launcher, never evidence. A sweep filed under the
# wrong iteration (2026-08-02) cannot be attributed after the fact, so no run of this
# script ends without these lines.
print_provenance() {
    echo "=============================================="
    echo " provenance (what any number from this endpoint describes)"
    echo " when      : $(TZ='Asia/Shanghai' date '+%Y-%m-%d %H:%M:%S %Z')"
    echo " host      : $(hostname) ($(hostname -i | awk '{print $1}'))"
    echo " ckpt src  : ${SRC_DIR}"
    echo "             kind=${CKPT_KIND}${ITERATION:+   iteration=${ITERATION}}"
    echo " converted : ${DEPLOY_NOTE}"
    if [[ -n "${CONVERT_SHARDS}" ]]; then
        echo "             ${CONVERT_SHARDS} shards + index + config.json, ${CONVERT_SECONDS}s, all written by this run"
        echo "             origin hf : ${ORIGIN_HF}   (model.visual.* + MTP head, absent from the ckpt)"
        echo "             convert log: ${CONVERT_LOG}"
    fi
    echo " serve dir : ${HF_DIR}   <- sglang --model-path"
    echo " engine    : ${SERVED_PATH:-<never verified>}   <- /get_model_info model_path"
    echo " label     : ${MODEL_NAME}   (--served-model-name: a LABEL, not evidence)"
    echo " endpoint  : ${LLM_URL}   gpus=${GPUS} tp=${TP} dp=${DP}"
    echo " budget    : context_len=${CONTEXT_LEN}  max_model_len=${EVAL_MAX_MODEL_LEN}  max_tokens=${EVAL_MAX_TOKENS}"
    echo " serve log : ${SERVER_LOG}   (pid $(cat "${PID_FILE}" 2>/dev/null || echo '?'))"
    echo " log dir   : ${LOG_DIR}"
    [[ -n "${OUT_DIR:-}" ]] && echo " eval out  : ${OUT_DIR}"
    echo "=============================================="
}

# ============ stage 3: reuse the existing eval scripts ============
# MAX_TOKENS / MAX_MODEL_LEN are forwarded so the eval runs the SAME budget the
# preflight above validated against this server's window -- otherwise
# run_eval.sh would fall back to its own defaults.
export LLM_URL MODEL_NAME
export MAX_TOKENS="${EVAL_MAX_TOKENS}"
export MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN}"
if [[ "${RUN_EVAL:-0}" == "1" ]]; then
    EVAL_SCRIPT="${EVAL_SCRIPT:-run_eval.sh}"
    TS="$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S)"
    export OUT_DIR="${OUT_DIR:-examples/coding_agent_opd/logs/eval_${TAG}_${TS}}"
    export DUMP_DIR="${DUMP_DIR:-examples/coding_agent_opd/logs/dump_${TAG}_${TS}}"
    echo "[eval] bash ${SCRIPT_DIR}/${EVAL_SCRIPT}  (out_dir=${OUT_DIR})"
    # rc is kept instead of let-it-die so the provenance is printed even when the
    # eval fails -- a failed run still needs to say which weights it was scoring.
    EVAL_RC=0
    bash "${SCRIPT_DIR}/${EVAL_SCRIPT}" || EVAL_RC=$?
    print_provenance
    exit "${EVAL_RC}"
else
    echo "=============================================="
    echo " server ready. evaluate this ckpt with:"
    echo "   LLM_URL='${LLM_URL}' MODEL_NAME='${MODEL_NAME}' \\"
    echo "     MAX_TOKENS=${EVAL_MAX_TOKENS} MAX_MODEL_LEN=${EVAL_MAX_MODEL_LEN} \\"
    echo "     bash examples/coding_agent_opd/eval/run_eval.sh"
    echo " stop the server:  kill -TERM -\$(cat ${PID_FILE})"
    print_provenance
fi
