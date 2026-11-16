#!/bin/bash
set -euo pipefail

# Resume the completed sampled-token + Top-128 diagnostic experiment from its
# iter_0000059 checkpoint and emit detailed importance-ratio tail diagnostics.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="${OFFLINE_OPD_BASE:-/path/to/OPD/offline_opd}"
SOURCE_RUN="${SOURCE_RUN:-${BASE_DIR}/qwen35-9b-token-reverse-kl-run4-reuse-k128-phase1-c500-20260806_020246}"

if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${BASE_DIR}/qwen35-9b-token-importance-debug-from59-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
[[ "${RUN_ROOT}" == "${BASE_DIR}"/* ]] || {
   echo "RUN_ROOT must be under ${BASE_DIR}: ${RUN_ROOT}" >&2
   exit 1
}

export RUN_ROOT
export RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/workspace/env_run/<user>/Megatron-LM}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/SFT/results/qwen35-9b-scaleswe5000-sft-246step-add-finish/checkpoint-246}"
export LOAD_PATH="${LOAD_PATH:-${RUN_ROOT}/load_checkpoint}"
export SAVE_PATH="${SAVE_PATH:-${RUN_ROOT}/checkpoints}"
export SCORED_DIR="${SCORED_DIR:-${SOURCE_RUN}/scored/epochs_x10}"
# SCORED_DIR is already the materialized ten-epoch replay (100 files).  Phase 3
# normally fans a one-pass scored directory out TRAIN_EPOCHS times, so force one
# here to avoid accidentally nesting epochs_x10/epochs_x10 (1000 files).
export TRAIN_EPOCHS=1

[[ -r "${LOAD_PATH}/latest_checkpointed_iteration.txt" ]] || {
   echo "missing isolated checkpoint pointer: ${LOAD_PATH}/latest_checkpointed_iteration.txt" >&2
   exit 1
}
[[ "$(<"${LOAD_PATH}/latest_checkpointed_iteration.txt")" == "59" ]] || {
   echo "debug resume must point to iteration 59: ${LOAD_PATH}" >&2
   exit 1
}
[[ -d "${LOAD_PATH}/iter_0000059" ]] || {
   echo "missing ${LOAD_PATH}/iter_0000059" >&2
   exit 1
}
[[ -r "${SCORED_DIR}/rollout_59.pt" ]] || {
   echo "missing offline replay data: ${SCORED_DIR}/rollout_59.pt" >&2
   exit 1
}

export ROLLOUT_BATCH_SIZE=50
export GLOBAL_BATCH_SIZE=50
export NUM_STEPS_PER_ROLLOUT=1
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-2}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024 * 128))}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-30}"
export PREFETCH_ROLLOUT_DATA="${PREFETCH_ROLLOUT_DATA:-1}"
export OFFLINE_OPD_MANIFEST_LOAD_WORKERS="${OFFLINE_OPD_MANIFEST_LOAD_WORKERS:-8}"
export OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS=1
export OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_IMPORTANCE_DEBUG=1
export OFFLINE_OPD_RESUME=1

export USE_WANDB="${USE_WANDB:-1}"
export WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-qwen35-9b-offline-token-loss-importance-tail-debug-resume-from59-top128}"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "${RUN_ROOT}"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
PIPELINE_RUNNING="${RUN_ROOT}/.pipeline.running"
PIPELINE_COMPLETE="${RUN_ROOT}/.pipeline.complete"
PIPELINE_FAILED="${RUN_ROOT}/.pipeline.failed"

if [[ "${1:-}" != "--worker" ]]; then
   nohup setsid bash "$0" --worker > "${PIPELINE_LOG}" 2>&1 < /dev/null &
   pipeline_pid=$!
   printf '%s\n' "${pipeline_pid}" > "${RUN_ROOT}/pipeline.pid"
   echo "Importance-debug Phase 3 started: pid=${pipeline_pid}"
   echo "run=${RUN_ROOT}"
   echo "load=${LOAD_PATH}"
   echo "data=${SCORED_DIR}"
   echo "wandb=${WANDB_GROUP}"
   echo "log=${PIPELINE_LOG}"
   exit 0
fi

printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(date -Is)" > "${PIPELINE_RUNNING}"
pipeline_failed() {
   local rc=$?
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${PIPELINE_FAILED}"
   rm -f "${PIPELINE_RUNNING}"
   exit "${rc}"
}
trap pipeline_failed ERR INT TERM

export RUN_DIR="${RUN_ROOT}/logs-phase3"
mkdir -p "${RUN_DIR}"
echo "[$(date -Is)] importance-debug Phase 3 starts from iter_0000059"
bash "${SCRIPT_DIR}/start-offline-token-opd.sh" phase3 > "${RUN_DIR}/launcher.log" 2>&1
printf 'completed=%s\n' "$(date -Is)" > "${PIPELINE_COMPLETE}"
rm -f "${PIPELINE_RUNNING}" "${PIPELINE_FAILED}"
trap - ERR INT TERM
echo "[$(date -Is)] importance-debug Phase 3 completed"
