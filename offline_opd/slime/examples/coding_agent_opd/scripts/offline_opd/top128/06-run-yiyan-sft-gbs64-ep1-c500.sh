#!/bin/bash
set -euo pipefail

# One-pass fully-offline OPD on the YiYan SWE factory dataset:
#   Phase 1: SFT student rolls out every one of the 4093 prompts exactly once.
#   Phase 2: the fixed Qwen3.6 teacher scores that cached support exactly once.
#   Phase 3: two continuous epochs at GBS=64 (128 steps; the final batch reuses
#            six cached samples from the beginning of the next epoch).
#
# 4093 is prime. Phase 1 therefore persists one sample per atomic rollout file
# while its continuous producer still keeps 500 requests globally in flight.
# Phase 2 emits sample artifacts/manifests, and the existing cross-epoch
# rebatcher assembles the training-time batches without regenerating data.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../../../../../" &>/dev/null && pwd)"

OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-sft-yiyan-k128-ep2-gbs64-c500-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
[[ "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
   echo "RUN_ROOT must be under ${OFFLINE_OPD_BASE}: ${RUN_ROOT}" >&2
   exit 1
}

export OFFLINE_OPD_BASE RUN_ROOT
export OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}" 
export RLE_SESSION_IDS_DIR="${RLE_SESSION_IDS_DIR:-${RUN_ROOT}/rle_sessions}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/SFT/results/qwen35-9b-scaleswe5000-sft-246step-add-finish/checkpoint-246}"
export LOAD_PATH="${LOAD_PATH:-/path/to/OPD/model/qwen35-9b-scaleswe5000-sft-246-add-finish_torch_dist}"
export TEACHER_MODEL="${TEACHER_MODEL:-/path/to/SFT/model/Qwen3.6-27B}"
export PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/data/rl_query_final_20260709_issuetype_train.jsonl}"
export OPD_ID_TOPK=128
export OFFLINE_OPD_LOSS_TYPE=topk_reverse_kl

DATASET_SIZE="$(python3 - "${PROMPT_DATA}" <<'PY'
import sys
with open(sys.argv[1], "rb") as stream:
    print(sum(bool(line.strip()) for line in stream))
PY
)"
[[ "${DATASET_SIZE}" == 4093 ]] || {
   echo "expected 4093 YiYan training rows, found ${DATASET_SIZE}: ${PROMPT_DATA}" >&2
   exit 1
}

mkdir -p "${RUN_ROOT}" "${RLE_SESSION_IDS_DIR}"
{
   printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
   printf 'PROMPT_DATA=%s\n' "${PROMPT_DATA}"
   printf 'DATASET_SIZE=%s\n' "${DATASET_SIZE}"
   printf 'HF_CHECKPOINT=%s\n' "${HF_CHECKPOINT}"
   printf 'LOAD_PATH=%s\n' "${LOAD_PATH}"
   printf 'TEACHER_MODEL=%s\n' "${TEACHER_MODEL}"
   printf 'OPD_ID_TOPK=128\n'
   printf 'OFFLINE_OPD_LOSS_TYPE=topk_reverse_kl\n'
   printf 'PHASE1_CONCURRENCY=500\n'
   printf 'PHASE1_ROLLOUT_ENGINE_PORT_BASE=30000\n'
   printf 'PHASE1_PERSIST_BATCH_SIZE=1\n'
   printf 'PHASE1_NUM_FILES=4093\n'
   printf 'PHASE2_MANIFEST_BATCH_SIZE=1\n'
   printf 'PHASE2_BALANCE_MANIFESTS=1\n'
   printf 'PHASE3_GLOBAL_BATCH_SIZE=64\n'
   printf 'PHASE3_TRAIN_STEPS=128\n'
   printf 'PHASE3_EPOCHS=2\n'
   printf 'PHASE3_CROSS_EPOCH_PADDING_SAMPLES=6\n'
   printf 'TP_SIZE=8\nPP_SIZE=1\nCP_SIZE=2\n'
   printf 'MAX_CONTEXT_LEN=131072\nMAX_GEN_LEN=16384\n'
   printf 'LR=1e-6\nLR_DECAY_STYLE=constant\n'
   printf 'SAVE_INTERVAL=30\n'
   printf 'WANDB_RUN_ID=%s\n' "${OFFLINE_OPD_RUN_TAG}"
   printf 'RLE_SESSION_IDS_DIR=%s\n' "${RLE_SESSION_IDS_DIR}"
} > "${RUN_ROOT}/experiment.env"

PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
PIPELINE_RUNNING="${RUN_ROOT}/.pipeline.running"
PIPELINE_COMPLETE="${RUN_ROOT}/.pipeline.complete"
PIPELINE_FAILED="${RUN_ROOT}/.pipeline.failed"

if [[ "${1:-}" != "--worker" ]]; then
   nohup setsid bash "${SCRIPT_PATH}" --worker >> "${PIPELINE_LOG}" 2>&1 < /dev/null &
   pipeline_pid=$!
   printf '%s\n' "${pipeline_pid}" > "${RUN_ROOT}/pipeline.pid"
   echo "YiYan SFT Top-128 offline pipeline started: pid=${pipeline_pid}"
   echo "run=${RUN_ROOT}"
   echo "log=${PIPELINE_LOG}"
   exit 0
fi

printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(date -Is)" > "${PIPELINE_RUNNING}"
rm -f "${PIPELINE_FAILED}" "${PIPELINE_COMPLETE}"

pipeline_failed() {
   local rc=$?
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${PIPELINE_FAILED}"
   rm -f "${PIPELINE_RUNNING}"
   exit "${rc}"
}
trap pipeline_failed ERR INT TERM

run_phase() {
   local phase="$1"
   shift
   if [[ -f "${RUN_ROOT}/.${phase}.complete" ]]; then
      echo "[$(date -Is)] ${phase} already complete; skipping"
      return 0
   fi
   echo "[$(date -Is)] ${phase} starts"
   "$@"
   echo "[$(date -Is)] ${phase} completed"
}

echo "[$(date -Is)] YiYan SFT Top-128 offline experiment starts: ${RUN_ROOT}"

run_phase phase1 env \
   RUN_ROOT="${RUN_ROOT}" \
   OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE}" \
   OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" \
   LOAD_PATH="${LOAD_PATH}" \
   PROMPT_DATA="${PROMPT_DATA}" \
   SLIME_ROLLOUT_ENGINE_PORT_BASE=30000 \
   OFFLINE_OPD_CONCURRENCY=500 \
   ROLLOUT_BATCH_SIZE=1 \
   PREFERRED_ROLLOUT_BATCH_SIZE=1 \
   bash "${SCRIPT_DIR}/01-phase1-rollout.sh" --foreground

run_phase phase2 env \
   RUN_ROOT="${RUN_ROOT}" \
   OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE}" \
   OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" \
   LOAD_PATH="${LOAD_PATH}" \
   TEACHER_MODEL="${TEACHER_MODEL}" \
   ROLLOUT_BATCH_SIZE=1 \
   PHASE2_SAMPLES_PER_FILE=1 \
   PHASE2_BALANCE_MANIFESTS=1 \
   bash "${SCRIPT_DIR}/02-phase2-teacher.sh" --foreground

run_phase phase3 env \
   RUN_ROOT="${RUN_ROOT}" \
   OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE}" \
   OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" \
   LOAD_PATH="${LOAD_PATH}" \
   TRAIN_EPOCHS=2 \
   EPOCH_SHUFFLE=1 \
   ROLLOUT_BATCH_SIZE=64 \
   GLOBAL_BATCH_SIZE=64 \
   NUM_STEPS_PER_ROLLOUT=1 \
   TP_SIZE=8 \
   PP_SIZE=1 \
   CP_SIZE=2 \
   SAVE_INTERVAL=30 \
   USE_WANDB=1 \
   WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}" \
   WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}" \
   WANDB_GROUP="${WANDB_GROUP:-offline-opd-k128-yiyan-sft-ep2}" \
   WANDB_RUN_ID="${OFFLINE_OPD_RUN_TAG}" \
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground

printf 'completed=%s\n' "$(date -Is)" > "${PIPELINE_COMPLETE}"
rm -f "${PIPELINE_RUNNING}" "${PIPELINE_FAILED}"
trap - ERR INT TERM
echo "[$(date -Is)] YiYan SFT Top-128 offline experiment completed: ${RUN_ROOT}"
