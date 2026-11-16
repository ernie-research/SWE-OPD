#!/bin/bash
set -euo pipefail

# Base Qwen3.5-9B, YiYan 4093 rows, dynamic current Top-128 against a cached
# teacher Top-4096. This is the isolated counterpart of the 20260812 base run.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../../../../../" &>/dev/null && pwd)"

OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-base-yiyan-dynamic-t4k-s128-ep2-gbs64-c500-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
[[ "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
   echo "RUN_ROOT must be under ${OFFLINE_OPD_BASE}: ${RUN_ROOT}" >&2
   exit 1
}

export OFFLINE_OPD_BASE RUN_ROOT
export OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/SFT/model/Qwen3.5-9B}"
export LOAD_PATH="${LOAD_PATH:-/path/to/SFT/model/Qwen3.5-9B_torch_dist}"
export TEACHER_MODEL="${TEACHER_MODEL:-/path/to/SFT/model/Qwen3.6-27B}"
export PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/data/rl_query_final_20260709_issuetype_train.jsonl}"
export OPD_ID_TOPK=128 OPD_DYNAMIC_SUPPORT_TOPK=128 OPD_TEACHER_CACHE_TOPK=4096
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_dynamic_topk_reverse_kl

DATASET_SIZE="$(python3 - "${PROMPT_DATA}" <<'PY'
import sys
with open(sys.argv[1], "rb") as stream:
    print(sum(bool(line.strip()) for line in stream))
PY
)"
[[ "${DATASET_SIZE}" == 4093 ]] || {
   echo "expected 4093 YiYan rows, found ${DATASET_SIZE}: ${PROMPT_DATA}" >&2
   exit 1
}

mkdir -p "${RUN_ROOT}"
{
   printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
   printf 'PROMPT_DATA=%s\nDATASET_SIZE=%s\n' "${PROMPT_DATA}" "${DATASET_SIZE}"
   printf 'HF_CHECKPOINT=%s\nLOAD_PATH=%s\nTEACHER_MODEL=%s\n' "${HF_CHECKPOINT}" "${LOAD_PATH}" "${TEACHER_MODEL}"
   printf 'OFFLINE_OPD_LOSS_TYPE=cached_teacher_dynamic_topk_reverse_kl\n'
   printf 'OPD_DYNAMIC_SUPPORT_TOPK=128\nOPD_TEACHER_CACHE_TOPK=4096\n'
   printf 'PHASE1_CONCURRENCY=500\nPHASE1_PERSIST_BATCH_SIZE=1\nPHASE1_NUM_FILES=4093\n'
   printf 'PHASE2_MANIFEST_BATCH_SIZE=1\nPHASE2_BALANCE_MANIFESTS=1\nPHASE2_FILE_CONCURRENCY=8\nPHASE2_TEACHER_WATCHDOG_TIMEOUT=7200\n'
   printf 'PHASE3_GLOBAL_BATCH_SIZE=64\nPHASE3_TRAIN_STEPS=128\nPHASE3_EPOCHS=2\n'
   printf 'TP_SIZE=8\nPP_SIZE=1\nCP_SIZE=2\nMAX_CONTEXT_LEN=131072\nMAX_GEN_LEN=16384\n'
   printf 'LR=1e-6\nLR_DECAY_STYLE=constant\nSAVE_INTERVAL=20\n'
   printf 'WANDB_RUN_ID=%s\n' "${OFFLINE_OPD_RUN_TAG}"
} > "${RUN_ROOT}/experiment.env"

PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "${SCRIPT_PATH}" --worker >> "${PIPELINE_LOG}" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "dynamic base YiYan pipeline started: pid=$!, run=${RUN_ROOT}, log=${PIPELINE_LOG}"
   exit 0
fi

printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(date -Is)" > "${RUN_ROOT}/.pipeline.running"
rm -f "${RUN_ROOT}/.pipeline.failed" "${RUN_ROOT}/.pipeline.complete"
pipeline_failed() {
   local rc=$?
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${RUN_ROOT}/.pipeline.failed"
   rm -f "${RUN_ROOT}/.pipeline.running"
   exit "${rc}"
}
trap pipeline_failed ERR INT TERM

run_phase() {
   local phase="$1"; shift
   [[ -f "${RUN_ROOT}/.${phase}.complete" ]] && { echo "${phase} already complete; skipping"; return; }
   echo "[$(date -Is)] ${phase} starts"
   "$@"
   echo "[$(date -Is)] ${phase} completed"
}

run_phase phase1 env \
   RUN_ROOT="${RUN_ROOT}" OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE}" OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" LOAD_PATH="${LOAD_PATH}" PROMPT_DATA="${PROMPT_DATA}" \
   SLIME_ROLLOUT_ENGINE_PORT_BASE=30000 OFFLINE_OPD_CONCURRENCY=500 \
   ROLLOUT_BATCH_SIZE=1 PREFERRED_ROLLOUT_BATCH_SIZE=1 \
   bash "${SCRIPT_DIR}/01-phase1-rollout.sh" --foreground

run_phase phase2 env \
   RUN_ROOT="${RUN_ROOT}" OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE}" OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" LOAD_PATH="${LOAD_PATH}" TEACHER_MODEL="${TEACHER_MODEL}" \
   ROLLOUT_BATCH_SIZE=1 PHASE2_SAMPLES_PER_FILE=1 PHASE2_BALANCE_MANIFESTS=1 \
   SCORE_CONCURRENCY=8 SCORE_FILE_CONCURRENCY=8 TEACHER_WATCHDOG_TIMEOUT=7200 \
   bash "${SCRIPT_DIR}/02-phase2-teacher.sh" --foreground

run_phase phase3 env \
   RUN_ROOT="${RUN_ROOT}" OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE}" OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" LOAD_PATH="${LOAD_PATH}" \
   TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1 ROLLOUT_BATCH_SIZE=64 GLOBAL_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 \
   TP_SIZE=8 PP_SIZE=1 CP_SIZE=2 SAVE_INTERVAL=20 USE_WANDB=1 \
   WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}" WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}" \
   WANDB_GROUP="${WANDB_GROUP:-offline-opd-dynamic-t4k-s128-base-yiyan-ep2}" \
   WANDB_RUN_ID="${OFFLINE_OPD_RUN_TAG}" \
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground

printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
rm -f "${RUN_ROOT}/.pipeline.running" "${RUN_ROOT}/.pipeline.failed"
trap - ERR INT TERM
