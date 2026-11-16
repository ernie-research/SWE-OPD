#!/bin/bash
set -euo pipefail

# Full Top-128 offline-OPD experiment starting from the Qwen3.5-9B base model.
# Unlike the replay-only launchers, this script always keeps Phase 1, Phase 2,
# replay manifests, and Phase 3 checkpoints under one fresh RUN_ROOT.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../../../../../" &>/dev/null && pwd)"

OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-base-k128-scratch-gbs64-s100-c500-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
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
export OPD_ID_TOPK=128
export OFFLINE_OPD_LOSS_TYPE=topk_reverse_kl

mkdir -p "${RUN_ROOT}"
{
   printf 'RUN_ROOT=%s\n' "${RUN_ROOT}"
   printf 'HF_CHECKPOINT=%s\n' "${HF_CHECKPOINT}"
   printf 'LOAD_PATH=%s\n' "${LOAD_PATH}"
   printf 'TEACHER_MODEL=%s\n' "${TEACHER_MODEL}"
   printf 'OPD_ID_TOPK=128\n'
   printf 'OFFLINE_OPD_LOSS_TYPE=topk_reverse_kl\n'
   printf 'PHASE1_CONCURRENCY=500\n'
   printf 'PHASE1_BATCH_SIZE=50\n'
   printf 'PHASE2_BATCH_SIZE=50\n'
   printf 'PHASE3_GLOBAL_BATCH_SIZE=64\n'
   printf 'PHASE3_TRAIN_STEPS=100\n'
   printf 'PHASE3_EFFECTIVE_EPOCHS=12.8\n'
   printf 'TP_SIZE=8\nPP_SIZE=1\nCP_SIZE=2\n'
   printf 'SAVE_INTERVAL=30\n'
   printf 'WANDB_RUN_ID=%s\n' "${OFFLINE_OPD_RUN_TAG}"
} > "${RUN_ROOT}/experiment.env"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
PIPELINE_RUNNING="${RUN_ROOT}/.pipeline.running"
PIPELINE_COMPLETE="${RUN_ROOT}/.pipeline.complete"
PIPELINE_FAILED="${RUN_ROOT}/.pipeline.failed"

if [[ "${1:-}" != "--worker" ]]; then
   nohup setsid bash "$0" --worker >> "${PIPELINE_LOG}" 2>&1 < /dev/null &
   pipeline_pid=$!
   printf '%s\n' "${pipeline_pid}" > "${RUN_ROOT}/pipeline.pid"
   echo "Base Top-128 scratch pipeline started: pid=${pipeline_pid}"
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

echo "[$(date -Is)] base Top-128 scratch experiment starts: ${RUN_ROOT}"

# Phase 1: exactly one base-model rollout over all 500 prompts. Keep the
# validated 50-sample on-disk grouping while maintaining 500 global requests.
run_phase phase1 env \
   RUN_ROOT="${RUN_ROOT}" \
   OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE}" \
   OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" \
   LOAD_PATH="${LOAD_PATH}" \
   OFFLINE_OPD_CONCURRENCY=500 \
   ROLLOUT_BATCH_SIZE=50 \
   PREFERRED_ROLLOUT_BATCH_SIZE=50 \
   bash "${SCRIPT_DIR}/01-phase1-rollout.sh" --foreground

# Phase 2: score the new base-model support with the teacher once. Sample
# artifacts remain local to this run and are never linked from an older run.
run_phase phase2 env \
   RUN_ROOT="${RUN_ROOT}" \
   OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE}" \
   OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" \
   LOAD_PATH="${LOAD_PATH}" \
   TEACHER_MODEL="${TEACHER_MODEL}" \
   ROLLOUT_BATCH_SIZE=50 \
   PHASE2_SAMPLES_PER_FILE=50 \
   PHASE2_BALANCE_MANIFESTS=1 \
   bash "${SCRIPT_DIR}/02-phase2-teacher.sh" --foreground

# Phase 3: concatenate shuffled scored epochs into exactly 100 full batches of
# 64 samples (effective 12.8 epochs over 500 samples), then train from base.
run_phase phase3 env \
   RUN_ROOT="${RUN_ROOT}" \
   OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE}" \
   OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" \
   LOAD_PATH="${LOAD_PATH}" \
   TRAIN_EPOCHS=13 \
   TRAIN_STEPS=100 \
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
   WANDB_GROUP="${WANDB_GROUP:-offline-opd-k128-base-scratch}" \
   WANDB_RUN_ID="${OFFLINE_OPD_RUN_TAG}" \
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground

printf 'completed=%s\n' "$(date -Is)" > "${PIPELINE_COMPLETE}"
rm -f "${PIPELINE_RUNNING}" "${PIPELINE_FAILED}"
trap - ERR INT TERM
echo "[$(date -Is)] base Top-128 scratch experiment completed: ${RUN_ROOT}"
