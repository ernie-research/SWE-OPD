#!/bin/bash
set -euo pipefail

# Resume the sampled-token advantage experiment from its latest native
# Megatron checkpoint.  The W&B id is deliberately identical to the original
# run, so the newly added train/entropy series continues on the same chart.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"

export RUN_ROOT="${RUN_ROOT:-/path/to/offline_opd/glm45air-glm52-crossvocab-sampled-token-pg-t1k-high-full4093-a1-ep2-gbs64-20260902_181754}"
export RUN_TAG="$(basename "${RUN_ROOT}")"
export LOAD_PATH="${RUN_ROOT}/checkpoints"
export SAVE_PATH="${RUN_ROOT}/checkpoints"
export SCORED_DIR="${RUN_ROOT}/scored"
export FANOUT_DIR="${RUN_ROOT}/replay_epochs_x2_b64"
export OPD_TOKEN_ID_BRIDGE="${RUN_ROOT}/glm45air_to_glm52_token_ids.json"

[[ -f "${RUN_ROOT}/.phase2.complete" ]] || {
   echo "Phase 2 marker is missing: ${RUN_ROOT}/.phase2.complete" >&2
   exit 1
}
[[ -f "${LOAD_PATH}/latest_checkpointed_iteration.txt" ]] || {
   echo "checkpoint tracker is missing: ${LOAD_PATH}" >&2
   exit 1
}

export OFFLINE_OPD_LOSS_TYPE=cached_teacher_sampled_token_pg_with_student_anchor
export OPD_TEACHER_CACHE_TOPK=1000 OPD_DYNAMIC_SUPPORT_TOPK=128 OPD_ID_TOPK=128
export OPD_KL_COEF=1.0 OPD_STUDENT_ANCHOR_KL_COEF=1.0
export OPD_SAMPLED_TOKEN_CACHE_MISS_POLICY=anchor
export OPD_SAMPLED_TOKEN_EPS_CLIP=0.2 OPD_SAMPLED_TOKEN_EPS_CLIP_HIGH=0.2
# Keep the detached Student/Teacher Top-1/2/4/8 overlap and mass diagnostics
# enabled when resuming the sampled-token run.
export OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_POSTHOC_ANCHOR_PROTOCOL_TOKENS=0
export OFFLINE_OPD_POSTHOC_MASK_TOOL_FORMAT=0
export OFFLINE_OPD_POSTHOC_MASK_LET_ME_PROB=0

export TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1
unset TRAIN_STEPS
export ROLLOUT_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=8 PP_SIZE=2 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export SAVE_INTERVAL=20 PREFETCH_ROLLOUT_DATA=1
export OFFLINE_OPD_BALANCE_DATA=0 DISTRIBUTED_TIMEOUT_MINUTES=60
export OFFLINE_OPD_RESUME=1 OFFLINE_OPD_FINETUNE=0
export OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS=1

export USE_WANDB=1 WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-crossvocab-sampled-token-pg-t1k}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_TAG}}"
export WANDB_RESUME=must WANDB_MODE=online

export RUN_DIR="${RUN_ROOT}/logs-phase3-resume-entropy"
mkdir -p "${RUN_DIR}"

if [[ "${1:-}" != --worker ]]; then
   if [[ -s "${RUN_ROOT}/phase3-resume-entropy.pid" ]]; then
      old_pid="$(<"${RUN_ROOT}/phase3-resume-entropy.pid")"
      if [[ "${old_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
         echo "entropy resume is already running: pid=${old_pid}" >&2
         exit 1
      fi
   fi
   nohup setsid bash "$0" --worker > "${RUN_DIR}/launcher.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/phase3-resume-entropy.pid"
   echo "sampled-token Phase 3 entropy resume started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

printf 'pid=%s started=%s checkpoint=%s wandb_id=%s metric=train/entropy\n' \
   "$$" "$(date -Is)" "$(<"${LOAD_PATH}/latest_checkpointed_iteration.txt")" \
   "${WANDB_RUN_ID}" > "${RUN_ROOT}/.phase3-resume-entropy.running"
rm -f "${RUN_ROOT}/.phase3-resume-entropy.failed" \
   "${RUN_ROOT}/.phase3-resume-entropy.complete"
set +e
bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
rc=$?
set -e
if (( rc == 0 )); then
   printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.phase3-resume-entropy.complete"
   rm -f "${RUN_ROOT}/.phase3-resume-entropy.running"
else
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" \
      > "${RUN_ROOT}/.phase3-resume-entropy.failed"
   rm -f "${RUN_ROOT}/.phase3-resume-entropy.running"
fi
exit "${rc}"
