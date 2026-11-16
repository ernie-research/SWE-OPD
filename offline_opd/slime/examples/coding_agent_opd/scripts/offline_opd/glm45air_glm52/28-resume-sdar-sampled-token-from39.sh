#!/bin/bash
set -euo pipefail

# Resume the SDAR-gated sampled-token Phase 3 in-place from the latest native
# Megatron checkpoint.  Optimizer/RNG/progress are restored, so checkpoint 39
# continues at rollout/step 40.  Reuse the original W&B run id.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"

export RUN_ROOT="${RUN_ROOT:-/path/to/offline_opd/glm45air-glm52-crossvocab-sdar-sampled-token-pg-beta5-t1k-high-full4093-a1-ep2-gbs64-20260903_152208}"
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
[[ "$(<"${LOAD_PATH}/latest_checkpointed_iteration.txt")" == 39 ]] || {
   echo "Expected latest checkpoint iteration 39 under ${LOAD_PATH}" >&2
   exit 1
}
[[ -f "${LOAD_PATH}/iter_0000039/.metadata" && -f "${LOAD_PATH}/iter_0000039/common.pt" ]] || {
   echo "Checkpoint iter_0000039 is incomplete" >&2
   exit 1
}

export OFFLINE_OPD_LOSS_TYPE=cached_teacher_sampled_token_pg_with_student_anchor
export OPD_TEACHER_CACHE_TOPK=1000 OPD_DYNAMIC_SUPPORT_TOPK=128 OPD_ID_TOPK=128
export OPD_KL_COEF=1.0 OPD_STUDENT_ANCHOR_KL_COEF=1.0
export OPD_SAMPLED_TOKEN_CACHE_MISS_POLICY=anchor
export OPD_SAMPLED_TOKEN_EPS_CLIP=0.2 OPD_SAMPLED_TOKEN_EPS_CLIP_HIGH=0.2
export OPD_SAMPLED_TOKEN_SDAR_GATE=1 OPD_SDAR_GATE_BETA=5.0

export OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_POSTHOC_ANCHOR_PROTOCOL_TOKENS=0
export OFFLINE_OPD_POSTHOC_MASK_TOOL_FORMAT=0
export OFFLINE_OPD_POSTHOC_MASK_LET_ME_PROB=0

export TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1
unset TRAIN_STEPS START_ROLLOUT_ID
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
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-crossvocab-sdar-sampled-token-beta5-t1k}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_TAG}}"
export WANDB_RESUME=must WANDB_MODE=online

export RUN_DIR="${RUN_ROOT}/logs-phase3-resume-from39"
mkdir -p "${RUN_DIR}"

if [[ "${1:-}" != --worker ]]; then
   pid_file="${RUN_ROOT}/phase3-resume-from39.pid"
   if [[ -s "${pid_file}" ]]; then
      old_pid="$(<"${pid_file}")"
      if [[ "${old_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
         echo "SDAR resume is already running: pid=${old_pid}" >&2
         exit 1
      fi
   fi
   nohup setsid bash "$0" --worker > "${RUN_DIR}/launcher.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${pid_file}"
   echo "SDAR sampled-token Phase 3 resumed: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

marker="${RUN_ROOT}/.phase3-resume-from39"
printf 'pid=%s started=%s checkpoint=39 next_rollout=40 wandb_id=%s\n' \
   "$$" "$(date -Is)" "${WANDB_RUN_ID}" > "${marker}.running"
rm -f "${marker}.failed" "${marker}.complete"
set +e
bash "${OFFLINE_OPD_DIR}/start-offline-opd.sh" phase3
rc=$?
set -e
if (( rc == 0 )); then
   printf 'completed=%s\n' "$(date -Is)" > "${marker}.complete"
else
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${marker}.failed"
fi
rm -f "${marker}.running"
exit "${rc}"
