#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_glm_run phase3
require_glm_phase phase2
ensure_glm_token_bridge
export SCORED_DIR="${RUN_ROOT}/scored"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}" EPOCH_SHUFFLE="${EPOCH_SHUFFLE:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE / NUM_STEPS_PER_ROLLOUT))}"
export TP_SIZE="${TP_SIZE:-8}" PP_SIZE="${PP_SIZE:-2}" CP_SIZE="${CP_SIZE:-2}"
export EXPERT_MP_SIZE="${EXPERT_MP_SIZE:-1}" EXPERT_TP_SIZE="${EXPERT_TP_SIZE:-8}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((128 * 1024))}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-4096}"
export OPD_TEACHER_CACHE_H2D_ROWS="${OPD_TEACHER_CACHE_H2D_ROWS:-512}"
# A GLM-4.5-Air optimizer checkpoint is about 1.4 TiB.  On shared storage,
# faster ranks can reach the final save barrier more than 10 minutes before
# the slowest rank, so Megatron's default timeout can abort an otherwise
# healthy checkpoint write.
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}" PREFETCH_ROLLOUT_DATA="${PREFETCH_ROLLOUT_DATA:-1}"
export OFFLINE_OPD_TOPK_OVERLAP="${OFFLINE_OPD_TOPK_OVERLAP:-1}" USE_WANDB="${USE_WANDB:-1}"
export WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}" WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-student-glm52-teacher-t1k-current128}"
export WANDB_MODE="${WANDB_MODE:-online}"
launch_glm_phase phase3 "${1:-}"
