#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_glm47_run phase3
require_glm47_phase phase2
ensure_shared_token_ids

export SCORED_DIR="${RUN_ROOT}/scored"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}" EPOCH_SHUFFLE="${EPOCH_SHUFFLE:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE / NUM_STEPS_PER_ROLLOUT))}"

# Eight-way dense model parallelism (TP4 x CP2), DP4 over 32 GPUs.  PP1 is
# sufficient for the much smaller 30B-A3B Student; experts stay EP8.
export TP_SIZE="${TP_SIZE:-4}" PP_SIZE="${PP_SIZE:-1}" CP_SIZE="${CP_SIZE:-2}"
export EXPERT_MP_SIZE="${EXPERT_MP_SIZE:-8}" EXPERT_TP_SIZE="${EXPERT_TP_SIZE:-1}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((128 * 1024))}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-4096}"
export OPD_TEACHER_CACHE_H2D_ROWS="${OPD_TEACHER_CACHE_H2D_ROWS:-512}"
export DISTRIBUTED_TIMEOUT_MINUTES="${DISTRIBUTED_TIMEOUT_MINUTES:-60}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}" PREFETCH_ROLLOUT_DATA="${PREFETCH_ROLLOUT_DATA:-1}"
export OFFLINE_OPD_TOPK_OVERLAP="${OFFLINE_OPD_TOPK_OVERLAP:-1}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}" WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm47flash-student-glm52-teacher-student-template-reverse-t1k-s128}"
export WANDB_MODE="${WANDB_MODE:-online}"
launch_glm47_phase phase3 "${1:-}"
