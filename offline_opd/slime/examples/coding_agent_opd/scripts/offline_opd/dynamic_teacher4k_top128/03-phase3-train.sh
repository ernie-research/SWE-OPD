#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_dynamic_run phase3
require_dynamic_phase phase2
export SCORED_DIR="${RUN_ROOT}/scored"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}" EPOCH_SHUFFLE="${EPOCH_SHUFFLE:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE / NUM_STEPS_PER_ROLLOUT))}"
export TP_SIZE="${TP_SIZE:-8}" PP_SIZE="${PP_SIZE:-1}" CP_SIZE="${CP_SIZE:-2}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((128 * 1024))}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"
export OPD_TEACHER_CACHE_H2D_ROWS="${OPD_TEACHER_CACHE_H2D_ROWS:-512}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-30}" PREFETCH_ROLLOUT_DATA="${PREFETCH_ROLLOUT_DATA:-1}"
export OFFLINE_OPD_TOPK_OVERLAP="${OFFLINE_OPD_TOPK_OVERLAP:-1}" USE_WANDB="${USE_WANDB:-1}"
export WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}" WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-dynamic-teacher4k-current128}"
export WANDB_MODE="${WANDB_MODE:-online}"
launch_dynamic_phase phase3 "${1:-}"
