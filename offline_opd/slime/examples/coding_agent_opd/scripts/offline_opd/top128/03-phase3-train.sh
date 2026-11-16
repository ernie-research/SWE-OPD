#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_top128_run phase3
require_previous_phase phase2

# Replay the one-pass teacher cache for ten shuffled epochs.  This is the final
# validated K=1024 trainer layout, with only OPD_ID_TOPK changed to 128.
export SCORED_DIR="${RUN_ROOT}/scored"
# A run-scoped override allows a queued Phase 3 to be adjusted while Phase 1/2
# are still running, without restarting or rescoring either immutable cache.
TRAIN_EPOCHS_OVERRIDE_FILE="${RUN_ROOT}/train_epochs.override"
if [[ -f "${TRAIN_EPOCHS_OVERRIDE_FILE}" ]]; then
   IFS= read -r TRAIN_EPOCHS < "${TRAIN_EPOCHS_OVERRIDE_FILE}"
   export TRAIN_EPOCHS
   echo "Phase-3 epoch override: ${TRAIN_EPOCHS_OVERRIDE_FILE} -> ${TRAIN_EPOCHS}"
else
   export TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
fi
export EPOCH_SHUFFLE="${EPOCH_SHUFFLE:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-50}"
export NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE / NUM_STEPS_PER_ROLLOUT))}"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-2}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024 * 128))}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-30}"
export PREFETCH_ROLLOUT_DATA="${PREFETCH_ROLLOUT_DATA:-1}"
export OFFLINE_OPD_MANIFEST_LOAD_WORKERS="${OFFLINE_OPD_MANIFEST_LOAD_WORKERS:-8}"
export OFFLINE_OPD_TOPK_OVERLAP="${OFFLINE_OPD_TOPK_OVERLAP:-1}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-k128}"
export WANDB_MODE="${WANDB_MODE:-online}"

# Selecting this run's checkpoint root as LOAD_PATH means this is a resume, not
# a fresh replay.  Let Megatron restore the next rollout id from checkpoint.
if [[ "${LOAD_PATH}" == "${SAVE_PATH}" && -f "${LOAD_PATH}/latest_checkpointed_iteration.txt" ]]; then
   export OFFLINE_OPD_RESUME=1
else
   export OFFLINE_OPD_RESUME="${OFFLINE_OPD_RESUME:-0}"
fi

launch_top128_phase phase3 "${1:-}"
