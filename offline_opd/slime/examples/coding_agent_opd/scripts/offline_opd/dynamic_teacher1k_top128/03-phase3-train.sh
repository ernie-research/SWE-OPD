#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"

: "${RUN_ROOT:?set RUN_ROOT}"
export SCORED_DIR="${RUN_ROOT}/scored"
export SAVE_PATH="${RUN_ROOT}/checkpoints"
export RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/SFT/model/Qwen3.5-9B}"
export LOAD_PATH="${LOAD_PATH:-/path/to/SFT/model/Qwen3.5-9B_torch_dist}"
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_dynamic_topk_reverse_kl
export OPD_ID_TOPK=128 OPD_DYNAMIC_SUPPORT_TOPK=128 OPD_TEACHER_CACHE_TOPK=1024
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}" EPOCH_SHUFFLE="${EPOCH_SHUFFLE:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
export NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
# Let the rebatcher derive ceil(4093 * 2 / 64) = 128 batches. The final six
# slots are filled from the following shuffled epoch. An explicit 128 is
# rejected because 128 * 64 is six samples beyond the exact two-epoch count.
unset TRAIN_STEPS
export TP_SIZE="${TP_SIZE:-8}" PP_SIZE="${PP_SIZE:-1}" CP_SIZE="${CP_SIZE:-2}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-131072}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-65536}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"
export OPD_TEACHER_CACHE_H2D_ROWS="${OPD_TEACHER_CACHE_H2D_ROWS:-512}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}" PREFETCH_ROLLOUT_DATA="${PREFETCH_ROLLOUT_DATA:-1}"
export OFFLINE_OPD_TOPK_OVERLAP=1 USE_WANDB="${USE_WANDB:-1}"
export WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}" WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-dynamic-t1k-s128-base-yiyan-ep2}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_TAG}}"
mkdir -p "${RUN_ROOT}/logs-phase3"
export RUN_DIR="${RUN_ROOT}/logs-phase3"
exec bash "${OFFLINE_OPD_DIR}/start-offline-opd.sh" phase3
