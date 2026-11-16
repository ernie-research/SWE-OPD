#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_top128_run phase2
require_previous_phase phase1

# Score the cached K=128 student support exactly once.  Four nodes each run one
# TP4 x DP2 teacher; dynamic claiming balances long and short trajectories.
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-50}"
export PHASE2_SAMPLES_PER_FILE="${PHASE2_SAMPLES_PER_FILE:-50}"
# 1 rebalances manifests by trainable-token length; set to 0 to preserve the
# exact Phase-1 rollout_N batch grouping and in-batch order.
export PHASE2_BALANCE_MANIFESTS="${PHASE2_BALANCE_MANIFESTS:-1}"
export PHASE2_DYNAMIC_SHARDING=1
export PHASE2_SAMPLE_OUTPUT=1
export PHASE1_SHUFFLE=0
export PHASE1_SHUFFLE_SEED="${PHASE1_SHUFFLE_SEED:-42}"
export PHASE1_SHUFFLE_BALANCE_KEY="${PHASE1_SHUFFLE_BALANCE_KEY:-trainable_tokens}"
export TEACHER_TP="${TEACHER_TP:-4}"
export TEACHER_DP="${TEACHER_DP:-2}"
export TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3,4,5,6,7}"
export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-8}"
export SCORE_MAX_ATTEMPTS="${SCORE_MAX_ATTEMPTS:-3}"
export TEACHER_LOGITS_CHUNK="${TEACHER_LOGITS_CHUNK:-2048}"
export USE_WANDB=0

launch_top128_phase phase2 "${1:-}"
