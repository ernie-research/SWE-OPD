#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_top128_run phase1

# Same Phase-1 recipe as the validated K=1024 run: 500 globally in-flight
# trajectories, 16 TP2 rollout engines, and 50 samples in each persisted file.
export OFFLINE_OPD_CONCURRENCY="${OFFLINE_OPD_CONCURRENCY:-500}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-50}"
export PREFERRED_ROLLOUT_BATCH_SIZE="${PREFERRED_ROLLOUT_BATCH_SIZE:-50}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024 * 128))}"
export MAX_GEN_LEN="${MAX_GEN_LEN:-$((1024 * 16))}"
export OFFLINE_OPD_SAVE_STUDENT_TOPK_LOGPROBS=1
export SGLANG_OUTPUT_TOP_LOGPROBS_B64=true
export USE_WANDB="${USE_WANDB:-0}"

launch_top128_phase phase1 "${1:-}"
