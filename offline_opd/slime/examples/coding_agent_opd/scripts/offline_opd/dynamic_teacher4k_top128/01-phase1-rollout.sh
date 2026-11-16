#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_dynamic_run phase1
export OFFLINE_OPD_CONCURRENCY="${OFFLINE_OPD_CONCURRENCY:-500}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-50}"
export PREFERRED_ROLLOUT_BATCH_SIZE="${PREFERRED_ROLLOUT_BATCH_SIZE:-50}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((128 * 1024))}"
export MAX_GEN_LEN="${MAX_GEN_LEN:-$((16 * 1024))}"
export OFFLINE_OPD_SAVE_STUDENT_TOPK_LOGPROBS=1
export SGLANG_OUTPUT_TOP_LOGPROBS_B64=true
export USE_WANDB=0
launch_dynamic_phase phase1 "${1:-}"
