#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_glm47_run phase1
ensure_shared_token_ids

# Eight TP4 engines on 32 H800s.  Agent trajectories spend most of their time
# waiting on tools, so keep substantially more samples in flight than the old
# GLM-4.5-Air run while retaining a conservative 50-request engine semaphore.
export OFFLINE_OPD_CONCURRENCY="${OFFLINE_OPD_CONCURRENCY:-400}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export PREFERRED_ROLLOUT_BATCH_SIZE="${PREFERRED_ROLLOUT_BATCH_SIZE:-1}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-4}"
export TP_SIZE="${TP_SIZE:-4}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((128 * 1024))}"
export MAX_GEN_LEN="${MAX_GEN_LEN:-$((16 * 1024))}"
export OFFLINE_OPD_SAVE_STUDENT_TOPK_LOGPROBS=1
export OFFLINE_OPD_VALID_VOCAB_SIZE=154856
export OFFLINE_OPD_TOPK_REQUEST_EXTRA=24
# Compute rollout-policy logprobs directly with log_softmax on the original
# logits.  SGLang's default softmax-then-log path can underflow low-probability
# Top-K entries to -inf, which cannot be persisted in the float16 snapshot.
export SGLANG_RETURN_ORIGINAL_LOGPROB=true
export SGLANG_OUTPUT_TOP_LOGPROBS_B64=true
export SGLANG_ATTN_BACKEND="${SGLANG_ATTN_BACKEND:-fa3}"
export USE_WANDB=0
launch_glm47_phase phase1 "${1:-}"
