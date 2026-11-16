#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_dynamic_run phase2
require_dynamic_phase phase1
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-50}"
export PHASE2_SAMPLES_PER_FILE="${PHASE2_SAMPLES_PER_FILE:-50}"
export PHASE2_BALANCE_MANIFESTS="${PHASE2_BALANCE_MANIFESTS:-1}"
export PHASE2_DYNAMIC_SHARDING=1 PHASE2_SAMPLE_OUTPUT=1 PHASE1_SHUFFLE=0
export TEACHER_TP="${TEACHER_TP:-4}" TEACHER_DP="${TEACHER_DP:-2}"
export TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3,4,5,6,7}"
export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-8}"
export SCORE_FILE_CONCURRENCY="${SCORE_FILE_CONCURRENCY:-${SCORE_CONCURRENCY}}"
export SCORE_MAX_ATTEMPTS="${SCORE_MAX_ATTEMPTS:-3}"
export TEACHER_LOGITS_CHUNK="${TEACHER_LOGITS_CHUNK:-512}"
export TEACHER_WATCHDOG_TIMEOUT="${TEACHER_WATCHDOG_TIMEOUT:-7200}"
# Dynamic cache only: compute/return teacher Top-4096 at trainable response
# positions carried by token_ids_logprob_b64.  Standard offline/online OPD leave
# this unset and retain the original all-row behavior.
export SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS=1
# The Phase-3 objective decides what Phase 2 stores:
#   topk_kl (default): teacher Top-OPD_TEACHER_CACHE_TOPK cache at trainable rows.
#   sampled_token: token_reverse_kl artifact -- the teacher's log-prob of each
#     sampled token, no Top-K; build_chunk_scored.sh with CHUNK_ALIGNER=identity
#     turns it into the chunk artifact.
case "${OPD_PHASE3_OBJECTIVE:-topk_kl}" in
   topk_kl) ;;
   sampled_token)
      export OFFLINE_OPD_LOSS_TYPE=token_reverse_kl
      unset SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS
      ;;
   *) echo "OPD_PHASE3_OBJECTIVE must be topk_kl or sampled_token" >&2; exit 1 ;;
esac
export USE_WANDB=0
launch_dynamic_phase phase2 "${1:-}"
