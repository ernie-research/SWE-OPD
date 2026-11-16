#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_glm_run phase2
require_glm_phase phase1
ensure_glm_token_bridge
export OPD_TEACHER_TEMPLATE_MODEL="${OPD_TEACHER_TEMPLATE_MODEL:-${TEACHER_MODEL}}"
export OPD_TEACHER_NATIVE_REASONING_PREFIX="${OPD_TEACHER_NATIVE_REASONING_PREFIX:-1}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-1}"
export PHASE2_SAMPLES_PER_FILE="${PHASE2_SAMPLES_PER_FILE:-1}"
export PHASE2_BALANCE_MANIFESTS="${PHASE2_BALANCE_MANIFESTS:-1}"
export PHASE2_DYNAMIC_SHARDING=1 PHASE2_SAMPLE_OUTPUT=1 PHASE1_SHUFFLE=0
export TEACHER_ATTENTION_MODE="${TEACHER_ATTENTION_MODE:-dsa}"
case "${TEACHER_ATTENTION_MODE}" in
   dsa)
      export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-2}"
      export SCORE_FILE_CONCURRENCY="${SCORE_FILE_CONCURRENCY:-2}"
      export TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.77}"
      export TEACHER_CHUNKED_PREFILL_SIZE="${TEACHER_CHUNKED_PREFILL_SIZE:-16384}"
      export TEACHER_MAX_PREFILL_TOKENS="${TEACHER_MAX_PREFILL_TOKENS:-16384}"
      export TEACHER_LOGITS_CHUNK="${TEACHER_LOGITS_CHUNK:-256}"
      ;;
   dense)
      # Match the validated remote GLM-5.2 setup.  Dense DSA dispatch is an
      # MHA_ONE_SHOT path, so scheduler-level chunking and concurrent prefills
      # must both be disabled.  FlashAttention still tiles internally.
      export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-1}"
      export SCORE_FILE_CONCURRENCY="${SCORE_FILE_CONCURRENCY:-1}"
      export TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.76}"
      export TEACHER_CHUNKED_PREFILL_SIZE="${TEACHER_CHUNKED_PREFILL_SIZE:--1}"
      export TEACHER_MAX_PREFILL_TOKENS="${TEACHER_MAX_PREFILL_TOKENS:-262144}"
      export TEACHER_MAX_RUNNING_REQUESTS="${TEACHER_MAX_RUNNING_REQUESTS:-1}"
      export TEACHER_KV_CACHE_DTYPE="${TEACHER_KV_CACHE_DTYPE:-bf16}"
      export TEACHER_DENSE_ATTN_KV_LEN_THRESHOLD="${TEACHER_DENSE_ATTN_KV_LEN_THRESHOLD:-131072}"
      export TEACHER_MAX_KV_CHUNK_CAPACITY="${TEACHER_MAX_KV_CHUNK_CAPACITY:-262144}"
      export TEACHER_NSA_SPARSE_PREFILL_Q_CHUNK="${TEACHER_NSA_SPARSE_PREFILL_Q_CHUNK:-8192}"
      export TEACHER_NSA_MQA_LOGITS_MAX_ROWS="${TEACHER_NSA_MQA_LOGITS_MAX_ROWS:-4096}"
      export TEACHER_MOE_TOKEN_CHUNK_SIZE="${TEACHER_MOE_TOKEN_CHUNK_SIZE:-16384}"
      export TEACHER_LOGITS_CHUNK="${TEACHER_LOGITS_CHUNK:-1024}"
      ;;
   *)
      echo "TEACHER_ATTENTION_MODE must be dsa or dense: ${TEACHER_ATTENTION_MODE}" >&2
      exit 1
      ;;
esac
export SCORE_MAX_ATTEMPTS="${SCORE_MAX_ATTEMPTS:-3}"
export TEACHER_TP="${TEACHER_TP:-16}"
export TEACHER_NNODES="${TEACHER_NNODES:-2}"
export TEACHER_REPLICAS="${TEACHER_REPLICAS:-2}"
export TEACHER_WATCHDOG_TIMEOUT="${TEACHER_WATCHDOG_TIMEOUT:-7200}"
export SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS=1
export OFFLINE_OPD_RUN_SCRIPT_OVERRIDE="${SCRIPT_DIR}/run-phase2-multinode-teacher.sh"
export USE_WANDB=0

# Dynamic claims are process-lifetime coordination state, not completed work.
# If Phase 2 is restarted after a scorer dies, a leftover claim can otherwise
# hide an unfinished rollout forever.  Rebuild only the claim/finalize state;
# per-sample artifacts and source manifests remain in OUT_DIR and are validated
# and skipped by score_teacher_offline.py on the resumed run.
SCORE_COORD_TOKEN="${RUN_TAG//\//_}"
export SCORE_COORD_DIR="${OUT_DIR}/.claims_${SCORE_COORD_TOKEN}"
PHASE2_FINALIZE_DIR="${OUT_DIR}/.finalize_${SCORE_COORD_TOKEN}"
case "${SCORE_COORD_DIR}" in
   "${RUN_ROOT}/scored/".claims_*) ;;
   *) echo "refusing to reset unexpected Phase-2 coordination dir: ${SCORE_COORD_DIR}" >&2; exit 1 ;;
esac
case "${PHASE2_FINALIZE_DIR}" in
   "${RUN_ROOT}/scored/".finalize_*) ;;
   *) echo "refusing to reset unexpected Phase-2 finalize dir: ${PHASE2_FINALIZE_DIR}" >&2; exit 1 ;;
esac
rm -rf -- "${SCORE_COORD_DIR}" "${PHASE2_FINALIZE_DIR}"
# Clear shared group state before mpirun starts. Clearing it inside each group
# leader races with secondary ranks, which can otherwise observe a stale
# failure marker from the previous attempt and abort a healthy restart.
rm -f -- "${RUN_ROOT}"/.phase2_teacher_group*.done "${RUN_ROOT}"/.phase2_teacher_group*.failed \
   "${RUN_ROOT}"/.phase2_teacher_group*.attempt "${RUN_ROOT}"/.phase2_teacher_group*.attempt.tmp.*

launch_glm_phase phase2 "${1:-}"
