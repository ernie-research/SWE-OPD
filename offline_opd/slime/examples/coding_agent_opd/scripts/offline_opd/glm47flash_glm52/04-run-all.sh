#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_glm47_run phase1
ensure_shared_token_ids

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   DATASET_SIZE="$(python3 - "${PROMPT_DATA}" <<'PY'
import sys
with open(sys.argv[1], "rb") as stream:
    print(sum(bool(line.strip()) for line in stream))
PY
)"
   {
      printf 'RUN_ROOT=%s\nPROMPT_DATA=%s\nDATASET_SIZE=%s\n' "${RUN_ROOT}" "${PROMPT_DATA}" "${DATASET_SIZE}"
      printf 'STUDENT_MODEL=%s\nSTUDENT_TORCH_DIST=%s\nTEACHER_MODEL=%s\n' "${HF_CHECKPOINT}" "${LOAD_PATH}" "${TEACHER_MODEL}"
      printf 'LOSS=cached_teacher_dynamic_topk_reverse_kl\nFORWARD_KL=0\nSTUDENT_ANCHOR=0\n'
      printf 'STUDENT_CURRENT_TOPK=128\nTEACHER_RAW_CACHE_TOPK=1000\nMISS_POLICY=cutoff\n'
      printf 'TOKEN_ALIGNMENT=shared_exact_ids\nTOKEN_ID_BRIDGE=none\nSTUDENT_VALID_VOCAB_SIZE=154856\n'
      printf 'TEMPLATE_POLICY=exact_student_token_replay\nTEACHER_TEMPLATE_RERENDER=0\nFORMAT_TOKEN_MASK=0\n'
      printf 'PHASE1_TP=4\nPHASE1_CONCURRENCY=400\nPHASE1_FILE_BATCH=1\n'
      printf 'PHASE1_LOGPROB_PATH=original_logits_log_softmax\n'
      printf 'TOOL_CALL_PARSER=glm47\nREASONING_PARSER=glm45\n'
      printf 'PHASE2_ATTENTION=dense\nPHASE2_REPLICAS=2\nPHASE2_NNODES_PER_REPLICA=2\nPHASE2_TP=16\nPHASE2_SCORE_CONCURRENCY_PER_REPLICA=1\nPHASE2_FILE_CONCURRENCY_PER_REPLICA=4\n'
      printf 'PHASE3_TP=4\nPHASE3_PP=1\nPHASE3_CP=2\nPHASE3_EP=8\nPHASE3_ETP=1\nPHASE3_GBS=64\nPHASE3_EPOCHS=2\n'
      printf 'SAVE_INTERVAL=20\nLR=1e-6\n'
   } > "${RUN_ROOT}/experiment.env"
fi

# Keep the run manifest truthful when a compatible in-place resume picks up a
# newer numerical-stability fix.  The existing rollout artifacts remain valid,
# but all newly generated Student logprobs use the direct logits/log-softmax
# path recorded here.
if ! grep -qxF 'PHASE1_LOGPROB_PATH=original_logits_log_softmax' "${RUN_ROOT}/experiment.env"; then
   printf 'PHASE1_LOGPROB_PATH=original_logits_log_softmax\n' >> "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   if [[ -s "${RUN_ROOT}/pipeline.pid" ]]; then
      EXISTING_PIPELINE_PID="$(<"${RUN_ROOT}/pipeline.pid")"
      if [[ "${EXISTING_PIPELINE_PID}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${EXISTING_PIPELINE_PID}" 2>/dev/null; then
         echo "pipeline is already running: pid=${EXISTING_PIPELINE_PID}, run=${RUN_ROOT}" >&2
         exit 1
      fi
   fi
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "GLM-4.7-Flash/GLM-5.2 offline OPD pipeline started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

PIPELINE_LOCK="${RUN_ROOT}/.pipeline.lock"
if ! mkdir "${PIPELINE_LOCK}" 2>/dev/null; then
   echo "another pipeline worker owns ${PIPELINE_LOCK}" >&2
   exit 1
fi
cleanup_pipeline_lock() {
   rmdir "${PIPELINE_LOCK}" 2>/dev/null || true
}
trap cleanup_pipeline_lock EXIT

bash "${SCRIPT_DIR}/00-prepare-student.sh"
for entry in phase1:01-phase1-rollout.sh phase2:02-phase2-teacher.sh phase3:03-phase3-train.sh; do
   phase="${entry%%:*}"
   script="${entry#*:}"
   [[ -f "${RUN_ROOT}/.${phase}.complete" ]] && continue
   bash "${SCRIPT_DIR}/${script}" --foreground
done
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
