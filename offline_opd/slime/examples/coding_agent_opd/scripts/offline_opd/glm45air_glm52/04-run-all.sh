#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_glm_run phase1
ensure_glm_token_bridge
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
      printf 'LOSS=cached_teacher_dynamic_topk_reverse_kl\nSTUDENT_CURRENT_TOPK=128\nTEACHER_RAW_CACHE_TOPK=1000\nMISS_POLICY=cutoff\n'
      printf 'TOKEN_ID_BRIDGE=%s\nSTUDENT_VALID_VOCAB_SIZE=151365\n' "${OPD_TOKEN_ID_BRIDGE}"
      printf 'TEACHER_TEMPLATE_MODEL=%s\n' "${TEACHER_MODEL}"
      printf 'PHASE1_TP=4\nPHASE1_CONCURRENCY=100\nPHASE1_FILE_BATCH=1\n'
      printf 'TOOL_CALL_PARSER=glm45\nREASONING_PARSER=glm45\n'
      printf 'PHASE2_REPLICAS=2\nPHASE2_NNODES_PER_REPLICA=2\nPHASE2_TP=16\nPHASE2_SCORE_CONCURRENCY_PER_REPLICA=2\nPHASE2_TEACHER_MEM_FRACTION=0.77\nPHASE2_CHUNKED_PREFILL_SIZE=16384\nPHASE2_MAX_PREFILL_TOKENS=16384\n'
      printf 'PHASE3_TP=8\nPHASE3_PP=2\nPHASE3_CP=2\nPHASE3_EP=1\nPHASE3_ETP=8\nPHASE3_GBS=64\nPHASE3_EPOCHS=2\nSAVE_INTERVAL=20\nLR=1e-6\n'
   } > "${RUN_ROOT}/experiment.env"
fi
if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "GLM offline OPD pipeline started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi
bash "${SCRIPT_DIR}/00-prepare-student.sh"
for entry in phase1:01-phase1-rollout.sh phase2:02-phase2-teacher.sh phase3:03-phase3-train.sh; do
   phase="${entry%%:*}" script="${entry#*:}"
   [[ -f "${RUN_ROOT}/.${phase}.complete" ]] && continue
   bash "${SCRIPT_DIR}/${script}" --foreground
done
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
