#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"

export RUN_ROOT="${RUN_ROOT:-/path/to/offline_opd/glm47flash-glm52-student-template-reverse-t1k-s128-partial-n1280-s20-20260831_190200}"
export MAIN_RUN="${MAIN_RUN:-/path/to/offline_opd/glm47flash-glm52-student-template-reverse-t1k-s128-20260831_014946}"
export RESUME_DATA_DIR="${RESUME_DATA_DIR:-${RUN_ROOT}/scored-resume-full-ep2}"
export SAVE_PATH="${RUN_ROOT}/checkpoints" LOAD_PATH="${RUN_ROOT}/checkpoints"
export RUN_TAG="$(basename "${RUN_ROOT}")"

[[ -f "${MAIN_RUN}/.phase2.complete" ]] || { echo "full Phase 2 is incomplete: ${MAIN_RUN}" >&2; exit 1; }
[[ -f "${LOAD_PATH}/latest_checkpointed_iteration.txt" ]] || { echo "checkpoint tracker is missing: ${LOAD_PATH}" >&2; exit 1; }
ensure_shared_token_ids

"${SLIME_DIR}/.slime_venv/bin/python3" "${SCRIPT_DIR}/build_resume_stream.py" \
   --full-scored-dir "${MAIN_RUN}/scored" \
   --prefix-dir "${RUN_ROOT}/scored" \
   --output-dir "${RESUME_DATA_DIR}" \
   --epochs 2 --batch-size 64 --seed 42

export SCORED_DIR="${RESUME_DATA_DIR}"
export TRAIN_EPOCHS=1 EPOCH_SHUFFLE=0
export ROLLOUT_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=4 PP_SIZE=1 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export DISTRIBUTED_TIMEOUT_MINUTES=60 SAVE_INTERVAL=20 PREFETCH_ROLLOUT_DATA=1
export OFFLINE_OPD_TOPK_OVERLAP=1 OFFLINE_OPD_RESUME=1 OFFLINE_OPD_FINETUNE=0
export OFFLINE_OPD_OVERRIDE_OPT_PARAM_SCHEDULER=1
export USE_WANDB=1 WANDB_TEAM=czguocs-baidu WANDB_PROJECT=slime-OPD
export WANDB_GROUP=offline-opd-glm47flash-student-glm52-teacher-student-template-reverse-t1k-s128
export WANDB_RUN_ID="${RUN_TAG}" WANDB_RESUME=must WANDB_MODE=online
export RUN_DIR="${RUN_ROOT}/logs-phase3-resume"
mkdir -p "${RUN_DIR}"

if [[ "${1:-}" != --worker ]]; then
   if [[ -s "${RUN_ROOT}/phase3-resume.pid" ]]; then
      old_pid="$(<"${RUN_ROOT}/phase3-resume.pid")"
      if [[ "${old_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
         echo "resume is already running: pid=${old_pid}" >&2
         exit 1
      fi
   fi
   nohup setsid bash "$0" --worker > "${RUN_DIR}/launcher.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/phase3-resume.pid"
   echo "GLM-4.7-Flash Phase 3 resume started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

printf 'pid=%s started=%s\n' "$$" "$(date -Is)" > "${RUN_ROOT}/.phase3-resume.running"
rm -f "${RUN_ROOT}/.phase3-resume.failed" "${RUN_ROOT}/.phase3-resume.complete"
set +e
bash "${OFFLINE_OPD_DIR}/start-offline-opd.sh" phase3
rc=$?
set -e
if (( rc == 0 )); then
   printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.phase3-resume.complete"
   rm -f "${RUN_ROOT}/.phase3-resume.running"
else
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${RUN_ROOT}/.phase3-resume.failed"
   rm -f "${RUN_ROOT}/.phase3-resume.running"
fi
exit "${rc}"
