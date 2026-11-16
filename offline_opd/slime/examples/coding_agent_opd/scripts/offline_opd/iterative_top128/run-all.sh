#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ -z "${RUN_ROOT:-}" ]]; then
   if [[ "${NEW_RUN:-0}" == "1" || ! -s "${ITERATIVE_STATE_FILE}" ]]; then
      RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-k128-refresh${ITERATIVE_CYCLES}x${EPOCHS_PER_CYCLE}-c${OFFLINE_OPD_CONCURRENCY:-500}-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
   else
      IFS= read -r RUN_ROOT < "${ITERATIVE_STATE_FILE}"
   fi
fi
validate_run_root "${RUN_ROOT}"
export RUN_ROOT
mkdir -p "${RUN_ROOT}" "$(dirname "${ITERATIVE_STATE_FILE}")"
printf '%s\n' "${RUN_ROOT}" > "${ITERATIVE_STATE_FILE}.tmp.$$"
mv "${ITERATIVE_STATE_FILE}.tmp.$$" "${ITERATIVE_STATE_FILE}"

# Every two-epoch training phase belongs to one logical experiment.  Persist a
# single W&B run id beside the pipeline state so retries and later cycles append
# to the same run instead of creating five unrelated runs.  An existing run may
# be adopted by passing ITERATIVE_WANDB_RUN_ID once when resuming.
WANDB_RUN_ID_FILE="${RUN_ROOT}/wandb.run_id"
if [[ -n "${ITERATIVE_WANDB_RUN_ID:-}" ]]; then
   if [[ -s "${WANDB_RUN_ID_FILE}" ]]; then
      EXISTING_WANDB_RUN_ID="$(<"${WANDB_RUN_ID_FILE}")"
      [[ "${EXISTING_WANDB_RUN_ID}" == "${ITERATIVE_WANDB_RUN_ID}" ]] || {
         echo "W&B run id mismatch: persisted=${EXISTING_WANDB_RUN_ID}, requested=${ITERATIVE_WANDB_RUN_ID}" >&2
         exit 1
      }
   else
      printf '%s\n' "${ITERATIVE_WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}.tmp.$$"
      mv "${WANDB_RUN_ID_FILE}.tmp.$$" "${WANDB_RUN_ID_FILE}"
   fi
elif [[ -s "${WANDB_RUN_ID_FILE}" ]]; then
   ITERATIVE_WANDB_RUN_ID="$(<"${WANDB_RUN_ID_FILE}")"
else
   ITERATIVE_WANDB_RUN_ID="$(python3 - <<'PY'
import secrets
import string

alphabet = string.ascii_lowercase + string.digits
print("".join(secrets.choice(alphabet) for _ in range(8)))
PY
)"
   printf '%s\n' "${ITERATIVE_WANDB_RUN_ID}" > "${WANDB_RUN_ID_FILE}.tmp.$$"
   mv "${WANDB_RUN_ID_FILE}.tmp.$$" "${WANDB_RUN_ID_FILE}"
fi
export ITERATIVE_WANDB_RUN_ID

PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
PIPELINE_PID="${RUN_ROOT}/pipeline.pid"
PIPELINE_RUNNING="${RUN_ROOT}/.pipeline.running"
PIPELINE_COMPLETE="${RUN_ROOT}/.pipeline.complete"
PIPELINE_FAILED="${RUN_ROOT}/.pipeline.failed"

if [[ "${1:-}" != "--worker" ]]; then
   nohup setsid env RUN_ROOT="${RUN_ROOT}" bash "$0" --worker \
      > "${PIPELINE_LOG}" 2>&1 < /dev/null &
   pipeline_pid=$!
   printf '%s\n' "${pipeline_pid}" > "${PIPELINE_PID}"
   echo "Iterative Top-128 OPD started: pid=${pipeline_pid}"
   echo "run=${RUN_ROOT}"
   echo "schedule=${ITERATIVE_CYCLES} cycles x ${EPOCHS_PER_CYCLE} epochs = $(( ITERATIVE_CYCLES * EPOCHS_PER_CYCLE )) epochs"
   echo "log=${PIPELINE_LOG}"
   exit 0
fi

if [[ -f "${PIPELINE_COMPLETE}" ]]; then
   echo "pipeline already complete: ${RUN_ROOT}"
   exit 0
fi
printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(date -Is)" > "${PIPELINE_RUNNING}"
rm -f "${PIPELINE_FAILED}"
pipeline_exit() {
   local rc=$?
   if (( rc != 0 )); then
      printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${PIPELINE_FAILED}"
      rm -f "${PIPELINE_RUNNING}"
   fi
}
trap pipeline_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cat > "${RUN_ROOT}/experiment.env" <<EOF
RUN_ROOT=${RUN_ROOT}
LOSS_TYPE=${OFFLINE_OPD_LOSS_TYPE}
TOPK=${OPD_ID_TOPK}
CYCLES=${ITERATIVE_CYCLES}
EPOCHS_PER_CYCLE=${EPOCHS_PER_CYCLE}
TOTAL_EPOCHS=$(( ITERATIVE_CYCLES * EPOCHS_PER_CYCLE ))
INITIAL_HF=${INITIAL_HF_CHECKPOINT}
INITIAL_MEGATRON=${INITIAL_LOAD_PATH}
TEACHER=${TEACHER_MODEL}
PROMPT_DATA=${PROMPT_DATA}
ROLLOUT_CONCURRENCY=${OFFLINE_OPD_CONCURRENCY:-500}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-50}
PHASE2_BALANCE_MANIFESTS=${PHASE2_BALANCE_MANIFESTS:-1}
ROLLOUT_ENGINE_PORT_BASE_ROOT=${SLIME_ROLLOUT_ENGINE_PORT_BASE_ROOT}
ROLLOUT_ENGINE_PORT_STRIDE=${SLIME_ROLLOUT_ENGINE_PORT_STRIDE}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-50}
WANDB_RUN_ID=${ITERATIVE_WANDB_RUN_ID}
EOF

echo "[$(date -Is)] iterative Top-128 pipeline starts: ${RUN_ROOT}"
echo "[$(date -Is)] shared W&B run id: ${ITERATIVE_WANDB_RUN_ID}"
for (( cycle = 1; cycle <= ITERATIVE_CYCLES; cycle++ )); do
   RUN_ROOT="${RUN_ROOT}" bash "${SCRIPT_DIR}/run-cycle.sh" "${cycle}"
done

write_marker "${PIPELINE_COMPLETE}" \
   "cycles=${ITERATIVE_CYCLES} epochs_per_cycle=${EPOCHS_PER_CYCLE} total_epochs=$(( ITERATIVE_CYCLES * EPOCHS_PER_CYCLE ))"
rm -f "${PIPELINE_RUNNING}" "${PIPELINE_FAILED}"
trap - EXIT INT TERM
echo "[$(date -Is)] iterative Top-128 pipeline completed: ${RUN_ROOT}"
