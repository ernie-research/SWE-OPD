#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/OPD/offline_opd}"

# Select the directory once before detaching so every child phase is immune to
# changes in the shared "current run" pointer.
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-k128-run3-c500-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
[[ "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
   echo "RUN_ROOT must be under ${OFFLINE_OPD_BASE}: ${RUN_ROOT}" >&2
   exit 1
}
export RUN_ROOT
mkdir -p "${RUN_ROOT}"

PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
PIPELINE_RUNNING="${RUN_ROOT}/.pipeline.running"
PIPELINE_COMPLETE="${RUN_ROOT}/.pipeline.complete"
PIPELINE_FAILED="${RUN_ROOT}/.pipeline.failed"

if [[ "${1:-}" != "--worker" ]]; then
   # Start a new session as well as using nohup.  Some cluster command runners
   # terminate their whole original process group after the foreground shell
   # returns; nohup alone does not protect against that SIGKILL.
   nohup setsid bash "$0" --worker >> "${PIPELINE_LOG}" 2>&1 < /dev/null &
   pipeline_pid=$!
   printf '%s\n' "${pipeline_pid}" > "${RUN_ROOT}/pipeline.pid"
   echo "Top-128 continuous pipeline started: pid=${pipeline_pid}"
   echo "run=${RUN_ROOT}"
   echo "log=${PIPELINE_LOG}"
   exit 0
fi

printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(date -Is)" > "${PIPELINE_RUNNING}"
rm -f "${PIPELINE_FAILED}"

pipeline_failed() {
   local rc=$?
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${PIPELINE_FAILED}"
   rm -f "${PIPELINE_RUNNING}"
   exit "${rc}"
}
trap pipeline_failed ERR INT TERM

echo "[$(date -Is)] continuous Top-128 run starts: ${RUN_ROOT}"

for entry in \
   "phase1:01-phase1-rollout.sh" \
   "phase2:02-phase2-teacher.sh" \
   "phase3:03-phase3-train.sh"; do
   phase="${entry%%:*}"
   script="${entry#*:}"
   if [[ -f "${RUN_ROOT}/.${phase}.complete" ]]; then
      echo "[$(date -Is)] ${phase} already complete; skipping"
      continue
   fi
   phase_log_dir="${RUN_ROOT}/logs-${phase}"
   mkdir -p "${phase_log_dir}"
   echo "[$(date -Is)] ${phase} starts; log=${phase_log_dir}/launcher.log"
   bash "${SCRIPT_DIR}/${script}" --foreground > "${phase_log_dir}/launcher.log" 2>&1
   echo "[$(date -Is)] ${phase} completed"
done

printf 'completed=%s\n' "$(date -Is)" > "${PIPELINE_COMPLETE}"
rm -f "${PIPELINE_RUNNING}" "${PIPELINE_FAILED}"
trap - ERR INT TERM
echo "[$(date -Is)] continuous Top-128 run completed: ${RUN_ROOT}"
