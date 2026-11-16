#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RUN_ROOT="${RUN_ROOT:-/path/to/offline_opd/glm47flash-glm52-student-template-reverse-t1k-s128-partial-n1280-s20-20260831_190200}"
POLL_SECONDS="${POLL_SECONDS:-1800}"
LOG_DIR="${RUN_ROOT}/logs-phase3-resume"
LOG_FILE="${LOG_DIR}/monitor.log"
PID_FILE="${RUN_ROOT}/phase3-resume-monitor.pid"
LOCK_DIR="${RUN_ROOT}/.phase3-resume.monitor.lock"
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" != --worker ]]; then
   if [[ -s "${PID_FILE}" ]]; then
      old_pid="$(<"${PID_FILE}")"
      if [[ "${old_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
         echo "resume monitor is already running: pid=${old_pid}" >&2
         exit 1
      fi
   fi
   nohup setsid env RUN_ROOT="${RUN_ROOT}" POLL_SECONDS="${POLL_SECONDS}" \
      bash "${SCRIPT_DIR}/08-monitor-phase3-resume.sh" --worker >> "${LOG_FILE}" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${PID_FILE}"
   echo "Phase-3 monitor started: pid=$!, poll=${POLL_SECONDS}s, log=${LOG_FILE}"
   exit 0
fi

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
   echo "another resume monitor owns ${LOCK_DIR}" >&2
   exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

while [[ ! -f "${RUN_ROOT}/.phase3-resume.complete" ]]; do
   sleep "${POLL_SECONDS}"

   set +e
   RUN_ROOT="${RUN_ROOT}" "${SCRIPT_DIR}/07-status-phase3-resume.sh"
   status_rc=$?
   set -e
   if (( status_rc != 0 )); then
      printf '[%s] status check returned rc=%s\n' "$(date -Is)" "${status_rc}"
   fi

   [[ ! -f "${RUN_ROOT}/.phase3-resume.complete" ]] || break

   queue_alive=0
   if [[ -s "${RUN_ROOT}/phase3-resume-queue.pid" ]]; then
      queue_pid="$(<"${RUN_ROOT}/phase3-resume-queue.pid")"
      if [[ "${queue_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${queue_pid}" 2>/dev/null; then
         queue_alive=1
      fi
   fi
   if (( queue_alive == 0 )); then
      printf '[%s] resume queue is not alive; restarting it\n' "$(date -Is)"
      RUN_ROOT="${RUN_ROOT}" POLL_SECONDS="${POLL_SECONDS}" \
         bash "${SCRIPT_DIR}/06-queue-resume-when-gpus-free.sh" || true
   fi
done

printf '[%s] Phase-3 monitor finished\n' "$(date -Is)"
