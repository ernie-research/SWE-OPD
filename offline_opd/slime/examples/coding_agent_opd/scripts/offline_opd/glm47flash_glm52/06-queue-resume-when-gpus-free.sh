#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RUN_ROOT="${RUN_ROOT:-/path/to/offline_opd/glm47flash-glm52-student-template-reverse-t1k-s128-partial-n1280-s20-20260831_190200}"
POLL_SECONDS="${POLL_SECONDS:-1800}"
LOG_DIR="${RUN_ROOT}/logs-phase3-resume"
LOG_FILE="${LOG_DIR}/queue.log"
LOCK_DIR="${RUN_ROOT}/.phase3-resume.queue.lock"
mkdir -p "${LOG_DIR}"

if [[ "${1:-}" != --worker ]]; then
   if [[ -s "${RUN_ROOT}/phase3-resume-queue.pid" ]]; then
      old_pid="$(<"${RUN_ROOT}/phase3-resume-queue.pid")"
      if [[ "${old_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${old_pid}" 2>/dev/null; then
         echo "resume queue is already running: pid=${old_pid}" >&2
         exit 1
      fi
   fi
   nohup setsid bash "$0" --worker >> "${LOG_FILE}" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/phase3-resume-queue.pid"
   echo "Phase-3 resume queued: pid=$!, poll=${POLL_SECONDS}s, log=${LOG_FILE}"
   exit 0
fi

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
   echo "another resume queue owns ${LOCK_DIR}" >&2
   exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

while [[ ! -f "${RUN_ROOT}/.phase3-resume.complete" ]]; do
   # These are cGPU pods: nvidia-smi includes co-tenants on the physical H800s,
   # while this job retains its own 60.9-GiB / 75%-core slice on every GPU.
   # Waiting for physical memory to reach zero would therefore never launch.
   printf '[%s] starting resume in this job cGPU allocation\n' "$(date -Is)"
   if bash "${SCRIPT_DIR}/05-resume-full-from-partial.sh" --worker; then
      printf '[%s] Phase-3 resume completed\n' "$(date -Is)"
      exit 0
   fi
   printf '[%s] resume exited unsuccessfully; retrying in %s seconds from the latest checkpoint\n' \
      "$(date -Is)" "${POLL_SECONDS}"
   sleep "${POLL_SECONDS}"
done
