#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"

if [[ -z "${RUN_ROOT:-}" ]]; then
   [[ -s "${ITERATIVE_STATE_FILE}" ]] || {
      echo "no iterative run selected: ${ITERATIVE_STATE_FILE}" >&2
      exit 1
   }
   IFS= read -r RUN_ROOT < "${ITERATIVE_STATE_FILE}"
fi
validate_run_root "${RUN_ROOT}"

echo "run=${RUN_ROOT}"
if [[ -r "${RUN_ROOT}/pipeline.pid" ]]; then
   pid="$(<"${RUN_ROOT}/pipeline.pid")"
   if kill -0 "${pid}" 2>/dev/null; then
      echo "pipeline=running pid=${pid}"
   else
      echo "pipeline=not-running last_pid=${pid}"
   fi
fi
for (( cycle = 1; cycle <= ITERATIVE_CYCLES; cycle++ )); do
   dir="$(cycle_dir "${cycle}")"
   state="pending"
   [[ -f "${dir}/.cycle.running" ]] && state="running"
   [[ -f "${dir}/.cycle.failed" ]] && state="failed"
   [[ -f "${dir}/.cycle.complete" ]] && state="complete"
   phases=""
   for phase in phase1 phase2 phase3 hf; do
      [[ -f "${dir}/.${phase}.complete" ]] && phases+=" ${phase}"
   done
   echo "$(cycle_name "${cycle}"): ${state}; completed:${phases:- none}"
done
if [[ -f "${RUN_ROOT}/.pipeline.failed" ]]; then
   echo "failure=$(<"${RUN_ROOT}/.pipeline.failed")"
fi
if [[ -f "${RUN_ROOT}/pipeline.log" ]]; then
   echo "--- pipeline tail ---"
   tail -n 30 "${RUN_ROOT}/pipeline.log"
fi

