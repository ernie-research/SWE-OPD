#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
select_dynamic_run phase1
if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   echo "dynamic teacher4k/current128 pipeline started: pid=$!, run=${RUN_ROOT}"
   exit
fi
for entry in phase1:01-phase1-rollout.sh phase2:02-phase2-teacher.sh phase3:03-phase3-train.sh; do
   phase="${entry%%:*}" script="${entry#*:}"
   [[ -f "${RUN_ROOT}/.${phase}.complete" ]] && continue
   bash "${SCRIPT_DIR}/${script}" --foreground
done
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
