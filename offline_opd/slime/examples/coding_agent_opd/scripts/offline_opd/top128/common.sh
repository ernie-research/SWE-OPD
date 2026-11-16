#!/bin/bash
set -euo pipefail

# Shared, validated configuration for the second offline-OPD run.  The generic
# implementation stays in ../; these wrappers only pin the experiment recipe
# and keep all three phases on the same persistent run directory.

TOP128_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_DIR="$(cd -- "${TOP128_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"

export OPD_ID_TOPK=128
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/workspace/env_run/<user>/Megatron-LM}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/SFT/results/qwen35-9b-scaleswe5000-sft-246step-add-finish/checkpoint-246}"
export LOAD_PATH="${LOAD_PATH:-/path/to/OPD/model/qwen35-9b-scaleswe5000-sft-246-add-finish_torch_dist}"
export TEACHER_MODEL="${TEACHER_MODEL:-/path/to/SFT/model/Qwen3.6-27B}"
export PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/data/swe_train.jsonl}"

OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/OPD/offline_opd}"
TOP128_STATE_FILE="${TOP128_STATE_FILE:-${OFFLINE_OPD_BASE}/.qwen35-9b-k128-current}"

_validate_run_root() {
   local run_root="$1"
   [[ "${run_root}" == "${OFFLINE_OPD_BASE}"/* ]] || {
      echo "RUN_ROOT must be under ${OFFLINE_OPD_BASE}: ${run_root}" >&2
      return 1
   }
}

# Phase 1 creates a timestamped run by default.  A failed/repeated invocation
# resumes the same run through TOP128_STATE_FILE.  Set NEW_RUN=1 to deliberately
# start another run, or RUN_ROOT=/absolute/path to select one explicitly.
select_top128_run() {
   local phase="$1"
   if [[ -n "${RUN_ROOT:-}" ]]; then
      _validate_run_root "${RUN_ROOT}"
   elif [[ "${phase}" == "phase1" && "${NEW_RUN:-0}" == "1" ]]; then
      RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-k128-run3-c500-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
   elif [[ -s "${TOP128_STATE_FILE}" ]]; then
      IFS= read -r RUN_ROOT < "${TOP128_STATE_FILE}"
      _validate_run_root "${RUN_ROOT}"
   elif [[ "${phase}" == "phase1" ]]; then
      RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-k128-run3-c500-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
   else
      echo "No Top-128 run selected. Run 01-phase1-rollout.sh first or export RUN_ROOT." >&2
      return 1
   fi

   mkdir -p "${RUN_ROOT}" "$(dirname "${TOP128_STATE_FILE}")"
   printf '%s\n' "${RUN_ROOT}" > "${TOP128_STATE_FILE}.tmp.$$"
   mv "${TOP128_STATE_FILE}.tmp.$$" "${TOP128_STATE_FILE}"

   export RUN_ROOT
   # Do not inherit a stale RUN_TAG from an earlier K=1024 shell: it is used by
   # Phase-2 coordination and as the Phase-3 W&B run id.
   export RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
   export DUMP_DIR="${RUN_ROOT}/dump"
   export IN_DIR="${DUMP_DIR}"
   export OUT_DIR="${RUN_ROOT}/scored"
   export SAVE_PATH="${RUN_ROOT}/checkpoints"
}

require_previous_phase() {
   local previous="$1"
   local marker="${RUN_ROOT}/.${previous}.complete"
   if [[ ! -f "${marker}" && "${ALLOW_INCOMPLETE_PREVIOUS_PHASE:-0}" != "1" ]]; then
      echo "Missing completion marker ${marker}. Wait for ${previous}, or set ALLOW_INCOMPLETE_PREVIOUS_PHASE=1 only for a manually verified run." >&2
      return 1
   fi
}

launch_top128_phase() {
   local phase="$1"
   local mode="${2:-}"
   local phase_number="${phase#phase}"
   local log_dir="${OFFLINE_OPD_PHASE_LOG_DIR:-${RUN_ROOT}/logs-phase${phase_number}}"
   local launcher_log="${log_dir}/launcher.log"
   local running_marker="${RUN_ROOT}/.${phase}.running"
   local complete_marker="${RUN_ROOT}/.${phase}.complete"
   local failed_marker="${RUN_ROOT}/.${phase}.failed"
   mkdir -p "${log_dir}"

   export RUN_DIR="${log_dir}"

   if [[ "${mode}" != "--foreground" ]]; then
      nohup bash "$0" --foreground > "${launcher_log}" 2>&1 < /dev/null &
      echo "Top-128 ${phase} started: pid=$!"
      echo "run=${RUN_ROOT}"
      echo "log=${launcher_log}"
      return 0
   fi

   printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(date -Is)" > "${running_marker}"
   rm -f "${failed_marker}"
   set +e
   bash "${OFFLINE_OPD_DIR}/start-offline-opd.sh" "${phase}"
   local rc=$?
   set -e
   if (( rc == 0 )); then
      printf 'completed=%s log=%s\n' "$(date -Is)" "${launcher_log}" > "${complete_marker}"
      rm -f "${running_marker}" "${failed_marker}"
   else
      printf 'rc=%s failed=%s log=%s\n' "${rc}" "$(date -Is)" "${launcher_log}" > "${failed_marker}"
      rm -f "${running_marker}"
   fi
   return "${rc}"
}
