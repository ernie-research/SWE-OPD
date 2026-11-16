#!/bin/bash
set -euo pipefail

DYNAMIC_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_DIR="$(cd -- "${DYNAMIC_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"

export OPD_ID_TOPK=128
export OPD_DYNAMIC_SUPPORT_TOPK=128
# Overridable so 07 can store the teacher Top-1024 directly.
export OPD_TEACHER_CACHE_TOPK="${OPD_TEACHER_CACHE_TOPK:-4096}"
# Overridable so the chunk sampled-token recipe (06) can reuse 03-phase3-train.sh.
export OFFLINE_OPD_LOSS_TYPE="${OFFLINE_OPD_LOSS_TYPE:-cached_teacher_dynamic_topk_reverse_kl}"
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/workspace/env_run/<user>/Megatron-LM}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/SFT/results/qwen35-9b-scaleswe5000-sft-246step-add-finish/checkpoint-246}"
export LOAD_PATH="${LOAD_PATH:-/path/to/OPD/model/qwen35-9b-scaleswe5000-sft-246-add-finish_torch_dist}"
export TEACHER_MODEL="${TEACHER_MODEL:-/path/to/SFT/model/Qwen3.6-27B}"
export PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/data/swe_train.jsonl}"

OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"
DYNAMIC_STATE_FILE="${DYNAMIC_STATE_FILE:-${OFFLINE_OPD_BASE}/.qwen35-9b-dynamic-t4k-s128-current}"

select_dynamic_run() {
   local phase="$1"
   if [[ -z "${RUN_ROOT:-}" && -s "${DYNAMIC_STATE_FILE}" && "${NEW_RUN:-0}" != 1 ]]; then
      IFS= read -r RUN_ROOT < "${DYNAMIC_STATE_FILE}"
   elif [[ -z "${RUN_ROOT:-}" ]]; then
      [[ "${phase}" == phase1 ]] || { echo "run phase1 first or export RUN_ROOT" >&2; return 1; }
      RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-dynamic-t4k-s128-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
   fi
   [[ "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
      echo "RUN_ROOT must be under ${OFFLINE_OPD_BASE}: ${RUN_ROOT}" >&2; return 1;
   }
   mkdir -p "${RUN_ROOT}" "$(dirname "${DYNAMIC_STATE_FILE}")"
   printf '%s\n' "${RUN_ROOT}" > "${DYNAMIC_STATE_FILE}.tmp.$$"
   mv "${DYNAMIC_STATE_FILE}.tmp.$$" "${DYNAMIC_STATE_FILE}"
   export RUN_ROOT RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
   export DUMP_DIR="${RUN_ROOT}/dump" IN_DIR="${RUN_ROOT}/dump"
   export OUT_DIR="${RUN_ROOT}/scored" SAVE_PATH="${RUN_ROOT}/checkpoints"
}

require_dynamic_phase() {
   [[ -f "${RUN_ROOT}/.$1.complete" || "${ALLOW_INCOMPLETE_PREVIOUS_PHASE:-0}" == 1 ]] || {
      echo "missing ${RUN_ROOT}/.$1.complete" >&2; return 1;
   }
}

launch_dynamic_phase() {
   local phase="$1"
   local mode="${2:-}"
   local log_dir="${RUN_ROOT}/logs-${phase}"
   mkdir -p "${log_dir}"
   export RUN_DIR="${log_dir}"
   if [[ "${mode}" != --foreground ]]; then
      nohup bash "$0" --foreground > "${log_dir}/launcher.log" 2>&1 < /dev/null &
      echo "dynamic teacher4k/current128 ${phase} started: pid=$!, run=${RUN_ROOT}"
      return
   fi
   printf 'pid=%s started=%s\n' "$$" "$(date -Is)" > "${RUN_ROOT}/.${phase}.running"
   rm -f "${RUN_ROOT}/.${phase}.failed"
   set +e
   bash "${OFFLINE_OPD_DIR}/start-offline-opd.sh" "${phase}"
   local rc=$?
   set -e
   if (( rc == 0 )); then
      printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.${phase}.complete"
      rm -f "${RUN_ROOT}/.${phase}.running"
   else
      printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${RUN_ROOT}/.${phase}.failed"
      rm -f "${RUN_ROOT}/.${phase}.running"
   fi
   return "${rc}"
}
