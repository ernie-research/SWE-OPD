#!/bin/bash
set -euo pipefail

GLM_OPD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_DIR="$(cd -- "${GLM_OPD_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"

export OPD_ID_TOPK="${OPD_ID_TOPK:-128}"
export OPD_DYNAMIC_SUPPORT_TOPK="${OPD_DYNAMIC_SUPPORT_TOPK:-128}"
export OPD_TEACHER_CACHE_TOPK="${OPD_TEACHER_CACHE_TOPK:-1000}"
export OPD_STUDENT_VALID_VOCAB_SIZE=151365
export OFFLINE_OPD_LOSS_TYPE="${OFFLINE_OPD_LOSS_TYPE:-cached_teacher_dynamic_topk_reverse_kl}"
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/workspace/env_run/<user>/Megatron-LM}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/model/<org>/GLM-4.5-Air}"
export LOAD_PATH="${LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist}"
export TEACHER_MODEL="${TEACHER_MODEL:-/path/to/model/<org>/GLM-5.2-FP8}"
export MODEL_ARGS_SH="${MODEL_ARGS_SH:-${GLM_OPD_DIR}/model_args.sh}"
export PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/data/rl_query_final_20260709_issuetype_train.jsonl}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-glm45}"
export REASONING_PARSER="${REASONING_PARSER:-glm45}"

OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"
GLM_STATE_FILE="${GLM_STATE_FILE:-${OFFLINE_OPD_BASE}/.glm45air-glm52-dynamic-t1k-s128-current}"

select_glm_run() {
   local phase="$1"
   if [[ -z "${RUN_ROOT:-}" && -s "${GLM_STATE_FILE}" && "${NEW_RUN:-0}" != 1 ]]; then
      IFS= read -r RUN_ROOT < "${GLM_STATE_FILE}"
   elif [[ -z "${RUN_ROOT:-}" ]]; then
      [[ "${phase}" == phase1 ]] || {
         echo "run phase1 first or export RUN_ROOT" >&2
         return 1
      }
      RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-dynamic-t1k-s128-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
   fi
   [[ "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
      echo "RUN_ROOT must be under ${OFFLINE_OPD_BASE}: ${RUN_ROOT}" >&2
      return 1
   }
   mkdir -p "${RUN_ROOT}" "$(dirname "${GLM_STATE_FILE}")"
   printf '%s\n' "${RUN_ROOT}" > "${GLM_STATE_FILE}.tmp.$$"
   mv "${GLM_STATE_FILE}.tmp.$$" "${GLM_STATE_FILE}"
   export RUN_ROOT RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
   export DUMP_DIR="${RUN_ROOT}/dump" IN_DIR="${RUN_ROOT}/dump"
   export OUT_DIR="${RUN_ROOT}/scored" SAVE_PATH="${RUN_ROOT}/checkpoints"
   export OPD_TOKEN_ID_BRIDGE="${OPD_TOKEN_ID_BRIDGE:-${RUN_ROOT}/glm45air_to_glm52_token_ids.json}"
}

ensure_glm_token_bridge() {
   if [[ ! -s "${OPD_TOKEN_ID_BRIDGE}" ]]; then
      "${SLIME_DIR}/.slime_venv/bin/python3" \
         "${GLM_OPD_DIR}/build_token_id_bridge.py" \
         --student-model "${HF_CHECKPOINT}" \
         --teacher-model "${TEACHER_MODEL}" \
         --output "${OPD_TOKEN_ID_BRIDGE}"
   fi
   PYTHONPATH="${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${SLIME_DIR}/.slime_venv/bin/python3" - "${OPD_TOKEN_ID_BRIDGE}" <<'PY'
import sys
from slime.utils.token_id_bridge import load_token_id_bridge
b = load_token_id_bridge(sys.argv[1])
assert b.student_tokenizer_size == 151365
assert b.student_model_vocab_size == 151552
assert b.teacher_tokenizer_size == 154856
assert b.teacher_model_vocab_size == 154880
print(f"token bridge verified: {b.mapping_sha256}")
PY
}

require_glm_phase() {
   [[ -f "${RUN_ROOT}/.$1.complete" || "${ALLOW_INCOMPLETE_PREVIOUS_PHASE:-0}" == 1 ]] || {
      echo "missing ${RUN_ROOT}/.$1.complete" >&2
      return 1
   }
}

launch_glm_phase() {
   local phase="$1"
   local mode="${2:-}"
   local log_dir="${RUN_ROOT}/logs-${phase}"
   mkdir -p "${log_dir}"
   export RUN_DIR="${log_dir}"
   if [[ "${mode}" != --foreground ]]; then
      # Detach the complete phase launcher from the caller's process group.
      # nohup alone only ignores SIGHUP; cluster launch wrappers may still
      # reap the inherited process group after the submitting shell exits,
      # leaving the setsid SGLang children alive without their scorer.
      nohup setsid bash "$0" --foreground > "${log_dir}/launcher.log" 2>&1 < /dev/null &
      echo "GLM offline OPD ${phase} started: pid=$!, run=${RUN_ROOT}"
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
