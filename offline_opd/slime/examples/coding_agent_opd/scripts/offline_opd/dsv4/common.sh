#!/bin/bash
# Shared settings for GLM-4.5-Air (student) <- DeepSeek-V4-Flash-0731 (teacher) offline OPD.
#
# Derived from glm45air_glm52/common.sh: only the teacher side differs. Student-side knobs
# stay identical, including TOOL_CALL_PARSER / REASONING_PARSER, which parse the
# Student's own output (teacher_trajectory._canonical_assistant_message) and must stay glm45.
set -euo pipefail

DSV4_OPD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# The slime tree is mounted at different paths in the Agent sandbox and the training
# container, so probe and fail loudly rather than resolve to a wrong tree.
if [[ -z "${OFFLINE_OPD_DIR:-}" ]]; then
   for candidate in \
      "${DSV4_OPD_DIR}/.." \
      "/path/to/<user>/slime" \
      "${DSV4_OPD_DIR}/../../<user>_space/<user>/slime"; do
      if [[ -f "${candidate}/examples/coding_agent_opd/scripts/offline_opd/start-offline-opd.sh" ]]; then
         OFFLINE_OPD_DIR="${candidate}/examples/coding_agent_opd/scripts/offline_opd"
         break
      fi
      # A tree whose offline_opd dir is the immediate parent of this recipe.
      if [[ -f "${candidate}/start-offline-opd.sh" ]]; then
         OFFLINE_OPD_DIR="${candidate}"
         break
      fi
   done
fi
[[ -n "${OFFLINE_OPD_DIR:-}" ]] || {
   echo "cannot locate slime offline_opd scripts; export OFFLINE_OPD_DIR" >&2
   return 1 2>/dev/null || exit 1
}
OFFLINE_OPD_DIR="$(cd -- "${OFFLINE_OPD_DIR}" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"

export OPD_ID_TOPK="${OPD_ID_TOPK:-128}"
export OPD_DYNAMIC_SUPPORT_TOPK="${OPD_DYNAMIC_SUPPORT_TOPK:-128}"
export OPD_TEACHER_CACHE_TOPK="${OPD_TEACHER_CACHE_TOPK:-1000}"
# GLM-4.5-Air tokenizer size, not the teacher's 129280.
export OPD_STUDENT_VALID_VOCAB_SIZE=151365
# The anchor schema is required: it is the only Phase-2 artifact loss that sets
# partition_student_anchors, which is what lets _rebuild_canonical_teacher_trajectory
# skip the OPD_TEACHER_MIN_ALIGNMENT_RATE abort. DSV4 aligns ~85% of trainable
# positions 1:1, below the 0.90 default, so any other schema fails every sample.
export OFFLINE_OPD_LOSS_TYPE="${OFFLINE_OPD_LOSS_TYPE:-cached_teacher_bidirectional_topk_kl_with_student_anchor}"
# Needed by phase1/phase3 (start-offline-opd.sh checks the dir). The upstream default
# is missing on the trainer nodes, so probe; it stays first so an env that has it wins.
if [[ -z "${MEGATRON_LM_PATH:-}" ]]; then
   for candidate in \
      "/path/to/workspace/env_run/<user>/Megatron-LM" \
      "/path/to/<user>/Megatron-LM"; do
      [[ -f "${candidate}/megatron/core/__init__.py" ]] && { MEGATRON_LM_PATH="${candidate}"; break; }
   done
fi
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:?cannot locate Megatron-LM}"
# Sandbox and container paths, as above. _ropefix is the checkpoint the glm45air_glm52
# canonical runs trained from; the un-suffixed torch_dist predates the RoPE fix.
for root in "/path/to/model/<org>" \
   "/path/to/<user>/<user>_space/model/<org>"; do
   [[ -d "${root}/GLM-4.5-Air" ]] || continue
   : "${HF_CHECKPOINT:=${root}/GLM-4.5-Air}"
   : "${LOAD_PATH:=${root}/GLM-4.5-Air_torch_dist_ropefix}"
   break
done
export HF_CHECKPOINT="${HF_CHECKPOINT:?cannot locate GLM-4.5-Air HF dir}"
export LOAD_PATH="${LOAD_PATH:?cannot locate GLM-4.5-Air torch_dist checkpoint}"
# teacher_serving_0731: the 0731 weights (symlinks into <user>'s model_hub) plus the
# chat_template.jinja from make_dsv4_0731_chat_template.py (0731 ships only
# encoding_dsv4.py). Probed as: job read-only view, job rw view, Agent-sandbox mount.
if [[ -z "${TEACHER_MODEL:-}" ]]; then
   for candidate in \
      "/path/to/<user>/<user>/opd_glm45air_dsv4flash/teacher_serving_0731" \
      "/path/to/<space>-<user>/<user>/opd_glm45air_dsv4flash/teacher_serving_0731" \
      "/path/to/<space>_<user>/<user>/opd_glm45air_dsv4flash/teacher_serving_0731"; do
      [[ -f "${candidate}/chat_template.jinja" ]] && { TEACHER_MODEL="${candidate}"; break; }
   done
fi
export TEACHER_MODEL="${TEACHER_MODEL:?cannot locate DSV4-Flash-0731 serving dir (teacher_serving_0731)}"
# Tokenizer/template-only consumers read this instead of TEACHER_MODEL: the serving
# config.json says model_type=deepseek_v4, which transformers in .slime_venv cannot
# load. teacher_shim_0731 = the 0731 tokenizer.json and chat_template.jinja plus a
# llama-typed config.json with vocab_size=129280. SGLang still loads TEACHER_MODEL.
if [[ -z "${OPD_TEACHER_TEMPLATE_MODEL:-}" ]]; then
   for candidate in "${TEACHER_MODEL%/*}/teacher_shim_0731" \
      "/path/to/<space>_<user>/<user>/opd_glm45air_dsv4flash/teacher_shim_0731"; do
      [[ -f "${candidate}/chat_template.jinja" ]] && { OPD_TEACHER_TEMPLATE_MODEL="${candidate}"; break; }
   done
fi
export OPD_TEACHER_TEMPLATE_MODEL="${OPD_TEACHER_TEMPLATE_MODEL:?cannot locate teacher_shim_0731}"
# 0731 protocol pairs: <tool_call> = first |DSML| of the turn, <|observation|> =
# <|end_of_sentence|>; </tool_call> has no 1:1 token. That maps 2 of the 58298
# surface-unmapped student ids, hence 58296.
export DSV4_BRIDGE_PROTOCOL_PAIRS="${DSV4_BRIDGE_PROTOCOL_PAIRS:-v0731}"
export EXPECT_UNMAPPED_STUDENT_IDS="${EXPECT_UNMAPPED_STUDENT_IDS:-58296}"
[[ -f "${OPD_TEACHER_TEMPLATE_MODEL}/chat_template.jinja" ]] || {
   echo "teacher template dir lacks chat_template.jinja: ${OPD_TEACHER_TEMPLATE_MODEL}" >&2
   return 1 2>/dev/null || exit 1
}
# Byte-identical copy of glm45air_glm52/model_args.sh (Student side only).
export MODEL_ARGS_SH="${MODEL_ARGS_SH:-${DSV4_OPD_DIR}/model_args.sh}"
export PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/data/rl_query_final_20260709_issuetype_train.jsonl}"
# Student-side parsers. Do NOT switch to a deepseek detector: these consume the
# GLM-4.5-Air rollout text, not the teacher's.
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-glm45}"
export REASONING_PARSER="${REASONING_PARSER:-glm45}"

# Runs (teacher cache plus ~1.4 TiB of optimizer checkpoints) go to our own PFS
# namespace; Phase 1 is read from the GLM-5.2 run by symlink.
OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/<user>/offline_opd}"
DSV4_STATE_FILE="${DSV4_STATE_FILE:-${OFFLINE_OPD_BASE}/.glm45air-dsv4flash-dynamic-t1k-s128-current}"

# Same function names as glm45air_glm52/common.sh, so dsv4/02-phase2-teacher.sh and
# 03-phase3-train.sh stay near-copies of the GLM-5.2 ones.
select_glm_run() {
   local phase="$1"
   if [[ -z "${RUN_ROOT:-}" && -s "${DSV4_STATE_FILE}" && "${NEW_RUN:-0}" != 1 ]]; then
      IFS= read -r RUN_ROOT < "${DSV4_STATE_FILE}"
   elif [[ -z "${RUN_ROOT:-}" ]]; then
      [[ "${phase}" == phase1 ]] || {
         echo "run phase1 first or export RUN_ROOT" >&2
         return 1
      }
      RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-dsv4flash-dynamic-t1k-s128-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
   fi
   [[ "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
      echo "RUN_ROOT must be under ${OFFLINE_OPD_BASE}: ${RUN_ROOT}" >&2
      return 1
   }
   mkdir -p "${RUN_ROOT}" "$(dirname "${DSV4_STATE_FILE}")"
   printf '%s\n' "${RUN_ROOT}" > "${DSV4_STATE_FILE}.tmp.$$"
   mv "${DSV4_STATE_FILE}.tmp.$$" "${DSV4_STATE_FILE}"
   export RUN_ROOT RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
   export DUMP_DIR="${RUN_ROOT}/dump" IN_DIR="${RUN_ROOT}/dump"
   export OUT_DIR="${RUN_ROOT}/scored" SAVE_PATH="${RUN_ROOT}/checkpoints"
   export OPD_TOKEN_ID_BRIDGE="${OPD_TOKEN_ID_BRIDGE:-${RUN_ROOT}/glm45air_to_dsv4flash_token_ids.json}"
}

ensure_glm_token_bridge() {
   if [[ ! -s "${OPD_TOKEN_ID_BRIDGE}" ]]; then
      # Not glm45air_glm52/build_token_id_bridge.py: it fails on the 58298 Student
      # tokens with no DSV4 counterpart and on a GLM-family-only identity prefix.
      "${SLIME_DIR}/.slime_venv/bin/python3" \
         "${DSV4_OPD_DIR}/build_token_id_bridge.py" \
         --student-model "${HF_CHECKPOINT}" \
         --teacher-model "${OPD_TEACHER_TEMPLATE_MODEL}" \
         ${DSV4_BRIDGE_PROTOCOL_PAIRS:+--protocol-pairs "${DSV4_BRIDGE_PROTOCOL_PAIRS}"} \
         --output "${OPD_TOKEN_ID_BRIDGE}"
   fi
   PYTHONPATH="${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
      "${SLIME_DIR}/.slime_venv/bin/python3" - "${OPD_TOKEN_ID_BRIDGE}" <<'PY'
import sys
from slime.utils.token_id_bridge import load_token_id_bridge
b = load_token_id_bridge(sys.argv[1])
assert b.student_tokenizer_size == 151365, b.student_tokenizer_size
assert b.student_model_vocab_size == 151552, b.student_model_vocab_size
# DSV4-Flash: tokenizer and model vocabulary are the same size, no padding rows.
assert b.teacher_tokenizer_size == 129280, b.teacher_tokenizer_size
assert b.teacher_model_vocab_size == 129280, b.teacher_model_vocab_size
# Pin the unmapped count so a swapped tokenizer cannot poison a teacher cache.
assert b.unmapped_student_ids == int(__import__("os").environ.get("EXPECT_UNMAPPED_STUDENT_IDS", "58298")), b.unmapped_student_ids
print(
    f"token bridge verified: {b.mapping_sha256} "
    f"({b.unmapped_student_ids}/{b.student_tokenizer_size} student ids unmapped)"
)
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
      nohup setsid bash "$0" --foreground > "${log_dir}/launcher.log" 2>&1 < /dev/null &
      echo "DSV4 offline OPD ${phase} started: pid=$!, run=${RUN_ROOT}"
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
