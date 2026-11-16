#!/bin/bash
set -euo pipefail

GLM47_OPD_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_DIR="$(cd -- "${GLM47_OPD_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"

# Fixed experiment contract: shared token ids, exact Student-token replay, and
# pure dynamic Top-K reverse KL.  Do not inherit a cross-tokenizer/template mode
# from an older GLM-4.5-Air shell.
export OPD_ID_TOPK=128
export OPD_DYNAMIC_SUPPORT_TOPK=128
export OPD_TEACHER_CACHE_TOPK=1000
export OPD_STUDENT_VALID_VOCAB_SIZE=154856
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_dynamic_topk_reverse_kl
unset OPD_TOKEN_ID_BRIDGE
unset OPD_TEACHER_TEMPLATE_MODEL
unset OPD_TEACHER_NATIVE_REASONING_PREFIX
unset OPD_TEACHER_CANONICAL_TEMPLATE
unset OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS
unset OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE
unset OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT
unset OFFLINE_OPD_POSTHOC_ANCHOR_PROTOCOL_TOKENS
unset OPD_FORWARD_KL_COEF OPD_REVERSE_KL_COEF OPD_STUDENT_ANCHOR_KL_COEF

export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/workspace/env_run/<user>/Megatron-LM}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/model/<org>/GLM-4.7-Flash}"
export LOAD_PATH="${LOAD_PATH:-/path/to/model/<org>/GLM-4.7-Flash_torch_dist}"
export TEACHER_MODEL="${TEACHER_MODEL:-/path/to/model/<org>/GLM-5.2-FP8}"
export MODEL_ARGS_SH="${MODEL_ARGS_SH:-${SLIME_DIR}/scripts/models/glm4.7-30B-A3B.sh}"
export PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/data/rl_query_final_20260709_issuetype_train.jsonl}"
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-glm47}"
export REASONING_PARSER="${REASONING_PARSER:-glm45}"

OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"
GLM47_STATE_FILE="${GLM47_STATE_FILE:-${OFFLINE_OPD_BASE}/.glm47flash-glm52-reverse-t1k-s128-current}"

select_glm47_run() {
   local phase="$1"
   if [[ -z "${RUN_ROOT:-}" && -s "${GLM47_STATE_FILE}" && "${NEW_RUN:-0}" != 1 ]]; then
      IFS= read -r RUN_ROOT < "${GLM47_STATE_FILE}"
   elif [[ -z "${RUN_ROOT:-}" ]]; then
      [[ "${phase}" == phase1 ]] || {
         echo "run phase1 first or export RUN_ROOT" >&2
         return 1
      }
      RUN_ROOT="${OFFLINE_OPD_BASE}/glm47flash-glm52-student-template-reverse-t1k-s128-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
   fi
   [[ "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
      echo "RUN_ROOT must be under ${OFFLINE_OPD_BASE}: ${RUN_ROOT}" >&2
      return 1
   }
   mkdir -p "${RUN_ROOT}" "$(dirname "${GLM47_STATE_FILE}")"
   printf '%s\n' "${RUN_ROOT}" > "${GLM47_STATE_FILE}.tmp.$$"
   mv "${GLM47_STATE_FILE}.tmp.$$" "${GLM47_STATE_FILE}"
   export RUN_ROOT RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
   export DUMP_DIR="${RUN_ROOT}/dump" IN_DIR="${RUN_ROOT}/dump"
   export OUT_DIR="${RUN_ROOT}/scored" SAVE_PATH="${RUN_ROOT}/checkpoints"
}

ensure_shared_token_ids() {
   "${SLIME_DIR}/.slime_venv/bin/python3" - "${HF_CHECKPOINT}" "${TEACHER_MODEL}" <<'PY'
import json
import sys
from pathlib import Path

student, teacher = map(Path, sys.argv[1:])
for model in (student, teacher):
    if not model.is_dir():
        raise SystemExit(f"model directory is missing: {model}")

student_config = json.loads((student / "config.json").read_text())
teacher_config = json.loads((teacher / "config.json").read_text())
if student_config.get("vocab_size") != 154880 or teacher_config.get("vocab_size") != 154880:
    raise SystemExit(
        "expected Student/Teacher model vocab_size=154880, got "
        f"{student_config.get('vocab_size')}/{teacher_config.get('vocab_size')}"
    )

student_tokenizer = student / "tokenizer.json"
teacher_tokenizer = teacher / "tokenizer.json"
student_stat = student_tokenizer.stat()
teacher_stat = teacher_tokenizer.stat()
if student_stat.st_size != 20217442 or teacher_stat.st_size != 20217442:
    raise SystemExit(
        "unexpected Student/Teacher tokenizer.json sizes: "
        f"{student_stat.st_size}/{teacher_stat.st_size}"
    )

# The complete files were audited once when this recipe was created
# (SHA256 below). Re-reading both 20 MiB JSON files before every phase can block
# for minutes on the shared PFS, so subsequent launch guards compare bounded
# head/middle/tail probes plus the exact size and model vocab configuration.
def probes(path: Path) -> list[bytes]:
    chunk = 65536
    size = path.stat().st_size
    with path.open("rb") as stream:
        output = [stream.read(chunk)]
        stream.seek(size // 2)
        output.append(stream.read(chunk))
        stream.seek(max(size - chunk, 0))
        output.append(stream.read(chunk))
    return output

if probes(student_tokenizer) != probes(teacher_tokenizer):
    raise SystemExit("Student and Teacher tokenizer.json probes differ")
print(
    "shared token ids verified: "
    "size=154856, model_vocab=154880, "
    "audited_sha256=19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d"
)
PY
}

require_glm47_phase() {
   [[ -f "${RUN_ROOT}/.$1.complete" || "${ALLOW_INCOMPLETE_PREVIOUS_PHASE:-0}" == 1 ]] || {
      echo "missing ${RUN_ROOT}/.$1.complete" >&2
      return 1
   }
}

launch_glm47_phase() {
   local phase="$1"
   local mode="${2:-}"
   local log_dir="${RUN_ROOT}/logs-${phase}"
   mkdir -p "${log_dir}"
   export RUN_DIR="${log_dir}"
   if [[ "${mode}" != --foreground ]]; then
      nohup setsid bash "$0" --foreground > "${log_dir}/launcher.log" 2>&1 < /dev/null &
      echo "GLM-4.7-Flash/GLM-5.2 offline OPD ${phase} started: pid=$!, run=${RUN_ROOT}"
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
