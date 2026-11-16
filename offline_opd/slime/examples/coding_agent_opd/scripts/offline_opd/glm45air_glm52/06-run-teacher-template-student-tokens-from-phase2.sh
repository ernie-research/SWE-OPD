#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# common.sh supplies a historical GLM checkpoint default. Preserve an explicit
# caller override, but otherwise select the checkpoint converted for the
# non-interleaved (NeoX split-half) RoPE used by this isolated experiment.
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-dynamic-t1k-s128-20260819_001719}"
SOURCE_SAMPLE_COUNT="${SOURCE_SAMPLE_COUNT:-0}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-teacher-template-student-tokens-t1k-s128-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
[[ "${SOURCE_RUN}" == "${OFFLINE_OPD_BASE}"/* && "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
   echo "SOURCE_RUN and RUN_ROOT must be under ${OFFLINE_OPD_BASE}" >&2
   exit 1
}
[[ -d "${SOURCE_RUN}/dump" && -f "${SOURCE_RUN}/.phase1.complete" ]] || {
   echo "source Phase 1 is incomplete: ${SOURCE_RUN}" >&2
   exit 1
}
[[ "${SOURCE_SAMPLE_COUNT}" =~ ^[0-9]+$ ]] || {
   echo "SOURCE_SAMPLE_COUNT must be a non-negative integer: ${SOURCE_SAMPLE_COUNT}" >&2
   exit 1
}

mkdir -p "${RUN_ROOT}"
link_once() {
   local source="$1" target="$2"
   if [[ -L "${target}" ]]; then
      [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
         echo "existing symlink points elsewhere: ${target}" >&2
         return 1
      }
      return
   fi
   [[ ! -e "${target}" ]] || { echo "refusing to replace ${target}" >&2; return 1; }
   ln -s "${source}" "${target}"
}
if (( SOURCE_SAMPLE_COUNT > 0 )); then
   # Keep the original Phase-1 dataset immutable.  A per-file symlink view also
   # makes Phase 2 discover exactly rollout_0..rollout_{N-1}, without copying
   # the large student support tensors.
   [[ ! -L "${RUN_ROOT}/dump" ]] || {
      echo "subset dump must be a directory, not a symlink: ${RUN_ROOT}/dump" >&2
      exit 1
   }
   mkdir -p "${RUN_ROOT}/dump"
   for (( i = 0; i < SOURCE_SAMPLE_COUNT; i++ )); do
      source_file="${SOURCE_RUN}/dump/rollout_${i}.pt"
      [[ -f "${source_file}" ]] || {
         echo "missing requested Phase-1 sample: ${source_file}" >&2
         exit 1
      }
      link_once "${source_file}" "${RUN_ROOT}/dump/rollout_${i}.pt"
   done
   unexpected="$(find "${RUN_ROOT}/dump" -maxdepth 1 -type f -o -type l | sed -n "/\/rollout_[0-9][0-9]*\.pt$/p" | wc -l)"
   (( unexpected == SOURCE_SAMPLE_COUNT )) || {
      echo "subset dump contains ${unexpected} rollout files, expected ${SOURCE_SAMPLE_COUNT}" >&2
      exit 1
   }
else
   link_once "${SOURCE_RUN}/dump" "${RUN_ROOT}/dump"
fi
link_once "${SOURCE_RUN}/.phase1.complete" "${RUN_ROOT}/.phase1.complete"
link_once "${SOURCE_RUN}/glm45air_to_glm52_token_ids.json" \
   "${RUN_ROOT}/glm45air_to_glm52_token_ids.json"

export RUN_ROOT
export OPD_TOKEN_ID_BRIDGE="${RUN_ROOT}/glm45air_to_glm52_token_ids.json"
export LOAD_PATH="${CALLER_LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export EXPERT_MP_SIZE="${EXPERT_MP_SIZE:-8}"
export EXPERT_TP_SIZE="${EXPERT_TP_SIZE:-1}"
export OPD_TEACHER_NATIVE_REASONING_PREFIX=1
export OPD_TEACHER_CANONICAL_TEMPLATE=0
export OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS=1
if (( SOURCE_SAMPLE_COUNT > 0 )); then
   export OFFLINE_OPD_ALLOW_SAMPLE_SUBSET=1
fi
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-teacher-template-student-tokens-t1k-current128}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE1=%s\nSOURCE_SAMPLE_COUNT=%s\n' \
         "${RUN_ROOT}" "${SOURCE_RUN}" "${SOURCE_SAMPLE_COUNT}"
      printf 'STUDENT_MODEL=%s\nSTUDENT_TORCH_DIST=%s\nTEACHER_MODEL=%s\n' \
         "${HF_CHECKPOINT}" "${LOAD_PATH}" "${TEACHER_MODEL}"
      printf 'LOSS=cached_teacher_dynamic_topk_reverse_kl\nSTUDENT_CURRENT_TOPK=128\nTEACHER_RAW_CACHE_TOPK=1000\n'
      printf 'TEACHER_TEMPLATE=official_glm52_prompt_prefix_connectors\nASSISTANT_CONTENT=verbatim_student_token_ids\n'
      printf 'MASKED_STUDENT_FORMAT_TOKENS=role,think,tool_call,tool_response,arg_tags\nARTIFACT_SCHEMA=teacher_template_student_tokens_v1\n'
      printf 'PHASE2_REPLICAS=2\nPHASE2_NNODES_PER_REPLICA=2\nPHASE2_TP=16\n'
      printf 'PHASE2_ATTENTION_MODE=%s\nPHASE2_SCORE_CONCURRENCY_PER_REPLICA=%s\n' \
         "${TEACHER_ATTENTION_MODE:-dsa}" "${SCORE_CONCURRENCY:-$([[ "${TEACHER_ATTENTION_MODE:-dsa}" == dense ]] && echo 1 || echo 2)}"
      if [[ "${TEACHER_ATTENTION_MODE:-dsa}" == dense ]]; then
         printf 'PHASE2_DENSE_ATTN_KV_LEN_THRESHOLD=%s\nPHASE2_MAX_KV_CHUNK_CAPACITY=%s\n' \
            "${TEACHER_DENSE_ATTN_KV_LEN_THRESHOLD:-131072}" "${TEACHER_MAX_KV_CHUNK_CAPACITY:-262144}"
         printf 'PHASE2_NSA_SPARSE_PREFILL_Q_CHUNK=%s\nPHASE2_NSA_MQA_LOGITS_MAX_ROWS=%s\nPHASE2_MOE_TOKEN_CHUNK_SIZE=%s\n' \
            "${TEACHER_NSA_SPARSE_PREFILL_Q_CHUNK:-8192}" "${TEACHER_NSA_MQA_LOGITS_MAX_ROWS:-4096}" "${TEACHER_MOE_TOKEN_CHUNK_SIZE:-16384}"
         printf 'PHASE2_CHUNKED_PREFILL_SIZE=%s\nPHASE2_MAX_PREFILL_TOKENS=%s\nPHASE2_MAX_RUNNING_REQUESTS=%s\n' \
            "${TEACHER_CHUNKED_PREFILL_SIZE:--1}" "${TEACHER_MAX_PREFILL_TOKENS:-262144}" "${TEACHER_MAX_RUNNING_REQUESTS:-1}"
      fi
      printf 'PHASE3_TP=8\nPHASE3_PP=2\nPHASE3_CP=2\nPHASE3_EP=8\nPHASE3_ETP=1\nPHASE3_GBS=64\nPHASE3_EPOCHS=2\nSAVE_INTERVAL=%s\nLR=1e-6\n' \
         "${SAVE_INTERVAL:-20}"
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "Teacher-template/Student-token GLM offline OPD pipeline started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

for entry in phase2:02-phase2-teacher.sh phase3:03-phase3-train.sh; do
   phase="${entry%%:*}"
   script="${entry#*:}"
   [[ -f "${RUN_ROOT}/.${phase}.complete" ]] && continue
   bash "${SCRIPT_DIR}/${script}" --foreground
done
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
