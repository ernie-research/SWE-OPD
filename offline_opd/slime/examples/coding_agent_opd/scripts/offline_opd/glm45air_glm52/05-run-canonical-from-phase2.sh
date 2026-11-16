#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-dynamic-t1k-s128-20260819_001719}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-canonical-t1k-s128-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
[[ "${SOURCE_RUN}" == "${OFFLINE_OPD_BASE}"/* && "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
   echo "SOURCE_RUN and RUN_ROOT must be under ${OFFLINE_OPD_BASE}" >&2
   exit 1
}
[[ -d "${SOURCE_RUN}/dump" && -f "${SOURCE_RUN}/.phase1.complete" ]] || {
   echo "source Phase 1 is incomplete: ${SOURCE_RUN}" >&2
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
link_once "${SOURCE_RUN}/dump" "${RUN_ROOT}/dump"
link_once "${SOURCE_RUN}/.phase1.complete" "${RUN_ROOT}/.phase1.complete"
link_once "${SOURCE_RUN}/glm45air_to_glm52_token_ids.json" \
   "${RUN_ROOT}/glm45air_to_glm52_token_ids.json"

export RUN_ROOT
export OPD_TOKEN_ID_BRIDGE="${RUN_ROOT}/glm45air_to_glm52_token_ids.json"
export LOAD_PATH="${LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export OPD_TEACHER_NATIVE_REASONING_PREFIX=0
export OPD_TEACHER_CANONICAL_TEMPLATE=1
export OPD_TEACHER_MIN_ALIGNMENT_RATE="${OPD_TEACHER_MIN_ALIGNMENT_RATE:-0.90}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-canonical-t1k-current128}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE1=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'STUDENT_MODEL=%s\nSTUDENT_TORCH_DIST=%s\nTEACHER_MODEL=%s\n' \
         "${HF_CHECKPOINT}" "${LOAD_PATH}" "${TEACHER_MODEL}"
      printf 'LOSS=cached_teacher_dynamic_topk_reverse_kl\nSTUDENT_CURRENT_TOPK=128\nTEACHER_RAW_CACHE_TOPK=1000\n'
      printf 'TEACHER_TEMPLATE=official_glm52_canonical\nTOKEN_ALIGNMENT=exact_monotonic_v1\nMIN_ALIGNMENT_RATE=%s\n' \
         "${OPD_TEACHER_MIN_ALIGNMENT_RATE}"
      printf 'PHASE2_REPLICAS=2\nPHASE2_NNODES_PER_REPLICA=2\nPHASE2_TP=16\nPHASE2_SCORE_CONCURRENCY_PER_REPLICA=2\n'
      printf 'PHASE3_TP=8\nPHASE3_PP=2\nPHASE3_CP=2\nPHASE3_EP=1\nPHASE3_ETP=8\nPHASE3_GBS=64\nPHASE3_EPOCHS=2\nSAVE_INTERVAL=20\nLR=1e-6\n'
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "canonical GLM offline OPD pipeline started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

for entry in phase2:02-phase2-teacher.sh phase3:03-phase3-train.sh; do
   phase="${entry%%:*}"
   script="${entry#*:}"
   [[ -f "${RUN_ROOT}/.${phase}.complete" ]] && continue
   bash "${SCRIPT_DIR}/${script}" --foreground
done
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
