#!/bin/bash
set -euo pipefail

# Phase-3-only experiment: reuse immutable dense-Teacher Top-1000 scores,
# derive the first 128 Student-representable Teacher tokens at training time,
# and optimize truncated Teacher-owned forward-KL. Teacher-only vocabulary ids
# are skipped and replaced by the next cached Teacher token.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-dense-t1k-s128-rescore-mask-tool-newlines-pretool-20260823_013530}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-dense-teacherowned-forwardkl-t1k-s128-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
[[ "${SOURCE_RUN}" == "${OFFLINE_OPD_BASE}"/* && "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || {
   echo "SOURCE_RUN and RUN_ROOT must be under ${OFFLINE_OPD_BASE}" >&2
   exit 1
}
[[ -d "${SOURCE_RUN}/scored" && -f "${SOURCE_RUN}/.phase2.complete" ]] || {
   echo "source Phase 2 is incomplete: ${SOURCE_RUN}" >&2
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
link_once "${SOURCE_RUN}/scored" "${RUN_ROOT}/scored"
link_once "${SOURCE_RUN}/.phase2.complete" "${RUN_ROOT}/.phase2.complete"
link_once "${SOURCE_RUN}/glm45air_to_glm52_token_ids.json" \
   "${RUN_ROOT}/glm45air_to_glm52_token_ids.json"

export RUN_ROOT
export OPD_TOKEN_ID_BRIDGE="${RUN_ROOT}/glm45air_to_glm52_token_ids.json"
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_topk_forward_kl
export LOAD_PATH="${CALLER_LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export SCORED_DIR="${RUN_ROOT}/scored"
export FANOUT_DIR="${RUN_ROOT}/train_data_epochs_x2_b64"
export TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1 ROLLOUT_BATCH_SIZE=64
export NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=8 PP_SIZE=2 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export SAVE_INTERVAL=20 PREFETCH_ROLLOUT_DATA=1 OFFLINE_OPD_TOPK_OVERLAP=1
# The source Phase-2 loss_mask already contains the structural-newline and
# post-think/pre-tool masks. Do not apply any additional Phase-3 masking.
export OFFLINE_OPD_POSTHOC_MASK_TOOL_FORMAT=0
export OFFLINE_OPD_POSTHOC_MASK_LET_ME_PROB=0
export USE_WANDB="${USE_WANDB:-1}" WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-teacherowned-forwardkl-t1k-s128}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE2=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'STUDENT_MODEL=%s\nSTUDENT_TORCH_DIST=%s\n' "${HF_CHECKPOINT}" "${LOAD_PATH}"
      printf 'LOSS=cached_teacher_topk_forward_kl\nTEACHER_OWNED_SUPPORT_TOPK=128\nTEACHER_RAW_CACHE_TOPK=1000\n'
      printf 'TEACHER_ONLY_TOKEN_POLICY=skip_and_fill_from_next_rank\nSOURCE_SCORED_IMMUTABLE=1\n'
      printf 'MASK_LET_ME=0\nADDITIONAL_PHASE3_MASKING=0\n'
      printf 'PHASE3_TP=8\nPHASE3_PP=2\nPHASE3_CP=2\nPHASE3_EP=8\nPHASE3_ETP=1\n'
      printf 'PHASE3_GBS=64\nPHASE3_EPOCHS=2\nPHASE3_STEPS=128\nSAVE_INTERVAL=20\nLR=1e-6\nLR_SCHEDULE=constant\n'
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "cached-Teacher forward-KL GLM Phase 3 started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
