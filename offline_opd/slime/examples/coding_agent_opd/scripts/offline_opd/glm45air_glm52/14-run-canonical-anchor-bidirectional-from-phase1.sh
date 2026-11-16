#!/bin/bash
set -euo pipefail

# Isolated cross-tokenizer experiment:
#   * Phase 1 is reused unchanged from the GLM-4.5-Air rollout.
#   * Exact 1:1 positions are scored under the official GLM-5.2 template.
#   * Every remaining trainable position is anchored to the Phase-1 Student
#     Top-128 distribution with an explicit complementary tail bucket.
#   * Phase 3 uses 0.5 forward + 0.5 reverse Teacher KL and 1.0 anchor KL,
#     all reduced over the original Student trainable-token denominator.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-dynamic-t1k-s128-20260819_001719}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-canonical-exact-anchor-bidir-f0.5-r0.5-a1-t1k-s128-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
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
export LOAD_PATH="${CALLER_LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_bidirectional_topk_kl_with_student_anchor

# Canonical Teacher template is mandatory for the exact/anchor partition.
export OPD_TEACHER_NATIVE_REASONING_PREFIX=0
export OPD_TEACHER_CANONICAL_TEMPLATE=1
export OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS=0
export OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE=0
export OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT=0
export TEACHER_ATTENTION_MODE="${TEACHER_ATTENTION_MODE:-dense}"
export TEACHER_PORT="${TEACHER_PORT:-15141}"
export TEACHER_NCCL_PORT_BASE="${TEACHER_NCCL_PORT_BASE:-24600}"
export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-1}"
export SCORE_FILE_CONCURRENCY="${SCORE_FILE_CONCURRENCY:-1}"
export PHASE2_BALANCE_MANIFESTS=0

export OPD_FORWARD_KL_COEF="${OPD_FORWARD_KL_COEF:-0.5}"
export OPD_REVERSE_KL_COEF="${OPD_REVERSE_KL_COEF:-0.5}"
export OPD_STUDENT_ANCHOR_KL_COEF="${OPD_STUDENT_ANCHOR_KL_COEF:-1.0}"
export TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1 ROLLOUT_BATCH_SIZE=64
export NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=8 PP_SIZE=2 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export SAVE_INTERVAL=20 PREFETCH_ROLLOUT_DATA=1 OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_BALANCE_DATA=0
export OFFLINE_OPD_POSTHOC_MASK_TOOL_FORMAT=0
export OFFLINE_OPD_POSTHOC_MASK_LET_ME_PROB=0
export USE_WANDB="${USE_WANDB:-1}" WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-canonical-exact-anchor-bidir}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE1=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'LOSS=cached_teacher_bidirectional_topk_kl_with_student_anchor\n'
      printf 'TEACHER_TEMPLATE=official_glm52_canonical\nPOSITION_POLICY=exact_1to1_teacher_else_rollout_student_anchor\n'
      printf 'TEACHER_FORWARD_COEF=%s\nTEACHER_REVERSE_COEF=%s\nSTUDENT_ANCHOR_COEF=%s\n' \
         "${OPD_FORWARD_KL_COEF}" "${OPD_REVERSE_KL_COEF}" "${OPD_STUDENT_ANCHOR_KL_COEF}"
      printf 'TEACHER_RAW_CACHE_TOPK=1000\nSUPPORT_TOPK=128\nANCHOR_TAIL_BUCKET=1\n'
      printf 'PHASE2_ATTENTION_MODE=%s\nPHASE2_SCORE_CONCURRENCY=%s\n' \
         "${TEACHER_ATTENTION_MODE}" "${SCORE_CONCURRENCY}"
      printf 'PHASE3_GBS=64\nPHASE3_EPOCHS=2\nSAVE_INTERVAL=20\nLR=1e-6\nLR_SCHEDULE=constant\n'
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "canonical cross-tokenizer anchor pipeline started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

if [[ ! -f "${RUN_ROOT}/.phase2.complete" ]]; then
   bash "${SCRIPT_DIR}/02-phase2-teacher.sh" --foreground
fi
if [[ ! -f "${RUN_ROOT}/.phase3.complete" ]]; then
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
fi
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
