#!/bin/bash
set -euo pipefail

# Phase-3-only conditional-KL experiment over the immutable High-effort Phase-2
# cache. Exact Student ``Let/let + me + X`` triples use Teacher-owned Top-128
# forward-KL; all other exact rows use current-Student Top-128 reverse-KL;
# cross-tokenizer/template mismatch rows retain rollout-Student anchor KL.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-canonical-anchor-bidir-effort-high-f0.5-r0.5-a1-n1280-s20-20260827_123616}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-letme-forward-else-reverse-anchor-high-t1k-s128-n1280-s20-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
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
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor
export OPD_LET_TOKEN_IDS="${OPD_LET_TOKEN_IDS:-1077,1149,6771,10056,10212}"
export OPD_ME_TOKEN_IDS="${OPD_ME_TOKEN_IDS:-752,2660,48977}"
export OPD_STUDENT_ANCHOR_KL_COEF="${OPD_STUDENT_ANCHOR_KL_COEF:-1.0}"
export LOAD_PATH="${CALLER_LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export SCORED_DIR="${RUN_ROOT}/scored"
export FANOUT_DIR="${RUN_ROOT}/train_data_epochs_x1_b64_s20"
export TRAIN_EPOCHS=1 TRAIN_STEPS=20 EPOCH_SHUFFLE=1
export ROLLOUT_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=8 PP_SIZE=2 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export SAVE_INTERVAL=10 PREFETCH_ROLLOUT_DATA=1 OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_BALANCE_DATA=0 DISTRIBUTED_TIMEOUT_MINUTES=60
# The reused High-effort Phase 2 is a deterministic 1280-sample subset of the
# original 4093 trajectories, so its preserved sample indices are intentionally
# non-contiguous even though rollout_0..1279 are contiguous.
export OFFLINE_OPD_ALLOW_SAMPLE_SUBSET=1
export USE_WANDB="${USE_WANDB:-1}" WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-letme-forward-else-reverse-high}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE2=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'LOSS=cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor\n'
      printf 'LET_TOKEN_IDS=%s\nME_TOKEN_IDS=%s\n' "${OPD_LET_TOKEN_IDS}" "${OPD_ME_TOKEN_IDS}"
      printf 'EXACT_LET_ME_X_ROWS=teacher_owned_forward_kl_top128\n'
      printf 'OTHER_EXACT_ROWS=current_student_reverse_kl_top128\n'
      printf 'UNALIGNED_ROWS=rollout_student_anchor_kl_top128_plus_tail\n'
      printf 'FORWARD_ROW_COEF=1\nREVERSE_ROW_COEF=1\nSTUDENT_ANCHOR_COEF=%s\n' "${OPD_STUDENT_ANCHOR_KL_COEF}"
      printf 'TEACHER_RAW_CACHE_TOPK=1000\nSUPPORT_TOPK=128\n'
      printf 'SOURCE_SCORED_IMMUTABLE=1\nPHASE3_SHUFFLE=1\nPHASE3_BALANCE=0\n'
      printf 'PHASE3_TP=8\nPHASE3_PP=2\nPHASE3_CP=2\nPHASE3_EP=8\nPHASE3_ETP=1\n'
      printf 'PHASE3_GBS=64\nPHASE3_TRAIN_STEPS=20\nSAVE_INTERVAL=10\nLR=1e-6\nLR_SCHEDULE=constant\n'
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "conditional Let-me KL Phase 3 started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
