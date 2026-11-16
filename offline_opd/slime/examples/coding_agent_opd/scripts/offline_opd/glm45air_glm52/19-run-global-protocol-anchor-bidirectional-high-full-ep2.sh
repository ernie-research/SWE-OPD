#!/bin/bash
set -euo pipefail

# Phase-3-only experiment over the immutable full High-effort Phase-2 cache.
# Start from the original GLM-4.5-Air weights and repartition protocol-critical
# rows to the rollout-Student anchor while each sample is loaded. The source
# scored artifacts remain byte-for-byte unchanged.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-canonical-anchor-bidir-high-full4093-f0.5-r0.5-a1-ep2-gbs64-20260827_214929}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-global-protocol-anchor-bidir-high-full4093-f0.5-r0.5-a1-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
for path in "${SOURCE_RUN}" "${RUN_ROOT}"; do
   [[ "${path}" == "${OFFLINE_OPD_BASE}"/* ]] || {
      echo "experiment paths must be under ${OFFLINE_OPD_BASE}: ${path}" >&2
      exit 1
   }
done
[[ -d "${SOURCE_RUN}/scored/samples" && -f "${SOURCE_RUN}/.phase2.complete" ]] || {
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
link_once "${SOURCE_RUN}/dump" "${RUN_ROOT}/dump"
link_once "${SOURCE_RUN}/.phase2.complete" "${RUN_ROOT}/.phase2.complete"
link_once "${SOURCE_RUN}/glm45air_to_glm52_token_ids.json" \
   "${RUN_ROOT}/glm45air_to_glm52_token_ids.json"

export RUN_ROOT
export OPD_TOKEN_ID_BRIDGE="${RUN_ROOT}/glm45air_to_glm52_token_ids.json"
# This is the original model conversion, not a trained experiment checkpoint.
export LOAD_PATH="${CALLER_LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_bidirectional_topk_kl_with_student_anchor
export OPD_FORWARD_KL_COEF=0.5
export OPD_REVERSE_KL_COEF=0.5
export OPD_STUDENT_ANCHOR_KL_COEF=1.0

# New isolated policy: globally locate protocol token ids and move those rows
# to the rollout-Student anchor. Legacy destructive masks must remain disabled.
export OFFLINE_OPD_POSTHOC_ANCHOR_PROTOCOL_TOKENS=1
export OFFLINE_OPD_POSTHOC_MASK_TOOL_FORMAT=0
export OFFLINE_OPD_POSTHOC_MASK_LET_ME_PROB=0

export TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1
unset TRAIN_STEPS
export ROLLOUT_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=8 PP_SIZE=2 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export SAVE_INTERVAL=20 PREFETCH_ROLLOUT_DATA=1 OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_BALANCE_DATA=0 DISTRIBUTED_TIMEOUT_MINUTES=60
export OFFLINE_OPD_RESUME=0 OFFLINE_OPD_FINETUNE=0
# Keep the new randomized replay manifests inside this experiment rather than
# rebuilding the source experiment's replay directory.
export FANOUT_DIR="${FANOUT_DIR:-${RUN_ROOT}/replay_epochs_x2_b64}"
export OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS=1

export USE_WANDB="${USE_WANDB:-1}" WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-global-protocol-anchor-bidir-high-full}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE2=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'INITIAL_MODEL=%s\n' "${LOAD_PATH}"
      printf 'LOSS=cached_teacher_bidirectional_topk_kl_with_student_anchor\n'
      printf 'POSITION_POLICY=global_protocol_token_ids_to_rollout_student_anchor\n'
      printf 'POSTHOC_PROTOCOL_ANCHOR=1\nLEGACY_TOOL_MASK=0\nLET_ME_MASK=0\n'
      printf 'TEACHER_FORWARD_COEF=0.5\nTEACHER_REVERSE_COEF=0.5\nSTUDENT_ANCHOR_COEF=1.0\n'
      printf 'FORWARD_KL_ANNEAL_STEPS=%s\n' "${OPD_FORWARD_KL_ANNEAL_STEPS:-0}"
      printf 'TEACHER_RAW_CACHE_TOPK=1000\nSUPPORT_TOPK=128\nANCHOR_TAIL_BUCKET=1\n'
      printf 'PHASE3_GBS=64\nPHASE3_EPOCHS=2\nPHASE3_STEPS=128\nPHASE3_SHUFFLE=1\nPHASE3_BALANCE=0\n'
      printf 'PHASE3_TP=8\nPHASE3_PP=2\nPHASE3_CP=2\nPHASE3_EP=8\nPHASE3_ETP=1\n'
      printf 'MAX_CONTEXT_LEN=131072\nMAX_TOKENS_PER_GPU=65536\n'
      printf 'SAVE_INTERVAL=20\nLR=1e-6\nLR_SCHEDULE=constant\n'
      printf 'WANDB_RUN_ID=%s\n' "${WANDB_RUN_ID}"
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "global protocol-anchor Phase 3 started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

if [[ ! -f "${RUN_ROOT}/.phase3.complete" ]]; then
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
fi
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
