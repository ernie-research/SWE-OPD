#!/bin/bash
set -euo pipefail

# Phase-3-only SDAR-gated sampled-token OPD over the immutable canonical
# High-effort Teacher Top-1000 cache.  There is no RL advantage in this offline
# experiment. Exact Teacher hits use:
#   delta = log p_teacher(sampled_token) - log p_current_student(sampled_token)
#   gate  = stopgrad(sigmoid(beta * delta)), beta=5
#   advantage = gate * delta
# with the existing clipped current/rollout PPO ratio. Tokenizer/template rows
# without an exact Teacher value retain the rollout-Student anchor.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-canonical-anchor-bidir-high-full4093-f0.5-r0.5-a1-ep2-gbs64-20260827_214929}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-crossvocab-sdar-sampled-token-pg-beta5-t1k-high-full4093-a1-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
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
export LOAD_PATH="${CALLER_LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_sampled_token_pg_with_student_anchor
export OPD_TEACHER_CACHE_TOPK=1000 OPD_DYNAMIC_SUPPORT_TOPK=128 OPD_ID_TOPK=128
export OPD_KL_COEF=1.0 OPD_STUDENT_ANCHOR_KL_COEF=1.0
export OPD_SAMPLED_TOKEN_CACHE_MISS_POLICY=anchor
export OPD_SAMPLED_TOKEN_EPS_CLIP=0.2 OPD_SAMPLED_TOKEN_EPS_CLIP_HIGH=0.2
export OPD_SAMPLED_TOKEN_SDAR_GATE=1 OPD_SDAR_GATE_BETA=5.0
export OPD_SDAR_GATE_SCALE="${OPD_SDAR_GATE_SCALE:-1.0}"

export OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_POSTHOC_ANCHOR_PROTOCOL_TOKENS=0
export OFFLINE_OPD_POSTHOC_MASK_TOOL_FORMAT=0
export OFFLINE_OPD_POSTHOC_MASK_LET_ME_PROB=0

export TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1
unset TRAIN_STEPS
export ROLLOUT_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=8 PP_SIZE=2 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export SAVE_INTERVAL=20 PREFETCH_ROLLOUT_DATA=1
export OFFLINE_OPD_BALANCE_DATA=0 DISTRIBUTED_TIMEOUT_MINUTES=60
export OFFLINE_OPD_RESUME=0 OFFLINE_OPD_FINETUNE=0
export FANOUT_DIR="${FANOUT_DIR:-${RUN_ROOT}/replay_epochs_x2_b64}"
export OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS=1

export USE_WANDB="${USE_WANDB:-1}" WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-crossvocab-sdar-sampled-token-beta5-t1k}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE2=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'INITIAL_MODEL=%s\n' "${LOAD_PATH}"
      printf 'LOSS=cached_teacher_sampled_token_pg_with_student_anchor\n'
      printf 'SDAR_GATE=1\nSDAR_BETA=5.0\nSDAR_GATE_SCALE=%s\nRL_ADVANTAGE=none\n' \
         "${OPD_SDAR_GATE_SCALE}"
      printf 'TEACHER_GAP=logp_teacher-logp_current_student\n'
      printf 'GATE=scale*stopgrad(sigmoid(beta*teacher_gap))\n'
      printf 'TEACHER_ADVANTAGE=gate*stopgrad(teacher_gap)\n'
      printf 'POLICY_RATIO=exp(logp_current-logp_rollout)\n'
      printf 'PPO_CLIP_LOW=0.2\nPPO_CLIP_HIGH=0.2\n'
      printf 'TEACHER_COEF=1.0\nROLLOUT_ANCHOR_COEF=1.0\n'
      printf 'CACHE_MISS_POLICY=rollout_sampled_token_anchor\n'
      printf 'TEACHER_RAW_CACHE_TOPK=1000\n'
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
   echo "SDAR-gated sampled-token Phase 3 started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

if [[ ! -f "${RUN_ROOT}/.phase3.complete" ]]; then
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
fi
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
