#!/bin/bash
set -euo pipefail

# Phase-3-only Top-16 experiment over the immutable canonical High-effort
# Phase-2 cache. Exact Student/Teacher rows use current-Student reverse-KL.
# Only rows that Phase 2 could not align one-to-one retain the rollout-Student
# anchor; no additional protocol-token anchors are introduced during loading.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-canonical-anchor-bidir-high-full4093-f0.5-r0.5-a1-ep2-gbs64-20260827_214929}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-crossvocab-anchor-reverse-t1k-s16-high-full4093-a1-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
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
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_dynamic_topk_reverse_kl_with_student_anchor
export OPD_ID_TOPK=16 OPD_DYNAMIC_SUPPORT_TOPK=16 OPD_TEACHER_CACHE_TOPK=1000
export OPD_STUDENT_ANCHOR_KL_COEF=1.0

# The immutable Phase-2 cache stores Top-128 snapshots. Narrow only the
# probability-ranked support tensors to their first 16 columns after loading;
# keep the full Teacher Top-1000 candidate cache for current-Student lookup.
export OFFLINE_OPD_ALLOW_SUPPORT_TOPK_TRUNCATION=1

# Do not add protocol anchors or destructive masks during Phase 3. The only
# anchor rows are the canonical cross-tokenizer non-1:1 rows saved by Phase 2.
export OFFLINE_OPD_POSTHOC_ANCHOR_PROTOCOL_TOKENS=0
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
export FANOUT_DIR="${FANOUT_DIR:-${RUN_ROOT}/replay_epochs_x2_b64}"
export OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS=1

export USE_WANDB="${USE_WANDB:-1}" WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-crossvocab-anchor-reverse-t1k-s16}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE2=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'INITIAL_MODEL=%s\n' "${LOAD_PATH}"
      printf 'LOSS=cached_teacher_dynamic_topk_reverse_kl_with_student_anchor\n'
      printf 'POSITION_POLICY=phase2_unaligned_rows_to_rollout_student_anchor_only\n'
      printf 'POSTHOC_PROTOCOL_ANCHOR=0\nLEGACY_TOOL_MASK=0\nLET_ME_MASK=0\n'
      printf 'TEACHER_FORWARD_COEF=0\nTEACHER_REVERSE_COEF=1\nSTUDENT_ANCHOR_COEF=1.0\n'
      printf 'TEACHER_RAW_CACHE_TOPK=1000\nSOURCE_SUPPORT_TOPK=128\nSUPPORT_TOPK=16\n'
      printf 'RUNTIME_SUPPORT_TRUNCATION=1\nANCHOR_TAIL_BUCKET=1\n'
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
   echo "cross-tokenizer-anchor reverse-only Top-16 Phase 3 started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

if [[ ! -f "${RUN_ROOT}/.phase3.complete" ]]; then
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
fi
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
