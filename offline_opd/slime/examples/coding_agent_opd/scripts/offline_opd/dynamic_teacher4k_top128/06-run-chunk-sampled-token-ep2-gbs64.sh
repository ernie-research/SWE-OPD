#!/bin/bash
set -euo pipefail

# Sampled-token OPD on the unified chunk artifact (Qwen3.5-9B base <- Qwen3.6-27B).
#
# Same tokenizer, so the identity aligner pairs every trainable row with the teacher's
# position for the same token. Trains chunk_teacher_sampled_token_pg_with_student_anchor,
# the loss the GLM-5.2 and DSV4-0731 recipes use.
#
# Source of the teacher value:
#   CHUNK_FROM_TEACHER_CACHE=1 (default): reuse SOURCE_RUN's teacher Top-1024 cache (the
#     dynamic_teacher1k_top128 run); the sampled token is in it for 99.98% of rows, the rest
#     become Student-anchor rows.
#   CHUNK_FROM_TEACHER_CACHE=0: SOURCE_RUN is a sampled-token Phase 2 (02-phase2-teacher.sh
#     with OPD_PHASE3_OBJECTIVE=sampled_token), which stores no Top-K.
# Every row is EXACT, so OPD_CHUNK_TIER exact/struct/chunk train the same rows.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
OFFLINE_OPD_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"
SOURCE_RUN="${SOURCE_RUN:-${OFFLINE_OPD_BASE}/qwen35-9b-base-yiyan-dynamic-t1k-s128-ep2-gbs64-reuse-t4k-20260815_005905}"
CHUNK_FROM_TEACHER_CACHE="${CHUNK_FROM_TEACHER_CACHE:-1}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-base-yiyan-chunk-sampled-token-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
for path in "${SOURCE_RUN}" "${RUN_ROOT}"; do
   [[ "${path}" == "${OFFLINE_OPD_BASE}"/* ]] || { echo "unsafe path: ${path}" >&2; exit 1; }
done
[[ -d "${SOURCE_RUN}/scored/samples" && -f "${SOURCE_RUN}/.phase2.complete" ]] || {
   echo "source Phase 2 is incomplete: ${SOURCE_RUN}" >&2
   exit 1
}

export OFFLINE_OPD_BASE RUN_ROOT
export OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/SFT/model/Qwen3.5-9B}"
export LOAD_PATH="${LOAD_PATH:-/path/to/SFT/model/Qwen3.5-9B_torch_dist}"
export OFFLINE_OPD_LOSS_TYPE=chunk_teacher_sampled_token_pg_with_student_anchor
export OPD_CHUNK_TIER="${OPD_CHUNK_TIER:-struct}"
export OPD_KL_COEF=1.0 OPD_STUDENT_ANCHOR_KL_COEF=1.0
export OPD_SAMPLED_TOKEN_CACHE_MISS_POLICY=anchor
export OPD_SAMPLED_TOKEN_CLIP_REFERENCE="${OPD_SAMPLED_TOKEN_CLIP_REFERENCE:-rollout}"
export OPD_SAMPLED_TOKEN_EPS_CLIP=0.2 OPD_SAMPLED_TOKEN_EPS_CLIP_HIGH=0.2
export OPD_SAMPLED_TOKEN_SDAR_GATE=0
# The chunk artifact carries no teacher Top-K, so there is nothing to measure overlap on.
export OFFLINE_OPD_TOPK_OVERLAP=0

mkdir -p "${RUN_ROOT}"
# For provenance only (Phase 3 reads scored/). The derived Top-1024 run reused its
# Phase 1 and has no dump/ of its own.
if [[ -d "${SOURCE_RUN}/dump" && ! -L "${RUN_ROOT}/dump" && ! -e "${RUN_ROOT}/dump" ]]; then
   ln -s "${SOURCE_RUN}/dump" "${RUN_ROOT}/dump"
fi
if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE2=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'HF_CHECKPOINT=%s\nLOAD_PATH=%s\n' "${HF_CHECKPOINT}" "${LOAD_PATH}"
      printf 'LOSS=chunk_teacher_sampled_token_pg_with_student_anchor\n'
      printf 'CHUNK_ALIGNER=identity\nCHUNK_FROM_TEACHER_CACHE=%s\nOPD_CHUNK_TIER=%s\n' \
         "${CHUNK_FROM_TEACHER_CACHE}" "${OPD_CHUNK_TIER}"
      printf 'PHASE3_GLOBAL_BATCH_SIZE=64\nPHASE3_EPOCHS=2\n'
      printf 'TP_SIZE=8\nPP_SIZE=1\nCP_SIZE=2\nMAX_CONTEXT_LEN=131072\nSAVE_INTERVAL=20\n'
      printf 'WANDB_RUN_ID=%s\n' "${OFFLINE_OPD_RUN_TAG}"
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "${SCRIPT_PATH}" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "Qwen chunk sampled-token Phase 3 started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

CHUNK_SRC_SCORED="${SOURCE_RUN}/scored" CHUNK_DST_SCORED="${RUN_ROOT}/scored" \
   CHUNK_ALIGNER=identity CHUNK_FROM_TEACHER_CACHE="${CHUNK_FROM_TEACHER_CACHE}" \
   bash "${OFFLINE_OPD_DIR}/build_chunk_scored.sh"
# 03-phase3-train.sh requires the Phase-2 marker; here it means "the chunk artifact is ready".
[[ -f "${RUN_ROOT}/.phase2.complete" ]] || cp "${RUN_ROOT}/scored/.chunk.complete" "${RUN_ROOT}/.phase2.complete"

if [[ ! -f "${RUN_ROOT}/.phase3.complete" ]]; then
   env TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1 ROLLOUT_BATCH_SIZE=64 GLOBAL_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 \
      TP_SIZE=8 PP_SIZE=1 CP_SIZE=2 SAVE_INTERVAL=20 USE_WANDB="${USE_WANDB:-1}" \
      WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}" WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}" \
      WANDB_GROUP="${WANDB_GROUP:-offline-opd-qwen35-9b-chunk-sampled-token-ep2}" \
      WANDB_RUN_ID="${OFFLINE_OPD_RUN_TAG}" \
      bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
fi
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
