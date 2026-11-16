#!/bin/bash
set -euo pipefail

# Short, isolated validation run for canonical cross-tokenizer Student anchors:
# random 640-sample Phase-1 subset -> canonical GLM-5.2 scoring -> 10 x GBS64.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"
SLIME_VENV="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-dynamic-t1k-s128-20260819_001719}"
SUBSET_COUNT="${SUBSET_COUNT:-640}"
SUBSET_SEED="${SUBSET_SEED:-20260827}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-canonical-anchor-bidir-f0.5-r0.5-a1-n${SUBSET_COUNT}-s10-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
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
PYTHONPATH="${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}" "${SLIME_VENV}/bin/python3" \
   "${OFFLINE_OPD_DIR}/create_phase1_subset.py" \
   --input-dir "${SOURCE_RUN}/dump" --output-dir "${RUN_ROOT}/dump" \
   --count "${SUBSET_COUNT}" --seed "${SUBSET_SEED}"
link_once "${SOURCE_RUN}/.phase1.complete" "${RUN_ROOT}/.phase1.complete"
link_once "${SOURCE_RUN}/glm45air_to_glm52_token_ids.json" \
   "${RUN_ROOT}/glm45air_to_glm52_token_ids.json"

export RUN_ROOT
export OPD_TOKEN_ID_BRIDGE="${RUN_ROOT}/glm45air_to_glm52_token_ids.json"
export LOAD_PATH="${CALLER_LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_bidirectional_topk_kl_with_student_anchor

export OPD_TEACHER_NATIVE_REASONING_PREFIX=0
export OPD_TEACHER_CANONICAL_TEMPLATE=1
export OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS=0
export OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE=0
export OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT=0
export TEACHER_ATTENTION_MODE="${TEACHER_ATTENTION_MODE:-dense}"
# Match the previously validated dense Phase-2 launch exactly: no separate
# page-cache pass and SGLang's native/default multithreaded weight loader.
export TEACHER_PREWARM_WEIGHTS=0
unset TEACHER_PREWARM_CONCURRENCY TEACHER_MODEL_LOADER_THREADS
export SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S="${SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S:-7200}"
export TEACHER_WAIT_TIMEOUT_SEC="${TEACHER_WAIT_TIMEOUT_SEC:-7200}"
export TEACHER_PORT="${TEACHER_PORT:-15141}"
export TEACHER_NCCL_PORT_BASE="${TEACHER_NCCL_PORT_BASE:-24600}"
export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-1}"
export SCORE_FILE_CONCURRENCY="${SCORE_FILE_CONCURRENCY:-1}"
export PHASE2_SAMPLES_PER_FILE=1
export PHASE2_BALANCE_MANIFESTS=0
export PHASE2_ROLLOUT_ID_FROM_FILENAME=1
export OFFLINE_OPD_ALLOW_SAMPLE_SUBSET=1

export OPD_FORWARD_KL_COEF="${OPD_FORWARD_KL_COEF:-0.5}"
export OPD_REVERSE_KL_COEF="${OPD_REVERSE_KL_COEF:-0.5}"
export OPD_STUDENT_ANCHOR_KL_COEF="${OPD_STUDENT_ANCHOR_KL_COEF:-1.0}"
export TRAIN_EPOCHS=1 TRAIN_STEPS=10 EPOCH_SHUFFLE=1
export ROLLOUT_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=8 PP_SIZE=2 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export SAVE_INTERVAL=10 PREFETCH_ROLLOUT_DATA=1 OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_BALANCE_DATA=0
export DISTRIBUTED_TIMEOUT_MINUTES=60
export USE_WANDB="${USE_WANDB:-1}" WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-canonical-anchor-bidir-n640-s10}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE1=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'PHASE1_SUBSET_COUNT=%s\nPHASE1_SUBSET_SEED=%s\n' "${SUBSET_COUNT}" "${SUBSET_SEED}"
      printf 'LOSS=cached_teacher_bidirectional_topk_kl_with_student_anchor\n'
      printf 'TEACHER_TEMPLATE=official_glm52_canonical\nPOSITION_POLICY=exact_1to1_teacher_else_rollout_student_anchor\n'
      printf 'TEACHER_FORWARD_COEF=%s\nTEACHER_REVERSE_COEF=%s\nSTUDENT_ANCHOR_COEF=%s\n' \
         "${OPD_FORWARD_KL_COEF}" "${OPD_REVERSE_KL_COEF}" "${OPD_STUDENT_ANCHOR_KL_COEF}"
      printf 'TEACHER_RAW_CACHE_TOPK=1000\nSUPPORT_TOPK=128\nANCHOR_TAIL_BUCKET=1\n'
      printf 'PHASE2_SAMPLES=640\nPHASE2_ATTENTION_MODE=%s\nPHASE2_BALANCE=0\n' "${TEACHER_ATTENTION_MODE}"
      printf 'PHASE3_GBS=64\nPHASE3_TRAIN_STEPS=10\nPHASE3_SHUFFLE=1\nPHASE3_BALANCE=0\n'
      printf 'SAVE_INTERVAL=10\nDISTRIBUTED_TIMEOUT_MINUTES=60\nLR=1e-6\nLR_SCHEDULE=constant\n'
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "canonical anchor n640/s10 pipeline started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

if [[ ! -f "${RUN_ROOT}/.phase2.complete" ]]; then
   bash "${SCRIPT_DIR}/02-phase2-teacher.sh" --foreground
fi
if [[ ! -f "${RUN_ROOT}/.phase3.complete" ]]; then
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
fi
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
