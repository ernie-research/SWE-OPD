#!/bin/bash
set -euo pipefail

# Re-score the original GLM-4.5-Air Phase-1 trajectories with the dense GLM-5.2
# teacher while retaining all ordinary Student content between </think> and the
# final valid <tool_call>. Then train the isolated 0.5 forward + 0.5 reverse KL
# objective from the base Student for two epochs.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-dynamic-t1k-s128-20260819_001719}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-dense-bidirectional-keep-postthink-f0.5-r0.5-t1k-s128-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
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

# Phase 2: Teacher-native template/prefix, Student semantic token stream.
export OPD_TEACHER_NATIVE_REASONING_PREFIX=1
export OPD_TEACHER_CANONICAL_TEMPLATE=0
export OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS=1
export OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE=1
export OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT=0
export TEACHER_ATTENTION_MODE="${TEACHER_ATTENTION_MODE:-dense}"
export TEACHER_PORT="${TEACHER_PORT:-15141}"
export TEACHER_NCCL_PORT_BASE="${TEACHER_NCCL_PORT_BASE:-24600}"
export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-1}"
export SCORE_FILE_CONCURRENCY="${SCORE_FILE_CONCURRENCY:-1}"
export PHASE2_BALANCE_MANIFESTS=0

# Phase 3: same settings as the prior bidirectional run, from the base model.
export OPD_FORWARD_KL_COEF="${OPD_FORWARD_KL_COEF:-0.5}"
export OPD_REVERSE_KL_COEF="${OPD_REVERSE_KL_COEF:-0.5}"
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
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-bidirectional-keep-postthink-f0.5-r0.5-t1k-s128}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE1=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'STUDENT_MODEL=%s\nSTUDENT_TORCH_DIST=%s\nTEACHER_MODEL=%s\n' \
         "${HF_CHECKPOINT}" "${LOAD_PATH}" "${TEACHER_MODEL}"
      printf 'LOSS=cached_teacher_bidirectional_topk_kl\nFORWARD_KL_COEF=%s\nREVERSE_KL_COEF=%s\n' \
         "${OPD_FORWARD_KL_COEF}" "${OPD_REVERSE_KL_COEF}"
      printf 'TEACHER_OWNED_FORWARD_SUPPORT_TOPK=128\nCURRENT_STUDENT_REVERSE_SUPPORT_TOPK=128\nTEACHER_RAW_CACHE_TOPK=1000\n'
      printf 'TEACHER_TEMPLATE=glm52_native_prefix_and_connectors_with_student_semantic_tokens\n'
      printf 'MASK_STUDENT_FORMAT_TOKENS=1\nMASK_STRUCTURAL_TOOL_NEWLINES=1\nMASK_POST_THINK_PRE_TOOL_CONTENT=0\n'
      printf 'KEEP_POST_THINK_PRE_TOOL_CONTENT=1\nMASK_LET_ME=0\nADDITIONAL_PHASE3_MASKING=0\n'
      printf 'PHASE2_REPLICAS=2\nPHASE2_NNODES_PER_REPLICA=2\nPHASE2_TP=16\nPHASE2_ATTENTION_MODE=dense\n'
      printf 'PHASE2_SCORE_CONCURRENCY_PER_REPLICA=1\nPHASE2_SAMPLES_PER_FILE=1\nPHASE2_BALANCE_MANIFESTS=0\n'
      printf 'PHASE3_DATA_SELECTION=per_epoch_random_shuffle\nPHASE3_BALANCE_DATA=0\nPHASE3_SHUFFLE_SEEDS=epoch_index\n'
      printf 'PHASE3_TP=8\nPHASE3_PP=2\nPHASE3_CP=2\nPHASE3_EP=8\nPHASE3_ETP=1\n'
      printf 'PHASE3_GBS=64\nPHASE3_EPOCHS=2\nPHASE3_STEPS=128\nSAVE_INTERVAL=20\nLR=1e-6\nLR_SCHEDULE=constant\n'
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "keep-postthink bidirectional GLM pipeline started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

if [[ ! -f "${RUN_ROOT}/.phase2.complete" ]]; then
   export OFFLINE_OPD_LOSS_TYPE=cached_teacher_dynamic_topk_reverse_kl
   bash "${SCRIPT_DIR}/02-phase2-teacher.sh" --foreground
fi
if [[ ! -f "${RUN_ROOT}/.phase3.complete" ]]; then
   export OFFLINE_OPD_LOSS_TYPE=cached_teacher_bidirectional_topk_kl
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
fi
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
