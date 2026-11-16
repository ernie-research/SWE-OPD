#!/bin/bash
set -euo pipefail

# New two-epoch reverse-KL experiment initialized from the final iter127 model
# weights of the canonical anchored bidirectional run.  Phase-2 scores are
# immutable and reused; optimizer/RNG/progress are deliberately not resumed.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"

SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/glm45air-glm52-canonical-anchor-bidir-high-full4093-f0.5-r0.5-a1-ep2-gbs64-20260827_214929}"
SOURCE_CKPT="${SOURCE_CKPT:-${SOURCE_RUN}/checkpoints}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-canonical-anchor-reverse-only-from-bidir-iter127-t1k-s128-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
for path in "${SOURCE_RUN}" "${RUN_ROOT}"; do
   [[ "${path}" == "${OFFLINE_OPD_BASE}"/* ]] || {
      echo "experiment path must be under ${OFFLINE_OPD_BASE}: ${path}" >&2
      exit 1
   }
done
[[ -f "${SOURCE_RUN}/.phase2.complete" && -d "${SOURCE_RUN}/scored/samples" ]] || {
   echo "source Phase 2 is incomplete: ${SOURCE_RUN}" >&2
   exit 1
}
[[ "$(<"${SOURCE_CKPT}/latest_checkpointed_iteration.txt")" == 127 ]] || {
   echo "source checkpoint tracker is not iter127: ${SOURCE_CKPT}" >&2
   exit 1
}
[[ -d "${SOURCE_CKPT}/iter_0000127" ]] || {
   echo "missing source iter127 checkpoint: ${SOURCE_CKPT}/iter_0000127" >&2
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
   elif [[ -e "${target}" ]]; then
      echo "refusing to replace existing path: ${target}" >&2
      return 1
   else
      ln -s "${source}" "${target}"
   fi
}
link_once "${SOURCE_RUN}/scored" "${RUN_ROOT}/scored"
link_once "${SOURCE_RUN}/.phase2.complete" "${RUN_ROOT}/.phase2.complete"
link_once "${SOURCE_RUN}/glm45air_to_glm52_token_ids.json" \
   "${RUN_ROOT}/glm45air_to_glm52_token_ids.json"

export RUN_ROOT RUN_TAG="$(basename "${RUN_ROOT}")"
export SCORED_DIR="${SOURCE_RUN}/scored"
export FANOUT_DIR="${RUN_ROOT}/replay-ep2-b64"
export SAVE_PATH="${RUN_ROOT}/checkpoints"
export RUN_DIR="${RUN_ROOT}/logs-phase3"
export LOAD_PATH="${SOURCE_CKPT}"
export OPD_TOKEN_ID_BRIDGE="${RUN_ROOT}/glm45air_to_glm52_token_ids.json"

export OFFLINE_OPD_LOSS_TYPE=cached_teacher_bidirectional_topk_kl_with_student_anchor
export OPD_FORWARD_KL_COEF=0
export OPD_REVERSE_KL_COEF=1
export OPD_STUDENT_ANCHOR_KL_COEF=1
export OFFLINE_OPD_FINETUNE=1
export OFFLINE_OPD_RESUME=0
export START_ROLLOUT_ID=0

export TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1
unset TRAIN_STEPS
export ROLLOUT_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=8 PP_SIZE=2 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export SAVE_INTERVAL=20 PREFETCH_ROLLOUT_DATA=1 OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_BALANCE_DATA=0 DISTRIBUTED_TIMEOUT_MINUTES=60
export OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS=1
export USE_WANDB="${USE_WANDB:-1}" WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-canonical-anchor-reverse-only-from-bidir-iter127}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${RUN_TAG}}"

mkdir -p "${RUN_DIR}"
if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_RUN=%s\nSOURCE_CHECKPOINT=%s\n' \
         "${RUN_ROOT}" "${SOURCE_RUN}" "${SOURCE_CKPT}/iter_0000127"
      printf 'INIT_POLICY=weights_only_finetune_no_optim_no_rng\nSTART_ITERATION=0\nSTART_ROLLOUT_ID=0\n'
      printf 'LOSS=cached_teacher_bidirectional_topk_kl_with_student_anchor\n'
      printf 'TEACHER_FORWARD_COEF=0\nTEACHER_REVERSE_COEF=1\nSTUDENT_ANCHOR_COEF=1\n'
      printf 'TEACHER_RAW_CACHE_TOPK=1000\nSUPPORT_TOPK=128\n'
      printf 'PHASE3_GBS=64\nPHASE3_EPOCHS=2\nPHASE3_EXPECTED_STEPS=128\nPHASE3_SHUFFLE=1\nPHASE3_BALANCE=0\n'
      printf 'PHASE3_TP=8\nPHASE3_PP=2\nPHASE3_CP=2\nPHASE3_EP=8\nPHASE3_ETP=1\n'
      printf 'SAVE_INTERVAL=20\nLR=1e-6\nLR_SCHEDULE=constant\n'
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "reverse-only continuation started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

printf 'pid=%s started=%s\n' "$$" "$(date -Is)" > "${RUN_ROOT}/.phase3.running"
rm -f "${RUN_ROOT}/.phase3.failed"
set +e
bash "${OFFLINE_OPD_DIR}/start-offline-opd.sh" phase3
rc=$?
set -e
if (( rc == 0 )); then
   printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.phase3.complete"
   rm -f "${RUN_ROOT}/.phase3.running"
else
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${RUN_ROOT}/.phase3.failed"
   rm -f "${RUN_ROOT}/.phase3.running"
fi
exit "${rc}"
