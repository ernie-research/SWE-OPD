#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
OFFLINE_OPD_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"
SOURCE_RUN="${SOURCE_RUN:-/path/to/offline_opd/qwen35-9b-base-yiyan-dynamic-t4k-s128-ep2-gbs64-c500-20260813_024427}"
OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/qwen35-9b-base-yiyan-dynamic-t1k-s128-ep2-gbs64-reuse-t4k-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
[[ "${RUN_ROOT}" == "${OFFLINE_OPD_BASE}"/* ]] || { echo "unsafe RUN_ROOT: ${RUN_ROOT}" >&2; exit 1; }
SOURCE_SCORED_DIR="${SOURCE_RUN}/scored"
TARGET_SCORED_DIR="${RUN_ROOT}/scored"
export RUN_ROOT SOURCE_RUN SOURCE_SCORED_DIR TARGET_SCORED_DIR OFFLINE_OPD_BASE
export OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/SFT/model/Qwen3.5-9B}"
export LOAD_PATH="${LOAD_PATH:-/path/to/SFT/model/Qwen3.5-9B_torch_dist}"
export SLIME_VENV="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}"
export DERIVE_ROWS_PER_CHUNK="${DERIVE_ROWS_PER_CHUNK:-512}"
export DERIVE_DEVICE="${DERIVE_DEVICE:-auto}"
mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs-derive" "${RUN_ROOT}/logs-phase3"

if [[ "${1:-}" != --worker ]]; then
   {
      printf 'RUN_ROOT=%s\nSOURCE_RUN=%s\n' "${RUN_ROOT}" "${SOURCE_RUN}"
      printf 'SOURCE_TEACHER_CACHE_TOPK=4096\nOPD_TEACHER_CACHE_TOPK=1024\nOPD_DYNAMIC_SUPPORT_TOPK=128\n'
      printf 'HF_CHECKPOINT=%s\nLOAD_PATH=%s\n' "${HF_CHECKPOINT}" "${LOAD_PATH}"
      printf 'PHASE1=reused\nPHASE2=offline_derived_from_top4096\n'
      printf 'PHASE3_GLOBAL_BATCH_SIZE=64\nPHASE3_TRAIN_STEPS=128\nPHASE3_EPOCHS=2\n'
      printf 'TP_SIZE=8\nPP_SIZE=1\nCP_SIZE=2\nMAX_CONTEXT_LEN=131072\n'
      printf 'LR=1e-6\nLR_DECAY_STYLE=constant\nSAVE_INTERVAL=20\n'
      printf 'WANDB_RUN_ID=%s\n' "${OFFLINE_OPD_RUN_TAG}"
   } > "${RUN_ROOT}/experiment.env"
   nohup setsid bash "${SCRIPT_PATH}" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "Top-1024 derived-cache pipeline started: pid=$!, run=${RUN_ROOT}, log=${RUN_ROOT}/pipeline.log"
   exit 0
fi

printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(date -Is)" > "${RUN_ROOT}/.pipeline.running"
rm -f "${RUN_ROOT}/.pipeline.failed" "${RUN_ROOT}/.pipeline.complete"
pipeline_failed() {
   local rc=$?
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${RUN_ROOT}/.pipeline.failed"
   rm -f "${RUN_ROOT}/.pipeline.running"
   exit "${rc}"
}
trap pipeline_failed ERR INT TERM

if [[ ! -f "${TARGET_SCORED_DIR}/.teacher_cache_derived.json" ]]; then
   echo "[$(date -Is)] four-node Top-4096 -> Top-1024 derivation starts"
   export OFFLINE_OPD_RUN_SCRIPT="${SCRIPT_DIR}/derive-rank.sh"
   mpirun -n "${PADDLE_TRAINERS_NUM:?}" -N 1 \
      -x PADDLE_TRAINERS -x PADDLE_TRAINERS_NUM -x OFFLINE_OPD_RUN_SCRIPT \
      -x SOURCE_SCORED_DIR -x TARGET_SCORED_DIR -x SLIME_VENV \
      -x DERIVE_ROWS_PER_CHUNK -x DERIVE_DEVICE \
      "${OFFLINE_OPD_DIR}/_rank_dispatch.sh" \
      2>&1 | tee "${RUN_ROOT}/logs-derive/derive.log"
   "${SLIME_VENV}/bin/python3" -u "${SCRIPT_DIR}/derive_teacher_cache.py" \
      --source-dir "${SOURCE_SCORED_DIR}" --output-dir "${TARGET_SCORED_DIR}" \
      --source-k 4096 --target-k 1024 --support-k 128 --finalize-only \
      2>&1 | tee "${RUN_ROOT}/logs-derive/finalize.log"
fi
printf 'source=%s\ncompleted=%s\n' "${SOURCE_RUN}" "$(date -Is)" > "${RUN_ROOT}/.phase2.complete"

echo "[$(date -Is)] Phase 3 starts"
env \
   RUN_ROOT="${RUN_ROOT}" OFFLINE_OPD_RUN_TAG="${OFFLINE_OPD_RUN_TAG}" \
   HF_CHECKPOINT="${HF_CHECKPOINT}" LOAD_PATH="${LOAD_PATH}" \
   TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1 ROLLOUT_BATCH_SIZE=64 GLOBAL_BATCH_SIZE=64 \
   TP_SIZE=8 PP_SIZE=1 CP_SIZE=2 SAVE_INTERVAL=20 USE_WANDB=1 \
   WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}" WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}" \
   WANDB_GROUP="${WANDB_GROUP:-offline-opd-dynamic-t1k-s128-base-yiyan-ep2}" \
   WANDB_RUN_ID="${OFFLINE_OPD_RUN_TAG}" \
   bash "${SCRIPT_DIR}/03-phase3-train.sh" \
   > "${RUN_ROOT}/logs-phase3/launcher.log" 2>&1
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.phase3.complete"
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
rm -f "${RUN_ROOT}/.pipeline.running" "${RUN_ROOT}/.pipeline.failed"
trap - ERR INT TERM
