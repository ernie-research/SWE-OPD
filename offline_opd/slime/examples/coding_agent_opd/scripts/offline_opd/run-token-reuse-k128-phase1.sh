#!/bin/bash
set -euo pipefail

# Continuous sampled-token offline OPD experiment:
# reuse the completed K=128 Phase-1 trajectories, then run a fresh token-mode
# Phase 2 and a fresh ten-epoch Phase 3 under an independent experiment root.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR="${OFFLINE_OPD_BASE:-/path/to/OPD/offline_opd}"
SOURCE_PHASE1_DIR="${SOURCE_PHASE1_DIR:-${BASE_DIR}/qwen35-9b-k128-run3-c500-20260805_024453/dump}"

if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${BASE_DIR}/qwen35-9b-token-reverse-kl-run4-reuse-k128-phase1-c500-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
[[ "${RUN_ROOT}" == "${BASE_DIR}"/* ]] || {
   echo "RUN_ROOT must be under ${BASE_DIR}: ${RUN_ROOT}" >&2
   exit 1
}
[[ -r "${SOURCE_PHASE1_DIR}/rollout_0.pt" ]] || {
   echo "missing reusable Phase-1 data: ${SOURCE_PHASE1_DIR}/rollout_0.pt" >&2
   exit 1
}

export RUN_ROOT SOURCE_PHASE1_DIR
export RUN_TAG="${OFFLINE_OPD_RUN_TAG:-$(basename "${RUN_ROOT}")}"
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/workspace/env_run/<user>/Megatron-LM}"
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/SFT/results/qwen35-9b-scaleswe5000-sft-246step-add-finish/checkpoint-246}"
export LOAD_PATH="${LOAD_PATH:-/path/to/OPD/model/qwen35-9b-scaleswe5000-sft-246-add-finish_torch_dist}"
export TEACHER_MODEL="${TEACHER_MODEL:-/path/to/SFT/model/Qwen3.6-27B}"
export SAVE_PATH="${RUN_ROOT}/checkpoints"

# Shared batch layout.
export ROLLOUT_BATCH_SIZE=50
export GLOBAL_BATCH_SIZE=50
export NUM_STEPS_PER_ROLLOUT=1

# Phase 2: sampled-token teacher scoring, no Top-K payload.
export IN_DIR="${SOURCE_PHASE1_DIR}"
export OUT_DIR="${RUN_ROOT}/scored"
export PHASE2_SAMPLES_PER_FILE=50
export PHASE2_DYNAMIC_SHARDING=1
export PHASE2_SAMPLE_OUTPUT=1
export OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS=1
export OFFLINE_OPD_DIAGNOSTIC_TOPK=128
export PHASE1_SHUFFLE=0
export PHASE1_SHUFFLE_SEED=42
export PHASE1_SHUFFLE_BALANCE_KEY=trainable_tokens
export TEACHER_TP="${TEACHER_TP:-4}"
export TEACHER_DP="${TEACHER_DP:-2}"
export TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3,4,5,6,7}"
export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-8}"
export SCORE_MAX_ATTEMPTS="${SCORE_MAX_ATTEMPTS:-3}"
export TEACHER_LOGITS_CHUNK="${TEACHER_LOGITS_CHUNK:-2048}"

# Phase 3: same recipe as the completed K=128 run, except sampled-token loss.
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-10}"
export EPOCH_SHUFFLE="${EPOCH_SHUFFLE:-1}"
export TP_SIZE="${TP_SIZE:-8}"
export PP_SIZE="${PP_SIZE:-1}"
export CP_SIZE="${CP_SIZE:-2}"
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024 * 128))}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-30}"
export PREFETCH_ROLLOUT_DATA="${PREFETCH_ROLLOUT_DATA:-1}"
export OFFLINE_OPD_MANIFEST_LOAD_WORKERS="${OFFLINE_OPD_MANIFEST_LOAD_WORKERS:-8}"
export OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_RESUME=0

# wandb uses group as the visible run name when random suffixing is disabled.
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-qwen35-9b-offline-token-loss-with-top128-diagnostics-reuse-phase1-10ep-gbs50-tp8-pp1-cp2-lr1e-6}"
export WANDB_MODE="${WANDB_MODE:-online}"

mkdir -p "${RUN_ROOT}"
PIPELINE_LOG="${RUN_ROOT}/pipeline.log"
PIPELINE_RUNNING="${RUN_ROOT}/.pipeline.running"
PIPELINE_COMPLETE="${RUN_ROOT}/.pipeline.complete"
PIPELINE_FAILED="${RUN_ROOT}/.pipeline.failed"

if [[ "${1:-}" != "--worker" ]]; then
   nohup setsid bash "$0" --worker > "${PIPELINE_LOG}" 2>&1 < /dev/null &
   pipeline_pid=$!
   printf '%s\n' "${pipeline_pid}" > "${RUN_ROOT}/pipeline.pid"
   echo "Sampled-token continuous pipeline started: pid=${pipeline_pid}"
   echo "run=${RUN_ROOT}"
   echo "source_phase1=${SOURCE_PHASE1_DIR}"
   echo "wandb=${WANDB_GROUP}"
   echo "log=${PIPELINE_LOG}"
   exit 0
fi

printf 'pid=%s host=%s started=%s\n' "$$" "$(hostname)" "$(date -Is)" > "${PIPELINE_RUNNING}"
pipeline_failed() {
   local rc=$?
   printf 'rc=%s failed=%s\n' "${rc}" "$(date -Is)" > "${PIPELINE_FAILED}"
   rm -f "${PIPELINE_RUNNING}"
   exit "${rc}"
}
trap pipeline_failed ERR INT TERM

echo "[$(date -Is)] sampled-token pipeline starts"
echo "run=${RUN_ROOT}"
echo "source_phase1=${SOURCE_PHASE1_DIR}"
echo "wandb=${WANDB_GROUP}"

export RUN_DIR="${RUN_ROOT}/logs-phase2"
mkdir -p "${RUN_DIR}"
echo "[$(date -Is)] phase2 starts; log=${RUN_DIR}/launcher.log"
bash "${SCRIPT_DIR}/start-offline-token-opd.sh" phase2 > "${RUN_DIR}/launcher.log" 2>&1
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.phase2.complete"
echo "[$(date -Is)] phase2 completed"

export SCORED_DIR="${RUN_ROOT}/scored"
export RUN_DIR="${RUN_ROOT}/logs-phase3"
mkdir -p "${RUN_DIR}"
echo "[$(date -Is)] phase3 starts; log=${RUN_DIR}/launcher.log"
bash "${SCRIPT_DIR}/start-offline-token-opd.sh" phase3 > "${RUN_DIR}/launcher.log" 2>&1
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.phase3.complete"
echo "[$(date -Is)] phase3 completed"

printf 'completed=%s\n' "$(date -Is)" > "${PIPELINE_COMPLETE}"
rm -f "${PIPELINE_RUNNING}" "${PIPELINE_FAILED}"
trap - ERR INT TERM
echo "[$(date -Is)] sampled-token pipeline completed"
