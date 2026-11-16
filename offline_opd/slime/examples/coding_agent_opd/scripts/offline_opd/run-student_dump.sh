#!/bin/bash
set -euo pipefail
set -x
# OFFLINE OPD PHASE 1: exact student rollout over the dataset, with no teacher/trainer.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# The shared settings predate pipefail and probe optional NVLink support with a
# grep pipeline whose no-match status is expected. Keep the stricter setting for
# this script, but do not let that probe terminate the phase before Ray starts.
set +o pipefail
source "${SCRIPT_DIR}/../multi_nodes/general_settings.sh"
set -o pipefail
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

RAY_GPUS="${RAY_GPUS:-0,1,2,3,4,5,6,7}"
RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-1}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS_PER_NODE="${ROLLOUT_NUM_GPUS_PER_NODE:-8}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-$(( NNODES * ROLLOUT_NUM_GPUS_PER_NODE ))}"
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"

if (( ROLLOUT_NUM_GPUS % ROLLOUT_GPUS_PER_ENGINE != 0 )); then
   echo "ROLLOUT_NUM_GPUS=${ROLLOUT_NUM_GPUS} must be divisible by ROLLOUT_GPUS_PER_ENGINE=${ROLLOUT_GPUS_PER_ENGINE}" >&2
   exit 1
fi

preclean_ray
source "${MODEL_ARGS_SH:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}"

MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024*128))}"
MAX_GEN_LEN="${MAX_GEN_LEN:-$((1024*16))}"
PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/data/swe_train.jsonl}"
DUMP_DIR="${DUMP_DIR:-/path/to/OPD/offline_opd/${EXP_NAME}_${RUN_TAG}/dump}"
OPD_ID_TOPK="${OPD_ID_TOPK:-1024}"
OFFLINE_OPD_LOSS_TYPE="${OFFLINE_OPD_LOSS_TYPE:-topk_reverse_kl}"
case "${OFFLINE_OPD_LOSS_TYPE}" in
   topk_reverse_kl|token_reverse_kl|cached_teacher_dynamic_topk_reverse_kl) ;;
   *) echo "unsupported OFFLINE_OPD_LOSS_TYPE=${OFFLINE_OPD_LOSS_TYPE}" >&2; exit 1 ;;
esac
# Dynamic Phase 3 still needs the rollout student's ranked Top-K snapshot for
# drift metrics. Phase 1 therefore uses the existing capture-only reverse schema;
# no teacher or training loss runs in this phase.
PHASE1_OPD_LOSS_TYPE="${OFFLINE_OPD_LOSS_TYPE}"
if [[ "${PHASE1_OPD_LOSS_TYPE}" == "cached_teacher_dynamic_topk_reverse_kl" ]]; then
   PHASE1_OPD_LOSS_TYPE=topk_reverse_kl
fi
OFFLINE_OPD_CONCURRENCY="${OFFLINE_OPD_CONCURRENCY:-100}"
[[ "${OFFLINE_OPD_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || {
   echo "OFFLINE_OPD_CONCURRENCY must be a positive integer: ${OFFLINE_OPD_CONCURRENCY}" >&2
   exit 1
}
export OFFLINE_OPD_CONCURRENCY
# Phase-1-only sparse snapshot used by offline drift diagnostics.  The SGLang
# switch is separate from the ids-only fast path, so standard online OPD does not
# return or persist this additional [N_trainable, K] matrix.
export OFFLINE_OPD_SAVE_STUDENT_TOPK_LOGPROBS="${OFFLINE_OPD_SAVE_STUDENT_TOPK_LOGPROBS:-1}"
if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "token_reverse_kl" ]]; then
   # Sampled-token OPD consumes the generated token ids already present in
   # sample.tokens; do not request or persist a separate Top-K snapshot.
   export OFFLINE_OPD_SAVE_STUDENT_TOPK_LOGPROBS=0
fi
export SGLANG_OUTPUT_TOP_LOGPROBS_B64="${SGLANG_OUTPUT_TOP_LOGPROBS_B64:-true}"
# Ray dynamically opens worker ports below 20000 while rollout engines are being
# allocated.  Use a dedicated non-overlapping range for this four-node phase;
# the core default remains unchanged for standard online OPD.
export SLIME_ROLLOUT_ENGINE_PORT_BASE="${SLIME_ROLLOUT_ENGINE_PORT_BASE:-22000}"

[[ -r "${PROMPT_DATA}" ]] || { echo "PROMPT_DATA not readable: ${PROMPT_DATA}" >&2; exit 1; }
mkdir -p "${DUMP_DIR}"

DATASET_SIZE="$(python3 - "${PROMPT_DATA}" <<'PY'
import sys
with open(sys.argv[1], "rb") as f:
    print(sum(bool(line.strip()) for line in f))
PY
)"
if (( DATASET_SIZE <= 0 )); then
   echo "empty dataset: ${PROMPT_DATA}" >&2
   exit 1
fi

# A fixed batch is required by slime's rollout/train contract. Pick the largest divisor
# no greater than the preferred batch so the final rollout never wraps into epoch two.
if [[ -n "${ROLLOUT_BATCH_SIZE:-}" ]]; then
   if (( DATASET_SIZE % ROLLOUT_BATCH_SIZE != 0 )); then
      echo "ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE} does not divide DATASET_SIZE=${DATASET_SIZE}; choose a divisor to guarantee exactly-once coverage." >&2
      exit 1
   fi
else
   # Offline coding trajectories spend substantial time waiting on sandbox/tool
   # calls.  Keep enough samples in flight to hide that latency across all 16
   # TP2 engines in the four-node layout.  For the 500-sample dataset this picks
   # batch=50 (10 atomic files); callers can still select another exact divisor.
   PREFERRED_ROLLOUT_BATCH_SIZE="${PREFERRED_ROLLOUT_BATCH_SIZE:-50}"
   ROLLOUT_BATCH_SIZE="$(python3 - "${DATASET_SIZE}" "${PREFERRED_ROLLOUT_BATCH_SIZE}" <<'PY'
import sys
n, preferred = map(int, sys.argv[1:])
print(next(size for size in range(min(n, preferred), 0, -1) if n % size == 0))
PY
)"
fi
N_SAMPLES_PER_PROMPT=1
EXPECTED_NUM_ROLLOUT=$(( DATASET_SIZE / ROLLOUT_BATCH_SIZE ))
if [[ -n "${NUM_ROLLOUT:-}" && "${NUM_ROLLOUT}" -ne "${EXPECTED_NUM_ROLLOUT}" ]]; then
   echo "NUM_ROLLOUT=${NUM_ROLLOUT} would not cover the dataset exactly once; expected ${EXPECTED_NUM_ROLLOUT}" >&2
   exit 1
fi
NUM_ROLLOUT="${EXPECTED_NUM_ROLLOUT}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}"
if (( GLOBAL_BATCH_SIZE != ROLLOUT_BATCH_SIZE )); then
   echo "Phase 1 uses one debug-only step per rollout, so GLOBAL_BATCH_SIZE must equal ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE}" >&2
   exit 1
fi

# Resume only from a fully validated contiguous prefix.  The custom rollout function
# consumes the matching dataset prefix once, preserving sample indices and shuffle order.
EXISTING_NUM_ROLLOUT=0
while [[ -f "${DUMP_DIR}/rollout_${EXISTING_NUM_ROLLOUT}.pt" ]]; do
   EXISTING_NUM_ROLLOUT=$(( EXISTING_NUM_ROLLOUT + 1 ))
done
if (( EXISTING_NUM_ROLLOUT > NUM_ROLLOUT )); then
   echo "${DUMP_DIR} has ${EXISTING_NUM_ROLLOUT} files, more than expected ${NUM_ROLLOUT}" >&2
   exit 1
fi
if (( EXISTING_NUM_ROLLOUT > 0 )); then
   VERIFY_ARGS=(--opd-loss-type "${PHASE1_OPD_LOSS_TYPE}")
   if [[ "${PHASE1_OPD_LOSS_TYPE}" == "topk_reverse_kl" ]]; then
      VERIFY_ARGS+=(--require-student-topk-logprobs)
   fi
   python3 "${SCRIPT_DIR}/verify_offline_pipeline.py" --mode dump-dir \
      --dir "${DUMP_DIR}" --opd-id-topk "${OPD_ID_TOPK}" "${VERIFY_ARGS[@]}" \
      --expected-samples "$(( EXISTING_NUM_ROLLOUT * ROLLOUT_BATCH_SIZE ))" \
      --allow-partial-index-set --dataset-size "${DATASET_SIZE}"
fi
START_ROLLOUT_ID="${START_ROLLOUT_ID:-${EXISTING_NUM_ROLLOUT}}"
if (( START_ROLLOUT_ID != EXISTING_NUM_ROLLOUT )); then
   echo "START_ROLLOUT_ID=${START_ROLLOUT_ID} must equal the validated file prefix ${EXISTING_NUM_ROLLOUT}" >&2
   exit 1
fi
if (( START_ROLLOUT_ID == NUM_ROLLOUT )); then
   echo "PHASE1 already complete: ${NUM_ROLLOUT} validated files in ${DUMP_DIR}"
   exit 0
fi

echo "PHASE1 exact continuous dump: dataset=${DATASET_SIZE}, file_batch=${ROLLOUT_BATCH_SIZE}, concurrency=${OFFLINE_OPD_CONCURRENCY}, files=${NUM_ROLLOUT}, resume=${START_ROLLOUT_ID}, GPUs=${ROLLOUT_NUM_GPUS}, output=${DUMP_DIR}"

WANDB_ARGS=()
if [[ "${USE_WANDB:-0}" -eq 1 ]]; then
   WANDB_ARGS=(
      --use-wandb
      --wandb-team "${WANDB_TEAM:-swe_opd}"
      --wandb-project "${WANDB_PROJECT:-slime-OPD}"
      --wandb-group "${WANDB_GROUP:-${EXP_NAME}}"
      --wandb-id "${RUN_TAG}"
      --wandb-mode "${WANDB_MODE:-online}"
   )
fi

CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/path/to/SFT/results/qwen35-9b-scaleswe5000-sft-246step-add-finish/checkpoint-246}"
   --load "${LOAD_PATH:-/path/to/OPD/model/qwen35-9b-scaleswe5000-sft-246-add-finish_torch_dist}"
)
ROLLOUT_ARGS=(
   --rollout-function-path examples.coding_agent_opd.scripts.offline_opd.exact_rollout.generate_rollout_exact
   --custom-generate-function-path examples.coding_agent_opd.generate.generate
   --prompt-data "${PROMPT_DATA}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT}"
   --start-rollout-id "${START_ROLLOUT_ID}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt 1
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --rollout-max-response-len "${MAX_GEN_LEN}"
   --rollout-temperature 1.0
   --num-steps-per-rollout 1
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
   --log-multi-turn
   --balance-data
)
if [[ "${PHASE1_OPD_LOSS_TYPE}" == "token_reverse_kl" ]]; then
   OPD_ARGS=(--opd-loss-type token_reverse_kl --opd-id-topk 0)
else
   OPD_ARGS=(--opd-loss-type topk_reverse_kl --opd-id-topk "${OPD_ID_TOPK}")
fi
OPTIMIZER_ARGS=(--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98)
PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-4}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_CONTEXT_LEN}"
)

NUM_ROLLOUT_ENGINES=$(( ROLLOUT_NUM_GPUS / ROLLOUT_GPUS_PER_ENGINE ))
# This semaphore is per engine and therefore has to round up.  The offline
# producer itself remains the exact global cap (100 by default).
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-$(( (OFFLINE_OPD_CONCURRENCY + NUM_ROLLOUT_ENGINES - 1) / NUM_ROLLOUT_ENGINES ))}"
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.85}"
   # slime prefixes SGLang ServerArgs with `sglang_`.  The custom rollout loop
   # reads args.sglang_tool_call_parser to parse raw /generate output back into
   # executable actions.  The unprefixed flag can be accepted by another parser
   # but leaves that field unset; Qwen's XML fallback masked this, while GLM's
   # native <arg_key>/<arg_value> calls then became 100 no-op turns.
   --sglang-tool-call-parser "${TOOL_CALL_PARSER:-qwen3_coder}"
   --use-distributed-post
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   --sglang-attention-backend "${SGLANG_ATTN_BACKEND:-flashinfer}"
)
if [[ -n "${REASONING_PARSER:-}" ]]; then
   SGLANG_ARGS+=(--sglang-reasoning-parser "${REASONING_PARSER}")
fi
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash)
DEBUG_ARGS=(--save-debug-rollout-data "${DUMP_DIR}/rollout_{rollout_id}.pt")

export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-1}"
setup_proxy
cd "${SLIME_DIR}"
build_runtime_env_json

if [[ "${RANK}" -eq 0 ]]; then
   CUDA_VISIBLE_DEVICES="${RAY_GPUS}" "${RAY_LAUNCH[@]}" start --head --node-ip-address "${POD_IP}" \
      --port "${RAY_HEAD_PORT}" --num-gpus "${RAY_NUM_GPUS_PER_NODE}" \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"
   sleep "${WAIT_WORKER_SECS}"
   ray status

   set +x
   SUBMIT_RC=0
   ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- "${SLIME_VENV}/bin/python3" -u train.py \
      --debug-rollout-only \
      --actor-num-nodes "${ACTOR_NUM_NODES}" \
      --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
      --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
      --num-gpus-per-node "${ROLLOUT_NUM_GPUS_PER_NODE}" \
      --placement-group-head-ip "${MASTER_ADDR}" \
      "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" "${OPD_ARGS[@]}" "${WANDB_ARGS[@]}" \
      "${PERF_ARGS[@]}" "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" "${DEBUG_ARGS[@]}" || SUBMIT_RC=$?
   set -x

   if [[ "${SUBMIT_RC}" -eq 0 ]]; then
      VERIFY_ARGS=(--opd-loss-type "${PHASE1_OPD_LOSS_TYPE}")
      if [[ "${PHASE1_OPD_LOSS_TYPE}" == "topk_reverse_kl" ]]; then
         VERIFY_ARGS+=(--require-student-topk-logprobs)
      fi
      python3 "${SCRIPT_DIR}/verify_offline_pipeline.py" --mode dump-dir \
         --dir "${DUMP_DIR}" --opd-id-topk "${OPD_ID_TOPK}" \
         "${VERIFY_ARGS[@]}" \
         --expected-samples "${DATASET_SIZE}" || SUBMIT_RC=$?
   else
      dump_engine_logs
   fi
   ray stop --force || true
   exit "${SUBMIT_RC}"
else
   # Retry only the initial join.  A successful non-blocking `ray start` leaves
   # the worker daemon running while this mpirun rank exits.  Using `--block`
   # here makes a normal head shutdown return an error and re-enter the retry
   # loop forever, so a completed offline phase can never finish its launcher.
   until CUDA_VISIBLE_DEVICES="${RAY_GPUS}" "${RAY_LAUNCH[@]}" start --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" \
      --node-ip-address "${POD_IP}" --num-gpus "${RAY_NUM_GPUS_PER_NODE}" --disable-usage-stats; do
      ray stop --force || true
      sleep 5
   done
fi
