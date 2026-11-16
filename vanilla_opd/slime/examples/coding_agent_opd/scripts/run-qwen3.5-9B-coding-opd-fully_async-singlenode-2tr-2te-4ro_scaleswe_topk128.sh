#!/bin/bash
# ==============================================================================
# SINGLE-NODE (8 GPU) validation of the fully-async topk128 OPD run.
#
# Downstream of: run-qwen3.5-9B-coding-opd-fully_async-multinode-8tr-8te-16ro_scaleswe_topk128.sh
# Same experiment semantics (topk_reverse_kl, OPD_ID_TOPK=128, scale_swe data,
# pop_longest_first buffer filter), but squeezed onto ONE node with 8 GPUs so the
# whole teacher/actor/rollout triangle can be validated without a multi-node
# launch. Use it to smoke the environment + the pipeline; use the multinode
# script for the real sweep.
#
# ---------------- GPU allocation (8 x H800, 183GB) ----------------
#   teacher  Qwen3.6-27B sglang tp2 dp1    -> GPUs 6,7   (outside ray)
#   actor    Qwen3.5-9B  megatron TP2xDP1  -> GPUs 0,1   (ray)
#   rollout  Qwen3.5-9B  sglang 2 engines x 2 GPU (tp2) -> GPUs 2-5 (ray)
# ray sees 6 GPUs (0-5); 6,7 are held out via CUDA_VISIBLE_DEVICES.
#
# Why this split:
#   * teacher 2 GPUs is the floor: 27B bf16 ~= 54GB of weights, tp2 -> 27GB/GPU,
#     leaving ~150GB/GPU for kv-cache/activations -> healthy headroom for the
#     one-shot-prefill logprob path logits (--chunked-prefill-size -1).
#   * actor 2 GPUs TP2: num_attention_heads 16 % 2 == 0, num_query_groups 4 % 2 == 0
#     -> clean TP split, no KV-head-copy / output-gate patch needed (that patch is
#     only exercised at TP > num_query_groups, i.e. TP8 in the multinode variant).
#   * rollout 4 GPUs as 2 engines x 2 GPUs keeps ROLLOUT_GPUS_PER_ENGINE=2 and the
#     in-flight arithmetic IDENTICAL to the multinode script (ports start at 15000,
#     stride 2) so knobs keep their meaning.
#   * DP1 on the actor: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True and
#     --log-probs-chunk-size 8192 keep the peak bounded; --max-tokens-per-gpu is
#     LOWERED to 32768 vs the multinode 131072 because TP2 (vs TP8) leaves the
#     [T, V/TP] fp32 logits 4x wider: 32k -> ~16GB/rank, 131k -> ~64GB/rank.
#
# ---- TRAJECTORY BUDGET (measured 2026-11-15 on this box) ----
# Un-capped scale_swe trajectories run to 65k-73k response tokens (100 agent turns
# x ~700 tok). At TP2 that is a [T, V/TP] fp32 logits tensor of ~32-36GB per rank
# and it broke BOTH sides of the triangle:
#   * actor (@ --max-tokens-per-gpu 65536): torch.OutOfMemoryError during
#     vocab_parallel_logprobs_on_ids (32.41 GiB request vs 32.39 GiB free);
#   * teacher (Qwen3.6-27B tp2, mem-fraction 0.8 -> 34.5GB pool headroom): a
#     73123-token scoring prefill needs ~36GB of logits -> GPU pressure, and the
#     run died moments later with a SIGKILLed detokenizer.
# So this validation pins the trajectory budget: MAX_CONTEXT_LEN=32768 (hard
# truncation ceiling) AND SLIME_AGENT_MAX_STEPS=40 (turn count -> ~28k tokens in
# practice), which keeps the scoring prefill + student logits at ~16GB/rank on
# both sides. The multinode sweep keeps 131072 @ TP8 (logits ~16GB/rank there).
#
# NUM_ROLLOUT defaults to 4 for a quick end-to-end validation; the teacher/actor/
# rollout code paths are the same as the sweep.
# ==============================================================================
set -ex

# ============ force single-node topology ============
# This box runs inside a PDC multi-node job (PADDLE_TRAINERS = 5 hosts) but this
# script is single-node by design: pin the topology vars BEFORE general_settings
# reads them. Anything already exported in the environment wins; the prefix below
# is the documented one-node setup.
export PADDLE_TRAINERS="${PADDLE_TRAINERS_SINGLE:-127.0.0.1}"
export PADDLE_TRAINERS_NUM="${PADDLE_TRAINERS_NUM_SINGLE:-1}"
export PADDLE_TRAINER_ID="${PADDLE_TRAINER_ID_SINGLE:-0}"
# Trajectory budget (see the header): must be set BEFORE general_settings sources
# the RLE env, since it exports SLIME_AGENT_MAX_STEPS into the sandbox runtime.
export SLIME_AGENT_MAX_STEPS="${SLIME_AGENT_MAX_STEPS:-40}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# ============ shared layer (see general_settings.sh) ============
# Provides, at source time: slime-venv PATH + RAY_LAUNCH, PDC topology with the
# overrides above (NNODES=1, RANK=0, MASTER_ADDR=127.0.0.1), RUN_DIR + awk log
# fan-out, NCCL/GLOO ifname, cudnn/NVLink, SWE+RLE sandbox env (incl. the RLE
# secret file), SLIME_HTTP_TIMING, SGLANG_TOP_LOGPROBS_IDS_B64, metrics vars,
# and the helpers: preclean_ray, dump_engine_logs, metrics_poller,
# build_debug_args, setup_proxy, build_runtime_env_json.
source "${SCRIPT_DIR}/utils/general_settings.sh"

# ============ teacher: local sglang server on the held-out GPUs ============
# Qwen3.6-27B bf16 ~54GB -> tp2 leaves plenty of room for the kv-cache. Served
# straight from the HF dir. Same flags as the multinode variant -- chunked-prefill
# -1 is REQUIRED for topk_reverse_kl (one-shot prefill), ports pinned to dodge the
# node's reserved-port band.
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_NCCL_PORT_BASE="${TEACHER_NCCL_PORT_BASE:-13150}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/Qwen3.6-27B}"
TEACHER_GPUS="${TEACHER_GPUS:-6,7}"
TEACHER_TP="${TEACHER_TP:-2}"
TEACHER_DP="${TEACHER_DP:-1}"
TEACHER_LAUNCH_RETRIES="${TEACHER_LAUNCH_RETRIES:-3}"
TEACHER_READY_TIMEOUT_SEC="${TEACHER_READY_TIMEOUT_SEC:-600}"
TEACHER_LOG="${RUN_DIR}/sglang_teacher.log"
TEACHER_BG_PID=""

TEACHER_METRICS_ARGS=()
if [[ "${ENABLE_TEACHER_METRICS:-1}" -eq 1 ]]; then
   TEACHER_METRICS_ARGS=(
      --enable-metrics
      --enable-metrics-for-all-schedulers
      --enable-cache-report
      --decode-log-interval "${SGLANG_DECODE_LOG_INTERVAL:-100}"
   )
fi

teacher_healthy() {
   curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" > /dev/null 2>&1
}

# ONE sglang boot; sets TEACHER_BG_PID.
launch_teacher_process() {
   # --chunked-prefill-size -1 (one-shot prefill) is REQUIRED for topk_reverse_kl:
   # reward_func sends a per-position token_ids_logprob matrix for the whole response
   # region; with chunked prefill the id matrix stays full-length while pruned_len
   # shrinks -> get_token_ids_logprobs_raw asserts and the teacher SIGQUITs.
   # --disable-custom-all-reduce: at TP2 the custom allreduce's
   # cudaIpcGetMemHandle dies with CUDA error: invalid argument during CUDA graph
   # capture (custom_all_reduce.cuh:37) -> "Capture cuda graph failed" on boot.
   # Falls back to NCCL allreduce; slight cost on the teacher only.
   SGLANG_OPD_TOPK_OVERLAP=1 \
   SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1 SGLANG_LOGITS_PROCESSER_CHUNK_SIZE=$((1024*8)) \
   CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" python3 -m sglang.launch_server \
      --model-path "${TEACHER_MODEL}" \
      --host 0.0.0.0 \
      --port ${TEACHER_PORT} \
      --tp ${TEACHER_TP} \
      --dp-size ${TEACHER_DP} \
      --nccl-port ${TEACHER_NCCL_PORT_BASE} \
      --trust-remote-code \
      --tool-call-parser qwen3_coder \
      --attention-backend "${TEACHER_ATTN_BACKEND:-flashinfer}" \
      --chunked-prefill-size -1 \
      --disable-custom-all-reduce \
      --disable-radix-cache \
      --max-prefill-tokens $((1024*256)) \
      --mem-fraction-static "${TEACHER_MEM_FRACTION:-0.8}" \
      "${TEACHER_METRICS_ARGS[@]}" \
      > "${TEACHER_LOG}" 2>&1 &
   TEACHER_BG_PID=$!
   echo "Starting teacher server (${TEACHER_MODEL}) on GPUs ${TEACHER_GPUS} (tp=${TEACHER_TP} dp=${TEACHER_DP}) pid=${TEACHER_BG_PID}, log=${TEACHER_LOG}"
}

wait_teacher_ready_local() {
   local waited=0
   while true; do
      if teacher_healthy; then
         echo "[teacher] healthy after ${waited}s (pid=${TEACHER_BG_PID})."
         return 0
      fi
      if ! kill -0 "${TEACHER_BG_PID}" 2>/dev/null; then
         echo "[teacher] sglang pid ${TEACHER_BG_PID} exited before becoming healthy (${waited}s)."
         return 1
      fi
      if [[ "${waited}" -ge "${TEACHER_READY_TIMEOUT_SEC}" ]]; then
         echo "[teacher] still not healthy after ${waited}s (TEACHER_READY_TIMEOUT_SEC)."
         return 1
      fi
      sleep 10
      waited=$(( waited + 10 ))
   done
}

# Idempotent boot with retries; reuses an already-serving teacher on this port.
start_teacher() {
   if teacher_healthy; then
      echo "Teacher already serving on ${TEACHER_IP}:${TEACHER_PORT}; reusing it."
      return 0
   fi
   local attempt
   for (( attempt = 1; attempt <= TEACHER_LAUNCH_RETRIES; attempt++ )); do
      echo "[teacher] boot attempt ${attempt}/${TEACHER_LAUNCH_RETRIES} ..."
      launch_teacher_process
      if wait_teacher_ready_local; then
         return 0
      fi
      echo "[teacher] attempt ${attempt} failed; tail of ${TEACHER_LOG}:"
      tail -n 40 "${TEACHER_LOG}" 2>/dev/null || true
      kill -9 "${TEACHER_BG_PID}" 2>/dev/null || true
      TEACHER_BG_PID=""
      mv "${TEACHER_LOG}" "${TEACHER_LOG}.attempt${attempt}" 2>/dev/null || true
   done
   echo "[teacher] FATAL: teacher did not come up in ${TEACHER_LAUNCH_RETRIES} attempts (logs: ${TEACHER_LOG}.attempt*)." >&2
   exit 1
}

# ============ context budget ============
# 32k (vs the multinode 131k): hard-truncation ceiling for the TP2 trajectory
# budget -- the [T, V/TP] fp32 logits at 32k are ~16GB/rank on the actor AND on
# the teacher's one-shot scoring prefill (see the header notes).
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024*32))}"
# PER-TURN generation cap (-> --rollout-max-response-len). 0 = no per-turn cap.
MAX_GEN_LEN="${MAX_GEN_LEN:-0}"
# Student checkpoints. HF_CHECKPOINT is the ORIGINAL VLM dir (sglang rollout needs
# its image processor); load/save use the torch_dist megatron checkpoint. Pure OPD
# (reverse-KL toward the teacher server via opd_loss_function); no --ref-load.
CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/path/to/Qwen3.5-9B}"
   --load "${LOAD_PATH:-/path/to/Qwen3.5-9B_torch_dist}"
   --save "${SAVE_PATH:-/path/to/Qwen3.5-9B_save_ckpt_scaleswe_topk128_singlenode_verify/}"
   --save-interval 40
)

# Student (Qwen3.5-9B) megatron model args (in-repo, self-contained).
source "${MODEL_ARGS_SH:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}"

# ============ rollout: fully-async, student drives the SWE agent loop ============
NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT:-2}
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"     # prompt/group 为单位
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-1}"
ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path examples.coding_agent_opd.generate.generate
   --buffer-filter-path examples.coding_agent_opd.utils.filters.pop_longest_first
   --prompt-data "${PROMPT_DATA:-${SLIME_DIR}/data/scale_swe_python_sample5000_train_ccr.jsonl}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-4}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --num-steps-per-rollout ${NUM_STEPS_PER_ROLLOUT}
   --log-multi-turn
   --balance-data
)

# REWARD: reward_func POSTs the token sequence to the sglang teacher (--rm-url,
# return_logprob=True); post_process promotes metadata["swe_reward"] and attaches
# the teacher logprobs the OPD loss consumes.
RM_ARGS=(
   --custom-rm-path examples.coding_agent_opd.opd.reward_func
   --custom-reward-post-process-path examples.coding_agent_opd.opd.post_process_rewards
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

EVAL_ARGS=(
)

# TP2 (vs the multinode TP8) leaves the [T, V/TP] fp32 logits 4x wider, so this
# cap is the peak knob: 32k -> ~16GB/rank (the 65k-73k trajectories this box
# actually rolled out OOM'd the actor at 32.41 GiB -- see the header notes).
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((1024*32))}"
LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"
PERF_ARGS=(
   # TP2 x CP1 x DP1 on 2 GPUs. TP2 is fine for TP <= num_query_groups (4).
   --tensor-model-parallel-size "${TP_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE:-1}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-loss-function
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS_PER_GPU}
   --log-probs-chunk-size ${LOG_PROBS_CHUNK_SIZE}
)

# OPD: reverse-KL toward the teacher server's token logprobs. --opd-type sglang
# fetches them via the reward hook (cross-arch teacher OK: shared vocab).
OPD_LOSS_TYPE="${OPD_LOSS_TYPE:-topk_reverse_kl}"
OPD_ID_TOPK="${OPD_ID_TOPK:-128}"
OPD_ARGS=(
   --opd-type sglang
   --loss-type custom_loss
   --custom-loss-function-path slime.backends.megatron_utils.loss.opd_loss_function
   --opd-loss-type "${OPD_LOSS_TYPE}"
   --opd-id-topk "${OPD_ID_TOPK}"
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

# Validation only: wandb left OFF (no credentials / not needed for the smoke).
WANDB_ARGS=(
)

# ============ async in-flight pool sizing ============
# Designed pool = INFLIGHT_STEPS x rollout_batch_size groups x n_samples_per_prompt
# samples; --sglang-server-concurrency is PER ENGINE = ceil(samples / engines).
# 2 engines x 2 GPUs, ports start at 15000 with stride 2 (same convention as the
# multinode script).
ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
INFLIGHT_STEPS="${INFLIGHT_STEPS:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"

NUM_ROLLOUT_ENGINES=$(( ROLLOUT_NUM_GPUS / ROLLOUT_GPUS_PER_ENGINE ))
INFLIGHT_SAMPLES=$(( INFLIGHT_STEPS * ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))
SGLANG_SERVER_CONCURRENCY="${SGLANG_SERVER_CONCURRENCY:-$(( (INFLIGHT_SAMPLES + NUM_ROLLOUT_ENGINES - 1) / NUM_ROLLOUT_ENGINES ))}"
ADMITTED_SAMPLES=$(( SGLANG_SERVER_CONCURRENCY * NUM_ROLLOUT_ENGINES ))
echo "in-flight sizing: ${INFLIGHT_STEPS} steps x ${ROLLOUT_BATCH_SIZE} groups x ${N_SAMPLES_PER_PROMPT} samples = ${INFLIGHT_SAMPLES} samples designed | ${NUM_ROLLOUT_ENGINES} engines -> sglang_server_concurrency=${SGLANG_SERVER_CONCURRENCY} -> admits ${ADMITTED_SAMPLES} samples = RLE container ceiling"

if [[ -z "${METRICS_ENGINE_PORTS:-}" ]]; then
   METRICS_ENGINE_PORTS=""
   for (( i = 0; i < NUM_ROLLOUT_ENGINES; i++ )); do
      METRICS_ENGINE_PORTS+="$(( 15000 + i * ROLLOUT_GPUS_PER_ENGINE )) "
   done
fi
echo "metrics engine ports (poller fallback): ${METRICS_ENGINE_PORTS}"

SGLANG_METRICS_ARGS=()
if [[ "${ENABLE_SGLANG_METRICS:-1}" -eq 1 ]]; then
   SGLANG_METRICS_ARGS=(
      --sglang-enable-metrics
      --sglang-enable-metrics-for-all-schedulers
      --sglang-enable-cache-report
      --sglang-decode-log-interval "${SGLANG_DECODE_LOG_INTERVAL:-100}"
   )
fi
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
   --sglang-mem-fraction-static 0.85
   --tool-call-parser qwen3_coder
   --use-distributed-post
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
   --sglang-attention-backend "${SGLANG_ATTN_BACKEND:-flashinfer}"
   # rollout engines run at TP2 too -> same custom-allreduce CUDA-graph-capture
   # failure as the teacher; disable it there as well (NCCL allreduce fallback).
   --sglang-disable-custom-all-reduce
   "${SGLANG_METRICS_ARGS[@]}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============ optional: dump trajectories + train data ============
build_debug_args

# ============ allocator: 抗变长打包造成的碎片 ============
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ============ teacher (OPD rm_url) HTTP timeouts + concurrency ============
export OPD_TEACHER_TIMEOUT_SEC="${OPD_TEACHER_TIMEOUT_SEC:-7200}"
export OPD_TEACHER_CONNECT_TIMEOUT_SEC="${OPD_TEACHER_CONNECT_TIMEOUT_SEC:-360}"
export OPD_TEACHER_CONCURRENCY="${OPD_TEACHER_CONCURRENCY:-$((TEACHER_DP * 2))}"

# ============ proxy: github fetch via proxy, in-cluster traffic direct ============
setup_proxy

cd "${SLIME_DIR}"

# ============ runtime env propagated to ray workers ============
build_runtime_env_json

# ============ teacher first (it is the rm-url; worst case fail fast) ============
start_teacher

# ============ ray head on the 6 role GPUs (teacher cards held out) ============
preclean_ray
RAY_GPUS="${RAY_GPUS:-0,1,2,3,4,5}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-6}"
CUDA_VISIBLE_DEVICES="${RAY_GPUS}" "${RAY_LAUNCH[@]}" start --head --node-ip-address "${MASTER_ADDR}" \
   --port "${RAY_HEAD_PORT}" --num-gpus "${RAY_NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"
ray status || true

METRICS_POLLER_PID=""
if [[ "${ENABLE_SGLANG_METRICS:-1}" -eq 1 && "${METRICS_POLL_INTERVAL:-120}" -gt 0 ]]; then
   echo "[head] starting sglang metrics poller (interval=${METRICS_POLL_INTERVAL}s) -> ${METRICS_LOG}"
   metrics_poller 2>&1 | tee -a "${METRICS_LOG}" &
   METRICS_POLLER_PID=$!
fi

# ============ submit (driver = train_async.py, disaggregated) ============
SUBMIT_RC=0
ray job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${SLIME_VENV}/bin/python3" -u train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE:-2}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS}" \
   --num-gpus-per-node "${ROLLOUT_NUM_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${OPD_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" \
   "${RM_ARGS[@]}" || SUBMIT_RC=$?

if [[ "${SUBMIT_RC}" -ne 0 ]]; then
   echo "[head] ray job submit failed (rc=${SUBMIT_RC}); dumping engine logs for post-mortem."
   dump_engine_logs
fi

# ============ cleanup ============
[[ -n "${METRICS_POLLER_PID}" ]] && kill "${METRICS_POLLER_PID}" 2>/dev/null || true
pkill -9 sglang || true > /dev/null 2>&1
sleep 3
ray stop --force || true > /dev/null 2>&1
exit "${SUBMIT_RC}"