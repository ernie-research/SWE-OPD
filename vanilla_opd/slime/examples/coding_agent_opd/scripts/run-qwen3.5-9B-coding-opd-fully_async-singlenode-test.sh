#!/bin/bash
# Model-in-the-loop SWE rollout + On-Policy Distillation (OPD), SINGLE NODE,
# FULLY-ASYNC / DISAGGREGATE. Student = Qwen3.5-9B, Teacher = Qwen3.6-27B.
#
#   bash examples/coding_agent_opd/scripts/run-qwen3.5-9B-coding-opd-fully_async-singlenode-test.sh
#
# Based on run-qwen3-8B-coding-opd-fully_async.sh. The student (Qwen3.5-9B) drives
# the sweagent 3-tool loop; reward = SWE test pass/fail + reverse-KL toward the
# teacher. Two knobs put it on slime's fully-async path (see examples/fully_async):
#   1. driver = train_async.py (overlaps rollout(N+1) with training(N), 1-step
#      off-policy; asserts `not args.colocate` so roles MUST be disaggregated).
#   2. --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
#
# ---- WHY --opd-type sglang (NOT megatron) ----
# The teacher (Qwen3.6-27B: hidden 5120 / 64 layers / 24 heads) is a DIFFERENT
# architecture from the student (Qwen3.5-9B: hidden 4096 / 32 layers / 16 heads).
# --opd-type megatron loads the teacher checkpoint INTO the student's own megatron
# model definition (actor.load_other_checkpoint -> load_checkpoint(self.model,...)),
# so it can only take a SAME-ARCH teacher. A 27B checkpoint cannot load into a 9B
# model. Instead we serve the teacher as a SEPARATE sglang server and fetch its
# token logprobs over HTTP (--opd-type sglang, --rm-url). This only needs a
# SHARED TOKENIZER/VOCAB, which holds here: both are vocab_size 248320, eos 248044.
#
# ---- single-node GPU layout (8 GPUs) ----
#   * teacher  Qwen3.6-27B sglang server, tp=2  -> GPUs 6,7 (outside ray)
#   * student-actor   megatron TP2              -> 2 GPUs (ray)
#   * student-rollout sglang, 4 engines         -> 4 GPUs (ray)
# ray only sees the 6 role GPUs (0-5); the teacher GPUs are held out via
# CUDA_VISIBLE_DEVICES so ray never schedules a role onto the teacher.
#
# Prerequisites:
#   * Student Qwen3.5-9B HF (VLM dir, sglang needs its image processor) + a
#     torch_dist megatron checkpoint for --ref-load/--load.
#   * Teacher Qwen3.6-27B HF dir servable by sglang (NO megatron conversion --
#     the sglang teacher loads the HF checkpoint directly).
#   * RLE credentials + an SWE dataset (metadata.image / workdir / grading fields).

set -ex

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"

# ============ python env: run EVERYTHING under slime-venv ============
# ray, sglang (teacher + rollout) and train_async.py must all use the venv
# interpreter. Prepending the venv's bin to PATH makes `python3`, `ray` and the
# `python3 -m sglang...` calls resolve to the venv, and ray workers spawned by
# `ray start` inherit this same interpreter (so the rollout picks up the venv's
# R2E-Gym deps rather than the system python3).
SLIME_VENV="${SLIME_VENV:-${SLIME_DIR}/../slime-venv}"
if [[ ! -x "${SLIME_VENV}/bin/python3" ]]; then
   echo "slime-venv python not found at ${SLIME_VENV}/bin/python3" >&2
   exit 1
fi
export VIRTUAL_ENV="${SLIME_VENV}"
export PATH="${SLIME_VENV}/bin:${PATH}"
echo "Using python: $(command -v python3)  ray: $(command -v ray)"

# ============ per-run output dir ============
EXP_NAME="${EXP_NAME:-$(basename "${BASH_SOURCE[0]}" .sh)}"
RUN_DIR="${RUN_DIR:-${SCRIPT_DIR}/../logs/${EXP_NAME}_$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S)}"
mkdir -p "${RUN_DIR}"

exec > >(tee -a "${RUN_DIR}/run.log" \
             >(grep --line-buffered -v 'SGLangEngine' >> "${RUN_DIR}/train.log")) 2>&1
echo "RUN_DIR=${RUN_DIR}"

export PYTHONUNBUFFERED=1

# ============ cudnn load-order workaround ============
# transformer_engine dlopens libcudnn_*.so.9 in glob order and can hit
# libcudnn_engines_precompiled.so.9 before libcudnn_graph.so.9 -> undefined symbol.
# Pointing CUDNN_PATH at the wheel's cudnn dir makes TE load the dispatcher, which
# pulls the sublibs in the right DT_NEEDED order.
if [[ -z "${CUDNN_PATH:-}" ]]; then
   CUDNN_PATH="$(python3 -c 'import sysconfig,os;print(os.path.join(sysconfig.get_path("purelib"),"nvidia/cudnn"))' 2>/dev/null)"
   [[ -d "${CUDNN_PATH}" ]] && export CUDNN_PATH || unset CUDNN_PATH
fi

# ============ teacher (external sglang server, Qwen3.6-27B) ============
# 27B in bf16 is ~54GB of weights -> needs tp>=2 to leave room for the kv-cache.
# Served straight from the HF dir (no megatron conversion). Held out from ray via
# CUDA_VISIBLE_DEVICES so its GPUs never receive a student role.
TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/Qwen3.6-27B}"
TEACHER_GPUS="${TEACHER_GPUS:-6,7}"
TEACHER_TP="${TEACHER_TP:-2}"
LOG_FILE="${RUN_DIR}/sglang_teacher.log"

# Idempotent: if a teacher is already serving on this port (e.g. a prior run left
# it up), reuse it instead of relaunching the slow-loading 27B. This lets the
# whole script be re-run to recover from a student-side crash without paying the
# teacher's multi-minute load again.
if curl -sf http://${TEACHER_IP}:${TEACHER_PORT}/health_generate > /dev/null 2>&1; then
    echo "Teacher already serving on ${TEACHER_IP}:${TEACHER_PORT}; reusing it."
else
    CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" python3 -m sglang.launch_server \
        --model-path "${TEACHER_MODEL}" \
        --host 0.0.0.0 \
        --port ${TEACHER_PORT} \
        --tp ${TEACHER_TP} \
        --trust-remote-code \
        --attention-backend "${TEACHER_ATTN_BACKEND:-flashinfer}" \
        --chunked-prefill-size 4096 \
        --mem-fraction-static "${TEACHER_MEM_FRACTION:-0.8}" \
        > "${LOG_FILE}" 2>&1 &

    echo "Starting teacher server (${TEACHER_MODEL}) on GPUs ${TEACHER_GPUS} (tp=${TEACHER_TP})..."
    until curl -sf http://${TEACHER_IP}:${TEACHER_PORT}/health_generate > /dev/null; do
        echo "Waiting for teacher server..."
        tail -n 5 "${LOG_FILE}" || true
        sleep 5
    done
fi
curl -s http://${TEACHER_IP}:${TEACHER_PORT}/get_model_info || true
echo "Teacher server is up at ${TEACHER_IP}:${TEACHER_PORT}."
sleep 5

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

# Student (Qwen3.5-9B) megatron model args (in-repo, self-contained).
source "${MODEL_ARGS_SH:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}"

# ============ context budget ============
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-32768}"
MAX_GEN_LEN="${MAX_GEN_LEN:-0}"

# Student checkpoints. HF_CHECKPOINT is the ORIGINAL VLM dir (sglang rollout needs
# its image processor); ref/load/save use the torch_dist megatron checkpoint.
REF_MODEL_PATH="${REF_MODEL_PATH:-/path/to/Qwen3.5-9B_torch_dist}"
CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/path/to/Qwen3.5-9B}"
   --ref-load "${REF_MODEL_PATH}"
   --load "${LOAD_PATH:-/path/to/Qwen3.5-9B_slime/}"
   --save "${SAVE_PATH:-/path/to/Qwen3.5-9B_slime/}"
   --save-interval 20
)
# __PLACEHOLDER__

# ROLLOUT: student drives the sweagent tool loop; fully-async rollout function.
ROLLOUT_ARGS=(
   --rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async
   --custom-generate-function-path examples.coding_agent_opd.generate.generate
   --prompt-data "${PROMPT_DATA:-${SLIME_DIR}/data/swe_train.jsonl}"
   --input-key prompt
   --label-key label
   --metadata-key metadata
   --rollout-shuffle
   --num-rollout "${NUM_ROLLOUT:-100}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-8}"
   --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-context-len ${MAX_CONTEXT_LEN}
   --rollout-max-response-len ${MAX_GEN_LEN}
   --rollout-temperature 1.0
   --num-steps-per-rollout 1
   --log-multi-turn
   --global-batch-size "${GLOBAL_BATCH_SIZE:-64}"
   --balance-data
)

# REWARD: reward_func POSTs the full token sequence to the sglang teacher
# (--rm-url, return_logprob=True); coding_agent_opd's post-process promotes
# metadata["swe_reward"] to the scalar reward and stores teacher logprobs on the
# sample for the reverse-KL penalty.
RM_ARGS=(
   --custom-rm-path examples.coding_agent_opd.opd.reward_func
   --custom-reward-post-process-path examples.coding_agent_opd.opd.post_process_rewards
   --rm-url http://${TEACHER_IP}:${TEACHER_PORT}/generate
)

# fully-async rollout raises on eval, so no --eval-interval here.
EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size "${CP_SIZE:-1}"
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_CONTEXT_LEN}
)

# OPD: reverse-KL toward the teacher server's token logprobs. --opd-type sglang
# fetches them via the reward hook above (cross-arch teacher OK: shared vocab).
GRPO_ARGS=(
   --advantage-estimator grpo
   --use-opd-adjust-advantage
   --opd-type sglang
   --opd-kl-coef "${OPD_KL_COEF:-1.0}"
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   #--use-wandb
   #--wandb-dir "${RUN_DIR}/wandb"
)
# __PLACEHOLDER2__

# ============ async in-flight pool sizing ============
# The background worker keeps  sglang_server_concurrency * num_engines  GROUPS in
# flight. SPEED-UP CONFIG: one big engine at gen_tp=4 spanning all 4 rollout GPUs
# (num_engines = rollout_num_gpus / rollout_num_gpus_per_engine = 4/4 = 1), so a
# single decode process serves every group -> no cache_aware single-hotspot skew
# (the group-dispatch-balancing TODO is moot with 1 engine). mem-fraction 0.9 gives
# the engine a big kv-cache; concurrency 64 keeps 64 groups in flight to saturate
# decode against the IO-bound agentic loop. Each group = n_samples_per_prompt live
# RLE pods (boots throttled by SWE_BOOT_CONCURRENCY).
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-mem-fraction-static 0.9
   --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY:-64}"
   # Qwen3.5/3.6 are hybrid-GDN (mamba + full-attention) models with head_dim=256.
   # On Blackwell (SM100) the supported full-attention backends are
   # triton/trtllm_mha/fa4/flashinfer, BUT fa4's CUTE varlen kernel asserts out on
   # head_dim=256 (SM100 fwd does not support seqused_q/seqused_k), so we use
   # flashinfer. Override with SGLANG_ATTN_BACKEND=triton if flashinfer misbehaves.
   --sglang-attention-backend "${SGLANG_ATTN_BACKEND:-flashinfer}"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# ============ SWE + RLE sandbox knobs ============
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-rle}"
# RLE_ENV / RLE_TOOL_TYPE / RLE_API_KEY come from an untracked secret file, never
# from this script. Anything already exported in the env wins over the file.
RLE_SECRET_FILE="${RLE_SECRET_FILE:-${SLIME_DIR}/.secret/.key.sh}"
source "${RLE_SECRET_FILE}" || { echo "FATAL: RLE secret file not found: ${RLE_SECRET_FILE}" >&2; exit 1; }
export RLE_SESSION_IDS_DIR="${RLE_SESSION_IDS_DIR:-${RUN_DIR}/rle_sessions}"

export SWE_TRAIN_PROTOCOL="${SWE_TRAIN_PROTOCOL:-scaleswe}"
export SWE_EVAL_PROTOCOL="${SWE_EVAL_PROTOCOL:-scaleswe}"
export SLIME_AGENT_MAX_STEPS="${SLIME_AGENT_MAX_STEPS:-50}"
export SLIME_AGENT_TOOL_TIMEOUT_SEC="${SLIME_AGENT_TOOL_TIMEOUT_SEC:-180}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-16}"

# ============ proxy: github fetch via proxy, in-cluster traffic direct ============
# swebench's test-spec builder (get_requirements_by_commit) does an UNTIMED
# requests.get to raw.githubusercontent.com while creating the SWE grading spec.
# github raw is NOT directly reachable from this cluster, so that call hangs
# forever on connect and freezes the whole rollout (all r2egym-runtime threads
# stuck in create_connection, 0% GPU util). Route http(s) through the
# github-reaching proxy (verified: proxy.example.com:8188 reaches github raw) and
# bypass it for everything in-cluster: teacher (127.0.0.1), the sglang rollout
# engines (this node's IP) and the RLE prod hosts (.baidu*.com + prod gateway
# IPs). Mirrors the sibling package's eval/run_eval.sh.
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
NODE_IP="$(hostname -i 2>/dev/null | awk '{print $1}')"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR},${TEACHER_IP},${NODE_IP},10.0.0.18,10.0.0.19,.example.com,.internal.example.com,.baidubce.com,${no_proxy:-}"
export NO_PROXY="${no_proxy}"
GITHUB_PROXY="${GITHUB_PROXY:-http://proxy.example.com:8188}"
export HTTPS_PROXY="${GITHUB_PROXY}"
export HTTP_PROXY="${GITHUB_PROXY}"
export https_proxy="${GITHUB_PROXY}"
export http_proxy="${GITHUB_PROXY}"

cd "${SLIME_DIR}"

# ============ ray head (student roles only, teacher GPUs held out) ============
# The student actor (2 GPUs) + rollout (4 GPUs) = 6 GPUs. We PIN ray to the 6
# non-teacher GPUs via CUDA_VISIBLE_DEVICES so it never schedules a student role
# onto the teacher's cards. ray keeps its sockets under /tmp/ray (short path --
# avoids the AF_UNIX 107-byte limit that kills the dashboard).
RAY_GPUS="${RAY_GPUS:-0,1,2,3,4,5}"
RAY_NUM_GPUS="${RAY_NUM_GPUS:-6}"
# Clear any stale ray head from a prior (crashed) launch before starting a new
# one. This does NOT touch the teacher, which runs as a standalone sglang process
# outside ray, so a re-run reuses the already-up teacher.
ray stop --force > /dev/null 2>&1 || true
sleep 2
CUDA_VISIBLE_DEVICES="${RAY_GPUS}" ray start --head --node-ip-address ${MASTER_ADDR} \
   --num-gpus "${RAY_NUM_GPUS}" \
   --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

# ============ runtime env propagated to ray workers ============
export SLIME_DIR
RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = (
    "no_proxy", "NO_PROXY",
    "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
    "SLIME_AGENT_SANDBOX_BACKEND",
    "RLE_ENV", "RLE_TOOL_TYPE", "RLE_API_KEY", "RLE_SESSION_IDS_DIR",
    "SLIME_AGENT_RLE_IMAGE_PREFIX",
    "SWE_TRAIN_PROTOCOL", "SWE_EVAL_PROTOCOL",
    "SLIME_AGENT_MAX_STEPS", "SLIME_AGENT_TOOL_TIMEOUT_SEC",
    "SWE_EVAL_TIMEOUT_SEC", "SWE_BOOT_CONCURRENCY",
    "CUDNN_PATH",
)
env = {k: os.environ[k] for k in keys if k in os.environ}
env["PYTHONPATH"] = f"/root/Megatron-LM/:{os.environ['SLIME_DIR']}"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
# Pin ray workers to the slime-venv interpreter (PATH prepended at script top).
env["PATH"] = os.environ["PATH"]
env["VIRTUAL_ENV"] = os.environ["VIRTUAL_ENV"]
print(json.dumps({"env_vars": env}))
PY
)

# ============ submit (driver = train_async.py, disaggregated) ============
# train_async.py asserts `not args.colocate`: actor (2 GPUs, megatron TP2) and
# rollout (4 GPUs, sglang) are disaggregated. It overlaps rollout(N+1) with
# training(N) and syncs student weights to the rollout engines every
# --update-weights-interval steps.
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- "${SLIME_VENV}/bin/python3" -u train_async.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE:-2}" \
   --rollout-num-gpus "${ROLLOUT_NUM_GPUS:-4}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${GRPO_ARGS[@]}" \
   "${WANDB_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${EVAL_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${RM_ARGS[@]}"

# ============ cleanup after training ============
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
