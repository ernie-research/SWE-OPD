#!/bin/bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"
SLIME_VENV="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}"
export PATH="${SLIME_VENV}/bin:${PATH}"
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_LM_PATH:-/root/Megatron-LM}${PYTHONPATH:+:${PYTHONPATH}}"

RANK="${PADDLE_TRAINER_ID:?}"
WORLD="${PADDLE_TRAINERS_NUM:?}"
TEACHER_NNODES="${TEACHER_NNODES:-2}"
TEACHER_REPLICAS="${TEACHER_REPLICAS:-2}"
TEACHER_TP="${TEACHER_TP:-16}"
TEACHER_PORT="${TEACHER_PORT:-13141}"
if (( TEACHER_NNODES * TEACHER_REPLICAS != WORLD )); then
   echo "teacher groups ${TEACHER_REPLICAS}x${TEACHER_NNODES} do not cover ${WORLD} nodes" >&2
   exit 1
fi
if (( TEACHER_TP % TEACHER_NNODES != 0 || TEACHER_TP / TEACHER_NNODES != 8 )); then
   echo "this launcher requires eight GPUs/node: TP=${TEACHER_TP}, nnodes=${TEACHER_NNODES}" >&2
   exit 1
fi
IFS=',' read -ra TRAINER_HOSTS <<< "${PADDLE_TRAINERS:?}"
if (( ${#TRAINER_HOSTS[@]} != WORLD )); then
   echo "PADDLE_TRAINERS contains ${#TRAINER_HOSTS[@]} hosts, expected ${WORLD}" >&2
   exit 1
fi

GROUP_ID=$(( RANK / TEACHER_NNODES ))
NODE_RANK=$(( RANK % TEACHER_NNODES ))
LEADER_RANK=$(( GROUP_ID * TEACHER_NNODES ))
LEADER_IP="${TRAINER_HOSTS[LEADER_RANK]}"
DIST_PORT=$(( ${TEACHER_NCCL_PORT_BASE:-23000} + GROUP_ID * 100 ))
RUN_DIR="${RUN_DIR:-${RUN_ROOT:?}/logs-phase2}"
# start-offline-opd propagates RUN_DIR to every multinode rank, but RUN_ROOT is
# only guaranteed in the local launcher environment.  Recover the experiment
# root from the phase-specific log directory on remote ranks so the shared
# group completion/failure markers always resolve to the same location.
if [[ -z "${RUN_ROOT:-}" ]]; then
   case "${RUN_DIR}" in
      */logs-phase2) RUN_ROOT="${RUN_DIR%/logs-phase2}" ;;
      *) echo "cannot recover RUN_ROOT from unexpected RUN_DIR=${RUN_DIR}" >&2; exit 1 ;;
   esac
fi
export RUN_ROOT
mkdir -p "${RUN_DIR}"
TEACHER_LOG="${RUN_DIR}/phase2.teacher.group${GROUP_ID}.node${NODE_RANK}.rank${RANK}.log"
GROUP_DONE="${RUN_ROOT}/.phase2_teacher_group${GROUP_ID}.done"
GROUP_FAILED="${RUN_ROOT}/.phase2_teacher_group${GROUP_ID}.failed"
if (( NODE_RANK == 0 )); then
   rm -f "${GROUP_DONE}" "${GROUP_FAILED}"
fi

TEACHER_PID=""
cleanup() {
   if [[ -n "${TEACHER_PID}" ]] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
      kill -TERM -- "-${TEACHER_PID}" 2>/dev/null || true
      for _ in {1..30}; do
         kill -0 "${TEACHER_PID}" 2>/dev/null || break
         sleep 1
      done
      kill -KILL -- "-${TEACHER_PID}" 2>/dev/null || true
      wait "${TEACHER_PID}" 2>/dev/null || true
   fi
}
trap cleanup EXIT INT TERM

export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-1}"
export SGLANG_OPD_TOPK_OVERLAP=1
export SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS="${SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS:-1}"
export SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1
export SGLANG_LOGITS_PROCESSER_CHUNK_SIZE="${TEACHER_LOGITS_CHUNK:-256}"
export no_proxy="127.0.0.1,localhost,${LEADER_IP},${no_proxy:-}"
export NO_PROXY="${no_proxy}"

TEACHER_ATTENTION_MODE="${TEACHER_ATTENTION_MODE:-dsa}"
TEACHER_MODE_ARGS=()
TEACHER_LOAD_ARGS=()
if [[ -n "${TEACHER_MODEL_LOADER_THREADS:-}" ]]; then
   [[ "${TEACHER_MODEL_LOADER_THREADS}" =~ ^[1-9][0-9]*$ ]] || {
      echo "TEACHER_MODEL_LOADER_THREADS must be a positive integer: ${TEACHER_MODEL_LOADER_THREADS}" >&2
      exit 1
   }
   TEACHER_LOAD_ARGS+=(
      --model-loader-extra-config
      "{\"num_threads\": ${TEACHER_MODEL_LOADER_THREADS}}"
   )
fi
case "${TEACHER_ATTENTION_MODE}" in
   dsa) ;;
   dense)
      # Local SGLang uses the pre-rename NSA environment names.  Newer SGLang
      # exposes the same switch as SGLANG_DSA_* and keeps NSA as an alias.
      export SGLANG_NSA_PREFILL_DENSE_ATTN_KV_LEN_THRESHOLD="${TEACHER_DENSE_ATTN_KV_LEN_THRESHOLD:-131072}"
      export SGLANG_MAX_KV_CHUNK_CAPACITY="${TEACHER_MAX_KV_CHUNK_CAPACITY:-262144}"
      export SGLANG_NSA_SPARSE_PREFILL_Q_CHUNK="${TEACHER_NSA_SPARSE_PREFILL_Q_CHUNK:-8192}"
      export SGLANG_NSA_MQA_LOGITS_MAX_ROWS="${TEACHER_NSA_MQA_LOGITS_MAX_ROWS:-4096}"
      export SGLANG_MOE_TOKEN_CHUNK_SIZE="${TEACHER_MOE_TOKEN_CHUNK_SIZE:-16384}"
      TEACHER_MODE_ARGS+=(
         --kv-cache-dtype "${TEACHER_KV_CACHE_DTYPE:-bf16}"
         --max-running-requests "${TEACHER_MAX_RUNNING_REQUESTS:-1}"
      )
      ;;
   *)
      echo "TEACHER_ATTENTION_MODE must be dsa or dense: ${TEACHER_ATTENTION_MODE}" >&2
      exit 1
      ;;
esac

echo "teacher attention mode=${TEACHER_ATTENTION_MODE}, chunked_prefill=${TEACHER_CHUNKED_PREFILL_SIZE:-16384}, max_prefill=${TEACHER_MAX_PREFILL_TOKENS:-16384}, score_concurrency=${SCORE_CONCURRENCY:-2}, nsa_rows=${TEACHER_NSA_MQA_LOGITS_MAX_ROWS:-upstream}, moe_chunk=${TEACHER_MOE_TOKEN_CHUNK_SIZE:-upstream}"

# The default loader makes every TP rank read checkpoint tensors independently.
# On a shared filesystem this produces highly contended, non-sequential reads.
# These hosts have enough RAM to cache the complete checkpoint, so one ordered
# pass per host turns the subsequent per-rank reads into local page-cache hits.
if [[ "${TEACHER_PREWARM_WEIGHTS:-0}" == 1 ]]; then
   [[ -d "${TEACHER_MODEL}" ]] || {
      echo "cannot prewarm non-local teacher model: ${TEACHER_MODEL}" >&2
      exit 1
   }
   mapfile -d '' TEACHER_WEIGHT_FILES < <(
      find "${TEACHER_MODEL}" -maxdepth 1 -type f -name '*.safetensors' -print0 | sort -z
   )
   (( ${#TEACHER_WEIGHT_FILES[@]} > 0 )) || {
      echo "no safetensors files found for prewarm: ${TEACHER_MODEL}" >&2
      exit 1
   }
   prewarm_concurrency="${TEACHER_PREWARM_CONCURRENCY:-1}"
   [[ "${prewarm_concurrency}" =~ ^[1-9][0-9]*$ ]] || {
      echo "TEACHER_PREWARM_CONCURRENCY must be a positive integer: ${prewarm_concurrency}" >&2
      exit 1
   }
   prewarm_start="$(date +%s)"
   echo "prewarming ${#TEACHER_WEIGHT_FILES[@]} teacher shards into local page cache with ${prewarm_concurrency} readers"
   printf '%s\0' "${TEACHER_WEIGHT_FILES[@]}" | \
      xargs -0 -r -n 1 -P "${prewarm_concurrency}" sh -c 'cat -- "$1" > /dev/null' _
   echo "teacher page-cache prewarm complete in $(( $(date +%s) - prewarm_start ))s"
fi

: > "${TEACHER_LOG}"
setsid env CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
   "${SLIME_VENV}/bin/python3" -m sglang.launch_server \
   --model-path "${TEACHER_MODEL:?}" \
   --host 0.0.0.0 \
   --port "${TEACHER_PORT}" \
   --tp "${TEACHER_TP}" \
   --dp-size 1 \
   --nnodes "${TEACHER_NNODES}" \
   --node-rank "${NODE_RANK}" \
   --dist-init-addr "${LEADER_IP}:${DIST_PORT}" \
   --trust-remote-code \
   --disable-cuda-graph \
   --chunked-prefill-size "${TEACHER_CHUNKED_PREFILL_SIZE:-16384}" \
   --disable-radix-cache \
   --max-prefill-tokens "${TEACHER_MAX_PREFILL_TOKENS:-16384}" \
   --watchdog-timeout "${TEACHER_WATCHDOG_TIMEOUT:-7200}" \
   --mem-fraction-static "${TEACHER_MEM_FRACTION:-0.77}" \
   "${TEACHER_LOAD_ARGS[@]}" \
   "${TEACHER_MODE_ARGS[@]}" \
   > "${TEACHER_LOG}" 2>&1 &
TEACHER_PID=$!

if (( NODE_RANK == 0 )); then
   waited=0
   timeout="${TEACHER_WAIT_TIMEOUT_SEC:-7200}"
   until curl --noproxy '*' -sf "http://127.0.0.1:${TEACHER_PORT}/health_generate" >/dev/null; do
      if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
         printf 'teacher exited before health: rank=%s\n' "${RANK}" > "${GROUP_FAILED}"
         tail -n 200 "${TEACHER_LOG}" >&2 || true
         exit 1
      fi
      if (( waited >= timeout )); then
         printf 'teacher health timeout=%s rank=%s\n' "${timeout}" "${RANK}" > "${GROUP_FAILED}"
         tail -n 200 "${TEACHER_LOG}" >&2 || true
         exit 1
      fi
      sleep 5
      waited=$(( waited + 5 ))
   done
   echo "teacher group ${GROUP_ID} healthy after ${waited}s"
   set +e
   SKIP_SERVE=1 TEACHER_IP=127.0.0.1 TEACHER_PORT="${TEACHER_PORT}" \
      SCORE_SHARD_INDEX="${GROUP_ID}" SCORE_NUM_SHARDS="${TEACHER_REPLICAS}" \
      bash "${OFFLINE_OPD_DIR}/run_phase2.sh"
   rc=$?
   set -e
   if (( rc == 0 )); then
      printf 'completed=%s leader=%s\n' "$(date -Is)" "${RANK}" > "${GROUP_DONE}"
   else
      printf 'rc=%s failed=%s leader=%s\n' "${rc}" "$(date -Is)" "${RANK}" > "${GROUP_FAILED}"
      exit "${rc}"
   fi
else
   timeout="${TEACHER_PHASE2_WAIT_TIMEOUT_SEC:-172800}"
   waited=0
   until [[ -f "${GROUP_DONE}" ]]; do
      if [[ -f "${GROUP_FAILED}" ]]; then
         cat "${GROUP_FAILED}" >&2
         exit 1
      fi
      if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
         echo "secondary teacher process exited: group=${GROUP_ID}, node=${NODE_RANK}" >&2
         tail -n 200 "${TEACHER_LOG}" >&2 || true
         exit 1
      fi
      if (( waited >= timeout )); then
         echo "timed out waiting for teacher group ${GROUP_ID} scorer" >&2
         exit 1
      fi
      sleep 10
      waited=$(( waited + 10 ))
   done
fi
