#!/bin/bash
set -euo pipefail
set -x
# OFFLINE OPD PHASE 2 teacher. Defaults match the currently validated H800 online setup.

TEACHER_PORT="${TEACHER_PORT:-13141}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/SFT/model/Qwen3.6-27B}"
TEACHER_TP="${TEACHER_TP:-4}"
TEACHER_DP="${TEACHER_DP:-2}"
TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3,4,5,6,7}"
TEACHER_MEM_FRACTION="${TEACHER_MEM_FRACTION:-0.65}"
TEACHER_ATTN_BACKEND="${TEACHER_ATTN_BACKEND:-flashinfer}"
TEACHER_LOGITS_CHUNK="${TEACHER_LOGITS_CHUNK:-2048}"
TEACHER_NCCL_PORT_BASE="${TEACHER_NCCL_PORT_BASE:-8000}"
TEACHER_MAX_PREFILL_TOKENS="${TEACHER_MAX_PREFILL_TOKENS:-$((1024*256))}"
TEACHER_WATCHDOG_TIMEOUT="${TEACHER_WATCHDOG_TIMEOUT:-300}"

IFS=',' read -ra _teacher_gpu_array <<< "${TEACHER_GPUS}"
if (( TEACHER_TP * TEACHER_DP != ${#_teacher_gpu_array[@]} )); then
   echo "teacher requires TP*DP GPUs: ${TEACHER_TP}*${TEACHER_DP} != ${#_teacher_gpu_array[@]} from TEACHER_GPUS=${TEACHER_GPUS}" >&2
   exit 1
fi
[[ -d "${TEACHER_MODEL}" ]] || { echo "TEACHER_MODEL not found: ${TEACHER_MODEL}" >&2; exit 1; }

export FLASHINFER_USE_CUDA_NORM="${FLASHINFER_USE_CUDA_NORM:-1}"
echo "PHASE2 teacher: model=${TEACHER_MODEL} port=${TEACHER_PORT} tp=${TEACHER_TP} dp=${TEACHER_DP} gpus=${TEACHER_GPUS}"

SGLANG_OPD_TOPK_OVERLAP="${SGLANG_OPD_TOPK_OVERLAP:-1}" \
SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS="${SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS:-0}" \
SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK="${SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK:-1}" \
SGLANG_LOGITS_PROCESSER_CHUNK_SIZE="${TEACHER_LOGITS_CHUNK}" \
CUDA_VISIBLE_DEVICES="${TEACHER_GPUS}" exec python3 -m sglang.launch_server \
   --model-path "${TEACHER_MODEL}" \
   --host 0.0.0.0 \
   --port "${TEACHER_PORT}" \
   --tp "${TEACHER_TP}" \
   --dp-size "${TEACHER_DP}" \
   --nccl-port "${TEACHER_NCCL_PORT_BASE}" \
   --trust-remote-code \
   --tool-call-parser qwen3_coder \
   --attention-backend "${TEACHER_ATTN_BACKEND}" \
   --disable-cuda-graph \
   --chunked-prefill-size -1 \
   --disable-radix-cache \
   --max-prefill-tokens "${TEACHER_MAX_PREFILL_TOKENS}" \
   --watchdog-timeout "${TEACHER_WATCHDOG_TIMEOUT}" \
   --mem-fraction-static "${TEACHER_MEM_FRACTION}"
