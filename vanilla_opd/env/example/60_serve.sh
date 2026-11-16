#!/bin/bash
# Phase 3a/3b: bring up Qwen3.5-9B twice under sglang -- once as the STUDENT
# (OpenAI /v1/chat/completions, drives the sweagent loop) and once as the
# TEACHER (/generate with return_logprob, the OPD reverse-KL source). Same
# weights for both roles, which is what the acceptance criterion asks for.
#
# GPU layout on this 8x H800 node: student = 0,1 (tp2), teacher = 2,3 (tp2).
# attention backend = flashinfer: Qwen3.5-9B has head_dim=256 and FA4's CUTE
# varlen kernel asserts on the sglang side for that head_dim.
#
#   bash env/example/60_serve.sh          # start (idempotent) and wait for health
#   bash env/example/60_serve.sh stop     # tear both down
set -eux
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

RUN_DIR=${RUN_DIR:-$SWE_OPD_DIR/env/example/serve}
mkdir -p "$RUN_DIR"

MODEL=${MODEL:-/path/to/Qwen3.5-9B-HF-dir}   # e.g. /path/to/Qwen3.5-9B
STUDENT_PORT=${STUDENT_PORT:-13140}
TEACHER_PORT=${TEACHER_PORT:-13141}
STUDENT_GPUS=${STUDENT_GPUS:-0,1}
TEACHER_GPUS=${TEACHER_GPUS:-2,3}
TP=${TP:-2}
MEM_FRACTION=${MEM_FRACTION:-0.75}
# 131072 is what the sweagent loop actually needs: eval_chat's default
# --max-model-len is 131072 and --max-tokens 0 asks the server for the whole
# remaining window each turn. At --context-length 32768 a mid-trajectory turn
# hits HTTP 400 "Requested token count exceeds the model's maximum context
# length" once the transcript passes ~28k tokens. The weights allow it
# (text_config.max_position_embeddings = 262144) and the KV pool is not the
# constraint: 765096 tokens even at mem-fraction 0.40.
CTX_LEN=${CTX_LEN:-131072}
ATTN_BACKEND=${ATTN_BACKEND:-flashinfer}

if [[ "${1:-start}" == "stop" ]]; then
   pkill -9 -f "sglang.launch_server.*--port ${STUDENT_PORT}" || true
   pkill -9 -f "sglang.launch_server.*--port ${TEACHER_PORT}" || true
   echo "stopped"
   exit 0
fi

launch() {
   local role=$1 gpus=$2 port=$3
   if curl -sf "http://127.0.0.1:${port}/health_generate" >/dev/null 2>&1; then
      echo "${role} already serving on ${port}; reusing."
      return
   fi
   CUDA_VISIBLE_DEVICES="$gpus" python3 -m sglang.launch_server \
      --model-path "$MODEL" \
      --served-model-name "qwen3.5-9b" \
      --host 0.0.0.0 --port "$port" \
      --tp "$TP" \
      --context-length "$CTX_LEN" \
      --attention-backend "$ATTN_BACKEND" \
      --mem-fraction-static "$MEM_FRACTION" \
      --chunked-prefill-size 4096 \
      --trust-remote-code \
      > "$RUN_DIR/${role}.log" 2>&1 &
   echo "launched ${role} pid=$! gpus=${gpus} port=${port} -> $RUN_DIR/${role}.log"
}

wait_health() {
   local role=$1 port=$2 tries=0
   until curl -sf "http://127.0.0.1:${port}/health_generate" >/dev/null; do
      tries=$((tries + 1))
      if (( tries > 180 )); then
         echo "FATAL: ${role} did not come up; last log lines:" >&2
         tail -n 40 "$RUN_DIR/${role}.log" >&2
         return 1
      fi
      sleep 10
   done
   echo "${role} healthy on ${port}"
}

launch student "$STUDENT_GPUS" "$STUDENT_PORT"
launch teacher "$TEACHER_GPUS" "$TEACHER_PORT"
wait_health student "$STUDENT_PORT"
wait_health teacher "$TEACHER_PORT"

curl -s "http://127.0.0.1:${STUDENT_PORT}/get_model_info"; echo
curl -s "http://127.0.0.1:${TEACHER_PORT}/get_model_info"; echo
