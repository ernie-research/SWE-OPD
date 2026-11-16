#!/bin/bash
# Chat-completions SWE eval for coding_agent_opd (plan S10).
#
#   bash examples/coding_agent_opd/eval/run_eval.sh
#
# Scores an external OpenAI-compatible /v1/chat/completions endpoint on a SWE
# dataset by driving the SAME sweagent tool loop + RLE grading the training
# rollout uses (examples/coding_agent_opd/generate.py + r2egym.*): identical
# scaffold prompts, budget, stepcount, action parser and observation formatting.
#
# Transport: we send `messages` and let the SERVER apply its own chat template
# with its own tokenizer -- like R2E-Gym's own eval. (Token-in/TITO is only
# byte-identical when the eval tokenizer matches the served model; this endpoint
# is an external model, Qwen3.5-9B, whose tokenizer we do not hold, so local
# token-in produced garbage. Everything above the transport is still the shared
# harness, so the interaction protocol is identical.)
#
# Smoke test (1 instance, tiny budget -- the endpoint is a slow reasoning model):
#   LIMIT=1 MAX_STEPS_ABS=8 MAX_TOKENS=4096 CONCURRENT=1 LOG_TURNS=1 \
#     bash examples/coding_agent_opd/eval/run_eval.sh
#
# Full run mirrors the upstream R2E-Gym full-eval budget (max_steps_absolute 300).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
cd "${SLIME_DIR}"

MEGATRON_LM="${MEGATRON_LM:-$(cd "${SLIME_DIR}/../Megatron-LM" 2>/dev/null && pwd || echo /root/Megatron-LM)}"

# Activate the training venv if not already inside one (generate.py imports the
# full slime rollout stack: sglang_router / megatron / torch). The venv dir is
# ".slime_venv" on these nodes; "slime-venv" is honored for other environments.
for _VENV_DIR in ".slime_venv" "slime-venv"; do
    if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "${SLIME_DIR}/${_VENV_DIR}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${SLIME_DIR}/${_VENV_DIR}/bin/activate"
        break
    fi
done

# cuDNN: match the venv's bundled cudnn (see smoke-test.sh for the rationale).
if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib" ]; then
    export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python3.12/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
fi

# ============ model endpoint (OpenAI-compatible /v1/chat/completions) ============
# We send `messages`; the server applies its own chat template + tokenizer. No
# local tokenizer needed (unlike the old TITO transport). The defaults are
# placeholders -- ALWAYS set LLM_URL / MODEL_NAME (see QUICK_EVAL_GUIDE.md).
LLM_URL="${LLM_URL:-http://<MODEL_HOST>:<PORT>/v1/chat/completions}"
MODEL_NAME="${MODEL_NAME:-<MODEL_NAME>}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
# extra non-top-level sampling params (-> request body), like the upstream
# R2E-Gym eval's EXTRA_SAMPLING_PARAMS. Defaults mirror it (top_k/min_p/penalties).
# NOTE: keep the default in its own var -- inlining a brace-containing JSON as the
# ${VAR:-default} fallback makes bash stop at the JSON's first "}" and append the
# expansion's closing "}" as a literal, corrupting the JSON with a trailing "}".
_DEFAULT_EXTRA_SAMPLING='{"top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}'
EXTRA_SAMPLING="${EXTRA_SAMPLING:-${_DEFAULT_EXTRA_SAMPLING}}"

# ============ loop / token budget (defaults follow the upstream R2E-Gym full eval) ============
MAX_STEPS_ABS="${MAX_STEPS_ABS:-300}"
MAX_TOKENS="${MAX_TOKENS:-0}"   # 0 = no per-turn cap: the server grants each turn
                                # context_len - input_len - 2 (see eval_chat.py)
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-1200}"
RETRY="${RETRY:-3}"
# Per-trajectory wall-clock ceiling (seconds): the chat loop aborts an instance
# once its total elapsed exceeds this, capping the worst-case duration a single
# instance can burn even under a retry storm. 0 disables. (eval_chat --max-total-time)
MAX_TOTAL_TIME="${MAX_TOTAL_TIME:-5400}"
CONCURRENT="${CONCURRENT:-16}"
LIMIT="${LIMIT:-0}"   # 0 = all 500 rows

# ============ observability ============
# LOG_TURNS=1 logs each turn's think / tool calls / observation (like
# visualize_rollouts.py); LOG_MAX_CHARS truncates each field; HEARTBEAT is the
# seconds between aggregate progress logs (0 disables). No detokenize needed --
# the chat transport already returns turns as text.
LOG_TURNS="${LOG_TURNS:-0}"
LOG_MAX_CHARS="${LOG_MAX_CHARS:-400}"
HEARTBEAT="${HEARTBEAT:-60}"

# DUMP_DIR: if non-empty, dump the full self-contained replayable trajectory
# (run context + env repro fields + complete messages transcript) to
# ${DUMP_DIR}/<instance_id>.json, one file per instance. Empty (default) = off.
#   DUMP_DIR="${OUT_DIR}/dump" bash examples/coding_agent_opd/eval/run_eval.sh
DUMP_DIR="${DUMP_DIR:-}"

# RESUME=1: reuse an EXISTING run dir. Point OUT_DIR at a finished run and set
# RESUME=1; instances already recorded in results.jsonl WITHOUT an error are kept
# and skipped, only the failed/never-run ones are re-attempted (补跑). Combine with
# a lower CONCURRENT to clear the timeout/pod_unavailable failures. Example:
#   OUT_DIR=examples/coding_agent_opd/logs/eval_20260718_210413 RESUME=1 CONCURRENT=8 \
#     bash examples/coding_agent_opd/eval/run_eval.sh
RESUME="${RESUME:-0}"

# ============ benchmark + data + output ============
# BENCHMARK picks the dataset (and the host-side pre-flight checks) from
# eval/benchmarks.json -- swebench | swebench-pro | multilingual. DATA, when
# given explicitly, wins over the registry-resolved file.
BENCHMARK="${BENCHMARK:-swebench}"
if [ -z "${DATA:-}" ]; then
    if ! DATA="$(python3 -m examples.coding_agent_opd.eval.benchmarks "${BENCHMARK}")"; then
        echo "error: cannot resolve benchmark '${BENCHMARK}' via eval/benchmarks.py" >&2
        echo "       (available names are listed in eval/benchmarks.json)" >&2
        exit 1
    fi
fi
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/../logs/eval_${BENCHMARK}_$(TZ='Asia/Shanghai' date +%Y%m%d_%H%M%S)}"

# ============ RLE sandbox (grading pod) ============
# swebench-latest rows carry a full ccr/-pub.cnc docker_image, so no
# docker_image_prefix is needed (docker.py leaves ccr images untouched and swaps
# -pub.cnc -> -vpc.cnc for the RLE backend). RLE_TOOL_TYPE defaults to swe-pdc
# (r2egym __init__), matching the upstream R2E-Gym full eval which sets neither.
export SLIME_AGENT_SANDBOX_BACKEND="${SLIME_AGENT_SANDBOX_BACKEND:-rle}"
export RLE_ENV="${RLE_ENV:-prod}"
export RLE_API_KEY="${RLE_API_KEY:?set RLE_API_KEY}"
export RLE_SESSION_TIMEOUT="${RLE_SESSION_TIMEOUT:-180}"
export RLE_SESSION_IDS_DIR="${RLE_SESSION_IDS_DIR:-${OUT_DIR}/rle_sessions}"

# Boot / tool / grading timeouts consumed by utils/rollout.py's _boot_env + env.
export SWE_BOOT_CONCURRENCY="${SWE_BOOT_CONCURRENCY:-${CONCURRENT}}"
export SLIME_AGENT_TOOL_TIMEOUT_SEC="${SLIME_AGENT_TOOL_TIMEOUT_SEC:-180}"
export SWE_EVAL_TIMEOUT_SEC="${SWE_EVAL_TIMEOUT_SEC:-600}"

# Never route the in-cluster endpoint / RLE API through a proxy. The model
# endpoint's host is derived from LLM_URL above so it always tracks the
# configured endpoint (no hard-coded IP); the internal domain suffixes below
# keep the RLE API direct; append any other internal hosts to the export.
_LLM_HOST="${LLM_URL#http://}"
_LLM_HOST="${_LLM_HOST%%:*}"
export no_proxy="127.0.0.1,localhost,${_LLM_HOST},.baidu.com,.baidu-int.com,.baidubce.com,${no_proxy:-}"
export NO_PROXY="${no_proxy}"

# ============ outbound proxy for GitHub (host-side only) ============
# swebench's test-spec build (get_requirements_by_commit) does a TIMEOUT-LESS
# requests.get to raw.githubusercontent.com to fetch each repo's requirements
# (e.g. django IS in MAP_REPO_TO_REQS_PATHS -> tests/requirements/py3.txt;
# astropy is NOT, so it never fetches). This cluster cannot reach GitHub
# directly, so django-family instances hang FOREVER inside RepoEnv setup_env
# during boot -- before the chat loop, so none of our step/model/reward
# timeouts cover it. Route host-side python requests (requests.trust_env=True)
# through the web proxy. Safe because: the model endpoint uses aiohttp with
# trust_env=False (stays direct), and RLE / internal hosts are in no_proxy
# above -- so ONLY external fetches (GitHub) traverse the proxy.
# Override with a different proxy via:  GITHUB_PROXY=http://host:port bash run_eval.sh
GITHUB_PROXY="${GITHUB_PROXY:-http://agent.baidu.com:8188}"
export HTTPS_PROXY="${GITHUB_PROXY}"
export HTTP_PROXY="${GITHUB_PROXY}"
export https_proxy="${GITHUB_PROXY}"
export http_proxy="${GITHUB_PROXY}"

export PYTHONPATH="${MEGATRON_LM}:${SLIME_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${OUT_DIR}"

echo "=============================================="
echo " benchmark    : ${BENCHMARK}"
echo " data         : ${DATA}"
echo " out_dir      : ${OUT_DIR}"
echo " model        : ${MODEL_NAME}"
echo " url          : ${LLM_URL}"
echo " sampling     : temperature=${TEMPERATURE} top_p=${TOP_P} extra=${EXTRA_SAMPLING:-none}"
echo " budget       : max_steps_absolute=${MAX_STEPS_ABS} max_tokens=${MAX_TOKENS} max_model_len=${MAX_MODEL_LEN}"
echo " timeouts     : llm_req=${REQUEST_TIMEOUT}s retry=${RETRY} max_total_time=${MAX_TOTAL_TIME}s | exec=${SLIME_AGENT_TOOL_TIMEOUT_SEC}s reward=${SWE_EVAL_TIMEOUT_SEC}s boot_retry=${SWE_BOOT_RETRIES:-2} rle_session=${RLE_SESSION_TIMEOUT}min"
echo " concurrent   : ${CONCURRENT}  limit=${LIMIT}"
echo " backend      : ${SLIME_AGENT_SANDBOX_BACKEND} (RLE_ENV=${RLE_ENV})"
echo " observ.      : log_turns=${LOG_TURNS} log_max_chars=${LOG_MAX_CHARS} heartbeat=${HEARTBEAT}s"
echo " dump         : ${DUMP_DIR:-off}"
echo " resume       : ${RESUME}  (1 = skip已成功, 只补跑失败/未跑)"
echo "=============================================="

EXTRA_ARGS=()
[ -n "${EXTRA_SAMPLING}" ] && EXTRA_ARGS+=(--extra-sampling "${EXTRA_SAMPLING}")
[ "${SAVE_TRAJ:-0}" = "1" ] && EXTRA_ARGS+=(--save-traj)
[ -n "${DUMP_DIR}" ] && EXTRA_ARGS+=(--dump-dir "${DUMP_DIR}")
[ "${RESUME}" = "1" ] && EXTRA_ARGS+=(--resume)
[ "${LOG_TURNS}" = "1" ] && EXTRA_ARGS+=(--log-turns)

python3 -u -m examples.coding_agent_opd.eval.eval_chat \
    --data "${DATA}" \
    --benchmark "${BENCHMARK}" \
    --out-dir "${OUT_DIR}" \
    --url "${LLM_URL}" \
    --model "${MODEL_NAME}" \
    --temperature "${TEMPERATURE}" \
    --top-p "${TOP_P}" \
    --max-tokens "${MAX_TOKENS}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-steps-absolute "${MAX_STEPS_ABS}" \
    --request-timeout "${REQUEST_TIMEOUT}" \
    --retry "${RETRY}" \
    --max-total-time "${MAX_TOTAL_TIME}" \
    --concurrent "${CONCURRENT}" \
    --limit "${LIMIT}" \
    --log-max-chars "${LOG_MAX_CHARS}" \
    --heartbeat-interval "${HEARTBEAT}" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "${OUT_DIR}/run.log"

echo "=============================================="
echo " eval done. summary -> ${OUT_DIR}/summary.json"
echo "=============================================="
