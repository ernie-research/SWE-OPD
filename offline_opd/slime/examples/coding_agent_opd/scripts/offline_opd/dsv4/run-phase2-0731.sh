#!/bin/bash
# Phase 2 (teacher scoring) with DeepSeek-V4-Flash-0731 as teacher.
#
# Teacher-side specifics:
#   * template: 0731 ships only encoding_dsv4.py; chat_template.jinja comes from
#     dsv4/make_dsv4_0731_chat_template.py (byte-checked against
#     encode_messages(thinking_mode="thinking")). Earlier-turn reasoning is kept.
#   * bridge: --protocol-pairs v0731 -> <tool_call> = first |DSML| of the turn,
#     <|observation|> = <|end_of_sentence|>; </tool_call> has no 1:1 token (anchor).
#   * weights: FP8 attention + MXFP4 routed experts (156G), sglang's fp4-expert path.
# Run on pod 0 of a 4-pod x 8-GPU job: 2 teacher replicas x (2 nodes, TP16).
set -euo pipefail
# B: rw view of the run base, RO: read-only view of the same storage that resolves on every pod.
B="${DSV4_0731_RW_BASE:-/path/to/<space>-<user>/<user>}"
RO="${DSV4_0731_RO_BASE:-/path/to/<user>/<user>}"
DSV4="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OFFLINE_OPD_DIR="$(dirname "${DSV4}")"
export OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-${B}/offline_opd}"
export RUN_ROOT="${RUN_ROOT:-${OFFLINE_OPD_BASE}/glm45air-dsv4flash0731-protocolpairs-anchor-lt88k-ep2-gbs64-20261016}"
export OFFLINE_OPD_RUN_TAG="$(basename "${RUN_ROOT}")"
export DSV4_STATE_FILE="${RUN_ROOT}/.state"

# Read-only views that resolve identically on every pod.
export HF_CHECKPOINT="${HF_CHECKPOINT:-/path/to/<user>/model/<org>/GLM-4.5-Air}"
export LOAD_PATH="${LOAD_PATH:-/path/to/<user>/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/<user>/<user>/Megatron-LM}"
export TEACHER_MODEL="${TEACHER_MODEL:-${RO}/opd_glm45air_dsv4flash/teacher_serving_0731}"
export OPD_TEACHER_TEMPLATE_MODEL="${OPD_TEACHER_TEMPLATE_MODEL:-${RO}/opd_glm45air_dsv4flash/teacher_shim_0731}"
export OPD_TOKEN_ID_BRIDGE="${OPD_TOKEN_ID_BRIDGE:-${RO}/offline_opd/$(basename "${RUN_ROOT}")/glm45air_to_dsv4flash_token_ids.json}"
export DSV4_FLASH_MLA_SITE="${DSV4_FLASH_MLA_SITE:-${RO}/dsv4_glm45air_opd_probe/pysite}"
export EXPECT_UNMAPPED_STUDENT_IDS=58296   # 58298 - <tool_call> - <|observation|> (</tool_call> stays unmapped for 0731)

# Alignment/teacher settings for the canonical-template rebuild.
export OPD_TEACHER_NATIVE_REASONING_PREFIX=0
export OPD_TEACHER_CANONICAL_TEMPLATE=1
export OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS=0
export OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE=0
export OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT=0
export OPD_TEACHER_MIN_ALIGNMENT_RATE=0.75
export TEACHER_ATTENTION_MODE=dsa
export TEACHER_PREWARM_WEIGHTS=0
unset TEACHER_PREWARM_CONCURRENCY TEACHER_MODEL_LOADER_THREADS
export SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S=7200
export TEACHER_WAIT_TIMEOUT_SEC=7200
# sglang uses dist_init_port + small offsets internally. A co-located job holds ray
# worker ports 22000-25000 and its own sglang in 25xxx, so the NCCL base is 40000
# (free on all 4 pods when checked).
export TEACHER_PORT="${TEACHER_PORT:-15161}"
export TEACHER_NCCL_PORT_BASE="${TEACHER_NCCL_PORT_BASE:-40000}"
export TEACHER_LAUNCH_ATTEMPTS="${TEACHER_LAUNCH_ATTEMPTS:-6}"
export OPD_FORWARD_KL_COEF=0 OPD_REVERSE_KL_COEF=1 OPD_STUDENT_ANCHOR_KL_COEF=1.0

# The Phase-3 objective decides what Phase 2 stores (always the Teacher-indexed dense
# dump, then build_sampled_token_set.py for the chunk artifact):
#   sampled_token -> own-token log-probs only (teacher_token_logprob), no Top-K
#   topk_kl       -> additionally the teacher's own Top-OPD_TEACHER_CACHE_TOPK (1000)
export OFFLINE_OPD_TEACHER_DENSE_TOPK=1
case "${OPD_PHASE3_OBJECTIVE:-sampled_token}" in
   sampled_token) export OPD_TEACHER_CACHE_TOPK=0 ;;
   topk_kl) export OPD_TEACHER_CACHE_TOPK="${OPD_TEACHER_CACHE_TOPK:-1000}" ;;
   *) echo "OPD_PHASE3_OBJECTIVE must be sampled_token or topk_kl" >&2; exit 1 ;;
esac

[[ -f "${RUN_ROOT}/.phase1.complete" && -d "${RUN_ROOT}/dump" ]] || { echo "phase1 links missing in ${RUN_ROOT}" >&2; exit 1; }
[[ -r "${OPD_TOKEN_ID_BRIDGE}" ]] || { echo "bridge not readable: ${OPD_TOKEN_ID_BRIDGE}" >&2; exit 1; }
[[ -f "${DSV4_FLASH_MLA_SITE}/flash_mla/flash_mla_interface.py" ]] || { echo "flash_mla missing" >&2; exit 1; }
printf '%s\n' "${RUN_ROOT}" > "${DSV4_STATE_FILE}"

exec bash "${DSV4}/02-phase2-teacher.sh" "$@"
