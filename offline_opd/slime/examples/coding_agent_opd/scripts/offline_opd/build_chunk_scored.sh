#!/bin/bash
# Phase-2 scored dir -> chunk artifact for chunk_teacher_sampled_token_pg_with_student_anchor.
#
# One entry point for the three teacher setups (see build_sampled_token_set.py):
#   CHUNK_ALIGNER=identity   same tokenizer (Qwen3.5-9B <- Qwen3.6-27B)
#   CHUNK_ALIGNER=canonical  bridged GLM pair (GLM-4.5-Air <- GLM-5.2), needs OPD_TOKEN_ID_BRIDGE
#                            when reading a cache
#   CHUNK_ALIGNER=dsv4       GLM-4.5-Air <- DeepSeek-V4-Flash-0731, needs XVOCAB_ASSETS and
#                            OPD_TEACHER_TEMPLATE_MODEL
# CHUNK_EMIT_TOPK_CACHE=K (e.g. 1000) additionally writes the anchored Top-K schema, so the
# top-K reverse KL losses train on the same artifact (with OPD_NM_ROWS_SAMPLED_TOKEN=1 the n:m
# rows train the sampled-token term).
# CHUNK_FROM_TEACHER_CACHE=1 reuses an existing Top-K artifact (identity/canonical) instead
# of a sampled-token Phase 2; the sampled token's value is looked up in the cache and a miss
# becomes a Student-anchor row, the same rule as the cached sampled-token loss.
#
# The batch manifests are copied, so the output keeps the source run's batch layout.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../../../.." &>/dev/null && pwd)"

CHUNK_SRC_SCORED="${CHUNK_SRC_SCORED:?set CHUNK_SRC_SCORED to the Phase-2 scored/ dir}"
CHUNK_DST_SCORED="${CHUNK_DST_SCORED:?set CHUNK_DST_SCORED to the output scored/ dir}"
CHUNK_ALIGNER="${CHUNK_ALIGNER:?set CHUNK_ALIGNER to identity, canonical or dsv4}"
CHUNK_FROM_TEACHER_CACHE="${CHUNK_FROM_TEACHER_CACHE:-0}"
CHUNK_WORKERS="${CHUNK_WORKERS:-48}"

if [[ -f "${CHUNK_DST_SCORED}/.chunk.complete" ]]; then
   echo "chunk artifact already complete: ${CHUNK_DST_SCORED}"
   exit 0
fi
[[ -d "${CHUNK_SRC_SCORED}/samples" ]] || { echo "no samples/ under ${CHUNK_SRC_SCORED}" >&2; exit 1; }

ARGS=(--aligner "${CHUNK_ALIGNER}" --in-dir "${CHUNK_SRC_SCORED}" --out-dir "${CHUNK_DST_SCORED}"
      --workers "${CHUNK_WORKERS}")
case "${CHUNK_FROM_TEACHER_CACHE}" in
   0) ;;
   1) ARGS+=(--from-teacher-cache) ;;
   *) echo "CHUNK_FROM_TEACHER_CACHE must be 0 or 1" >&2; exit 1 ;;
esac
# Top-K KL on the same artifact: also write the anchored Top-K schema from a dense dump
# that stored a teacher Top-K (Phase 2 with OPD_PHASE3_OBJECTIVE=topk_kl).
if [[ "${CHUNK_EMIT_TOPK_CACHE:-0}" != 0 ]]; then
   ARGS+=(--emit-topk-cache "${CHUNK_EMIT_TOPK_CACHE}"
          --dynamic-support-topk "${OPD_DYNAMIC_SUPPORT_TOPK:-128}")
   [[ -n "${OPD_TOKEN_ID_BRIDGE:-}" ]] && ARGS+=(--token-id-bridge "${OPD_TOKEN_ID_BRIDGE}")
fi
case "${CHUNK_ALIGNER}" in
   identity) ;;
   canonical)
      if [[ "${CHUNK_FROM_TEACHER_CACHE}" == 1 ]]; then
         ARGS+=(--token-id-bridge "${OPD_TOKEN_ID_BRIDGE:?canonical cache reuse needs OPD_TOKEN_ID_BRIDGE}")
      fi
      ;;
   dsv4)
      ARGS+=(--assets "${XVOCAB_ASSETS:?dsv4 needs XVOCAB_ASSETS}"
             --teacher-template-model "${OPD_TEACHER_TEMPLATE_MODEL:?dsv4 needs OPD_TEACHER_TEMPLATE_MODEL}")
      [[ -n "${OPD_TEACHER_REASONING_EFFORT:-}" ]] && ARGS+=(--reasoning-effort "${OPD_TEACHER_REASONING_EFFORT}")
      ;;
   *) echo "CHUNK_ALIGNER must be identity, canonical or dsv4" >&2; exit 1 ;;
esac

PYTHON="${SLIME_DIR}/.slime_venv/bin/python3"
[[ -x "${PYTHON}" ]] || PYTHON=python3
mkdir -p "${CHUNK_DST_SCORED}"
PYTHONPATH="${SLIME_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
   "${PYTHON}" "${SCRIPT_DIR}/build_sampled_token_set.py" "${ARGS[@]}"
[[ -r "${CHUNK_DST_SCORED}/rollout_0.pt" ]] || {
   echo "source run has no sample manifests to carry over; run build_scored_manifests.py" >&2
   exit 1
}
printf 'aligner=%s from_teacher_cache=%s source=%s completed=%s\n' \
   "${CHUNK_ALIGNER}" "${CHUNK_FROM_TEACHER_CACHE}" "${CHUNK_SRC_SCORED}" "$(date -Is)" \
   > "${CHUNK_DST_SCORED}/.chunk.complete"
