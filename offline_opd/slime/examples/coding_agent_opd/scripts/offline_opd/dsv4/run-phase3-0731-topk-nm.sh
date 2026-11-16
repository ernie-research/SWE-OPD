#!/bin/bash
# Phase 3 for GLM-4.5-Air <- DeepSeek-V4-Flash-0731: top-K reverse KL on the chunk artifact.
#
# 1:1 chunk rows train reverse KL against the teacher Top-1000, n:m chunk rows of a kind
# OPD_CHUNK_TIER accepts train the chunk sampled-token term (--opd-nm-rows-sampled-token),
# every other trainable row anchors on the rollout Student. The loss is
#   mean(1:1 KL) + mean(n:m PG) + OPD_STUDENT_ANCHOR_KL_COEF * mean(anchor KL).
# The pipeline before this script:
#   1. Phase 2 with the teacher Top-1000 kept:
#        OPD_PHASE3_OBJECTIVE=topk_kl bash dsv4/run-phase2-0731.sh
#   2. Dense -> chunk artifact with the anchored Top-K schema:
#        CHUNK_SRC_SCORED=<dense>/scored CHUNK_DST_SCORED=<run>/scored CHUNK_ALIGNER=dsv4 \
#          CHUNK_EMIT_TOPK_CACHE=1000 OPD_TOKEN_ID_BRIDGE=<bridge> XVOCAB_ASSETS=... \
#          OPD_TEACHER_TEMPLATE_MODEL=<teacher_shim_0731> bash build_chunk_scored.sh
#   3. A length-filtered view that fits MAX_TOKENS_PER_GPU x CP_SIZE, passed as SCORED_DIR.
# OPD_NM_ROWS_SAMPLED_TOKEN=0 gives the plain anchored reverse KL (n:m rows anchor).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OFFLINE_OPD_LOSS_TYPE="${OFFLINE_OPD_LOSS_TYPE:-cached_teacher_dynamic_topk_reverse_kl_with_student_anchor}"
export OPD_NM_ROWS_SAMPLED_TOKEN="${OPD_NM_ROWS_SAMPLED_TOKEN:-1}"
export OPD_CHUNK_TIER="${OPD_CHUNK_TIER:-struct}"
export OPD_TEACHER_CACHE_TOPK="${OPD_TEACHER_CACHE_TOPK:-1000}" OPD_DYNAMIC_SUPPORT_TOPK="${OPD_DYNAMIC_SUPPORT_TOPK:-128}"
export SCORED_DIR="${SCORED_DIR:?set SCORED_DIR to the length-filtered chunk artifact with the Top-K schema}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-45056}"
export OPD_STUDENT_ANCHOR_KL_COEF="${OPD_STUDENT_ANCHOR_KL_COEF:-1.0}"
export OPD_SAMPLED_TOKEN_CLIP_REFERENCE="${OPD_SAMPLED_TOKEN_CLIP_REFERENCE:-rollout}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-dsv4flash0731-topk-nm${OPD_NM_ROWS_SAMPLED_TOKEN}-${OPD_CHUNK_TIER}}"

exec bash "${SCRIPT_DIR}/03-phase3-train.sh" "$@"
