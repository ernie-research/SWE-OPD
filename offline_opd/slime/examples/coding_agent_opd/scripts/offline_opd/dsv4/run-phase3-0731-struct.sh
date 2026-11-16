#!/bin/bash
# Phase 3 for GLM-4.5-Air <- DeepSeek-V4-Flash-0731: chunk-aligned sampled-token OPD.
#
# Settings are the ones glm45air-dsv4flash0731-chunk-sampledtoken-full4093-20261029
# trained with (taken from its argument dump). The pipeline before this script:
#   1. Phase 2, Teacher-indexed dense dump with own-token log-probs only (no Top-K;
#      the 10-29 run's Phase 2 also stored a Top-1000 that this loss never reads):
#        bash dsv4/run-phase2-0731.sh            # OPD_PHASE3_OBJECTIVE=sampled_token
#   2. Dense -> chunk artifact (always mode=chunk, so every tier stays selectable; the
#      batch manifests are carried over):
#        CHUNK_SRC_SCORED=<dense>/scored CHUNK_DST_SCORED=<run>/scored CHUNK_ALIGNER=dsv4 \
#          XVOCAB_ASSETS=... OPD_TEACHER_TEMPLATE_MODEL=<teacher_shim_0731> \
#          bash build_chunk_scored.sh
#   3. A length-filtered view of <run>/scored that fits MAX_TOKENS_PER_GPU x CP_SIZE
#      (45056 x 2 ~ 88k; the 10-29 run used scored_lt88k), passed as SCORED_DIR.
#
# OPD_CHUNK_TIER picks exact / struct / chunk at train time; struct is what the 10-29
# run used. The tier must not be wider than the artifact's chunk mode (checked in
# slime/ray/rollout.py).
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OFFLINE_OPD_LOSS_TYPE=chunk_teacher_sampled_token_pg_with_student_anchor
export OPD_CHUNK_TIER="${OPD_CHUNK_TIER:-struct}"
export SCORED_DIR="${SCORED_DIR:?set SCORED_DIR to the length-filtered chunk artifact (e.g. <run>/scored_lt88k)}"
export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-45056}"
# The chunk artifact carries no teacher Top-K, so there is nothing to measure overlap on.
export OFFLINE_OPD_TOPK_OVERLAP="${OFFLINE_OPD_TOPK_OVERLAP:-0}"
export OPD_KL_COEF="${OPD_KL_COEF:-1.0}"
export OPD_STUDENT_ANCHOR_KL_COEF="${OPD_STUDENT_ANCHOR_KL_COEF:-1.0}"
export OPD_SAMPLED_TOKEN_CACHE_MISS_POLICY="${OPD_SAMPLED_TOKEN_CACHE_MISS_POLICY:-anchor}"
export OPD_SAMPLED_TOKEN_CLIP_REFERENCE="${OPD_SAMPLED_TOKEN_CLIP_REFERENCE:-rollout}"
export OPD_SAMPLED_TOKEN_SDAR_GATE="${OPD_SAMPLED_TOKEN_SDAR_GATE:-0}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-dsv4flash0731-chunk-${OPD_CHUNK_TIER}}"

exec bash "${SCRIPT_DIR}/03-phase3-train.sh" "$@"
