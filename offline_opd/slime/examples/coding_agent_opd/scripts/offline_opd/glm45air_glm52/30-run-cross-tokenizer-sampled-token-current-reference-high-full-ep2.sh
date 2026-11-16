#!/bin/bash
set -euo pipefail

# Isolated pure sampled-token OPD experiment.  It reuses the canonical High
# Teacher Top-1000 cache and the original GLM-4.5-Air Student, but snapshots
# each current forward as the detached PPO reference.  Therefore the numerical
# ratio is one and clipfrac is zero, while the sampled-token policy-gradient
# through the current log-prob remains active.  Existing experiments keep the
# default rollout reference.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_SAMPLED_TOKEN_CLIP_REFERENCE=current
export OPD_SAMPLED_TOKEN_SDAR_GATE=0
export RUN_ROOT="${RUN_ROOT:-/path/to/offline_opd/glm45air-glm52-crossvocab-pure-sampled-token-current-ref-t1k-high-full4093-a1-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-crossvocab-pure-sampled-token-current-ref-t1k}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

exec bash "${SCRIPT_DIR}/25-run-cross-tokenizer-sampled-token-pg-high-full-ep2.sh" "$@"
