#!/bin/bash
set -euo pipefail

# Isolated normalized-SDAR experiment.  Relative to experiment 27, only the
# exact cached-Teacher gate changes from sigmoid(beta*gap) to
# 2*sigmoid(beta*gap).  Thus positive Teacher-current gaps receive gain 1..2,
# while negative gaps receive gain 0..1.  Legacy launches retain scale=1.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_SDAR_GATE_SCALE=2.0
export RUN_ROOT="${RUN_ROOT:-/path/to/offline_opd/glm45air-glm52-crossvocab-sdar-normalized-gate2-sampled-token-pg-beta5-t1k-high-full4093-a1-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-crossvocab-sdar-normalized-gate2-sampled-token-beta5-t1k}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

exec bash "${SCRIPT_DIR}/27-run-cross-tokenizer-sdar-sampled-token-pg-beta5-high-full-ep2.sh" "$@"
