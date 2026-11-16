#!/bin/bash
set -euo pipefail

# Isolated Phase-3 experiment: keep the global protocol-token rollout-Student
# anchor unchanged while transferring Teacher forward-KL weight to reverse-KL:
#   step 0:  forward=0.5, reverse=0.5
#   step 10: forward=0.25, reverse=0.75
#   step 20+: forward=0.0, reverse=1.0
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"

if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-global-protocol-anchor-fwdanneal20-reverse-high-full4093-a1-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi

export RUN_ROOT
export OPD_FORWARD_KL_ANNEAL_STEPS=20
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-global-protocol-anchor-fwdanneal20-reverse}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

exec bash "${SCRIPT_DIR}/19-run-global-protocol-anchor-bidirectional-high-full-ep2.sh" "$@"
