#!/bin/bash
set -euo pipefail

# Sampled-token offline OPD entry point. The shared three-phase implementation
# keeps topk_reverse_kl as its default, so this wrapper opts into the native
# token_reverse_kl schema without changing existing Top-K launches.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
export OFFLINE_OPD_LOSS_TYPE=token_reverse_kl
export OPD_ID_TOPK=0
export OFFLINE_OPD_SAVE_STUDENT_TOPK_LOGPROBS=0
export OFFLINE_OPD_TOPK_OVERLAP="${OFFLINE_OPD_TOPK_OVERLAP:-0}"

exec bash "${SCRIPT_DIR}/start-offline-opd.sh" "$@"
