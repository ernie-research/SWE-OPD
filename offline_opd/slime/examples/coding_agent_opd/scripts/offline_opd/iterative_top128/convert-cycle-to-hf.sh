#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"

CYCLE="${1:?usage: convert-cycle-to-hf.sh <cycle-number>}"
[[ "${CYCLE}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid cycle: ${CYCLE}" >&2; exit 1; }
: "${RUN_ROOT:?RUN_ROOT is required}"
validate_run_root "${RUN_ROOT}"

CYCLE_DIR="$(cycle_dir "${CYCLE}")"
CHECKPOINT_ROOT="${RUN_ROOT}/checkpoints"
HF_DIR="${CYCLE_DIR}/hf"
LOG_DIR="${CYCLE_DIR}/logs-convert"
MARKER="${CYCLE_DIR}/.hf.complete"
mkdir -p "${LOG_DIR}"

[[ -r "${CYCLE_DIR}/checkpoint.iter" ]] || {
   echo "missing cycle checkpoint reference: ${CYCLE_DIR}/checkpoint.iter" >&2
   exit 1
}
ITERATION="$(<"${CYCLE_DIR}/checkpoint.iter")"
[[ "${ITERATION}" =~ ^[0-9]+$ ]] || {
   echo "invalid cycle checkpoint iteration: ${ITERATION}" >&2
   exit 1
}
ITER_DIR="$(printf '%s/iter_%07d' "${CHECKPOINT_ROOT}" "${ITERATION}")"
[[ -f "${ITER_DIR}/common.pt" ]] || {
   echo "not a complete torch_dist checkpoint: ${ITER_DIR}" >&2
   exit 1
}

if [[ -r "${HF_DIR}/config.json" && -f "${MARKER}" ]]; then
   echo "HF export already complete: ${HF_DIR}"
   exit 0
fi

SLIME_VENV="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}"
[[ -x "${SLIME_VENV}/bin/python3" ]] || {
   echo "missing Python environment: ${SLIME_VENV}/bin/python3" >&2
   exit 1
}
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

CONVERT_ARGS=(
   --input-dir "${ITER_DIR}"
   --output-dir "${HF_DIR}"
   --origin-hf-dir "${ORIGIN_HF}"
   --add-missing-from-origin-hf
)
# An interrupted conversion leaves a directory without the completion marker.
# The converter safely overwrites named shards and writes config.json last.
[[ -e "${HF_DIR}" ]] && CONVERT_ARGS+=(--force)

echo "[$(date -Is)] converting ${ITER_DIR} -> ${HF_DIR}"
"${SLIME_VENV}/bin/python3" "${SLIME_DIR}/tools/convert_torch_dist_to_hf.py" \
   "${CONVERT_ARGS[@]}" > "${LOG_DIR}/convert.log" 2>&1

[[ -r "${HF_DIR}/config.json" && -r "${HF_DIR}/model.safetensors.index.json" ]] || {
   echo "HF conversion did not produce a complete model: ${HF_DIR}" >&2
   exit 1
}
write_marker "${MARKER}" "cycle=${CYCLE} source=${ITER_DIR} output=${HF_DIR}"
echo "[$(date -Is)] HF export completed: ${HF_DIR}"
