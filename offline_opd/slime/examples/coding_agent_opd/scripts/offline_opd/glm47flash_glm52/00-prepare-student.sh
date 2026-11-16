#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"
ensure_shared_token_ids

if [[ -s "${LOAD_PATH}/latest_checkpointed_iteration.txt" ]]; then
   echo "GLM-4.7-Flash torch_dist checkpoint already exists: ${LOAD_PATH}"
   exit 0
fi
[[ -d "${HF_CHECKPOINT}" ]] || { echo "student HF model missing: ${HF_CHECKPOINT}" >&2; exit 1; }
[[ ! -e "${LOAD_PATH}" || -d "${LOAD_PATH}" ]] || {
   echo "LOAD_PATH is not a directory: ${LOAD_PATH}" >&2
   exit 1
}
mkdir -p "${LOAD_PATH}"
source "${MODEL_ARGS_SH}"
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "${SLIME_DIR}"
"${SLIME_DIR}/.slime_venv/bin/python3" -m torch.distributed.run \
   --standalone --nproc-per-node "${CONVERT_GPUS:-8}" \
   tools/convert_hf_to_torch_dist.py \
   "${MODEL_ARGS[@]}" \
   --hf-checkpoint "${HF_CHECKPOINT}" \
   --save "${LOAD_PATH}"
[[ -s "${LOAD_PATH}/latest_checkpointed_iteration.txt" ]] || {
   echo "conversion exited without a checkpoint tracker: ${LOAD_PATH}" >&2
   exit 1
}
