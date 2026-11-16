#!/bin/bash
# Stage 30: flash-attn 2.7.4.post1 from source (sm90 only).
# This is the longest compile in the build; TORCH_CUDA_ARCH_LIST=9.0 (set in
# common.sh) keeps it to the one arch this H800 node needs.
set -ex
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv

nvcc --version | tail -1
MAX_JOBS=${MAX_JOBS:-64} uv pip install -v flash-attn==2.7.4.post1 --no-build-isolation

python -c "import flash_attn, flash_attn_2_cuda; print('flash_attn', flash_attn.__version__)"
