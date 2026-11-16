#!/bin/bash
# Stage 00: create the venv, install torch 2.11.0+cu129, and put the pip
# CUDA 12.9 toolkit wheels in place. Nothing here compiles CUDA yet -- stage 10
# validates the shim before we spend an hour on flash-attn / apex / TE.
set -ex
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

mkdir -p "$BUILD_DIR"

# rustc/cargo + protoc are needed by sglang's editable install (sglang-grpc is
# built with setuptools-rust).
if ! command -v cargo >/dev/null 2>&1; then
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
      | sh -s -- -y --profile minimal --default-toolchain stable
fi
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo
if ! command -v protoc >/dev/null 2>&1; then
   apt-get install -y --no-install-recommends protobuf-compiler
fi
command -v protoc

uv venv "$VENV_DIR" --python 3.12
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# Pin the cu12 line before anything can drag in cuda 13 bindings.
uv pip install "cuda-python==12.9.*"

# torch must come from pytorch's own cu129 index; pypi only has the cu130 build
# and a +cu130 torch makes every downstream CUDA extension fail with
# "CUDA_MISMATCH 12.9 vs 13.0".
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url "$PT_CU129"

# Full CUDA 12.9 toolkit as wheels: this is where nvcc/cicc/ptxas, the CUDA
# headers and cccl come from, since the container only has CUDA 13.0.
uv pip install "cuda-toolkit[all]==12.9.1"

uv pip install wheel cmake ninja "setuptools<80.0.0" pybind11 "packaging>=24.2"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
