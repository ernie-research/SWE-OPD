#!/bin/bash
# Shared env for the swe_opd slime uv build. `source` this, don't execute.
#
# Node facts (H800 x8, sm90; container ships CUDA 13.0 ONLY):
#   * There is no /usr/local/cuda-12.9 here, so CUDA_HOME points at a shim dir
#     synthesized from NVIDIA's pip cuda-toolkit 12.9 wheels (see 10_cuda_shim.sh).
#   * torch 2.11.0+cu129 was compiled against cudnn 9.17.1; the container's
#     /usr/lib/x86_64-linux-gnu cudnn would shadow the wheel one, hence the
#     LD_LIBRARY_PATH prepend below.

BASE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
export BASE_DIR
export VENV_DIR=$BASE_DIR/slime-venv
export SLIME_DIR=$BASE_DIR/slime
# sglang / Megatron-LM are NOT vendored here: this repo reuses the identical
# source trees from the sibling standalone package (see README.md "Reuse").
# GOLDEN_OPD_DIR defaults to ../offline_opd (the swe-opd/ layout) and can be
# overridden by exporting it before the build; it must contain sglang/ and
# Megatron-LM/ (commit pins in repos.lock).
export GOLDEN_OPD_DIR="${GOLDEN_OPD_DIR:-$(cd "$BASE_DIR/../offline_opd" && pwd)}"
export SGLANG_DIR="${SGLANG_DIR:-${GOLDEN_OPD_DIR}/sglang}"
export MEGATRON_DIR="${MEGATRON_DIR:-${GOLDEN_OPD_DIR}/Megatron-LM}"
export CUDA_SHIM=$BASE_DIR/cuda-12.9-shim
export BUILD_DIR=$BASE_DIR/env/example

# --- pip indexes -----------------------------------------------------------
# The original build used an internal PyPI mirror (its host is masked here);
# defaults now point at public PyPI and every value can be overridden BEFORE
# sourcing this file, e.g.:
#   export UV_INDEX_URL=https://<your-mirror>/simple
#   export UV_INSECURE_HOST=<your-mirror-host>   # only if that mirror's cert is untrusted
# Note: `rle-open-sdk` is only published on the internal mirror, so if a build
# step needs it, override UV_INDEX_URL with the internal mirror.
export UV_INDEX_STRATEGY=unsafe-best-match
export UV_INDEX_URL="${UV_INDEX_URL:-https://pypi.org/simple}"
export UV_EXTRA_INDEX_URL="${UV_EXTRA_INDEX_URL:-https://pypi.org/simple}"
export UV_INSECURE_HOST="${UV_INSECURE_HOST:-}"
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=180

export no_proxy="${no_proxy:-localhost,127.0.0.1,10.0.0.0/8,100.64.0.0/10,192.168.0.0/16,172.16.0.0/12}"
export NO_PROXY="$no_proxy"
export http_proxy="${HTTP_PROXY:-}"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

# --- CUDA / build ----------------------------------------------------------
export CUDA_HOME="$CUDA_SHIM"
export CUDA_PATH="$CUDA_SHIM"
export PATH="$CUDA_SHIM/bin:$PATH"
# H800 = sm90 only. Keeps flash-attn / apex / TE compiles from fanning out to
# every arch (and from tripping over sm103 codegen this nvcc predates).
export TORCH_CUDA_ARCH_LIST="9.0"
export MAX_JOBS=${MAX_JOBS:-64}
export NVCC_APPEND_FLAGS="--threads 4"

activate_venv() {
   # shellcheck disable=SC1091
   source "$VENV_DIR/bin/activate"
   export LD_LIBRARY_PATH="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV_DIR/lib/python3.12/site-packages/nvidia/nccl/lib:$CUDA_SHIM/lib64:${LD_LIBRARY_PATH:-}"
   export CUDNN_PATH="$VENV_DIR/lib/python3.12/site-packages/nvidia/cudnn"
}

PT_CU129=https://download.pytorch.org/whl/cu129
SGL_CU129=https://docs.sglang.io/whl/cu129/
export PT_CU129 SGL_CU129
