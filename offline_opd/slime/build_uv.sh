#!/bin/bash

# Do not enable xtrace here: proxy URLs may contain credentials.
set -eo pipefail

# capture the directory the script was invoked from before we cd around, so
# the final `.slime_venv` symlink lands where the user launched the build.
ORIG_DIR="$(pwd)"

export SGLANG_VERSION="v0.5.12.post1"
export SGLANG_COMMIT="5a15cde858ea09b77116212a39356f2fc51b8584"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"

export BASE_DIR=${BASE_DIR:-"/root"}
mkdir -p $BASE_DIR
cd $BASE_DIR

# make uv available even when installed to ~/.local/bin
export PATH="$HOME/.local/bin:$PATH"

# Primary index for pure-python deps. Override UV_INDEX_URL to use a mirror.
# If you need an HTTP proxy, export HTTP_PROXY / HTTPS_PROXY / NO_PROXY before
# running this script; uv honors them.
export UV_INDEX_STRATEGY=${UV_INDEX_STRATEGY:-unsafe-best-match}
export UV_INDEX_URL=${UV_INDEX_URL:-https://pypi.org/simple}

# --- bootstrap uv ---------------------------------------------------------
# A fresh machine may not have uv; install it to ~/.local/bin (already on PATH).
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# copy (don't hardlink) into the venv: site-packages may live on a different
# filesystem than uv's cache on some hosts, which breaks hardlinking.
export UV_LINK_MODE=${UV_LINK_MODE:-copy}

# sglang's editable install compiles `sglang-grpc` via setuptools-rust, so we
# need a working rustc/cargo + protoc. Install if missing.
if ! command -v cargo >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal --default-toolchain stable
fi
export PATH="$HOME/.cargo/bin:$PATH"
if ! command -v protoc >/dev/null 2>&1; then
  apt-get install -y --no-install-recommends protobuf-compiler
fi

# create venv with uv
uv venv slime-venv --python 3.12
source $BASE_DIR/slime-venv/bin/activate

# CUDA_HOME should point to system CUDA 12.9 on the target B200 machine
# Adjust this path if your CUDA installation is elsewhere
export CUDA_HOME=${CUDA_HOME:-"/usr/local/cuda-12.9"}
export PATH="$CUDA_HOME/bin:$PATH"
# Deliberately DON'T add $CUDA_HOME/lib64 to LD_LIBRARY_PATH: the system
# install ships cudnn 9.16, which would shadow the pip-installed
# nvidia-cudnn-cu12 that torch 2.11.0+cu129 was compiled against (9.17.1).
# torch's own bundled nvidia-*-cu12 wheels resolve runtime libs on their own.

# prevent installing cuda 13.0 for sglang (pin to the cu12 line)
uv pip install "cuda-python==12.9.*"
# torch pinned to match sglang v0.5.12.post1's runtime expectation (>=2.11);
# the cu129 wheel is what pytorch publishes at their custom index.
uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu129

# install sglang
if [ ! -d "$BASE_DIR/sglang" ]; then
  git clone https://github.com/sgl-project/sglang.git
fi
cd $BASE_DIR/sglang
git checkout ${SGLANG_COMMIT}
# Install the python packages
# --extra-index-url cu129 so sglang's torch/nvidia-* deps resolve to the
# CUDA 12.9 wheels instead of the cu13 defaults on pypi.
uv pip install wheel
uv pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
# sglang can pull in a newer/other torch; re-pin to our cu129 build.
uv pip install --reinstall --no-deps \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129

# sglang's native kernels (sglang-kernel, sgl-deep-gemm) resolve to cu13
# builds from pypi (they link libcudart.so.13 / libnvrtc.so.13 and fail to
# load in this cu12 env, crashing `sglang.launch_server` on Hopper/Blackwell
# via the deep_gemm import). Reinstall the +cu129 builds from sglang's own
# wheel index. NB: use docs.sglang.io (docs.sglang.ai 301-redirects and the
# simple-index client won't follow it).
uv pip install --reinstall --no-deps \
  sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url https://docs.sglang.io/whl/cu129/

# sglang v0.5.12.post1's deps drag in cu13-suffix nvidia-* wheels that
# co-own the `site-packages/nvidia/*` namespace with the cu12 ones. Purge
# them, then reinstall the exact cu12 versions the cu129 torch was
# compiled against.
uv pip uninstall \
  nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
  nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
  nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 || true
# Versions here match torch 2.11.0+cu129's Requires-Dist pins.
uv pip install --reinstall --no-deps \
  nvidia-cudnn-cu12==9.17.1.4 \
  nvidia-cusparselt-cu12==0.7.1 \
  nvidia-nccl-cu12==2.28.9 \
  nvidia-nvshmem-cu12==3.4.5 \
  triton==3.6.0
# The cu13 and base CUTLASS wheels own overlapping files. Uninstalling cu13
# can therefore delete the CUDA 12 payload while leaving base metadata behind.
uv pip install --reinstall --no-deps \
  nvidia-cutlass-dsl-libs-base==4.5.1 \
  nvidia-cutlass-dsl==4.5.1

uv pip install cmake ninja

# flash attn
# the newest version megatron supports is v2.7.4.post1
MAX_JOBS=64 uv pip install -v flash-attn==2.7.4.post1 --no-build-isolation

uv pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.17.0"
uv pip install flash-linear-attention==0.4.1

uv pip install pip
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

uv pip install poetry pybind11
uv pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache --reinstall-package torch-memory-saver
uv pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
uv pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# megatron
cd $BASE_DIR
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
cd Megatron-LM/ && git checkout ${MEGATRON_COMMIT}
# pre-install Megatron's build deps for --no-build-isolation
uv pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
# --no-build-isolation so the helpers_cpp C++ extension actually builds;
# otherwise it's silently skipped and GPT dataset loading breaks.
uv pip install -e . --no-build-isolation

# install slime and apply patches
if [ ! -d "$BASE_DIR/slime" ]; then
  cd $BASE_DIR
  git clone https://github.com/THUDM/slime.git
fi
export SLIME_DIR=$BASE_DIR/slime
cd $SLIME_DIR
# Install pure-python runtime deps first, then slime itself with --no-deps so
# pip doesn't re-resolve and stomp our pinned native libs (torch+cu129,
# sglang, ...). Mirrors build_conda.sh.
uv pip install -r requirements.txt
uv pip install -e . --no-deps

# DON'T install nvidia-cudnn-cu12==9.16.0.29: torch 2.11.0+cu129 was compiled
# against cudnn 9.17.1 and will refuse to init with 9.16 at runtime. The
# 9.17.1.4 version pinned above is what stays.
# Runtime note: this container ships cudnn 9.16 under /usr/lib/x86_64-linux-gnu,
# which is on the default LD_LIBRARY_PATH and shadows the pip-installed 9.17.1.
# Anything that imports torch must prepend the venv's nvidia/cudnn/lib dir, e.g.:
#   export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
uv pip install "numpy<2"

# apply patch
# SGLANG_COMMIT/MEGATRON_COMMIT above are pinned to the v0.5.12.post1 line,
# so use the matching patch directory instead of the older v0.5.9 one.
cd $BASE_DIR/sglang
git apply $SLIME_DIR/docker/patch/v0.5.12.post1/sglang.patch
cd $BASE_DIR/Megatron-LM
git apply $SLIME_DIR/docker/patch/v0.5.12.post1/megatron.patch

# --- verify ---------------------------------------------------------------
# The pip-installed cudnn 9.17.1 must win over the system 9.16; prepend it.
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
python - <<'PY'
import hashlib
import importlib.metadata as metadata
from pathlib import Path

import torch
print("torch:", torch.__version__)
print("cuda :", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("devs :", torch.cuda.device_count())
assert torch.cuda.is_available(), "CUDA not available"
assert metadata.version("transformer-engine") == "2.17.0"
assert not any(
    dist.metadata["Name"] == "nvidia-cutlass-dsl-libs-cu13"
    for dist in metadata.distributions()
), "cu13 CUTLASS runtime must not be installed in the CUDA 12.9 environment"
runtime = next(
    Path(metadata.distribution("nvidia-cutlass-dsl").locate_file("nvidia_cutlass_dsl/lib")).glob(
        "libcute_dsl_runtime.so"
    )
)
runtime_sha = hashlib.sha256(runtime.read_bytes()).hexdigest()
print("cutlass runtime:", runtime_sha, runtime)
assert runtime_sha == "7b50763f72b058d348885b40b9b2e99d1b7181b7df51ec65c90fd07387ecebf4"
PY
# Expected: torch 2.11.0+cu129 / cuda 12.9 / cudnn 91701 / devs > 0

# create a convenience symlink to the venv in the invocation dir. -f/-n so
# re-running the build replaces the link instead of nesting inside it.
ln -sfn "$BASE_DIR/slime-venv" "$ORIG_DIR/.slime_venv"
echo "Symlinked $ORIG_DIR/.slime_venv -> $BASE_DIR/slime-venv"

cat <<EOF

Build finished. To use this env in a NEW shell:
  source $BASE_DIR/slime-venv/bin/activate
  export LD_LIBRARY_PATH="\$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:\$LD_LIBRARY_PATH"
  export CUDA_HOME=/usr/local/cuda-12.9
  export PYTHONPATH=$BASE_DIR/Megatron-LM:\$PYTHONPATH
EOF
