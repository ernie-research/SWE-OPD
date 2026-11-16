#!/bin/bash
# Stage 32: apex (cpp_ext + cuda_ext) and the remaining native/git deps.
# apex is the most header-sensitive consumer of the CUDA shim -- if the shim is
# incomplete, this is where it shows up.
# Commits for torch_memory_saver and Megatron-Bridge are taken from the
# internal reference venv (an already validated environment of the same
# lineage).
set -ex
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv

uv pip install pip poetry pybind11

SKIP_APEX=${SKIP_APEX:-0}
if [[ "$SKIP_APEX" != "1" ]]; then
NVCC_APPEND_FLAGS="--threads 4" \
   pip -v install --disable-pip-version-check --no-cache-dir \
   --no-build-isolation \
   --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
   git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
fi

# torch_memory_saver's setup.py refuses to build a CUDA extension unless
# TMS_CUDA_MAJOR is set (upstream expects you to go through
# scripts/build_multi_cuda.sh). The reference venv's artifacts are named
# torch_memory_saver_hook_mode_{torch,preload}_cu12.abi3.so, i.e. major 12.
TMS_CUDA_MAJOR=12 \
   uv pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@6d5bce48c898c64497be2022fbdc08cc7ddb5637 \
   --no-cache --reinstall-package torch-memory-saver

uv pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a --no-build-isolation
uv pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

python - <<'PY'
import apex, amp_C, fused_layer_norm_cuda
print("apex ok:", apex.__file__)
import torch_memory_saver
print("torch_memory_saver ok")
# megatron-bridge installs under the `megatron` namespace, not `megatron_bridge`.
import importlib.util
print("megatron.bridge present:", importlib.util.find_spec("megatron.bridge") is not None)
PY
