#!/bin/bash
# Stage 31: transformer_engine 2.17.0 + the FA4 path.
#

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv

uv pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps

# transformer_engine_cu12 2.17.0 ships as a prebuilt manylinux wheel; the
# torch bindings (transformer_engine_torch) are compiled locally -> needs the
# 12.9 nvcc from the shim.
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.17.0"

# The `transformer_engine[pytorch]` extra resolves to transformer-engine-CU13 on
# this box, which is the wrong CUDA line for a +cu129 torch. Force the cu12
# variant (the reference venv has transformer_engine_cu12==2.17.0).
uv pip uninstall transformer-engine-cu13 || true
uv pip install --no-deps "transformer_engine_cu12==2.17.0" "nvidia-cudnn-frontend==1.26.0"

uv pip install flash-linear-attention==0.4.1

# Pure-python (py3-none-any); it JIT-builds its CUTE kernels through
# nvidia-cutlass-dsl at first use, and installs into the `flash_attn.cute`
# namespace (there is no top-level `flash_attn_4` module).
uv pip install flash-attn-4==4.0.0b15
# TE 2.17's dot_product_attention/backends.py imports flash_attn.cute.interface
# at module import time, which needs `cutlass` from nvidia-cutlass-dsl. Its libs
# package only exists in a cu13 flavour at 4.5.1, and stage 20's cu13 purge
# (inherited from the upstream build_uv line) removes it -- the reference venv keeps it, so put
# it back. Without it `import transformer_engine.pytorch` dies with
# ModuleNotFoundError: No module named 'cutlass'.
uv pip install --no-deps "nvidia-cutlass-dsl-libs-cu13==4.5.1"

python - <<'PY'
import transformer_engine as te, transformer_engine.pytorch as tep
print("transformer_engine:", te.__version__)
import transformer_engine_torch as tex
print("te_torch ext ok")
import cutlass
from flash_attn.cute.interface import flash_attn_func as _fa4
print("flash_attn.cute (FA4) ok")
PY
