#!/bin/bash
# Stage 20: sglang (local fork, editable) + re-pin everything back onto the cu12
set -ex
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv

cd "$SGLANG_DIR"
# The reused tree (GOLDEN_OPD_DIR/sglang) is a plain source snapshot without
# .git (version pinned in ../repos.lock), so rev-parse is best-effort.
git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no git metadata at $SGLANG_DIR (snapshot; see repos.lock)"
git rev-parse --short HEAD 2>/dev/null || true

uv pip install -e "python[all]" --extra-index-url "$PT_CU129"

# sglang's resolve can pull a different torch; force ours back.
uv pip install --reinstall --no-deps \
   torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
   --index-url "$PT_CU129"

# sglang's native kernels resolve to cu13 builds on pypi (they link
# libcudart.so.13 and fail to load here). Take the +cu129 builds from sglang's
# own wheel index.
uv pip install --reinstall --no-deps \
   sglang-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
   --index-url "$SGL_CU129"

# Purge the cu13-suffixed nvidia wheels sglang dragged in: they co-own
# site-packages/nvidia/* with the cu12 ones and shadow them.
uv pip uninstall \
   nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime \
   nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand \
   nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 \
   nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx \
   nvidia-cutlass-dsl-libs-cu13 || true

# Exactly what torch 2.11.0+cu129 pins in its Requires-Dist.
uv pip install --reinstall --no-deps \
   nvidia-cudnn-cu12==9.17.1.4 \
   nvidia-cusparselt-cu12==0.7.1 \
   nvidia-nccl-cu12==2.28.9 \
   nvidia-nvshmem-cu12==3.4.5 \
   triton==3.6.0

# The purge/reinstall dance can drop the 12.9 toolkit wheels the shim links
# against; put any missing one back without touching the rest of the resolve.
uv pip install --no-deps "cuda-toolkit[all]==12.9.1" "cuda-python==12.9.*"

# `kernels` 0.16.x changed LayerRepository's signature, so transformers 5.6.0's
# integrations/hub_kernels.py dies at import with
#   ValueError: Either a revision or a version must be specified.
# -> `import sglang` fails. Pin the pair the reference venv uses.
uv pip install --no-deps "kernels==0.12.3" "kernels-data==0.16.0"

python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version())
assert torch.__version__.endswith("+cu129"), torch.__version__
import sglang
from sglang.version import __version__ as sgl_ver
print("sglang:", sgl_ver, sglang.__file__)
PY
