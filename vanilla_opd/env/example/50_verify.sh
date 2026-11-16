#!/bin/bash
# Stage 50: import-level + GPU-level smoke of the finished venv. No model
# weights, no network sandbox -- this only has to prove the env is coherent
# before Phase 3 (sglang serve + rollout + RLE scoring) is attempted.
set -ex
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

python - <<'PY'
import importlib, sys, os
print("python:", sys.version.split()[0], sys.executable)

import torch
print("torch      :", torch.__version__, "| cuda", torch.version.cuda,
      "| cudnn", torch.backends.cudnn.version(), "| devs", torch.cuda.device_count())
assert torch.__version__.endswith("+cu129"), torch.__version__
assert torch.cuda.is_available()
assert torch.backends.cudnn.version() == 91701, torch.backends.cudnn.version()

for mod in ("sglang", "megatron.core", "transformer_engine.pytorch",
            "flash_attn", "apex", "torch_memory_saver", "slime",
            "examples.coding_agent_opd.r2egym", "rle_open_sdk"):
    m = importlib.import_module(mod)
    print(f"{mod:38s} {getattr(m, '__version__', 'ok')}")

# numpy 1.x ABI: scipy>=1.18 hard-requires numpy>=2 and blows up on import.
import numpy, scipy, scipy.sparse, scipy.stats
assert numpy.__version__.startswith("1."), numpy.__version__
print(f"{'numpy/scipy':38s} {numpy.__version__} / {scipy.__version__}")

# the actual OPD entrypoints, not just the libraries underneath them
for mod in ("examples.coding_agent_opd.generate",
            "examples.coding_agent_opd.opd",
            "slime.rollout.sglang_rollout",
            "slime.backends.megatron_utils", "slime.backends.sglang_utils"):
    importlib.import_module(mod)
    print(f"{mod:38s} ok")

# cu12/cu13 namespace hygiene: a stray nvidia-*-cu13 wheel co-owning
# site-packages/nvidia/* is what breaks sglang's deep_gemm import.
# nvidia-cutlass-dsl-libs-cu13 is the one sanctioned exception -- it only ships
# under site-packages/nvidia_cutlass_dsl/ (no nvidia/* collision), the package
# has no cu12 flavour at 4.5.1, TE 2.17 needs `import cutlass` for its FA4
# backend, and the reference venv keeps it too.
CU13_ALLOWED = {"nvidia-cutlass-dsl-libs-cu13"}
import subprocess
out = subprocess.run([sys.executable, "-m", "pip", "list", "--format=freeze"],
                     capture_output=True, text=True).stdout
bad = [l for l in out.splitlines()
       if "cu13" in l and l.split("==")[0] not in CU13_ALLOWED]
print("cu13 leftovers:", bad or "none (only the sanctioned cutlass-dsl-libs-cu13)")
assert not bad, bad

# real GPU matmul + a cudnn conv, i.e. both libs actually load
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
print("matmul ok:", (a @ a).float().abs().mean().item() > 0)
x = torch.randn(2, 4, 16, 16, device="cuda")
w = torch.randn(4, 4, 3, 3, device="cuda")
print("cudnn conv ok:", torch.nn.functional.conv2d(x, w).shape)

# flash-attn 2 forward on sm90
from flash_attn import flash_attn_func
q = torch.randn(1, 128, 8, 128, device="cuda", dtype=torch.bfloat16)
print("flash_attn2 ok:", flash_attn_func(q, q, q, causal=True).shape)
PY

# TE's dot-product-attention backend selection for head_dim=256 thd/packed --
# this is the exact case that aborted under TE 2.10 and is why the build uses
# TE 2.17.0 + flash-attn-4. env.sh sets NVTE_FUSED_ATTN=0 because TE 2.17's
# cuDNN FusedAttention bwd is broken for this shape at cuDNN 9.17.1
# (CUDNN_STATUS_BAD_PARAM on CUDNN_ATTR_OPERATION_RESHAPE_MODE); the assert
# below therefore also checks that FA4 -- not FusedAttention -- was selected.
NVTE_DEBUG=1 NVTE_DEBUG_LEVEL=1 python - <<'PY'
import io, logging, torch
from transformer_engine.pytorch.attention import DotProductAttention
buf = io.StringIO()
logging.getLogger().addHandler(logging.StreamHandler(buf))
h, d, b, s = 8, 256, 2, 64
dpa = DotProductAttention(num_attention_heads=h, kv_channels=d,
                          attn_mask_type="padding_causal", qkv_format="thd").cuda()
cu = torch.arange(0, (b + 1) * s, s, dtype=torch.int32, device="cuda")
q, k, v = (torch.randn(b * s, h, d, device="cuda", dtype=torch.bfloat16,
                       requires_grad=True) for _ in range(3))
o = dpa(q, k, v, cu_seqlens_q=cu, cu_seqlens_kv=cu,
        max_seqlen_q=s, max_seqlen_kv=s)
o.sum().backward()
log = buf.getvalue()
assert "FlashAttention backend" in log, log
print("TE thd head_dim=256 fwd+bwd ok via FA4:", tuple(o.shape))
PY
echo "ENV SMOKE OK"
