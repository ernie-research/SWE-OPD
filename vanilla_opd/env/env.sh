#!/bin/bash
# vanilla_opd slime environment. `source` this before running anything.
#   source /path/to/swe-opd/vanilla_opd/env/env.sh
#
# Node: 8x H800 (sm90). Container ships CUDA 13.0 only -- CUDA_HOME points at a
# self-contained 12.9 tree under this workspace (see env/example/10_cuda_shim.sh),
# built from NVIDIA's cuda_nvcc 12.9.86 redist archive + the venv's cu12 wheels.
# /usr/local/cuda-13.0 is never used by this env.

SWE_OPD_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export SWE_OPD_DIR
export SLIME_DIR=$SWE_OPD_DIR/slime
# sglang / Megatron-LM are NOT vendored here -- this repo reuses the identical
# source trees from the sibling standalone package (see README.md "Reuse").
# GOLDEN_OPD_DIR defaults to ../offline_opd (the swe-opd/ layout) and can be
# overridden by exporting it before sourcing; the venv's editable installs
# point at the same two trees (sglang finder / megatron_core finder), and
# PYTHONPATH keeps megatron.training resolvable.
export GOLDEN_OPD_DIR="${GOLDEN_OPD_DIR:-$(cd "$SWE_OPD_DIR/../offline_opd" && pwd)}"
export SGLANG_DIR=$GOLDEN_OPD_DIR/sglang
export MEGATRON_DIR=$GOLDEN_OPD_DIR/Megatron-LM
export SLIME_VENV=$SWE_OPD_DIR/slime-venv

# shellcheck disable=SC1091
source "$SLIME_VENV/bin/activate"

export CUDA_HOME=$SWE_OPD_DIR/cuda-12.9-shim
export CUDA_PATH=$CUDA_HOME
export PATH="$CUDA_HOME/bin:$PATH"

# torch 2.11.0+cu129 was built against cudnn 9.17.1; the container's system
# cudnn is on the default loader path and would shadow the wheel one, so the
# venv's cudnn MUST come first.
export CUDNN_PATH="$SLIME_VENV/lib/python3.12/site-packages/nvidia/cudnn"
# TE's locally-built torch ext links against libnccl.so.2; torch 2.11+cu129 is
# built as a NON-bundled-nccl build, so the nvidia-nccl-cu12 wheel lib has to be
# on the loader path explicitly.
export NCCL_PATH="$SLIME_VENV/lib/python3.12/site-packages/nvidia/nccl/lib"
export LD_LIBRARY_PATH="$CUDNN_PATH/lib:$NCCL_PATH:$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# SLIME_DIR MUST precede MEGATRON_DIR: both trees have an `examples/` package
# with an __init__.py, so whichever comes first owns the `examples` name. slime's
# entrypoints import examples.coding_agent_opd.*, so slime has to win.
#
# CAVEAT: the sibling package's eval/run_eval.sh re-exports
# PYTHONPATH="$MEGATRON_LM:$SLIME_DIR:$PYTHONPATH", i.e. Megatron FIRST, undoing
# this ordering. It only works because that script cd's into $SLIME_DIR, so
# sys.path[0] (the cwd) shadows both. Do not run eval_chat from another cwd with
# that ordering -- `import examples` then resolves to Megatron's package.
export PYTHONPATH="$SLIME_DIR:$MEGATRON_DIR${PYTHONPATH:+:$PYTHONPATH}"
# The sibling package's eval/serve scripts (../offline_opd: eval/run_eval.sh,
# eval/serve_ckpt.sh) read MEGATRON_LM. They default it to $SLIME_DIR/../Megatron-LM,
# which does NOT exist in this layout (Megatron lives in $GOLDEN_OPD_DIR), so
# pinning it here is required -- see env/env.readme.md §6.
export MEGATRON_LM=$MEGATRON_DIR
export TORCH_CUDA_ARCH_LIST="9.0"
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Qwen3.5-9B has head_dim=256. TE 2.17 prefers its cuDNN FusedAttention backend
# for thd/packed sequences, but that path is broken at cuDNN 9.17.1: the bwd
# aborts with CUDNN_STATUS_BAD_PARAM on CUDNN_ATTR_OPERATION_RESHAPE_MODE
# (fused_attn_f16_arbitrary_seqlen.cu:934). Disabling it makes TE dispatch to
# the flash-attn-4 4.0.0b15 CUTE kernel, which does head_dim=256 thd fwd+bwd
# correctly -- that is the whole reason this build pins TE 2.17 + FA4.
# Only affects TE (megatron training); the sglang serving path never sees it.
export NVTE_FUSED_ATTN="${NVTE_FUSED_ATTN:-0}"

# Proxy policy. The original build ran inside an intranet that routes external
# traffic through a corporate proxy; that host is masked in this snapshot. If
# you need external egress (swebench's test-spec builder does an UNTIMED
# requests.get to raw.githubusercontent.com, so the proxy really must be set for
# rollout), export HTTP_PROXY/HTTPS_PROXY (and NO_PROXY for in-cluster hosts)
# BEFORE sourcing this file. Left unset, no proxy is configured (in-cluster
# hosts stay in no_proxy via the private ranges below; RLE hosts are served by
# the sandbox backend directly, see slime/agent/sandbox_rle.py).
export no_proxy="127.0.0.1,localhost,10.0.0.0/8,100.64.0.0/10,192.168.0.0/16,172.16.0.0/12"
export NO_PROXY="$no_proxy"
export http_proxy="${HTTP_PROXY:-}"
export https_proxy="$http_proxy"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$http_proxy"

# RLE sandbox (credentials live in $SLIME_DIR/.secret/.key.sh, gitignored).
if [[ -f "$SLIME_DIR/.secret/.key.sh" ]]; then
   # shellcheck disable=SC1091
   source "$SLIME_DIR/.secret/.key.sh"
fi
