#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
BUILD_DIR="$ROOT_DIR/env/example"

need_cmds=(bash curl git python3 uv)
for cmd in "${need_cmds[@]}"; do
  command -v "$cmd" >/dev/null 2>&1 || {
    printf 'missing required command: %s\n' "$cmd" >&2
    exit 1
  }
done

# slime is vendored here; sglang / Megatron-LM are reused from the sibling
# standalone package (GOLDEN_OPD_DIR, default ../offline_opd) -- see README.
GOLDEN_OPD_DIR="${GOLDEN_OPD_DIR:-$ROOT_DIR/../offline_opd}"
[[ -d "$ROOT_DIR/slime" && -d "$GOLDEN_OPD_DIR/sglang" && -d "$GOLDEN_OPD_DIR/Megatron-LM" ]] || {
  printf 'missing source checkout; slime must exist here and sglang/Megatron-LM under %s (see repos.lock, README "Reuse")\n' "$GOLDEN_OPD_DIR" >&2
  exit 1
}

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null || {
    printf 'GPU query failed\n' >&2
    exit 1
  }
fi

for script in 00_venv_torch.sh 10_cuda_shim.sh 20_sglang.sh 30_flash_attn.sh 31_te.sh 32_apex.sh 40_megatron_slime.sh 50_verify.sh; do
  bash "$BUILD_DIR/$script"
done

printf 'setup complete; source %s/env/env.sh\n' "$ROOT_DIR"
