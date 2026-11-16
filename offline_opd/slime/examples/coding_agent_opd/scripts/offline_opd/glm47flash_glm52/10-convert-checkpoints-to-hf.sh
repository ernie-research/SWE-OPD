#!/bin/bash
set -euo pipefail

SLIME_DIR="${SLIME_DIR:-/path/to/workspace/env_run/<user>/slime}"
RUN_DIR="${RUN_DIR:-/path/to/offline_opd/glm47flash-glm52-student-template-reverse-t1k-s128-partial-n1280-s20-20260831_190200}"
ORIGIN_HF="${ORIGIN_HF:-/path/to/model/<org>/GLM-4.7-Flash}"
NUM_SHARDS="${NUM_SHARDS:-48}"
SAVE_WORKERS="${SAVE_WORKERS:-16}"
if (( $# > 0 )); then
   ITERATIONS=("$@")
else
   ITERATIONS=(iter_0000059 iter_0000099)
fi

[[ "${NUM_SHARDS}" == 48 ]] || {
   echo "This GLM-4.7-Flash export must use exactly 48 shards; got NUM_SHARDS=${NUM_SHARDS}" >&2
   exit 1
}
[[ "${SAVE_WORKERS}" =~ ^[1-9][0-9]*$ ]] || {
   echo "SAVE_WORKERS must be a positive integer; got ${SAVE_WORKERS}" >&2
   exit 1
}
[[ -d "${ORIGIN_HF}" ]] || { echo "Missing origin HF model: ${ORIGIN_HF}" >&2; exit 1; }

mkdir -p "${RUN_DIR}/hf" "${RUN_DIR}/logs-hf"
cd "${SLIME_DIR}"

for iter_name in "${ITERATIONS[@]}"; do
   [[ "${iter_name}" =~ ^iter_[0-9]{7}$ ]] || {
      echo "Invalid checkpoint name: ${iter_name}" >&2
      exit 1
   }
   input_dir="${RUN_DIR}/checkpoints/${iter_name}"
   output_dir="${RUN_DIR}/hf/${iter_name}"
   log_file="${RUN_DIR}/logs-hf/${iter_name}.convert48.log"

   [[ -f "${input_dir}/common.pt" ]] || {
      echo "Incomplete checkpoint: ${input_dir}" >&2
      exit 1
   }
   if [[ -e "${output_dir}" ]]; then
      backup="${output_dir}.incomplete-$(date -u +%Y%m%dT%H%M%SZ)"
      echo "Moving pre-existing output to ${backup}"
      mv "${output_dir}" "${backup}"
   fi

   echo "[$(date -Is)] converting ${iter_name} to exactly ${NUM_SHARDS} shards (${SAVE_WORKERS} parallel writers)"
   "${SLIME_DIR}/.slime_venv/bin/python3" tools/convert_torch_dist_to_hf.py \
      --input-dir "${input_dir}" \
      --output-dir "${output_dir}" \
      --origin-hf-dir "${ORIGIN_HF}" \
      --num-shards "${NUM_SHARDS}" \
      --save-workers "${SAVE_WORKERS}" \
      >"${log_file}" 2>&1

   "${SLIME_DIR}/.slime_venv/bin/python3" - "${output_dir}" "${ORIGIN_HF}" "${NUM_SHARDS}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
origin = Path(sys.argv[2])
expected = int(sys.argv[3])
index_path = output / "model.safetensors.index.json"
with index_path.open() as f:
    index = json.load(f)
referenced = set(index["weight_map"].values())
actual = {p.name for p in output.glob("model-*-of-*.safetensors")}
weight_names = set(index["weight_map"])
layer_ids = {
    int(name.split(".")[2])
    for name in weight_names
    if name.startswith("model.layers.")
}
assert len(referenced) == expected, (len(referenced), expected)
assert len(actual) == expected, (len(actual), expected)
assert referenced == actual, (referenced - actual, actual - referenced)
assert all((output / name).stat().st_size > 0 for name in referenced)
assert (output / "config.json").is_file()
assert (output / "tokenizer.json").read_bytes() == (origin / "tokenizer.json").read_bytes()
assert layer_ids == set(range(47)), (min(layer_ids), max(layer_ids), len(layer_ids))
assert not any(name.startswith("model.layers.47.") for name in weight_names)
print(
    f"validated {output}: {expected} non-empty shards, index and files agree, "
    "tokenizer matches origin, model layers are exactly 0..46"
)
PY
   echo "[$(date -Is)] completed ${iter_name}"
done
