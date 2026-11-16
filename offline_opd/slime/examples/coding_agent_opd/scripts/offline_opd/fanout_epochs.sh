#!/bin/bash
set -euo pipefail
# ==============================================================================
# OFFLINE OPD — PHASE 2.5: build the fixed-size batch stream consumed by Phase 3.
# When the requested batch equals the original manifest batch, retain the old
# zero-copy symlink fanout exactly. Otherwise, reuse the per-sample Phase-2
# artifacts through lightweight manifests and cut a continuous multi-epoch
# stream into the requested batch size.
#
# Usage:
#   bash fanout_epochs.sh <scored_dir> <num_epochs> [<fanout_dir>]
#
# ROLLOUT_BATCH_SIZE optionally selects a Phase-3 batch different from the
# Phase-2 manifest size. With SHUFFLE=1, each epoch deterministically permutes
# the original manifest blocks before they are flattened and rebatched.
# ==============================================================================

SCORED_DIR="${1:?usage: fanout_epochs.sh <scored_dir> <num_epochs> [fanout_dir]}"
E="${2:?usage: fanout_epochs.sh <scored_dir> <num_epochs> [fanout_dir]}"
FANOUT_DIR="${3:-${SCORED_DIR}/epochs_x${E}}"
SHUFFLE="${SHUFFLE:-0}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
PYTHON="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}/bin/python3"
[[ -x "${PYTHON}" ]] || PYTHON=python3
OWNER_MARKER=".offline_opd_rebatch_v1.json"

[[ "${E}" =~ ^[1-9][0-9]*$ ]] || { echo "num_epochs must be a positive integer: ${E}" >&2; exit 1; }
[[ "${SHUFFLE}" == 0 || "${SHUFFLE}" == 1 ]] || { echo "SHUFFLE must be 0 or 1" >&2; exit 1; }
[[ "${OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS:-0}" == 0 || \
   "${OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS:-0}" == 1 ]] || {
   echo "OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS must be 0 or 1" >&2
   exit 1
}

SCORED_DIR="$(cd "${SCORED_DIR}" && pwd)"
mkdir -p "${FANOUT_DIR}"
FANOUT_DIR="$(cd "${FANOUT_DIR}" && pwd)"

# This is a generated directory. Refuse to replace unknown regular files, but
# permit a previous rebatched output carrying our ownership marker to be rebuilt.
if find "${FANOUT_DIR}" -maxdepth 1 -type f -name 'rollout_*.pt' -print -quit | grep -q .; then
   if [[ ! -f "${FANOUT_DIR}/${OWNER_MARKER}" ]]; then
      echo "fanout directory contains an unowned regular rollout file; refusing to replace data: ${FANOUT_DIR}" >&2
      exit 1
   fi
   find "${FANOUT_DIR}" -maxdepth 1 -type f -name 'rollout_*.pt' -delete
fi
find "${FANOUT_DIR}" -maxdepth 1 -type l -name 'rollout_*.pt' -delete
rm -f "${FANOUT_DIR}/${OWNER_MARKER}"

# Count the distinct base files rollout_0.pt .. rollout_{M-1}.pt (must be contiguous).
M=0
while [[ -f "${SCORED_DIR}/rollout_${M}.pt" ]]; do
   M=$(( M + 1 ))
done
if [[ "${M}" -eq 0 ]]; then
   echo "no rollout_0.pt.. under ${SCORED_DIR}" >&2
   exit 1
fi
for path in "${SCORED_DIR}"/rollout_*.pt; do
   name="${path##*/}"
   index="${name#rollout_}"
   index="${index%.pt}"
   if [[ "${index}" =~ ^[0-9]+$ ]] && (( index >= M )); then
      echo "non-contiguous rollout files: ${name} exists after first gap at rollout_${M}.pt" >&2
      exit 1
   fi
done

# Inspect only the small batch manifests. The legacy bundled format remains
# supported by the unchanged symlink path, but sample-level rebatching requires
# modern Phase-2 manifests.
read -r INSPECTED_M DATASET_SIZE SOURCE_BATCH MODERN_MANIFESTS <<< "$("${PYTHON}" - "${SCORED_DIR}" <<'PY'
import re, sys, torch
from pathlib import Path

directory = Path(sys.argv[1])
pattern = re.compile(r"rollout_(\d+)\.pt$")
indexed = sorted((int(m.group(1)), p) for p in directory.glob("rollout_*.pt") if (m := pattern.fullmatch(p.name)))
sizes = []
modern = True
for _, path in indexed:
    blob = torch.load(path, weights_only=False)
    if isinstance(blob, dict) and blob.get("format") == "offline_opd_sample_manifest_v1":
        sizes.append(len(blob.get("sample_files", [])))
    else:
        modern = False
        sizes.append(len(blob.get("samples", [])) if isinstance(blob, dict) else 0)
if not sizes or any(size <= 0 for size in sizes) or len(set(sizes)) != 1:
    raise SystemExit(f"source rollout files must have one uniform positive batch size: {sizes}")
print(len(indexed), sum(sizes), sizes[0], int(modern))
PY
)"
(( INSPECTED_M == M )) || { echo "source rollout count changed during inspection" >&2; exit 1; }
TARGET_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-${SOURCE_BATCH}}"
[[ "${TARGET_BATCH_SIZE}" =~ ^[1-9][0-9]*$ ]] || {
   echo "ROLLOUT_BATCH_SIZE must be a positive integer: ${TARGET_BATCH_SIZE}" >&2
   exit 1
}

# Exact compatibility path: an unchanged batch that divides one epoch never
# crosses an epoch boundary and keeps the original symlink layout byte-for-byte.
EXPECTED_LINK_ROLLOUTS=$(( M * E ))
if (( TARGET_BATCH_SIZE == SOURCE_BATCH && DATASET_SIZE % TARGET_BATCH_SIZE == 0 )) && \
   [[ -z "${TRAIN_STEPS:-}" || "${TRAIN_STEPS}" -eq "${EXPECTED_LINK_ROLLOUTS}" ]]; then
   echo "[fanout] base M=${M} files in ${SCORED_DIR}; epochs E=${E} -> ${M}*${E}=$(( M * E )) links in ${FANOUT_DIR} (shuffle=${SHUFFLE}, batch=${TARGET_BATCH_SIZE}, cross_epoch=0)"

   # Deterministic per-epoch permutation of 0..M-1 (Python for a real shuffle; identity otherwise).
   gen_perm() {  # $1 = epoch seed
      if [[ "${SHUFFLE}" -eq 1 ]]; then
         "${PYTHON}" -c "import random,sys; m=int(sys.argv[1]); s=int(sys.argv[2]); r=random.Random(s); a=list(range(m)); r.shuffle(a); print(' '.join(map(str,a)))" "${M}" "$1"
      else
         # space-separated on ONE line: `read -ra` below stops at the first newline.
         echo $(seq 0 $(( M - 1 )))
      fi
   }

   idx=0
   for (( e = 0; e < E; e++ )); do
      read -ra perm <<< "$(gen_perm "${e}")"
      for (( j = 0; j < M; j++ )); do
         src="${SCORED_DIR}/rollout_${perm[$j]}.pt"
         dst="${FANOUT_DIR}/rollout_${idx}.pt"
         ln -sfn "${src}" "${dst}"
         idx=$(( idx + 1 ))
      done
   done

   echo "[fanout] wrote ${idx} symlinks. Phase 3: --load-debug-rollout-data '${FANOUT_DIR}/rollout_{rollout_id}.pt' --num-rollout ${idx}"
   exit 0
fi

(( MODERN_MANIFESTS == 1 )) || {
   echo "ROLLOUT_BATCH_SIZE=${TARGET_BATCH_SIZE} requires Phase-2 sample manifests; legacy bundled files can only use batch=${SOURCE_BATCH}" >&2
   exit 1
}
# Mark the directory before writing so an interrupted generated output can be
# safely recognized and rebuilt on the next launch.
printf '{"status":"building"}\n' > "${FANOUT_DIR}/${OWNER_MARKER}"
REBATCH_ARGS=(
   --scored-dir "${SCORED_DIR}"
   --output-dir "${FANOUT_DIR}"
   --epochs "${E}"
   --batch-size "${TARGET_BATCH_SIZE}"
   --shuffle "${SHUFFLE}"
)
if [[ "${OFFLINE_OPD_ALLOW_SAMPLE_SUBSET:-0}" == 1 ]]; then
   REBATCH_ARGS+=(--allow-noncontiguous-sample-indices)
fi
if [[ "${OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS:-0}" == 1 ]]; then
   REBATCH_ARGS+=(--allow-external-sample-symlinks)
fi
if [[ -n "${TRAIN_STEPS:-}" ]]; then
   [[ "${TRAIN_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
      echo "TRAIN_STEPS must be a positive integer: ${TRAIN_STEPS}" >&2
      exit 1
   }
   REBATCH_ARGS+=(--num-rollouts "${TRAIN_STEPS}")
fi
"${PYTHON}" -u "${SCRIPT_DIR}/rebatch_scored_epochs.py" \
   "${REBATCH_ARGS[@]}"
