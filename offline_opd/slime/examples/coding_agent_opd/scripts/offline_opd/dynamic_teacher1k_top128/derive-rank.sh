#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../../../../../" &>/dev/null && pwd)"
PYTHON="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}/bin/python3"
DERIVE_RANK_OFFSET="${DERIVE_RANK_OFFSET:-0}"
DERIVE_WORLD_SIZE="${DERIVE_WORLD_SIZE_OVERRIDE:-${PADDLE_TRAINERS_NUM:?}}"
DERIVE_RANK=$(( PADDLE_TRAINER_ID + DERIVE_RANK_OFFSET ))
exec "${PYTHON}" -u "${SCRIPT_DIR}/derive_teacher_cache.py" \
   --source-dir "${SOURCE_SCORED_DIR:?}" \
   --output-dir "${TARGET_SCORED_DIR:?}" \
   --source-k 4096 --target-k 1024 --support-k 128 \
   --rows-per-chunk "${DERIVE_ROWS_PER_CHUNK:-512}" \
   --rank "${DERIVE_RANK}" --world-size "${DERIVE_WORLD_SIZE}" \
   --device "${DERIVE_DEVICE:-auto}"
