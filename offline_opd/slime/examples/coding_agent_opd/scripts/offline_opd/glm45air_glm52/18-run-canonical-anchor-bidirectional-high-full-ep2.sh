#!/bin/bash
set -euo pipefail

# Full-data continuation of the validated High-reasoning canonical alignment:
#   * reuse the immutable 4093-sample GLM-4.5-Air Phase-1 rollout;
#   * reuse the already-scored 1280-sample High Phase-2 subset by symlink;
#   * score only missing source manifests with the GLM-5.2 dense Teacher;
#   * train two epochs with 0.5 forward KL + 0.5 reverse KL, while anchoring
#     every non-exact Student/Teacher row to the rollout Student distribution.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CALLER_LOAD_PATH="${LOAD_PATH:-}"
source "${SCRIPT_DIR}/common.sh"

SOURCE_PHASE1="${SOURCE_PHASE1:-/path/to/offline_opd/glm45air-glm52-dynamic-t1k-s128-20260819_001719}"
REUSE_PHASE2="${REUSE_PHASE2:-/path/to/offline_opd/glm45air-glm52-canonical-anchor-bidir-effort-high-f0.5-r0.5-a1-n1280-s20-20260827_123616}"
if [[ -z "${RUN_ROOT:-}" ]]; then
   RUN_ROOT="${OFFLINE_OPD_BASE}/glm45air-glm52-canonical-anchor-bidir-high-full4093-f0.5-r0.5-a1-ep2-gbs64-$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
fi
for path in "${SOURCE_PHASE1}" "${REUSE_PHASE2}" "${RUN_ROOT}"; do
   [[ "${path}" == "${OFFLINE_OPD_BASE}"/* ]] || {
      echo "all experiment paths must be under ${OFFLINE_OPD_BASE}: ${path}" >&2
      exit 1
   }
done
[[ -d "${SOURCE_PHASE1}/dump" && -f "${SOURCE_PHASE1}/.phase1.complete" ]] || {
   echo "source Phase 1 is incomplete: ${SOURCE_PHASE1}" >&2
   exit 1
}
[[ -d "${REUSE_PHASE2}/scored/samples" && -d "${REUSE_PHASE2}/scored/source_manifests" \
   && -f "${REUSE_PHASE2}/.phase2.complete" ]] || {
   echo "reused High Phase 2 is incomplete: ${REUSE_PHASE2}" >&2
   exit 1
}

mkdir -p "${RUN_ROOT}/scored/samples" "${RUN_ROOT}/scored/source_manifests"
link_once() {
   local source="$1" target="$2"
   if [[ -L "${target}" ]]; then
      [[ "$(readlink -f "${target}")" == "$(readlink -f "${source}")" ]] || {
         echo "existing symlink points elsewhere: ${target}" >&2
         return 1
      }
      return
   fi
   [[ ! -e "${target}" ]] || { echo "refusing to replace ${target}" >&2; return 1; }
   ln -s "${source}" "${target}"
}
link_once "${SOURCE_PHASE1}/dump" "${RUN_ROOT}/dump"
link_once "${SOURCE_PHASE1}/.phase1.complete" "${RUN_ROOT}/.phase1.complete"
link_once "${SOURCE_PHASE1}/glm45air_to_glm52_token_ids.json" \
   "${RUN_ROOT}/glm45air_to_glm52_token_ids.json"

# A source manifest is the scorer's transactional completion record. The reused
# subset renamed its files to rollout_0..1279, so those manifest names MUST NOT
# be copied verbatim into the full run. Recover each original rollout id from
# the subset dump symlink (for example subset rollout_0 -> full rollout_2152),
# and publish the manifest under that original name. Sample payload names use
# the stable Sample.index and can be linked directly. Deliberately do not reuse
# the old aggregate summary or rebatches: finalization rebuilds them for 4093.
"${SLIME_DIR}/.slime_venv/bin/python3" - \
   "${SOURCE_PHASE1}" "${REUSE_PHASE2}" "${RUN_ROOT}" <<'PY'
import json
import os
import sys
from pathlib import Path

phase1 = Path(sys.argv[1]).resolve()
reuse = Path(sys.argv[2]).resolve()
run = Path(sys.argv[3]).resolve()
sample_out = run / "scored" / "samples"
manifest_out = run / "scored" / "source_manifests"
sample_out.mkdir(parents=True, exist_ok=True)
manifest_out.mkdir(parents=True, exist_ok=True)

sample_sources = sorted((reuse / "scored" / "samples").glob("sample_*.pt"))
subset_inputs = sorted((reuse / "dump").glob("rollout_*.pt"))
if len(sample_sources) != 1280 or len(subset_inputs) != 1280:
    raise SystemExit(
        f"expected 1280 reusable samples/inputs, got "
        f"{len(sample_sources)}/{len(subset_inputs)}"
    )

for source in sample_sources:
    target = sample_out / source.name
    if target.is_symlink():
        if target.resolve() != source.resolve():
            raise SystemExit(f"sample symlink points elsewhere: {target}")
    elif target.exists():
        raise SystemExit(f"refusing to replace regular sample: {target}")
    else:
        target.symlink_to(source)

expected_full_dump = (phase1 / "dump").resolve()
published = set()
for subset_input in subset_inputs:
    if not subset_input.is_symlink():
        raise SystemExit(f"subset input is not a symlink: {subset_input}")
    original_input = subset_input.resolve()
    if original_input.parent != expected_full_dump:
        raise SystemExit(f"subset input escapes full Phase 1 dump: {subset_input}")
    subset_id = int(subset_input.stem.removeprefix("rollout_"))
    original_id = int(original_input.stem.removeprefix("rollout_"))
    source_manifest = reuse / "scored" / "source_manifests" / f"rollout_{subset_id}.json"
    payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if int(payload.get("source_rollout_id", -1)) != subset_id:
        raise SystemExit(f"unexpected source rollout id in {source_manifest}")
    rows = payload.get("samples")
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"empty reusable source manifest: {source_manifest}")
    for row in rows:
        relative = row.get("path") if isinstance(row, dict) else None
        if not isinstance(relative, str) or not (run / "scored" / relative).is_file():
            raise SystemExit(f"missing reused sample for {source_manifest}: {relative!r}")
    payload["source_rollout_id"] = original_id
    payload["reused_from_subset_rollout_id"] = subset_id
    target = manifest_out / f"rollout_{original_id}.json"
    if target.name in published:
        raise SystemExit(f"duplicate original rollout mapping: {target.name}")
    published.add(target.name)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, target)

if len(published) != 1280:
    raise SystemExit(f"expected 1280 mapped manifests, got {len(published)}")
print(f"reused {len(sample_sources)} samples under {len(published)} original rollout ids")
PY

export RUN_ROOT
export OPD_TOKEN_ID_BRIDGE="${RUN_ROOT}/glm45air_to_glm52_token_ids.json"
export LOAD_PATH="${CALLER_LOAD_PATH:-/path/to/model/<org>/GLM-4.5-Air_torch_dist_ropefix}"
export OFFLINE_OPD_LOSS_TYPE=cached_teacher_bidirectional_topk_kl_with_student_anchor

export OPD_TEACHER_NATIVE_REASONING_PREFIX=0
export OPD_TEACHER_CANONICAL_TEMPLATE=1
export OPD_TEACHER_REASONING_EFFORT=high
export OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS=0
export OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE=0
export OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT=0
export TEACHER_ATTENTION_MODE="${TEACHER_ATTENTION_MODE:-dense}"
export TEACHER_PREWARM_WEIGHTS=0
unset TEACHER_PREWARM_CONCURRENCY TEACHER_MODEL_LOADER_THREADS
export SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S="${SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S:-7200}"
export TEACHER_WAIT_TIMEOUT_SEC="${TEACHER_WAIT_TIMEOUT_SEC:-7200}"
export TEACHER_PORT="${TEACHER_PORT:-15141}"
export TEACHER_NCCL_PORT_BASE="${TEACHER_NCCL_PORT_BASE:-25000}"
export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-4}"
export SCORE_FILE_CONCURRENCY="${SCORE_FILE_CONCURRENCY:-4}"
export PHASE2_SAMPLES_PER_FILE=1
export PHASE2_BALANCE_MANIFESTS=0
export PHASE2_ROLLOUT_ID_FROM_FILENAME=1

export OPD_FORWARD_KL_COEF=0.5
export OPD_REVERSE_KL_COEF=0.5
export OPD_STUDENT_ANCHOR_KL_COEF="${OPD_STUDENT_ANCHOR_KL_COEF:-1.0}"
export TRAIN_EPOCHS=2 EPOCH_SHUFFLE=1
unset TRAIN_STEPS
export ROLLOUT_BATCH_SIZE=64 NUM_STEPS_PER_ROLLOUT=1 GLOBAL_BATCH_SIZE=64
export TP_SIZE=8 PP_SIZE=2 CP_SIZE=2 EXPERT_MP_SIZE=8 EXPERT_TP_SIZE=1
export MAX_CONTEXT_LEN=$((128 * 1024)) MAX_TOKENS_PER_GPU=$((64 * 1024))
export LOG_PROBS_CHUNK_SIZE=4096 OPD_TEACHER_CACHE_H2D_ROWS=512
export SAVE_INTERVAL=20 PREFETCH_ROLLOUT_DATA=1 OFFLINE_OPD_TOPK_OVERLAP=1
export OFFLINE_OPD_BALANCE_DATA=0 DISTRIBUTED_TIMEOUT_MINUTES=60
# This run deliberately reuses 1280 immutable Phase-2 sample artifacts through
# symlinks into REUSE_PHASE2.  Keep the rebatcher's stricter default for every
# other experiment and opt in only here.
export OFFLINE_OPD_ALLOW_EXTERNAL_SAMPLE_SYMLINKS=1
export USE_WANDB="${USE_WANDB:-1}" WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
export WANDB_GROUP="${WANDB_GROUP:-offline-opd-glm45air-glm52-canonical-anchor-bidir-high-full}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "${RUN_ROOT}")}"

if [[ ! -f "${RUN_ROOT}/experiment.env" ]]; then
   reusable_samples="$(find -L "${RUN_ROOT}/scored/samples" -maxdepth 1 -type f -name 'sample_*.pt' | wc -l)"
   reusable_manifests="$(find -L "${RUN_ROOT}/scored/source_manifests" -maxdepth 1 -type f -name 'rollout_*.json' | wc -l)"
   [[ "${reusable_samples}" == 1280 && "${reusable_manifests}" == 1280 ]] || {
      echo "expected 1280 reusable samples/manifests, got ${reusable_samples}/${reusable_manifests}" >&2
      exit 1
   }
   {
      printf 'RUN_ROOT=%s\nSOURCE_PHASE1=%s\nREUSE_PHASE2=%s\n' \
         "${RUN_ROOT}" "${SOURCE_PHASE1}" "${REUSE_PHASE2}"
      printf 'PHASE1_SAMPLES=4093\nPHASE2_REUSED_SAMPLES=1280\nPHASE2_REMAINING_SAMPLES=2813\n'
      printf 'LOSS=cached_teacher_bidirectional_topk_kl_with_student_anchor\n'
      printf 'TEACHER_TEMPLATE=official_glm52_canonical\nTEACHER_REASONING_EFFORT=high\n'
      printf 'POSITION_POLICY=exact_1to1_teacher_else_rollout_student_anchor\n'
      printf 'TEACHER_FORWARD_COEF=0.5\nTEACHER_REVERSE_COEF=0.5\nSTUDENT_ANCHOR_COEF=%s\n' \
         "${OPD_STUDENT_ANCHOR_KL_COEF}"
      printf 'TEACHER_RAW_CACHE_TOPK=1000\nSUPPORT_TOPK=128\nANCHOR_TAIL_BUCKET=1\n'
      printf 'PHASE2_ATTENTION_MODE=%s\nPHASE2_BALANCE=0\n' "${TEACHER_ATTENTION_MODE}"
      printf 'PHASE2_SCORE_CONCURRENCY=%s\nPHASE2_FILE_CONCURRENCY=%s\nTEACHER_MAX_RUNNING_REQUESTS=1\n' \
         "${SCORE_CONCURRENCY}" "${SCORE_FILE_CONCURRENCY}"
      printf 'PHASE3_GBS=64\nPHASE3_EPOCHS=2\nPHASE3_SHUFFLE=1\nPHASE3_BALANCE=0\n'
      printf 'PHASE3_TP=8\nPHASE3_PP=2\nPHASE3_CP=2\nPHASE3_EP=8\nPHASE3_ETP=1\n'
      printf 'SAVE_INTERVAL=20\nDISTRIBUTED_TIMEOUT_MINUTES=60\nLR=1e-6\nLR_SCHEDULE=constant\n'
   } > "${RUN_ROOT}/experiment.env"
fi

if [[ "${1:-}" != --worker ]]; then
   nohup setsid bash "$0" --worker >> "${RUN_ROOT}/pipeline.log" 2>&1 < /dev/null &
   printf '%s\n' "$!" > "${RUN_ROOT}/pipeline.pid"
   echo "canonical anchor High full4093/ep2 pipeline started: pid=$!, run=${RUN_ROOT}"
   exit 0
fi

if [[ ! -f "${RUN_ROOT}/.phase2.complete" ]]; then
   bash "${SCRIPT_DIR}/02-phase2-teacher.sh" --foreground
fi
if [[ ! -f "${RUN_ROOT}/.phase3.complete" ]]; then
   bash "${SCRIPT_DIR}/03-phase3-train.sh" --foreground
fi
printf 'completed=%s\n' "$(date -Is)" > "${RUN_ROOT}/.pipeline.complete"
