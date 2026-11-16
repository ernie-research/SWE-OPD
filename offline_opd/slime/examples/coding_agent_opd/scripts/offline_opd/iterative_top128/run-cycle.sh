#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/common.sh"

CYCLE="${1:?usage: run-cycle.sh <cycle-number>}"
[[ "${CYCLE}" =~ ^[1-9][0-9]*$ ]] || { echo "invalid cycle: ${CYCLE}" >&2; exit 1; }
(( CYCLE <= ITERATIVE_CYCLES )) || {
   echo "cycle ${CYCLE} exceeds ITERATIVE_CYCLES=${ITERATIVE_CYCLES}" >&2
   exit 1
}
: "${RUN_ROOT:?RUN_ROOT is required}"
validate_run_root "${RUN_ROOT}"

CYCLE_NAME="$(cycle_name "${CYCLE}")"
CYCLE_DIR="$(cycle_dir "${CYCLE}")"
mkdir -p "${CYCLE_DIR}"
COMPLETE_MARKER="${CYCLE_DIR}/.cycle.complete"
# A later cycle advances the experiment-wide checkpoint pointer.  Therefore a
# completed earlier cycle must be skipped before require_cycle_input validates
# that pointer against the immediately preceding boundary.  Doing this check
# later made a normal retry at cycle 3 reject completed cycle 2 because the
# shared pointer had already advanced from iter 19 to iter 39.
if [[ -f "${COMPLETE_MARKER}" ]]; then
   echo "${CYCLE_NAME} already complete: ${CYCLE_DIR}"
   exit 0
fi
require_cycle_input "${CYCLE}"
PHASE3_USE_WANDB="${USE_WANDB:-1}"

export RUN_TAG="$(basename "${RUN_ROOT}")-${CYCLE_NAME}"
export HF_CHECKPOINT="${CYCLE_INPUT_HF}"
export LOAD_PATH="${CYCLE_INPUT_MEGATRON}"
export DUMP_DIR="${CYCLE_DIR}/dump"
export IN_DIR="${DUMP_DIR}"
export OUT_DIR="${CYCLE_DIR}/scored"
export SAVE_PATH="${RUN_ROOT}/checkpoints"

# This experiment lives on a shared PFS whose metadata can be briefly stale
# across nodes.  Concurrent `mkdir -p` from all MPI ranks can therefore report
# EEXIST as an error even though another rank just created the directory.  The
# isolated orchestrator creates every shared phase directory once, before
# mpirun, and gives the mount a short visibility window.  Generic offline OPD
# launchers remain unchanged.
precreate_shared_dirs() {
   local dir
   for dir in "$@"; do
      mkdir -p "${dir}"
      [[ -d "${dir}" ]] || {
         echo "shared directory is not visible after creation: ${dir}" >&2
         return 1
      }
   done
   sleep "${PFS_METADATA_SETTLE_SEC:-3}"
}

precreate_shared_dirs "${CYCLE_DIR}" "${DUMP_DIR}" "${OUT_DIR}"

RUNNING_MARKER="${CYCLE_DIR}/.cycle.running"
FAILED_MARKER="${CYCLE_DIR}/.cycle.failed"
printf 'pid=%s host=%s started=%s input=%s\n' \
   "$$" "$(hostname)" "$(date -Is)" "${CYCLE_INPUT_LABEL}" > "${RUNNING_MARKER}"
rm -f "${FAILED_MARKER}"
cycle_exit() {
   local rc=$?
   if (( rc != 0 )); then
      printf 'rc=%s failed=%s cycle=%s\n' "${rc}" "$(date -Is)" "${CYCLE}" > "${FAILED_MARKER}"
      rm -f "${RUNNING_MARKER}"
   fi
}
trap cycle_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cat > "${CYCLE_DIR}/cycle.env" <<EOF
CYCLE=${CYCLE}
CYCLE_NAME=${CYCLE_NAME}
INPUT_LABEL=${CYCLE_INPUT_LABEL}
INPUT_HF=${CYCLE_INPUT_HF}
INPUT_MEGATRON=${CYCLE_INPUT_MEGATRON}
LOSS_TYPE=${OFFLINE_OPD_LOSS_TYPE}
TOPK=${OPD_ID_TOPK}
EPOCHS=${EPOCHS_PER_CYCLE}
EOF

echo "[$(date -Is)] ${CYCLE_NAME} begins from ${CYCLE_INPUT_LABEL}"

# ---------------------------------------------------------------------------
# Phase 1: fresh full-dataset rollout from this cycle's input HF model.
# ---------------------------------------------------------------------------
if [[ ! -f "${CYCLE_DIR}/.phase1.complete" ]]; then
   export OFFLINE_OPD_CONCURRENCY="${OFFLINE_OPD_CONCURRENCY:-500}"
   export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-50}"
   export PREFERRED_ROLLOUT_BATCH_SIZE="${PREFERRED_ROLLOUT_BATCH_SIZE:-50}"
   export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-2}"
   # Give every refresh its own legal port range.  A failed SGLang engine can
   # leave a rendezvous listener behind briefly; the compact stride stays below
   # slime's hard 32000 base-port ceiling across all five cycles.
   ROLLOUT_ENGINE_PORT_BASE_ROOT="${SLIME_ROLLOUT_ENGINE_PORT_BASE_ROOT:-${SLIME_ROLLOUT_ENGINE_PORT_BASE:-30000}}"
   ROLLOUT_ENGINE_PORT_STRIDE="${SLIME_ROLLOUT_ENGINE_PORT_STRIDE:-200}"
   export SLIME_ROLLOUT_ENGINE_PORT_BASE="$(( ROLLOUT_ENGINE_PORT_BASE_ROOT + (CYCLE - 1) * ROLLOUT_ENGINE_PORT_STRIDE ))"
   export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024 * 128))}"
   export MAX_GEN_LEN="${MAX_GEN_LEN:-$((1024 * 16))}"
   export OFFLINE_OPD_SAVE_STUDENT_TOPK_LOGPROBS=1
   export SGLANG_OUTPUT_TOP_LOGPROBS_B64=true
   export USE_WANDB=0
   # Let run-student_dump discover a validated partial prefix on retry.
   unset START_ROLLOUT_ID
   run_multinode_phase phase1 "${CYCLE_DIR}/logs-phase1"
   write_marker "${CYCLE_DIR}/.phase1.complete" \
      "cycle=${CYCLE} model=${HF_CHECKPOINT} dump=${DUMP_DIR}"
else
   echo "[$(date -Is)] ${CYCLE_NAME} phase1 already complete"
fi

# ---------------------------------------------------------------------------
# Phase 2: score the newly generated support exactly once with the fixed teacher.
# ---------------------------------------------------------------------------
if [[ ! -f "${CYCLE_DIR}/.phase2.complete" ]]; then
   export START_ROLLOUT_ID=0
   export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-50}"
   export PHASE2_SAMPLES_PER_FILE="${PHASE2_SAMPLES_PER_FILE:-50}"
   export PHASE2_BALANCE_MANIFESTS="${PHASE2_BALANCE_MANIFESTS:-1}"
   export PHASE2_DYNAMIC_SHARDING=1
   export PHASE2_SAMPLE_OUTPUT=1
   export PHASE1_SHUFFLE=0
   export PHASE1_SHUFFLE_SEED="$(( 42 + CYCLE - 1 ))"
   export PHASE1_SHUFFLE_BALANCE_KEY="${PHASE1_SHUFFLE_BALANCE_KEY:-trainable_tokens}"
   export TEACHER_TP="${TEACHER_TP:-4}"
   export TEACHER_DP="${TEACHER_DP:-2}"
   export TEACHER_GPUS="${TEACHER_GPUS:-0,1,2,3,4,5,6,7}"
   export SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-8}"
   export SCORE_MAX_ATTEMPTS="${SCORE_MAX_ATTEMPTS:-3}"
   export TEACHER_LOGITS_CHUNK="${TEACHER_LOGITS_CHUNK:-2048}"
   export USE_WANDB=0
   SCORE_COORD_TOKEN="${RUN_TAG//\//_}"
   export SCORE_COORD_DIR="${OUT_DIR}/.claims_${SCORE_COORD_TOKEN}"
   precreate_shared_dirs \
      "${OUT_DIR}" \
      "${OUT_DIR}/samples" \
      "${OUT_DIR}/source_manifests" \
      "${SCORE_COORD_DIR}" \
      "${OUT_DIR}/.finalize_${SCORE_COORD_TOKEN}"
   run_multinode_phase phase2 "${CYCLE_DIR}/logs-phase2"
   write_marker "${CYCLE_DIR}/.phase2.complete" \
      "cycle=${CYCLE} input=${DUMP_DIR} scored=${OUT_DIR}"
else
   echo "[$(date -Is)] ${CYCLE_NAME} phase2 already complete"
fi

# ---------------------------------------------------------------------------
# Phase 3: train exactly two epochs on this cycle's freshly scored trajectories.
# ---------------------------------------------------------------------------
# A completed Megatron save can precede the phase marker if the orchestrator is
# interrupted immediately after training.  Recover that boundary instead of
# needlessly submitting the same two epochs again.  This also repairs runs made
# before newline-less Megatron checkpoint pointers were handled correctly.
if [[ ! -f "${CYCLE_DIR}/.phase3.complete" && \
      -r "${SAVE_PATH}/latest_checkpointed_iteration.txt" ]]; then
   RECOVERY_BASE_FILES="$(count_contiguous_rollouts "${OUT_DIR}")"
   RECOVERY_STEPS=$(( RECOVERY_BASE_FILES * EPOCHS_PER_CYCLE ))
   RECOVERY_EXPECTED_ITER=$(( CYCLE * RECOVERY_STEPS - 1 ))
   RECOVERY_ACTUAL_ITER="$(<"${SAVE_PATH}/latest_checkpointed_iteration.txt")"
   RECOVERY_ITER_DIR="$(printf '%s/iter_%07d' "${SAVE_PATH}" "${RECOVERY_EXPECTED_ITER}")"
   if [[ "${RECOVERY_ACTUAL_ITER}" =~ ^[0-9]+$ ]] && \
      (( RECOVERY_ACTUAL_ITER == RECOVERY_EXPECTED_ITER )) && \
      [[ -f "${RECOVERY_ITER_DIR}/common.pt" ]]; then
      printf '%s\n' "${RECOVERY_EXPECTED_ITER}" > "${CYCLE_DIR}/checkpoint.iter.tmp.$$"
      mv "${CYCLE_DIR}/checkpoint.iter.tmp.$$" "${CYCLE_DIR}/checkpoint.iter"
      write_marker "${CYCLE_DIR}/.phase3.complete" \
         "cycle=${CYCLE} epochs=${EPOCHS_PER_CYCLE} steps=${RECOVERY_STEPS} checkpoint=${RECOVERY_ITER_DIR} recovered=1"
      echo "[$(date -Is)] ${CYCLE_NAME} recovered completed phase3 at iter ${RECOVERY_EXPECTED_ITER}"
   fi
fi

if [[ ! -f "${CYCLE_DIR}/.phase3.complete" ]]; then
   BASE_FILES="$(count_contiguous_rollouts "${OUT_DIR}")"
   CYCLE_STEPS=$(( BASE_FILES * EPOCHS_PER_CYCLE ))
   EXPECTED_OFFSET=$(( (CYCLE - 1) * CYCLE_STEPS ))
   EXPECTED_END=$(( EXPECTED_OFFSET + CYCLE_STEPS ))
   EXPECTED_ITER=$(( EXPECTED_END - 1 ))

   # Expand this cycle into two shuffled epochs, then append them to the
   # experiment-wide replay index. Earlier entries keep rollout/checkpoint ids
   # cumulative; checkpoint resume skips them and reads only this cycle's range.
   LOCAL_REPLAY_DIR="${CYCLE_DIR}/replay_x${EPOCHS_PER_CYCLE}"
   SHUFFLE=1 bash "${OFFLINE_OPD_DIR}/fanout_epochs.sh" \
      "${OUT_DIR}" "${EPOCHS_PER_CYCLE}" "${LOCAL_REPLAY_DIR}" \
      > "${CYCLE_DIR}/fanout.log" 2>&1
   LOCAL_REPLAY_FILES="$(count_contiguous_rollouts "${LOCAL_REPLAY_DIR}")"
   (( LOCAL_REPLAY_FILES == CYCLE_STEPS )) || {
      echo "local replay count ${LOCAL_REPLAY_FILES} != expected ${CYCLE_STEPS}" >&2
      exit 1
   }
   GLOBAL_REPLAY_DIR="${RUN_ROOT}/replay"
   mkdir -p "${GLOBAL_REPLAY_DIR}"
   for (( local_id = 0; local_id < CYCLE_STEPS; local_id++ )); do
      global_id=$(( EXPECTED_OFFSET + local_id ))
      src="${LOCAL_REPLAY_DIR}/rollout_${local_id}.pt"
      dst="${GLOBAL_REPLAY_DIR}/rollout_${global_id}.pt"
      if [[ -e "${dst}" && ! -L "${dst}" ]]; then
         echo "refusing to replace regular replay file: ${dst}" >&2
         exit 1
      fi
      ln -sfn "${src}" "${dst}"
   done
   GLOBAL_REPLAY_FILES="$(count_contiguous_rollouts "${GLOBAL_REPLAY_DIR}")"
   (( GLOBAL_REPLAY_FILES == EXPECTED_END )) || {
      echo "global replay count ${GLOBAL_REPLAY_FILES} != expected ${EXPECTED_END}" >&2
      exit 1
   }

   export SCORED_DIR="${GLOBAL_REPLAY_DIR}"
   # The global replay is already epoch-expanded; do not fan it out again.
   export TRAIN_EPOCHS=1
   export EPOCH_SHUFFLE=0
   export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-50}"
   export NUM_STEPS_PER_ROLLOUT=1
   export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-50}"
   export TP_SIZE="${TP_SIZE:-8}"
   export PP_SIZE="${PP_SIZE:-1}"
   export CP_SIZE="${CP_SIZE:-2}"
   export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024 * 128))}"
   export MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$(( MAX_CONTEXT_LEN / CP_SIZE ))}"
   export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-8192}"
   export SAVE_INTERVAL="${SAVE_INTERVAL:-${CYCLE_STEPS}}"
   export PREFETCH_ROLLOUT_DATA="${PREFETCH_ROLLOUT_DATA:-1}"
   export OFFLINE_OPD_MANIFEST_LOAD_WORKERS="${OFFLINE_OPD_MANIFEST_LOAD_WORKERS:-8}"
   # The cumulative replay grows from 20 to 40/60/80/100 files across cycles.
   # Override only the scheduler's derived horizon when loading the prior
   # cycle; optimizer/model state and the consumed step remain restored.
   export OFFLINE_OPD_OVERRIDE_OPT_PARAM_SCHEDULER=1
   # Match the prior Top-128 experiment: overlap enabled, verbose outlier
   # payload logging disabled. The ordinary detached offline metrics remain.
   export OFFLINE_OPD_TOPK_OVERLAP=1
   export OFFLINE_OPD_IMPORTANCE_DEBUG=0
   export USE_WANDB="${PHASE3_USE_WANDB}"
   export WANDB_TEAM="${WANDB_TEAM:-czguocs-baidu}"
   export WANDB_PROJECT="${WANDB_PROJECT:-slime-OPD}"
   export WANDB_GROUP="${WANDB_GROUP_PREFIX:-qwen35-9b-offline-online-k128-refresh${ITERATIVE_CYCLES}x${EPOCHS_PER_CYCLE}}"
   export WANDB_MODE="${WANDB_MODE:-online}"
   export WANDB_RUN_ID="${ITERATIVE_WANDB_RUN_ID:?ITERATIVE_WANDB_RUN_ID is required}"
   export WANDB_RESUME="allow"

   # A shared checkpoint root produces cumulative cycle ends 19/39/59/79/99
   # with the default 10 files per epoch.
   if [[ -r "${SAVE_PATH}/latest_checkpointed_iteration.txt" ]]; then
      export LOAD_PATH="${SAVE_PATH}"
      export OFFLINE_OPD_RESUME=1
      LOADED_ITER="$(<"${SAVE_PATH}/latest_checkpointed_iteration.txt")"
      [[ "${LOADED_ITER}" =~ ^[0-9]+$ ]] || {
         echo "invalid shared checkpoint iteration: ${LOADED_ITER}" >&2
         exit 1
      }
      (( LOADED_ITER < EXPECTED_END )) || {
         echo "checkpoint iteration ${LOADED_ITER} is beyond ${CYCLE_NAME} end ${EXPECTED_ITER}" >&2
         exit 1
      }
      echo "[$(date -Is)] ${CYCLE_NAME} continues after shared checkpoint iter ${LOADED_ITER}"
   else
      (( CYCLE == 1 )) || {
         echo "${CYCLE_NAME} requires the previous cycle checkpoint under ${SAVE_PATH}" >&2
         exit 1
      }
      export LOAD_PATH="${CYCLE_INPUT_MEGATRON}"
      export OFFLINE_OPD_RESUME=0
   fi
   run_multinode_phase phase3 "${CYCLE_DIR}/logs-phase3"

   ACTUAL_ITER="$(<"${SAVE_PATH}/latest_checkpointed_iteration.txt")"
   (( ACTUAL_ITER == EXPECTED_ITER )) || {
      echo "${CYCLE_NAME} ended at iter ${ACTUAL_ITER}; expected ${EXPECTED_ITER}" >&2
      exit 1
   }
   FINAL_ITER_DIR="$(printf '%s/iter_%07d' "${SAVE_PATH}" "${EXPECTED_ITER}")"
   [[ -f "${FINAL_ITER_DIR}/common.pt" ]] || {
      echo "Phase 3 returned without a complete checkpoint: ${FINAL_ITER_DIR}" >&2
      exit 1
   }
   printf '%s\n' "${EXPECTED_ITER}" > "${CYCLE_DIR}/checkpoint.iter.tmp.$$"
   mv "${CYCLE_DIR}/checkpoint.iter.tmp.$$" "${CYCLE_DIR}/checkpoint.iter"
   write_marker "${CYCLE_DIR}/.phase3.complete" \
      "cycle=${CYCLE} epochs=${EPOCHS_PER_CYCLE} steps=${CYCLE_STEPS} checkpoint=${FINAL_ITER_DIR}"
else
   echo "[$(date -Is)] ${CYCLE_NAME} phase3 already complete"
fi

# Phase 4: make the just-trained model directly loadable by the next SGLang rollout.
RUN_ROOT="${RUN_ROOT}" bash "${SCRIPT_DIR}/convert-cycle-to-hf.sh" "${CYCLE}"

FINAL_ITER="$(<"${CYCLE_DIR}/checkpoint.iter")"
FINAL_ITER_DIR="$(printf '%s/iter_%07d' "${SAVE_PATH}" "${FINAL_ITER}")"
write_marker "${COMPLETE_MARKER}" \
   "cycle=${CYCLE} epochs=${EPOCHS_PER_CYCLE} checkpoint=${FINAL_ITER_DIR} hf=${CYCLE_DIR}/hf"
rm -f "${RUNNING_MARKER}" "${FAILED_MARKER}"
trap - EXIT INT TERM
echo "[$(date -Is)] ${CYCLE_NAME} complete: checkpoint=${FINAL_ITER_DIR}, hf=${CYCLE_DIR}/hf"
