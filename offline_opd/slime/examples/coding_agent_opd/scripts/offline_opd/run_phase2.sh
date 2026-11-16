#!/bin/bash
set -euo pipefail
set -x
# OFFLINE OPD PHASE 2: one local teacher per node; ranks dynamically claim rollout files.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
SLIME_VENV="${SLIME_VENV:-${SLIME_DIR}/.slime_venv}"
[[ -x "${SLIME_VENV}/bin/python3" ]] || { echo "slime venv missing: ${SLIME_VENV}" >&2; exit 1; }
export PATH="${SLIME_VENV}/bin:${PATH}"
# Keep slime first: Megatron-LM also owns a top-level `examples` tree, which can
# otherwise shadow examples.coding_agent_opd when this script is launched from
# outside the repository.
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_LM_PATH:-/root/Megatron-LM}${PYTHONPATH:+:${PYTHONPATH}}"

IN_DIR="${IN_DIR:?set IN_DIR to the Phase-1 dump dir}"
# Resolve the default before IN_DIR is redirected to its shuffled Phase-1.5
# sibling, preserving the pre-shuffle output location for existing launchers.
OUT_DIR="${OUT_DIR:-${IN_DIR}/scored}"
OPD_ID_TOPK="${OPD_ID_TOPK:-1024}"
OFFLINE_OPD_LOSS_TYPE="${OFFLINE_OPD_LOSS_TYPE:-topk_reverse_kl}"
case "${OFFLINE_OPD_LOSS_TYPE}" in
   topk_reverse_kl|token_reverse_kl|cached_teacher_dynamic_topk_reverse_kl|cached_teacher_bidirectional_topk_kl_with_student_anchor) ;;
   *) echo "unsupported OFFLINE_OPD_LOSS_TYPE=${OFFLINE_OPD_LOSS_TYPE}" >&2; exit 1 ;;
esac
OPD_TEACHER_CACHE_TOPK="${OPD_TEACHER_CACHE_TOPK:-4096}"
OPD_DYNAMIC_SUPPORT_TOPK="${OPD_DYNAMIC_SUPPORT_TOPK:-128}"
TEACHER_DP="${TEACHER_DP:-2}"
SCORE_CONCURRENCY="${SCORE_CONCURRENCY:-$((TEACHER_DP * 2))}"
SCORE_FILE_CONCURRENCY="${SCORE_FILE_CONCURRENCY:-1}"
SCORE_MAX_ATTEMPTS="${SCORE_MAX_ATTEMPTS:-3}"
RANK="${PADDLE_TRAINER_ID:-0}"
NNODES="${PADDLE_TRAINERS_NUM:-1}"
SCORE_RANK="${SCORE_SHARD_INDEX:-${RANK}}"
SCORE_WORLD="${SCORE_NUM_SHARDS:-${NNODES}}"
[[ "${SCORE_RANK}" =~ ^[0-9]+$ && "${SCORE_WORLD}" =~ ^[1-9][0-9]*$ && "${SCORE_RANK}" -lt "${SCORE_WORLD}" ]] || {
   echo "invalid scoring shard ${SCORE_RANK}/${SCORE_WORLD}" >&2
   exit 1
}
TEACHER_PORT="${TEACHER_PORT:-13141}"

# Concurrent mkdir -p calls on the shared PFS can occasionally return EEXIST
# to one rank even though another rank has successfully created the directory.
# Treat an already-materialized directory as success and retry only while it is
# genuinely absent.
ensure_shared_dir() {
   local directory="$1"
   local attempt
   for attempt in {1..10}; do
      if mkdir -p "${directory}" 2>/dev/null || [[ -d "${directory}" ]]; then
         return 0
      fi
      sleep 1
   done
   echo "failed to create shared directory after 10 attempts: ${directory}" >&2
   return 1
}

# The current sample-output path avoids a redundant Phase-1 rewrite: Phase 2
# writes each scored sample once and builds balanced manifests afterward.
# PHASE1_SHUFFLE remains as an opt-in compatibility path for legacy bundled
# outputs only.
PHASE1_SHUFFLE="${PHASE1_SHUFFLE:-0}"
PHASE1_SHUFFLE_SEED="${PHASE1_SHUFFLE_SEED:-42}"
PHASE1_SHUFFLE_BALANCE_KEY="${PHASE1_SHUFFLE_BALANCE_KEY:-trainable_tokens}"
# One scored file is one Phase-3 rollout batch.  With one train step per file
# this equals GBS; for S steps/file set it to GBS*S explicitly.
PHASE2_SAMPLES_PER_FILE="${PHASE2_SAMPLES_PER_FILE:-${ROLLOUT_BATCH_SIZE:-}}"
# 1: deterministically rebalance scored samples by trainable-token length.
# 0: preserve Phase-1 rollout_N batch grouping and in-batch order exactly.
PHASE2_BALANCE_MANIFESTS="${PHASE2_BALANCE_MANIFESTS:-1}"
PHASE2_DYNAMIC_SHARDING="${PHASE2_DYNAMIC_SHARDING:-1}"
PHASE2_SAMPLE_OUTPUT="${PHASE2_SAMPLE_OUTPUT:-1}"
PHASE2_ROLLOUT_ID_FROM_FILENAME="${PHASE2_ROLLOUT_ID_FROM_FILENAME:-0}"
OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS="${OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS:-0}"
OFFLINE_OPD_DIAGNOSTIC_TOPK="${OFFLINE_OPD_DIAGNOSTIC_TOPK:-}"
OPD_TEACHER_NATIVE_REASONING_PREFIX="${OPD_TEACHER_NATIVE_REASONING_PREFIX:-0}"
OPD_TEACHER_CANONICAL_TEMPLATE="${OPD_TEACHER_CANONICAL_TEMPLATE:-0}"
OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS="${OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS:-0}"
OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE="${OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE:-0}"
OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT="${OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT:-0}"
OPD_TEACHER_REASONING_EFFORT="${OPD_TEACHER_REASONING_EFFORT:-}"
[[ "${PHASE1_SHUFFLE}" == 0 || "${PHASE1_SHUFFLE}" == 1 ]] || {
   echo "PHASE1_SHUFFLE must be 0 or 1" >&2
   exit 1
}
[[ "${PHASE1_SHUFFLE_SEED}" =~ ^-?[0-9]+$ ]] || {
   echo "PHASE1_SHUFFLE_SEED must be an integer" >&2
   exit 1
}
case "${PHASE1_SHUFFLE_BALANCE_KEY}" in
   none|trainable_tokens|total_tokens) ;;
   *) echo "PHASE1_SHUFFLE_BALANCE_KEY must be none, trainable_tokens, or total_tokens" >&2; exit 1 ;;
esac
if [[ -n "${PHASE2_SAMPLES_PER_FILE}" && ! "${PHASE2_SAMPLES_PER_FILE}" =~ ^[1-9][0-9]*$ ]]; then
   echo "PHASE2_SAMPLES_PER_FILE must be a positive integer when set" >&2
   exit 1
fi
[[ "${PHASE2_BALANCE_MANIFESTS}" == 0 || "${PHASE2_BALANCE_MANIFESTS}" == 1 ]] || {
   echo "PHASE2_BALANCE_MANIFESTS must be 0 or 1" >&2
   exit 1
}
[[ "${PHASE2_DYNAMIC_SHARDING}" == 0 || "${PHASE2_DYNAMIC_SHARDING}" == 1 ]] || {
   echo "PHASE2_DYNAMIC_SHARDING must be 0 or 1" >&2
   exit 1
}
[[ "${PHASE2_SAMPLE_OUTPUT}" == 0 || "${PHASE2_SAMPLE_OUTPUT}" == 1 ]] || {
   echo "PHASE2_SAMPLE_OUTPUT must be 0 or 1" >&2
   exit 1
}
[[ "${PHASE2_ROLLOUT_ID_FROM_FILENAME}" == 0 || "${PHASE2_ROLLOUT_ID_FROM_FILENAME}" == 1 ]] || {
   echo "PHASE2_ROLLOUT_ID_FROM_FILENAME must be 0 or 1" >&2
   exit 1
}
[[ "${SCORE_FILE_CONCURRENCY}" =~ ^[1-9][0-9]*$ ]] || {
   echo "SCORE_FILE_CONCURRENCY must be a positive integer" >&2
   exit 1
}
[[ "${OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS}" == 0 || "${OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS}" == 1 ]] || {
   echo "OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS must be 0 or 1" >&2
   exit 1
}
[[ "${OPD_TEACHER_NATIVE_REASONING_PREFIX}" == 0 || "${OPD_TEACHER_NATIVE_REASONING_PREFIX}" == 1 ]] || {
   echo "OPD_TEACHER_NATIVE_REASONING_PREFIX must be 0 or 1" >&2
   exit 1
}
[[ "${OPD_TEACHER_CANONICAL_TEMPLATE}" == 0 || "${OPD_TEACHER_CANONICAL_TEMPLATE}" == 1 ]] || {
   echo "OPD_TEACHER_CANONICAL_TEMPLATE must be 0 or 1" >&2
   exit 1
}
[[ "${OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS}" == 0 || "${OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS}" == 1 ]] || {
   echo "OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS must be 0 or 1" >&2
   exit 1
}
[[ "${OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE}" == 0 || "${OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE}" == 1 ]] || {
   echo "OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE must be 0 or 1" >&2
   exit 1
}
[[ "${OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT}" == 0 || "${OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT}" == 1 ]] || {
   echo "OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT must be 0 or 1" >&2
   exit 1
}
case "${OPD_TEACHER_REASONING_EFFORT}" in
   ""|high|max) ;;
   *) echo "OPD_TEACHER_REASONING_EFFORT must be empty, high, or max" >&2; exit 1 ;;
esac
if [[ -n "${OPD_TEACHER_REASONING_EFFORT}" && "${OPD_TEACHER_CANONICAL_TEMPLATE}" != 1 ]]; then
   echo "OPD_TEACHER_REASONING_EFFORT requires canonical Teacher template mode" >&2
   exit 1
fi
if [[ "${OPD_TEACHER_CANONICAL_TEMPLATE}" == 1 && "${OPD_TEACHER_NATIVE_REASONING_PREFIX}" == 1 ]]; then
   echo "canonical Teacher template and native-prefix replay are mutually exclusive" >&2
   exit 1
fi
if [[ "${OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS}" == 1 && "${OPD_TEACHER_NATIVE_REASONING_PREFIX}" != 1 ]]; then
   echo "Student format masking requires native Teacher reasoning-prefix replay" >&2
   exit 1
fi
if [[ "${OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE}" == 1 || "${OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT}" == 1 ]]; then
   [[ "${OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS}" == 1 ]] || {
      echo "tool-template normalization and pre-tool masking require Student format masking" >&2
      exit 1
   }
fi
if [[ "${OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS}" == 1 && "${OFFLINE_OPD_LOSS_TYPE}" != "token_reverse_kl" ]]; then
   echo "OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS=1 requires token_reverse_kl" >&2
   exit 1
fi
if [[ "${OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS}" == 1 && ! "${OFFLINE_OPD_DIAGNOSTIC_TOPK}" =~ ^[1-9][0-9]*$ ]]; then
   echo "OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS=1 requires a positive OFFLINE_OPD_DIAGNOSTIC_TOPK" >&2
   exit 1
fi
if [[ "${PHASE2_SAMPLE_OUTPUT}" -eq 1 && -z "${PHASE2_SAMPLES_PER_FILE}" ]]; then
   echo "PHASE2_SAMPLES_PER_FILE is required with PHASE2_SAMPLE_OUTPUT=1" >&2
   exit 1
fi
if [[ "${PHASE2_SAMPLE_OUTPUT}" -eq 1 && "${PHASE1_SHUFFLE}" -eq 1 ]]; then
   echo "PHASE1_SHUFFLE=1 would add a redundant full-data pass; disable it with sample-output Phase 2" >&2
   exit 1
fi
if [[ "${PHASE1_SHUFFLE}" -eq 1 ]]; then
   ORIGINAL_IN_DIR="${IN_DIR%/}"
   SHUFFLE_BATCH_TAG="${PHASE2_SAMPLES_PER_FILE:-source}"
   PHASE1_SHUFFLED_DIR="${PHASE1_SHUFFLED_DIR:-${ORIGINAL_IN_DIR}_shuffled_seed${PHASE1_SHUFFLE_SEED}_${PHASE1_SHUFFLE_BALANCE_KEY}_b${SHUFFLE_BATCH_TAG}}"
   COORD_TOKEN="${RUN_TAG:-phase2}"
   COORD_TOKEN="${COORD_TOKEN//\//_}"
   READY_MARKER="${PHASE1_SHUFFLED_DIR}.ready.${COORD_TOKEN}"
   FAILED_MARKER="${PHASE1_SHUFFLED_DIR}.failed.${COORD_TOKEN}"
   if [[ "${RANK}" -eq 0 ]]; then
      rm -f "${READY_MARKER}" "${FAILED_MARKER}"
      SHUFFLE_ARGS=(
         --input-dir "${ORIGINAL_IN_DIR}"
         --output-dir "${PHASE1_SHUFFLED_DIR}"
         --seed "${PHASE1_SHUFFLE_SEED}"
         --balance-key "${PHASE1_SHUFFLE_BALANCE_KEY}"
      )
      if [[ -n "${PHASE2_SAMPLES_PER_FILE}" ]]; then
         SHUFFLE_ARGS+=(--samples-per-file "${PHASE2_SAMPLES_PER_FILE}")
      fi
      if python3 -u "${SCRIPT_DIR}/shuffle_phase1.py" "${SHUFFLE_ARGS[@]}"; then
         printf 'ready seed=%s input=%s output=%s\n' \
            "${PHASE1_SHUFFLE_SEED}" "${ORIGINAL_IN_DIR}" "${PHASE1_SHUFFLED_DIR}" \
            > "${READY_MARKER}.tmp.$$"
         mv "${READY_MARKER}.tmp.$$" "${READY_MARKER}"
      else
         rc=$?
         printf 'failed rc=%s seed=%s input=%s output=%s\n' \
            "${rc}" "${PHASE1_SHUFFLE_SEED}" "${ORIGINAL_IN_DIR}" "${PHASE1_SHUFFLED_DIR}" \
            > "${FAILED_MARKER}.tmp.$$"
         mv "${FAILED_MARKER}.tmp.$$" "${FAILED_MARKER}"
         exit "${rc}"
      fi
   else
      waited=0
      timeout="${PHASE1_SHUFFLE_WAIT_TIMEOUT_SEC:-7200}"
      [[ "${timeout}" =~ ^[1-9][0-9]*$ ]] || {
         echo "PHASE1_SHUFFLE_WAIT_TIMEOUT_SEC must be a positive integer" >&2
         exit 1
      }
      until [[ -f "${READY_MARKER}" ]]; do
         if [[ -f "${FAILED_MARKER}" ]]; then
            echo "[phase2 rank=${RANK}] Phase-1.5 shuffle failed: $(cat "${FAILED_MARKER}")" >&2
            exit 1
         fi
         if (( waited >= timeout )); then
            echo "[phase2 rank=${RANK}] timed out waiting ${timeout}s for ${READY_MARKER}" >&2
            exit 1
         fi
         sleep 5
         waited=$(( waited + 5 ))
      done
      echo "[phase2 rank=${RANK}] Phase-1.5 shuffle ready after ${waited}s"
   fi
   IN_DIR="${PHASE1_SHUFFLED_DIR}"
   echo "[phase2 rank=${RANK}] using globally shuffled Phase-1 input: ${IN_DIR} (seed=${PHASE1_SHUFFLE_SEED}, balance=${PHASE1_SHUFFLE_BALANCE_KEY}, samples_per_file=${SHUFFLE_BATCH_TAG})"
fi

if [[ "${SKIP_SERVE:-0}" -eq 1 ]]; then
   TEACHER_IP="${TEACHER_IP:-127.0.0.1}"
   RM_URL="${RM_URL:-http://${TEACHER_IP}:${TEACHER_PORT}/generate}"
else
   # Every mpirun rank owns a local teacher; never accidentally target an online
   # teacher address inherited from a previous launch shell.
   TEACHER_IP=127.0.0.1
   RM_URL="http://${TEACHER_IP}:${TEACHER_PORT}/generate"
fi

ensure_shared_dir "${OUT_DIR}"
SCORE_COORD_TOKEN="${RUN_TAG:-phase2}"
SCORE_COORD_TOKEN="${SCORE_COORD_TOKEN//\//_}"
SCORE_COORD_DIR="${SCORE_COORD_DIR:-${OUT_DIR}/.claims_${SCORE_COORD_TOKEN}}"
RUN_DIR="${RUN_DIR:-$(dirname "${OUT_DIR}")/logs}"
ensure_shared_dir "${RUN_DIR}"
TEACHER_LOG="${RUN_DIR}/phase2.teacher.rank${RANK}.log"
SCORER_LOG="${RUN_DIR}/phase2.scorer.rank${RANK}.log"
export no_proxy="127.0.0.1,localhost,${TEACHER_IP},${no_proxy:-}"
export NO_PROXY="${no_proxy}"
export OPD_TEACHER_CONCURRENCY="${OPD_TEACHER_CONCURRENCY:-${SCORE_CONCURRENCY}}"
export OPD_TEACHER_TIMEOUT_SEC="${OPD_TEACHER_TIMEOUT_SEC:-7200}"
export OPD_TEACHER_CONNECT_TIMEOUT_SEC="${OPD_TEACHER_CONNECT_TIMEOUT_SEC:-360}"

if [[ -z "${CUDNN_PATH:-}" ]]; then
   CUDNN_PATH="$(python3 -c 'import os,sysconfig; print(os.path.join(sysconfig.get_path("purelib"), "nvidia/cudnn"))')"
fi
if [[ -d "${CUDNN_PATH}/lib" ]]; then
   export LD_LIBRARY_PATH="${CUDNN_PATH}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

TEACHER_PID=""
cleanup() {
   if [[ -n "${TEACHER_PID}" ]] && kill -0 "${TEACHER_PID}" 2>/dev/null; then
      echo "[phase2 rank=${RANK}] stopping owned teacher process group ${TEACHER_PID}"
      kill -TERM -- "-${TEACHER_PID}" 2>/dev/null || true
      for _ in {1..30}; do
         kill -0 "${TEACHER_PID}" 2>/dev/null || break
         sleep 1
      done
      kill -KILL -- "-${TEACHER_PID}" 2>/dev/null || true
      wait "${TEACHER_PID}" 2>/dev/null || true
   fi
}
trap cleanup EXIT INT TERM

if [[ "${SKIP_SERVE:-0}" -ne 1 ]]; then
   : > "${TEACHER_LOG}"
   TEACHER_PORT="${TEACHER_PORT}" TEACHER_DP="${TEACHER_DP}" setsid bash "${SCRIPT_DIR}/serve_teacher.sh" \
      > "${TEACHER_LOG}" 2>&1 &
   TEACHER_PID=$!
   waited=0
   timeout="${TEACHER_WAIT_TIMEOUT_SEC:-1800}"
   until curl -sf "http://${TEACHER_IP}:${TEACHER_PORT}/health_generate" >/dev/null; do
      if ! kill -0 "${TEACHER_PID}" 2>/dev/null; then
         wait "${TEACHER_PID}" || true
         echo "[phase2 rank=${RANK}] teacher exited before becoming healthy" >&2
         tail -n 100 "${TEACHER_LOG}" >&2 || true
         exit 1
      fi
      if (( waited >= timeout )); then
         echo "[phase2 rank=${RANK}] teacher health timeout after ${timeout}s" >&2
         tail -n 100 "${TEACHER_LOG}" >&2 || true
         exit 1
      fi
      sleep 5
      waited=$(( waited + 5 ))
   done
   echo "[phase2 rank=${RANK}] teacher healthy after ${waited}s; log=${TEACHER_LOG}"
fi

echo "[phase2 rank=${RANK}/${NNODES}] scoring shard ${SCORE_RANK}/${SCORE_WORLD}: ${IN_DIR} -> ${OUT_DIR}"
SHARD_ARGS=(--shard-index "${SCORE_RANK}" --num-shards "${SCORE_WORLD}")
if [[ "${PHASE2_DYNAMIC_SHARDING}" -eq 1 ]]; then
   SHARD_ARGS+=(--dynamic-sharding --coord-dir "${SCORE_COORD_DIR}")
fi
if [[ "${PHASE2_SAMPLE_OUTPUT}" -eq 1 ]]; then
   SHARD_ARGS+=(--sample-output)
fi
if [[ "${PHASE2_ROLLOUT_ID_FROM_FILENAME}" -eq 1 ]]; then
   SHARD_ARGS+=(--rollout-id-from-filename)
fi
SCORE_SCHEMA_ARGS=(--opd-loss-type "${OFFLINE_OPD_LOSS_TYPE}")
if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "topk_reverse_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_dynamic_topk_reverse_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_bidirectional_topk_kl_with_student_anchor" ]]; then
   SCORE_SCHEMA_ARGS+=(--require-student-topk-logprobs)
fi
if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_dynamic_topk_reverse_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_bidirectional_topk_kl_with_student_anchor" ]]; then
   SCORE_SCHEMA_ARGS+=(
      --opd-teacher-cache-topk "${OPD_TEACHER_CACHE_TOPK}"
      --opd-dynamic-support-topk "${OPD_DYNAMIC_SUPPORT_TOPK}"
   )
fi
if [[ -n "${OPD_TOKEN_ID_BRIDGE:-}" ]]; then
   SCORE_SCHEMA_ARGS+=(--opd-token-id-bridge "${OPD_TOKEN_ID_BRIDGE}")
fi
if [[ -n "${OPD_TEACHER_TEMPLATE_MODEL:-}" ]]; then
   SCORE_SCHEMA_ARGS+=(--opd-teacher-template-model "${OPD_TEACHER_TEMPLATE_MODEL}")
fi
if [[ "${OPD_TEACHER_NATIVE_REASONING_PREFIX}" == 1 ]]; then
   SCORE_SCHEMA_ARGS+=(--opd-teacher-native-reasoning-prefix)
fi
if [[ "${OPD_TEACHER_CANONICAL_TEMPLATE}" == 1 ]]; then
   SCORE_SCHEMA_ARGS+=(--opd-teacher-canonical-template)
fi
if [[ "${OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS}" == 1 ]]; then
   SCORE_SCHEMA_ARGS+=(--opd-teacher-mask-student-format-tokens)
fi
if [[ "${OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE}" == 1 ]]; then
   SCORE_SCHEMA_ARGS+=(--opd-teacher-normalize-student-tool-template)
fi
if [[ "${OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT}" == 1 ]]; then
   SCORE_SCHEMA_ARGS+=(--opd-teacher-mask-student-pre-tool-content)
fi
if [[ -n "${OPD_TEACHER_REASONING_EFFORT}" ]]; then
   SCORE_SCHEMA_ARGS+=(--opd-teacher-reasoning-effort "${OPD_TEACHER_REASONING_EFFORT}")
fi
if [[ "${OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS}" == "1" ]]; then
   SCORE_SCHEMA_ARGS+=(
      --token-topk-diagnostics
      --diagnostic-topk "${OFFLINE_OPD_DIAGNOSTIC_TOPK}"
      --require-student-topk-logprobs
   )
fi
python3 "${SCRIPT_DIR}/score_teacher_offline.py" \
   --in-dir "${IN_DIR}" \
   --out-dir "${OUT_DIR}" \
   --rm-url "${RM_URL}" \
   --opd-id-topk "${OPD_ID_TOPK}" \
   "${SCORE_SCHEMA_ARGS[@]}" \
   --concurrency "${SCORE_CONCURRENCY}" \
   --file-concurrency "${SCORE_FILE_CONCURRENCY}" \
   --max-attempts "${SCORE_MAX_ATTEMPTS}" \
   "${SHARD_ARGS[@]}" 2>&1 | tee -a "${SCORER_LOG}"

if [[ "${PHASE2_SAMPLE_OUTPUT}" -eq 1 ]]; then
   FINALIZE_DIR="${OUT_DIR}/.finalize_${SCORE_COORD_TOKEN}"
   ensure_shared_dir "${FINALIZE_DIR}"
   printf 'rank=%s host=%s done\n' "${SCORE_RANK}" "$(hostname)" > "${FINALIZE_DIR}/rank_${SCORE_RANK}.tmp.$$"
   mv "${FINALIZE_DIR}/rank_${SCORE_RANK}.tmp.$$" "${FINALIZE_DIR}/rank_${SCORE_RANK}.done"
   FINAL_READY="${FINALIZE_DIR}/ready"
   if [[ "${SCORE_RANK}" -eq 0 ]]; then
      waited=0
      timeout="${PHASE2_FINALIZE_WAIT_TIMEOUT_SEC:-86400}"
      while true; do
         complete=1
         for ((rank=0; rank<SCORE_WORLD; rank++)); do
            [[ -f "${FINALIZE_DIR}/rank_${rank}.done" ]] || complete=0
         done
         (( complete == 1 )) && break
         if (( waited >= timeout )); then
            echo "timed out waiting for Phase-2 scorer ranks after ${timeout}s" >&2
            exit 1
         fi
         sleep 5
         waited=$(( waited + 5 ))
      done
      if [[ "${PHASE2_BALANCE_MANIFESTS}" == 1 ]]; then
         MANIFEST_BALANCE_STRATEGY=length_stratified_greedy
      else
         MANIFEST_BALANCE_STRATEGY=phase1
      fi
      MANIFEST_SUBSET_ARGS=()
      if [[ "${OFFLINE_OPD_ALLOW_SAMPLE_SUBSET:-0}" == 1 ]]; then
         MANIFEST_SUBSET_ARGS+=(--allow-noncontiguous-sample-indices)
      fi
      python3 -u "${SCRIPT_DIR}/build_scored_manifests.py" \
         --output-dir "${OUT_DIR}" \
         --samples-per-file "${PHASE2_SAMPLES_PER_FILE}" \
         --seed "${PHASE1_SHUFFLE_SEED}" \
         --balance-strategy "${MANIFEST_BALANCE_STRATEGY}" \
         "${MANIFEST_SUBSET_ARGS[@]}"
      printf 'ready samples_per_file=%s seed=%s balance_manifests=%s\n' \
         "${PHASE2_SAMPLES_PER_FILE}" "${PHASE1_SHUFFLE_SEED}" \
         "${PHASE2_BALANCE_MANIFESTS}" > "${FINAL_READY}.tmp.$$"
      mv "${FINAL_READY}.tmp.$$" "${FINAL_READY}"
   else
      until [[ -f "${FINAL_READY}" ]]; do sleep 5; done
   fi
fi

echo "[phase2 rank=${RANK}/${NNODES}] shard complete; log=${SCORER_LOG}"
