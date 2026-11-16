#!/bin/bash
set -euo pipefail

# Isolated configuration for iterative offline/online Top-128 OPD.
# Nothing in ../top128 or the generic phase implementations is modified here.

ITERATIVE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
OFFLINE_OPD_DIR="$(cd -- "${ITERATIVE_DIR}/.." &>/dev/null && pwd)"
SLIME_DIR="$(cd -- "${OFFLINE_OPD_DIR}/../../../.." &>/dev/null && pwd)"

export MEGATRON_LM_PATH="${MEGATRON_LM_PATH:-/path/to/workspace/env_run/<user>/Megatron-LM}"
export INITIAL_HF_CHECKPOINT="${INITIAL_HF_CHECKPOINT:-/path/to/SFT/results/qwen35-9b-scaleswe5000-sft-246step-add-finish/checkpoint-246}"
export INITIAL_LOAD_PATH="${INITIAL_LOAD_PATH:-/path/to/OPD/model/qwen35-9b-scaleswe5000-sft-246-add-finish_torch_dist}"
export ORIGIN_HF="${ORIGIN_HF:-${INITIAL_HF_CHECKPOINT}}"
export TEACHER_MODEL="${TEACHER_MODEL:-/path/to/SFT/model/Qwen3.6-27B}"
export PROMPT_DATA="${PROMPT_DATA:-${SLIME_DIR}/data/swe_train.jsonl}"

# Hard-pin the dedicated strategy. A stale sampled-token setting in the parent
# shell must never change this experiment's data schema or loss.
export OFFLINE_OPD_LOSS_TYPE=topk_reverse_kl
export OPD_ID_TOPK=128

# Keep the validated 5x2 schedule as the default, while allowing an isolated
# run to request a different refresh cadence with the same total epoch budget
# (for example 10x1: re-rollout after every training epoch).
export ITERATIVE_CYCLES="${ITERATIVE_CYCLES:-5}"
export EPOCHS_PER_CYCLE="${EPOCHS_PER_CYCLE:-2}"
[[ "${ITERATIVE_CYCLES}" =~ ^[1-9][0-9]*$ ]] || {
   echo "ITERATIVE_CYCLES must be a positive integer: ${ITERATIVE_CYCLES}" >&2
   return 1 2>/dev/null || exit 1
}
[[ "${EPOCHS_PER_CYCLE}" =~ ^[1-9][0-9]*$ ]] || {
   echo "EPOCHS_PER_CYCLE must be a positive integer: ${EPOCHS_PER_CYCLE}" >&2
   return 1 2>/dev/null || exit 1
}

# Keep this isolated iterative recipe in a compact, legal SGLang range.
# slime/ray/rollout.py caps the configured base at 32000; a 200-port stride is
# ample for this recipe's four local TP=2 engines while giving each refresh a
# fresh range. A 10x1 run maps cycles 1..10 to 30000/30200/.../31800.
export SLIME_ROLLOUT_ENGINE_PORT_BASE_ROOT="${SLIME_ROLLOUT_ENGINE_PORT_BASE_ROOT:-30000}"
export SLIME_ROLLOUT_ENGINE_PORT_STRIDE="${SLIME_ROLLOUT_ENGINE_PORT_STRIDE:-200}"

OFFLINE_OPD_BASE="${OFFLINE_OPD_BASE:-/path/to/offline_opd}"
ITERATIVE_STATE_FILE="${ITERATIVE_STATE_FILE:-${OFFLINE_OPD_BASE}/.qwen35-9b-k128-refresh${ITERATIVE_CYCLES}x${EPOCHS_PER_CYCLE}-current}"

validate_run_root() {
   local root="$1"
   [[ "${root}" == "${OFFLINE_OPD_BASE}"/* ]] || {
      echo "RUN_ROOT must be under ${OFFLINE_OPD_BASE}: ${root}" >&2
      return 1
   }
}

cycle_name() {
   printf 'cycle_%02d' "$1"
}

cycle_dir() {
   printf '%s/%s' "${RUN_ROOT}" "$(cycle_name "$1")"
}

checkpoint_iter_dir() {
   local checkpoint_root="$1"
   [[ -r "${checkpoint_root}/latest_checkpointed_iteration.txt" ]] || {
      echo "missing checkpoint pointer: ${checkpoint_root}/latest_checkpointed_iteration.txt" >&2
      return 1
   }
   local iteration
   # Megatron writes this pointer without a trailing newline.  `read` assigns
   # the value but returns 1 at EOF, which aborts our `set -e` orchestrators.
   iteration="$(<"${checkpoint_root}/latest_checkpointed_iteration.txt")"
   [[ "${iteration}" =~ ^[0-9]+$ ]] || {
      echo "invalid checkpoint iteration '${iteration}' in ${checkpoint_root}" >&2
      return 1
   }
   printf '%s/iter_%07d' "${checkpoint_root}" "${iteration}"
}

count_contiguous_rollouts() {
   local dir="$1"
   local count=0
   while [[ -f "${dir}/rollout_${count}.pt" ]]; do
      count=$(( count + 1 ))
   done
   (( count > 0 )) || {
      echo "no rollout_0.pt found under ${dir}" >&2
      return 1
   }
   local path name index
   for path in "${dir}"/rollout_*.pt; do
      name="${path##*/}"
      index="${name#rollout_}"
      index="${index%.pt}"
      if [[ "${index}" =~ ^[0-9]+$ ]] && (( index >= count )); then
         echo "non-contiguous rollout files under ${dir}: ${name} exists after gap ${count}" >&2
         return 1
      fi
   done
   printf '%d' "${count}"
}

require_cycle_input() {
   local cycle="$1"
   if (( cycle == 1 )); then
      CYCLE_INPUT_HF="${INITIAL_HF_CHECKPOINT}"
      CYCLE_INPUT_MEGATRON="${INITIAL_LOAD_PATH}"
      CYCLE_INPUT_LABEL="base-sft"
   else
      local previous_dir
      previous_dir="$(cycle_dir "$(( cycle - 1 ))")"
      CYCLE_INPUT_HF="${previous_dir}/hf"
      CYCLE_INPUT_MEGATRON="${RUN_ROOT}/checkpoints"
      CYCLE_INPUT_LABEL="$(cycle_name "$(( cycle - 1 ))")"
      [[ -f "${previous_dir}/.cycle.complete" ]] || {
         echo "previous cycle is incomplete: ${previous_dir}" >&2
         return 1
      }
      [[ -r "${CYCLE_INPUT_HF}/config.json" ]] || {
         echo "previous cycle HF export is incomplete: ${CYCLE_INPUT_HF}" >&2
         return 1
      }
      checkpoint_iter_dir "${CYCLE_INPUT_MEGATRON}" >/dev/null
      [[ -r "${previous_dir}/checkpoint.iter" ]] || {
         echo "previous cycle checkpoint reference is missing: ${previous_dir}/checkpoint.iter" >&2
         return 1
      }
      local expected_iteration latest_iteration
      expected_iteration="$(<"${previous_dir}/checkpoint.iter")"
      latest_iteration="$(<"${CYCLE_INPUT_MEGATRON}/latest_checkpointed_iteration.txt")"
      [[ "${expected_iteration}" == "${latest_iteration}" ]] || {
         echo "shared checkpoint is not at the previous cycle boundary: " \
              "latest=${latest_iteration}, expected=${expected_iteration}" >&2
         return 1
      }
   fi
   [[ -d "${CYCLE_INPUT_HF}" ]] || { echo "missing input HF model: ${CYCLE_INPUT_HF}" >&2; return 1; }
   [[ -d "${CYCLE_INPUT_MEGATRON}" ]] || {
      echo "missing input Megatron model: ${CYCLE_INPUT_MEGATRON}" >&2
      return 1
   }
   export CYCLE_INPUT_HF CYCLE_INPUT_MEGATRON CYCLE_INPUT_LABEL
}

write_marker() {
   local path="$1"
   shift
   mkdir -p "$(dirname "${path}")"
   printf 'completed=%s %s\n' "$(date -Is)" "$*" > "${path}.tmp.$$"
   mv "${path}.tmp.$$" "${path}"
}

run_multinode_phase() {
   local phase="$1"
   local log_dir="$2"
   mkdir -p "${log_dir}"
   export RUN_DIR="${log_dir}"
   echo "[$(date -Is)] ${phase} starts; log=${log_dir}/launcher.log"
   bash "${OFFLINE_OPD_DIR}/start-offline-opd.sh" "${phase}" \
      > "${log_dir}/launcher.log" 2>&1
   echo "[$(date -Is)] ${phase} completed"
}
