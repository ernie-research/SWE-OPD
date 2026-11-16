#!/bin/bash
set -euo pipefail
set -x
# OFFLINE OPD PHASE 3: four-node pure trainer over Phase-2 baked trajectories.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# general_settings.sh contains an optional NVLink grep probe that legitimately
# returns no matches; it was written without pipefail enabled.
set +o pipefail
source "${SCRIPT_DIR}/../multi_nodes/general_settings.sh"
set -o pipefail
export PYTHONPATH="${SLIME_DIR}:${MEGATRON_LM_PATH}${PYTHONPATH:+:${PYTHONPATH}}"

SCORED_DIR="${SCORED_DIR:?set SCORED_DIR to a Phase-2 scored or fanout directory}"
OPD_ID_TOPK="${OPD_ID_TOPK:-1024}"
OFFLINE_OPD_LOSS_TYPE="${OFFLINE_OPD_LOSS_TYPE:-topk_reverse_kl}"
case "${OFFLINE_OPD_LOSS_TYPE}" in
   topk_reverse_kl|token_reverse_kl|cached_teacher_topk_forward_kl|cached_teacher_dynamic_topk_reverse_kl|cached_teacher_dynamic_topk_reverse_kl_with_student_anchor|cached_teacher_sampled_token_pg_with_student_anchor|chunk_teacher_sampled_token_pg_with_student_anchor|cached_teacher_bidirectional_topk_kl|cached_teacher_bidirectional_topk_kl_with_student_anchor|cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor) ;;
   *) echo "unsupported OFFLINE_OPD_LOSS_TYPE=${OFFLINE_OPD_LOSS_TYPE}" >&2; exit 1 ;;
esac
OPD_TEACHER_CACHE_TOPK="${OPD_TEACHER_CACHE_TOPK:-4096}"
OPD_DYNAMIC_SUPPORT_TOPK="${OPD_DYNAMIC_SUPPORT_TOPK:-128}"
OPD_TEACHER_CACHE_H2D_ROWS="${OPD_TEACHER_CACHE_H2D_ROWS:-512}"
if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "token_reverse_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_sampled_token_pg_with_student_anchor" ]]; then
   OFFLINE_OPD_TOPK_OVERLAP="${OFFLINE_OPD_TOPK_OVERLAP:-0}"
else
   OFFLINE_OPD_TOPK_OVERLAP="${OFFLINE_OPD_TOPK_OVERLAP:-1}"
fi
if [[ "${OFFLINE_OPD_TOPK_OVERLAP}" != "0" && "${OFFLINE_OPD_TOPK_OVERLAP}" != "1" ]]; then
   echo "OFFLINE_OPD_TOPK_OVERLAP must be 0 or 1, got ${OFFLINE_OPD_TOPK_OVERLAP}" >&2
   exit 1
fi
OFFLINE_OPD_IMPORTANCE_DEBUG="${OFFLINE_OPD_IMPORTANCE_DEBUG:-0}"
if [[ "${OFFLINE_OPD_IMPORTANCE_DEBUG}" != "0" && "${OFFLINE_OPD_IMPORTANCE_DEBUG}" != "1" ]]; then
   echo "OFFLINE_OPD_IMPORTANCE_DEBUG must be 0 or 1, got ${OFFLINE_OPD_IMPORTANCE_DEBUG}" >&2
   exit 1
fi
OPD_REVERSE_KL_TAIL_BUCKET="${OPD_REVERSE_KL_TAIL_BUCKET:-0}"
if [[ "${OPD_REVERSE_KL_TAIL_BUCKET}" != "0" && "${OPD_REVERSE_KL_TAIL_BUCKET}" != "1" ]]; then
   echo "OPD_REVERSE_KL_TAIL_BUCKET must be 0 or 1, got ${OPD_REVERSE_KL_TAIL_BUCKET}" >&2
   exit 1
fi
[[ -r "${SCORED_DIR}/rollout_0.pt" ]] || { echo "missing ${SCORED_DIR}/rollout_0.pt" >&2; exit 1; }

DISCOVERED_NUM_ROLLOUT="$(python3 - "${SCORED_DIR}" <<'PY'
import re, sys
from pathlib import Path

directory = Path(sys.argv[1])
pattern = re.compile(r"rollout_(\d+)\.pt$")
indices = sorted(int(m.group(1)) for p in directory.glob("rollout_*.pt") if (m := pattern.fullmatch(p.name)))
if not indices or indices != list(range(len(indices))):
    raise SystemExit(f"rollout files are not contiguous under {directory}: {indices}")
print(len(indices))
PY
)"
NUM_ROLLOUT="${NUM_ROLLOUT:-${DISCOVERED_NUM_ROLLOUT}}"
if (( NUM_ROLLOUT != DISCOVERED_NUM_ROLLOUT )); then
   echo "NUM_ROLLOUT=${NUM_ROLLOUT} does not match ${DISCOVERED_NUM_ROLLOUT} contiguous files in ${SCORED_DIR}" >&2
   exit 1
fi
if [[ -z "${ROLLOUT_BATCH_SIZE:-}" ]]; then
   # Convenience fallback for ad-hoc launches. Production launches pass the
   # known Phase-2 shard size and therefore never deserialize data here.
   ROLLOUT_BATCH_SIZE="$(python3 - "${SCORED_DIR}/rollout_0.pt" <<'PY'
import sys, torch
blob = torch.load(sys.argv[1], weights_only=False)
if blob.get("format") in {"offline_opd_sample_manifest_v1", "offline_opd_sample_manifest_v2"}:
    print(len(blob["sample_files"]))
else:
    print(len(blob["samples"]))
PY
)"
fi

NUM_STEPS_PER_ROLLOUT="${NUM_STEPS_PER_ROLLOUT:-1}"
if (( NUM_STEPS_PER_ROLLOUT <= 0 || ROLLOUT_BATCH_SIZE % NUM_STEPS_PER_ROLLOUT != 0 )); then
   echo "NUM_STEPS_PER_ROLLOUT=${NUM_STEPS_PER_ROLLOUT} must be a positive divisor of batch=${ROLLOUT_BATCH_SIZE}" >&2
   exit 1
fi
EXPECTED_GLOBAL_BATCH_SIZE=$(( ROLLOUT_BATCH_SIZE / NUM_STEPS_PER_ROLLOUT ))
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${EXPECTED_GLOBAL_BATCH_SIZE}}"
if (( GLOBAL_BATCH_SIZE != EXPECTED_GLOBAL_BATCH_SIZE )); then
   echo "GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}; expected ${EXPECTED_GLOBAL_BATCH_SIZE}=batch/steps" >&2
   exit 1
fi

RAY_GPUS="${RAY_GPUS:-0,1,2,3,4,5,6,7}"
RAY_NUM_GPUS_PER_NODE="${RAY_NUM_GPUS_PER_NODE:-8}"
ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-${NNODES}}"
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-8}"
if (( ACTOR_NUM_NODES != NNODES )); then
   echo "offline Phase 3 expects all ${NNODES} nodes for training; got ACTOR_NUM_NODES=${ACTOR_NUM_NODES}" >&2
   exit 1
fi

# Phase 2 already validates every artifact while producing it.  Re-reading the
# complete scored dataset here costs hundreds of GiB and delays Ray startup on
# the other nodes.  Keep the deep check available for manual/debug launches,
# but skip it during normal Phase-3 training.
if [[ "${RANK}" -eq 0 && "${OFFLINE_OPD_FULL_VERIFY:-0}" -eq 1 ]]; then
   VERIFY_ARGS=(--opd-loss-type "${OFFLINE_OPD_LOSS_TYPE}")
   if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "topk_reverse_kl" ]]; then
      VERIFY_ARGS+=(--require-student-topk-logprobs --require-sparse-teacher)
   elif [[ "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_dynamic_topk_reverse_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_sampled_token_pg_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_topk_forward_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_bidirectional_topk_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_bidirectional_topk_kl_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor" ]]; then
      VERIFY_ARGS+=(
         --require-student-topk-logprobs
         --opd-dynamic-support-topk "${OPD_DYNAMIC_SUPPORT_TOPK}"
         --opd-teacher-cache-topk "${OPD_TEACHER_CACHE_TOPK}"
      )
      if [[ "${OFFLINE_OPD_ALLOW_SUPPORT_TOPK_TRUNCATION:-0}" == "1" ]]; then
         VERIFY_ARGS+=(--truncate-dynamic-support)
      fi
   elif [[ "${OFFLINE_OPD_TOKEN_TOPK_DIAGNOSTICS:-0}" == "1" ]]; then
      VERIFY_ARGS+=(--token-topk-diagnostics)
   fi
   python3 "${SCRIPT_DIR}/verify_offline_pipeline.py" --mode scored-dir \
      --dir "${SCORED_DIR}" --opd-id-topk "${OPD_ID_TOPK}" \
      "${VERIFY_ARGS[@]}" \
      --allow-repeated-samples
elif [[ "${RANK}" -eq 0 ]]; then
   echo "PHASE3 full scored-data verification skipped (set OFFLINE_OPD_FULL_VERIFY=1 to enable)"
fi

preclean_ray
source "${MODEL_ARGS_SH:-${SLIME_DIR}/scripts/models/qwen3.5-9B.sh}"

WANDB_ARGS=()
if [[ "${USE_WANDB:-1}" -eq 1 ]]; then
   WANDB_ARGS=(
      --use-wandb
      --disable-wandb-random-suffix
      --wandb-team "${WANDB_TEAM:-czguocs-baidu}"
      --wandb-project "${WANDB_PROJECT:-slime-OPD}"
      --wandb-group "${WANDB_GROUP:-${EXP_NAME}-${RUN_TAG}}"
      --wandb-id "${WANDB_RUN_ID:-${RUN_TAG}}"
      --wandb-mode "${WANDB_MODE:-online}"
   )
fi

MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-$((1024*128))}"
TP_SIZE="${TP_SIZE:-8}"
PP_SIZE="${PP_SIZE:-2}"
CP_SIZE="${CP_SIZE:-1}"
EXPERT_MP_SIZE="${EXPERT_MP_SIZE:-1}"
EXPERT_TP_SIZE="${EXPERT_TP_SIZE:-1}"
if (( CP_SIZE <= 0 || MAX_CONTEXT_LEN % CP_SIZE != 0 )); then
   echo "CP_SIZE=${CP_SIZE} must be positive and divide MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN}" >&2
   exit 1
fi
# max-tokens-per-gpu is a *local* token budget.  The dynamic packer multiplies
# it by CP_SIZE, so dividing the 128K global packing budget here keeps the
# effective per-microbatch cap unchanged when CP is enabled.
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-$((MAX_CONTEXT_LEN / CP_SIZE))}"
TRAIN_WORLD_SIZE=$(( ACTOR_NUM_NODES * ACTOR_NUM_GPUS_PER_NODE ))
MODEL_PARALLEL_SIZE=$(( TP_SIZE * PP_SIZE * CP_SIZE ))
if (( TRAIN_WORLD_SIZE % MODEL_PARALLEL_SIZE != 0 )); then
   echo "training world size ${TRAIN_WORLD_SIZE} is not divisible by TP*PP*CP=${MODEL_PARALLEL_SIZE}" >&2
   exit 1
fi
DP_SIZE=$(( TRAIN_WORLD_SIZE / MODEL_PARALLEL_SIZE ))
EXPERT_MODEL_PARALLEL_SIZE=$(( EXPERT_TP_SIZE * EXPERT_MP_SIZE * PP_SIZE ))
if (( TRAIN_WORLD_SIZE % EXPERT_MODEL_PARALLEL_SIZE != 0 )); then
   echo "training world size ${TRAIN_WORLD_SIZE} is not divisible by ETP*EP*PP=${EXPERT_MODEL_PARALLEL_SIZE}" >&2
   exit 1
fi
CKPT_ARGS=(
   --hf-checkpoint "${HF_CHECKPOINT:-/path/to/SFT/results/qwen35-9b-scaleswe5000-sft-246step-add-finish/checkpoint-246}"
   --load "${LOAD_PATH:-/path/to/OPD/model/qwen35-9b-scaleswe5000-sft-246-add-finish_torch_dist}"
   --save "${SAVE_PATH:-/path/to/OPD/results/qwen35-9b-offline-opd/}"
   --save-interval "${SAVE_INTERVAL:-30}"
)
OFFLINE_OPD_FINETUNE="${OFFLINE_OPD_FINETUNE:-0}"
if [[ "${OFFLINE_OPD_FINETUNE}" == "1" ]]; then
   # Start a new experiment from checkpoint model weights only.  In
   # particular, do not inherit optimizer moments, RNG state, iteration, or
   # consumed-sample progress from the source experiment.
   CKPT_ARGS+=(--finetune --no-load-optim --no-load-rng)
elif [[ "${OFFLINE_OPD_FINETUNE}" != "0" ]]; then
   echo "OFFLINE_OPD_FINETUNE must be 0 or 1, got ${OFFLINE_OPD_FINETUNE}" >&2
   exit 1
fi
if [[ "${OFFLINE_OPD_OVERRIDE_OPT_PARAM_SCHEDULER:-0}" == "1" ]]; then
   # Iterative offline/online refreshes grow the cumulative replay index after
   # every cycle. Megatron derives the scheduler horizon from that index, so it
   # legitimately differs from the previous cycle's checkpoint. Keep the
   # current CLI schedule while still restoring optimizer state and progress.
   CKPT_ARGS+=(--override-opt_param-scheduler)
fi
ROLLOUT_ARGS=(
   --disable-rollout-global-dataset
   # Phase 3 uses a custom replay loss. Even the sampled-token PG variant builds
   # its detached OPD advantage and current/rollout ratio inside that one loss
   # forward, so it does not need the actor's separate advantage/log-prob pass.
   --disable-compute-advantages-and-returns
   --num-rollout "${NUM_ROLLOUT}"
   --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
   --n-samples-per-prompt 1
   --rollout-max-context-len "${MAX_CONTEXT_LEN}"
   --num-steps-per-rollout "${NUM_STEPS_PER_ROLLOUT}"
   --global-batch-size "${GLOBAL_BATCH_SIZE}"
)
OFFLINE_OPD_BALANCE_DATA="${OFFLINE_OPD_BALANCE_DATA:-1}"
[[ "${OFFLINE_OPD_BALANCE_DATA}" == 0 || "${OFFLINE_OPD_BALANCE_DATA}" == 1 ]] || {
   echo "OFFLINE_OPD_BALANCE_DATA must be 0 or 1" >&2
   exit 1
}
if [[ "${OFFLINE_OPD_BALANCE_DATA}" == 1 ]]; then
   ROLLOUT_ARGS+=(--balance-data)
fi
# A regular Phase-3 launch starts from the base model and must consume file 0.
# On Megatron-checkpoint resume, omitting this argument lets actor.create()
# recover loaded_rollout_id+1 from checkpoint state.  Passing an explicit zero
# here would silently reload the trained model at step N but replay data from
# step 0, producing N extra optimizer steps.
OFFLINE_OPD_RESUME="${OFFLINE_OPD_RESUME:-0}"
if [[ "${OFFLINE_OPD_RESUME}" == "1" ]]; then
   [[ -f "${LOAD_PATH}/latest_checkpointed_iteration.txt" ]] || {
      echo "OFFLINE_OPD_RESUME=1 requires a Megatron checkpoint root: ${LOAD_PATH}" >&2
      exit 1
   }
elif [[ "${OFFLINE_OPD_RESUME}" == "0" ]]; then
   ROLLOUT_ARGS+=(--start-rollout-id "${START_ROLLOUT_ID:-0}")
else
   echo "OFFLINE_OPD_RESUME must be 0 or 1, got ${OFFLINE_OPD_RESUME}" >&2
   exit 1
fi
if [[ "${PREFETCH_ROLLOUT_DATA:-1}" == "1" ]]; then
   ROLLOUT_ARGS+=(--prefetch-rollout-data)
fi
OPD_ARGS=(
   --loss-type custom_loss
   --custom-loss-function-path slime.backends.megatron_utils.loss.opd_loss_function
   --opd-loss-type "${OFFLINE_OPD_LOSS_TYPE}"
)
if [[ "${OPD_REVERSE_KL_TAIL_BUCKET}" == "1" ]]; then
   OPD_ARGS+=(--opd-reverse-kl-tail-bucket)
fi
if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "token_reverse_kl" ]]; then
   OPD_ARGS+=(--opd-id-topk 0)
elif [[ "${OFFLINE_OPD_LOSS_TYPE}" == "chunk_teacher_sampled_token_pg_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_dynamic_topk_reverse_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_sampled_token_pg_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_topk_forward_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_bidirectional_topk_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_bidirectional_topk_kl_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor" ]]; then
   OPD_ARGS+=(
      --opd-id-topk "${OPD_DYNAMIC_SUPPORT_TOPK}"
      --opd-teacher-cache-topk "${OPD_TEACHER_CACHE_TOPK}"
      --opd-dynamic-support-topk "${OPD_DYNAMIC_SUPPORT_TOPK}"
      --opd-teacher-cache-miss-policy cutoff
      --opd-teacher-cache-h2d-rows "${OPD_TEACHER_CACHE_H2D_ROWS}"
   )
   if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_bidirectional_topk_kl" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_bidirectional_topk_kl_with_student_anchor" ]]; then
      OPD_ARGS+=(
         --opd-forward-kl-coef "${OPD_FORWARD_KL_COEF:-0.5}"
         --opd-reverse-kl-coef "${OPD_REVERSE_KL_COEF:-0.5}"
         --opd-forward-kl-anneal-steps "${OPD_FORWARD_KL_ANNEAL_STEPS:-0}"
      )
   elif [[ "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_sampled_token_pg_with_student_anchor" ]]; then
      # The reverse-only implementation does not consume the bidirectional
      # coefficients, but pass truthful values so the resolved argument dump
      # and experiment record cannot misleadingly report the parser defaults.
      OPD_ARGS+=(--opd-forward-kl-coef 0 --opd-reverse-kl-coef 1)
   fi
   if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_sampled_token_pg_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_bidirectional_topk_kl_with_student_anchor" || "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor" ]]; then
      OPD_ARGS+=(--opd-student-anchor-kl-coef "${OPD_STUDENT_ANCHOR_KL_COEF:-1.0}")
   fi
   # Top-K KL on a chunk artifact: n:m anchor rows train the chunk sampled-token term.
   if [[ "${OPD_NM_ROWS_SAMPLED_TOKEN:-0}" == "1" ]]; then
      case "${OFFLINE_OPD_LOSS_TYPE}" in
         cached_teacher_dynamic_topk_reverse_kl_with_student_anchor|cached_teacher_bidirectional_topk_kl_with_student_anchor) ;;
         *) echo "OPD_NM_ROWS_SAMPLED_TOKEN=1 needs an anchored top-K loss, got ${OFFLINE_OPD_LOSS_TYPE}" >&2; exit 1 ;;
      esac
      OPD_ARGS+=(
         --opd-nm-rows-sampled-token
         --opd-chunk-tier "${OPD_CHUNK_TIER:-struct}"
         --opd-sampled-token-clip-reference "${OPD_SAMPLED_TOKEN_CLIP_REFERENCE:-rollout}"
         --eps-clip "${OPD_SAMPLED_TOKEN_EPS_CLIP:-0.2}"
         --eps-clip-high "${OPD_SAMPLED_TOKEN_EPS_CLIP_HIGH:-0.2}"
      )
   fi
   if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_sampled_token_pg_with_student_anchor" ]]; then
      OPD_ARGS+=(
         --opd-kl-coef "${OPD_KL_COEF:-1.0}"
         --opd-sampled-token-cache-miss-policy "${OPD_SAMPLED_TOKEN_CACHE_MISS_POLICY:-anchor}"
         --opd-sampled-token-clip-reference "${OPD_SAMPLED_TOKEN_CLIP_REFERENCE:-rollout}"
         --eps-clip "${OPD_SAMPLED_TOKEN_EPS_CLIP:-0.2}"
         --eps-clip-high "${OPD_SAMPLED_TOKEN_EPS_CLIP_HIGH:-0.2}"
      )
      if [[ "${OPD_SAMPLED_TOKEN_SDAR_GATE:-0}" == "1" ]]; then
         OPD_ARGS+=(
            --opd-sampled-token-sdar-gate
            --opd-sdar-gate-beta "${OPD_SDAR_GATE_BETA:-5.0}"
            --opd-sdar-gate-scale "${OPD_SDAR_GATE_SCALE:-1.0}"
         )
      fi
   fi
   if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "chunk_teacher_sampled_token_pg_with_student_anchor" ]]; then
      # Same estimator knobs as the cached variant; the Teacher value comes from
      # the byte-chunk artifact, so the tier replaces the cache-topk settings.
      OPD_ARGS+=(
         --opd-kl-coef "${OPD_KL_COEF:-1.0}"
         --opd-student-anchor-kl-coef "${OPD_STUDENT_ANCHOR_KL_COEF:-1.0}"
         --opd-chunk-tier "${OPD_CHUNK_TIER:-struct}"
         --opd-sampled-token-cache-miss-policy "${OPD_SAMPLED_TOKEN_CACHE_MISS_POLICY:-anchor}"
         --opd-sampled-token-clip-reference "${OPD_SAMPLED_TOKEN_CLIP_REFERENCE:-rollout}"
         --eps-clip "${OPD_SAMPLED_TOKEN_EPS_CLIP:-0.2}"
         --eps-clip-high "${OPD_SAMPLED_TOKEN_EPS_CLIP_HIGH:-0.2}"
         --opd-forward-kl-coef 0 --opd-reverse-kl-coef 1
      )
      if [[ "${OPD_SAMPLED_TOKEN_SDAR_GATE:-0}" == "1" ]]; then
         OPD_ARGS+=(
            --opd-sampled-token-sdar-gate
            --opd-sdar-gate-beta "${OPD_SDAR_GATE_BETA:-5.0}"
            --opd-sdar-gate-scale "${OPD_SDAR_GATE_SCALE:-1.0}"
         )
      fi
   fi
   if [[ "${OFFLINE_OPD_LOSS_TYPE}" == "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor" ]]; then
      OPD_ARGS+=(
         --opd-let-token-ids "${OPD_LET_TOKEN_IDS:?set OPD_LET_TOKEN_IDS for conditional KL}"
         --opd-me-token-ids "${OPD_ME_TOKEN_IDS:?set OPD_ME_TOKEN_IDS for conditional KL}"
      )
   fi
   if [[ -n "${OPD_TOKEN_ID_BRIDGE:-}" ]]; then
      OPD_ARGS+=(--opd-token-id-bridge "${OPD_TOKEN_ID_BRIDGE}")
   fi
   if [[ -n "${OPD_STUDENT_VALID_VOCAB_SIZE:-}" ]]; then
      OPD_ARGS+=(--opd-student-valid-vocab-size "${OPD_STUDENT_VALID_VOCAB_SIZE}")
   fi
else
   OPD_ARGS+=(--opd-id-topk "${OPD_ID_TOPK}")
fi
if [[ "${OFFLINE_OPD_TOPK_OVERLAP}" == "1" ]]; then
   # Enabled by default for the current monitoring run. This computes on every
   # training forward; set OFFLINE_OPD_TOPK_OVERLAP=0 to skip the expensive
   # full-vocabulary top-k while retaining all sparse offline drift metrics.
   OPD_ARGS+=(--offline-opd-topk-overlap)
fi
if [[ "${OFFLINE_OPD_IMPORTANCE_DEBUG}" == "1" ]]; then
   OPD_ARGS+=(--offline-opd-importance-debug)
fi
OPTIMIZER_ARGS=(
   # Keep this block identical to the standard online OPD training recipe.
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
PERF_ARGS=(
   --tensor-model-parallel-size "${TP_SIZE}"
   --sequence-parallel
   --pipeline-model-parallel-size "${PP_SIZE}"
   --context-parallel-size "${CP_SIZE}"
   --expert-model-parallel-size "${EXPERT_MP_SIZE}"
   --expert-tensor-parallel-size "${EXPERT_TP_SIZE}"
   --recompute-loss-function
   --optimizer-cpu-offload
   --use-precision-aware-optimizer
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
   --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE:-8192}"
)
MISC_ARGS=(--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --attention-backend flash)
if [[ -n "${DISTRIBUTED_TIMEOUT_MINUTES:-}" ]]; then
   MISC_ARGS+=(--distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES}")
fi
DEBUG_ARGS=(--load-debug-rollout-data "${SCORED_DIR}/rollout_{rollout_id}.pt")

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
setup_proxy
cd "${SLIME_DIR}"
build_runtime_env_json

echo "PHASE3 train: loss=${OFFLINE_OPD_LOSS_TYPE}, support_topk=${OPD_DYNAMIC_SUPPORT_TOPK}, teacher_cache_topk=${OPD_TEACHER_CACHE_TOPK}, files=${NUM_ROLLOUT}, batch=${ROLLOUT_BATCH_SIZE}, steps/file=${NUM_STEPS_PER_ROLLOUT}, global_batch=${GLOBAL_BATCH_SIZE}, balance_data=${OFFLINE_OPD_BALANCE_DATA}, TP=${TP_SIZE}, PP=${PP_SIZE}, CP=${CP_SIZE}, DP=${DP_SIZE}, EP=${EXPERT_MP_SIZE}, ETP=${EXPERT_TP_SIZE}, local_token_cap=${MAX_TOKENS_PER_GPU}, global_pack_cap=$(( MAX_TOKENS_PER_GPU * CP_SIZE )), offline_topk_overlap=${OFFLINE_OPD_TOPK_OVERLAP}, nodes=${NNODES}"

if [[ "${RANK}" -eq 0 ]]; then
   CUDA_VISIBLE_DEVICES="${RAY_GPUS}" "${RAY_LAUNCH[@]}" start --head --node-ip-address "${POD_IP}" \
      --port "${RAY_HEAD_PORT}" --num-gpus "${RAY_NUM_GPUS_PER_NODE}" \
      --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port="${RAY_DASHBOARD_PORT}"
   sleep "${WAIT_WORKER_SECS}"
   "${RAY_LAUNCH[@]}" status
   # The job HTTP API lives in a dashboard subprocess module that loads lazily.
   # On this cluster importing it costs ~20s of CPU, so `job submit` can hit a
   # 504 from an otherwise healthy cluster. Poll until it serves instead of
   # trusting WAIT_WORKER_SECS to have covered it.
   DASH_READY=0
   for _ in $(seq 1 "${RAY_DASHBOARD_READY_TRIES:-120}"); do
      if curl -sf -o /dev/null --max-time 5 \
         "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/jobs/"; then
         DASH_READY=1
         break
      fi
      sleep 5
   done
   echo "[phase3 rank=0] ray job API ready=${DASH_READY}"

   set +x
   SUBMIT_RC=0
   "${RAY_LAUNCH[@]}" job submit --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
      --runtime-env-json="${RUNTIME_ENV_JSON}" \
      -- "${SLIME_VENV}/bin/python3" -u train.py \
      --actor-num-nodes "${ACTOR_NUM_NODES}" \
      --actor-num-gpus-per-node "${ACTOR_NUM_GPUS_PER_NODE}" \
      --placement-group-head-ip "${MASTER_ADDR}" \
      "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
      "${OPTIMIZER_ARGS[@]}" "${OPD_ARGS[@]}" "${WANDB_ARGS[@]}" \
      "${PERF_ARGS[@]}" "${MISC_ARGS[@]}" "${DEBUG_ARGS[@]}" || SUBMIT_RC=$?
   set -x
   if [[ "${SUBMIT_RC}" -ne 0 ]]; then
      dump_engine_logs
   fi
   "${RAY_LAUNCH[@]}" stop --force || true
   exit "${SUBMIT_RC}"
else
   # Give the head a quiet window first. Importing ray's dashboard modules out
   # of the shared-storage venv costs ~20s of CPU per process, and three
   # workers retrying their join in parallel starve the head's dashboard agent
   # past the raylet's WaitForDashboardAgentPorts deadline -- the raylet then
   # aborts and no node ever registers in GCS. Default 0 keeps the old behaviour.
   sleep "${RAY_WORKER_JOIN_DELAY_SECS:-0}"
   # Retry only until this node has joined.  The raylet is daemonized by a
   # successful `ray start`; keeping the CLI in `--block` mode would turn the
   # head's normal shutdown into an endless reconnect loop.
   until CUDA_VISIBLE_DEVICES="${RAY_GPUS}" "${RAY_LAUNCH[@]}" start --address="${MASTER_ADDR}:${RAY_HEAD_PORT}" \
      --node-ip-address "${POD_IP}" --num-gpus "${RAY_NUM_GPUS_PER_NODE}" --disable-usage-stats; do
      "${RAY_LAUNCH[@]}" stop --force || true
      sleep 5
   done
fi
