#!/bin/bash
set -uo pipefail

MAIN_RUN="${MAIN_RUN:-/path/to/offline_opd/glm47flash-glm52-student-template-reverse-t1k-s128-20260831_014946}"
RUN_ROOT="${RUN_ROOT:-/path/to/offline_opd/glm47flash-glm52-student-template-reverse-t1k-s128-partial-n1280-s20-20260831_190200}"
STUDENT_MODEL="${STUDENT_MODEL:-/path/to/model/<org>/GLM-4.7-Flash}"
TEACHER_MODEL="${TEACHER_MODEL:-/path/to/model/<org>/GLM-5.2-FP8}"
TARGET_STEP="${TARGET_STEP:-127}"
MEGATRON_LOG="${RUN_ROOT}/logs-phase3-resume/megatron.rank0.log"
QUEUE_LOG="${RUN_ROOT}/logs-phase3-resume/queue.log"
failures=0

pass() {
   echo "PASS $*"
}

fail() {
   echo "FAIL $*"
   failures=$((failures + 1))
}

require_file() {
   if [[ -e "$1" ]]; then
      pass "$2"
   else
      fail "$2: missing $1"
   fi
}

require_line() {
   if grep -Fqx -- "$2" "$1" 2>/dev/null; then
      pass "$3"
   else
      fail "$3"
   fi
}

require_file "${MAIN_RUN}/.phase1.complete" "Phase 1 completion marker"
require_file "${MAIN_RUN}/.phase2.complete" "Phase 2 completion marker"

phase1_count="$(find "${MAIN_RUN}/dump" -maxdepth 1 -type f -name 'rollout_*.pt' 2>/dev/null | wc -l)"
phase2_count="$(find "${MAIN_RUN}/scored/samples" -maxdepth 1 -type f -name 'sample_*.pt' 2>/dev/null | wc -l)"
manifest_count="$(find "${MAIN_RUN}/scored/source_manifests" -maxdepth 1 -type f -name 'rollout_*.json' 2>/dev/null | wc -l)"
[[ "${phase1_count}" == 4093 ]] && pass "Phase 1 has 4093 rollouts" || fail "Phase 1 rollout count=${phase1_count}, expected=4093"
[[ "${phase2_count}" == 4093 ]] && pass "Phase 2 has 4093 scored samples" || fail "Phase 2 sample count=${phase2_count}, expected=4093"
[[ "${manifest_count}" == 4093 ]] && pass "Phase 2 has 4093 source manifests" || fail "Phase 2 manifest count=${manifest_count}, expected=4093"

if cmp -s "${STUDENT_MODEL}/tokenizer.json" "${TEACHER_MODEL}/tokenizer.json"; then
   pass "Student and Teacher tokenizer.json are byte-identical"
else
   fail "Student and Teacher tokenizer.json differ"
fi

contract="${MAIN_RUN}/experiment.env"
require_line "${contract}" 'LOSS=cached_teacher_dynamic_topk_reverse_kl' "pure cached dynamic Top-K reverse-KL"
require_line "${contract}" 'FORWARD_KL=0' "forward-KL disabled"
require_line "${contract}" 'STUDENT_ANCHOR=0' "Student anchor disabled"
require_line "${contract}" 'TOKEN_ALIGNMENT=shared_exact_ids' "shared exact token ids"
require_line "${contract}" 'TOKEN_ID_BRIDGE=none' "token-id bridge disabled"
require_line "${contract}" 'TEMPLATE_POLICY=exact_student_token_replay' "exact Student-token replay"
require_line "${contract}" 'TEACHER_TEMPLATE_RERENDER=0' "Teacher template re-render disabled"
require_line "${contract}" 'FORMAT_TOKEN_MASK=0' "extra format-token mask disabled"
require_line "${contract}" 'PHASE1_TP=4' "rollout TP=4"
require_line "${contract}" 'PHASE2_ATTENTION=dense' "Teacher dense attention contract"

dense_nodes="$(grep -l 'Dense attention kv len threshold is manually set to 131072' "${MAIN_RUN}"/logs-phase2/phase2.teacher.group*.log 2>/dev/null | wc -l)"
[[ "${dense_nodes}" == 4 ]] && pass "Teacher dense-attention runtime confirmed on 4 nodes" || fail "Teacher dense-attention runtime nodes=${dense_nodes}, expected=4"

require_file "${RUN_ROOT}/.phase3-resume.complete" "Phase 3 resume completion marker"
tracker=missing
if [[ -r "${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt" ]]; then
   tracker="$(<"${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt")"
fi
[[ "${tracker}" == "${TARGET_STEP}" ]] && pass "checkpoint tracker reached ${TARGET_STEP}" || fail "checkpoint tracker=${tracker}, expected=${TARGET_STEP}"
require_file "${RUN_ROOT}/checkpoints/iter_$(printf '%07d' "${TARGET_STEP}")" "final distributed checkpoint"

latest_step=none
if [[ -r "${MEGATRON_LOG}" ]]; then
   latest_step="$(grep -E 'model.py:[0-9]+ - step [0-9]+:' "${MEGATRON_LOG}" | tail -n 1 | sed -nE 's/.* - step ([0-9]+):.*/\1/p')"
fi
[[ "${latest_step}" == "${TARGET_STEP}" ]] && pass "training log reached step ${TARGET_STEP}" || fail "latest training step=${latest_step:-none}, expected=${TARGET_STEP}"

if grep -Fq 'wandb: Resuming run offline-opd-glm47flash-student-glm52-teacher-student-template-reverse-t1k-s128' "${QUEUE_LOG}" 2>/dev/null; then
   pass "W&B run resumed with the original id"
else
   fail "W&B resume evidence missing"
fi

if grep -E 'model.py:[0-9]+ - step [0-9]+:' "${MEGATRON_LOG}" 2>/dev/null | grep -Eq "'train/opd_(forward_kl|student_anchor)"; then
   fail "unexpected forward-KL or anchor metric found in Phase 3"
else
   pass "no forward-KL or anchor metric in Phase 3 steps"
fi

last_start="$(grep -n 'starting resume in this job cGPU allocation' "${QUEUE_LOG}" 2>/dev/null | tail -n 1 | cut -d: -f1 || true)"
[[ -n "${last_start}" ]] || last_start=1
fatal_count="$(tail -n "+${last_start}" "${QUEUE_LOG}" 2>/dev/null | grep -Eic 'Traceback \(most recent call last\):|RayTaskError|CUDA out of memory|OutOfMemoryError|NCCL[^[:cntrl:]]*(error|failed)|Segmentation fault' || true)"
[[ "${fatal_count}" == 0 ]] && pass "no fatal error in the final Phase 3 attempt" || fail "fatal error count in final Phase 3 attempt=${fatal_count}"

if (( failures > 0 )); then
   echo "AUDIT_RESULT=incomplete failures=${failures}"
   exit 1
fi

echo "AUDIT_RESULT=complete failures=0"
