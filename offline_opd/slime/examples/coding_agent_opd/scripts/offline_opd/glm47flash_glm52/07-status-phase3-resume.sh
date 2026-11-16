#!/bin/bash
set -euo pipefail

RUN_ROOT="${RUN_ROOT:-/path/to/offline_opd/glm47flash-glm52-student-template-reverse-t1k-s128-partial-n1280-s20-20260831_190200}"
LOG_DIR="${RUN_ROOT}/logs-phase3-resume"
MEGATRON_LOG="${LOG_DIR}/megatron.rank0.log"
QUEUE_LOG="${LOG_DIR}/queue.log"
TRACKER="${RUN_ROOT}/checkpoints/latest_checkpointed_iteration.txt"
STALE_SECONDS="${STALE_SECONDS:-3600}"

strip_ansi() {
   sed -E 's/\x1B\[[0-9;]*[mK]//g'
}

state=missing
if [[ -f "${RUN_ROOT}/.phase3-resume.running" ]]; then
   state=running
fi
if [[ -f "${RUN_ROOT}/.phase3-resume.failed" ]]; then
   state=failed
fi
if [[ -f "${RUN_ROOT}/.phase3-resume.complete" ]]; then
   state=complete
fi

queue_state=missing
if [[ -s "${RUN_ROOT}/phase3-resume-queue.pid" ]]; then
   queue_pid="$(<"${RUN_ROOT}/phase3-resume-queue.pid")"
   if [[ "${queue_pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${queue_pid}" 2>/dev/null; then
      queue_state="running:${queue_pid}"
   else
      queue_state="stopped:${queue_pid}"
   fi
fi

tracker=missing
if [[ -r "${TRACKER}" ]]; then
   tracker="$(<"${TRACKER}")"
fi

log_age=missing
if [[ -r "${MEGATRON_LOG}" ]]; then
   log_age="$(( $(date +%s) - $(stat -c %Y "${MEGATRON_LOG}") ))"
fi

echo "time=$(date -Is) state=${state} queue=${queue_state} checkpoint=${tracker} megatron_log_age_seconds=${log_age}"

if [[ -r "${MEGATRON_LOG}" ]]; then
   step_line="$(grep -E 'model.py:[0-9]+ - step [0-9]+:' "${MEGATRON_LOG}" | tail -n 1 | strip_ansi || true)"
   if [[ -n "${step_line}" ]]; then
      step="$(sed -nE 's/.* - step ([0-9]+):.*/\1/p' <<<"${step_line}")"
      loss="$(sed -nE "s/.*'train\/loss': ([^,}]+).*/\1/p" <<<"${step_line}")"
      reverse_kl="$(sed -nE "s/.*'train\/opd_reverse_kl': ([^,}]+).*/\1/p" <<<"${step_line}")"
      grad_norm="$(sed -nE "s/.*'train\/grad_norm': ([^,}]+).*/\1/p" <<<"${step_line}")"
      lr="$(sed -nE "s/.*'train\/lr-pg_0': ([^,}]+).*/\1/p" <<<"${step_line}")"
      gbs="$(sed -nE "s/.*'train\/global_batch_size': ([^,}]+).*/\1/p" <<<"${step_line}")"
      echo "latest_step=${step} loss=${loss:-missing} reverse_kl=${reverse_kl:-missing} grad_norm=${grad_norm:-missing} lr=${lr:-missing} global_batch_size=${gbs:-missing}"
   else
      echo "latest_step=none"
   fi

   rollout_line="$(grep -E 'data.py:[0-9]+ - rollout [0-9]+:' "${MEGATRON_LOG}" | tail -n 1 | strip_ansi || true)"
   if [[ -n "${rollout_line}" ]]; then
      rollout="$(sed -nE 's/.* - rollout ([0-9]+):.*/\1/p' <<<"${rollout_line}")"
      echo "latest_rollout=${rollout}"
   else
      echo "latest_rollout=none"
   fi
fi

fatal_count=0
if [[ -r "${QUEUE_LOG}" ]]; then
   last_start="$(grep -n 'starting resume in this job cGPU allocation' "${QUEUE_LOG}" | tail -n 1 | cut -d: -f1 || true)"
   [[ -n "${last_start}" ]] || last_start=1
   fatal_count="$(tail -n "+${last_start}" "${QUEUE_LOG}" | grep -Eic 'Traceback \(most recent call last\):|RayTaskError|CUDA out of memory|OutOfMemoryError|NCCL[^[:cntrl:]]*(error|failed)|Segmentation fault' || true)"
fi
echo "fatal_count_current_attempt=${fatal_count}"

if [[ "${state}" == failed ]]; then
   exit 2
fi
if [[ "${state}" == running && "${log_age}" != missing && "${log_age}" -gt "${STALE_SECONDS}" ]]; then
   exit 3
fi
if [[ "${state}" == running && "${queue_state}" != running:* ]]; then
   exit 4
fi
if (( fatal_count > 0 )); then
   exit 5
fi
