#!/usr/bin/env bash
###############################################################################
# Clean slime training leftovers on the LOCAL node.
#
# Run standalone on a node, or fan it out to every node via:
#   bash dispatch.sh clean
#
# Kills, in dependency order, the processes a slime coding_agent_opd run leaves
# behind (sglang rollout engines -> ray cluster -> the train driver / megatron
# workers), then clears the stale /tmp/ray session dir so the next `ray start`
# does not fail on a session-name mismatch.
#
# Every step is best-effort (|| true) so a node with nothing to clean is a
# no-op, and the script never aborts a fan-out over other nodes.
###############################################################################
set -uo pipefail

HOST="$(hostname)"
log() { echo "[cleanup:${HOST}] $*"; }

# 0) (removed) remote RLE sandbox session cleanup. It used to stop the pods a
#    dead run had booked with the RLE service via the eval helper
#    ``examples.coding_agent_opd.eval.kill_rle_sessions``; that helper was
#    removed together with the rest of ``eval/utils`` (normalize_dataset /
#    build_eval_index / probe_rle). Leaked sessions are now left to the RLE
#    platform TTL, or stopped by hand with the RLE SDK (session_stop,
#    tool_type=swe-pdc) using the ids run_eval.sh logged to
#    ``${RLE_SESSION_IDS_DIR:-<out_dir>/rle_sessions}/session_ids.jsonl``.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# scripts/test/ is 4 levels below the slime root (same as the run scripts).
SLIME_DIR="${SLIME_DIR:-$(cd "${SCRIPT_DIR}/../../../.." && pwd)}"
log "RLE session cleanup: skipped (eval.kill_rle_sessions helper removed)."

# 1) sglang rollout engines (each owns a GPU; must die before ray tears down).
log "stopping sglang engines ..."
pkill -9 -f "sglang" || true

# 2) slime training driver + custom rollout workers.
log "stopping slime train drivers ..."
pkill -9 -f "train_async.py" || true
pkill -9 -f "train.py" || true
pkill -9 -f "slime.rollout" || true
pkill -9 -f "fully_async_rollout" || true

# 2b) the coding_agent_opd launch scripts themselves. On workers these run an
# `until ray start --block; do retry; done` loop that OUTLIVES a `ray stop`:
# without killing the launcher, it just re-registers the node on the next
# `ray start`, stacking multiple raylets on one physical node (port/GPU
# collision -> raylet SIGABRT). Kill the launcher before tearing ray down.
pkill -9 -f "run-qwen3.5-9B-coding-opd" || true
pkill -9 -f "run-qwen3-8B-coding-opd" || true

# 3) ray cluster (graceful stop first, then hard-kill any survivors).
log "stopping ray ..."
ray stop --force || true
pkill -9 -f "ray::" || true
pkill -9 -f "raylet" || true
pkill -9 -f "gcs_server" || true
pkill -9 -f "plasma_store" || true
pkill -9 -f "dashboard" || true
pkill -9 ray || true

# 4) megatron / torch.distributed workers that may outlive their launcher.
log "stopping megatron / torch.distributed workers ..."
pkill -9 -f "megatron" || true
pkill -9 -f "torch.distributed.run" || true
pkill -9 -f "torchrun" || true

# 5) orphaned multiprocessing / spawn workers from the rollout pool.
log "stopping orphaned multiprocessing workers ..."
pkill -9 -f "multiprocessing.spawn" || true
pkill -9 -f "multiprocessing.resource_tracker" || true

# 6) GPU-holding / keep-alive tasks (self-started placeholder that pins GPUs).
#    Kill the multi_dispatch.sh launcher FIRST so it stops re-spawning via ssh,
#    then the dynamic_gpu_task.py children it left behind.
log "stopping GPU keep-alive tasks (multi_dispatch / dynamic_gpu_task) ..."
pkill -9 -f "multi_dispatch.sh" || true
pkill -9 -f "dynamic_gpu_task.py" || true

# 6b) stray GPU monitors. A bare `nvitop` (without --once) is a persistent TUI
#     that keeps polling every GPU forever. gpu_status.sh only ever runs
#     `nvitop --once` (self-terminating), so nothing here should leak from it;
#     this reaps any interactive nvitop a human started by hand and forgot.
#     Bracketed pattern so this pkill never matches another helper's cmdline
#     that merely CONTAINS the word "nvitop" (e.g. this cleanup fan-out itself).
log "stopping stray GPU monitors (nvitop) ..."
pkill -9 -f "[n]vitop" || true

# 7) port-based fallback kill. The process-name kills above miss any engine
#    that re-exec'd or whose cmdline no longer matches (yet still holds a bound
#    port). A stale holder here is exactly what makes the NEXT run die with
#    `zmq.error.ZMQError: Address already in use`. Force-kill whatever still
#    listens on the stack's known ports:
#      6379  ray GCS/head            8265  ray dashboard
#      13141 teacher sglang server   15000-15040 sglang rollout engine HTTP
#    (The dynamic sglang dist/ZMQ ports like 18207 belong to `sglang` procs
#    already reaped in step 1; they are freed once that process dies.)
CLEAN_PORTS=(6379 8265 13141)
for p in $(seq 15000 15040); do CLEAN_PORTS+=("${p}"); done
log "freeing known ports (${CLEAN_PORTS[0]}..${CLEAN_PORTS[-1]}) ..."
for p in "${CLEAN_PORTS[@]}"; do
   fuser -k -9 "${p}/tcp" 2>/dev/null || true
done

# 8) clear stale ray session state + sglang shared-memory segments so the next
#    `ray start` / engine boot is clean (shm_broadcast leaves /dev/shm files).
log "clearing /tmp/ray and sglang /dev/shm ..."
rm -rf /tmp/ray/* 2>/dev/null || true
rm -f /dev/shm/sglang* /dev/shm/nccl-* 2>/dev/null || true

sleep 2

# 9) verify: no target process survived and no known port is still LISTENing.
#    Use bracketed patterns so pgrep never matches this script's own cmdline.
#    Exit non-zero on residue so `dispatch.sh clean` reports the node as failed
#    instead of silently letting the next launch collide.
log "verifying ..."
rc=0
leftover="$(pgrep -af '[s]glang|[t]rain_async.py|[r]aylet|[g]cs_server' || true)"
if [[ -n "${leftover}" ]]; then
   log "WARNING: processes still alive after cleanup:"
   echo "${leftover}" | sed "s/^/[cleanup:${HOST}]   /"
   rc=1
fi
busy=""
for p in "${CLEAN_PORTS[@]}"; do
   if ss -ltnH 2>/dev/null | grep -q ":${p} "; then
      busy+=" ${p}"
   fi
done
if [[ -n "${busy}" ]]; then
   log "WARNING: ports still LISTENing after cleanup:${busy}"
   rc=1
fi

if [[ "${rc}" -eq 0 ]]; then
   log "done (clean: no target procs, no busy ports)."
else
   log "done WITH RESIDUE (rc=1); re-run clean or inspect the node."
fi
exit "${rc}"
