"""Fully-async rollout for slime.

Decouples ``max_concurrent_tasks`` from ``rollout_batch_size``: a background
asyncio worker keeps a fixed pool of in-flight trajectories across rollout
boundaries, so the next training step doesn't have to wait for the slowest
in-flight sample to finish.

Use with ``--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async``.
Plug in per-sample logic via ``--custom-generate-function-path`` and
per-sample reward via ``--custom-rm-path`` — the worker calls slime's stock
:func:`generate_and_rm_group` which dispatches to those.

Concurrency is sourced from ``args.sglang_server_concurrency``, scaled by the number of
sglang engines and divided by ``n_samples_per_prompt`` so the GROUP pool matches the
per-SAMPLE semaphore cap in :mod:`slime.rollout.sglang_rollout`.

The worker is intentionally oblivious to slime's higher-level pause /
weight-update signalling (e.g. ``GenerateState.aborted``). Each in-flight
generation short-circuits on those signals on its own and surfaces
:data:`Sample.Status.ABORTED`; the only piece the worker owns is
**redirecting ABORTED groups back to ``data_buffer``** instead of shipping
them to training, so the next rollout (with refreshed weights) can pick
them up.
"""

from __future__ import annotations

import asyncio
import atexit
import logging
import queue
import threading
import time
from collections import Counter

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.http_utils import get_rollout_num_engines
from slime.utils.types import Sample

__all__ = [
    "AsyncRolloutWorker",
    "generate_rollout_fully_async",
]

logger = logging.getLogger("slime.rollout.fully_async")

# Cadence (s) of the in-flight stage histogram. The worker already wakes once a
# second, so this only adds a Counter over the live groups -- no extra awaits.
STAGE_LOG_EVERY = 30.0

# Least- to most-advanced. A group's slot is released only when its LAST sample is
# done, so the least-advanced sample is what the slot is waiting on.
_STAGE_ORDER = ("boot", "generate", "score", "abort", "done")


def _sample_stage(sample: Sample) -> str:
    """Coarse pipeline stage of one in-flight sample, read off fields the stages mutate
    in place (a generate function hands back the very Sample it was given), so no
    cross-module bookkeeping is needed:

      * ``reward``: set by ``generate_and_rm`` only after the reward hook returns.
      * ``metadata["swe_reward"]``: stashed by the generate function once the rollout is
        over, i.e. the group is inside the reward hook -- for OPD, the teacher logprob
        fetch, the stage worth seeing separately.
      * ``response_length``: bumped per turn, so 0 means still booting the environment or
        waiting on the turn-0 prefill.
    """
    if sample.status == Sample.Status.ABORTED:
        return "abort"
    if sample.reward is not None:
        return "done"
    if (sample.metadata or {}).get("swe_reward") is not None:
        return "score"
    if sample.response_length:
        return "generate"
    return "boot"


def _group_stage(group: list[Sample]) -> str:
    return min((_sample_stage(s) for s in group), key=_STAGE_ORDER.index, default="boot")


# Abort-reason buckets for the rollout/abort/by_reason/* metric. The raw
# ``metadata["abort_reason"]`` is high-cardinality (10 distinct strings in
# coding_agent_opd, several carrying an interpolated instance id / exception
# message), which would spray one wandb series per variant. Collapse to the two
# families worth separating -- ``loop_abort:*`` (generation was preempted or the
# token loop hit an unrecoverable state; in practice ~all of it is weight-update
# preemption) and ``pod_unavailable:*`` (sandbox could not be obtained) -- plus a
# catch-all. ``_OTHER`` also absorbs samples whose status was set to ABORTED
# generically by ``Sample._update_terminal_info`` (finish_reason == "abort"),
# which never writes an abort_reason at all.
_ABORT_REASON_BUCKETS = ("loop_abort", "pod_unavailable")
_ABORT_REASON_OTHER = "__other__"


def _abort_reason_bucket(reason: str) -> str:
    for bucket in _ABORT_REASON_BUCKETS:
        if reason.startswith(bucket):
            return bucket
    return _ABORT_REASON_OTHER


# Global worker, shared across rollout calls so the queue stays warm.
_global_worker: AsyncRolloutWorker | None = None
_worker_lock = threading.Lock()


def _get_global_worker(args, data_buffer) -> AsyncRolloutWorker:
    global _global_worker
    with _worker_lock:
        if _global_worker is None or not _global_worker.worker_thread.is_alive():
            logger.info("starting fully-async rollout worker")
            # One task per GROUP, but slime's generate semaphore is per SAMPLE
            # (``sglang_rollout.generate_and_rm`` acquires it once per sample), so a pool
            # sized in samples oversubscribes it by n_samples_per_prompt: the surplus tasks
            # park in the semaphore queue holding a slot and a Sample copy without ever
            # reaching an engine, and they show up as permanently "not yet generating".
            concurrency = max(
                1,
                args.sglang_server_concurrency * get_rollout_num_engines(args) // args.n_samples_per_prompt,
            )
            _global_worker = AsyncRolloutWorker(args, data_buffer, concurrency=concurrency)
            _global_worker.start()
        return _global_worker


def _stop_global_worker() -> None:
    global _global_worker
    with _worker_lock:
        if _global_worker is not None:
            _global_worker.stop()
            _global_worker = None


atexit.register(_stop_global_worker)


class AsyncRolloutWorker:
    """Background thread + asyncio loop that continuously consumes groups
    from ``data_buffer`` and runs :func:`generate_and_rm_group` on each."""

    def __init__(self, args, data_buffer, concurrency: int = 10):
        self.args = args
        self.data_buffer = data_buffer
        self.concurrency = concurrency
        self.running = True
        self.output_queue: queue.Queue[tuple[int, list[Sample]]] = queue.Queue(maxsize=1000)
        self.worker_thread: threading.Thread | None = None
        self.state = GenerateState(args)
        # Requeued-abort tally, accumulated by _make_done_cb (worker thread) and
        # drained once per rollout by _generate_rollout_async (main thread), hence
        # the lock: Counter's ``+=`` is a read-modify-write, not atomic.
        self._abort_stats: Counter[str] = Counter()
        self._abort_stats_lock = threading.Lock()

    # -- public --------------------------------------------------------------

    def start(self) -> None:
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._thread_main, name="fully-async-rollout", daemon=True)
            self.worker_thread.start()

    def stop(self) -> None:
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=5)

    def get_completed_groups(self) -> list[tuple[int, list[Sample]]]:
        completed: list[tuple[int, list[Sample]]] = []
        while True:
            try:
                completed.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return completed

    def queue_size(self) -> int:
        return self.output_queue.qsize()

    def drain_abort_stats(self) -> dict[str, int]:
        """Pop the abort tally accumulated since the last call.

        Unlike the ``abort=N`` field of the stage histogram -- a point-in-time gauge
        over the in-flight pool -- this is a monotonically accumulating event count,
        so it must be read-and-reset exactly once per rollout. The window is "since
        the previous rollout's drain", which includes aborts that happened while the
        trainer was busy, not just those inside the collection loop.
        """
        with self._abort_stats_lock:
            stats = dict(self._abort_stats)
            self._abort_stats.clear()
        return stats

    # -- internals -----------------------------------------------------------

    def _thread_main(self) -> None:
        asyncio.run(self._loop())

    async def _loop(self) -> None:
        # task -> the group it is running, so the stage histogram can read progress
        # straight off the Sample objects the stages mutate in place.
        active: dict[asyncio.Task, list[Sample]] = {}
        max_concurrent = self.concurrency
        gid_counter = 0
        last_stage_log = time.time()

        while self.running:
            try:
                # Reap done tasks
                for t in [t for t in active if t.done()]:
                    try:
                        t.result()  # results already handled in callback
                    except Exception as e:  # noqa: BLE001
                        logger.warning("fully-async task crashed: %r", e)
                    del active[t]

                # Top up.
                starved = False
                while len(active) < max_concurrent and self.running:
                    groups = self.data_buffer.get_samples(1)
                    if not groups:
                        starved = True
                        break
                    for group in groups:
                        gid = gid_counter
                        gid_counter += 1
                        task = asyncio.create_task(
                            generate_and_rm_group(
                                self.args,
                                group,
                                sampling_params=self.state.sampling_params.copy(),
                                evaluation=False,
                            )
                        )
                        task.add_done_callback(self._make_done_cb(gid))
                        active[task] = group

                # Where the fixed pool of slots sits: in_flight below the cap means the
                # data_buffer is starving it, at the cap with everything in one stage
                # names the bottleneck.
                now = time.time()
                if now - last_stage_log >= STAGE_LOG_EVERY:
                    last_stage_log = now
                    hist = Counter(_group_stage(g) for g in active.values())
                    logger.info(
                        "fully-async stages: in_flight=%d/%d out_queue=%d%s | %s",
                        len(active),
                        max_concurrent,
                        self.output_queue.qsize(),
                        " data_buffer_starved" if starved else "",
                        " ".join(f"{s}={hist.get(s, 0)}" for s in _STAGE_ORDER),
                    )

                await asyncio.sleep(1)
            except Exception as e:  # noqa: BLE001
                logger.exception("fully-async loop iteration error: %s", e)
                await asyncio.sleep(1)

        if active:
            logger.info(
                "fully-async: waiting for %d in-flight tasks to drain",
                len(active),
            )
            try:
                await asyncio.wait(list(active), timeout=30)
            except Exception:  # noqa: BLE001
                pass

    def _make_done_cb(self, gid: int):
        def _cb(done_task: asyncio.Task) -> None:
            try:
                result = done_task.result()
            except Exception:  # noqa: BLE001
                logger.exception("fully-async: process task raised")
                return
            if not isinstance(result, list):
                logger.warning(
                    "fully-async: generate_and_rm_group returned %r, expected list[Sample]; dropping",
                    type(result).__name__,
                )
                return
            # Aborted group → requeue, don't ship to training.
            aborted = [s for s in result if getattr(s, "status", None) == Sample.Status.ABORTED]
            if aborted:
                # Aborts are routine in async RL (weight-update preemption, guard
                # timeouts, pod failures), but a silent requeue hides abort churn.
                # Surface it with a colour-highlighted line so it stands out in the
                # log stream without being an error.
                raw_reasons = [
                    (getattr(s, "metadata", None) or {}).get("abort_reason", "unknown") for s in aborted
                ]
                reasons = sorted(set(raw_reasons))
                logger.info(
                    "\033[1;33m[fully-async] group %d ABORTED → requeue (reasons: %s)\033[0m",
                    gid,
                    ", ".join(reasons),
                )
                # The whole GROUP goes back, so one aborted sample drags its healthy
                # siblings along. They are NOT recomputed: sglang_rollout.generate_and_rm
                # early-returns COMPLETED/TRUNCATED samples, and the buffer requeues the
                # same objects, so their response/reward/teacher data survive intact.
                # The cost is staleness -- a sibling can sit through arbitrarily many
                # weight updates while the group keeps getting preempted. Track it
                # separately; it is invisible in any existing log line.
                with self._abort_stats_lock:
                    self._abort_stats["groups"] += 1
                    self._abort_stats["samples"] += len(aborted)
                    self._abort_stats["collateral_samples"] += len(result) - len(aborted)
                    for reason in raw_reasons:
                        self._abort_stats[f"by_reason/{_abort_reason_bucket(reason)}"] += 1
                try:
                    self.data_buffer.add_samples([result])
                except Exception:  # noqa: BLE001
                    logger.exception("fully-async: failed to requeue aborted group")
                return
            self.output_queue.put((gid, result))

        return _cb


async def _generate_rollout_async(args, rollout_id: int, data_buffer) -> RolloutFnTrainOutput:
    assert args.rollout_global_dataset
    worker = _get_global_worker(args, data_buffer)

    target = args.rollout_batch_size
    logger.info(
        "fully-async rollout %d: target=%d queue_warm=%d",
        rollout_id,
        target,
        worker.queue_size(),
    )

    collected: dict[int, list[Sample]] = {}
    started = time.time()
    last_log = started
    LOG_EVERY = 30.0

    while len(collected) < target:
        # Pull whatever's done.
        drained = 0
        for gid, group in worker.get_completed_groups():
            collected[gid] = group
            drained += 1

        if not drained:
            await asyncio.sleep(0.05)

        now = time.time()
        if now - last_log > LOG_EVERY:
            logger.info(
                "fully-async rollout %d: collected %d/%d, queue=%d, elapsed=%.1fs",
                rollout_id,
                len(collected),
                target,
                worker.queue_size(),
                now - started,
            )
            last_log = now

    # Order by sample.index for determinism (slime convention).
    def _key(group: list[Sample]) -> int:
        for s in group:
            idx = getattr(s, "index", None)
            if idx is not None:
                return int(idx)
        return 0

    out = sorted(collected.values(), key=_key)[:target]
    # Keys carry the full metric path: _log_rollout_data merges rollout_extra_metrics
    # verbatim (no prefix is added), unlike compute_metrics_from_samples' output.
    # The buckets are always emitted, zeros included, so the wandb series is
    # continuous instead of appearing only on rollouts that had aborts.
    abort_stats = worker.drain_abort_stats()
    metrics: dict[str, float] = {
        f"rollout/abort/{key}": float(abort_stats.get(key, 0))
        for key in ("samples", "groups", "collateral_samples")
    }
    metrics |= {
        f"rollout/abort/by_reason/{bucket}": float(abort_stats.get(f"by_reason/{bucket}", 0))
        for bucket in (*_ABORT_REASON_BUCKETS, _ABORT_REASON_OTHER)
    }
    logger.info(
        "fully-async rollout %d: done in %.1fs, queue_left=%d, "
        "aborted samples=%d groups=%d collateral=%d",
        rollout_id,
        time.time() - started,
        worker.queue_size(),
        int(metrics["rollout/abort/samples"]),
        int(metrics["rollout/abort/groups"]),
        int(metrics["rollout/abort/collateral_samples"]),
    )
    return RolloutFnTrainOutput(samples=out, metrics=metrics)


def generate_rollout_fully_async(args, rollout_id, data_buffer, evaluation: bool = False):
    """Slime ``--rollout-function-path`` entrypoint."""

    if evaluation:
        raise ValueError("fully-async rollout doesn't support evaluation mode")
    return run(_generate_rollout_async(args, rollout_id, data_buffer))
