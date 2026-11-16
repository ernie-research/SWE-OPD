"""Exact, continuously-fed rollout used only by offline OPD Phase 1.

A persistent asyncio pool spans output-file boundaries.  It keeps a fixed number of
dataset groups in flight, replacing each completed group immediately, while the caller
periodically drains a small batch for atomic ``rollout_N.pt`` persistence.  Every
dataset row is submitted exactly once (apart from retry-in-place after an abort).
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
from collections import deque
from pathlib import Path

import torch

from slime.rollout.base_types import RolloutFnTrainOutput
from slime.rollout.sglang_rollout import GenerateState, generate_and_rm_group
from slime.utils.async_utils import run
from slime.utils.types import Sample


logger = logging.getLogger(__name__)


def _group_index(group) -> int:
    sample = group[0][0] if isinstance(group[0], list) else group[0]
    if sample.index is None:
        raise RuntimeError("offline OPD sample has no dataset index")
    return int(sample.index)


def _aborted_samples(group) -> list[Sample]:
    samples: list[Sample] = []
    for item in group:
        samples.extend(item if isinstance(item, list) else [item])
    return [sample for sample in samples if sample.status == Sample.Status.ABORTED]


async def _generate_group_with_retry(args, group, sampling_params, max_attempts: int):
    # generate_and_rm_group (and the coding-agent generate function below it) mutate
    # Sample objects in place.  Retrying the returned ABORTED object therefore also
    # carries a stale routing session and any rollout state which a custom generator
    # forgot to clear.  Keep an immutable, pre-attempt template and start every try
    # from a clean copy.  This is deliberately local to offline Phase 1; online OPD's
    # partial-rollout path must continue resuming the same object/session.
    template = copy.deepcopy(group)
    for attempt in range(1, max_attempts + 1):
        attempt_group = copy.deepcopy(template)
        for sample in _all_samples(attempt_group):
            sample.status = Sample.Status.PENDING
            sample.session_id = None
        result = await generate_and_rm_group(
            args, attempt_group, sampling_params=sampling_params, evaluation=False
        )
        aborted = _aborted_samples(result)
        if not aborted:
            return result
        reasons = [(sample.metadata or {}).get("abort_reason", "unknown") for sample in aborted]
        if attempt == max_attempts:
            raise RuntimeError(f"offline OPD rollout exhausted {max_attempts} attempts for sample index={group[0].index}; reasons={reasons}")
        logger.warning(
            "offline OPD retrying the same dataset sample index=%s after abort (attempt %d/%d, reasons=%s)",
            group[0].index,
            attempt,
            max_attempts,
            reasons,
        )
        await asyncio.sleep(min(attempt, 5))


def _all_samples(group) -> list[Sample]:
    samples: list[Sample] = []
    for item in group:
        samples.extend(item if isinstance(item, list) else [item])
    return samples


def _persisted_indices(args, rollout_id: int, expected_count: int, dataset_size: int) -> set[int]:
    """Read the completion-order files from a previous interrupted run."""
    if rollout_id == 0:
        return set()
    template = args.save_debug_rollout_data
    if not template:
        raise RuntimeError("offline OPD resume requires --save-debug-rollout-data")

    indices: set[int] = set()
    for rid in range(rollout_id):
        path = Path(template.format(rollout_id=rid))
        if not path.is_file():
            raise RuntimeError(f"offline OPD resume file is missing: {path}")
        blob = torch.load(path, map_location="cpu", weights_only=False)
        rows = blob.get("samples") if isinstance(blob, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError(f"offline OPD resume file has no sample list: {path}")
        for row in rows:
            index = row.get("index") if isinstance(row, dict) else getattr(row, "index", None)
            if not isinstance(index, int) or not 0 <= index < dataset_size:
                raise RuntimeError(f"offline OPD resume has invalid sample index={index!r}: {path}")
            if index in indices:
                raise RuntimeError(f"offline OPD resume has duplicate sample index={index}: {path}")
            indices.add(index)
    if len(indices) != expected_count:
        raise RuntimeError(
            f"offline OPD resume found {len(indices)} persisted samples, expected {expected_count}"
        )
    return indices


class _ContinuousExactPool:
    """Finite producer pool whose tasks survive across rollout file drains."""

    def __init__(self, args, groups, concurrency: int):
        self.args = args
        self.pending = deque(groups)
        self.concurrency = concurrency
        self.sampling_params = GenerateState(args).sampling_params
        self.completed: asyncio.Queue = asyncio.Queue()
        self.active: set[asyncio.Task] = set()
        self.submitted = 0
        self.returned = 0
        self.running = True

    def start(self) -> None:
        while len(self.active) < self.concurrency and self.pending:
            self._launch_next()
        logger.info(
            "offline OPD continuous pool started: in_flight=%d pending=%d concurrency=%d",
            len(self.active),
            len(self.pending),
            self.concurrency,
        )

    def _launch_next(self) -> None:
        if not self.running or not self.pending:
            return
        group = self.pending.popleft()
        task = asyncio.create_task(self._run_one(group))
        self.active.add(task)
        self.submitted += 1
        task.add_done_callback(self._task_done)

    async def _run_one(self, group) -> None:
        try:
            # A 100-turn SWE trajectory can remain exposed to the external RLE
            # service for tens of minutes.  A single pathological row has already
            # seen six independent session evictions in one run, so a small global
            # cap can discard thousands of otherwise-complete exact-rollout files.
            # Keep this bounded, but high enough that infrastructure churn cannot
            # routinely abort the whole offline epoch.  Online OPD is unaffected.
            max_attempts = int(os.environ.get("OFFLINE_OPD_MAX_ATTEMPTS", "100"))
            if max_attempts <= 0:
                raise RuntimeError(
                    f"OFFLINE_OPD_MAX_ATTEMPTS must be positive, got {max_attempts}"
                )
            result = await _generate_group_with_retry(
                self.args,
                group,
                self.sampling_params.copy(),
                max_attempts=max_attempts,
            )
            await self.completed.put((result, None))
        except BaseException as error:  # propagate to the RolloutManager instead of hanging
            await self.completed.put((None, error))

    def _task_done(self, task: asyncio.Task) -> None:
        self.active.discard(task)
        # Refill before the completed result is drained/saved by the caller.
        self._launch_next()

    async def take(self, count: int):
        outputs = []
        while len(outputs) < count:
            if not self.active and not self.pending and self.completed.empty():
                raise RuntimeError(
                    f"offline OPD continuous pool exhausted after {self.returned} samples; "
                    f"requested another {count}"
                )
            result, error = await self.completed.get()
            if error is not None:
                await self.cancel()
                raise RuntimeError("offline OPD continuous rollout task failed") from error
            outputs.append(result)
        self.returned += len(outputs)
        logger.info(
            "offline OPD continuous drain: returned=%d batch=%d in_flight=%d pending=%d ready=%d",
            self.returned,
            len(outputs),
            len(self.active),
            len(self.pending),
            self.completed.qsize(),
        )
        return outputs

    async def cancel(self) -> None:
        self.running = False
        tasks = list(self.active)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.active.clear()


_continuous_pool: _ContinuousExactPool | None = None


def _get_continuous_pool(args, rollout_id: int, data_source) -> _ContinuousExactPool:
    global _continuous_pool
    if _continuous_pool is not None:
        return _continuous_pool

    dataset_size = int(args.num_rollout) * int(args.rollout_batch_size)
    persisted_count = int(rollout_id) * int(args.rollout_batch_size)
    persisted = _persisted_indices(args, rollout_id, persisted_count, dataset_size)

    # Consume epoch zero exactly once so every row receives its deterministic index,
    # then remove whichever completion-order rows were already persisted.
    all_groups = data_source.get_samples(dataset_size)
    if len(all_groups) != dataset_size:
        raise RuntimeError(f"dataset returned {len(all_groups)} groups, expected {dataset_size}")
    remaining = [group for group in all_groups if _group_index(group) not in persisted]
    expected_remaining = dataset_size - persisted_count
    if len(remaining) != expected_remaining:
        raise RuntimeError(
            f"offline OPD resume selected {len(remaining)} remaining samples, expected {expected_remaining}"
        )

    concurrency = int(os.environ.get("OFFLINE_OPD_CONCURRENCY", "100"))
    if concurrency <= 0:
        raise RuntimeError(f"OFFLINE_OPD_CONCURRENCY must be positive, got {concurrency}")
    _continuous_pool = _ContinuousExactPool(args, remaining, min(concurrency, len(remaining)))
    _continuous_pool.start()
    logger.info(
        "offline OPD finite epoch initialized: dataset=%d persisted=%d remaining=%d concurrency=%d",
        dataset_size,
        persisted_count,
        len(remaining),
        _continuous_pool.concurrency,
    )
    return _continuous_pool


async def _generate_exact(args, rollout_id: int, data_source) -> RolloutFnTrainOutput:
    pool = _get_continuous_pool(args, rollout_id, data_source)
    outputs = await pool.take(args.rollout_batch_size)
    outputs.sort(key=_group_index)
    return RolloutFnTrainOutput(samples=outputs, metrics={})


def generate_rollout_exact(args, rollout_id, data_source, evaluation: bool = False):
    if evaluation:
        raise ValueError("offline OPD exact rollout does not support evaluation")
    return run(_generate_exact(args, rollout_id, data_source))
