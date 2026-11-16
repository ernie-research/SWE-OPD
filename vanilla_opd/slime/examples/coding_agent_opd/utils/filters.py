"""Dynamic sampling filter: drop aborted rollout groups, halt if aborts persist.

An ABORT is *not* normal. ``generate.py`` only aborts a sample when the harness
hits an unrecoverable state -- sglang returned ``finish=="abort"``, the engine
gave back no ``output_token_logprobs``, the chat-template delta failed to render,
the pod never came up, or the wall-clock guard fired. None of these are valid
training data: an aborted sample's ``tokens`` are a ``[0, 0]`` stub with no real
trajectory, so scoring it against the teacher (OPD) or training on it is
meaningless.

The pure-distillation reference (``slime.rollout.on_policy_distillation``) never
aborts and assumes every sample carries a real teacher json. To keep that
invariant we drop any group containing an ABORTED sample here, *before* it enters
the training batch: slime's over-sampling loop (``generate_rollout_async``) then
replaces the dropped group with a fresh rollout, so ``post_process_rewards`` only
ever sees complete, teacher-scorable samples.

But dropping silently would let a systematic bug (a broken checkpoint, a template
mismatch, a dead RLE backend) burn compute forever by aborting-then-resampling in
a loop. So we also treat abort as the error it is:

  * every dropped group is logged at WARNING with its abort reasons;
  * once the cumulative number of dropped groups this process exceeds
    ``CODING_OPD_MAX_ABORTED_GROUPS`` (env, default 8) we ``raise`` -- which
    propagates out of the rollout and halts training, so an operator investigates
    instead of training on a trickle of degraded/duplicated data.

Wire via ``--dynamic-sampling-filter-path examples.coding_agent_opd.utils.filters.drop_aborted_groups``.

This module also hosts ``pop_longest_first``, the ``--buffer-filter-path`` hook that
orders the fully-async requeue backlog deepest-trajectory-first (see its docstring).
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace

from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

# Cumulative aborted-group budget for the whole process. Kept deliberately small:
# a healthy run aborts ~never, so this only trips when something is structurally
# wrong (bad checkpoint / template drift / RLE backend down), at which point we
# want to fail loud rather than silently resample.
_MAX_ABORTED_GROUPS = int(os.environ.get("CODING_OPD_MAX_ABORTED_GROUPS", "8"))

# Process-cumulative count of groups dropped for containing an aborted sample.
_aborted_groups_dropped = 0


def _flatten(group) -> list[Sample]:
    """A group is ``list[Sample]`` (our custom generate) or ``list[list[Sample]]``."""
    flat: list[Sample] = []
    for item in group:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)
    return flat


def drop_aborted_groups(args: Namespace, group) -> DynamicFilterOutput:
    """Keep only groups with no ABORTED sample; halt training if aborts persist.

    Returns ``keep=True`` for clean groups. For a group with any ABORTED sample:
    logs a warning with the per-sample abort reasons, counts it against the
    process budget, and either drops it (``keep=False``) so over-sampling refills
    the batch, or -- once the cumulative budget is exceeded -- raises to interrupt
    training (abort is a harness error, not something to tolerate indefinitely).
    """
    samples = _flatten(group)
    aborted = [s for s in samples if s.status == Sample.Status.ABORTED]
    if not aborted:
        return DynamicFilterOutput(keep=True)

    global _aborted_groups_dropped
    _aborted_groups_dropped += 1

    reasons = [(s.metadata or {}).get("abort_reason", "unknown") for s in aborted]
    instance = next(
        ((s.metadata or {}).get("instance_id") for s in aborted if (s.metadata or {}).get("instance_id")),
        "?",
    )
    logger.warning(
        "[coding_agent_opd] dropping aborted group instance=%s reasons=%s "
        "(%d/%d aborted-group budget)",
        instance,
        reasons,
        _aborted_groups_dropped,
        _MAX_ABORTED_GROUPS,
    )

    if _aborted_groups_dropped > _MAX_ABORTED_GROUPS:
        raise RuntimeError(
            f"[coding_agent_opd] aborted-group drops exceeded "
            f"CODING_OPD_MAX_ABORTED_GROUPS={_MAX_ABORTED_GROUPS}. Abort is an "
            f"unrecoverable harness state (sglang abort / missing output_token_logprobs "
            f"/ chat-template delta failure / pod boot failure / wall-clock timeout), "
            f"not valid training data. Persistent aborts indicate a structural problem "
            f"(broken checkpoint, template drift, or a down RLE backend) -- halting "
            f"training for investigation. Most recent reasons={reasons} instance={instance}."
        )

    return DynamicFilterOutput(keep=False, reason="aborted_group")


__all__ = ["drop_aborted_groups", "pop_longest_first"]


# --------------------------------------------------------------------------
# Buffer filter: requeue-backlog priority
# --------------------------------------------------------------------------
def _group_priority(group: list[Sample]) -> tuple[int, int]:
    """(best turns reached, attempts) over a requeued group, from abort metadata.

    Both keys are written by ``utils.rollout._abort_sample`` and accumulate across
    attempts because ``generate()``'s per-attempt reset leaves ``metadata`` alone. A
    group's value is the max over its samples: with ``n_samples_per_prompt > 1`` the
    siblings share one instance, and the deepest attempt is what says how expensive
    the instance is.
    """
    best_turns = 0
    attempts = 0
    for s in group:
        md = s.metadata or {}
        best_turns = max(best_turns, int(md.get("turns_best", 0)))
        attempts = max(attempts, int(md.get("attempts", 0)))
    return best_turns, attempts


def pop_longest_first(args, rollout_id, buffer: list[list[Sample]], num_samples: int) -> list[list[Sample]]:
    """Buffer filter: dequeue requeued groups deepest-trajectory-first.

    Wire via ``--buffer-filter-path examples.coding_agent_opd.utils.filters.pop_longest_first``
    (replaces slime's default ``data_source.pop_first``).

    ``RolloutDataSourceWithBuffer.buffer`` holds *only* groups the fully-async worker
    requeued after an abort (fresh prompts come from the dataset path), and the buffer is
    already drained before the dataset, so this changes nothing about buffer-vs-dataset
    priority -- only the ORDER WITHIN the abort backlog. Long trajectories are the ones
    that keep losing the race (measured: abort count and turns reached are strictly
    monotone, and a handful of instances never produced a single complete trajectory), so
    when the backlog is larger than the number of free slots, restarting the deepest /
    most-retried instances first is what gives them a whole inter-update window instead of
    the tail of one.

    Must mutate ``buffer`` in place: the caller passes ``self.buffer`` itself and relies on
    the filter to remove what it returns (same contract as ``pop_first``).
    """
    num_to_pop = min(len(buffer), num_samples)
    if num_to_pop <= 0:
        return []

    # Stable sort => groups with equal priority keep FIFO order, so this degrades to
    # pop_first exactly when no group carries abort metadata (e.g. a buffer populated by
    # some other path).
    order = sorted(range(len(buffer)), key=lambda i: _group_priority(buffer[i]), reverse=True)
    chosen = order[:num_to_pop]
    top_turns, top_attempts = _group_priority(buffer[chosen[0]])
    # Return in buffer order (the worker asks for one group at a time, so this only matters
    # for a batched caller) but delete high-index-first so the earlier indices stay valid.
    keep_order = sorted(chosen)
    samples = [buffer[i] for i in keep_order]
    for i in reversed(keep_order):
        del buffer[i]

    logger.info(
        "[coding_agent_opd] buffer_filter: popped %d of %d requeued groups, "
        "top priority turns_best=%d attempts=%d",
        num_to_pop,
        num_to_pop + len(buffer),
        top_turns,
        top_attempts,
    )
    return samples
