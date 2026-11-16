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


__all__ = ["drop_aborted_groups"]
