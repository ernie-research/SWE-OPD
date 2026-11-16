"""Coding-Agent OPD: reward post-processing (KL teacher + SWE test reward).

The launcher wires the two hooks in as:

  --custom-rm-path                 examples.coding_agent_opd.opd.reward_func
  --custom-reward-post-process-path examples.coding_agent_opd.opd.post_process_rewards

``reward_func`` is re-exported unchanged from ``slime.rollout.on_policy_distillation``: it
POSTs the token sequence to the external sglang *teacher* (``args.rm_url``) and returns the
raw json, which slime stashes on ``sample.reward`` until post-processing.

``post_process_rewards`` promotes ``sample.metadata["swe_reward"]`` to the scalar reward in
both OPD modes, and under ``--opd-type sglang`` also attaches the teacher's logprobs (which
fields depends on ``--opd-loss-type``). Under ``--opd-type megatron`` nothing rides these
hooks: ``generate.py`` sets ``sample.reward`` directly (so ``async_rm`` never runs) and an
in-backend forward fills ``rollout_data["teacher_log_probs"]`` instead.
"""

from __future__ import annotations

import logging

import torch

from slime.utils.types import Sample

# Re-exported so --custom-rm-path can point at a single coding_agent_opd import path.
from slime.rollout.on_policy_distillation import (  # noqa: F401
    build_student_per_position_ids,
    extract_teacher_logprobs_on_student_ids,
    extract_teacher_topk,
    extract_teacher_topk_overlap,
    reward_func,
)

logger = logging.getLogger(__name__)

# Flip after the first token-in alignment probe so the (hot) reward path only
# checks once per process. See _warn_once_if_teacher_misaligned.
_TEACHER_ALIGNMENT_CHECKED = False


def _logprob_of(item) -> float:
    """Scalar logprob from one sglang ``input_token_logprobs`` entry: either a bare float
    or a ``[logprob, token_id, token_text]`` tuple, depending on the sglang build."""
    if isinstance(item, (list, tuple)):
        return float(item[0])
    return float(item)


def _needs_sglang_teacher_logprobs(args) -> bool:
    """True iff teacher log-probs ride this reward hook (sglang OPD mode only)."""
    return getattr(args, "opd_type", None) == "sglang"


def _warn_once_if_teacher_misaligned(sample: Sample, reward: dict) -> None:
    """Coarse, once-per-process check that the teacher scores the student's exact token
    grid (token-in). See docs/opd_tito_same_tokenizer.md.
    """
    global _TEACHER_ALIGNMENT_CHECKED
    if _TEACHER_ALIGNMENT_CHECKED:
        return
    _TEACHER_ALIGNMENT_CHECKED = True

    n_teacher = len(reward["meta_info"]["input_token_logprobs"])
    n_student = len(sample.tokens)
    if n_teacher != n_student:
        logger.warning(
            "OPD teacher returned %d token logprobs for a %d-token student "
            "sequence. Token-in teacher scoring assumes student and teacher share "
            "one tokenizer/vocab so logprobs align 1:1 with student tokens; this "
            "length mismatch suggests the teacher (args.rm_url) re-tokenized or "
            "uses a different vocab, which would corrupt the OPD KL. Verify "
            "--rm-url points at a same-tokenizer teacher "
            "(see docs/opd_tito_same_tokenizer.md).",
            n_teacher,
            n_student,
        )


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Emit SWE-test scalar rewards; in sglang mode also attach the teacher's logprobs.

    Returns ``(scalar_rewards, raw_rewards)``, both the SWE test reward stashed by
    ``generate.py`` (no raw-vs-scaled distinction here). Which teacher fields get
    attached follows ``--opd-loss-type``, as in
    ``slime.rollout.on_policy_distillation.post_process_rewards``.

    Aborted samples never reach here -- the ``drop_aborted_groups`` dynamic sampling
    filter discards their whole group first -- so every sample carries a real teacher
    json; the 0.0 fallback only covers a missing ``swe_reward`` key.
    """
    scalar_rewards = [float((sample.metadata or {}).get("swe_reward", 0.0)) for sample in samples]

    # Clear every teacher-derived field before filling in the ones this loss type uses.
    # The branches below SKIP a sample whose extraction returns None, so without this a
    # failed fetch leaves whatever was already there. Nothing downstream validates these
    # against response_length (Sample._validate_response_metadata_lengths covers only
    # loss_mask / rollout_log_probs / the top-p pair), so a stale [R, K] block of a
    # matching length would reach _opd_topk_kl_loss and pair an old teacher distribution
    # with new student logits -- wrong numbers, no error. Unconditional because megatron
    # mode fills teacher_log_probs from an in-backend forward that runs after this hook.
    for sample in samples:
        sample.teacher_log_probs = None
        sample.teacher_topk_logprobs = None
        sample.teacher_topk_ids = None
        sample.student_support_ids = None
        sample.teacher_logprobs_on_student = None
        sample.opd_topk_overlap = None

    if _needs_sglang_teacher_logprobs(args):
        # After async_rm, sample.reward holds the teacher server's raw json.
        raw_rewards = [sample.get_reward_value(args) for sample in samples]

        loss_type = getattr(args, "opd_loss_type", "token_reverse_kl")
        teacher_topk = int(getattr(args, "opd_id_topk", 0) or 0)

        if loss_type == "topk_reverse_kl" and teacher_topk > 0:
            # Teacher was queried on the student's per-position top-k ids at the
            # trainable positions only: scatter those ids and the teacher logprobs at
            # them back to the [R, K] each that the loss indexes by response position.
            for sample, reward in zip(samples, raw_rewards, strict=False):
                per_pos = build_student_per_position_ids(sample)
                if per_pos is None:
                    continue
                rows, positions = per_pos
                teacher_lp = extract_teacher_logprobs_on_student_ids(
                    reward, sample.response_length, teacher_topk, positions
                )
                if teacher_lp is None:
                    continue
                support_ids = torch.zeros((sample.response_length, teacher_topk), dtype=torch.long)
                support_ids[torch.from_numpy(positions).long()] = torch.from_numpy(rows).long()
                sample.student_support_ids = support_ids
                sample.teacher_logprobs_on_student = teacher_lp
                sample.opd_topk_overlap = extract_teacher_topk_overlap(
                    reward, sample.response_length, teacher_topk, positions
                )
            return scalar_rewards, scalar_rewards

        if loss_type == "topk_forward_kl" and teacher_topk > 0:
            # Return early like the branch above: the teacher scored the response region
            # only (reward_func narrows logprob_start_len), so the alignment probe below
            # -- which expects one logprob per POSTed token -- no longer applies, and
            # _opd_topk_kl_loss never reads sample.teacher_log_probs.
            for sample, reward in zip(samples, raw_rewards, strict=False):
                topk = extract_teacher_topk(reward, sample.response_length)
                if topk is not None:
                    sample.teacher_topk_logprobs, sample.teacher_topk_ids = topk
            return scalar_rewards, scalar_rewards

        # Once-per-process, coarse token-in alignment probe (warns, never raises).
        if samples:
            _warn_once_if_teacher_misaligned(samples[0], raw_rewards[0])

        for sample, reward in zip(samples, raw_rewards, strict=False):
            t_log_probs = torch.tensor(
                [_logprob_of(item) for item in reward["meta_info"]["input_token_logprobs"][1:]],
                dtype=torch.float32,
            )
            sample.teacher_log_probs = t_log_probs[-sample.response_length :]

            # token_reverse_kl with --opd-id-topk > 0: reward_func still asks for the
            # teacher's own top-k (full window, JSON list); the token-level loss ignores
            # it, kept for diagnostics.
            topk = extract_teacher_topk(reward, sample.response_length)
            if topk is not None:
                sample.teacher_topk_logprobs, sample.teacher_topk_ids = topk

    return scalar_rewards, scalar_rewards
