"""CPU tests for the truncated top-k OPD losses (topk_forward_kl / topk_reverse_kl)."""

import sys
import types
from argparse import Namespace

import pytest
import torch

NUM_GPUS = 0

VOCAB, K = 32, 4
PROMPT_LEN, RESPONSE_LEN = 2, 4
TOTAL_LEN = PROMPT_LEN + RESPONSE_LEN

ARGS = Namespace(
    rollout_temperature=1.0,
    allgather_cp=False,
    offline_opd_topk_overlap=True,
)


@pytest.fixture
def opd_topk_kl_loss(monkeypatch):
    """``_opd_topk_kl_loss`` imported with ``megatron.core.mpu`` stubbed: CPU, no TP/CP."""
    stubbed = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    previous = {name: sys.modules.pop(name, None) for name in stubbed}

    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    try:
        from slime.backends.megatron_utils.loss import _opd_topk_kl_loss

        yield _opd_topk_kl_loss
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _batch(ids, teacher_lp, direction):
    ids_key, lp_key = (
        ("teacher_topk_ids", "teacher_topk_logprobs")
        if direction == "forward"
        else ("student_support_ids", "teacher_logprobs_on_student")
    )
    return {
        "unconcat_tokens": [torch.arange(TOTAL_LEN)],
        "total_lengths": [TOTAL_LEN],
        "response_lengths": [RESPONSE_LEN],
        # Required by the masked-max diagnostics; all-trainable so the assertions below
        # can compare against the unmasked reference.
        "loss_masks": [torch.ones(RESPONSE_LEN, dtype=torch.int)],
        ids_key: [ids],
        lp_key: [teacher_lp],
    }


def _support_ids():
    return torch.stack([torch.randperm(VOCAB)[:K] for _ in range(RESPONSE_LEN)])


def _student_logprobs_on_support(logits, ids):
    """``log pi_s`` at the support ids, on the rows that predict the response tokens."""
    log_probs = logits.squeeze(0).log_softmax(dim=-1)
    return log_probs[PROMPT_LEN - 1 : TOTAL_LEN - 1].gather(1, ids)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_kl_weight_is_renormalized_over_the_support(opd_topk_kl_loss, direction):
    torch.manual_seed(0)
    logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = _support_ids()
    # Sub-normalized over K, as real full-vocab teacher logprobs are.
    teacher_lp = torch.randn(RESPONSE_LEN, K).log_softmax(dim=-1) - 0.7

    loss, log = opd_topk_kl_loss(
        ARGS, _batch(ids, teacher_lp, direction), logits, lambda x: x.sum(), direction=direction
    )

    student_lp = _student_logprobs_on_support(logits, ids)
    log_weight, log_ratio = (
        (teacher_lp, teacher_lp - student_lp)
        if direction == "forward"
        else (student_lp, student_lp - teacher_lp)
    )
    weight = log_weight.softmax(dim=-1)
    expected = (weight * log_ratio).sum(dim=-1).sum()

    torch.testing.assert_close(loss, expected)
    # softmax over log-probs == pi(v) / sum_{u in S} pi(u): the renormalized support weight.
    torch.testing.assert_close(weight, log_weight.exp() / log_weight.exp().sum(dim=-1, keepdim=True))
    torch.testing.assert_close(weight.sum(dim=-1), torch.ones(RESPONSE_LEN))
    torch.testing.assert_close(log[f"opd_{direction}_kl"], loss.detach())


def test_forward_kl_is_invariant_to_the_teacher_support_mass(opd_topk_kl_loss):
    """What renormalizing the weight buys: a per-position scale that no longer depends on
    how much of its own mass the teacher's top-k captures.

    Adding a constant to a row of teacher log-probs changes ``sum_{v in S} pi_t(v)``
    without touching its shape. With the raw ``pi_t`` as weight that rescaled the whole
    position; now it may only shift the value by that constant (weights sum to 1) and must
    leave the gradient alone.
    """
    torch.manual_seed(0)
    base_logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = _support_ids()
    teacher_lp = torch.randn(RESPONSE_LEN, K).log_softmax(dim=-1) - 0.7
    shift = 2.0

    def loss_and_grad(lp):
        logits = base_logits.clone().requires_grad_(True)
        loss, _ = opd_topk_kl_loss(
            ARGS, _batch(ids, lp, "forward"), logits, lambda x: x.sum(), direction="forward"
        )
        loss.backward()
        return loss.detach(), logits.grad

    loss, grad = loss_and_grad(teacher_lp)
    shifted_loss, shifted_grad = loss_and_grad(teacher_lp - shift)

    torch.testing.assert_close(shifted_grad, grad)
    torch.testing.assert_close(shifted_loss, loss - shift * RESPONSE_LEN)


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_infinite_teacher_logprob_does_not_poison_the_gradient(opd_topk_kl_loss, direction):
    """A -inf teacher log-prob must stay out of the log-ratio.

    The teacher server's non-base64 top-k path serializes an unscored id as
    ``-Infinity``. Unguarded, ``log_ratio`` becomes +-inf and ``weight * log_ratio`` is inf
    where the weight is positive and NaN where it is 0 -- the sum then carries that into
    every parameter's gradient for the whole micro-batch.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, TOTAL_LEN, VOCAB, requires_grad=True)
    ids = _support_ids()
    teacher_lp = torch.randn(RESPONSE_LEN, K).log_softmax(dim=-1) - 0.7
    teacher_lp[0, 0] = float("-inf")

    loss, log = opd_topk_kl_loss(
        ARGS, _batch(ids, teacher_lp, direction), logits, lambda x: x.sum(), direction=direction
    )
    loss.backward()

    assert torch.isfinite(loss), loss
    assert torch.isfinite(logits.grad).all()
    # exp(floor) underflows to 0, so the support-mass diagnostic is unchanged by clamping.
    assert torch.isfinite(log["teacher_mass"])


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_chunked_topk_logprobs_match_unchunked_loss_and_gradient(opd_topk_kl_loss, direction):
    torch.manual_seed(0)
    base_logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = _support_ids()
    teacher_lp = torch.randn(RESPONSE_LEN, K).log_softmax(dim=-1) - 0.7

    def loss_and_grad(chunk_size):
        args = Namespace(rollout_temperature=1.0, allgather_cp=False, log_probs_chunk_size=chunk_size)
        logits = base_logits.clone().requires_grad_(True)
        loss, log = opd_topk_kl_loss(
            args,
            _batch(ids, teacher_lp, direction),
            logits,
            lambda x: x.sum(),
            direction=direction,
        )
        loss.backward()
        return loss.detach(), log, logits.grad

    full_loss, full_log, full_grad = loss_and_grad(-1)
    chunked_loss, chunked_log, chunked_grad = loss_and_grad(2)
    torch.testing.assert_close(chunked_loss, full_loss)
    torch.testing.assert_close(chunked_log[f"opd_{direction}_kl"], full_log[f"opd_{direction}_kl"])
    torch.testing.assert_close(chunked_grad, full_grad)


OVERLAP_K = 8


def _teacher_lp_with_top4(columns: list[list[int]]) -> torch.Tensor:
    """``[R, OVERLAP_K]`` teacher log-probs whose per-row top-4 columns are ``columns``."""
    lp = torch.full((len(columns), OVERLAP_K), -10.0)
    for row, cols in enumerate(columns):
        for rank, col in enumerate(cols):
            lp[row, col] = -1.0 - rank
    return lp.log_softmax(dim=-1)


def test_overlap_at_m_counts_head_agreement(opd_topk_kl_loss):
    """overlap@m = |TopM_student ∩ TopM_teacher| / m, read off the support's column order.

    The support arrives sorted by the own model's probability, so its first m columns are
    the student's top-m and the teacher's top-m is ``topk(teacher_lp, m)``: the intersection
    is how many of those column indices are < m.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = torch.stack([torch.randperm(VOCAB)[:OVERLAP_K] for _ in range(RESPONSE_LEN)])
    # Per row: all 4 / none / 2 / 2 of the teacher's top-4 sit in the student's top-4.
    teacher_lp = _teacher_lp_with_top4([[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 4, 5], [3, 2, 7, 6]])

    _, log = opd_topk_kl_loss(
        ARGS, _batch(ids, teacher_lp, "reverse"), logits, lambda x: x.sum(), direction="reverse"
    )

    torch.testing.assert_close(log["opd_topk_overlap_at4"], torch.tensor(1.0 + 0.0 + 0.5 + 0.5))
    # m == K compares the support with itself, so it is 1 at every position.
    torch.testing.assert_close(log["opd_topk_overlap_at8"], torch.tensor(float(RESPONSE_LEN)))
    # Widths wider than the support are skipped rather than clamped to K.
    assert "opd_topk_overlap_at64" not in log


def test_offline_opd_drift_metrics_match_saved_distribution_formulas(opd_topk_kl_loss):
    torch.manual_seed(7)
    logits = torch.randn(1, TOTAL_LEN, VOCAB)
    rollout_logits = torch.randn(1, TOTAL_LEN, VOCAB)
    teacher_logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = _support_ids()

    current_full_lp = logits.squeeze(0).log_softmax(dim=-1)[PROMPT_LEN - 1 : TOTAL_LEN - 1]
    rollout_full_lp = rollout_logits.squeeze(0).log_softmax(dim=-1)[PROMPT_LEN - 1 : TOTAL_LEN - 1]
    teacher_full_lp = teacher_logits.squeeze(0).log_softmax(dim=-1)[PROMPT_LEN - 1 : TOTAL_LEN - 1]
    current_support_lp = current_full_lp.gather(1, ids)
    rollout_support_lp = rollout_full_lp.gather(1, ids)
    teacher_support_lp = teacher_full_lp.gather(1, ids)
    sampled_tokens = torch.arange(TOTAL_LEN)[-RESPONSE_LEN:]
    rollout_sample_lp = rollout_full_lp.gather(1, sampled_tokens[:, None]).squeeze(1)

    batch = _batch(ids, teacher_support_lp, "reverse")
    batch["rollout_student_logprobs_on_support"] = [rollout_support_lp]
    batch["rollout_log_probs"] = [rollout_sample_lp]
    _, log = opd_topk_kl_loss(
        ARGS,
        batch,
        logits,
        lambda x: x.mean(),
        direction="reverse",
    )

    current_log_q = current_support_lp.log_softmax(dim=-1)
    rollout_log_q = rollout_support_lp.log_softmax(dim=-1)
    teacher_log_q = teacher_support_lp.log_softmax(dim=-1)
    current_q = current_log_q.exp()
    rollout_q = rollout_log_q.exp()
    current_vs_rollout = (
        rollout_q * (rollout_log_q - current_log_q)
    ).sum(dim=-1)
    current_vs_teacher = (
        current_q * (current_log_q - teacher_log_q)
    ).sum(dim=-1)
    rollout_vs_teacher = (
        rollout_q * (rollout_log_q - teacher_log_q)
    ).sum(dim=-1)
    current_mass = current_support_lp.exp().sum(dim=-1)
    rollout_mass = rollout_support_lp.exp().sum(dim=-1)
    teacher_mass = teacher_support_lp.exp().sum(dim=-1)

    torch.testing.assert_close(
        log["offline_opd/current_vs_rollout_sparse_kl"], current_vs_rollout.mean()
    )
    torch.testing.assert_close(
        log["offline_opd/current_mass_on_rollout_topk"], current_mass.mean()
    )
    torch.testing.assert_close(
        log["offline_opd/support_mass_retention"], (current_mass / rollout_mass).mean()
    )
    torch.testing.assert_close(
        log["offline_opd/current_vs_cached_teacher_sparse_kl"], current_vs_teacher.mean()
    )
    torch.testing.assert_close(
        log["offline_opd/teacher_kl_improvement"],
        (rollout_vs_teacher - current_vs_teacher).mean(),
    )
    torch.testing.assert_close(
        log["offline_opd/teacher_mass_on_rollout_topk"], teacher_mass.mean()
    )

    current_sample_lp = current_full_lp.gather(
        1, sampled_tokens[:, None]
    ).squeeze(1)
    importance_weight = (current_sample_lp - rollout_sample_lp).exp()
    sequence_token_mean_weight = importance_weight.mean()
    sequence_weight = (current_sample_lp - rollout_sample_lp).sum().clamp(-20, 20).exp()
    torch.testing.assert_close(
        log["offline_importance_weight_sum"], sequence_token_mean_weight
    )
    torch.testing.assert_close(
        log["offline_importance_weight_sqsum"], sequence_token_mean_weight.square()
    )
    torch.testing.assert_close(
        log["offline_importance_weight_count"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        log["offline_sequence_importance_weight_sum"], sequence_weight
    )
    torch.testing.assert_close(
        log["offline_sequence_importance_weight_sqsum"], sequence_weight.square()
    )
    torch.testing.assert_close(
        log["offline_sequence_importance_weight_count"], torch.tensor(1.0)
    )

    current_top_ids = current_full_lp.topk(K, dim=-1).indices
    sorted_rollout_ids = ids.sort(dim=-1).values
    positions = torch.searchsorted(sorted_rollout_ids, current_top_ids)
    matched = (positions < K) & sorted_rollout_ids.gather(
        1, positions.clamp_max(K - 1)
    ).eq(current_top_ids)
    expected_overlap = matched.sum(dim=-1).float().div(K).mean()
    torch.testing.assert_close(
        log["offline_opd/current_vs_rollout_topk_overlap_at4"], expected_overlap
    )
    torch.testing.assert_close(
        log["offline_opd/current_vs_rollout_topk_overlap"], expected_overlap
    )


def test_offline_importance_debug_reports_exact_local_extrema_and_tail_counts(
    opd_topk_kl_loss,
):
    torch.manual_seed(17)
    logits = torch.randn(1, TOTAL_LEN, VOCAB)
    rollout_logits = torch.randn(1, TOTAL_LEN, VOCAB)
    teacher_logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = _support_ids()
    current_full_lp = logits.squeeze(0).log_softmax(dim=-1)[PROMPT_LEN - 1 : TOTAL_LEN - 1]
    rollout_full_lp = rollout_logits.squeeze(0).log_softmax(dim=-1)[PROMPT_LEN - 1 : TOTAL_LEN - 1]
    sampled_tokens = torch.arange(TOTAL_LEN)[-RESPONSE_LEN:]
    rollout_sample_lp = rollout_full_lp.gather(1, sampled_tokens[:, None]).squeeze(1)
    teacher_support_lp = teacher_logits.squeeze(0).log_softmax(dim=-1)[
        PROMPT_LEN - 1 : TOTAL_LEN - 1
    ].gather(1, ids)
    batch = _batch(ids, teacher_support_lp, "reverse")
    batch["rollout_student_logprobs_on_support"] = [rollout_full_lp.gather(1, ids)]
    batch["rollout_log_probs"] = [rollout_sample_lp]

    ARGS.offline_opd_importance_debug = True
    try:
        _, log = opd_topk_kl_loss(
            ARGS, batch, logits, lambda x: x.mean(), direction="reverse"
        )
    finally:
        ARGS.offline_opd_importance_debug = False

    current_sample_lp = current_full_lp.gather(1, sampled_tokens[:, None]).squeeze(1)
    log_ratio = current_sample_lp - rollout_sample_lp
    weight = log_ratio.clamp(-20.0, 20.0).exp()
    torch.testing.assert_close(log["__max__offline_importance_weight_max"], weight.max())
    torch.testing.assert_close(log["__max__offline_importance_weight_neg_min"], -weight.min())
    torch.testing.assert_close(log["__max__offline_importance_log_ratio_max"], log_ratio.max())
    torch.testing.assert_close(log["__max__offline_importance_log_ratio_neg_min"], -log_ratio.min())
    torch.testing.assert_close(
        log["offline_importance_weight_gt_2_count"], (weight > 2.0).sum()
    )
    torch.testing.assert_close(
        log["offline_importance_weight_lt_0_5_count"], (weight < 0.5).sum()
    )
    torch.testing.assert_close(
        log["offline_importance_token_count"], torch.tensor(float(RESPONSE_LEN))
    )


def test_offline_importance_moments_weight_sequences_equally(opd_topk_kl_loss):
    torch.manual_seed(23)
    total_lengths = [4, 6]
    response_lengths = [2, 4]
    logits = torch.randn(1, sum(total_lengths), VOCAB)
    tokens = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8, 9, 10])]
    ids = [
        torch.stack([torch.randperm(VOCAB)[:K] for _ in range(response_length)])
        for response_length in response_lengths
    ]

    response_logits = [logits[0, 1:3], logits[0, 5:9]]
    current_full_lp = [value.log_softmax(dim=-1) for value in response_logits]
    current_sample_lp = [
        lp.gather(1, sample_tokens[-response_length:, None]).squeeze(1)
        for lp, sample_tokens, response_length in zip(
            current_full_lp, tokens, response_lengths, strict=True
        )
    ]
    desired_token_ratios = [torch.tensor([1.0, 3.0]), torch.tensor([1.0, 1.0, 1.0, 7.0])]
    rollout_sample_lp = [
        current - ratio.log()
        for current, ratio in zip(current_sample_lp, desired_token_ratios, strict=True)
    ]
    support_lp = [
        lp.gather(1, sample_ids)
        for lp, sample_ids in zip(current_full_lp, ids, strict=True)
    ]
    batch = {
        "unconcat_tokens": tokens,
        "total_lengths": total_lengths,
        "response_lengths": response_lengths,
        "loss_masks": [torch.ones(length) for length in response_lengths],
        "student_support_ids": ids,
        "teacher_logprobs_on_student": support_lp,
        "rollout_student_logprobs_on_support": support_lp,
        "rollout_log_probs": rollout_sample_lp,
    }
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        offline_opd_topk_overlap=False,
    )

    _, log = opd_topk_kl_loss(
        args, batch, logits, lambda value: value.sum(), direction="reverse"
    )

    per_sequence_token_means = torch.tensor([2.0, 2.5])
    per_sequence_product_ratios = torch.tensor([3.0, 7.0])
    torch.testing.assert_close(
        log["offline_importance_weight_sum"], per_sequence_token_means.sum()
    )
    torch.testing.assert_close(
        log["offline_importance_weight_sqsum"],
        per_sequence_token_means.square().sum(),
    )
    torch.testing.assert_close(log["offline_importance_weight_count"], torch.tensor(2.0))
    torch.testing.assert_close(
        log["offline_sequence_importance_weight_sum"],
        per_sequence_product_ratios.sum(),
    )
    torch.testing.assert_close(
        log["offline_sequence_importance_weight_sqsum"],
        per_sequence_product_ratios.square().sum(),
    )


def test_offline_metrics_do_not_change_reverse_kl_loss_or_gradient(opd_topk_kl_loss):
    torch.manual_seed(11)
    base_logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = _support_ids()
    teacher_lp = torch.randn(RESPONSE_LEN, K).log_softmax(dim=-1) - 0.4
    old_support_lp = torch.randn(RESPONSE_LEN, K).log_softmax(dim=-1) - 0.3
    rollout_sample_lp = torch.randn(RESPONSE_LEN).log_softmax(dim=0)

    def run(with_metrics: bool):
        logits = base_logits.clone().requires_grad_(True)
        batch = _batch(ids, teacher_lp, "reverse")
        if with_metrics:
            batch["rollout_student_logprobs_on_support"] = [old_support_lp]
            batch["rollout_log_probs"] = [rollout_sample_lp]
        loss, _ = opd_topk_kl_loss(
            ARGS, batch, logits, lambda x: x.sum(), direction="reverse"
        )
        loss.backward()
        return loss.detach(), logits.grad

    baseline_loss, baseline_grad = run(False)
    metrics_loss, metrics_grad = run(True)
    torch.testing.assert_close(metrics_loss, baseline_loss)
    torch.testing.assert_close(metrics_grad, baseline_grad)


def test_offline_overlap_switch_skips_only_full_vocab_metrics(
    opd_topk_kl_loss, monkeypatch
):
    logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = _support_ids()
    teacher_lp = torch.randn(RESPONSE_LEN, K).log_softmax(dim=-1)
    rollout_lp = torch.randn(RESPONSE_LEN, K).log_softmax(dim=-1)
    batch = _batch(ids, teacher_lp, "reverse")
    batch["rollout_student_logprobs_on_support"] = [rollout_lp]

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("full-vocabulary top-k must be disabled")

    monkeypatch.setitem(
        opd_topk_kl_loss.__globals__, "vocab_parallel_topk_ids", fail_if_called
    )
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        offline_opd_topk_overlap=False,
    )
    _, log = opd_topk_kl_loss(
        args, batch, logits, lambda value: value.mean(), direction="reverse"
    )

    assert "offline_opd/current_vs_rollout_sparse_kl" in log
    assert not any(
        key.startswith("offline_opd/current_vs_rollout_topk_overlap") for key in log
    )


def test_offline_current_vs_rollout_overlap_reports_all_requested_widths(
    opd_topk_kl_loss,
):
    torch.manual_seed(19)
    vocab, support_width = 128, 64
    logits = torch.randn(1, TOTAL_LEN, vocab)
    current_lp = logits.squeeze(0).log_softmax(dim=-1)[
        PROMPT_LEN - 1 : TOTAL_LEN - 1
    ]
    current_top_ids = current_lp.topk(support_width, dim=-1).indices
    # Same top-64 set, but different rollout head ordering on each response row.
    ids = torch.stack(
        [
            torch.roll(current_top_ids[row], shifts=-4 * row)
            for row in range(RESPONSE_LEN)
        ]
    )
    teacher_lp = torch.randn(RESPONSE_LEN, support_width).log_softmax(dim=-1)
    rollout_lp = torch.randn(RESPONSE_LEN, support_width).log_softmax(dim=-1)
    batch = _batch(ids, teacher_lp, "reverse")
    batch["rollout_student_logprobs_on_support"] = [rollout_lp]

    _, log = opd_topk_kl_loss(
        ARGS, batch, logits, lambda value: value.mean(), direction="reverse"
    )

    for width in (4, 8, 64):
        old = ids[:, :width].sort(dim=-1).values
        new = current_top_ids[:, :width].contiguous()
        positions = torch.searchsorted(old, new)
        matched = (positions < width) & old.gather(
            1, positions.clamp_max(width - 1)
        ).eq(new)
        expected = matched.sum(dim=-1).float().div(width).mean()
        torch.testing.assert_close(
            log[f"offline_opd/current_vs_rollout_topk_overlap_at{width}"], expected
        )

    # The un-suffixed metric is overlap at the stored support's maximum K.
    torch.testing.assert_close(
        log["offline_opd/current_vs_rollout_topk_overlap"],
        log["offline_opd/current_vs_rollout_topk_overlap_at64"],
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
