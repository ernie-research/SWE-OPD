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

ARGS = Namespace(rollout_temperature=1.0, allgather_cp=False)


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
    # 0 * floor == 0 in the -p log p term as well, so the entropy stays finite too.
    assert torch.isfinite(log["teacher_entropy"])


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


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_teacher_mass_in_student_topm(opd_topk_kl_loss, direction):
    """teacher_mass_in_student_top{m} = sum_{v in TopM_student} pi_t(v), summed over positions.

    The student's top-m is ranked by the training-time student log-probs inside the support in
    both directions, so the reference here gathers the teacher probs at ``topk(student_lp, m)``.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = torch.stack([torch.randperm(VOCAB)[:OVERLAP_K] for _ in range(RESPONSE_LEN)])
    teacher_lp = torch.randn(RESPONSE_LEN, OVERLAP_K).log_softmax(dim=-1) - 0.7

    _, log = opd_topk_kl_loss(
        ARGS, _batch(ids, teacher_lp, direction), logits, lambda x: x.sum(), direction=direction
    )

    student_lp = _student_logprobs_on_support(logits, ids)
    student_top = torch.topk(student_lp, OVERLAP_K, dim=-1).indices
    for m in (4, OVERLAP_K):
        expected = teacher_lp.exp().gather(-1, student_top[:, :m]).sum(dim=-1).sum()
        torch.testing.assert_close(log[f"teacher_mass_in_student_top{m}"], expected)
    # m == K covers the whole support, i.e. teacher_mass itself.
    torch.testing.assert_close(log[f"teacher_mass_in_student_top{OVERLAP_K}"], log["teacher_mass"])
    # Widths wider than the support are skipped rather than clamped to K.
    assert "teacher_mass_in_student_top64" not in log


@pytest.mark.parametrize("direction", ["forward", "reverse"])
def test_support_entropies(opd_topk_kl_loss, direction):
    """student_entropy / teacher_entropy = -sum_{v in S} p(v) log p(v), summed over positions.

    Both are truncated to the shared support, hence lower bounds on the full-vocab entropy;
    this pins the truncated form that is logged and the one-sided direction of the bias.
    """
    torch.manual_seed(0)
    logits = torch.randn(1, TOTAL_LEN, VOCAB)
    ids = _support_ids()
    teacher_lp = torch.randn(RESPONSE_LEN, K).log_softmax(dim=-1) - 0.7

    _, log = opd_topk_kl_loss(
        ARGS, _batch(ids, teacher_lp, direction), logits, lambda x: x.sum(), direction=direction
    )

    student_lp = _student_logprobs_on_support(logits, ids)
    torch.testing.assert_close(log["student_entropy"], -(student_lp.exp() * student_lp).sum())
    torch.testing.assert_close(log["teacher_entropy"], -(teacher_lp.exp() * teacher_lp).sum())

    # Never above the student's exact full-vocab entropy: the dropped tail terms are >= 0.
    rows = logits.squeeze(0)[PROMPT_LEN - 1 : TOTAL_LEN - 1]
    full_entropy = -(rows.softmax(dim=-1) * rows.log_softmax(dim=-1)).sum()
    assert log["student_entropy"] <= full_entropy


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
