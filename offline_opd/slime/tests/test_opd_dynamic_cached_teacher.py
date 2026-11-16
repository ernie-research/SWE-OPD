"""CPU checks for dynamic current-support / cached-teacher offline OPD."""

import sys
import types
import hashlib
import json
from argparse import Namespace

import numpy as np
import pytest
import torch

NUM_GPUS = 0


@pytest.fixture
def bidirectional_coefficients(monkeypatch):
    stubbed = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    previous = {name: sys.modules.pop(name, None) for name in stubbed}
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    try:
        from slime.backends.megatron_utils.loss import (
            _resolve_opd_bidirectional_kl_coefficients,
        )

        yield _resolve_opd_bidirectional_kl_coefficients
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, (0.5, 0.5)),
        (10, (0.25, 0.75)),
        (20, (0.0, 1.0)),
        (40, (0.0, 1.0)),
    ],
)
def test_bidirectional_coefficients_transfer_forward_weight(
    bidirectional_coefficients, step, expected
):
    args = Namespace(
        opd_forward_kl_coef=0.5,
        opd_reverse_kl_coef=0.5,
        opd_forward_kl_anneal_steps=20,
        opd_kl_schedule_step=step,
    )
    assert bidirectional_coefficients(args) == pytest.approx(expected)


def test_coarsened_reverse_kl_tail_bucket_prevents_support_mass_escape(monkeypatch):
    stubbed = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    previous = {name: sys.modules.pop(name, None) for name in stubbed}
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = types.SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    try:
        from slime.backends.megatron_utils.loss import (
            _coarsened_reverse_kl_with_tail,
        )

        current = torch.tensor([[0.2, 0.1]], requires_grad=True)
        teacher = torch.tensor([[0.6, 0.3]])
        current_lp = current.log()
        teacher_lp = teacher.log()

        # The legacy head-only estimator is negative because current head mass
        # (0.3) is lower than Teacher head mass (0.9).
        legacy_q = current / current.sum(dim=-1, keepdim=True)
        legacy = (legacy_q * (current_lp - teacher_lp)).sum(dim=-1)
        assert legacy.item() < 0

        actual = _coarsened_reverse_kl_with_tail(current_lp, teacher_lp)
        expected = (
            current * (current_lp - teacher_lp)
        ).sum(dim=-1) + 0.7 * (torch.log(torch.tensor(0.7)) - torch.log(torch.tensor(0.1)))
        torch.testing.assert_close(actual, expected)
        assert actual.item() > 0
        actual.sum().backward()
        assert torch.isfinite(current.grad).all()
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def dynamic_loss(monkeypatch):
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
        from slime.backends.megatron_utils.loss import _opd_dynamic_cached_teacher_topk_loss

        yield _opd_dynamic_cached_teacher_topk_loss
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def cached_forward_loss(monkeypatch):
    stubbed = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    previous = {name: sys.modules.pop(name, None) for name in stubbed}
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    try:
        from slime.backends.megatron_utils.loss import (
            _opd_cached_teacher_topk_forward_loss,
        )

        yield _opd_cached_teacher_topk_forward_loss
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def cached_bidirectional_loss(monkeypatch):
    stubbed = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    previous = {name: sys.modules.pop(name, None) for name in stubbed}
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    try:
        from slime.backends.megatron_utils.loss import (
            _opd_cached_teacher_bidirectional_topk_loss,
        )

        yield _opd_cached_teacher_bidirectional_topk_loss
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def cross_tokenizer_anchor_loss(monkeypatch):
    stubbed = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    previous = {name: sys.modules.pop(name, None) for name in stubbed}
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    try:
        from slime.backends.megatron_utils.loss import (
            _opd_cross_tokenizer_anchored_bidirectional_topk_loss,
        )

        yield _opd_cross_tokenizer_anchored_bidirectional_topk_loss
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def cross_tokenizer_reverse_anchor_loss(monkeypatch):
    stubbed = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    previous = {name: sys.modules.pop(name, None) for name in stubbed}
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    try:
        from slime.backends.megatron_utils.loss import (
            _opd_cross_tokenizer_anchored_reverse_topk_loss,
        )

        yield _opd_cross_tokenizer_anchored_reverse_topk_loss
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@pytest.fixture
def cached_sampled_token_pg_loss(monkeypatch):
    stubbed = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    previous = {name: sys.modules.pop(name, None) for name in stubbed}
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    try:
        from slime.backends.megatron_utils.loss import (
            _opd_cached_teacher_sampled_token_pg_loss,
        )

        yield _opd_cached_teacher_sampled_token_pg_loss
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_cached_sampled_token_pg_uses_advantage_and_anchors_misses(
    cached_sampled_token_pg_loss,
):
    torch.manual_seed(20260902)
    prompt_len, response_len, vocab = 2, 3, 8
    total_len, cache_k = prompt_len + response_len, 4
    logits = torch.randn(1, total_len, vocab, requires_grad=True)
    response_logits = logits[0, prompt_len - 1 : total_len - 1]
    response_tokens = torch.tensor([1, 2, 3])
    current_lp = response_logits.detach().log_softmax(dim=-1).gather(
        1, response_tokens[:, None]
    ).squeeze(1)

    # Row 0 is an exact Teacher-cache hit. Row 1 is aligned but its sampled
    # token is outside Top-K. Row 2 is a tokenizer/template anchor row.
    candidate_ids = torch.tensor([[0, 1, 4, 5], [0, 3, 4, 5]], dtype=torch.int32)
    candidate_lp = torch.full((2, cache_k), -5.0, dtype=torch.float16)
    candidate_lp[0, 1] = float(current_lp[0] + 0.7)
    rollout_lp = current_lp.clone()
    batch = {
        "unconcat_tokens": [torch.cat((torch.tensor([6, 7]), response_tokens))],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "rollout_log_probs": [rollout_lp],
        "teacher_candidate_positions": [torch.tensor([0, 1], dtype=torch.int32)],
        "teacher_candidate_ids": [candidate_ids],
        "teacher_candidate_logprobs": [candidate_lp],
    }
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_kl_coef=1.0,
        opd_student_anchor_kl_coef=1.0,
        opd_sampled_token_cache_miss_policy="anchor",
        opd_token_id_bridge=None,
        eps_clip=0.2,
        eps_clip_high=0.2,
        log_probs_chunk_size=-1,
    )

    loss, log = cached_sampled_token_pg_loss(
        args, batch, logits, lambda value: value.mean()
    )

    expected_advantage = candidate_lp[0, 1].float() - current_lp[0]
    torch.testing.assert_close(loss.detach(), -expected_advantage / response_len)
    torch.testing.assert_close(
        log["offline_opd/sampled_token_exact_row_fraction"], torch.tensor(2 / 3)
    )
    torch.testing.assert_close(
        log["offline_opd/sampled_token_teacher_hit_fraction"], torch.tensor(1 / 3)
    )
    torch.testing.assert_close(
        log["offline_opd/sampled_token_teacher_cache_miss_fraction"], torch.tensor(1 / 3)
    )
    torch.testing.assert_close(
        log["offline_opd/sampled_token_rollout_anchor_fraction"], torch.tensor(2 / 3)
    )
    expected_entropy = -(
        response_logits.detach().softmax(dim=-1)
        * response_logits.detach().log_softmax(dim=-1)
    ).sum(dim=-1).mean()
    torch.testing.assert_close(
        log["entropy"], expected_entropy
    )

    loss.backward()
    # A positive Teacher-current advantage must increase the sampled token's
    # logit under gradient descent. Cache-miss/unaligned anchor rows start at
    # zero advantage because current == rollout.
    assert logits.grad[0, prompt_len - 1, response_tokens[0]] < 0
    torch.testing.assert_close(
        logits.grad[0, prompt_len : total_len - 1],
        torch.zeros_like(logits.grad[0, prompt_len : total_len - 1]),
    )


def test_cached_sampled_token_pg_current_clip_reference_keeps_gradient_without_clipping(
    cached_sampled_token_pg_loss,
):
    prompt_len, response_len, vocab, cache_k = 2, 1, 8, 4
    total_len = prompt_len + response_len
    torch.manual_seed(20260905)
    rollout_reference_logits = torch.randn(
        1, total_len, vocab, requires_grad=True
    )
    current_reference_logits = (
        rollout_reference_logits.detach().clone().requires_grad_(True)
    )
    response_token = torch.tensor([1])
    current_lp = rollout_reference_logits.detach()[
        0, prompt_len - 1 : total_len - 1
    ].log_softmax(dim=-1).gather(1, response_token[:, None]).squeeze(1)
    advantage = torch.tensor(0.5)
    candidate_ids = torch.tensor([[0, 1, 4, 5]], dtype=torch.int32)
    candidate_lp = torch.full((1, cache_k), -5.0, dtype=torch.float16)
    candidate_lp[0, 1] = float(current_lp[0] + advantage)
    # current/rollout ratio=2, above the 1.2 PPO upper clip.
    rollout_lp = current_lp - torch.log(torch.tensor(2.0))
    batch = {
        "unconcat_tokens": [
            torch.cat((torch.tensor([6, 7]), response_token))
        ],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "rollout_log_probs": [rollout_lp],
        "teacher_candidate_positions": [torch.tensor([0], dtype=torch.int32)],
        "teacher_candidate_ids": [candidate_ids],
        "teacher_candidate_logprobs": [candidate_lp],
    }
    common_args = dict(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=1,
        opd_kl_coef=1.0,
        opd_student_anchor_kl_coef=1.0,
        opd_sampled_token_cache_miss_policy="anchor",
        opd_token_id_bridge=None,
        eps_clip=0.2,
        eps_clip_high=0.2,
        log_probs_chunk_size=-1,
        offline_opd_topk_overlap=False,
    )

    rollout_loss, rollout_log = cached_sampled_token_pg_loss(
        Namespace(
            **common_args,
            opd_sampled_token_clip_reference="rollout",
        ),
        batch,
        rollout_reference_logits,
        lambda value: value.mean(),
    )
    rollout_loss.backward()
    torch.testing.assert_close(
        rollout_log["offline_opd/sampled_token_ppo_clipfrac"],
        torch.tensor(1.0),
    )
    torch.testing.assert_close(
        rollout_log[
            "__max__offline_opd/sampled_token_clip_reference_current"
        ],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        rollout_reference_logits.grad,
        torch.zeros_like(rollout_reference_logits.grad),
    )

    current_loss, current_log = cached_sampled_token_pg_loss(
        Namespace(
            **common_args,
            opd_sampled_token_clip_reference="current",
        ),
        batch,
        current_reference_logits,
        lambda value: value.mean(),
    )
    current_loss.backward()
    expected_advantage = candidate_lp[0, 1].float() - current_lp[0]
    torch.testing.assert_close(current_loss.detach(), -expected_advantage)
    torch.testing.assert_close(
        current_log["offline_opd/sampled_token_ppo_clipfrac"],
        torch.tensor(0.0),
    )
    torch.testing.assert_close(
        current_log[
            "__max__offline_opd/sampled_token_clip_reference_current"
        ],
        torch.tensor(1.0),
    )
    assert current_reference_logits.grad[
        0, prompt_len - 1, response_token[0]
    ] < 0


def test_cached_sampled_token_pg_reports_detached_top8_teacher_metrics(
    cached_sampled_token_pg_loss,
):
    prompt_len, response_len, vocab = 2, 3, 10
    total_len, cache_k = prompt_len + response_len, vocab
    response_student_logits = torch.tensor(
        [
            [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0],
            [2.0, 1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        ]
    )
    logits_without_metrics = torch.zeros(
        1, total_len, vocab, requires_grad=True
    )
    with torch.no_grad():
        logits_without_metrics[0, prompt_len - 1 : total_len - 1] = (
            response_student_logits
        )
    logits_with_metrics = logits_without_metrics.detach().clone().requires_grad_(True)
    response_tokens = torch.tensor([0, 1, 2])
    student_lp = response_student_logits.log_softmax(dim=-1)

    teacher_ranked_ids = torch.tensor(
        [
            [0, 2, 1, 3, 4, 5, 6, 7, 8, 9],
            [2, 1, 4, 3, 6, 5, 8, 7, 0, 9],
        ]
    )
    teacher_ranked_logits = torch.tensor(
        [
            [10.5, 9.5, 8.5, 7.5, 6.5, 5.5, 4.5, 3.5, 2.5, 1.5],
            [10.2, 9.2, 8.2, 7.2, 6.2, 5.2, 4.2, 3.2, 2.2, 1.2],
        ]
    )
    teacher_ranked_lp = teacher_ranked_logits.log_softmax(dim=-1)
    id_order = teacher_ranked_ids.argsort(dim=-1)
    candidate_ids = teacher_ranked_ids.gather(1, id_order)
    candidate_lp = teacher_ranked_lp.gather(1, id_order)
    teacher_lp_by_id = torch.empty_like(candidate_lp).scatter(
        1, teacher_ranked_ids, teacher_ranked_lp
    )
    rollout_lp = student_lp.gather(1, response_tokens[:, None]).squeeze(1)
    batch = {
        "unconcat_tokens": [torch.cat((torch.tensor([8, 9]), response_tokens))],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "rollout_log_probs": [rollout_lp],
        "teacher_candidate_positions": [torch.tensor([0, 1], dtype=torch.int32)],
        "teacher_candidate_ids": [candidate_ids.to(torch.int32)],
        "teacher_candidate_logprobs": [candidate_lp.to(torch.float16)],
        "teacher_top128_ids": [teacher_ranked_ids.to(torch.int32)],
        "teacher_cutoff_logprobs": [teacher_ranked_lp[:, -1].to(torch.float16)],
        "teacher_top4096_mass": [teacher_ranked_lp.exp().sum(dim=-1)],
    }
    common_args = dict(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_kl_coef=1.0,
        opd_student_anchor_kl_coef=1.0,
        opd_sampled_token_cache_miss_policy="anchor",
        opd_token_id_bridge=None,
        eps_clip=0.2,
        eps_clip_high=0.2,
        log_probs_chunk_size=-1,
        vocab_size=vocab,
    )
    loss_without_metrics, _ = cached_sampled_token_pg_loss(
        Namespace(**common_args, offline_opd_topk_overlap=False),
        batch,
        logits_without_metrics,
        lambda value: value.mean(),
    )
    loss_without_metrics.backward()
    baseline_grad = logits_without_metrics.grad.detach().clone()

    loss, log = cached_sampled_token_pg_loss(
        Namespace(**common_args, offline_opd_topk_overlap=True),
        batch,
        logits_with_metrics,
        lambda value: value.mean(),
    )
    loss.backward()
    torch.testing.assert_close(loss, loss_without_metrics)
    torch.testing.assert_close(logits_with_metrics.grad, baseline_grad)

    expected_overlaps = {1: 0.5, 2: 0.75, 4: 1.0, 8: 1.0}
    current_top8 = response_student_logits[:2].topk(8, dim=-1).indices
    for width, expected_overlap in expected_overlaps.items():
        torch.testing.assert_close(
            log[f"opd_topk_overlap_at{width}"],
            torch.tensor(expected_overlap),
        )
        expected_student_mass = student_lp[:2].gather(
            1, current_top8[:, :width]
        ).exp().sum(dim=-1).mean()
        expected_teacher_mass = teacher_lp_by_id.gather(
            1, current_top8[:, :width]
        ).exp().sum(dim=-1).mean()
        torch.testing.assert_close(
            log[f"student_mass_at{width}"], expected_student_mass
        )
        torch.testing.assert_close(
            log[f"teacher_mass_at{width}"],
            expected_teacher_mass,
            atol=2e-4,
            rtol=2e-4,
        )
    torch.testing.assert_close(log["opd_topk_overlap"], log["opd_topk_overlap_at8"])
    torch.testing.assert_close(log["student_mass"], log["student_mass_at8"])
    torch.testing.assert_close(log["teacher_mass"], log["teacher_mass_at8"])


def test_cached_sampled_token_pg_sdar_uses_current_gap_sigmoid_gate(
    cached_sampled_token_pg_loss,
):
    prompt_len, response_len, vocab, cache_k = 2, 3, 8, 4
    total_len = prompt_len + response_len
    torch.manual_seed(260515155)
    baseline_logits = torch.randn(
        1, total_len, vocab, requires_grad=True
    )
    sdar_logits = baseline_logits.detach().clone().requires_grad_(True)
    response_tokens = torch.tensor([1, 2, 3])
    response_logits = baseline_logits.detach()[
        0, prompt_len - 1 : total_len - 1
    ]
    current_lp = response_logits.log_softmax(dim=-1).gather(
        1, response_tokens[:, None]
    ).squeeze(1)
    candidate_ids = torch.tensor(
        [[0, 1, 4, 5], [0, 2, 4, 5]], dtype=torch.int32
    )
    candidate_lp = torch.full((2, cache_k), -5.0, dtype=torch.float16)
    candidate_lp[0, 1] = float(current_lp[0] + 0.4)
    candidate_lp[1, 1] = float(current_lp[1] - 0.4)
    batch = {
        "unconcat_tokens": [torch.cat((torch.tensor([6, 7]), response_tokens))],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "rollout_log_probs": [current_lp.clone()],
        "teacher_candidate_positions": [torch.tensor([0, 1], dtype=torch.int32)],
        "teacher_candidate_ids": [candidate_ids],
        "teacher_candidate_logprobs": [candidate_lp],
    }
    common_args = dict(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_kl_coef=1.0,
        opd_student_anchor_kl_coef=1.0,
        opd_sampled_token_cache_miss_policy="anchor",
        opd_token_id_bridge=None,
        eps_clip=0.2,
        eps_clip_high=0.2,
        log_probs_chunk_size=-1,
        offline_opd_topk_overlap=False,
    )
    baseline_loss, _ = cached_sampled_token_pg_loss(
        Namespace(**common_args, opd_sampled_token_sdar_gate=False),
        batch,
        baseline_logits,
        lambda value: value.mean(),
    )
    baseline_loss.backward()
    baseline_grad = baseline_logits.grad.detach().clone()

    beta = 5.0
    loss, log = cached_sampled_token_pg_loss(
        Namespace(
            **common_args,
            opd_sampled_token_sdar_gate=True,
            opd_sdar_gate_beta=beta,
        ),
        batch,
        sdar_logits,
        lambda value: value.mean(),
    )
    positive_gap = candidate_lp[0, 1].float() - current_lp[0]
    negative_gap = candidate_lp[1, 1].float() - current_lp[1]
    positive_gate = torch.sigmoid(beta * positive_gap)
    negative_gate = torch.sigmoid(beta * negative_gap)
    expected_loss = -(
        positive_gate * positive_gap + negative_gate * negative_gap
    ) / response_len
    torch.testing.assert_close(loss.detach(), expected_loss)
    torch.testing.assert_close(
        log["offline_opd/sdar_positive_row_fraction"], torch.tensor(1 / 3)
    )
    torch.testing.assert_close(
        log["offline_opd/sdar_negative_row_fraction"], torch.tensor(1 / 3)
    )
    torch.testing.assert_close(
        log["offline_opd/sdar_gate_mass_per_train_token"],
        (positive_gate + negative_gate) / response_len,
    )
    torch.testing.assert_close(
        log["__max__offline_opd/sdar_gate_beta"], torch.tensor(beta)
    )

    loss.backward()
    torch.testing.assert_close(
        sdar_logits.grad[0, prompt_len - 1],
        baseline_grad[0, prompt_len - 1] * positive_gate,
    )
    torch.testing.assert_close(
        sdar_logits.grad[0, prompt_len],
        baseline_grad[0, prompt_len] * negative_gate,
    )
    torch.testing.assert_close(
        sdar_logits.grad[0, prompt_len + 1 :],
        baseline_grad[0, prompt_len + 1 :],
    )

    normalized_logits = baseline_logits.detach().clone().requires_grad_(True)
    gate_scale = 2.0
    normalized_loss, normalized_log = cached_sampled_token_pg_loss(
        Namespace(
            **common_args,
            opd_sampled_token_sdar_gate=True,
            opd_sdar_gate_beta=beta,
            opd_sdar_gate_scale=gate_scale,
        ),
        batch,
        normalized_logits,
        lambda value: value.mean(),
    )
    torch.testing.assert_close(
        normalized_loss.detach(), gate_scale * expected_loss
    )
    torch.testing.assert_close(
        normalized_log["offline_opd/sdar_gate_mass_per_train_token"],
        gate_scale * (positive_gate + negative_gate) / response_len,
    )
    torch.testing.assert_close(
        normalized_log["__max__offline_opd/sdar_gate_scale"],
        torch.tensor(gate_scale),
    )
    normalized_loss.backward()
    torch.testing.assert_close(
        normalized_logits.grad[0, prompt_len - 1],
        baseline_grad[0, prompt_len - 1] * gate_scale * positive_gate,
    )
    torch.testing.assert_close(
        normalized_logits.grad[0, prompt_len],
        baseline_grad[0, prompt_len] * gate_scale * negative_gate,
    )


def test_cross_tokenizer_reverse_only_keeps_unaligned_student_anchor(
    cross_tokenizer_reverse_anchor_loss,
):
    torch.manual_seed(321)
    prompt_len, response_len, vocab = 2, 2, 6
    total_len, support_k, cache_k = prompt_len + response_len, 1, 2
    logits = torch.randn(1, total_len, vocab, requires_grad=True)
    teacher_ids = torch.tensor([[0, 3]], dtype=torch.int32)
    teacher_lp = torch.tensor([[-0.2, -2.0]], dtype=torch.float16)
    anchor_ids = torch.tensor([[2]], dtype=torch.int32)
    anchor_lp = torch.tensor([[-0.4]], dtype=torch.float16)
    batch = {
        "unconcat_tokens": [torch.arange(total_len)],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "teacher_candidate_positions": [torch.tensor([0], dtype=torch.int32)],
        "teacher_candidate_ids": [teacher_ids],
        "teacher_candidate_logprobs": [teacher_lp],
        "teacher_top128_ids": [torch.tensor([[0]], dtype=torch.int32)],
        "teacher_cutoff_logprobs": [torch.tensor([-2.0], dtype=torch.float16)],
        "teacher_top4096_mass": [teacher_lp.float().exp().sum(dim=-1)],
        "student_anchor_positions": [torch.tensor([1], dtype=torch.int32)],
        "student_anchor_topk_ids": [anchor_ids],
        "student_anchor_topk_logprobs": [anchor_lp],
        "student_anchor_reason": [torch.zeros(1, dtype=torch.int8)],
    }
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_dynamic_support_topk=support_k,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_teacher_cache_miss_policy="cutoff",
        opd_student_anchor_kl_coef=1.0,
        offline_opd_topk_overlap=False,
        log_probs_chunk_size=-1,
        vocab_size=vocab,
    )

    loss, log = cross_tokenizer_reverse_anchor_loss(
        args, batch, logits, lambda value: value.mean()
    )

    assert "opd_forward_kl" not in log
    torch.testing.assert_close(
        loss.detach(),
        (log["opd_reverse_kl"] + log["offline_opd/student_anchor_kl"]) / 2,
        atol=2e-4,
        rtol=2e-4,
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, prompt_len - 1 : total_len - 1].abs().sum() > 0


@pytest.fixture
def let_me_conditional_loss(monkeypatch):
    stubbed = ("slime.backends.megatron_utils.loss", "slime.backends.megatron_utils.cp_utils")
    previous = {name: sys.modules.pop(name, None) for name in stubbed}
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_tensor_model_parallel_rank=lambda: 0,
    )
    monkeypatch.setitem(sys.modules, "megatron", types.ModuleType("megatron"))
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)
    try:
        from slime.backends.megatron_utils.loss import (
            _opd_cross_tokenizer_let_me_forward_else_reverse_topk_loss,
        )

        yield _opd_cross_tokenizer_let_me_forward_else_reverse_topk_loss
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_let_me_triplet_uses_forward_and_other_exact_rows_use_reverse(
    let_me_conditional_loss,
):
    torch.manual_seed(20260827)
    prompt_len, response_len, vocab = 2, 4, 8
    total_len, support_k, cache_k = prompt_len + response_len, 2, 4
    logits = torch.randn(1, total_len, vocab, requires_grad=True)
    response_logits = logits[0, prompt_len - 1 : total_len - 1]
    teacher_full_lp = torch.randn(response_len, vocab).log_softmax(dim=-1)
    teacher_ranked_ids = teacher_full_lp.topk(cache_k, dim=-1).indices
    teacher_ranked_lp = teacher_full_lp.gather(1, teacher_ranked_ids)
    sort_order = teacher_ranked_ids.argsort(dim=-1)
    candidate_ids = teacher_ranked_ids.gather(1, sort_order)
    candidate_lp = teacher_ranked_lp.gather(1, sort_order)
    response_tokens = torch.tensor([1, 2, 3, 4])
    batch = {
        "unconcat_tokens": [torch.cat((torch.tensor([6, 7]), response_tokens))],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "teacher_candidate_positions": [torch.arange(response_len, dtype=torch.int32)],
        "teacher_candidate_ids": [candidate_ids.to(torch.int32)],
        "teacher_candidate_logprobs": [candidate_lp.to(torch.float16)],
        "teacher_top128_ids": [teacher_ranked_ids[:, :support_k].to(torch.int32)],
        "teacher_cutoff_logprobs": [teacher_ranked_lp[:, -1].to(torch.float16)],
        "teacher_top4096_mass": [teacher_ranked_lp.exp().sum(dim=-1)],
        "student_anchor_positions": [torch.empty(0, dtype=torch.int32)],
        "student_anchor_topk_ids": [torch.empty((0, support_k), dtype=torch.int32)],
        "student_anchor_topk_logprobs": [torch.empty((0, support_k), dtype=torch.float16)],
        "student_anchor_reason": [torch.empty(0, dtype=torch.int8)],
    }
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_dynamic_support_topk=support_k,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_teacher_cache_miss_policy="cutoff",
        opd_student_anchor_kl_coef=1.0,
        opd_let_token_ids="1",
        opd_me_token_ids="2",
        offline_opd_topk_overlap=True,
        log_probs_chunk_size=-1,
        vocab_size=vocab,
    )
    loss, log = let_me_conditional_loss(
        args, batch, logits, lambda value: value.mean()
    )

    current_full_lp = response_logits.log_softmax(dim=-1)
    forward_ids = teacher_ranked_ids[:, :support_k]
    forward_teacher_lp = teacher_full_lp.gather(1, forward_ids)
    forward_student_lp = current_full_lp.gather(1, forward_ids)
    forward_log_q = forward_teacher_lp - forward_teacher_lp.logsumexp(
        dim=-1, keepdim=True
    )
    forward_rows = (
        forward_log_q.exp() * (forward_teacher_lp - forward_student_lp)
    ).sum(dim=-1)

    current_ids = response_logits.topk(support_k, dim=-1).indices
    current_lp = current_full_lp.gather(1, current_ids)
    reverse_teacher_lp = torch.empty_like(current_lp)
    for row in range(response_len):
        cache = {
            int(token_id): float(token_lp)
            for token_id, token_lp in zip(
                candidate_ids[row], candidate_lp[row], strict=True
            )
        }
        cutoff = float(teacher_ranked_lp[row, -1].to(torch.float16))
        reverse_teacher_lp[row] = torch.tensor(
            [cache.get(int(token_id), cutoff) for token_id in current_ids[row]]
        )
    reverse_log_q = current_lp - current_lp.logsumexp(dim=-1, keepdim=True)
    reverse_rows = (
        reverse_log_q.exp() * (current_lp - reverse_teacher_lp)
    ).sum(dim=-1)
    expected = (forward_rows[:3].sum() + reverse_rows[3]) / response_len

    torch.testing.assert_close(loss, expected, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(
        log["offline_opd/let_me_forward_row_fraction"], torch.tensor(0.75)
    )
    # Direction-specific logs are contributions in the common Teacher-row mean:
    # phrase rows are zero in reverse and all other rows are zero in forward.
    torch.testing.assert_close(log["opd_forward_kl"], forward_rows[:3].sum() / 4, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(log["opd_reverse_kl"], reverse_rows[3] / 4, atol=3e-3, rtol=3e-3)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, prompt_len - 1 : total_len - 1].abs().sum() > 0


def test_cross_tokenizer_anchor_uses_sparse_head_and_tail_with_gradient(
    cross_tokenizer_anchor_loss,
):
    torch.manual_seed(123)
    prompt_len, response_len, vocab = 2, 3, 8
    total_len, support_k, cache_k = prompt_len + response_len, 2, 4
    logits = torch.randn(1, total_len, vocab, requires_grad=True)
    response_logits = logits[0, prompt_len - 1 : total_len - 1]
    teacher_positions = torch.tensor([0, 2], dtype=torch.int32)
    teacher_full_lp = torch.randn(2, vocab).log_softmax(dim=-1)
    teacher_ranked_ids = teacher_full_lp.topk(cache_k, dim=-1).indices
    teacher_ranked_lp = teacher_full_lp.gather(1, teacher_ranked_ids)
    order = teacher_ranked_ids.argsort(dim=-1)
    candidate_ids = teacher_ranked_ids.gather(1, order)
    candidate_lp = teacher_ranked_lp.gather(1, order)

    anchor_ids = torch.tensor([[1, 5]], dtype=torch.int32)
    old_full_lp = torch.randn(vocab).log_softmax(dim=-1)
    anchor_old_lp = old_full_lp[anchor_ids.long()].to(torch.float16)
    rollout_ids = response_logits.detach()[[0, 2]].topk(support_k, dim=-1).indices
    rollout_lp = response_logits.detach()[[0, 2]].log_softmax(dim=-1).gather(1, rollout_ids)
    batch = {
        "unconcat_tokens": [torch.arange(total_len)],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "teacher_candidate_positions": [teacher_positions],
        "teacher_candidate_ids": [candidate_ids.to(torch.int32)],
        "teacher_candidate_logprobs": [candidate_lp.to(torch.float16)],
        "teacher_top128_ids": [teacher_ranked_ids[:, :support_k].to(torch.int32)],
        "teacher_cutoff_logprobs": [teacher_ranked_lp[:, -1].to(torch.float16)],
        "teacher_top4096_mass": [teacher_ranked_lp.exp().sum(dim=-1)],
        "rollout_student_topk_ids_sparse": [rollout_ids.to(torch.int32)],
        "rollout_student_topk_logprobs_sparse": [rollout_lp.to(torch.float16)],
        "student_anchor_positions": [torch.tensor([1], dtype=torch.int32)],
        "student_anchor_topk_ids": [anchor_ids],
        "student_anchor_topk_logprobs": [anchor_old_lp],
        "student_anchor_reason": [torch.zeros(1, dtype=torch.int8)],
    }
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_dynamic_support_topk=support_k,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_teacher_cache_miss_policy="cutoff",
        opd_forward_kl_coef=0.5,
        opd_reverse_kl_coef=0.5,
        opd_student_anchor_kl_coef=2.0,
        offline_opd_topk_overlap=True,
        log_probs_chunk_size=-1,
        vocab_size=vocab,
    )
    loss, log = cross_tokenizer_anchor_loss(
        args, batch, logits, lambda value: value.mean()
    )

    current_anchor_lp = response_logits[1].log_softmax(dim=-1)[anchor_ids.long()[0]]
    old_prob = anchor_old_lp.float().exp()[0]
    old_tail = 1.0 - old_prob.sum()
    current_tail = 1.0 - current_anchor_lp.exp().sum()
    expected_anchor_row = (
        old_prob * (anchor_old_lp.float()[0] - current_anchor_lp)
    ).sum() + old_tail * (old_tail.log() - current_tail.log())
    # The shared reducer divides by all three original trainable positions.
    torch.testing.assert_close(
        log["offline_opd/student_anchor_kl"], expected_anchor_row,
        atol=2e-4,
        rtol=2e-4,
    )
    torch.testing.assert_close(
        log["offline_opd/student_anchor_row_fraction"], torch.tensor(1 / 3)
    )
    expected_total = (
        2
        * (
            0.5 * log["opd_forward_kl"]
            + 0.5 * log["opd_reverse_kl"]
        )
        + 2.0 * expected_anchor_row
    ) / 4.0
    torch.testing.assert_close(loss.detach(), expected_total, atol=2e-4, rtol=2e-4)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, prompt_len].abs().sum() > 0


def test_dynamic_loss_uses_current_topk_and_cutoff_for_cache_misses(dynamic_loss):
    torch.manual_seed(41)
    prompt_len, response_len, vocab = 2, 3, 10
    total_len = prompt_len + response_len
    support_k, cache_k = 3, 5
    logits = torch.randn(1, total_len, vocab, requires_grad=True)
    response_logits = logits[0, prompt_len - 1 : total_len - 1]
    current_ids = response_logits.topk(support_k, dim=-1).indices

    # Cache four of the five lowest ids plus one current id per row. This
    # deliberately produces both hits and misses on current Top-3.
    ranked_ids = torch.stack(
        [
            torch.cat(
                (
                    current_ids[row, :1],
                    torch.arange(vocab)[torch.arange(vocab) != current_ids[row, 0]][: cache_k - 1],
                )
            )
            for row in range(response_len)
        ]
    )
    teacher_full_lp = torch.randn(response_len, vocab).log_softmax(dim=-1)
    ranked_lp = teacher_full_lp.gather(1, ranked_ids)
    order = ranked_ids.argsort(dim=-1)
    sorted_ids = ranked_ids.gather(1, order)
    sorted_lp = ranked_lp.gather(1, order)
    cutoff = ranked_lp.min(dim=-1).values

    batch = {
        "unconcat_tokens": [torch.arange(total_len)],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "teacher_candidate_positions": [torch.arange(response_len, dtype=torch.int32)],
        "teacher_candidate_ids": [sorted_ids.to(torch.int32)],
        "teacher_candidate_logprobs": [sorted_lp.to(torch.float16)],
        "teacher_top128_ids": [ranked_ids[:, :support_k].to(torch.int32)],
        "teacher_cutoff_logprobs": [cutoff.to(torch.float16)],
        "teacher_top4096_mass": [ranked_lp.exp().sum(dim=-1)],
    }
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_dynamic_support_topk=support_k,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_teacher_cache_miss_policy="cutoff",
        log_probs_chunk_size=-1,
        vocab_size=vocab,
    )
    loss, log = dynamic_loss(args, batch, logits, lambda value: value.mean())

    current_lp = response_logits.log_softmax(dim=-1).gather(1, current_ids)
    columns = torch.searchsorted(sorted_ids, current_ids)
    safe = columns.clamp_max(cache_k - 1)
    hit = (columns < cache_k) & sorted_ids.gather(1, safe).eq(current_ids)
    cached_lp = sorted_lp.to(torch.float16).float().gather(1, safe)
    expected_teacher_lp = torch.where(hit, cached_lp, cutoff.to(torch.float16).float()[:, None])
    q = current_lp.softmax(dim=-1)
    expected = (q * (current_lp - expected_teacher_lp)).sum(dim=-1).mean()

    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        log["offline_opd_dynamic/teacher_cache_hit_ratio"], hit.float().mean()
    )
    torch.testing.assert_close(log["student_mass"], current_lp.exp().sum(dim=-1).mean())
    for width in (1, 2):
        torch.testing.assert_close(
            log[f"student_mass_at{width}"],
            current_lp[:, :width].exp().sum(dim=-1).mean(),
        )
        torch.testing.assert_close(
            log[f"teacher_mass_at{width}"],
            expected_teacher_lp[:, :width]
            .exp()
            .sum(dim=-1)
            .clamp_max(1.0)
            .mean(),
        )
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_dynamic_loss_maps_student_ids_before_teacher_cache_lookup(
    dynamic_loss, tmp_path
):
    prompt_len, response_len, vocab = 2, 2, 8
    total_len, support_k, cache_k = prompt_len + response_len, 2, 3
    logits = torch.randn(1, total_len, vocab, requires_grad=True)
    response_logits = logits[0, prompt_len - 1 : total_len - 1]
    current_ids = response_logits.topk(support_k, dim=-1).indices

    # A non-identity, injective student->teacher mapping makes a direct-id
    # lookup miss while the bridged lookup hits every current support token.
    mapping = torch.arange(vocab - 1, -1, -1, dtype=torch.int32)
    mapped_current = mapping[current_ids]
    extra = torch.tensor(
        [next(i for i in range(vocab) if i not in row.tolist()) for row in mapped_current]
    ).reshape(-1, 1)
    ranked_ids = torch.cat((mapped_current, extra), dim=-1)
    ranked_lp = torch.tensor([[-0.2, -1.0, -3.0], [-0.3, -1.2, -3.2]])
    order = ranked_ids.argsort(dim=-1)
    sorted_ids = ranked_ids.gather(1, order)
    sorted_lp = ranked_lp.gather(1, order)

    mapping_np = mapping.numpy()
    digest = hashlib.sha256(
        np.ascontiguousarray(mapping_np, dtype="<i4").tobytes()
    ).hexdigest()
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_text(
        json.dumps(
            {
                "format": "offline_opd_token_id_bridge_v1",
                "student_tokenizer_size": vocab,
                "student_model_vocab_size": vocab,
                "teacher_tokenizer_size": vocab,
                "teacher_model_vocab_size": vocab,
                "mapping_sha256": digest,
                "student_to_teacher": mapping_np.tolist(),
            }
        )
    )
    batch = {
        "unconcat_tokens": [torch.arange(total_len)],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "teacher_candidate_positions": [torch.arange(response_len, dtype=torch.int32)],
        "teacher_candidate_ids": [sorted_ids.to(torch.int32)],
        "teacher_candidate_logprobs": [sorted_lp.to(torch.float16)],
        "teacher_top128_ids": [ranked_ids[:, :support_k].to(torch.int32)],
        "teacher_cutoff_logprobs": [ranked_lp[:, -1].to(torch.float16)],
        "teacher_top4096_mass": [ranked_lp.exp().sum(dim=-1)],
    }
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_dynamic_support_topk=support_k,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_teacher_cache_miss_policy="cutoff",
        opd_token_id_bridge=str(bridge_path),
        opd_student_valid_vocab_size=vocab,
        log_probs_chunk_size=-1,
        vocab_size=vocab,
    )
    loss, log = dynamic_loss(args, batch, logits, lambda value: value.mean())
    current_lp = response_logits.log_softmax(dim=-1).gather(1, current_ids)
    expected = (
        current_lp.softmax(dim=-1)
        * (current_lp - ranked_lp[:, :support_k].to(torch.float16).float())
    ).sum(dim=-1).mean()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(
        log["offline_opd_dynamic/teacher_cache_hit_ratio"], torch.tensor(1.0)
    )
    torch.testing.assert_close(log["opd_topk_overlap"], torch.tensor(1.0))


def test_bidirectional_loss_combines_natural_supports_and_reports_entropy(
    cached_bidirectional_loss,
):
    torch.manual_seed(17)
    prompt_len, response_len, vocab = 2, 3, 9
    total_len, support_k, cache_k = prompt_len + response_len, 3, 6
    logits = torch.randn(1, total_len, vocab, requires_grad=True)
    response_logits = logits[0, prompt_len - 1 : total_len - 1]
    current_ids = response_logits.topk(support_k, dim=-1).indices

    teacher_full_lp = torch.randn(response_len, vocab).log_softmax(dim=-1)
    teacher_ranked_ids = teacher_full_lp.topk(cache_k, dim=-1).indices
    teacher_ranked_lp = teacher_full_lp.gather(1, teacher_ranked_ids)
    order = teacher_ranked_ids.argsort(dim=-1)
    candidate_ids = teacher_ranked_ids.gather(1, order)
    candidate_lp = teacher_ranked_lp.gather(1, order)
    cutoff = teacher_ranked_lp[:, -1]
    rollout_ids = torch.stack(
        [torch.randperm(vocab)[:support_k] for _ in range(response_len)]
    )
    rollout_lp = response_logits.detach().log_softmax(dim=-1).gather(1, rollout_ids)

    batch = {
        "unconcat_tokens": [torch.arange(total_len)],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "teacher_candidate_positions": [torch.arange(response_len, dtype=torch.int32)],
        "teacher_candidate_ids": [candidate_ids.to(torch.int32)],
        "teacher_candidate_logprobs": [candidate_lp.to(torch.float16)],
        "teacher_top128_ids": [teacher_ranked_ids[:, :support_k].to(torch.int32)],
        "teacher_cutoff_logprobs": [cutoff.to(torch.float16)],
        "teacher_top4096_mass": [teacher_ranked_lp.exp().sum(dim=-1)],
        "rollout_student_topk_ids_sparse": [rollout_ids.to(torch.int32)],
        "rollout_student_topk_logprobs_sparse": [rollout_lp.to(torch.float16)],
    }
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_dynamic_support_topk=support_k,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_teacher_cache_miss_policy="cutoff",
        opd_forward_kl_coef=0.5,
        opd_reverse_kl_coef=0.5,
        offline_opd_topk_overlap=True,
        log_probs_chunk_size=2,
        vocab_size=vocab,
    )
    loss, log = cached_bidirectional_loss(
        args, batch, logits, lambda value: value.mean()
    )

    student_full_lp = response_logits.log_softmax(dim=-1)
    current_lp = student_full_lp.gather(1, current_ids)
    columns = torch.searchsorted(candidate_ids, current_ids)
    safe = columns.clamp_max(cache_k - 1)
    hit = (columns < cache_k) & candidate_ids.gather(1, safe).eq(current_ids)
    reverse_teacher_lp = torch.where(
        hit,
        candidate_lp.to(torch.float16).float().gather(1, safe),
        cutoff.to(torch.float16).float()[:, None],
    )
    expected_reverse = (
        current_lp.softmax(dim=-1) * (current_lp - reverse_teacher_lp)
    ).sum(dim=-1).mean()

    forward_ids = teacher_ranked_ids[:, :support_k]
    forward_teacher_lp = teacher_ranked_lp[:, :support_k].to(torch.float16).float()
    forward_student_lp = student_full_lp.gather(1, forward_ids)
    expected_forward = (
        forward_teacher_lp.softmax(dim=-1)
        * (forward_teacher_lp - forward_student_lp)
    ).sum(dim=-1).mean()
    expected_entropy = -(student_full_lp.exp() * student_full_lp).sum(dim=-1).mean()
    old_log_q = rollout_lp.to(torch.float16).float().log_softmax(dim=-1)
    current_rollout_log_q = student_full_lp.gather(1, rollout_ids).log_softmax(dim=-1)
    expected_old_topk_entropy = -(old_log_q.exp() * old_log_q).sum(dim=-1).mean()
    expected_current_topk_entropy = -(
        current_rollout_log_q.exp() * current_rollout_log_q
    ).sum(dim=-1).mean()

    torch.testing.assert_close(log["opd_forward_kl"], expected_forward)
    torch.testing.assert_close(log["opd_reverse_kl"], expected_reverse)
    torch.testing.assert_close(loss, 0.5 * expected_forward + 0.5 * expected_reverse)
    torch.testing.assert_close(log["entropy"], expected_entropy)
    torch.testing.assert_close(log["rollout_topk_entropy"], expected_old_topk_entropy)
    torch.testing.assert_close(
        log["current_on_rollout_topk_entropy"], expected_current_topk_entropy
    )
    torch.testing.assert_close(
        log["entropy_change_vs_rollout_topk"],
        expected_current_topk_entropy - expected_old_topk_entropy,
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_cached_teacher_forward_skips_teacher_only_ids_and_fills_support(
    cached_forward_loss, tmp_path
):
    prompt_len, response_len = 2, 2
    student_vocab, teacher_vocab = 6, 9
    support_k, cache_k = 3, 5
    total_len = prompt_len + response_len
    logits = torch.randn(1, total_len, student_vocab, requires_grad=True)
    response_logits = logits[0, prompt_len - 1 : total_len - 1]

    # Teacher ids 2, 5 and 7 have no Student equivalent. The first row's
    # highest-ranked token is Teacher-only, so rank four must be pulled in to
    # retain a three-token support.
    mapping = np.array([0, 1, 3, 4, 6, 8], dtype=np.int32)
    digest = hashlib.sha256(
        np.ascontiguousarray(mapping, dtype="<i4").tobytes()
    ).hexdigest()
    bridge_path = tmp_path / "bridge.json"
    bridge_path.write_text(
        json.dumps(
            {
                "format": "offline_opd_token_id_bridge_v1",
                "student_tokenizer_size": student_vocab,
                "student_model_vocab_size": student_vocab,
                "teacher_tokenizer_size": teacher_vocab,
                "teacher_model_vocab_size": teacher_vocab,
                "mapping_sha256": digest,
                "student_to_teacher": mapping.tolist(),
            }
        )
    )
    ranked_ids = torch.tensor([[2, 0, 3, 1, 5], [8, 7, 4, 6, 0]])
    ranked_lp = torch.tensor([[-0.1, -0.3, -0.7, -1.1, -2.0], [-0.2, -0.4, -0.8, -1.2, -2.2]])
    order = ranked_ids.argsort(dim=-1)
    sorted_ids = ranked_ids.gather(1, order)
    sorted_lp = ranked_lp.gather(1, order)
    batch = {
        "unconcat_tokens": [torch.arange(total_len)],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "teacher_candidate_positions": [torch.arange(response_len, dtype=torch.int32)],
        "teacher_candidate_ids": [sorted_ids.to(torch.int32)],
        "teacher_candidate_logprobs": [sorted_lp.to(torch.float16)],
        "teacher_top128_ids": [ranked_ids[:, :support_k].to(torch.int32)],
        "teacher_top4096_mass": [ranked_lp.exp().sum(dim=-1)],
    }
    args = Namespace(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_dynamic_support_topk=support_k,
        opd_teacher_cache_topk=cache_k,
        opd_teacher_cache_h2d_rows=2,
        opd_token_id_bridge=str(bridge_path),
        opd_student_valid_vocab_size=student_vocab,
        offline_opd_topk_overlap=False,
        log_probs_chunk_size=-1,
        vocab_size=student_vocab,
    )
    loss, log = cached_forward_loss(args, batch, logits, lambda value: value.mean())

    selected_student_ids = torch.tensor([[0, 2, 1], [5, 3, 4]])
    selected_teacher_lp = torch.tensor([[-0.3, -0.7, -1.1], [-0.2, -0.8, -1.2]])
    selected_student_lp = response_logits.log_softmax(dim=-1).gather(
        1, selected_student_ids
    )
    expected = (
        selected_teacher_lp.softmax(dim=-1)
        * (selected_teacher_lp - selected_student_lp)
    ).sum(dim=-1).mean()
    torch.testing.assert_close(loss, expected, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(
        log["offline_opd_forward/teacher_vocab_skip_count"], torch.tensor(1.0)
    )
    torch.testing.assert_close(
        log["offline_opd_forward/teacher_vocab_skip_ratio"], torch.tensor(1 / 3)
    )
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def _nm_rows_batch(kinds):
    """One sample: response rows 0,2 are 1:1 teacher rows, rows 1,3,4 are anchors."""
    torch.manual_seed(20261105)
    prompt_len, response_len, vocab, support_k, cache_k = 2, 5, 9, 2, 4
    total_len = prompt_len + response_len
    candidate_ids = torch.tensor([[0, 3, 5, 7], [1, 2, 4, 8]], dtype=torch.int32)
    candidate_lp = torch.log_softmax(torch.randn(2, cache_k), dim=-1).to(torch.float16)
    batch = {
        "unconcat_tokens": [torch.randint(0, vocab, (total_len,))],
        "total_lengths": [total_len],
        "response_lengths": [response_len],
        "loss_masks": [torch.ones(response_len)],
        "teacher_candidate_positions": [torch.tensor([0, 2], dtype=torch.int32)],
        "teacher_candidate_ids": [candidate_ids],
        "teacher_candidate_logprobs": [candidate_lp],
        "teacher_top128_ids": [candidate_ids[:, :support_k]],
        "teacher_cutoff_logprobs": [candidate_lp.min(dim=-1).values],
        "teacher_top4096_mass": [candidate_lp.float().exp().sum(dim=-1)],
        "student_anchor_positions": [torch.tensor([1, 3, 4], dtype=torch.int32)],
        "student_anchor_topk_ids": [torch.tensor([[1, 6], [2, 4], [0, 8]], dtype=torch.int32)],
        "student_anchor_topk_logprobs": [torch.tensor([[-0.3, -1.6], [-0.5, -1.2], [-0.2, -2.0]], dtype=torch.float16)],
        "student_anchor_reason": [torch.zeros(3, dtype=torch.int8)],
        "rollout_log_probs": [-torch.rand(response_len) * 2],
        "teacher_chunk_logprob": [-torch.rand(response_len) * 2],
        "teacher_chunk_kind": [torch.tensor(kinds, dtype=torch.float32)],
        "teacher_chunk_mask": [torch.tensor([k > 0 for k in kinds], dtype=torch.float32)],
    }
    logits = torch.randn(1, total_len, vocab)
    return batch, logits


def _nm_args(**overrides):
    values = dict(
        rollout_temperature=1.0,
        allgather_cp=False,
        opd_dynamic_support_topk=2,
        opd_teacher_cache_topk=4,
        opd_teacher_cache_h2d_rows=2,
        opd_teacher_cache_miss_policy="cutoff",
        opd_student_anchor_kl_coef=0.5,
        offline_opd_topk_overlap=False,
        log_probs_chunk_size=-1,
        vocab_size=9,
        opd_chunk_tier="chunk",
    )
    values.update(overrides)
    return Namespace(**values)


def _run_loss(loss_fn, args, batch, logits):
    logits = logits.clone().requires_grad_(True)
    loss, log = loss_fn(args, batch, logits, lambda value: value.sum() / 5)
    loss.backward()
    return loss.detach(), logits.grad, log


def test_nm_rows_without_nm_chunks_add_the_two_group_means(cross_tokenizer_reverse_anchor_loss):
    # Anchors sit in masked (0) or 1:1 (2) chunks only: nothing for the flag to take over,
    # so the loss is the 1:1 KL mean plus lambda times the anchor mean.
    batch, logits = _nm_rows_batch([1, 0, 1, 2, 0])
    args = _nm_args(opd_nm_rows_sampled_token=True)
    loss, _, log = _run_loss(cross_tokenizer_reverse_anchor_loss, args, batch, logits)
    assert log["offline_opd/nm_sampled_token_row_fraction"].item() == 0
    torch.testing.assert_close(
        loss,
        log["opd_reverse_kl"] + args.opd_student_anchor_kl_coef * log["offline_opd/student_anchor_kl"],
        atol=1e-6,
        rtol=1e-5,
    )


def test_nm_rows_train_the_chunk_sampled_token_term(cross_tokenizer_reverse_anchor_loss):
    from slime.backends.megatron_utils.loss import _opd_chunk_teacher_sampled_token_pg_loss

    # Anchor rows 1 (struct_n) and 4 (text_multi) are n:m; anchor row 3 stays an anchor.
    kinds = [1, 3, 1, 0, 4]
    batch, logits = _nm_rows_batch(kinds)
    args = _nm_args(opd_nm_rows_sampled_token=True)
    loss, grad, log = _run_loss(cross_tokenizer_reverse_anchor_loss, args, batch, logits)

    teacher, anchors, nm, lam = 2, 3, 2, args.opd_student_anchor_kl_coef
    # Weights 1, 1, lambda on the three group means.
    expected = (
        log["opd_reverse_kl"]
        + log["offline_opd/nm_sampled_token_pg_loss"]
        + lam * log["offline_opd/student_anchor_kl"]
    )
    torch.testing.assert_close(loss, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        log["offline_opd/nm_sampled_token_row_fraction"], torch.tensor(nm / 5)
    )

    # The n:m term is the chunk loss's own term on those rows.
    chunk_batch = {key: batch[key] for key in (
        "unconcat_tokens", "total_lengths", "response_lengths", "loss_masks",
        "rollout_log_probs", "teacher_chunk_logprob", "teacher_chunk_kind",
    )}
    chunk_batch["teacher_chunk_mask"] = [torch.tensor([0, 1, 0, 0, 1], dtype=torch.float32)]
    chunk_loss, _ = _opd_chunk_teacher_sampled_token_pg_loss(
        _nm_args(opd_student_anchor_kl_coef=0.0), chunk_batch, logits.clone(),
        lambda value: value.sum() / 5,
    )
    torch.testing.assert_close(
        log["offline_opd/nm_sampled_token_pg_loss"] * nm / (teacher + anchors),
        chunk_loss,
        atol=1e-6,
        rtol=1e-5,
    )
    # Logit row r-1+prompt predicts response row r: rows 1 and 4 get a gradient.
    assert grad[0, 2 - 1 + 1].abs().sum() > 0 and grad[0, 2 - 1 + 4].abs().sum() > 0
    assert torch.isfinite(grad).all()
