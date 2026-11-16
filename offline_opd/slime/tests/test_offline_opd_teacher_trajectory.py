import json
from types import SimpleNamespace

import numpy as np
import torch

from examples.coding_agent_opd.scripts.offline_opd import teacher_trajectory as trajectory
from examples.coding_agent_opd.scripts.offline_opd import score_teacher_offline
from examples.coding_agent_opd.scripts.offline_opd.offline_utils import (
    compact_dynamic_teacher_cache_for_disk,
    validate_cross_tokenizer_anchor_scored_sample,
    validate_dynamic_scored_sample,
)
from slime.rollout.on_policy_distillation import build_student_per_position_ids
from slime.utils.types import Sample


def test_render_template_passes_reasoning_effort_only_when_requested():
    class TemplateTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return "rendered"

        def __call__(self, text, *, add_special_tokens=False):
            assert text == "rendered"
            assert not add_special_tokens
            return {"input_ids": [1]}

    tokenizer = TemplateTokenizer()
    trajectory._render_template(
        tokenizer,
        [{"role": "user", "content": "x"}],
        tools=None,
        add_generation_prompt=True,
    )
    assert "reasoning_effort" not in tokenizer.calls[-1][1]

    trajectory._render_template(
        tokenizer,
        [{"role": "user", "content": "x"}],
        tools=None,
        add_generation_prompt=True,
        reasoning_effort="high",
    )
    assert tokenizer.calls[-1][1]["reasoning_effort"] == "high"


class _Tokenizer:
    unk_token_id = -1

    def __init__(self, kind: str):
        self.kind = kind

    def __call__(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        if text == "\n<think>":
            return {"input_ids": [1, 2]}
        raise AssertionError(f"unexpected tokenization: {text!r}")

    def convert_tokens_to_ids(self, token):
        values = {
            "<|assistant|>": 90,
            "<tool_call>": 4,
        }
        return values.get(token, self.unk_token_id)

    def decode(self, ids, *, skip_special_tokens=False):
        return "TOKENS:" + ",".join(str(i) for i in ids)


class _Bridge:
    path = "/unused/bridge.json"

    def map_numpy(self, ids, *, context):
        del context
        return np.asarray(ids, dtype=np.int64) + 100


class _ToolTokenizer:
    unk_token_id = -1
    eos_token_id = 99

    ids = {
        "<|assistant|>": 51,
        "<|observation|>": 50,
        "</tool_response>": 52,
        "</think>": 12,
        "<tool_call>": 4,
        "</tool_call>": 5,
        "<arg_value>": 19,
        "</arg_value>": 22,
    }

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)

    def __call__(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        assert text == "\n"
        return {"input_ids": [6]}

    def decode(self, ids, *, skip_special_tokens=False):
        del ids, skip_special_tokens
        return "valid assistant tool call"


def test_student_tool_policy_omits_only_tool_structure_newlines(monkeypatch):
    monkeypatch.setattr(
        trajectory,
        "_canonical_assistant_message",
        lambda *args, **kwargs: {"tool_calls": [{"function": {"name": "tool"}}]},
    )
    # [prefix reasoning </think>] [visible pre-tool text] [tool call]. Newline
    # token 6 inside arg_value is payload and must remain trainable/contextual.
    run_ids = [1, 2, 11, 12, 6, 13, 6, 4, 15, 6, 16, 17, 18, 6, 19, 20, 6, 21, 22, 6, 5, 50]
    omitted, masked = trajectory._student_tool_call_policy(
        _ToolTokenizer(),
        SimpleNamespace(),
        run_ids,
        normalize_tool_template=True,
        mask_pre_tool_content=True,
        context="test",
    )

    assert omitted == {9, 13, 19}
    assert masked == {4, 5, 6, 9, 13, 19}
    assert 16 not in masked


def test_student_tool_policy_keeps_post_think_content_when_mask_disabled(monkeypatch):
    monkeypatch.setattr(
        trajectory,
        "_canonical_assistant_message",
        lambda *args, **kwargs: {"tool_calls": [{"function": {"name": "tool"}}]},
    )
    # Positions 4..6 are visible content after </think>. They must remain
    # trainable when pre-tool masking is disabled. Only GLM-4.5 structural
    # newlines inside the valid tool-call grammar are omitted.
    run_ids = [1, 2, 11, 12, 6, 13, 6, 4, 15, 6, 16, 17, 18, 6, 19, 20, 6, 21, 22, 6, 5, 50]
    omitted, masked = trajectory._student_tool_call_policy(
        _ToolTokenizer(),
        SimpleNamespace(),
        run_ids,
        normalize_tool_template=True,
        mask_pre_tool_content=False,
        context="test",
    )

    assert omitted == {9, 13, 19}
    assert masked == omitted
    assert not ({4, 5, 6} & masked)
    assert 16 not in masked


def test_student_tool_policy_fails_closed_for_unparsed_call(monkeypatch):
    monkeypatch.setattr(
        trajectory,
        "_canonical_assistant_message",
        lambda *args, **kwargs: {"content": "malformed"},
    )
    omitted, masked = trajectory._student_tool_call_policy(
        _ToolTokenizer(),
        SimpleNamespace(),
        [12, 6, 4, 15, 6, 5, 50],
        normalize_tool_template=True,
        mask_pre_tool_content=True,
        context="test",
    )

    assert omitted == set()
    assert masked == set()


def test_posthoc_tool_mask_compacts_every_dynamic_sparse_field(monkeypatch):
    monkeypatch.setattr(
        trajectory,
        "_canonical_assistant_message",
        lambda *args, **kwargs: {"tool_calls": [{"function": {"name": "tool"}}]},
    )
    # Structural newlines are at 9/13/19, while position 16 is inside
    # <arg_value> and must survive. Positions 4..6 are visible content between
    # </think> and the final valid <tool_call> and must also be removed.
    response = [1, 2, 11, 12, 6, 13, 6, 4, 15, 6, 16, 17, 18, 6, 19, 20, 6, 21, 22, 6, 5, 50]
    old_mask = np.ones(len(response), dtype=np.int8)
    # Match the old Teacher-template artifact: structural special tokens were
    # already excluded, but ordinary newline/visible-content rows were scored.
    old_mask[[0, 1, 3, 7, 14, 18, 20, 21]] = 0
    old_positions = np.flatnonzero(old_mask)
    rows = len(old_positions)
    candidate_ids = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4)
    sample = Sample(
        index=3,
        tokens=[101, 102] + response,
        response_length=len(response),
        loss_mask=old_mask.tolist(),
        metadata={},
        student_topk_ids=np.arange(rows * 3, dtype=np.int32).reshape(rows, 3),
        student_topk_logprobs=np.zeros((rows, 3), dtype=np.float16),
        teacher_candidate_positions=torch.from_numpy(old_positions.astype(np.int32)),
        teacher_candidate_ids=candidate_ids,
        teacher_candidate_logprobs=torch.full((rows, 4), -3.0, dtype=torch.float16),
        teacher_top128_ids=candidate_ids[:, :3].clone(),
        teacher_cutoff_logprobs=torch.full((rows,), -3.0, dtype=torch.float16),
        teacher_top4096_mass=torch.full((rows,), 0.2, dtype=torch.float32),
        offline_opd_dynamic_teacher_sparse=True,
    )

    stats = trajectory.posthoc_mask_dynamic_scored_sample(
        sample,
        _ToolTokenizer(),
        SimpleNamespace(),
        cache_topk=4,
        dynamic_topk=3,
        context="test",
    )

    expected_positions = np.asarray(
        [position for position in old_positions if position not in {4, 5, 6, 9, 13, 19}],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(
        sample.teacher_candidate_positions.numpy(), expected_positions
    )
    assert sample.loss_mask[16] == 1  # semantic newline inside arg_value
    assert all(sample.loss_mask[position] == 0 for position in (4, 5, 6, 9, 13, 19))
    assert sample.student_topk_ids.shape[0] == len(expected_positions)
    assert sample.student_topk_logprobs.shape[0] == len(expected_positions)
    assert sample.teacher_candidate_ids.shape[0] == len(expected_positions)
    assert sample.teacher_candidate_logprobs.shape[0] == len(expected_positions)
    assert sample.teacher_top128_ids.shape[0] == len(expected_positions)
    assert sample.teacher_cutoff_logprobs.shape[0] == len(expected_positions)
    assert sample.teacher_top4096_mass.shape[0] == len(expected_positions)
    assert stats == {
        "old_trainable_tokens": rows,
        "removed_trainable_tokens": 6,
        "trainable_tokens": rows - 6,
        "parsed_assistant_turns": 1,
    }


def test_posthoc_let_me_mask_is_deterministic_and_compacts_sparse_rows():
    # Two eligible Let/let + me + X triples plus one occurrence whose third
    # position was already masked. probability=1 makes the expected selection
    # independent of the hash values while exercising sparse-row compaction.
    response = [10, 20, 30, 7, 11, 20, 31, 10, 20, 32]
    old_mask = np.ones(len(response), dtype=np.int8)
    old_mask[-1] = 0
    old_positions = np.flatnonzero(old_mask)
    rows = len(old_positions)
    candidate_ids = torch.arange(rows * 4, dtype=torch.int32).reshape(rows, 4)
    sample = Sample(
        index=17,
        tokens=[101, 102] + response,
        response_length=len(response),
        loss_mask=old_mask.tolist(),
        metadata={},
        student_topk_ids=np.arange(rows * 3, dtype=np.int32).reshape(rows, 3),
        student_topk_logprobs=np.zeros((rows, 3), dtype=np.float16),
        teacher_candidate_positions=torch.from_numpy(old_positions.astype(np.int32)),
        teacher_candidate_ids=candidate_ids,
        teacher_candidate_logprobs=torch.full((rows, 4), -3.0, dtype=torch.float16),
        teacher_top128_ids=candidate_ids[:, :3].clone(),
        teacher_cutoff_logprobs=torch.full((rows,), -3.0, dtype=torch.float16),
        teacher_top4096_mass=torch.full((rows,), 0.2, dtype=torch.float32),
        offline_opd_dynamic_teacher_sparse=True,
    )

    stats = trajectory.posthoc_mask_let_me_triplets_dynamic_scored_sample(
        sample,
        let_token_ids={10, 11},
        me_token_ids={20},
        probability=1.0,
        seed=123,
        cache_topk=4,
        dynamic_topk=3,
        context="test",
    )

    np.testing.assert_array_equal(
        sample.teacher_candidate_positions.numpy(), np.asarray([3, 7, 8], dtype=np.int32)
    )
    assert sample.loss_mask == [0, 0, 0, 1, 0, 0, 0, 1, 1, 0]
    assert sample.teacher_candidate_ids.shape == (3, 4)
    assert sample.student_topk_ids.shape == (3, 3)
    assert stats == {
        "old_trainable_tokens": rows,
        "occurrences": 2,
        "selected_occurrences": 2,
        "removed_trainable_tokens": 6,
        "trainable_tokens": 3,
    }
    validate_dynamic_scored_sample(sample, cache_topk=4, dynamic_topk=3)


class _ProtocolTokenizer:
    unk_token_id = -1
    eos_token_id = 99

    ids = {
        "<think>": 10,
        "</think>": 11,
        "<tool_call>": 20,
        "</tool_call>": 21,
        "<arg_key>": 22,
        "</arg_key>": 23,
        "<arg_value>": 24,
        "</arg_value>": 25,
        "<|observation|>": 26,
        "<|endoftext|>": 99,
    }

    def convert_tokens_to_ids(self, token):
        return self.ids.get(token, self.unk_token_id)


def test_posthoc_protocol_anchor_uses_global_ids_and_keeps_semantic_rows():
    # Positions 1 and 5 model pre-existing masked template newlines. The first
    # trainable row after each think delimiter must still be anchored. Function
    # name, argument name/value, and ordinary reasoning/content remain exact
    # Teacher rows.
    response = [
        10,  # existing <think> anchor
        30,  # masked structural newline
        31,  # first reasoning token -> anchor
        32,  # ordinary reasoning -> Teacher
        11,  # </think> -> anchor
        30,  # masked structural newline
        33,  # first post-think token -> anchor
        34,  # ordinary content -> Teacher
        20,  # <tool_call> -> anchor
        40,  # function name -> Teacher
        22,  # <arg_key> -> anchor
        41,  # argument name -> Teacher
        23,  # </arg_key> -> anchor
        24,  # <arg_value> -> anchor
        42,  # argument value -> Teacher
        25,  # </arg_value> -> anchor
        21,  # </tool_call> -> anchor
        99,  # trainable EOS -> anchor
    ]
    loss_mask = np.ones(len(response), dtype=np.int8)
    loss_mask[[1, 5]] = 0
    old_anchor_positions = np.asarray([0], dtype=np.int32)
    teacher_positions = np.asarray(
        [position for position in np.flatnonzero(loss_mask) if position != 0],
        dtype=np.int32,
    )
    rows = len(teacher_positions)
    candidate_ids = torch.tensor([[1, 2, 3, 4]] * rows, dtype=torch.int32)
    student_ids = np.arange(rows * 3, dtype=np.int32).reshape(rows, 3)
    sample = Sample(
        index=23,
        tokens=[101, 102] + response,
        response_length=len(response),
        loss_mask=loss_mask.tolist(),
        metadata={"offline_opd_cross_tokenizer_anchor_sparse": True},
        student_topk_ids=student_ids.copy(),
        student_topk_logprobs=np.full((rows, 3), -2.0, dtype=np.float16),
        teacher_candidate_positions=torch.from_numpy(teacher_positions.copy()),
        teacher_candidate_ids=candidate_ids.clone(),
        teacher_candidate_logprobs=torch.full((rows, 4), -3.0, dtype=torch.float16),
        teacher_top128_ids=torch.tensor([[1, 2, 3]] * rows, dtype=torch.int32),
        teacher_cutoff_logprobs=torch.full((rows,), -3.0, dtype=torch.float16),
        teacher_top4096_mass=torch.full((rows,), 0.2, dtype=torch.float32),
        student_anchor_positions=torch.from_numpy(old_anchor_positions.copy()),
        student_anchor_topk_ids=torch.tensor([[70, 71, 72]], dtype=torch.int32),
        student_anchor_topk_logprobs=torch.full((1, 3), -1.0, dtype=torch.float16),
        student_anchor_reason=torch.tensor([1], dtype=torch.int8),
        offline_opd_dynamic_teacher_sparse=True,
        offline_opd_cross_tokenizer_anchor_sparse=True,
    )
    old_loss_mask = list(sample.loss_mask)

    stats = trajectory.posthoc_anchor_protocol_tokens_dynamic_scored_sample(
        sample,
        _ProtocolTokenizer(),
        cache_topk=4,
        dynamic_topk=3,
        context="test",
    )

    expected_teacher = np.asarray([3, 7, 9, 11, 14], dtype=np.int32)
    expected_anchor = np.asarray(
        [0, 2, 4, 6, 8, 10, 12, 13, 15, 16, 17], dtype=np.int32
    )
    np.testing.assert_array_equal(
        sample.teacher_candidate_positions.numpy(), expected_teacher
    )
    np.testing.assert_array_equal(
        sample.student_anchor_positions.numpy(), expected_anchor
    )
    np.testing.assert_array_equal(
        sample.student_anchor_reason.numpy(),
        np.asarray([1, 4, 5, 6, 7, 7, 7, 7, 7, 7, 8], dtype=np.int8),
    )
    assert sample.loss_mask == old_loss_mask
    assert sample.student_topk_ids.shape == (5, 3)
    assert sample.teacher_candidate_ids.shape == (5, 4)
    assert sample.teacher_candidate_logprobs.shape == (5, 4)
    assert sample.teacher_top128_ids.shape == (5, 3)
    assert sample.teacher_cutoff_logprobs.shape == (5,)
    assert sample.teacher_top4096_mass.shape == (5,)
    assert stats == {
        "old_teacher_rows": 15,
        "old_anchor_rows": 1,
        "selected_positions": 11,
        "moved_rows": 10,
        "new_teacher_rows": 5,
        "new_anchor_rows": 11,
        "reasoning_first_token_rows": 1,
        "reasoning_end_rows": 1,
        "post_reasoning_transition_rows": 1,
        "tool_protocol_rows": 6,
        "stop_token_rows": 1,
    }
    validate_cross_tokenizer_anchor_scored_sample(
        sample, cache_topk=4, dynamic_topk=3, context="test"
    )


def _patch_templates(monkeypatch):
    student = _Tokenizer("student")
    teacher = _Tokenizer("teacher")
    monkeypatch.setattr(
        trajectory,
        "_bridge_models",
        lambda bridge: ("/student", "/teacher"),
    )
    monkeypatch.setattr(
        trajectory,
        "_load_tokenizer",
        lambda model: student if model == "/student" else teacher,
    )
    monkeypatch.setattr(
        trajectory,
        "load_scaffold",
        lambda: SimpleNamespace(tools=()),
    )
    monkeypatch.setattr(
        trajectory.agent_prompt,
        "initial_messages",
        lambda scaffold, problem, workdir: [],
    )
    monkeypatch.setattr(
        trajectory.agent_prompt,
        "render_prompt_tokens",
        lambda tokenizer, messages, tools: (
            ([10], "student")
            if tokenizer.kind == "student"
            else ([80, 90, 91], "teacher")
        ),
    )
    monkeypatch.setattr(
        trajectory.agent_prompt,
        "render_turn_delta",
        lambda tokenizer, messages, obs, last_token_id, tools: [70, 90, 91],
    )
    monkeypatch.setattr(
        trajectory,
        "_extract_observation",
        lambda *args, **kwargs: ("OBS", {"role": "user", "content": "OBS"}),
    )


def _multi_turn_sample():
    # Two trainable assistant runs. Both begin with Student's [newline, <think>]
    # token pair [1, 2], separated by one masked observation connector.
    response = [1, 2, 3, 4, 50, 1, 2, 5]
    return Sample(
        index=7,
        tokens=[10] + response,
        response_length=len(response),
        loss_mask=[1, 1, 1, 1, 0, 1, 1, 1],
        metadata={"problem_statement": "p", "workdir": "/testbed"},
        student_topk_ids=np.arange(7 * 3, dtype=np.int32).reshape(7, 3),
        student_topk_logprobs=np.arange(7 * 3, dtype=np.float16).reshape(7, 3),
    )


def test_native_teacher_prefix_skips_every_student_reasoning_prefix(monkeypatch):
    _patch_templates(monkeypatch)
    rebuilt = trajectory.rebuild_teacher_trajectory(
        _multi_turn_sample(),
        _Bridge(),
        "/teacher",
        preserve_teacher_generation_prefix=True,
    )

    # Initial and post-observation Teacher generation prompts both retain [90, 91]
    # (<|assistant|><think>); only actual Student content is bridged.
    assert rebuilt.input_ids == [80, 90, 91, 103, 104, 70, 90, 91, 105]
    assert rebuilt.prompt_length == 3
    np.testing.assert_array_equal(rebuilt.scored_positions, [0, 1, 5])
    np.testing.assert_array_equal(rebuilt.student_scored_positions, [2, 3, 7])


def test_teacher_template_student_tokens_masks_only_format_rows(monkeypatch):
    _patch_templates(monkeypatch)
    rebuilt = trajectory.rebuild_teacher_trajectory(
        _multi_turn_sample(),
        _Bridge(),
        "/teacher",
        preserve_teacher_generation_prefix=True,
        mask_student_format_tokens=True,
    )

    # Student token 4 is the fake <tool_call> keyword. It remains in Teacher
    # context as mapped id 104, but is not a scored/training position.
    assert rebuilt.input_ids == [80, 90, 91, 103, 104, 70, 90, 91, 105]
    np.testing.assert_array_equal(rebuilt.scored_positions, [0, 5])
    np.testing.assert_array_equal(rebuilt.student_scored_positions, [2, 7])
    assert rebuilt.alignment_stats["student_reasoning_prefix_tokens_skipped"] == 4
    assert rebuilt.alignment_stats["student_format_tokens_masked"] == 1


def test_teacher_template_student_tokens_accepts_missing_student_prefix(monkeypatch):
    _patch_templates(monkeypatch)
    sample = _multi_turn_sample()
    sample.tokens[1:3] = [8, 9]
    rebuilt = trajectory.rebuild_teacher_trajectory(
        sample,
        _Bridge(),
        "/teacher",
        preserve_teacher_generation_prefix=True,
        mask_student_format_tokens=True,
    )

    # The Teacher already supplied its native <think>; malformed Student turns
    # without the usual prefix are replayed verbatim instead of being rejected.
    assert rebuilt.input_ids[:7] == [80, 90, 91, 108, 109, 103, 104]
    np.testing.assert_array_equal(rebuilt.student_scored_positions[:3], [0, 1, 2])


def test_default_teacher_replay_behavior_is_unchanged(monkeypatch):
    _patch_templates(monkeypatch)
    rebuilt = trajectory.rebuild_teacher_trajectory(
        _multi_turn_sample(), _Bridge(), "/teacher"
    )

    # Legacy mode removes Teacher's injected 91 and replays every Student token.
    assert rebuilt.input_ids == [80, 90, 101, 102, 103, 104, 70, 90, 101, 102, 105]
    assert rebuilt.prompt_length == 2
    np.testing.assert_array_equal(rebuilt.scored_positions, [0, 1, 2, 3, 6, 7, 8])
    np.testing.assert_array_equal(
        rebuilt.student_scored_positions, [0, 1, 2, 3, 5, 6, 7]
    )


def test_native_prefix_finalize_masks_and_crops_student_sparse_rows():
    sample = _multi_turn_sample()
    original_ids = sample.student_topk_ids.copy()
    original_logprobs = sample.student_topk_logprobs.copy()
    sample.opd_teacher_student_scored_positions = np.array([2, 3, 7], dtype=np.int32)

    # Request/post-process paths select these same original sparse rows before the
    # artifact is mutated.
    rows, positions = build_student_per_position_ids(sample)
    np.testing.assert_array_equal(positions, [2, 3, 7])
    np.testing.assert_array_equal(rows, original_ids[[2, 3, 6]])

    skipped = trajectory.finalize_native_reasoning_prefix_artifact(sample)
    assert skipped == 4
    assert sample.loss_mask == [0, 0, 1, 1, 0, 0, 0, 1]
    np.testing.assert_array_equal(sample.student_topk_ids, original_ids[[2, 3, 6]])
    np.testing.assert_array_equal(
        sample.student_topk_logprobs, original_logprobs[[2, 3, 6]]
    )
    assert sample.opd_teacher_student_scored_positions is None
    assert sample.metadata["opd_teacher_native_reasoning_prefix"] is True
    assert sample.metadata["opd_teacher_native_reasoning_prefix_skipped_tokens"] == 4


def test_native_prefix_artifact_compacts_into_existing_dynamic_schema():
    sample = _multi_turn_sample()
    sample.reward = 0.0
    sample.metadata["swe_reward"] = 0.0
    sample.opd_teacher_student_scored_positions = np.array([2, 3, 7], dtype=np.int32)
    # The optimized Teacher endpoint returns only the three requested rows.
    sample.teacher_topk_ids = torch.tensor(
        [[8, 7, 6, 5], [4, 3, 2, 1], [9, 10, 11, 12]], dtype=torch.long
    )
    sample.teacher_topk_logprobs = torch.tensor(
        [
            [-2.0, -3.0, -4.0, -5.0],
            [-2.1, -3.1, -4.1, -5.1],
            [-2.2, -3.2, -4.2, -5.2],
        ],
        dtype=torch.float32,
    )

    trajectory.finalize_native_reasoning_prefix_artifact(sample)
    compact_dynamic_teacher_cache_for_disk(sample, cache_topk=4, dynamic_topk=3)
    validate_dynamic_scored_sample(sample, cache_topk=4, dynamic_topk=3)
    torch.testing.assert_close(
        sample.teacher_candidate_positions,
        torch.tensor([2, 3, 7], dtype=torch.int32),
    )


def test_canonical_alignment_uses_distinct_artifact_schema():
    sample = _multi_turn_sample()
    sample.opd_teacher_student_scored_positions = np.array([2, 3, 7], dtype=np.int32)
    sample.opd_teacher_alignment_mode = "canonical_template_v1"
    sample.opd_teacher_alignment_stats = {
        "student_trainable_tokens_before_alignment": 7,
        "student_aligned_tokens": 3,
        "semantic_alignment_rate": 3 / 7,
    }

    trajectory.finalize_teacher_position_alignment_artifact(sample)
    assert sample.metadata["opd_teacher_canonical_template"] is True
    assert sample.metadata["opd_teacher_canonical_template_schema"] == 1
    alignment = sample.metadata["opd_teacher_canonical_alignment"]
    assert alignment["student_skipped_tokens"] == 4
    assert alignment["student_aligned_tokens"] == 3
    assert "opd_teacher_native_reasoning_prefix" not in sample.metadata
    assert sample.opd_teacher_alignment_mode is None
    assert sample.opd_teacher_alignment_stats is None


def test_teacher_template_student_tokens_uses_distinct_artifact_schema():
    sample = _multi_turn_sample()
    sample.opd_teacher_student_scored_positions = np.array([2, 7], dtype=np.int32)
    sample.opd_teacher_alignment_mode = "teacher_template_student_tokens_v1"
    sample.opd_teacher_alignment_stats = {
        "student_reasoning_prefix_tokens_skipped": 4,
        "student_format_tokens_masked": 1,
    }

    trajectory.finalize_teacher_position_alignment_artifact(sample)
    assert sample.metadata["opd_teacher_native_reasoning_prefix"] is True
    assert sample.metadata["opd_teacher_template_student_tokens"] is True
    assert sample.metadata["opd_teacher_template_student_tokens_schema"] == 1
    assert sample.loss_mask == [0, 0, 1, 0, 0, 0, 0, 1]
    assert sample.student_topk_ids.shape == (2, 3)


class _OffsetTokenizer:
    def __init__(self, decoded, encoded_ids, offsets):
        self.decoded = decoded
        self.encoded_ids = encoded_ids
        self.offsets = offsets

    def decode(self, ids, *, skip_special_tokens=False):
        del ids, skip_special_tokens
        return self.decoded

    def __call__(self, text, *, add_special_tokens=False, return_offsets_mapping=False):
        assert text == self.decoded
        assert not add_special_tokens
        value = {"input_ids": self.encoded_ids}
        if return_offsets_mapping:
            value["offset_mapping"] = self.offsets
        return value


def test_canonical_alignment_masks_many_to_one_tokens_without_shifting():
    # Stored Student ids use three pieces for the middle character. Its own
    # canonical encoder and the Teacher each use a different single merged id.
    # A and B remain aligned on both sides of the mismatch.
    student = _OffsetTokenizer(
        "A🎉B",
        [1, 9, 3],
        [(0, 1), (1, 2), (2, 3)],
    )
    teacher = _OffsetTokenizer(
        "A🎉B",
        [101, 10, 103],
        [(0, 1), (1, 2), (2, 3)],
    )

    aligned = trajectory._align_canonical_turn(
        student,
        teacher,
        _Bridge(),
        [1, 4, 5, 6, 3],
        [101, 10, 103],
        "A🎉B",
        context="test",
    )
    assert aligned == [(0, 0), (4, 2)]


def test_canonical_malformed_nested_tool_call_remains_plain_content(monkeypatch):
    # Phase 1 rejected this complete message (the outer call is malformed), even
    # though its tail contains a syntactically valid nested call.  Canonical replay
    # must not invent an execution of that nested call; the following observation
    # is the rollout parser's error response.
    from examples.coding_agent_opd.r2egym import action as agent_action

    monkeypatch.setattr(agent_action, "parse", lambda *args, **kwargs: ("", None))
    remainder = (
        "before<tool_call>try\n"
        "<tool_call>str_replace_editor\n"
        "<arg_key>command</arg_key><arg_value>view</arg_value>\n"
        "</tool_call>"
    )
    message = trajectory._canonical_assistant_message(
        f"\n<think>reasoning</think>{remainder}",
        SimpleNamespace(),
        context="test malformed call",
    )

    assert message == {
        "role": "assistant",
        "reasoning_content": "reasoning",
        "content": remainder,
    }


def test_canonical_parsed_tool_call_with_trailing_payload_remains_plain_content(monkeypatch):
    from examples.coding_agent_opd.r2egym import action as agent_action

    action = SimpleNamespace(
        function_name="str_replace_editor",
        parameters={"command": "view", "path": "/testbed"},
    )
    monkeypatch.setattr(agent_action, "parse", lambda *args, **kwargs: ("", action))
    remainder = "<tool_call>str_replace_editor</tool_call></tool_call></function>"
    message = trajectory._canonical_assistant_message(
        f"\n<think>reasoning</think>{remainder}",
        SimpleNamespace(),
        context="test parsed call with trailing payload",
    )

    assert message == {
        "role": "assistant",
        "reasoning_content": "reasoning",
        "content": remainder,
    }


def test_canonical_unterminated_reasoning_keeps_executed_final_call(monkeypatch):
    from examples.coding_agent_opd.r2egym import action as agent_action

    action = SimpleNamespace(
        function_name="str_replace_editor",
        parameters={"command": "view", "path": "/testbed"},
    )
    monkeypatch.setattr(agent_action, "parse", lambda *args, **kwargs: ("", action))
    raw_call = "<tool_call>str_replace_editor</tool_call>"
    message = trajectory._canonical_assistant_message(
        f"\n<think>unfinished reasoning</antml>\n{raw_call}",
        SimpleNamespace(),
        context="test unterminated reasoning",
    )

    assert message == {
        "role": "assistant",
        "reasoning_content": "unfinished reasoning</antml>\n",
        "content": "",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "str_replace_editor",
                    "arguments": {"command": "view", "path": "/testbed"},
                },
            }
        ],
    }


def test_canonical_missing_reasoning_prefix_keeps_reasoning_and_executed_call(monkeypatch):
    from examples.coding_agent_opd.r2egym import action as agent_action

    action = SimpleNamespace(
        function_name="str_replace_editor",
        parameters={"command": "view", "path": "/testbed/treenode/models.py"},
    )
    monkeypatch.setattr(agent_action, "parse", lambda *args, **kwargs: ("", action))
    raw_call = "<tool_call>str_replace_editor</tool_call>"
    message = trajectory._canonical_assistant_message(
        f"\n Oops, use the correct path.</think>\n\n{raw_call}",
        SimpleNamespace(),
        context="test missing reasoning prefix",
    )

    assert message == {
        "role": "assistant",
        "reasoning_content": "\n Oops, use the correct path.",
        "content": "\n\n",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "str_replace_editor",
                    "arguments": {
                        "command": "view",
                        "path": "/testbed/treenode/models.py",
                    },
                },
            }
        ],
    }


def test_canonical_no_reasoning_delimiters_is_visible_content(monkeypatch):
    from examples.coding_agent_opd.r2egym import action as agent_action

    action = SimpleNamespace(
        function_name="str_replace_editor",
        parameters={"command": "view", "path": "/testbed"},
    )
    monkeypatch.setattr(agent_action, "parse", lambda *args, **kwargs: ("", action))
    raw_call = "<tool_call>str_replace_editor</tool_call>"
    message = trajectory._canonical_assistant_message(
        f"Visible answer before call.\n{raw_call}",
        SimpleNamespace(),
        context="test visible content without reasoning delimiters",
    )

    assert message == {
        "role": "assistant",
        "reasoning_content": "",
        "content": "Visible answer before call.\n",
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": "str_replace_editor",
                    "arguments": {"command": "view", "path": "/testbed"},
                },
            }
        ],
    }


def test_native_prefix_rejects_a_missing_student_prefix(monkeypatch):
    _patch_templates(monkeypatch)
    sample = _multi_turn_sample()
    sample.tokens[1] = 9
    try:
        trajectory.rebuild_teacher_trajectory(
            sample,
            _Bridge(),
            "/teacher",
            preserve_teacher_generation_prefix=True,
        )
    except ValueError as error:
        assert "does not start with" in str(error)
    else:
        raise AssertionError("missing Student reasoning prefix was accepted")


def test_native_prefix_resume_does_not_accept_a_legacy_manifest(tmp_path):
    source_dir = tmp_path / "source_manifests"
    sample_dir = tmp_path / "samples"
    source_dir.mkdir()
    sample_dir.mkdir()
    sample_path = sample_dir / "sample_000007.pt"
    sample_path.write_bytes(b"artifact")
    manifest_path = source_dir / "rollout_0.json"

    legacy = {
        "format": "offline_opd_scored_source_v1",
        "source_rollout_id": "0",
        "samples": [{"index": 7, "path": "samples/sample_000007.pt"}],
    }
    manifest_path.write_text(json.dumps(legacy))
    assert score_teacher_offline._already_scored_as_samples(tmp_path, "0")
    assert not score_teacher_offline._already_scored_as_samples(
        tmp_path, "0", teacher_native_reasoning_prefix=True
    )

    legacy["opd_teacher_native_reasoning_prefix"] = True
    manifest_path.write_text(json.dumps(legacy))
    assert score_teacher_offline._already_scored_as_samples(
        tmp_path, "0", teacher_native_reasoning_prefix=True
    )

    assert not score_teacher_offline._already_scored_as_samples(
        tmp_path,
        "0",
        teacher_native_reasoning_prefix=True,
        teacher_mask_student_format_tokens=True,
    )
    legacy["opd_teacher_template_student_tokens"] = True
    manifest_path.write_text(json.dumps(legacy))
    assert score_teacher_offline._already_scored_as_samples(
        tmp_path,
        "0",
        teacher_native_reasoning_prefix=True,
        teacher_mask_student_format_tokens=True,
    )

    assert not score_teacher_offline._already_scored_as_samples(
        tmp_path,
        "0",
        teacher_native_reasoning_prefix=True,
        teacher_mask_student_format_tokens=True,
        teacher_reasoning_effort="high",
    )
    legacy["opd_teacher_reasoning_effort"] = "high"
    manifest_path.write_text(json.dumps(legacy))
    assert score_teacher_offline._already_scored_as_samples(
        tmp_path,
        "0",
        teacher_native_reasoning_prefix=True,
        teacher_mask_student_format_tokens=True,
        teacher_reasoning_effort="high",
    )
