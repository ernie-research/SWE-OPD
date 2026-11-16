"""Tests for the native sparse teacher cache used by dynamic offline OPD."""

import numpy as np
import pytest
import torch

from examples.coding_agent_opd.scripts.offline_opd.offline_utils import (
    compact_dynamic_teacher_cache_for_disk,
    truncate_dynamic_scored_sample_support,
    validate_cross_tokenizer_anchor_scored_sample,
    validate_dynamic_scored_sample,
)
from examples.coding_agent_opd.scripts.offline_opd.teacher_trajectory import (
    finalize_cross_tokenizer_anchor_artifact,
)
from slime.utils.types import Sample

NUM_GPUS = 0


def test_dynamic_teacher_cache_stays_native_sparse_and_token_id_sorted():
    sample = Sample(
        index=0,
        tokens=[10, 11, 12],
        response_length=2,
        loss_mask=[1, 0],
        reward=0.0,
        metadata={"swe_reward": 0.0},
        student_topk_ids=np.array([[0, 1, 2]], dtype=np.int32),
        student_topk_logprobs=np.array([[-0.1, -1.5, -2.0]], dtype=np.float16),
    )
    sample.teacher_topk_ids = torch.tensor(
        [[5, 2, 9, 1], [0, 0, 0, 0]], dtype=torch.long
    )
    sample.teacher_topk_logprobs = torch.tensor(
        [[-1.0, -2.0, -3.0, -4.0], [0.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    compact_dynamic_teacher_cache_for_disk(sample, 4, 3)

    assert sample.offline_opd_dynamic_teacher_sparse
    assert sample.teacher_topk_ids is None
    assert sample.teacher_topk_logprobs is None
    assert sample.teacher_candidate_ids.dtype == torch.int32
    assert sample.teacher_candidate_logprobs.dtype == torch.float16
    torch.testing.assert_close(
        sample.teacher_candidate_ids,
        torch.tensor([[1, 2, 5, 9]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        sample.teacher_top128_ids,
        torch.tensor([[5, 2, 9]], dtype=torch.int32),
    )
    validate_dynamic_scored_sample(sample, 4, 3)

    restored = Sample.from_dict(sample.to_dict())
    restored.expand_offline_opd_sparse_teacher()
    # The legacy expander must not turn [N, cache_K] back into [R, cache_K].
    assert restored.offline_opd_dynamic_teacher_sparse
    assert tuple(restored.teacher_candidate_ids.shape) == (1, 4)
    with pytest.raises(ValueError, match="artifact marker is missing"):
        validate_cross_tokenizer_anchor_scored_sample(restored, 4, 3)


def test_cross_tokenizer_artifact_partitions_every_trainable_position():
    student_ids = np.arange(3 * 2, dtype=np.int32).reshape(3, 2)
    student_lp = np.asarray(
        [[-0.2, -2.0], [-0.3, -2.1], [-0.4, -2.2]], dtype=np.float16
    )
    sample = Sample(
        index=7,
        tokens=[99, 10, 11, 12, 13],
        response_length=4,
        loss_mask=[1, 1, 0, 1],
        reward=0.0,
        metadata={"swe_reward": 0.0},
        student_topk_ids=student_ids,
        student_topk_logprobs=student_lp,
        opd_teacher_student_scored_positions=np.asarray([1], dtype=np.int32),
        opd_teacher_student_anchor_positions=np.asarray([0, 3], dtype=np.int32),
        opd_teacher_student_anchor_reasons=np.asarray([1, 3], dtype=np.int8),
        opd_teacher_alignment_mode="canonical_template_v1",
        opd_teacher_alignment_stats={"semantic_alignment_rate": 1 / 3},
    )
    # The optimized scorer returns only the one requested Teacher row.
    sample.teacher_topk_ids = torch.tensor([[8, 4, 6]], dtype=torch.long)
    sample.teacher_topk_logprobs = torch.tensor(
        [[-1.0, -2.0, -3.0]], dtype=torch.float32
    )

    assert finalize_cross_tokenizer_anchor_artifact(sample) == 2
    assert sample.loss_mask == [1, 1, 0, 1]
    np.testing.assert_array_equal(sample.student_topk_ids, student_ids[[1]])
    torch.testing.assert_close(
        sample.student_anchor_positions, torch.tensor([0, 3], dtype=torch.int32)
    )
    torch.testing.assert_close(
        sample.student_anchor_topk_ids,
        torch.from_numpy(student_ids[[0, 2]]),
    )

    compact_dynamic_teacher_cache_for_disk(sample, cache_topk=3, dynamic_topk=2)
    validate_cross_tokenizer_anchor_scored_sample(sample, 3, 2)
    assert sample.offline_opd_dynamic_teacher_sparse
    assert sample.offline_opd_cross_tokenizer_anchor_sparse
    torch.testing.assert_close(
        sample.teacher_candidate_positions, torch.tensor([1], dtype=torch.int32)
    )


def test_dynamic_support_can_be_narrowed_in_memory_without_touching_cache():
    sample = Sample(
        index=9,
        tokens=[99, 10, 11],
        response_length=2,
        loss_mask=[1, 1],
        reward=0.0,
        metadata={"swe_reward": 0.0},
        student_topk_ids=np.arange(4, dtype=np.int32).reshape(1, 4),
        student_topk_logprobs=np.asarray(
            [[-0.1, -1.0, -2.0, -3.0]], dtype=np.float16
        ),
    )
    sample.teacher_candidate_positions = torch.tensor([0], dtype=torch.int32)
    sample.teacher_candidate_ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32)
    sample.teacher_candidate_logprobs = torch.tensor(
        [[-0.1, -1.0, -2.0, -3.0, -4.0]], dtype=torch.float16
    )
    sample.teacher_top128_ids = torch.tensor([[5, 4, 3, 2]], dtype=torch.int32)
    sample.teacher_cutoff_logprobs = torch.tensor([-4.0], dtype=torch.float16)
    sample.teacher_top4096_mass = torch.tensor([0.9], dtype=torch.float32)
    sample.student_anchor_positions = torch.tensor([1], dtype=torch.int32)
    sample.student_anchor_topk_ids = torch.tensor([[9, 8, 7, 6]], dtype=torch.int32)
    sample.student_anchor_topk_logprobs = torch.tensor(
        [[-0.2, -1.2, -2.2, -3.2]], dtype=torch.float16
    )
    sample.student_anchor_reason = torch.tensor([0], dtype=torch.int8)
    sample.offline_opd_dynamic_teacher_sparse = True
    sample.offline_opd_cross_tokenizer_anchor_sparse = True
    sample.metadata["offline_opd_cross_tokenizer_anchor_sparse"] = True

    candidate_ids_before = sample.teacher_candidate_ids.clone()
    assert truncate_dynamic_scored_sample_support(sample, 2) == 4
    validate_cross_tokenizer_anchor_scored_sample(sample, 5, 2)

    torch.testing.assert_close(sample.teacher_candidate_ids, candidate_ids_before)
    torch.testing.assert_close(
        sample.teacher_top128_ids, torch.tensor([[5, 4]], dtype=torch.int32)
    )
    np.testing.assert_array_equal(
        sample.student_topk_ids, np.asarray([[0, 1]], dtype=np.int32)
    )
    torch.testing.assert_close(
        sample.student_anchor_topk_ids,
        torch.tensor([[9, 8]], dtype=torch.int32),
    )
    assert sample.metadata["offline_opd_runtime_support_source_topk"] == 4
    assert sample.metadata["offline_opd_runtime_support_topk"] == 2
