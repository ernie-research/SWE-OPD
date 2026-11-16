"""Tests for the sparse rollout-student snapshot saved by offline OPD Phase 1."""

import base64
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.coding_agent_opd.generate import _append_student_topk_support
from examples.coding_agent_opd.scripts.offline_opd.offline_utils import (
    compact_baked_teacher_for_disk,
    validate_dump_sample,
    validate_scored_sample,
)
from examples.coding_agent_opd.utils.rollout import _LoopAbort, _abort_sample
from slime.utils.types import Sample


def _blob(array: np.ndarray) -> dict:
    array = np.ascontiguousarray(array)
    return {
        "vals": base64.b64encode(array.tobytes()).decode("ascii"),
        "shape": list(array.shape),
        "dtype": array.dtype.name,
    }


def _phase1_sample(k: int = 3) -> Sample:
    return Sample(
        index=0,
        tokens=[10, 11, 12],
        response_length=2,
        loss_mask=[1, 0],
        reward=0.0,
        metadata={"swe_reward": 0.0},
        student_topk_ids=np.arange(k, dtype=np.int32).reshape(1, k),
        student_topk_logprobs=np.array([[-0.1, -1.5, -2.0]], dtype=np.float16)[:, :k],
    )


def test_append_student_topk_support_decodes_b64_and_appends_turn_chunks():
    sample = Sample()
    ids0 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    vals0 = np.array([[-0.1, -1.0, -2.0], [-0.2, -1.2, -2.2]], dtype=np.float32)
    ids1 = np.array([[7, 8, 9]], dtype=np.int32)
    vals1 = np.array([[-0.3, -1.3, -2.3]], dtype=np.float32)

    for ids, vals in ((ids0, vals0), (ids1, vals1)):
        _append_student_topk_support(
            sample,
            {
                "output_top_logprobs_ids_b64": _blob(ids),
                "output_top_logprobs_b64": _blob(vals),
            },
            len(ids),
            3,
            save_logprobs=True,
        )

    np.testing.assert_array_equal(np.concatenate(sample.student_topk_ids), np.vstack([ids0, ids1]))
    actual = np.concatenate(sample.student_topk_logprobs)
    assert actual.dtype == np.float16
    np.testing.assert_allclose(actual, np.vstack([vals0, vals1]), rtol=1e-3, atol=1e-3)


def test_append_student_topk_support_supports_legacy_tuple_json():
    sample = Sample()
    top = [
        [[-0.1, 5, "a"], [-1.2, 6, "b"]],
        [[-0.2, 7, "c"], [-1.3, 8, "d"]],
    ]
    _append_student_topk_support(
        sample,
        {"output_top_logprobs": top},
        2,
        2,
        save_logprobs=True,
    )
    np.testing.assert_array_equal(sample.student_topk_ids[0], [[5, 6], [7, 8]])
    np.testing.assert_allclose(sample.student_topk_logprobs[0], [[-0.1, -1.2], [-0.2, -1.3]], atol=1e-3)


def test_append_student_topk_support_rejects_missing_or_misaligned_values():
    ids = np.array([[1, 2]], dtype=np.int32)
    with pytest.raises(_LoopAbort, match="top-k logprobs"):
        _append_student_topk_support(
            Sample(),
            {"output_top_logprobs_ids_b64": _blob(ids)},
            1,
            2,
            save_logprobs=True,
        )


def test_online_ids_only_path_does_not_require_or_store_snapshot_values():
    sample = Sample()
    ids = np.array([[1, 2]], dtype=np.int32)
    _append_student_topk_support(
        sample,
        {"output_top_logprobs_ids_b64": _blob(ids)},
        1,
        2,
        save_logprobs=False,
    )
    np.testing.assert_array_equal(sample.student_topk_ids[0], ids)
    assert sample.student_topk_logprobs is None


def test_student_snapshot_filters_padded_vocab_and_keeps_rank_order(monkeypatch):
    monkeypatch.setenv("OFFLINE_OPD_VALID_VOCAB_SIZE", "10")
    sample = Sample()
    ids = np.array([[12, 3, 11, 5, 7], [2, 10, 4, 13, 6]], dtype=np.int32)
    vals = np.array(
        [[-0.1, -0.2, -0.3, -0.4, -0.5], [-0.6, -0.7, -0.8, -0.9, -1.0]],
        dtype=np.float32,
    )
    _append_student_topk_support(
        sample,
        {
            "output_top_logprobs_ids_b64": _blob(ids),
            "output_top_logprobs_b64": _blob(vals),
        },
        2,
        3,
        save_logprobs=True,
    )
    np.testing.assert_array_equal(sample.student_topk_ids[0], [[3, 5, 7], [2, 4, 6]])
    np.testing.assert_allclose(
        sample.student_topk_logprobs[0],
        [[-0.2, -0.4, -0.5], [-0.6, -0.8, -1.0]],
        atol=1e-3,
    )


def test_student_snapshot_ignores_nonfinite_padded_vocab_logprobs(monkeypatch):
    monkeypatch.setenv("OFFLINE_OPD_VALID_VOCAB_SIZE", "10")
    sample = Sample()
    ids = np.array([[12, 3, 11, 5, 7], [2, 10, 4, 13, 6]], dtype=np.int32)
    vals = np.array(
        [[-np.inf, -0.2, -np.inf, -0.4, -0.5], [-0.6, -np.inf, -0.8, -np.inf, -1.0]],
        dtype=np.float32,
    )
    _append_student_topk_support(
        sample,
        {
            "output_top_logprobs_ids_b64": _blob(ids),
            "output_top_logprobs_b64": _blob(vals),
        },
        2,
        3,
        save_logprobs=True,
    )
    np.testing.assert_array_equal(sample.student_topk_ids[0], [[3, 5, 7], [2, 4, 6]])
    np.testing.assert_allclose(
        sample.student_topk_logprobs[0],
        [[-0.2, -0.4, -0.5], [-0.6, -0.8, -1.0]],
        atol=1e-3,
    )


def test_student_snapshot_skips_constrained_nonfinite_valid_token(monkeypatch):
    monkeypatch.setenv("OFFLINE_OPD_VALID_VOCAB_SIZE", "10")
    sample = Sample()
    ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    vals = np.array([[-0.1, -np.inf, -0.3, -0.4]], dtype=np.float32)
    _append_student_topk_support(
        sample,
        {
            "output_top_logprobs_ids_b64": _blob(ids),
            "output_top_logprobs_b64": _blob(vals),
        },
        1,
        3,
        save_logprobs=True,
    )
    np.testing.assert_array_equal(sample.student_topk_ids[0], [[1, 3, 4]])
    np.testing.assert_allclose(sample.student_topk_logprobs[0], [[-0.1, -0.3, -0.4]], atol=1e-3)


def test_student_snapshot_skips_values_that_overflow_storage_dtype(monkeypatch):
    monkeypatch.setenv("OFFLINE_OPD_VALID_VOCAB_SIZE", "10")
    sample = Sample()
    ids = np.array([[1, 2, 3, 4]], dtype=np.int32)
    vals = np.array([[-0.1, -1e30, -0.3, -0.4]], dtype=np.float32)
    _append_student_topk_support(
        sample,
        {
            "output_top_logprobs_ids_b64": _blob(ids),
            "output_top_logprobs_b64": _blob(vals),
        },
        1,
        3,
        save_logprobs=True,
    )
    np.testing.assert_array_equal(sample.student_topk_ids[0], [[1, 3, 4]])
    np.testing.assert_allclose(sample.student_topk_logprobs[0], [[-0.1, -0.3, -0.4]], atol=1e-3)


def test_snapshot_validation_roundtrip_and_phase2_preservation():
    sample = _phase1_sample()
    validate_dump_sample(sample, 3, require_student_topk_logprobs=True)

    restored = Sample.from_dict(sample.to_dict())
    assert restored.student_topk_logprobs.dtype == np.float16
    np.testing.assert_array_equal(restored.student_topk_logprobs, sample.student_topk_logprobs)

    restored.student_support_ids = torch.tensor([[0, 1, 2], [9, 9, 9]], dtype=torch.long)
    restored.teacher_logprobs_on_student = torch.tensor(
        [[-0.2, -1.2, -2.2], [0.0, 0.0, 0.0]], dtype=torch.float32
    )
    restored.opd_topk_overlap = torch.tensor([0.5, 0.0], dtype=torch.float32)
    compact_baked_teacher_for_disk(restored, 3)
    assert restored.student_support_ids is None
    assert restored.offline_opd_teacher_sparse
    assert tuple(restored.teacher_logprobs_on_student.shape) == (1, 3)
    assert tuple(restored.opd_topk_overlap.shape) == (1,)
    validate_scored_sample(
        restored,
        3,
        require_student_topk_logprobs=True,
        require_sparse_teacher=True,
    )

    loaded = Sample.from_dict(restored.to_dict())
    loaded.expand_offline_opd_sparse_teacher()
    assert not loaded.offline_opd_teacher_sparse
    assert tuple(loaded.teacher_logprobs_on_student.shape) == (2, 3)
    assert tuple(loaded.student_support_ids.shape) == (2, 3)
    assert tuple(loaded.rollout_student_logprobs_on_support.shape) == (2, 3)
    torch.testing.assert_close(
        loaded.teacher_logprobs_on_student[0],
        torch.tensor([-0.2, -1.2, -2.2]),
    )
    torch.testing.assert_close(loaded.teacher_logprobs_on_student[1], torch.zeros(3))
    torch.testing.assert_close(
        loaded.rollout_student_logprobs_on_support[0],
        torch.tensor([-0.1, -1.5, -2.0], dtype=torch.float16),
        rtol=1e-3,
        atol=1e-3,
    )
    torch.testing.assert_close(
        loaded.rollout_student_logprobs_on_support[1],
        torch.zeros(3, dtype=torch.float16),
    )
    torch.testing.assert_close(loaded.opd_topk_overlap, torch.tensor([0.5, 0.0]))


def test_snapshot_is_backward_compatible_but_strict_mode_requires_it():
    sample = _phase1_sample()
    sample.student_topk_logprobs = None
    validate_dump_sample(sample, 3)
    with pytest.raises(ValueError, match="student_topk_logprobs is missing"):
        validate_dump_sample(sample, 3, require_student_topk_logprobs=True)


def test_snapshot_validator_requires_float16_and_abort_clears_it():
    sample = _phase1_sample()
    sample.student_topk_logprobs = sample.student_topk_logprobs.astype(np.float32)
    with pytest.raises(ValueError, match="must be float16"):
        validate_dump_sample(sample, 3, require_student_topk_logprobs=True)

    _abort_sample(sample, "test", "instance")
    assert sample.student_topk_ids is None
    assert sample.student_topk_logprobs is None


def test_debug_rollout_loader_expands_sparse_teacher_artifact(tmp_path):
    from slime.ray.rollout import RolloutManager

    sample = _phase1_sample()
    sample.student_support_ids = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    sample.teacher_logprobs_on_student = torch.tensor(
        [[-0.2, -1.2, -2.2], [0.0, 0.0, 0.0]], dtype=torch.float32
    )
    compact_baked_teacher_for_disk(sample, 3)
    path = tmp_path / "rollout_0.pt"
    torch.save({"samples": [sample.to_dict()]}, path)

    manager = SimpleNamespace(
        args=SimpleNamespace(
            load_debug_rollout_data=str(tmp_path / "rollout_{rollout_id}.pt"),
            load_debug_rollout_data_subsample=None,
        )
    )
    rollout_manager_class = RolloutManager.__ray_metadata__.modified_class
    loaded, metrics = rollout_manager_class._get_rollout_data(manager, 0)
    assert metrics is None
    assert not loaded[0].offline_opd_teacher_sparse
    assert tuple(loaded[0].student_support_ids.shape) == (2, 3)
    assert tuple(loaded[0].teacher_logprobs_on_student.shape) == (2, 3)
    assert tuple(loaded[0].rollout_student_logprobs_on_support.shape) == (2, 3)


def test_legacy_dense_teacher_sample_is_not_modified_by_sparse_expander():
    sample = _phase1_sample()
    support = torch.tensor([[0, 1, 2], [0, 0, 0]], dtype=torch.long)
    teacher = torch.zeros((2, 3), dtype=torch.float32)
    sample.student_support_ids = support
    sample.teacher_logprobs_on_student = teacher
    sample.expand_offline_opd_sparse_teacher()
    assert sample.student_support_ids is support
    assert sample.teacher_logprobs_on_student is teacher
