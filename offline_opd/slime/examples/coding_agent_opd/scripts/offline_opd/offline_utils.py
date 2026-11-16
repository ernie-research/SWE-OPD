"""Validation helpers shared by the three offline-OPD phases."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from numbers import Real
from pathlib import Path

import numpy as np
import torch

from slime.utils.types import Sample


ROLLOUT_RE = re.compile(r"rollout_(\d+)\.pt$")


@dataclass(frozen=True)
class RolloutDirectorySummary:
    num_files: int
    num_samples: int
    samples_per_file: int


def discover_rollout_files(directory: str | Path) -> list[Path]:
    """Return rollout files in numeric order, rejecting gaps and stray indices."""
    directory = Path(directory)
    indexed: list[tuple[int, Path]] = []
    for path in directory.glob("rollout_*.pt"):
        match = ROLLOUT_RE.fullmatch(path.name)
        if match is not None:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if not indexed:
        raise ValueError(f"no rollout_*.pt files under {directory}")

    indices = [index for index, _ in indexed]
    expected = list(range(len(indexed)))
    if indices != expected:
        raise ValueError(f"rollout files under {directory} are not contiguous: got {indices}, expected {expected}")
    return [path for _, path in indexed]


def load_rollout_file(path: str | Path) -> tuple[list[Sample], object]:
    path = Path(path)
    blob = torch.load(path, weights_only=False)
    if isinstance(blob, dict) and blob.get("format") in {
        "offline_opd_sample_manifest_v1",
        "offline_opd_sample_manifest_v2",
    }:
        files = blob.get("sample_files")
        indices = blob.get("sample_indices")
        if not isinstance(files, list) or not files or not isinstance(indices, list) or len(files) != len(indices):
            raise ValueError(f"{path}: invalid offline OPD sample manifest")
        manifest_base = path.resolve().parent
        if blob.get("format") == "offline_opd_sample_manifest_v2":
            sample_root = blob.get("sample_root")
            if not isinstance(sample_root, str) or Path(sample_root).is_absolute():
                raise ValueError(f"{path}: invalid offline OPD v2 sample_root")
            base = (manifest_base / sample_root).resolve()
            if not base.is_dir():
                raise ValueError(f"{path}: offline OPD v2 sample_root is not a directory: {base}")
        else:
            base = manifest_base
        allow_external_sample_symlinks = blob.get("allow_external_sample_symlinks", False)
        if not isinstance(allow_external_sample_symlinks, bool):
            raise ValueError(f"{path}: invalid allow_external_sample_symlinks flag")
        samples = []
        for relative, expected_index in zip(files, indices, strict=True):
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError(f"{path}: invalid offline OPD sample path: {relative!r}")
            logical_sample_path = Path(os.path.abspath(base / relative))
            try:
                logical_sample_path.relative_to(base)
            except ValueError as error:
                raise ValueError(f"{path}: offline OPD sample escapes sample root: {relative}") from error
            sample_path = logical_sample_path.resolve()
            if not allow_external_sample_symlinks:
                try:
                    sample_path.relative_to(base)
                except ValueError as error:
                    raise ValueError(
                        f"{path}: offline OPD sample escapes sample root: {relative}"
                    ) from error
            value = torch.load(logical_sample_path, weights_only=False)
            if (
                not isinstance(value, dict)
                or value.get("format") != "offline_opd_scored_sample_v1"
                or value.get("index") != expected_index
                or not isinstance(value.get("sample"), dict)
            ):
                raise ValueError(f"{sample_path}: invalid offline OPD scored sample")
            sample = Sample.from_dict(value["sample"])
            sample.rollout_id = blob.get("rollout_id")
            samples.append(sample)
        return samples, blob.get("rollout_id")
    if not isinstance(blob, dict) or not isinstance(blob.get("samples"), list) or not blob["samples"]:
        raise ValueError(f"{path}: expected a non-empty {{'samples': [...]}} rollout dump")
    return [Sample.from_dict(item) for item in blob["samples"]], blob.get("rollout_id")


def student_topk_array(sample: Sample) -> np.ndarray:
    value = sample.student_topk_ids
    if isinstance(value, list):
        value = np.concatenate(value) if value else np.empty((0, 0), dtype=np.int32)
    return np.asarray(value)


def student_topk_logprobs_array(sample: Sample) -> np.ndarray:
    value = sample.student_topk_logprobs
    if isinstance(value, list):
        value = np.concatenate(value) if value else np.empty((0, 0), dtype=np.float16)
    return np.asarray(value)


def _validate_student_topk_snapshot(
    sample: Sample,
    k: int,
    *,
    context: str,
    require_student_topk_logprobs: bool,
) -> None:
    if sample.student_topk_ids is None:
        raise ValueError(f"{context}: student_topk_ids is missing")

    ids = student_topk_array(sample)
    expected_rows = int(np.count_nonzero(np.asarray(sample.loss_mask, dtype=bool)))
    expected_shape = (expected_rows, k)
    if ids.shape != expected_shape:
        raise ValueError(f"{context}: student_topk_ids shape {ids.shape} != {expected_shape}")
    if not np.issubdtype(ids.dtype, np.integer):
        raise ValueError(f"{context}: student_topk_ids must be integer, got {ids.dtype}")
    if ids.size and int(ids.min()) < 0:
        raise ValueError(f"{context}: student_topk_ids contains a negative token id")

    if sample.student_topk_logprobs is None:
        if require_student_topk_logprobs:
            raise ValueError(f"{context}: student_topk_logprobs is missing")
        return
    logprobs = student_topk_logprobs_array(sample)
    if logprobs.shape != ids.shape:
        raise ValueError(
            f"{context}: student_topk_logprobs shape {logprobs.shape} != ids shape {ids.shape}"
        )
    if logprobs.dtype != np.float16:
        raise ValueError(
            f"{context}: student_topk_logprobs must be float16, got {logprobs.dtype}"
        )
    if not np.isfinite(logprobs).all():
        raise ValueError(f"{context}: student_topk_logprobs contains NaN/Inf")


def validate_dump_sample(
    sample: Sample,
    k: int,
    *,
    context: str = "sample",
    require_student_topk_logprobs: bool = False,
) -> None:
    if not sample.tokens:
        raise ValueError(f"{context}: empty tokens")
    if sample.response_length <= 0:
        raise ValueError(f"{context}: response_length={sample.response_length}")
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        actual = None if sample.loss_mask is None else len(sample.loss_mask)
        raise ValueError(f"{context}: loss_mask length {actual} != response_length {sample.response_length}")
    if not isinstance(sample.reward, Real):
        raise ValueError(f"{context}: reward must be scalar, got {type(sample.reward).__name__}")
    if (sample.metadata or {}).get("swe_reward") is None:
        raise ValueError(f"{context}: metadata.swe_reward is missing")
    if (
        sample.teacher_logprobs_on_student is not None
        or sample.student_support_ids is not None
        or sample.offline_opd_teacher_sparse
        or sample.rollout_student_logprobs_on_support is not None
    ):
        raise ValueError(f"{context}: Phase-1 dump unexpectedly contains baked teacher tensors")
    _validate_student_topk_snapshot(
        sample,
        k,
        context=context,
        require_student_topk_logprobs=require_student_topk_logprobs,
    )


def validate_token_dump_sample(sample: Sample, *, context: str = "sample") -> None:
    """Validate a Phase-1 trajectory for sampled-token offline OPD."""
    if not sample.tokens:
        raise ValueError(f"{context}: empty tokens")
    if sample.response_length <= 0:
        raise ValueError(f"{context}: response_length={sample.response_length}")
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        actual = None if sample.loss_mask is None else len(sample.loss_mask)
        raise ValueError(
            f"{context}: loss_mask length {actual} != response_length {sample.response_length}"
        )
    if not isinstance(sample.reward, Real):
        raise ValueError(f"{context}: reward must be scalar, got {type(sample.reward).__name__}")
    if (sample.metadata or {}).get("swe_reward") is None:
        raise ValueError(f"{context}: metadata.swe_reward is missing")
    if sample.rollout_log_probs is None or len(sample.rollout_log_probs) != sample.response_length:
        actual = None if sample.rollout_log_probs is None else len(sample.rollout_log_probs)
        raise ValueError(
            f"{context}: rollout_log_probs length {actual} != response_length {sample.response_length}"
        )
    if not np.isfinite(np.asarray(sample.rollout_log_probs, dtype=np.float32)).all():
        raise ValueError(f"{context}: rollout_log_probs contains NaN/Inf")
    if sample.teacher_log_probs is not None or sample.offline_opd_token_teacher_sparse:
        raise ValueError(f"{context}: Phase-1 dump unexpectedly contains token teacher data")


def compact_token_teacher_for_disk(
    sample: Sample,
    *,
    context: str = "sample",
    keep_topk_diagnostics: bool = False,
) -> None:
    """Persist token teacher log-probs only at trainable response positions."""
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        raise ValueError(f"{context}: invalid response-aligned loss_mask")
    teacher_lp = torch.as_tensor(sample.teacher_log_probs, dtype=torch.float32).reshape(-1)
    if teacher_lp.numel() != sample.response_length:
        raise ValueError(
            f"{context}: teacher_log_probs length {teacher_lp.numel()} "
            f"!= response_length {sample.response_length}"
        )
    if not torch.isfinite(teacher_lp).all():
        raise ValueError(f"{context}: teacher_log_probs contains NaN/Inf")
    positions = torch.from_numpy(
        np.flatnonzero(np.asarray(sample.loss_mask, dtype=bool))
    ).long()
    sample.teacher_log_probs = teacher_lp[positions].contiguous()
    sample.offline_opd_token_teacher_sparse = True
    if not keep_topk_diagnostics:
        # A token-mode Phase 2 may intentionally reuse trajectories produced by
        # a previous Top-K Phase 1. None of these K-wide snapshots is consumed
        # by the sampled-token loss, so drop them in the lightweight schema.
        sample.student_topk_ids = None
        sample.student_topk_logprobs = None
        sample.rollout_student_logprobs_on_support = None
        sample.student_support_ids = None
        sample.teacher_logprobs_on_student = None
        sample.offline_opd_teacher_sparse = False
        sample.teacher_topk_ids = None
        sample.teacher_topk_logprobs = None
        sample.opd_topk_overlap = None


def compact_dynamic_teacher_cache_for_disk(
    sample: Sample,
    cache_topk: int,
    dynamic_topk: int,
    *,
    context: str = "sample",
) -> None:
    """Convert a dense teacher Top-K response into the dynamic sparse schema.

    Only trainable response rows are retained. Candidate ids are sorted by token
    id (not probability) once during Phase 2, making every Phase-3 lookup an
    O(log K) ``searchsorted`` without constructing a dense vocabulary table.
    """
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        raise ValueError(f"{context}: invalid response-aligned loss_mask")
    if not 0 < dynamic_topk <= cache_topk:
        raise ValueError(
            f"{context}: require 0 < dynamic_topk <= cache_topk, got "
            f"{dynamic_topk} and {cache_topk}"
        )
    ids = torch.as_tensor(sample.teacher_topk_ids, dtype=torch.long)
    logprobs = torch.as_tensor(sample.teacher_topk_logprobs, dtype=torch.float32)
    if sample.offline_opd_cross_tokenizer_anchor_sparse:
        positions = torch.as_tensor(
            sample.teacher_candidate_positions, dtype=torch.int32
        ).reshape(-1)
    else:
        positions = torch.from_numpy(
            np.flatnonzero(np.asarray(sample.loss_mask, dtype=bool)).astype(np.int32)
        )
    dense_expected = (sample.response_length, cache_topk)
    sparse_expected = (int(positions.numel()), cache_topk)
    shapes = (tuple(ids.shape), tuple(logprobs.shape))
    if shapes not in {
        (dense_expected, dense_expected),
        (sparse_expected, sparse_expected),
    }:
        raise ValueError(
            f"{context}: teacher cache shape mismatch: ids={tuple(ids.shape)}, "
            f"logprobs={tuple(logprobs.shape)}, expected dense={dense_expected} "
            f"or sparse={sparse_expected}"
        )
    if not torch.isfinite(logprobs).all():
        raise ValueError(f"{context}: teacher cache contains NaN/Inf")
    # The optimized dynamic teacher already returns only trainable rows, in
    # ascending response-position order.  Keep accepting the original dense
    # response so old artifacts and the non-optimized path remain compatible.
    if tuple(ids.shape) == sparse_expected:
        ranked_ids = ids
        ranked_lp = logprobs
    else:
        row_index = positions.long()
        ranked_ids = ids[row_index]
        ranked_lp = logprobs[row_index]
    sorted_ids, order = torch.sort(ranked_ids, dim=-1)
    sorted_lp = torch.gather(ranked_lp, 1, order)

    sample.teacher_candidate_positions = positions.contiguous()
    sample.teacher_candidate_ids = sorted_ids.to(torch.int32).contiguous()
    sample.teacher_candidate_logprobs = sorted_lp.to(torch.float16).contiguous()
    sample.teacher_top128_ids = ranked_ids[:, :dynamic_topk].to(torch.int32).contiguous()
    sample.teacher_cutoff_logprobs = ranked_lp[:, -1].to(torch.float16).contiguous()
    sample.teacher_top4096_mass = ranked_lp.exp().sum(dim=-1).to(torch.float32).contiguous()
    sample.offline_opd_dynamic_teacher_sparse = True

    # These response-aligned tensors are the temporary SGLang extraction format.
    # Keeping them would duplicate the sparse cache and invite the legacy loss path.
    sample.teacher_topk_ids = None
    sample.teacher_topk_logprobs = None
    sample.student_support_ids = None
    sample.teacher_logprobs_on_student = None
    sample.offline_opd_teacher_sparse = False
    sample.opd_topk_overlap = None


def validate_dynamic_scored_sample(
    sample: Sample,
    cache_topk: int,
    dynamic_topk: int,
    *,
    context: str = "sample",
) -> None:
    """Validate the native sparse artifact consumed by dynamic Phase 3."""
    if not sample.offline_opd_dynamic_teacher_sparse:
        raise ValueError(f"{context}: dynamic teacher cache is not marked sparse")
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        raise ValueError(f"{context}: invalid response-aligned loss_mask")
    all_trainable = torch.from_numpy(
        np.flatnonzero(np.asarray(sample.loss_mask, dtype=bool)).astype(np.int32)
    )
    positions = torch.as_tensor(sample.teacher_candidate_positions)
    n = int(positions.numel())
    ids = torch.as_tensor(sample.teacher_candidate_ids)
    lp = torch.as_tensor(sample.teacher_candidate_logprobs)
    teacher_head = torch.as_tensor(sample.teacher_top128_ids)
    cutoff = torch.as_tensor(sample.teacher_cutoff_logprobs)
    mass = torch.as_tensor(sample.teacher_top4096_mass)
    if tuple(positions.shape) != (n,):
        raise ValueError(f"{context}: dynamic positions must be one-dimensional")
    positions_i32 = positions.to(torch.int32)
    if n and (positions_i32[1:] <= positions_i32[:-1]).any():
        raise ValueError(f"{context}: dynamic positions are not strictly increasing")
    if sample.offline_opd_cross_tokenizer_anchor_sparse:
        anchor_positions = torch.as_tensor(sample.student_anchor_positions).to(
            torch.int32
        )
        anchor_ids = torch.as_tensor(sample.student_anchor_topk_ids)
        anchor_lp = torch.as_tensor(sample.student_anchor_topk_logprobs)
        anchor_reason = torch.as_tensor(sample.student_anchor_reason)
        if anchor_positions.ndim != 1 or (
            anchor_positions.numel()
            and (anchor_positions[1:] <= anchor_positions[:-1]).any()
        ):
            raise ValueError(f"{context}: anchor positions are not strictly increasing")
        if tuple(anchor_ids.shape) != (anchor_positions.numel(), dynamic_topk):
            raise ValueError(f"{context}: invalid Student anchor id shape")
        if tuple(anchor_lp.shape) != tuple(anchor_ids.shape):
            raise ValueError(f"{context}: invalid Student anchor logprob shape")
        if tuple(anchor_reason.shape) != (anchor_positions.numel(),):
            raise ValueError(f"{context}: invalid Student anchor reason shape")
        if anchor_ids.dtype != torch.int32 or anchor_lp.dtype != torch.float16:
            raise ValueError(
                f"{context}: expected anchor int32 ids/float16 logprobs, got "
                f"{anchor_ids.dtype}/{anchor_lp.dtype}"
            )
        if not torch.isfinite(anchor_lp).all():
            raise ValueError(f"{context}: Student anchor logprobs contain NaN/Inf")
        merged = torch.sort(torch.cat((positions_i32, anchor_positions))).values
        if not torch.equal(merged, all_trainable):
            raise ValueError(
                f"{context}: Teacher and anchor positions do not partition loss_mask"
            )
        if torch.unique(merged).numel() != merged.numel():
            raise ValueError(f"{context}: Teacher and anchor positions overlap")
        rollout_ids = torch.as_tensor(sample.student_topk_ids)
        rollout_lp = torch.as_tensor(sample.student_topk_logprobs)
        if tuple(rollout_ids.shape) != (n, dynamic_topk):
            raise ValueError(f"{context}: Teacher-row Student snapshot shape mismatch")
        if tuple(rollout_lp.shape) != tuple(rollout_ids.shape):
            raise ValueError(f"{context}: Teacher-row Student logprobs shape mismatch")
    elif not torch.equal(positions_i32, all_trainable):
        raise ValueError(f"{context}: dynamic positions do not match loss_mask")
    if tuple(ids.shape) != (n, cache_topk) or tuple(lp.shape) != (n, cache_topk):
        raise ValueError(f"{context}: invalid dynamic candidate cache shape")
    if tuple(teacher_head.shape) != (n, dynamic_topk):
        raise ValueError(f"{context}: invalid teacher head shape")
    if tuple(cutoff.shape) != (n,) or tuple(mass.shape) != (n,):
        raise ValueError(f"{context}: invalid dynamic scalar-column shape")
    if ids.dtype != torch.int32 or lp.dtype != torch.float16:
        raise ValueError(
            f"{context}: expected int32 ids/float16 logprobs, got {ids.dtype}/{lp.dtype}"
        )
    if ids.numel() and not torch.all(ids[:, 1:] >= ids[:, :-1]):
        raise ValueError(f"{context}: candidate ids are not token-id sorted")
    if not torch.isfinite(lp).all() or not torch.isfinite(cutoff).all() or not torch.isfinite(mass).all():
        raise ValueError(f"{context}: dynamic teacher cache contains NaN/Inf")
    if (mass < 0).any() or (mass > 1.001).any():
        raise ValueError(f"{context}: invalid teacher cache mass")


def validate_cross_tokenizer_anchor_scored_sample(
    sample: Sample,
    cache_topk: int,
    dynamic_topk: int,
    *,
    context: str = "sample",
) -> None:
    """Require the isolated exact-Teacher/fallback-Student partition schema."""
    if not sample.offline_opd_cross_tokenizer_anchor_sparse:
        raise ValueError(
            f"{context}: cross-tokenizer Student-anchor artifact marker is missing"
        )
    if not (sample.metadata or {}).get(
        "offline_opd_cross_tokenizer_anchor_sparse", False
    ):
        raise ValueError(f"{context}: cross-tokenizer anchor metadata is missing")
    validate_dynamic_scored_sample(
        sample, cache_topk, dynamic_topk, context=context
    )


def truncate_dynamic_scored_sample_support(
    sample: Sample,
    target_topk: int,
    *,
    context: str = "sample",
) -> int:
    """Narrow a scored dynamic artifact's support-only tensors in memory.

    The large Teacher candidate cache is intentionally left unchanged.  This
    permits a Phase-3 experiment with K smaller than the K used by Phase 1/2
    without rewriting the immutable scored dataset.  Every probability-ranked
    Student/Teacher snapshot is truncated to the same prefix.

    Returns the source support width.  A return value equal to ``target_topk``
    means the artifact already had the requested width.
    """
    if not sample.offline_opd_dynamic_teacher_sparse:
        raise ValueError(f"{context}: dynamic teacher cache is not marked sparse")
    if target_topk <= 0:
        raise ValueError(f"{context}: target support Top-K must be positive")

    fields = (
        "teacher_top128_ids",
        "student_topk_ids",
        "student_topk_logprobs",
        "student_anchor_topk_ids",
        "student_anchor_topk_logprobs",
    )
    widths: dict[str, int] = {}
    for name in fields:
        value = getattr(sample, name, None)
        if value is None:
            continue
        if not isinstance(value, (np.ndarray, torch.Tensor)) or value.ndim != 2:
            raise ValueError(
                f"{context}: {name} must be a 2D numpy array or tensor"
            )
        widths[name] = int(value.shape[1])
    if "teacher_top128_ids" not in widths:
        raise ValueError(f"{context}: teacher_top128_ids is missing")
    source_topk = widths["teacher_top128_ids"]
    inconsistent = {name: width for name, width in widths.items() if width != source_topk}
    if inconsistent:
        raise ValueError(
            f"{context}: support widths disagree with teacher head K={source_topk}: "
            f"{inconsistent}"
        )
    if target_topk > source_topk:
        raise ValueError(
            f"{context}: cannot expand saved support Top-{source_topk} "
            f"to Top-{target_topk}"
        )
    if target_topk == source_topk:
        return source_topk

    for name in widths:
        value = getattr(sample, name)
        if isinstance(value, np.ndarray):
            narrowed = np.array(value[:, :target_topk], copy=True, order="C")
        else:
            narrowed = value[:, :target_topk].contiguous()
        setattr(sample, name, narrowed)
    sample.metadata = dict(sample.metadata or {})
    sample.metadata.update(
        {
            "offline_opd_runtime_support_source_topk": source_topk,
            "offline_opd_runtime_support_topk": target_topk,
        }
    )
    return source_topk


def validate_token_scored_sample(
    sample: Sample,
    *,
    context: str = "sample",
    allow_topk_diagnostics: bool = False,
) -> None:
    """Validate a sparse Phase-2 sampled-token teacher artifact."""
    if not isinstance(sample.reward, Real):
        raise ValueError(f"{context}: scored reward must be scalar")
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        raise ValueError(f"{context}: scored loss_mask is not response-aligned")
    if sample.rollout_log_probs is None or len(sample.rollout_log_probs) != sample.response_length:
        raise ValueError(f"{context}: rollout_log_probs is not response-aligned")
    if not sample.offline_opd_token_teacher_sparse:
        raise ValueError(f"{context}: token teacher data is not marked sparse")
    if (
        not allow_topk_diagnostics
        and (sample.student_support_ids is not None or sample.teacher_logprobs_on_student is not None)
    ):
        raise ValueError(f"{context}: sampled-token artifact contains unused Top-K teacher tensors")
    teacher_lp = torch.as_tensor(sample.teacher_log_probs, dtype=torch.float32).reshape(-1)
    expected = int(np.count_nonzero(np.asarray(sample.loss_mask, dtype=bool)))
    if teacher_lp.numel() != expected:
        raise ValueError(
            f"{context}: sparse teacher_log_probs length {teacher_lp.numel()} != {expected}"
        )
    if not torch.isfinite(teacher_lp).all():
        raise ValueError(f"{context}: sparse teacher_log_probs contains NaN/Inf")


def validate_baked_sample(
    sample: Sample,
    k: int,
    *,
    context: str = "sample",
    require_sparse_teacher: bool = False,
) -> None:
    teacher_lp = sample.teacher_logprobs_on_student
    support_ids = sample.student_support_ids
    if not isinstance(teacher_lp, torch.Tensor) or teacher_lp.dtype != torch.float32:
        raise ValueError(
            f"{context}: teacher_logprobs_on_student must be a float32 tensor; "
            f"got type={type(teacher_lp).__name__}, dtype={getattr(teacher_lp, 'dtype', None)}"
        )
    if not torch.isfinite(teacher_lp).all():
        raise ValueError(f"{context}: teacher_logprobs_on_student contains NaN/Inf")

    positions = np.flatnonzero(np.asarray(sample.loss_mask, dtype=bool))
    expected_ids = torch.from_numpy(student_topk_array(sample)).long()
    sparse_shape = (len(positions), k)
    if sample.offline_opd_teacher_sparse:
        if support_ids is not None:
            raise ValueError(f"{context}: sparse artifact unexpectedly contains student_support_ids")
        if sample.rollout_student_logprobs_on_support is not None:
            raise ValueError(
                f"{context}: sparse artifact unexpectedly contains expanded rollout-student data"
            )
        if tuple(teacher_lp.shape) != sparse_shape:
            raise ValueError(
                f"{context}: sparse teacher shape {tuple(teacher_lp.shape)} != {sparse_shape}"
            )
        if sample.opd_topk_overlap is not None and torch.as_tensor(
            sample.opd_topk_overlap
        ).numel() != len(positions):
            raise ValueError(f"{context}: sparse opd_topk_overlap is not trainable-row aligned")
        return

    if require_sparse_teacher:
        raise ValueError(f"{context}: scored artifact is not marked as sparse teacher data")
    if not isinstance(support_ids, torch.Tensor) or support_ids.dtype != torch.long:
        raise ValueError(f"{context}: student_support_ids must be an int64 tensor")
    expected_shape = (sample.response_length, k)
    if tuple(teacher_lp.shape) != expected_shape or tuple(support_ids.shape) != expected_shape:
        raise ValueError(
            f"{context}: baked tensor shapes {tuple(teacher_lp.shape)}, "
            f"{tuple(support_ids.shape)} != {expected_shape}"
        )
    actual_ids = support_ids[torch.from_numpy(positions).long()]
    if not torch.equal(actual_ids, expected_ids):
        raise ValueError(f"{context}: baked support ids no longer match Phase-1 student_topk_ids")


def compact_baked_teacher_for_disk(
    sample: Sample, k: int, *, context: str = "sample"
) -> None:
    """Drop masked rows and the duplicate support-id matrix before Phase-2 save."""
    validate_baked_sample(sample, k, context=context)
    positions = torch.from_numpy(
        np.flatnonzero(np.asarray(sample.loss_mask, dtype=bool))
    ).long()
    sample.teacher_logprobs_on_student = sample.teacher_logprobs_on_student[
        positions
    ].contiguous()
    sample.student_support_ids = None
    sample.offline_opd_teacher_sparse = True
    sample.rollout_student_logprobs_on_support = None
    if sample.opd_topk_overlap is not None:
        overlap = torch.as_tensor(sample.opd_topk_overlap, dtype=torch.float32).reshape(-1)
        if overlap.numel() != sample.response_length:
            raise ValueError(f"{context}: dense opd_topk_overlap is not response-aligned")
        sample.opd_topk_overlap = overlap[positions].contiguous()


def validate_scored_sample(
    sample: Sample,
    k: int,
    *,
    context: str = "sample",
    require_student_topk_logprobs: bool = False,
    require_sparse_teacher: bool = False,
) -> None:
    if not isinstance(sample.reward, Real):
        raise ValueError(f"{context}: scored reward must be scalar")
    if sample.loss_mask is None or len(sample.loss_mask) != sample.response_length:
        raise ValueError(f"{context}: scored loss_mask is not response-aligned")
    _validate_student_topk_snapshot(
        sample,
        k,
        context=context,
        require_student_topk_logprobs=require_student_topk_logprobs,
    )
    validate_baked_sample(
        sample,
        k,
        context=context,
        require_sparse_teacher=require_sparse_teacher,
    )


def validate_rollout_directory(
    directory: str | Path,
    k: int,
    *,
    require_teacher: bool,
    expected_samples: int | None = None,
    require_unique_samples: bool = True,
    require_complete_index_set: bool = True,
    dataset_size: int | None = None,
    require_student_topk_logprobs: bool = False,
    require_sparse_teacher: bool = False,
    opd_loss_type: str = "topk_reverse_kl",
    token_topk_diagnostics: bool = False,
    dynamic_support_topk: int = 128,
    teacher_cache_topk: int = 4096,
    truncate_dynamic_support: bool = False,
) -> RolloutDirectorySummary:
    files = discover_rollout_files(directory)
    seen_indices: set[int] = set()
    seen_instances: set[str] = set()
    samples_per_file: int | None = None
    total = 0

    for path in files:
        samples, _ = load_rollout_file(path)
        if samples_per_file is None:
            samples_per_file = len(samples)
        elif len(samples) != samples_per_file:
            raise ValueError(f"{path}: {len(samples)} samples != first file's {samples_per_file}")

        for offset, sample in enumerate(samples):
            context = f"{path.name}:sample[{offset}]"
            if truncate_dynamic_support and require_teacher:
                truncate_dynamic_scored_sample_support(
                    sample, dynamic_support_topk, context=context
                )
            if opd_loss_type == "token_reverse_kl" and require_teacher:
                validate_token_scored_sample(
                    sample,
                    context=context,
                    allow_topk_diagnostics=token_topk_diagnostics,
                )
                if token_topk_diagnostics:
                    validate_scored_sample(
                        sample,
                        k,
                        context=context,
                        require_student_topk_logprobs=True,
                        require_sparse_teacher=True,
                    )
            elif opd_loss_type == "token_reverse_kl":
                validate_token_dump_sample(sample, context=context)
            elif opd_loss_type in {
                "cached_teacher_dynamic_topk_reverse_kl",
                "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor",
                "cached_teacher_sampled_token_pg_with_student_anchor",
                "cached_teacher_topk_forward_kl",
                "cached_teacher_bidirectional_topk_kl",
                "cached_teacher_bidirectional_topk_kl_with_student_anchor",
                "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor",
            } and require_teacher:
                if opd_loss_type in {
                    "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor",
                    "cached_teacher_sampled_token_pg_with_student_anchor",
                    "cached_teacher_bidirectional_topk_kl_with_student_anchor",
                    "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor",
                }:
                    validate_cross_tokenizer_anchor_scored_sample(
                        sample,
                        teacher_cache_topk,
                        dynamic_support_topk,
                        context=context,
                    )
                else:
                    validate_dynamic_scored_sample(
                        sample,
                        teacher_cache_topk,
                        dynamic_support_topk,
                        context=context,
                    )
                    _validate_student_topk_snapshot(
                        sample,
                        dynamic_support_topk,
                        context=context,
                        require_student_topk_logprobs=require_student_topk_logprobs,
                    )
            elif opd_loss_type in {
                "cached_teacher_dynamic_topk_reverse_kl",
                "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor",
                "cached_teacher_sampled_token_pg_with_student_anchor",
                "cached_teacher_topk_forward_kl",
                "cached_teacher_bidirectional_topk_kl",
                "cached_teacher_bidirectional_topk_kl_with_student_anchor",
                "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor",
            }:
                validate_dump_sample(
                    sample,
                    dynamic_support_topk,
                    context=context,
                    require_student_topk_logprobs=require_student_topk_logprobs,
                )
            elif require_teacher:
                validate_scored_sample(
                    sample,
                    k,
                    context=context,
                    require_student_topk_logprobs=require_student_topk_logprobs,
                    require_sparse_teacher=require_sparse_teacher,
                )
            else:
                validate_dump_sample(
                    sample,
                    k,
                    context=context,
                    require_student_topk_logprobs=require_student_topk_logprobs,
                )

            if require_unique_samples and sample.index in seen_indices:
                raise ValueError(f"{context}: duplicate sample.index={sample.index}")
            if sample.index is not None:
                if dataset_size is not None and not 0 <= sample.index < dataset_size:
                    raise ValueError(
                        f"{context}: sample.index={sample.index} is outside dataset size {dataset_size}"
                    )
                seen_indices.add(sample.index)
            instance_id = (sample.metadata or {}).get("instance_id")
            if instance_id is not None:
                if require_unique_samples and instance_id in seen_instances:
                    raise ValueError(f"{context}: duplicate instance_id={instance_id}")
                seen_instances.add(instance_id)
            total += 1

    if expected_samples is not None and total != expected_samples:
        raise ValueError(f"{directory}: found {total} samples, expected {expected_samples}")
    if require_unique_samples and len(seen_indices) != total:
        raise ValueError(f"{directory}: one or more samples have no integer sample.index")
    if require_unique_samples and require_complete_index_set and seen_indices != set(range(total)):
        raise ValueError(f"{directory}: sample.index values are not exactly 0..{total - 1}")

    assert samples_per_file is not None
    return RolloutDirectorySummary(len(files), total, samples_per_file)
