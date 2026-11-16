#!/usr/bin/env python3
"""Rebatch scored offline-OPD samples as one continuous multi-epoch stream.

Phase 2 stores every scored trajectory once and writes small batch manifests.
This tool reuses those sample artifacts: it concatenates the requested epochs,
then cuts the stream into fixed-size Phase-3 batches.  Consequently, an epoch
boundary inside a batch is filled by the beginning of the next epoch.  Only the
very last batch may consume samples from one additional epoch to stay full.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Any

import torch


ROLLOUT_RE = re.compile(r"rollout_(\d+)\.pt$")
SOURCE_FORMAT = "offline_opd_sample_manifest_v1"
OUTPUT_FORMAT = "offline_opd_sample_manifest_v2"
OWNER_MARKER = ".offline_opd_rebatch_v1.json"


def _discover(directory: Path) -> list[Path]:
    indexed: list[tuple[int, Path]] = []
    for path in directory.glob("rollout_*.pt"):
        match = ROLLOUT_RE.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if not indexed or [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"rollout files are missing or non-contiguous under {directory}")
    return [path for _, path in indexed]


def _load_costs(scored_dir: Path) -> dict[int, int]:
    costs: dict[int, int] = {}
    source_dir = scored_dir / "source_manifests"
    if not source_dir.is_dir():
        return costs
    for path in source_dir.glob("rollout_*.json"):
        value = json.loads(path.read_text(encoding="utf-8"))
        for row in value.get("samples", []):
            index = row.get("index") if isinstance(row, dict) else None
            cost = row.get("trainable_tokens") if isinstance(row, dict) else None
            if isinstance(index, int) and isinstance(cost, int) and cost > 0:
                costs[index] = cost
    return costs


def _load_source_batches(
    scored_dir: Path,
    *,
    allow_noncontiguous_sample_indices: bool = False,
    allow_external_sample_symlinks: bool = False,
) -> tuple[Path, list[list[dict[str, Any]]]]:
    batches: list[list[dict[str, Any]]] = []
    sample_root: Path | None = None
    seen: set[int] = set()
    costs = _load_costs(scored_dir)

    for path in _discover(scored_dir):
        blob = torch.load(path, weights_only=False)
        if not isinstance(blob, dict) or blob.get("format") != SOURCE_FORMAT:
            raise ValueError(
                f"{path}: cross-epoch rebatching requires Phase-2 sample manifests "
                f"({SOURCE_FORMAT}); legacy bundled rollout files are unsupported"
            )
        files = blob.get("sample_files")
        indices = blob.get("sample_indices")
        if not isinstance(files, list) or not files or not isinstance(indices, list):
            raise ValueError(f"{path}: invalid sample manifest")
        if len(files) != len(indices):
            raise ValueError(f"{path}: sample_files/sample_indices length mismatch")

        root = path.resolve().parent
        if sample_root is None:
            sample_root = root
        elif root != sample_root:
            raise ValueError(f"source manifests resolve to multiple sample roots: {sample_root}, {root}")

        batch: list[dict[str, Any]] = []
        for relative, index in zip(files, indices, strict=True):
            if not isinstance(relative, str) or Path(relative).is_absolute():
                raise ValueError(f"{path}: invalid relative sample path {relative!r}")
            if not isinstance(index, int) or index < 0 or index in seen:
                raise ValueError(f"{path}: invalid or duplicate sample index {index!r}")
            # Keep the logical path rooted under ``scored_dir`` in the emitted
            # v2 manifest.  Some continuation experiments intentionally reuse
            # immutable sample artifacts through symlinks; resolving the link
            # here would both reject that layout and bake the old experiment
            # directory into every output manifest.
            logical_sample_path = Path(os.path.abspath(root / relative))
            try:
                normalized = logical_sample_path.relative_to(root)
            except ValueError as error:
                raise ValueError(f"{path}: sample escapes source root: {relative}") from error
            sample_path = logical_sample_path.resolve()
            if not allow_external_sample_symlinks:
                try:
                    sample_path.relative_to(root)
                except ValueError as error:
                    raise ValueError(f"{path}: sample escapes source root: {relative}") from error
            if not logical_sample_path.is_file():
                raise ValueError(f"{path}: missing scored sample: {logical_sample_path}")
            seen.add(index)
            batch.append(
                {
                    "index": index,
                    "path": normalized.as_posix(),
                    "trainable_tokens": costs.get(index),
                }
            )
        batches.append(batch)

    assert sample_root is not None
    num_samples = sum(map(len, batches))
    if not allow_noncontiguous_sample_indices and seen != set(range(num_samples)):
        raise ValueError(f"sample indices are not exactly 0..{num_samples - 1}")
    return sample_root, batches


def _epoch_rows(
    source_batches: list[list[dict[str, Any]]], epoch: int, shuffle: bool
) -> list[dict[str, Any]]:
    batch_order = list(range(len(source_batches)))
    if shuffle:
        random.Random(epoch).shuffle(batch_order)
    return [row for batch_id in batch_order for row in source_batches[batch_id]]


def _make_boundary_unique(
    stream: list[dict[str, Any]],
    epoch_rows: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    """Keep a cross-epoch batch from containing the same trajectory twice.

    Independently shuffled epochs can end and begin with overlapping sample ids.
    If that boundary falls inside one Phase-3 batch, promote just enough safe
    rows from the new epoch to fill the batch.  The epoch remains a permutation
    of the complete dataset; only its within-epoch order changes.
    """
    previous_count = len(stream) % batch_size
    if previous_count == 0:
        return epoch_rows

    prefix_count = batch_size - previous_count
    previous_ids = {row["index"] for row in stream[-previous_count:]}
    if all(row["index"] not in previous_ids for row in epoch_rows[:prefix_count]):
        return epoch_rows

    safe_prefix = [row for row in epoch_rows if row["index"] not in previous_ids][
        :prefix_count
    ]
    if len(safe_prefix) != prefix_count:
        raise ValueError(
            "cannot construct a unique cross-epoch batch; "
            f"batch_size={batch_size}, previous_count={previous_count}, "
            f"dataset_size={len(epoch_rows)}"
        )
    promoted_ids = {row["index"] for row in safe_prefix}
    return safe_prefix + [row for row in epoch_rows if row["index"] not in promoted_ids]


def build_rebatched_epochs(
    scored_dir: Path,
    output_dir: Path,
    epochs: int,
    batch_size: int,
    shuffle: bool,
    num_rollouts: int | None = None,
    allow_noncontiguous_sample_indices: bool = False,
    allow_external_sample_symlinks: bool = False,
) -> dict[str, Any]:
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if num_rollouts is not None and num_rollouts <= 0:
        raise ValueError("num_rollouts must be positive when specified")
    scored_dir = scored_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    if scored_dir == output_dir:
        raise ValueError("output_dir must differ from scored_dir")

    sample_root, source_batches = _load_source_batches(
        scored_dir,
        allow_noncontiguous_sample_indices=allow_noncontiguous_sample_indices,
        allow_external_sample_symlinks=allow_external_sample_symlinks,
    )
    dataset_size = sum(map(len, source_batches))
    if batch_size > dataset_size:
        raise ValueError(
            f"batch_size={batch_size} exceeds dataset_size={dataset_size}; "
            "a single batch would repeat sample indices and break rollout grouping"
        )
    epoch_samples = dataset_size * epochs
    if num_rollouts is None:
        output_rollouts = math.ceil(epoch_samples / batch_size)
    else:
        output_rollouts = num_rollouts
        if output_rollouts * batch_size > epoch_samples:
            raise ValueError(
                f"num_rollouts={output_rollouts} x batch_size={batch_size} exceeds "
                f"the requested {epochs} epochs ({epoch_samples} samples)"
            )
    emitted_samples = output_rollouts * batch_size

    stream: list[dict[str, Any]] = []
    epoch = 0
    while len(stream) < emitted_samples:
        rows = _epoch_rows(source_batches, epoch, shuffle)
        rows = _make_boundary_unique(stream, rows, batch_size)
        stream.extend(rows)
        epoch += 1
    stream = stream[:emitted_samples]

    sample_root_relative = os.path.relpath(sample_root, output_dir)
    for rollout_id, start in enumerate(range(0, emitted_samples, batch_size)):
        batch = stream[start : start + batch_size]
        path = output_dir / f"rollout_{rollout_id}.pt"
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        known_costs = [row["trainable_tokens"] for row in batch]
        torch.save(
            {
                "format": OUTPUT_FORMAT,
                "rollout_id": rollout_id,
                "samples_per_file": batch_size,
                "sample_root": sample_root_relative,
                "allow_external_sample_symlinks": allow_external_sample_symlinks,
                "sample_files": [row["path"] for row in batch],
                "sample_indices": [row["index"] for row in batch],
                "trainable_tokens": (
                    sum(known_costs)
                    if all(value is not None for value in known_costs)
                    else None
                ),
            },
            temporary,
        )
        os.replace(temporary, path)

    summary = {
        "format": OUTPUT_FORMAT,
        "source_dir": str(scored_dir),
        "epochs": epochs,
        "shuffle": shuffle,
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "num_rollouts": output_rollouts,
        "epoch_samples": epoch_samples,
        "requested_samples": emitted_samples if num_rollouts is not None else epoch_samples,
        "emitted_samples": emitted_samples,
        "padding_samples": max(emitted_samples - epoch_samples, 0),
        "truncated_samples": max(epoch_samples - emitted_samples, 0),
        "cross_epoch": dataset_size % batch_size != 0,
        "allow_noncontiguous_sample_indices": allow_noncontiguous_sample_indices,
        "allow_external_sample_symlinks": allow_external_sample_symlinks,
    }
    marker = output_dir / OWNER_MARKER
    temporary_marker = marker.with_name(f".{marker.name}.tmp.{os.getpid()}")
    temporary_marker.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_marker, marker)
    print(
        f"[rebatch] dataset={dataset_size} epochs={epochs} batch={batch_size} "
        f"rollouts={output_rollouts} emitted={emitted_samples} "
        f"padding={summary['padding_samples']} truncated={summary['truncated_samples']} "
        f"cross_epoch={summary['cross_epoch']}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--num-rollouts", type=int)
    parser.add_argument("--shuffle", choices=(0, 1), default=0, type=int)
    parser.add_argument(
        "--allow-noncontiguous-sample-indices",
        action="store_true",
        help="Allow an intentional subset whose original global sample ids have gaps.",
    )
    parser.add_argument(
        "--allow-external-sample-symlinks",
        action="store_true",
        help=(
            "Allow logical sample paths under scored_dir to be symlinks whose "
            "targets are outside scored_dir. The emitted manifests retain the "
            "logical in-tree paths."
        ),
    )
    args = parser.parse_args()
    build_rebatched_epochs(
        args.scored_dir,
        args.output_dir,
        args.epochs,
        args.batch_size,
        bool(args.shuffle),
        args.num_rollouts,
        args.allow_noncontiguous_sample_indices,
        args.allow_external_sample_symlinks,
    )


if __name__ == "__main__":
    main()
