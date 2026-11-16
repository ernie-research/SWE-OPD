#!/usr/bin/env python3
"""Globally shuffle Phase-1 offline-OPD samples and re-shard them safely.

Phase 1 writes trajectories in completion order, which correlates file position
with response length.  This tool loads the pre-teacher Phase-1 artifacts, applies
a deterministic sample-level permutation with optional length balancing, and
writes equally sized rollout files for Phase 2 and Phase 3.

The output directory is published atomically and contains a manifest with the
source fingerprint and exact permutation.  Re-running with the same inputs and
seed is therefore a cheap no-op; an incompatible existing output is never
silently replaced.
"""

from __future__ import annotations

import argparse
import heapq
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import torch

from examples.coding_agent_opd.scripts.offline_opd.offline_utils import discover_rollout_files


MANIFEST_NAME = ".shuffle_manifest.json"
MANIFEST_VERSION = 2


def _source_snapshot(paths: list[Path]) -> list[dict[str, Any]]:
    snapshot = []
    for path in paths:
        stat = path.stat()
        snapshot.append(
            {
                "name": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return snapshot


def _load_manifest(path: Path) -> dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as stream:
            value = json.load(stream)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _output_is_reusable(
    output_dir: Path,
    *,
    seed: int,
    source: list[dict[str, Any]],
    num_files: int,
    samples_per_file: int,
    balance_key: str,
) -> bool:
    manifest = _load_manifest(output_dir / MANIFEST_NAME)
    if manifest is None:
        return False
    if (
        manifest.get("version") != MANIFEST_VERSION
        or manifest.get("seed") != seed
        or manifest.get("source_files") != source
        or manifest.get("num_output_files") != num_files
        or manifest.get("samples_per_file") != samples_per_file
        or manifest.get("balance_key") != balance_key
    ):
        return False
    expected = {f"rollout_{index}.pt" for index in range(num_files)}
    actual = {path.name for path in output_dir.glob("rollout_*.pt") if path.is_file()}
    return actual == expected and all((output_dir / name).stat().st_size > 0 for name in expected)


def _sample_index(sample: Any, context: str) -> int:
    if not isinstance(sample, dict):
        raise ValueError(f"{context}: expected a serialized Sample dict, got {type(sample).__name__}")
    index = sample.get("index")
    if not isinstance(index, int):
        raise ValueError(f"{context}: sample.index must be an integer, got {index!r}")
    return index


def _sample_cost(sample: dict[str, Any], balance_key: str, context: str) -> int:
    if balance_key == "none":
        return 1
    if balance_key == "total_tokens":
        tokens = sample.get("tokens")
        if not hasattr(tokens, "__len__"):
            raise ValueError(f"{context}: tokens is missing or has no length")
        return max(int(len(tokens)), 1)

    # The sparse top-k rows are exactly the trainable response positions.  This
    # is both the dominant Phase-1/2 storage term and the expensive K-wide
    # teacher gather/loss term, so it is a better balancing signal than raw
    # completion order or response_length.
    topk_ids = sample.get("student_topk_ids")
    shape = getattr(topk_ids, "shape", None)
    if shape is not None and len(shape) == 2:
        return max(int(shape[0]), 1)
    loss_mask = sample.get("loss_mask")
    if loss_mask is None:
        raise ValueError(f"{context}: cannot determine trainable-token cost")
    return max(sum(bool(value) for value in loss_mask), 1)


def _balanced_batches(
    samples: list[dict[str, Any]],
    costs: list[int],
    *,
    samples_per_file: int,
    seed: int,
    balance_key: str,
) -> tuple[list[list[dict[str, Any]]], list[list[int]]]:
    """Return deterministic randomized batches with equal sample counts.

    ``balance_key=none`` is a uniform seeded permutation followed by chunking.
    Other modes randomize ties, greedily equalize total cost under a strict
    capacity, then randomize both within-batch order and batch order.  The
    latter removes completion-order length clustering without allowing a few
    long trajectories to dominate one teacher/training file.
    """

    total = len(samples)
    if samples_per_file <= 0:
        raise ValueError(f"samples_per_file must be positive, got {samples_per_file}")
    if total % samples_per_file != 0:
        raise ValueError(
            f"{total} samples are not divisible by samples_per_file={samples_per_file}; "
            "offline training currently requires uniform full global batches"
        )

    rng = random.Random(seed)
    order = list(range(total))
    rng.shuffle(order)
    num_batches = total // samples_per_file
    if balance_key == "none":
        index_batches = [
            order[start : start + samples_per_file]
            for start in range(0, total, samples_per_file)
        ]
    else:
        # Stable sort preserves the seeded random order for equal-cost samples.
        order.sort(key=lambda index: costs[index], reverse=True)
        tie_order = list(range(num_batches))
        rng.shuffle(tie_order)
        tie_rank = {batch_index: rank for rank, batch_index in enumerate(tie_order)}
        index_batches: list[list[int]] = [[] for _ in range(num_batches)]
        heap = [(0, 0, tie_rank[index], index) for index in range(num_batches)]
        heapq.heapify(heap)
        for sample_index in order:
            load, count, tie, batch_index = heapq.heappop(heap)
            index_batches[batch_index].append(sample_index)
            count += 1
            load += costs[sample_index]
            if count < samples_per_file:
                heapq.heappush(heap, (load, count, tie, batch_index))

        for batch in index_batches:
            rng.shuffle(batch)
        rng.shuffle(index_batches)

    return (
        [[samples[index] for index in batch] for batch in index_batches],
        index_batches,
    )


def shuffle_phase1(
    input_dir: Path,
    output_dir: Path,
    seed: int,
    *,
    samples_per_file: int | None = None,
    balance_key: str = "trainable_tokens",
) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if input_dir == output_dir:
        raise ValueError("input and output directories must differ")

    paths = discover_rollout_files(input_dir)
    source = _source_snapshot(paths)
    if output_dir.exists():
        manifest = _load_manifest(output_dir / MANIFEST_NAME)
        manifest_samples_per_file = (
            manifest.get("samples_per_file") if manifest is not None else None
        )
        manifest_num_files = (
            manifest.get("num_output_files") if manifest is not None else None
        )
        requested_samples_per_file = (
            samples_per_file
            if samples_per_file is not None
            else manifest_samples_per_file
        )
        if (
            isinstance(requested_samples_per_file, int)
            and isinstance(manifest_num_files, int)
            and _output_is_reusable(
                output_dir,
                seed=seed,
                source=source,
                num_files=manifest_num_files,
                samples_per_file=requested_samples_per_file,
                balance_key=balance_key,
            )
        ):
            print(
                f"[phase1-shuffle] reuse {output_dir}: "
                f"samples={manifest['num_samples']}, files={manifest_num_files}, seed={seed}, "
                f"balance={balance_key}"
            )
            return
        raise FileExistsError(
            f"{output_dir} exists but does not match the current Phase-1 inputs/seed/layout; "
            "choose a new PHASE1_SHUFFLED_DIR or remove the stale generated directory"
        )

    samples: list[dict[str, Any]] = []
    origins: list[dict[str, int]] = []
    source_samples_per_file: int | None = None
    seen_indices: set[int] = set()
    for file_index, path in enumerate(paths):
        blob = torch.load(path, weights_only=False)
        file_samples = blob.get("samples") if isinstance(blob, dict) else None
        if not isinstance(file_samples, list) or not file_samples:
            raise ValueError(f"{path}: expected a non-empty {{'samples': [...]}} dump")
        if source_samples_per_file is None:
            source_samples_per_file = len(file_samples)
        for offset, sample in enumerate(file_samples):
            index = _sample_index(sample, f"{path.name}:sample[{offset}]")
            if index in seen_indices:
                raise ValueError(f"{path.name}:sample[{offset}]: duplicate sample.index={index}")
            seen_indices.add(index)
            samples.append(sample)
            origins.append({"file": file_index, "offset": offset, "sample_index": index})
        print(f"[phase1-shuffle] loaded {path.name}: samples={len(file_samples)}")

    assert source_samples_per_file is not None
    if seen_indices != set(range(len(samples))):
        raise ValueError(f"sample.index values are not exactly 0..{len(samples) - 1}")

    output_samples_per_file = (
        samples_per_file
        if samples_per_file is not None
        else source_samples_per_file
    )
    if balance_key not in {"none", "trainable_tokens", "total_tokens"}:
        raise ValueError(f"unsupported balance_key={balance_key!r}")
    costs = [
        _sample_cost(sample, balance_key, f"sample[{offset}]")
        for offset, sample in enumerate(samples)
    ]
    sample_batches, index_batches = _balanced_batches(
        samples,
        costs,
        samples_per_file=output_samples_per_file,
        seed=seed,
        balance_key=balance_key,
    )
    num_output_files = len(sample_batches)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.tmp.{os.getpid()}"
    staging.mkdir()
    try:
        batch_costs = []
        for output_index, (batch, batch_indices) in enumerate(
            zip(sample_batches, index_batches, strict=True)
        ):
            out_path = staging / f"rollout_{output_index}.pt"
            torch.save(
                {
                    "rollout_id": output_index,
                    "samples": batch,
                },
                out_path,
            )
            batch_cost = sum(costs[index] for index in batch_indices)
            batch_costs.append(batch_cost)
            print(
                f"[phase1-shuffle] wrote {out_path.name}: "
                f"samples={len(batch)}, cost={batch_cost}, bytes={out_path.stat().st_size}"
            )

        shuffled_origins = [
            origins[index]
            for batch_indices in index_batches
            for index in batch_indices
        ]
        manifest = {
            "version": MANIFEST_VERSION,
            "seed": seed,
            "input_dir": str(input_dir),
            "num_samples": len(samples),
            "samples_per_file": output_samples_per_file,
            "num_output_files": num_output_files,
            "balance_key": balance_key,
            "batch_costs": batch_costs,
            "source_files": source,
            "permutation": shuffled_origins,
        }
        manifest_path = staging / MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(staging, output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(
        f"[phase1-shuffle] complete: {len(samples)} samples -> {num_output_files} files "
        f"x {output_samples_per_file}, seed={seed}, balance={balance_key}, output={output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=None,
        help="final Phase-3 samples per file; defaults to the first source file size",
    )
    parser.add_argument(
        "--balance-key",
        choices=("none", "trainable_tokens", "total_tokens"),
        default="trainable_tokens",
        help="seeded randomization with optional equal-cost batch assignment",
    )
    args = parser.parse_args()
    shuffle_phase1(
        args.input_dir,
        args.output_dir,
        args.seed,
        samples_per_file=args.samples_per_file,
        balance_key=args.balance_key,
    )


if __name__ == "__main__":
    main()
