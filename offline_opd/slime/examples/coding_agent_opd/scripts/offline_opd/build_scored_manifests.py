#!/usr/bin/env python3
"""Build Phase-3 batch manifests without rewriting scored samples.

Phase 2 writes each scored sample once under ``samples/`` and publishes one
small source manifest after each original rollout file is complete.  This tool
reads only those JSON manifests and writes tiny ``rollout_N.pt`` manifests
consumed by Phase 3.  It can either apply deterministic length balancing or
preserve the original Phase-1 batch grouping/order exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch


SOURCE_RE = re.compile(r"rollout_(\d+)\.json$")
FORMAT = "offline_opd_sample_manifest_v1"
SUMMARY_NAME = ".sample_manifest_summary.json"


def _load_source_batches(
    output_dir: Path, *, allow_noncontiguous_sample_indices: bool = False
) -> list[list[dict[str, Any]]]:
    source_dir = output_dir / "source_manifests"
    indexed: list[tuple[int, Path]] = []
    for path in source_dir.glob("rollout_*.json"):
        match = SOURCE_RE.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if not indexed or [i for i, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"source manifests are missing or non-contiguous under {source_dir}")

    batches: list[list[dict[str, Any]]] = []
    seen: set[int] = set()
    for source_id, path in indexed:
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("format") != "offline_opd_scored_source_v1":
            raise ValueError(f"{path}: unsupported source manifest format")
        if value.get("source_rollout_id") != source_id:
            raise ValueError(f"{path}: source_rollout_id mismatch")
        source_rows = value.get("samples")
        if not isinstance(source_rows, list) or not source_rows:
            raise ValueError(f"{path}: samples must be a non-empty list")
        validated_rows: list[dict[str, Any]] = []
        for row in source_rows:
            index = row.get("index") if isinstance(row, dict) else None
            relative_path = row.get("path") if isinstance(row, dict) else None
            cost = row.get("trainable_tokens") if isinstance(row, dict) else None
            if not isinstance(index, int) or index < 0 or index in seen:
                raise ValueError(f"{path}: invalid or duplicate sample index {index!r}")
            if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
                raise ValueError(f"{path}: sample path must be relative: {relative_path!r}")
            if not isinstance(cost, int) or cost <= 0:
                raise ValueError(f"{path}: invalid trainable_tokens={cost!r}")
            sample_path = output_dir / relative_path
            if not sample_path.is_file() or sample_path.stat().st_size <= 0:
                raise ValueError(f"{path}: scored sample is missing or empty: {sample_path}")
            seen.add(index)
            validated_rows.append(
                {"index": index, "path": relative_path, "trainable_tokens": cost}
            )
        batches.append(validated_rows)

    num_rows = sum(len(batch) for batch in batches)
    if not allow_noncontiguous_sample_indices and seen != set(range(num_rows)):
        raise ValueError(f"sample indices are not exactly 0..{num_rows - 1}")
    return batches


def _load_source_rows(output_dir: Path) -> list[dict[str, Any]]:
    return [row for batch in _load_source_batches(output_dir) for row in batch]


def _balanced_rows(
    rows: list[dict[str, Any]], samples_per_file: int, seed: int
) -> list[list[dict[str, Any]]]:
    if samples_per_file <= 0 or len(rows) % samples_per_file:
        raise ValueError(
            f"{len(rows)} samples are not divisible by samples_per_file={samples_per_file}"
        )
    rng = random.Random(seed)
    order = list(rows)
    rng.shuffle(order)
    # Split the global length ranking into ``samples_per_file`` adjacent strata,
    # each containing exactly one candidate per output batch.  Assign one row
    # from every stratum to every batch, pairing the longer rows in that stratum
    # with the currently lighter batches.  This balances total effective tokens
    # without allowing equal-token batches to have very different length mixes
    # (for example long+short versus all-medium).
    order.sort(key=lambda row: row["trainable_tokens"], reverse=True)
    count = len(rows) // samples_per_file
    batches: list[list[dict[str, Any]]] = [[] for _ in range(count)]
    loads = [0] * count
    for start in range(0, len(order), count):
        stratum = order[start : start + count]
        tie_order = list(range(count))
        rng.shuffle(tie_order)
        tie_rank = {batch: rank for rank, batch in enumerate(tie_order)}
        target_batches = sorted(range(count), key=lambda batch: (loads[batch], tie_rank[batch]))
        # ``order`` is descending, so the largest row in the stratum goes to the
        # lightest current batch.
        for row, batch in zip(stratum, target_batches, strict=True):
            batches[batch].append(row)
            loads[batch] += row["trainable_tokens"]

    if any(len(batch) != samples_per_file for batch in batches):
        raise AssertionError("stratified assignment produced a non-uniform batch")
    for batch in batches:
        rng.shuffle(batch)
    rng.shuffle(batches)
    return batches


def build_manifests(
    output_dir: Path,
    samples_per_file: int,
    seed: int,
    balance_strategy: str = "length_stratified_greedy",
    allow_noncontiguous_sample_indices: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    source_batches = _load_source_batches(
        output_dir,
        allow_noncontiguous_sample_indices=allow_noncontiguous_sample_indices,
    )
    rows = [row for batch in source_batches for row in batch]
    if balance_strategy == "length_stratified_greedy":
        batches = _balanced_rows(rows, samples_per_file, seed)
    elif balance_strategy == "phase1":
        invalid_sizes = [len(batch) for batch in source_batches if len(batch) != samples_per_file]
        if invalid_sizes:
            raise ValueError(
                "Phase-1 batch layout cannot be preserved with "
                f"samples_per_file={samples_per_file}; source batch sizes include "
                f"{sorted(set(invalid_sizes))}"
            )
        # Keep both source-manifest order and row order exactly as Phase 1 wrote
        # them.  Only the referenced artifact changes from Phase-1 rollout data
        # to the corresponding Phase-2 scored sample.
        batches = source_batches
    else:
        raise ValueError(f"unsupported balance_strategy={balance_strategy!r}")
    batch_costs: list[int] = []
    for rollout_id, batch in enumerate(batches):
        path = output_dir / f"rollout_{rollout_id}.pt"
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        cost = sum(row["trainable_tokens"] for row in batch)
        torch.save(
            {
                "format": FORMAT,
                "rollout_id": rollout_id,
                "samples_per_file": samples_per_file,
                "sample_files": [row["path"] for row in batch],
                "sample_indices": [row["index"] for row in batch],
                "trainable_tokens": cost,
            },
            temporary,
        )
        os.replace(temporary, path)
        batch_costs.append(cost)

    summary = {
        "format": FORMAT,
        "balance_strategy": balance_strategy,
        "seed": seed,
        "num_samples": len(rows),
        "samples_per_file": samples_per_file,
        "num_rollouts": len(batches),
        "allow_noncontiguous_sample_indices": allow_noncontiguous_sample_indices,
        "batch_trainable_tokens": batch_costs,
    }
    temporary_summary = output_dir / f".{SUMMARY_NAME}.tmp.{os.getpid()}"
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_summary, output_dir / SUMMARY_NAME)
    print(
        f"[phase2-manifest] {len(rows)} scored samples -> {len(batches)} manifests "
        f"x {samples_per_file}, strategy={balance_strategy}, seed={seed}, "
        f"cost_range={min(batch_costs)}..{max(batch_costs)}"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples-per-file", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance-strategy",
        choices=("length_stratified_greedy", "phase1"),
        default="length_stratified_greedy",
        help="Rebalance by trainable length, or preserve Phase-1 batch grouping/order.",
    )
    parser.add_argument(
        "--allow-noncontiguous-sample-indices",
        action="store_true",
        help="Allow an intentional subset whose original global sample ids have gaps.",
    )
    args = parser.parse_args()
    build_manifests(
        args.output_dir,
        args.samples_per_file,
        args.seed,
        balance_strategy=args.balance_strategy,
        allow_noncontiguous_sample_indices=args.allow_noncontiguous_sample_indices,
    )


if __name__ == "__main__":
    main()
