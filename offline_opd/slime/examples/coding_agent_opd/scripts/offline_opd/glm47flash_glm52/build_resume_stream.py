#!/usr/bin/env python3
"""Build a two-epoch stream that preserves an already-trained batch prefix."""

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
OWNER = ".offline_opd_partial_resume_v1.json"


def discover(directory: Path) -> list[Path]:
    indexed = []
    for path in directory.glob("rollout_*.pt"):
        if match := ROLLOUT_RE.fullmatch(path.name):
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if not indexed or [index for index, _ in indexed] != list(range(len(indexed))):
        raise ValueError(f"non-contiguous rollout files under {directory}")
    return [path for _, path in indexed]


def load_rows(directory: Path) -> tuple[list[list[int]], dict[int, dict[str, Any]]]:
    batches: list[list[int]] = []
    rows: dict[int, dict[str, Any]] = {}
    for path in discover(directory):
        blob = torch.load(path, weights_only=False)
        files = blob.get("sample_files")
        indices = blob.get("sample_indices")
        if not isinstance(files, list) or not isinstance(indices, list) or len(files) != len(indices):
            raise ValueError(f"invalid sample manifest: {path}")
        costs = blob.get("trainable_tokens")
        if len(indices) == 1 and isinstance(costs, int):
            per_row_costs = [costs]
        else:
            per_row_costs = [None] * len(indices)
        batch = []
        for sample_file, sample_index, cost in zip(files, indices, per_row_costs, strict=True):
            if not isinstance(sample_index, int) or sample_index in rows:
                raise ValueError(f"duplicate or invalid sample index in {path}: {sample_index!r}")
            rows[sample_index] = {"path": sample_file, "trainable_tokens": cost}
            batch.append(sample_index)
        batches.append(batch)
    return batches, rows


def load_prefix(directory: Path) -> list[list[int]]:
    batches = []
    seen = set()
    for path in discover(directory):
        blob = torch.load(path, weights_only=False)
        indices = blob.get("sample_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"invalid prefix manifest: {path}")
        if any(not isinstance(index, int) or index in seen for index in indices):
            raise ValueError(f"duplicate or invalid prefix index in {path}")
        seen.update(indices)
        batches.append(indices)
    return batches


def boundary_safe(stream: list[int], rows: list[int], batch_size: int) -> list[int]:
    previous_count = len(stream) % batch_size
    if not previous_count:
        return rows
    needed = batch_size - previous_count
    previous = set(stream[-previous_count:])
    safe = [index for index in rows if index not in previous][:needed]
    if len(safe) != needed:
        raise ValueError("cannot make an epoch boundary unique within a batch")
    promoted = set(safe)
    return safe + [index for index in rows if index not in promoted]


def shuffled(values: list[int], seed: int) -> list[int]:
    output = list(values)
    random.Random(seed).shuffle(output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-scored-dir", required=True, type=Path)
    parser.add_argument("--prefix-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    full_dir = args.full_scored_dir.resolve()
    prefix_dir = args.prefix_dir.resolve()
    output_dir = args.output_dir.resolve()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch size must be positive")

    _, full_rows = load_rows(full_dir)
    if set(full_rows) != set(range(len(full_rows))):
        raise ValueError("full scored sample indices are not contiguous")
    prefix_batches = load_prefix(prefix_dir)
    if any(len(batch) != args.batch_size for batch in prefix_batches):
        raise ValueError("every prefix batch must match --batch-size")
    prefix = [index for batch in prefix_batches for index in batch]
    if not set(prefix) <= set(full_rows):
        raise ValueError("prefix contains samples absent from the full scored set")

    all_indices = list(range(len(full_rows)))
    remaining = [index for index in all_indices if index not in set(prefix)]
    stream = list(prefix)
    first_remainder = shuffled(remaining, args.seed)
    stream.extend(boundary_safe(stream, first_remainder, args.batch_size))
    for epoch in range(1, args.epochs):
        epoch_rows = shuffled(all_indices, args.seed + epoch)
        stream.extend(boundary_safe(stream, epoch_rows, args.batch_size))

    requested_samples = len(full_rows) * args.epochs
    if len(stream) != requested_samples:
        raise AssertionError(f"constructed {len(stream)} rows, expected {requested_samples}")
    emitted_samples = math.ceil(requested_samples / args.batch_size) * args.batch_size
    if emitted_samples > len(stream):
        padding = shuffled(all_indices, args.seed + args.epochs)
        padding = boundary_safe(stream, padding, args.batch_size)
        stream.extend(padding[: emitted_samples - len(stream)])

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = list(output_dir.glob("rollout_*.pt"))
    if existing and not (output_dir / OWNER).is_file():
        raise ValueError(f"refusing to replace unowned rollout files under {output_dir}")
    for path in existing:
        path.unlink()

    sample_root = os.path.relpath(full_dir, output_dir)
    for rollout_id, start in enumerate(range(0, len(stream), args.batch_size)):
        indices = stream[start : start + args.batch_size]
        data = [full_rows[index] for index in indices]
        manifest = {
            "format": "offline_opd_sample_manifest_v2",
            "rollout_id": rollout_id,
            "samples_per_file": args.batch_size,
            "sample_root": sample_root,
            "allow_external_sample_symlinks": False,
            "sample_files": [row["path"] for row in data],
            "sample_indices": indices,
            "trainable_tokens": (
                sum(row["trainable_tokens"] for row in data)
                if all(row["trainable_tokens"] is not None for row in data)
                else None
            ),
        }
        temporary = output_dir / f".rollout_{rollout_id}.pt.tmp.{os.getpid()}"
        torch.save(manifest, temporary)
        os.replace(temporary, output_dir / f"rollout_{rollout_id}.pt")

    summary = {
        "format": "offline_opd_partial_resume_v1",
        "full_scored_dir": str(full_dir),
        "prefix_dir": str(prefix_dir),
        "dataset_size": len(full_rows),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "prefix_rollouts": len(prefix_batches),
        "prefix_samples": len(prefix),
        "first_epoch_remaining_samples": len(remaining),
        "requested_samples": requested_samples,
        "emitted_samples": emitted_samples,
        "padding_samples": emitted_samples - requested_samples,
        "num_rollouts": emitted_samples // args.batch_size,
        "resume_rollout_id": len(prefix_batches),
        "seed": args.seed,
    }
    temporary = output_dir / f".{OWNER}.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / OWNER)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
