#!/usr/bin/env python3
"""Derive a smaller dynamic Teacher cache without rerunning Teacher forward.

The source candidate columns are token-id sorted, not probability ranked.  This
tool therefore selects Top-K by cached log-probability and then restores token-id
ordering required by Phase-3 ``searchsorted``.  Files are handled independently,
written atomically, and may be sharded across nodes with ``--rank/--world-size``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import torch


SAMPLE_RE = re.compile(r"sample_(\d+)\.pt$")
SAMPLE_FORMAT = "offline_opd_scored_sample_v1"
MANIFEST_FORMATS = {
    "offline_opd_sample_manifest_v1",
    "offline_opd_sample_manifest_v2",
}


def discover_samples(source_dir: Path) -> list[tuple[int, Path]]:
    indexed: list[tuple[int, Path]] = []
    for path in (source_dir / "samples").glob("sample_*.pt"):
        match = SAMPLE_RE.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    indexed.sort()
    if not indexed:
        raise ValueError(f"no sample_*.pt files under {source_dir / 'samples'}")
    indices = [index for index, _ in indexed]
    if indices != list(range(len(indexed))):
        raise ValueError("source sample indices are not contiguous from zero")
    return indexed


def _stage_topk(
    candidate_ids: torch.Tensor,
    candidate_lp: torch.Tensor,
    teacher_head: torch.Tensor,
    target_k: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return probability Top-K, sorted by token id, preserving Teacher head."""
    work_lp = candidate_lp.to(device=device, non_blocking=False)
    values, columns = torch.topk(work_lp, target_k, dim=-1, largest=True, sorted=True)
    values = values.cpu()
    columns = columns.cpu()
    selected_ids = torch.gather(candidate_ids, 1, columns)

    # Float16 cache values can tie.  The exact Teacher Top-128 was captured
    # before token-id sorting, so make it a hard subset of the derived cache.
    sorted_selected = torch.sort(selected_ids, dim=-1).values
    head_columns = torch.searchsorted(candidate_ids, teacher_head)
    safe_head_columns = head_columns.clamp_max(candidate_ids.size(1) - 1)
    if not torch.gather(candidate_ids, 1, safe_head_columns).eq(teacher_head).all():
        raise ValueError("teacher head contains an id absent from the source cache")
    head_lookup = torch.searchsorted(sorted_selected, teacher_head)
    safe_head_lookup = head_lookup.clamp_max(target_k - 1)
    head_present = torch.gather(sorted_selected, 1, safe_head_lookup).eq(teacher_head)
    affected_rows = torch.nonzero(~head_present.all(dim=-1), as_tuple=False).flatten().tolist()
    for row in affected_rows:
        head_ids = set(map(int, teacher_head[row].tolist()))
        selected = selected_ids[row].tolist()
        missing = [token for token in teacher_head[row].tolist() if token not in set(selected)]
        removable = [
            column for column, token in enumerate(selected) if int(token) not in head_ids
        ]
        removable.sort(key=lambda column: (float(values[row, column]), int(selected[column])))
        if len(removable) < len(missing):
            raise ValueError(f"row {row}: cannot force Teacher head into Top-{target_k}")
        full_ids = candidate_ids[row]
        for token, output_column in zip(missing, removable, strict=False):
            source_column = int(torch.searchsorted(full_ids, torch.tensor(token)).item())
            if source_column >= full_ids.numel() or int(full_ids[source_column]) != int(token):
                raise ValueError(f"row {row}: Teacher head token {token} is absent")
            selected_ids[row, output_column] = int(token)
            values[row, output_column] = candidate_lp[row, source_column]

    sorted_ids, order = torch.sort(selected_ids, dim=-1)
    sorted_lp = torch.gather(values, 1, order)
    if sorted_ids.size(1) > 1 and not torch.all(sorted_ids[:, 1:] > sorted_ids[:, :-1]):
        raise ValueError("derived candidate ids are not unique and strictly sorted")
    head_lookup = torch.searchsorted(sorted_ids, teacher_head)
    if not torch.gather(sorted_ids, 1, head_lookup.clamp_max(target_k - 1)).eq(teacher_head).all():
        raise ValueError("derived cache does not contain every Teacher head id")
    return sorted_ids.to(torch.int32), sorted_lp.to(torch.float16)


def derive_sample(
    source: Path,
    target: Path,
    *,
    source_k: int,
    target_k: int,
    support_k: int,
    rows_per_chunk: int,
    device: torch.device,
) -> dict[str, int | float]:
    started = time.monotonic()
    blob = torch.load(source, weights_only=False, mmap=True)
    if not isinstance(blob, dict) or blob.get("format") != SAMPLE_FORMAT:
        raise ValueError(f"{source}: unsupported scored-sample format")
    sample = blob.get("sample")
    if not isinstance(sample, dict) or blob.get("index") is None:
        raise ValueError(f"{source}: malformed scored sample")
    candidate_ids = torch.as_tensor(sample.get("teacher_candidate_ids"))
    candidate_lp = torch.as_tensor(sample.get("teacher_candidate_logprobs"))
    teacher_head = torch.as_tensor(sample.get("teacher_top128_ids"), dtype=torch.int32)
    positions = torch.as_tensor(sample.get("teacher_candidate_positions"))
    n_rows = int(positions.numel())
    if tuple(candidate_ids.shape) != (n_rows, source_k):
        raise ValueError(f"{source}: ids shape {tuple(candidate_ids.shape)} != {(n_rows, source_k)}")
    if tuple(candidate_lp.shape) != (n_rows, source_k) or candidate_lp.dtype != torch.float16:
        raise ValueError(f"{source}: invalid source log-prob cache")
    if tuple(teacher_head.shape) != (n_rows, support_k):
        raise ValueError(f"{source}: invalid Teacher head shape {tuple(teacher_head.shape)}")
    if candidate_ids.dtype != torch.int32:
        raise ValueError(f"{source}: expected int32 candidate ids")
    if candidate_ids.numel() and not torch.all(candidate_ids[:, 1:] > candidate_ids[:, :-1]):
        raise ValueError(f"{source}: source candidate ids are not strictly sorted")

    output_ids = torch.empty((n_rows, target_k), dtype=torch.int32)
    output_lp = torch.empty((n_rows, target_k), dtype=torch.float16)
    for start in range(0, n_rows, rows_per_chunk):
        end = min(start + rows_per_chunk, n_rows)
        ids, lp = _stage_topk(
            candidate_ids[start:end],
            candidate_lp[start:end],
            teacher_head[start:end],
            target_k,
            device,
        )
        output_ids[start:end] = ids
        output_lp[start:end] = lp

    sample["teacher_candidate_ids"] = output_ids.contiguous()
    sample["teacher_candidate_logprobs"] = output_lp.contiguous()
    sample["teacher_cutoff_logprobs"] = output_lp.float().amin(dim=-1).to(torch.float16)
    # The field name is retained for artifact compatibility; its value is the
    # mass of the configured cache width (Top-1024 for this experiment).
    sample["teacher_top4096_mass"] = output_lp.float().exp().sum(dim=-1).to(torch.float32)
    sample["offline_opd_dynamic_teacher_sparse"] = True

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp.{os.getpid()}")
    torch.save(blob, temporary)
    os.replace(temporary, target)
    return {
        "index": int(blob["index"]),
        "rows": n_rows,
        "source_bytes": source.stat().st_size,
        "target_bytes": target.stat().st_size,
        "seconds": round(time.monotonic() - started, 3),
    }


def validate_existing(path: Path, index: int, target_k: int, support_k: int) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        blob = torch.load(path, weights_only=False, mmap=True)
        sample = blob["sample"]
        ids = torch.as_tensor(sample["teacher_candidate_ids"])
        lp = torch.as_tensor(sample["teacher_candidate_logprobs"])
        head = torch.as_tensor(sample["teacher_top128_ids"])
        valid = (
            blob.get("format") == SAMPLE_FORMAT
            and blob.get("index") == index
            and ids.ndim == 2
            and tuple(lp.shape) == tuple(ids.shape)
            and ids.shape[1] == target_k
            and head.shape == (ids.shape[0], support_k)
            and ids.dtype == torch.int32
            and lp.dtype == torch.float16
            and torch.as_tensor(sample["teacher_cutoff_logprobs"]).shape == (ids.shape[0],)
            and torch.as_tensor(sample["teacher_top4096_mass"]).shape == (ids.shape[0],)
        )
        # Conversion uses an atomic rename only after all invariants have been
        # checked. Header/shape validation keeps restart and finalization from
        # rereading hundreds of GiB merely to skip completed outputs.
        return bool(valid)
    except Exception:
        return False


def copy_manifests(source_dir: Path, output_dir: Path, expected_samples: int) -> None:
    source_manifests = source_dir / "source_manifests"
    if not source_manifests.is_dir():
        raise ValueError(f"missing source manifests: {source_manifests}")
    target_manifests = output_dir / "source_manifests"
    target_manifests.mkdir(parents=True, exist_ok=True)
    json_paths = sorted(source_manifests.glob("rollout_*.json"))
    rollout_paths = sorted(source_dir.glob("rollout_*.pt"))
    if len(json_paths) != expected_samples or len(rollout_paths) != expected_samples:
        raise ValueError(
            f"manifest count mismatch: json={len(json_paths)}, pt={len(rollout_paths)}, "
            f"samples={expected_samples}"
        )
    for source in json_paths:
        shutil.copy2(source, target_manifests / source.name)
    for source in rollout_paths:
        blob = torch.load(source, weights_only=False)
        if not isinstance(blob, dict) or blob.get("format") not in MANIFEST_FORMATS:
            raise ValueError(f"{source}: unsupported batch manifest")
        shutil.copy2(source, output_dir / source.name)
    summary = source_dir / ".sample_manifest_summary.json"
    if summary.is_file():
        shutil.copy2(summary, output_dir / summary.name)
    marker = {
        "format": "offline_opd_derived_teacher_cache_v1",
        "source_scored_dir": str(source_dir.resolve()),
        "source_topk": 4096,
        "target_topk": 1024,
        "dynamic_support_topk": 128,
        "num_samples": expected_samples,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    temporary = output_dir / f"..teacher_cache_derived.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output_dir / ".teacher_cache_derived.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--source-k", type=int, default=4096)
    parser.add_argument("--target-k", type=int, default=1024)
    parser.add_argument("--support-k", type=int, default=128)
    parser.add_argument("--rows-per-chunk", type=int, default=512)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--finalize-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.support_k <= args.target_k < args.source_k:
        parser.error("require 0 < support-k <= target-k < source-k")
    if not 0 <= args.rank < args.world_size:
        parser.error("rank must be in [0, world-size)")
    if args.rows_per_chunk <= 0:
        parser.error("rows-per-chunk must be positive")

    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    samples = discover_samples(source_dir)
    if args.limit is not None:
        samples = samples[: args.limit]
    if args.finalize_only:
        missing = [index for index, _ in samples if not validate_existing(
            output_dir / "samples" / f"sample_{index:06d}.pt",
            index,
            args.target_k,
            args.support_k,
        )]
        if missing:
            raise SystemExit(f"cannot finalize: {len(missing)} samples missing/invalid; first={missing[:10]}")
        copy_manifests(source_dir, output_dir, len(samples))
        print(f"finalized {len(samples)} derived samples under {output_dir}", flush=True)
        return

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    output_samples = output_dir / "samples"
    output_samples.mkdir(parents=True, exist_ok=True)
    assigned = [(index, path) for index, path in samples if index % args.world_size == args.rank]
    print(
        f"rank={args.rank}/{args.world_size} device={device} assigned={len(assigned)} "
        f"source={source_dir} output={output_dir}",
        flush=True,
    )
    completed = skipped = 0
    for position, (index, source) in enumerate(assigned, 1):
        target = output_samples / f"sample_{index:06d}.pt"
        if validate_existing(target, index, args.target_k, args.support_k):
            skipped += 1
            continue
        stats = derive_sample(
            source,
            target,
            source_k=args.source_k,
            target_k=args.target_k,
            support_k=args.support_k,
            rows_per_chunk=args.rows_per_chunk,
            device=device,
        )
        completed += 1
        print(
            f"rank={args.rank} progress={position}/{len(assigned)} "
            f"index={index} rows={stats['rows']} seconds={stats['seconds']} "
            f"size_mib={stats['target_bytes'] / 2**20:.1f}",
            flush=True,
        )
    marker_dir = output_dir / ".derive_ranks"
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f"rank_{args.rank}.done"
    marker.write_text(f"completed={completed} skipped={skipped} finished={time.time()}\n")
    print(f"rank={args.rank} done completed={completed} skipped={skipped}", flush=True)


if __name__ == "__main__":
    main()
