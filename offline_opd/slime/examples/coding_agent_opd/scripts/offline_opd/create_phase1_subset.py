#!/usr/bin/env python3
"""Create a deterministic, zero-copy subset of one-sample Phase-1 dumps."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from pathlib import Path

from examples.coding_agent_opd.scripts.offline_opd.offline_utils import discover_rollout_files


MANIFEST = ".phase1_subset.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--count", required=True, type=int)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.input_dir.resolve()
    output = args.output_dir.resolve()
    paths = discover_rollout_files(source)
    if args.count <= 0 or args.count > len(paths):
        raise ValueError(f"count must be in 1..{len(paths)}, got {args.count}")

    selected = random.Random(args.seed).sample(paths, args.count)
    expected = {
        "version": 1,
        "input_dir": str(source),
        "count": args.count,
        "seed": args.seed,
        "source_files": [path.name for path in selected],
    }
    manifest_path = output / MANIFEST
    if output.is_dir() and manifest_path.is_file():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        links_ok = all((output / f"rollout_{i}.pt").is_file() for i in range(args.count))
        if actual == expected and links_ok:
            print(f"reuse Phase-1 subset: {output} ({args.count} samples, seed={args.seed})")
            return
        raise FileExistsError(f"existing subset does not match requested selection: {output}")
    if output.exists():
        raise FileExistsError(f"refusing to replace existing path: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / f".{output.name}.tmp.{os.getpid()}"
    staging.mkdir()
    try:
        for new_id, source_path in enumerate(selected):
            os.symlink(source_path, staging / f"rollout_{new_id}.pt")
        (staging / MANIFEST).write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"created Phase-1 subset: {output} ({args.count}/{len(paths)}, seed={args.seed})")


if __name__ == "__main__":
    main()
