#!/usr/bin/env python3
"""End-to-end smoke test for ``slime.agent.sandbox_rle.RLESandbox``.

Boots a real RLE pod, exercises every method of the Sandbox Protocol, tears
the pod down, and prints one line per check.

Prerequisites
-------------
* ``rle_open_sdk`` importable in the current interpreter.
* ``RLE_API_KEY`` exported (see ``.venv_example`` for the test credential).
* An image reachable from the RLE session -- by default this is read from
  the first record of ``data/external_hf_v2_with_docker.jsonl``.

Usage
-----
    # Simplest form -- pulls image from the default dataset
    export RLE_API_KEY='code_swe_data_test:...'
    export RLE_ENV=test
    export RLE_TOOL_TYPE=swe-datagen
    python examples/coding_agent_rl/rle_scripts/smoke_test_sandbox.py

    # Override the image directly
    python examples/coding_agent_rl/rle_scripts/smoke_test_sandbox.py \
        --image ccr-<instance-id>-vpc.cnc.example.com/foo/bar:tag

    # Point at a different JSONL and pick a specific line
    python examples/coding_agent_rl/rle_scripts/smoke_test_sandbox.py \
        --dataset data/other.jsonl --line 3

Exit code is 0 on all-pass, 1 on any assertion failure.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import traceback
from pathlib import Path

# Make ``import slime.*`` work when running the script directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slime.agent.sandbox_rle import RLESandbox  # noqa: E402
from slime.agent.sandbox import install_r2egym_tools  # noqa: E402


DEFAULT_DATASET = _REPO_ROOT / "data" / "external_hf_v2_with_docker.jsonl"


def _load_image(dataset: Path, line_idx: int) -> str:
    with open(dataset, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == line_idx:
                rec = json.loads(line)
                img = rec.get("docker_image") or rec.get("image_name")
                if not img:
                    raise SystemExit(
                        f"[fatal] no docker_image/image_name field in {dataset}:{line_idx}"
                    )
                return img
    raise SystemExit(f"[fatal] {dataset} has fewer than {line_idx + 1} lines")


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")


async def run_checks(image: str) -> int:
    """Return number of failed checks."""
    failed = 0

    def check(cond: bool, label: str) -> None:
        nonlocal failed
        if cond:
            _ok(label)
        else:
            _fail(label)
            failed += 1

    print(f"[smoke] image      : {image}")
    print(f"[smoke] resolved   : {RLESandbox._resolve_image_path(image, '')}")

    t0 = time.time()
    async with RLESandbox(image, setup=install_r2egym_tools) as sb:
        boot = time.time() - t0
        print(f"[smoke] session_id : {sb.sandbox_id[:16]}...  (boot+setup {boot:.1f}s)")
        print(f"[smoke] pod_view   : {sb.pod_view_url[:120]}")

        # 1. echo
        rc, out, _ = await sb.exec("echo hello-from-rle", timeout=20)
        check(rc == 0 and "hello-from-rle" in out, "exec: echo returns rc=0 with stdout")

        # 2. str round-trip
        await sb.write_file("/tmp/slime_probe.txt", "line1\nline2\n")
        got = await sb.read_file("/tmp/slime_probe.txt")
        check(got == "line1\nline2\n", "write_file(str) -> read_file round-trip")

        # 3. bytes round-trip (verify binary is not corrupted)
        await sb.write_file("/tmp/slime_probe.bin", b"\x00\x01\x02binary")
        rc, out, _ = await sb.exec(
            "od -An -tx1 /tmp/slime_probe.bin | head -1", timeout=15
        )
        check(rc == 0 and "00 01 02" in out, "write_file(bytes) preserves binary")

        # 4. env dict survives shell wrapping (spaces + quoting)
        rc, out, _ = await sb.exec(
            "echo $FOO_BAR", env={"FOO_BAR": "quux 42"}, timeout=15
        )
        check(rc == 0 and out.strip() == "quux 42", "exec: env dict survives shell quoting")

        # 5. Non-zero exit surfaces as rc, not as exception
        rc, _, _ = await sb.exec("false", timeout=15)
        check(rc == 1, "exec: non-zero exit returned as rc")

        # 6. check=True raises RuntimeError on non-zero exit
        raised = False
        try:
            await sb.exec("false", timeout=15, check=True)
        except RuntimeError:
            raised = True
        check(raised, "exec: check=True raises on non-zero exit")

        # 7. Long-ish command (10s sleep) completes without stream sever
        rc, _, _ = await sb.exec("sleep 5 && echo done", timeout=30)
        check(rc == 0, "exec: sleep 5 completes within timeout")

        # 8. Timeout signal (inner shell 'timeout' returns 124)
        rc, _, _ = await sb.exec("sleep 30", timeout=3)
        check(rc == 124, "exec: outer timeout produces rc=124")

        # 9-12. rllm-parity tool install + invocation
        # ----------------------------------------------
        # ``setup=install_r2egym_tools`` already ran in __aenter__: shebang
        # symlink, pip install chardet, add_commands upload of the tool
        # scripts. We just assert the results are visible then exercise the
        # CLI shape agents will use (Action.to_bashcmd output).

        # 9. setup hook: shebang + tools present
        rc, out, _ = await sb.exec(
            "test -x /root/.venv/bin/python && "
            "test -x /usr/local/bin/str_replace_editor && "
            "test -x /usr/local/bin/execute_bash && "
            "python -c 'import chardet' && echo OK",
            timeout=30,
        )
        check(
            rc == 0 and out.strip() == "OK",
            "setup=install_r2egym_tools: shebang + tools + chardet in place",
        )

        # 10. str_replace_editor create -> view -> str_replace -> view round-trip
        # Argv shape matches Action.to_bashcmd() output. Real newlines inside
        # a single-quoted arg preserve as-is through /bin/sh -c (avoids
        # dash-vs-bash ``$'...'`` incompatibility).
        await sb.exec("rm -f /tmp/slime_editor_probe.txt", timeout=15)
        rc, out, err = await sb.exec(
            "str_replace_editor create --path /tmp/slime_editor_probe.txt "
            "--file_text 'alpha\nbeta\n'",
            timeout=30,
        )
        check(rc == 0 and "File created" in out, "str_replace_editor: create")

        rc, out, _ = await sb.exec(
            "str_replace_editor str_replace --path /tmp/slime_editor_probe.txt "
            "--old_str beta --new_str gamma",
            timeout=30,
        )
        check(rc == 0 and "has been edited" in out, "str_replace_editor: str_replace")

        got = await sb.read_file("/tmp/slime_editor_probe.txt")
        check(got == "alpha\ngamma\n", "str_replace_editor edit visible via read_file")

        # 11. execute_bash CLI usable via the same shell dispatch
        rc, out, _ = await sb.exec(
            "execute_bash 'echo bash-ok && cat /tmp/slime_editor_probe.txt'",
            timeout=30,
        )
        check(
            rc == 0 and "bash-ok" in out and "gamma" in out,
            "execute_bash: CLI dispatch through sb.exec",
        )

    # 12. session_ids.jsonl recorded (relative to $RLE_SESSION_IDS_DIR or CWD)
    sid_dir = os.environ.get("RLE_SESSION_IDS_DIR", "./")
    sid_path = Path(sid_dir) / "session_ids.jsonl"
    if sid_path.exists():
        last = sid_path.read_text().strip().splitlines()[-1]
        check(len(last) > 0, f"session_ids.jsonl recorded ({sid_path})")
    else:
        _fail(f"session_ids.jsonl missing at {sid_path}")
        failed += 1

    return failed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--image", help="Explicit image path (overrides --dataset)")
    p.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"JSONL dataset to pull docker_image from (default: {DEFAULT_DATASET})",
    )
    p.add_argument("--line", type=int, default=0, help="Line index in dataset (0-based)")
    args = p.parse_args()

    if not os.environ.get("RLE_API_KEY"):
        print(
            "[fatal] RLE_API_KEY not set. Export it before running (see .venv_example).",
            file=sys.stderr,
        )
        return 2

    if args.image:
        image = args.image
    else:
        dataset = Path(args.dataset)
        if not dataset.exists():
            print(f"[fatal] dataset not found: {dataset}", file=sys.stderr)
            return 2
        image = _load_image(dataset, args.line)

    try:
        failed = asyncio.run(run_checks(image))
    except Exception:
        traceback.print_exc()
        return 1

    print()
    if failed:
        print(f"[smoke] FAILED ({failed} check(s))")
        return 1
    print("[smoke] PASSED (all checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
