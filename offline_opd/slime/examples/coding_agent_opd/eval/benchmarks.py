"""Benchmark registry: resolve a benchmark NAME and its data file / checks.

This is the single place that decides which file + swe_types + capability
checks belong to which benchmark name. The whole eval chain
(``run_eval.sh`` -> ``eval_chat.py``) switches benchmarks by setting
``BENCHMARK=<name>``, and
(default when no ``DATA`` is given) resolves the data file from this registry,
so no script hardcodes a dataset path.

Switching benchmarks is meant to be a *data-level* operation: the runtime
already dispatches per row via ``ds["swe_type"]`` (DockerRuntimeMap), so a
benchmark entry is exactly {file, swe_types, checks}.

Preflight checks (host-side only, no pod, no server):

  * ``required_fields`` -- every row passes the same required-field rule the
    rollout applies before booting a pod (``rollout._missing_ds_field``); a row
    that fails here would otherwise waste a pod-boot and grade 0.
  * ``spec_coverage`` -- the *vendored* swebench spec map must answer
    ``(repo, version)`` for each declared-swe_type row. This is the gate that
    multilingual repos fail on swebench < 4.1 (KeyError at boot, before the pod
    exists).
  * ``run_scripts`` -- swebench-pro rows need their vendored run-script dir
    (``run_script.sh`` + ``parser.py``); missing dirs mean reward=0 league-wide.

Rows of a swe_type the benchmark does NOT declare (``foreign``) are reported
but not failed -- they would still boot under their own runtime.

CLI:

    python3 -m examples.coding_agent_opd.eval.benchmarks              # resolve swebench file
    python3 -m examples.coding_agent_opd.eval.benchmarks swebench-pro # resolve that file
    python3 -m examples.coding_agent_opd.eval.benchmarks --check-only swebench-pro
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

_EVAL_DIR = Path(__file__).resolve().parent            # .../coding_agent_opd/eval
_OPD_DIR = _EVAL_DIR.parent                            # .../coding_agent_opd
_SLIME_DIR = _OPD_DIR.parent.parent                    # repo root
_REGISTRY_PATH = _EVAL_DIR / "benchmarks.json"
_VENDORED_RUN_SCRIPTS = (
    _OPD_DIR / "3rdparty" / "R2E-Gym" / "third_party" / "swebench_pro_run_scripts"
)
_DEFAULT_BENCHMARK = os.environ.get("BENCHMARK", "swebench")


def _load_registry() -> dict[str, dict[str, Any]]:
    if not _REGISTRY_PATH.is_file():
        raise FileNotFoundError(
            f"benchmark registry missing: {_REGISTRY_PATH} "
            "(expected the eval/benchmarks.json file)"
        )
    with _REGISTRY_PATH.open(encoding="utf-8") as f:
        reg = json.load(f)
    if not isinstance(reg, dict) or not reg:
        raise ValueError(f"{_REGISTRY_PATH} must be a non-empty {{\"name\": {{...}}}} map")
    return reg


def available() -> list[str]:
    return sorted(_load_registry())


def get(name: str) -> dict[str, Any]:
    """Registry entry for ``name``."""
    reg = _load_registry()
    try:
        return reg[name]
    except KeyError:
        raise KeyError(
            f"unknown benchmark {name!r}; available: {', '.join(available())}"
        ) from None


def resolve_file(name: str) -> Path:
    """Absolute path of the benchmark's data file."""
    entry = get(name)
    raw = entry.get("file") or entry.get("source")
    if not raw:
        raise ValueError(f"benchmark {name!r} has no \"file\" entry in the registry")
    p = Path(raw)
    if not p.is_absolute():
        p = Path(os.environ.get("SLIME_DIR", str(_SLIME_DIR))) / p
    return p


# ---------------------------------------------------------------------------
# Host-side preflight checks
# ---------------------------------------------------------------------------
def _read_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _vendored_spec_map() -> dict:
    """The vendored swebench 4.1.x spec map -- exactly what make_test_spec indexes.

    The runtime imports ``swebench`` from the vendored tree under
    ``3rdparty/R2E-Gym/third_party``; a probe of THAT copy is a faithful
    "will boot?" predicate for the spec lookup.
    """
    import sys

    vendored_third_party = _OPD_DIR / "3rdparty" / "R2E-Gym" / "third_party"
    if str(vendored_third_party) not in sys.path:
        sys.path.insert(0, str(vendored_third_party))
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS  # noqa

    return MAP_REPO_VERSION_TO_SPECS


def check_rows(
    rows: list[dict],
    *,
    allowed_types: set[str] | None = None,
    checks: tuple[str, ...] = ("required_fields", "spec_coverage", "run_scripts"),
) -> dict[str, Any]:
    """Run host-side check groups over ``rows``; return the checks part of a report.

    ``allowed_types`` (optional) restricts the spec/script checks to rows whose
    ``swe_type`` is declared by the benchmark; other rows are kept out of those
    checks (they'd boot under their own runtime anyway, where the local check
    may not apply -- e.g. scale-swe vs swebench-pro).
    """
    kept = [r for r in rows if not allowed_types or r.get("swe_type") in allowed_types]
    res: list[dict[str, Any]] = []
    if "required_fields" in checks:
        # the same rule the rollout applies before booting each pod
        from ..utils.rollout import _missing_ds_field  # noqa

        dropped = []
        for r in rows:
            miss = _missing_ds_field(r)
            if miss is not None:
                dropped.append({"instance_id": r.get("instance_id"), "missing": miss})
        res.append({
            "check": "required_fields",
            "ok": not dropped,
            "detail": (
                f"{len(rows) - len(dropped)}/{len(rows)} rows have all required fields"
                + (f"; {len(dropped)} dropped" if dropped else "")
            ),
            "dropped": dropped[:10],
            "dropped_count": len(dropped),
        })
    if "spec_coverage" in checks:
        spec = _vendored_spec_map()
        uncovered = []
        for r in kept:
            repo = r.get("repo")
            ver = r.get("version") or ""
            if not repo or not ver or repo not in spec or ver not in spec[repo]:
                uncovered.append(
                    {"instance_id": r.get("instance_id"), "repo": repo, "version": ver}
                )
        res.append({
            "check": "spec_coverage",
            "ok": not uncovered,
            "detail": (
                f"vendored swebench spec map: {len(kept) - len(uncovered)}/{len(kept)} "
                f"rows have (repo, version) spec; uncovered={len(uncovered)}"
            ),
            "uncovered": uncovered[:10],
            "uncovered_count": len(uncovered),
        })
    if "run_scripts" in checks:
        missing = []
        for r in kept:
            d = _VENDORED_RUN_SCRIPTS / (r.get("instance_id") or "")
            if not (d / "run_script.sh").is_file() or not (d / "parser.py").is_file():
                missing.append({"instance_id": r.get("instance_id")})
        res.append({
            "check": "run_scripts",
            "ok": not missing,
            "detail": (
                f"vendored run-scripts: {len(kept) - len(missing)}/{len(kept)} rows have "
                f"run_script.sh + parser.py; missing={len(missing)}"
            ),
            "missing": missing[:10],
            "missing_count": len(missing),
        })
    return {"checks": res, "ok": all(c["ok"] for c in res)}


def check_benchmark(name: str) -> dict[str, Any]:
    """Run the benchmark's configured host-side checks; return a report dict.

    Shape::

        {
          "name", "display", "file",
          "rows", "rows_by_swe_type", "foreign",
          "checks": [ {check, ok, detail, ...} ... ],
          "ok": all(ok),
        }
    """
    entry = get(name)
    path = resolve_file(name)
    allowed = set(entry.get("swe_types") or [])
    rows = _read_rows(path)
    stats = {}
    for r in rows:
        st = r.get("swe_type") or "(none)"
        stats[st] = stats.get(st, 0) + 1
    checks = check_rows(
        rows, allowed_types=allowed, checks=tuple(entry.get("checks") or ())
    )
    return {
        "name": name,
        "display": entry.get("display_name", name),
        "file": str(path),
        "rows": len(rows),
        "rows_by_swe_type": stats,
        "foreign": len(rows) - sum(n for st, n in stats.items() if st in allowed),
        "checks": checks["checks"],
        "ok": checks["ok"],
    }


def _print_report(r: dict) -> None:
    print(f"benchmark : {r['name']}  ({r['display']})")
    print(f"data file : {r['file']}")
    print(
        f"rows      : {r['rows']}  by_swe_type={r['rows_by_swe_type']}  "
        f"foreign={r['foreign']}"
    )
    for c in r["checks"]:
        flag = "PASS" if c["ok"] else "FAIL"
        extra = ""
        for tag in ("dropped", "uncovered", "missing"):
            n = c.get(f"{tag}_count")
            if n:
                extra += f"  {tag}({n}): {c[tag]}"
        print(f"  [{flag}] {c['check']}: {c['detail']}{extra}")
    print(f"verdict   : {'PASS — runnable' if r['ok'] else 'FAIL — not runnable'}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="benchmarks", description=__doc__)
    ap.add_argument("--benchmark", default=_DEFAULT_BENCHMARK, help="benchmark name")
    ap.add_argument(
        "name", nargs="?", default=None,
        help="benchmark name (positional; overrides --benchmark)",
    )
    ap.add_argument(
        "--check-only", action="store_true",
        help="run preflight checks and exit 0/1 (no eval, no pod)",
    )
    args = ap.parse_args()
    benchmark = args.name or args.benchmark
    if args.check_only:
        r = check_benchmark(benchmark)
        _print_report(r)
        raise SystemExit(0 if r["ok"] else 1)
    # default: print the resolved data file path (used by the shell scripts)
    print(resolve_file(benchmark))


if __name__ == "__main__":
    main()