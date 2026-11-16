"""Wrap raw R2E-Gym rows into slime training rows -- SAME format as eval.

    python examples/coding_agent_opd/prepare_dataset.py \
        --in data/swebench-latest.jsonl \
        --out data/swe_train.jsonl

The training rollout (``generate.generate``) drives the student through the
sweagent tool loop against an ``AsyncRepoEnv`` and grades in place. The env is
booted straight off an R2E-Gym ``ds`` dict, which is *exactly* what each row of
the eval dataset already is (e.g. ``data/swebench-latest.jsonl``): the row
carries ``swe_type``, ``docker_image``, ``instance_id``, ``repo``,
``base_commit``, ``test_patch``, ``patch``, ``FAIL_TO_PASS`` / ``PASS_TO_PASS``
(swebench) or ``fail_to_pass`` / ``pass_to_pass`` (swebench-pro) -- the fields
``_calculate_reward`` / ``setup_env`` read off ``self.ds``.

Eval reads those rows verbatim (``eval_chat._load_dataset``). Training goes
through slime's ``Dataset``, which only surfaces ``prompt`` / ``label`` /
``metadata`` per row and drops every other key -- so the raw ds would be lost.
This script is the thin bridge: it keeps the eval dataset as the single on-disk
source of truth and wraps each row into the ``{prompt, label, metadata}`` shape
slime needs, stashing the whole row under ``metadata["ds"]`` unchanged.

Each emitted slime row is ``{prompt, label, metadata}`` where:

  * ``prompt`` / ``label`` are the (plain-text) problem statement and instance id;
  * ``metadata["ds"]`` is the full R2E-Gym ds dict passed verbatim to
    ``AsyncRepoEnv.create(ds)``;
  * ``metadata["workdir"]`` is the repo path the prompt should point at, derived
    from ``swe_type`` the same way eval does: ``/app`` for swebench-pro (whose
    ``SWEBenchProDocker`` hardcodes it), else ``/testbed`` (the sweagent template
    default that R2E-Gym's swebench images use).

The only transform is unwrapping ``problem_statement`` (some sources store it as
a JSON-quoted string). All grading fields are left as-is (the vendored reward
code ``eval()``s them), so no re-encoding happens here.

scale-swe rows (``swe_type == "scale-swe"``, e.g.
``data/scale_swe_python_sample5000.jsonl``) go through the same wrapper but have a
different grading schema, which ``ScaleSWEDocker`` (vendored ``runtime/docker.py``)
reads: no ``test_patch`` / ``base_commit``, instead ``f2p_patch`` + ``f2p_script``
+ ``parent_commit`` + ``pre_commands``, and the repo lives at the row's own
``workdir`` (``/workspace/<repo>``) rather than ``/testbed``. Rows are kept iff
they carry ``instance_id`` / ``docker_image`` / ``workdir``, at least one of
``f2p_patch`` / ``f2p_script``, and FAIL_TO_PASS + PASS_TO_PASS.

For swebench-pro rows we (by default) also skip any whose ``instance_id`` has no
vendored run-script directory
(``3rdparty/R2E-Gym/third_party/swebench_pro_run_scripts/<id>/``), because
``SWEBenchProDocker.setup_env`` requires ``run_script.sh`` + ``parser.py`` there
and would otherwise score reward=0. Pass ``--no-check-run-scripts`` to keep them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# swebench-pro's SWEBenchProDocker forces repo_path="/app"; every other swe_type
# uses the sweagent template's baked "/testbed". The prompt workdir must match so
# the model's paths resolve (render_instance rewrites /testbed -> workdir).
# scale-swe is the exception: its images bake the repo at /workspace/<repo> and the
# row carries that path in ``workdir``, which ScaleSWEDocker uses as repo_path.
_SWEBENCH_PRO_WORKDIR = "/app"
_DEFAULT_WORKDIR = "/testbed"
_SWEBENCH_PRO_TYPE = "swebench-pro"
_SCALE_SWE_TYPE = "scale-swe"
_YIYAN_FACTORY_TYPE = "yiyan-sweimages-factory"

# Vendored run-script tree (instance_<id>/{run_script.sh,parser.py}). Computed
# from __file__ to avoid importing the heavy r2egym package just for a path.
_OPD_DIR = Path(__file__).resolve().parent
_RUN_SCRIPTS_DIR = (
    _OPD_DIR / "3rdparty" / "R2E-Gym" / "third_party" / "swebench_pro_run_scripts"
)

# ds fields the rollout requires to boot + grade a pod. FAIL_TO_PASS /
# PASS_TO_PASS are uppercase in the canonical swebench schema and lowercase in
# swebench-pro; accept whichever case the row carries (mirrors
# generate._missing_ds_field so a row this script keeps is one the trainer boots).
_REQUIRED_FIELDS = (
    "instance_id",
    "docker_image",
    "test_patch",
)
_REQUIRED_FIELD_PAIRS = (
    ("fail_to_pass", "FAIL_TO_PASS"),
    ("pass_to_pass", "PASS_TO_PASS"),
)

# scale-swe rows carry no test_patch: the graded tests come from ``f2p_patch``
# (a diff over the repo's own test files) and/or ``f2p_script`` (a standalone
# pytest file), and the repo path is in ``workdir``. Mirrors
# rollout._REQUIRED_SCALE_SWE_DS_FIELDS / ScaleSWEDocker.
_REQUIRED_SCALE_SWE_FIELDS = (
    "instance_id",
    "docker_image",
    "workdir",
)
_REQUIRED_SCALE_SWE_FIELD_PAIRS = (
    ("f2p_patch", "f2p_script"),
    ("fail_to_pass", "FAIL_TO_PASS"),
    ("pass_to_pass", "PASS_TO_PASS"),
)

# yiyan-sweimages-factory rows carry no FAIL_TO_PASS/PASS_TO_PASS: grading runs
# the self-contained ``eval_script`` (uploaded as /run_tests.sh) and reads its
# ``OMNIGRIL_EXIT_CODE``. YiYanSWEFactoryImagesDocker.setup_env resets to
# ``base_commit`` and strips the added files of ``patch`` / ``test_patch``, so
# those are the boot-critical fields. The repo lives at the default /testbed.
# Mirrors rollout._REQUIRED_YIYAN_FACTORY_DS_FIELDS.
_REQUIRED_YIYAN_FACTORY_FIELDS = (
    "instance_id",
    "docker_image",
    "base_commit",
    "test_patch",
    "patch",
    "eval_script",
)


def _coerce_text(v) -> str:
    """problem_statement is sometimes stored as a JSON-quoted string; unwrap."""
    if isinstance(v, str):
        try:
            parsed = json.loads(v)
            if isinstance(parsed, str):
                return parsed
        except json.JSONDecodeError:
            pass
        return v
    return "" if v is None else str(v)


def _missing_field(raw: dict) -> str | None:
    if raw.get("swe_type") == _SCALE_SWE_TYPE:
        fields = _REQUIRED_SCALE_SWE_FIELDS
        pairs = _REQUIRED_SCALE_SWE_FIELD_PAIRS
    elif raw.get("swe_type") == _YIYAN_FACTORY_TYPE:
        fields = _REQUIRED_YIYAN_FACTORY_FIELDS
        pairs = ()
    else:
        fields = _REQUIRED_FIELDS
        pairs = _REQUIRED_FIELD_PAIRS
    for f in fields:
        if not raw.get(f):
            return f
    for first, second in pairs:
        if not raw.get(first) and not raw.get(second):
            return first
    return None


def _workdir_for(raw: dict) -> str:
    swe_type = raw.get("swe_type")
    if swe_type == _SCALE_SWE_TYPE:
        return raw["workdir"]
    return _SWEBENCH_PRO_WORKDIR if swe_type == _SWEBENCH_PRO_TYPE else _DEFAULT_WORKDIR


def _build_row(raw: dict, *, check_run_scripts: bool) -> dict | None:
    """Shape one raw R2E-Gym row into a slime ``{prompt,label,metadata}`` row.

    Returns None (skip) if the row lacks a required grading field, or -- for
    swebench-pro rows, unless disabled -- if its vendored run-script directory is
    absent. The row is otherwise passed through verbatim as the ds.
    """
    missing = _missing_field(raw)
    if missing is not None:
        return None

    instance_id = raw["instance_id"]
    swe_type = raw.get("swe_type")

    if check_run_scripts and swe_type == _SWEBENCH_PRO_TYPE:
        inst_dir = _RUN_SCRIPTS_DIR / instance_id
        if not (inst_dir / "run_script.sh").is_file() or not (inst_dir / "parser.py").is_file():
            return None

    problem = _coerce_text(raw.get("problem_statement"))

    # Pass the raw row through as the R2E-Gym ds, with problem_statement unwrapped
    # so the prompt sees plain text, not a JSON literal.
    ds = dict(raw)
    ds["problem_statement"] = problem

    return {
        "prompt": problem,
        "label": instance_id,
        "metadata": {
            "instance_id": instance_id,
            "workdir": _workdir_for(raw),
            "ds": ds,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/swebench-latest.jsonl")
    ap.add_argument("--out", dest="out", default="data/swe_train.jsonl")
    ap.add_argument(
        "--language",
        default="all",
        help="keep only rows with this language (js/python/go/ts); 'all' keeps every row",
    )
    ap.add_argument("--limit", type=int, default=0, help="max rows to emit (0 = no limit)")
    ap.add_argument(
        "--no-check-run-scripts",
        dest="check_run_scripts",
        action="store_false",
        help="keep swebench-pro rows even if their vendored run-script dir is missing",
    )
    args = ap.parse_args()

    src = Path(args.inp)
    dst = Path(args.out)
    kept = 0
    skipped = 0
    with src.open() as fin, dst.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if args.language != "all" and (raw.get("language") or "").lower() != args.language.lower():
                continue
            row = _build_row(raw, check_run_scripts=args.check_run_scripts)
            if row is None:
                skipped += 1
                continue
            fout.write(json.dumps(row) + "\n")
            kept += 1
            if args.limit and kept >= args.limit:
                break
    print(f"wrote {kept} rows -> {dst} (skipped {skipped})")


if __name__ == "__main__":
    main()
