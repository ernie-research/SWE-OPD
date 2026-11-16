"""slime glue package that structurally ports R2E-Gym's five roles
(Action / Observation / Runtime / Env / Trajectory) plus the two implicit
train/eval-consistency roles (Scaffold / Parser) into the coding_agent_opd
rollout.

Design contract (see the migration plan):

  * We do **not** ``pip install r2egym``. Instead the vendored source tree at
    ``examples/coding_agent_opd/3rdparty/R2E-Gym/src`` is inserted at the front
    of ``sys.path`` *before* the first ``import r2egym`` so it overrides any
    editable install (``other_infra/R2E-Gym/src`` via a ``.pth`` file). This
    makes any checkout reproducible.
  * The glue modules (``scaffold``/``action``/``observation``/``runtime``/
    ``env``/``trajectory``/``agent``) import the vendored classes directly and
    only add the thin async / token-level RL plumbing slime needs.

Importing this package has two side effects, both idempotent:

  1. ``3rdparty/R2E-Gym/src`` is prepended to ``sys.path`` and the resulting
     ``r2egym`` import is asserted to resolve into the vendored tree.
  2. RLE backend env-var defaults (``RLE_TOOL_TYPE`` / ``RLE_ENV``) are seeded
     to match ``slime.agent.sandbox_rle`` so the shared Runtime and the legacy
     sandbox agree. ``RLE_API_KEY`` is only passed through (never defaulted).

Downstream code should ``from examples.coding_agent_opd.r2egym import ...``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# 1. sys.path shim: vendored r2egym must win over any editable install.
# --------------------------------------------------------------------------
# .../examples/coding_agent_opd/r2egym/__init__.py -> .../coding_agent_opd
_OPD_DIR = Path(__file__).resolve().parent.parent
VENDORED_R2EGYM_SRC = _OPD_DIR / "3rdparty" / "R2E-Gym" / "src"
VENDORED_R2EGYM_ROOT = _OPD_DIR / "3rdparty" / "R2E-Gym"

if not (VENDORED_R2EGYM_SRC / "r2egym" / "__init__.py").is_file():
    raise ImportError(
        "vendored r2egym source not found at "
        f"{VENDORED_R2EGYM_SRC}; expected the in-repo R2E-Gym checkout under "
        "examples/coding_agent_opd/3rdparty/R2E-Gym/src."
    )

_src_str = str(VENDORED_R2EGYM_SRC)
if sys.path and sys.path[0] == _src_str:
    pass
else:
    # Drop any stale occurrence, then pin to the very front so the editable
    # install's .pth entry cannot shadow us.
    sys.path[:] = [p for p in sys.path if p != _src_str]
    sys.path.insert(0, _src_str)

# If r2egym was already imported earlier in this process from a *different*
# location (e.g. the editable install), the shim above is too late: sys.modules
# has cached the wrong module and every ``from r2egym...`` will keep resolving
# to it. Fail loudly rather than silently training/evaluating on a drifted copy.
_already = sys.modules.get("r2egym")
if _already is not None:
    _already_file = getattr(_already, "__file__", "") or ""
    if not _already_file.startswith(_src_str):
        raise ImportError(
            "r2egym was already imported from "
            f"{_already_file!r} before the vendored shim could run. "
            "Import examples.coding_agent_opd.r2egym before any other r2egym "
            "import (or uninstall the editable other_infra/R2E-Gym install)."
        )

import r2egym  # noqa: E402  (import intentionally after the sys.path shim)

_r2egym_file = getattr(r2egym, "__file__", "") or ""
if not _r2egym_file.startswith(_src_str):
    raise ImportError(
        f"r2egym resolved to {_r2egym_file!r}, not the vendored tree under "
        f"{_src_str!r}. An editable install is shadowing the vendored copy; "
        "uninstall it or fix sys.path ordering."
    )

# --------------------------------------------------------------------------
# 1b. vendored third_party shim (swebench et al).
# --------------------------------------------------------------------------
# ``runtime/docker.py`` imports ``trajectory.swebench_utils`` (which does
# ``from swebench.harness... import make_test_spec``) at module top *before* it
# appends ``<root>/third_party`` to sys.path a few lines later. That ordering
# works only if swebench is already importable. The vendored swebench lives at
# ``3rdparty/R2E-Gym/third_party/swebench``; prepend that dir here -- earlier
# than docker.py can -- so the runtime import chain resolves against the
# in-repo swebench (and multi_swe_bench) rather than needing a pip install.
_third_party_str = str(VENDORED_R2EGYM_ROOT / "third_party")
if (VENDORED_R2EGYM_ROOT / "third_party" / "swebench").is_dir():
    if _third_party_str not in sys.path:
        sys.path.insert(1, _third_party_str)

# --------------------------------------------------------------------------
# 2. RLE backend env defaults (align with slime.agent.sandbox_rle).
# --------------------------------------------------------------------------
# Only *default* the non-secret knobs; RLE_API_KEY is passed through untouched
# so a missing key surfaces as an explicit RuntimeError at pod-boot time.
os.environ.setdefault("RLE_TOOL_TYPE", "swe-pdc")
os.environ.setdefault("RLE_ENV", "test")

# --------------------------------------------------------------------------
# 3. huggingface_hub compat shim.
# --------------------------------------------------------------------------
# The vendored ``r2egym.agenthub.utils.utils`` does
# ``from huggingface_hub import ... HfFolder`` at module top. ``HfFolder`` was
# removed in huggingface_hub >= 1.0 (this env ships 1.23). The symbol is a
# *dead import* in the vendored file (never referenced), and it only ever
# mattered for HF Hub dataset uploads -- never on the train/eval rollout path.
# Rather than pin/downgrade a shared dep or edit the vendored tree, inject a
# harmless stub so the import resolves. Idempotent; no-op on older hub versions.
import huggingface_hub as _hf  # noqa: E402

if not hasattr(_hf, "HfFolder"):

    class _HfFolderStub:  # pragma: no cover - never instantiated on our path
        """Placeholder for the removed huggingface_hub.HfFolder (unused)."""

    _hf.HfFolder = _HfFolderStub

from .scaffold import Scaffold, load_scaffold  # noqa: E402
from .action import Action, guard_bashcmd, is_terminal, parse  # noqa: E402
from . import observation  # noqa: E402
from .runtime import POOL, AsyncRuntime, PodUnavailableError, run_in_pool  # noqa: E402
from .env import AsyncRepoEnv  # noqa: E402
from .trajectory import Recorder, Trajectory, TrajectoryStep  # noqa: E402

__all__ = [
    "VENDORED_R2EGYM_SRC",
    "VENDORED_R2EGYM_ROOT",
    "Scaffold",
    "load_scaffold",
    "Action",
    "parse",
    "is_terminal",
    "guard_bashcmd",
    "observation",
    "POOL",
    "AsyncRuntime",
    "AsyncRepoEnv",
    "PodUnavailableError",
    "run_in_pool",
    "Recorder",
    "Trajectory",
    "TrajectoryStep",
]
