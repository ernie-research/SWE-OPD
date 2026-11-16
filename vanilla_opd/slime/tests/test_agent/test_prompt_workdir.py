"""The turn-0 prompt must name the repo path the booted runtime actually uses.

The ``scale_swe`` ``instance_prompt`` bakes ``/testbed`` literally, but the repo
path is the swe_type's decision, not a config value: ``SWEBenchProDocker`` forces
``/app`` while ``SWEBenchDocker`` / ``SWEBenchMultilingualDocker`` keep
``/testbed`` and ``ScaleSWEDocker`` takes ``ds["workdir"]``. The reference harness
handles this by formatting the template with ``working_dir=env.runtime.repo_path``
and then calling ``runtime.post_process_for_template`` (agent.py:1416); slime
resolves the same value via ``eval_chat._instance_workdir`` and feeds it to the
shared ``r2egym.prompt.initial_messages``.

These tests pin the facts that make those two routes equivalent:

1. the scaffold's ``system_prompt`` contains no ``/testbed``, so rewriting only
   the instance prompt loses nothing (if a future scaffold yaml bakes the path
   into the system prompt, this fails and the rewrite must be widened);
2. each swe_type's ``repo_path`` is what ``_instance_workdir`` will read, and it
   is an ``__init__``-time value -- so it is already valid when ``run_instance``
   builds the messages right after ``_boot_env`` returns, before any pod I/O;
3. rendering with ``workdir=repo_path`` equals what that swe_type's
   ``post_process_for_template`` would produce.

The vendored ``docker.py`` is read via ``ast`` rather than imported: importing it
pulls in ``docker`` / ``kubernetes`` / ``rle_open_sdk`` / ``swebench`` /
``unidiff``, none of which a bare checkout has (and the repo's own top-level
``docker/`` directory shadows the pip SDK as a namespace package anyway). Static
resolution is sufficient because every value under test is a literal default or a
one-line ``.replace``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_GPUS = 0

_VENDORED = REPO_ROOT / "examples" / "coding_agent_opd" / "3rdparty" / "R2E-Gym" / "src" / "r2egym"
_YAML = _VENDORED / "agenthub" / "config" / "scale_swe" / "edit_fn_calling.yaml"
_DOCKER_PY = _VENDORED / "agenthub" / "runtime" / "docker.py"

# swe_type -> (runtime class, repo_path, class that owns post_process_for_template).
# Literals so a vendored-docker.py change that moves a repo path or drops an
# override fails loudly here instead of silently re-pointing every prompt.
_EXPECTED = {
    "swebench": ("SWEBenchDocker", "/testbed", "DockerRuntime"),
    "swe_bench_multilingual": ("SWEBenchMultilingualDocker", "/testbed", "DockerRuntime"),
    "swebench-pro": ("SWEBenchProDocker", "/app", "SWEBenchProDocker"),
}
_CASES = sorted(_EXPECTED.items())

_PROBLEM = "Something is broken.\nSee /testbed/foo.py for the failing case."


# --------------------------------------------------------------------------
# static view of the vendored runtime classes
# --------------------------------------------------------------------------
class _Classes:
    """Single-inheritance class table built from ``docker.py``'s AST."""

    def __init__(self, path: Path) -> None:
        self.table: dict[str, dict] = {}
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.ClassDef):
                continue
            methods = {
                c.name for c in node.body
                if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.table[node.name] = {
                "base": ast.unparse(node.bases[0]) if node.bases else None,
                "methods": methods,
                "repo_path": self._repo_path(node),
            }

    @staticmethod
    def _repo_path(node: ast.ClassDef) -> str | None:
        """``repo_path`` as this class fixes it: body override wins over default."""
        init = next(
            (c for c in node.body
             if isinstance(c, ast.FunctionDef) and c.name == "__init__"), None,
        )
        if init is None:
            return None
        for stmt in ast.walk(init):  # an unconditional `repo_path = "..."` override
            if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                    and isinstance(stmt.targets[0], ast.Name)
                    and stmt.targets[0].id == "repo_path"
                    and isinstance(stmt.value, ast.Constant)):
                return stmt.value.value
        names = [a.arg for a in init.args.args]
        if "repo_path" not in names:
            return None
        offset = len(names) - len(init.args.defaults)
        idx = names.index("repo_path")
        if idx < offset:
            return None
        default = init.args.defaults[idx - offset]
        return default.value if isinstance(default, ast.Constant) else None

    def _mro(self, cls: str):
        seen: set[str] = set()
        while cls in self.table and cls not in seen:
            seen.add(cls)
            yield cls
            cls = self.table[cls]["base"]

    def repo_path(self, cls: str) -> str | None:
        for name in self._mro(cls):
            value = self.table[name]["repo_path"]
            if value is not None:
                return value
        return None

    def method_owner(self, cls: str, method: str) -> str | None:
        for name in self._mro(cls):
            if method in self.table[name]["methods"]:
                return name
        return None


@pytest.fixture(scope="module")
def classes() -> _Classes:
    return _Classes(_DOCKER_PY)


@pytest.fixture(scope="module")
def prompts() -> tuple[str, str]:
    """``(system_prompt, instance_prompt)`` straight from the vendored yaml."""
    yaml = pytest.importorskip("yaml")
    cfg = yaml.safe_load(_YAML.read_text())
    return cfg["system_prompt"], cfg["instance_prompt"]


def _render(prompts: tuple[str, str], workdir: str) -> str:
    """What ``Scaffold.render_instance(problem, workdir)`` produces."""
    text = prompts[1].format(problem_statement=_PROBLEM)
    if workdir and workdir != "/testbed":
        text = text.replace("/testbed", workdir)
    return text


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_system_prompt_bakes_no_workdir(prompts):
    """Only the instance prompt needs the rewrite -- assert that stays true."""
    system = prompts[0]
    assert "/testbed" not in system
    assert "/app" not in system


@pytest.mark.parametrize("swe_type,expected", _CASES)
def test_runtime_repo_path_is_what_we_expect(classes, swe_type, expected):
    """``_instance_workdir`` reads ``runtime.repo_path``; pin what it will see."""
    cls, repo_path, _owner = expected
    assert cls in classes.table, f"{cls} vanished from the vendored docker.py"
    assert classes.repo_path(cls) == repo_path


@pytest.mark.parametrize("swe_type,expected", _CASES)
def test_post_process_owner(classes, swe_type, expected):
    """Which class supplies ``post_process_for_template`` -- Pro is the only override."""
    cls, _repo_path, owner = expected
    assert classes.method_owner(cls, "post_process_for_template") == owner


def test_pro_prompt_says_app(prompts, classes):
    """The regression this fixes: a Pro row must not be told /testbed."""
    pro = _render(prompts, classes.repo_path("SWEBenchProDocker"))
    assert "/testbed" not in pro
    assert "/app" in pro


@pytest.mark.parametrize("cls", ["SWEBenchDocker", "SWEBenchMultilingualDocker"])
def test_testbed_runtimes_are_untouched(prompts, classes, cls):
    """Verified / multilingual render byte-identically to the pre-fix behaviour."""
    rendered = _render(prompts, classes.repo_path(cls))
    assert rendered == prompts[1].format(problem_statement=_PROBLEM)
    assert "/testbed" in rendered
    assert "/app" not in rendered


def test_matches_pro_post_process(prompts):
    """Our rewrite == ``SWEBenchProDocker.post_process_for_template``'s.

    That override is literally ``s.replace("/testbed", "/app")`` on both prompts
    (docker.py:4545); applying it here proves the two routes agree on the exact
    bytes, including any ``/testbed`` inside the problem statement itself.
    """
    system, instance = prompts
    ref_system = system.replace("/testbed", "/app")
    ref_instance = instance.format(problem_statement=_PROBLEM).replace("/testbed", "/app")
    assert ref_system == system  # no-op: the system prompt has no /testbed
    assert _render(prompts, "/app") == ref_instance


def test_pro_post_process_body_is_still_a_plain_replace():
    """Guard the assumption test_matches_pro_post_process encodes."""
    source = _DOCKER_PY.read_text()
    assert (
        'return system_prompt.replace("/testbed", "/app"), '
        'instance_prompt.replace("/testbed", "/app")'
    ) in source


def test_instance_workdir_prefers_runtime_over_config():
    """``_instance_workdir`` reads the runtime, falling back to ``cfg.workdir``."""
    pytest.importorskip(
        "docker.models.containers",
        reason="pip 'docker' SDK absent; vendored r2egym runtime cannot import",
    )
    # The eval harness is not shipped with vanilla_opd (sibling offline_opd).
    pytest.importorskip(
        "examples.coding_agent_opd.eval.eval_chat",
        reason="eval harness is not shipped with vanilla_opd (see ../offline_opd)",
    )
    from examples.coding_agent_opd.eval.eval_chat import EvalConfig, _instance_workdir

    cfg = EvalConfig(url="http://x", model="m")

    class _Env:
        class runtime:  # noqa: N801 - stand-in for the vendored runtime object
            repo_path = "/app"

    class _NoRuntime:
        pass

    assert _instance_workdir(_Env(), cfg) == "/app"
    assert _instance_workdir(_NoRuntime(), cfg) == cfg.workdir
