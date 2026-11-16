"""The scale_swe fn-calling tool schemas must stay byte-identical to the reference eval.

In ``edit_fn_calling`` mode the ``tools=`` JSON array is rendered into the system
prompt's ``# Tools`` block by the chat template, so it is PROMPT content, not just
plumbing: a schema drift silently hands the model a different manual for the same
in-pod CLI. The reference benchmark
(``other_infra/R2E-Gym-benchmark/src/r2egym/agenthub/agent/agent.py:471-472``)
hardcodes ``[scale_swe_str_replace_editor_tool, scale_swe_execute_bash_tool,
scale_swe_finish_tool]`` for this scaffold; slime resolves the same three via
``r2egym.scaffold._SCAFFOLD_TOOL_ATTRS``.

``scale_swe_tools_expected.json`` is a verbatim copy of that harness's committed
``scale_swe-tools.json`` dump. It is checked in because the reference tree itself
is untracked; when the tree IS present we additionally assert the golden still
matches it, so the snapshot cannot silently rot.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_GPUS = 0

_GOLDEN = Path(__file__).resolve().parent / "scale_swe_tools_expected.json"
_REFERENCE = REPO_ROOT / "other_infra" / "R2E-Gym-benchmark" / "scale_swe-tools.json"


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    return json.loads(_GOLDEN.read_text())


@pytest.fixture(scope="module")
def scaffold():
    """The default scaffold (``scale_swe`` / ``edit_fn_calling``).

    ``load_scaffold`` reads the vendored yaml and resolves the tool schemas, so
    importing it also exercises the load-time ``command``-property contract check.
    """
    yaml = pytest.importorskip("yaml")  # noqa: F841
    from examples.coding_agent_opd.r2egym.scaffold import load_scaffold

    return load_scaffold()


def test_golden_matches_reference_dump(golden):
    """The checked-in snapshot still equals the reference harness's own dump."""
    if not _REFERENCE.is_file():
        pytest.skip(f"reference tree not present: {_REFERENCE}")
    assert golden == json.loads(_REFERENCE.read_text())


def test_advertised_tools_match_reference(scaffold, golden):
    """The exact array sent as ``tools=`` -- order, names, properties, descriptions."""
    assert scaffold.name == "scale_swe"
    assert scaffold.variant == "edit_fn_calling"
    assert scaffold.fn_calling is True
    # json round-trip normalises tuples/dict types without hiding content drift.
    assert json.loads(json.dumps(list(scaffold.tools))) == golden


def test_tools_cover_the_scaffold_command_files(scaffold):
    """One schema per installed CLI, in command_files order."""
    assert [t["function"]["name"] for t in scaffold.tools] == [
        p.stem for p in scaffold.command_files
    ]
    assert scaffold.allowed_tools == {"str_replace_editor", "execute_bash", "finish"}


def test_execute_bash_advertises_required_command_and_timeout(scaffold):
    """Regression guard for the generic schema leaking in.

    The generic ``execute_bash_tool`` has ``required: []`` and no ``timeout``; the
    scale_swe one requires ``command`` and offers ``timeout`` (which the in-pod CLI
    accepts as ``execute_bash <command> --timeout N``).
    """
    params = _params(scaffold, "execute_bash")
    assert params["required"] == ["command"]
    assert set(params["properties"]) == {"command", "timeout"}
    assert params["properties"]["timeout"]["type"] == "integer"


def test_finish_advertises_no_parameters(scaffold):
    """``finish`` takes nothing -- and must survive the contract check.

    ``_build_tools`` requires a ``command`` property on every non-terminal tool
    (``Action.to_bashcmd`` routes only that into ``argv[1]``). ``finish`` is exempt:
    ``to_bashcmd`` short-circuits it to ``echo '<<<Finished>>>'`` and ``RepoEnv.step``
    keys done-detection on ``function_name``, so an empty schema is correct here.
    A regression in that exemption makes ``load_scaffold()`` raise, failing the
    ``scaffold`` fixture rather than this assert.
    """
    params = _params(scaffold, "finish")
    assert params["properties"] == {}
    assert params["required"] == []


def test_str_replace_editor_command_enum(scaffold):
    """No ``undo_edit``: the in-pod CLI does not implement it."""
    params = _params(scaffold, "str_replace_editor")
    assert params["properties"]["command"]["enum"] == [
        "view",
        "create",
        "str_replace",
        "insert",
    ]
    assert params["required"] == ["command", "path"]


def _params(scaffold, name: str) -> dict:
    for tool in scaffold.tools:
        if tool["function"]["name"] == name:
            return tool["function"]["parameters"]
    raise AssertionError(f"{name!r} not advertised; got {[t['function']['name'] for t in scaffold.tools]}")


# --------------------------------------------------------------------------
# The off-schema tripwire (``r2egym.env._check_params``) reads these schemas, so
# swapping them changes what it flags. Two consequences are worth pinning:
#   * ``execute_bash`` gained ``required: ["command"]`` -> a call that omits it is
#     now reported (it used to pass, since the generic schema had ``required: []``).
#   * ``finish`` lost all properties -> every checkpoint trained before the swap
#     still emits ``finish(command="submit")``, which must stay silent: it is
#     provably inert (``to_bashcmd`` -> ``echo '<<<Finished>>>'``), and under
#     ``SLIME_AGENT_STRICT_TOOL_ARGS=1`` a report would abort the run.
# --------------------------------------------------------------------------


@pytest.fixture()
def check_params(monkeypatch):
    """``_check_params`` with a fresh counter, returning the reported signatures."""
    env = pytest.importorskip("examples.coding_agent_opd.r2egym.env")
    action_mod = pytest.importorskip("r2egym.agenthub.action")
    monkeypatch.setattr(env, "_OFF_SCHEMA_SEEN", Counter())

    def _run(scaffold, name: str, **params):
        env._check_params(action_mod.Action(function_name=name, parameters=params), scaffold)
        return dict(env._OFF_SCHEMA_SEEN)

    return _run


def test_old_schema_finish_call_is_not_flagged(check_params, scaffold):
    """``finish(command="submit")`` is inert, so the tripwire must stay quiet."""
    from r2egym.agenthub.action import Action

    assert Action(function_name="finish", parameters={"command": "submit"}).to_bashcmd() == (
        "echo '<<<Finished>>>'"
    )
    assert check_params(scaffold, "finish", command="submit") == {}


def test_execute_bash_without_command_is_flagged(check_params, scaffold):
    """``required: ["command"]`` is what makes the ``cmd`` regression visible."""
    reported = check_params(scaffold, "execute_bash", cmd="ls")
    assert reported == {("execute_bash", ("cmd",), ("command",)): 1}
