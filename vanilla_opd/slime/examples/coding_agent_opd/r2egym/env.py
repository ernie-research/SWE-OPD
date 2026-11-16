"""Env role: async wrapper over R2E-Gym's ``RepoEnv``.

``RepoEnv`` (vendored ``environment/env.py``) is the synchronous orchestrator
that owns a ``DockerRuntime`` and exposes the interactive tool loop
(``add_commands`` / ``step`` / ``compute_reward_with_output``). Construction
boots the pod, and every method makes blocking RLE calls, so we wrap it and run
each call on the shared :data:`~.runtime.POOL`.

Design notes:
  * The interactive rollout pod is *this* ``RepoEnv``. Grading may instead use a
    fresh :class:`~.runtime.AsyncRuntime` (no-cheat) -- both share ``POOL``.
  * ``step`` re-renders the observation through :mod:`~.observation` so the
    active :class:`~.scaffold.Scaffold` selects the ``__str__`` branch; this is
    the same rendering path the eval chat loop uses (train/eval consistency).
  * ``add_tools`` installs the scaffold's ``command_files`` via
    ``RepoEnv.add_commands`` (replaces the old ``install_r2egym_tools``); the
    tool CLIs come from the vendored ``agenthub/tools`` tree.
"""

from __future__ import annotations

import logging
import os
from collections import Counter
from typing import Any

# Package import first: runs the sys.path shim + RLE env defaults.
from . import observation as _observation  # noqa: F401
from .runtime import run_in_pool
from .scaffold import TERMINAL_TOOLS, Scaffold

from r2egym.agenthub.action import Action  # noqa: E402
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv  # noqa: E402

logger = logging.getLogger(__name__)

# SLIME_AGENT_STRICT_TOOL_ARGS=1 turns an off-schema tool call into a hard error
# instead of a warning. Off by default: model output is untrusted input, and one bad
# generation must not kill a training run -- but a smoke test / CI job wants it fatal.
_STRICT_TOOL_ARGS = os.environ.get("SLIME_AGENT_STRICT_TOOL_ARGS", "0") == "1"

# {(scaffold.name, scaffold.variant): {tool: (allowed_keys, required_keys)}}
_PARAM_SCHEMA_CACHE: dict[tuple[str, str], dict[str, tuple[frozenset[str], frozenset[str]]]] = {}
# Per-signature occurrence counter so the log carries a rate, not a flood.
_OFF_SCHEMA_SEEN: Counter = Counter()


def _param_schema(scaffold: Scaffold) -> dict[str, tuple[frozenset[str], frozenset[str]]]:
    """``{tool_name: (allowed_keys, required_keys)}`` from the ADVERTISED schemas."""
    key = (scaffold.name, scaffold.variant)
    cached = _PARAM_SCHEMA_CACHE.get(key)
    if cached is None:
        cached = {}
        for tool in scaffold.tools or ():
            fn = tool.get("function") or {}
            params = fn.get("parameters") or {}
            cached[fn.get("name") or ""] = (
                frozenset(params.get("properties") or {}),
                frozenset(params.get("required") or ()),
            )
        _PARAM_SCHEMA_CACHE[key] = cached
    return cached


def _check_params(action: Action, scaffold: Scaffold) -> None:
    """Report parameter keys that disagree with the advertised tool schema.

    Silent degradation is the failure mode: an off-schema key is rejected NOWHERE
    downstream -- ``to_bashcmd`` turns it into ``--key 'value'`` and the pod's CLI
    answers ``/bin/sh: 0: Illegal option --`` (exit 2), which the model reads as an
    ordinary failed command and retries. That is how ``cmd`` burned 93% of
    iter_0000059's trajectories with a 1.5% resolve rate and no error anywhere.
    This check is the tripwire: either the model drifted off the schema, or the
    schema does not match the tool (see scaffold._build_tools).

    Extra keys on a TERMINAL tool are exempt (see scaffold.TERMINAL_TOOLS):
    ``to_bashcmd`` short-circuits ``finish``/``submit`` to ``echo '<<<Finished>>>'``
    and never builds an argv for them, so their parameters are provably inert --
    the degradation this tripwire guards cannot happen there. This matters right
    now: the aligned ``scale_swe_finish_tool`` advertises ``properties: {}`` while
    the schema it replaced made ``command`` required, so every checkpoint trained
    before the switch still emits ``finish(command="submit")``. Without the
    exemption that harmless call reads as ``unknown={'command'}`` on every
    trajectory -- and under ``SLIME_AGENT_STRICT_TOOL_ARGS=1`` it would abort the
    run. Missing *required* keys are still reported for terminal tools.
    """
    schema = _param_schema(scaffold).get(action.function_name)
    if schema is None:  # unknown tool: RepoEnv.step rejects it on its own
        return
    allowed, required = schema
    keys = set(action.parameters or {})
    unknown = set() if action.function_name in TERMINAL_TOOLS else keys - allowed
    missing = required - keys
    if not unknown and not missing:
        return
    sig = (action.function_name, tuple(sorted(unknown)), tuple(sorted(missing)))
    _OFF_SCHEMA_SEEN[sig] += 1
    seen = _OFF_SCHEMA_SEEN[sig]
    msg = (
        f"off-schema tool call #{seen}: {action.function_name}("
        f"unknown={sorted(unknown)} missing_required={sorted(missing)}); "
        f"advertised={sorted(allowed)} required={sorted(required)}. "
        "Either the model is not following the advertised schema or the schema does "
        "not match the tool -- check r2egym.scaffold._build_tools."
    )
    if _STRICT_TOOL_ARGS:
        raise ValueError(msg)
    if seen == 1 or seen % 100 == 0:
        logger.warning(msg)


# --------------------------------------------------------------------------
# Grading infra-failure detection (shared by train generate.py + eval_chat.py).
# --------------------------------------------------------------------------
# ``compute_reward`` (-> the swe_type's ``_calculate_reward``) can *soft*-fail:
# instead of raising, it returns the RLE/grading error text AS the ``test_output``
# and scores reward=0. That 0 is a grading artifact (the tests never ran), NOT a
# real "unresolved" -- masking it as resolved=False silently depresses the score
# on eval and, worse, feeds a false-negative reward=0 training signal. These
# signatures are the soft returns that do not raise; a hard SDK/transport error
# raises ``PodUnavailableError`` and is handled separately by each caller.
GRADING_INFRA_FAILURE_SIGNATURES = {
    "The command took too long to execute (>": "grading_timeout",
    "RLE session_exec failed after retries": "rle_exec_failed",
    "RLE session_exec returned invalid type": "rle_exec_bad_type",
    "RLE session_exec missing return_code": "rle_exec_no_return_code",
}


def grading_failure_reason(reward: float, test_output: str | None) -> str | None:
    """Return an infra-failure label if grading did not complete, else ``None``.

    Only fires on a non-passing grade (``reward < 1.0``): a passing grade proves
    the grading command ran to completion, so a stray signature there is moot.
    On a non-passing grade whose ``test_output`` carries one of the grading
    infra-failure strings, the 0 reward is a grading artifact (the tests never
    ran), not a real "unresolved" -- we return the matched label as the reason.
    """
    if reward >= 1.0 or not test_output:
        return None
    for signature, label in GRADING_INFRA_FAILURE_SIGNATURES.items():
        if signature in test_output:
            return label
    return None


GRADING_NO_TEST_RESULT_SIGNATURES = {
    "[GRADING_NO_TEST_RESULTS]": "no_parseable_test_results",
    "[SWE_PRO_REWARD_PARSE_ERROR]": "pro_output_json_unparseable",
}


def grading_no_test_results_reason(reward: float, test_output: str | None) -> str | None:
    """Return a "graded 0 but nothing was measured" label, else ``None``.

    Same ``reward >= 1.0`` short-circuit as :func:`grading_failure_reason` (a
    passing grade proves results were parsed). Unlike that function, the returned
    label must NOT drive ``status`` / exclusion -- callers record it alongside the
    row so "how many rows were never actually graded" is answerable from
    ``results.jsonl`` alone.
    """
    if reward >= 1.0 or not test_output:
        return None
    for signature, label in GRADING_NO_TEST_RESULT_SIGNATURES.items():
        if signature in test_output:
            return label
    return None


def _make_repo_env(
    ds: dict,
    *,
    backend: str,
    verbose: bool,
    step_timeout: int,
    reward_timeout: int,
    rle_env_session_id: str | None,
    pod_cpu: int | str | None,
    pod_memory: int | str | None,
) -> RepoEnv:
    """Construct ``RepoEnv`` (boots the pod); run inside the pool."""
    return RepoEnv(
        EnvArgs(ds=ds),
        backend=backend,
        verbose=verbose,
        step_timeout=step_timeout,
        reward_timeout=reward_timeout,
        rle_env_session_id=rle_env_session_id,
        pod_cpu=pod_cpu,
        pod_memory=pod_memory,
    )


class AsyncRepoEnv:
    """Async facade over a single vendored ``RepoEnv`` (the interactive pod)."""

    def __init__(self, env: RepoEnv, scaffold: Scaffold | None) -> None:
        self._env = env
        self._scaffold = scaffold

    @property
    def env(self) -> RepoEnv:
        return self._env

    @property
    def runtime(self) -> Any:
        return self._env.runtime

    @classmethod
    async def create(
        cls,
        ds: dict,
        *,
        scaffold: Scaffold | None = None,
        backend: str = "rle",
        verbose: bool = False,
        step_timeout: int = 90,
        reward_timeout: int = 300,
        rle_session_id: str | None = None,
        pod_cpu: int | str | None = None,
        pod_memory: int | str | None = None,
    ) -> "AsyncRepoEnv":
        """Boot the interactive pod and return the wrapper (blocks in the pool)."""
        env = await run_in_pool(
            _make_repo_env,
            ds,
            backend=backend,
            verbose=verbose,
            step_timeout=step_timeout,
            reward_timeout=reward_timeout,
            rle_env_session_id=rle_session_id,
            pod_cpu=pod_cpu,
            pod_memory=pod_memory,
        )
        return cls(env, scaffold)

    async def add_tools(self, cmd_files: list[str]) -> None:
        """Install tool CLIs into the pod via ``RepoEnv.add_commands``.

        ``cmd_files`` are absolute paths (typically ``scaffold.command_files``);
        each is copied to ``/usr/local/bin`` and chmod/sourced by the vendored
        parser so ``run_action`` accepts the corresponding function names.
        """
        await run_in_pool(self._env.add_commands, list(cmd_files))

    async def step(self, action: Action, *, timeout: int | None = None) -> tuple[str, bool]:
        """Execute ``action`` in the pod; return ``(observation_text, done)``.

        Delegates dispatch + done-detection to ``RepoEnv.step`` (which validates
        the action against the installed commands and flags submit/finish), then
        renders the observation through :mod:`~.observation` so the active
        scaffold picks the correct ``__str__`` branch.

        ``cmd`` is normalised to ``command`` first. This is the ONE choke point both
        loops share (training parses the raw text itself, the eval takes the server's
        structured ``tool_calls``, and both end here), and ``Action.to_bashcmd`` only
        puts ``parameters["command"]`` in the positional slot the tool CLIs read
        (tools/scale_swe/execute_bash.py:212). A ``cmd`` key would therefore be emitted
        as ``execute_bash --cmd '...'`` -> ``argv[1] == "--cmd"`` -> the pod answers
        ``/bin/sh: 0: Illegal option --`` (exit 2) and every command silently fails.
        Checkpoints trained while the advertised schema still said ``cmd``
        (``r2egym_bash_execute_tool``, see scaffold._build_tools) keep that habit --
        iter_0000059 emitted ``cmd`` in 3279/3279 calls -- so accept the synonym
        instead of losing the whole trajectory.
        """
        params = action.parameters or {}
        _check_params(action, self._scaffold)
        if "command" not in params and "cmd" in params:
            action = Action(
                function_name=action.function_name,
                parameters={("command" if k == "cmd" else k): v for k, v in params.items()},
            )
        obs, _reward, done, _info = await run_in_pool(
            self._env.step, action, timeout
        )
        text = _observation.format(
            obs.bash_output,
            obs.error_code,
            obs.action,
            scaffold=self._scaffold,
            num_lines=obs.num_lines,
        )
        return text, bool(done)

    async def get_patch(self) -> str:
        """Return the model's git diff from the interactive pod."""
        return await run_in_pool(self._env.runtime.get_patch)

    async def compute_reward(
        self, *, timeout: int | None = None
    ) -> tuple[float, str]:
        """Grade the current pod state; return ``(reward, test_output)``.

        Wraps ``RepoEnv.compute_reward_with_output`` (-> the swe_type's
        ``_calculate_reward(get_test_output=True)``), which resets tests and
        applies the dataset's *test* patch on top of whatever is on the pod.
        """
        return await run_in_pool(
            self._env.compute_reward_with_output, timeout
        )

    async def close(self) -> None:
        """Tear down the interactive pod (best-effort; never raises)."""
        try:
            await run_in_pool(self._env.close)
        except Exception:  # noqa: BLE001  (teardown must not mask real errors)
            pass


__all__ = [
    "AsyncRepoEnv",
    "EnvArgs",
    "RepoEnv",
    "GRADING_INFRA_FAILURE_SIGNATURES",
    "GRADING_NO_TEST_RESULT_SIGNATURES",
    "grading_failure_reason",
    "grading_no_test_results_reason",
]
