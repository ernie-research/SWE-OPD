"""Trajectory role: thin recorder over R2E-Gym's ``Trajectory`` / ``TrajectoryStep``.

R2E-Gym's ``Agent.run`` accumulates one :class:`TrajectoryStep` per tool turn
and, at the end, packs them into a :class:`Trajectory` (agent.py:695-741). That
``Trajectory`` is the object the R2E-Gym analysis/eval harness consumes: it
exposes ``pass_1``, ``create_swebench_submission()``, ``editor_files``,
``swebench_reasoning_trace()``, etc. We reuse those pydantic models verbatim so
our rollout produces the *same* trajectory artifact the harness expects.

The only slime-side plumbing is a :class:`Recorder` that mirrors the per-step
``TrajectoryStep(...)`` construction from ``Agent.run`` (step_idx = step_count-1,
``action.to_xml_string()``, ``str(observation)``, cumulative ``total_time_traj``)
and a ``finalize`` that stamps the budget from the active :class:`Scaffold`.
Nothing here touches loss_mask / tokens -- that stays in the token loop; this is
purely an out-of-band record for ``sample.metadata["trajectory"]`` and analysis.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from r2egym.agenthub.trajectory.trajectory import Trajectory, TrajectoryStep

from .action import Action
from .scaffold import DEFAULT_MAX_STEPS, DEFAULT_MAX_TOKEN_LIMIT, Scaffold

# R2E-Gym ``Agent.run`` time-limit defaults (agent.py:429-431). The Scaffold
# only carries the step/token budget; the wall-time limits are Agent.run
# constants, kept here so the recorded Trajectory matches the harness.
DEFAULT_MAX_LLM_TIME = 7200
DEFAULT_MAX_EXEC_TIME = 90
DEFAULT_MAX_TOTAL_TIME = 50000


@dataclass
class Recorder:
    """Accumulate ``TrajectoryStep``s and emit an R2E-Gym ``Trajectory``.

    Usage mirrors ``Agent.run``'s loop:

        rec = Recorder(scaffold, problem_statement, docker_image)
        ...per turn...
        rec.build_step(thought=..., action=<Action|None>, observation=<str>, ...)
        traj = rec.finalize(output_patch=..., exit_reason=..., reward=...)
    """

    scaffold: Scaffold
    problem_statement: str
    docker_image: str
    ds: dict | None = None
    exp_name: str | None = None

    steps: list[TrajectoryStep] = field(default_factory=list)
    _start: float = field(default_factory=time.monotonic)

    def build_step(
        self,
        *,
        thought: str,
        action: Action | None,
        observation: str,
        done: bool,
        info: dict | None = None,
        token_usage_prompt: int = 0,
        token_usage_completion: int = 0,
        token_usage_total: int = 0,
        llm_exec_time: float = 0.0,
        env_exec_time: float = 0.0,
        total_step_time: float = 0.0,
    ) -> TrajectoryStep:
        """Append one step, mirroring ``Agent.run``'s ``TrajectoryStep(...)``.

        ``step_idx`` is 0-based and ``step_count`` 1-based (agent.py:697/712).
        A ``None`` action (model emitted no valid call) is stored as the empty
        XML string, matching the empty-``Action`` path in the vendored harness.
        ``total_time_traj`` is cumulative wall time from recorder construction.
        """
        step_count = len(self.steps) + 1
        step = TrajectoryStep(
            step_idx=step_count - 1,
            thought=thought,
            action=action.to_xml_string() if action is not None else "",
            observation=observation,
            done=done,
            info=info or {},
            token_usage_prompt=token_usage_prompt,
            token_usage_completion=token_usage_completion,
            token_usage_total=token_usage_total,
            llm_exec_time=llm_exec_time,
            env_exec_time=env_exec_time,
            total_step_time=total_step_time,
            total_time_traj=time.monotonic() - self._start,
            step_count=step_count,
        )
        self.steps.append(step)
        return step

    def finalize(
        self,
        *,
        output_patch: str,
        exit_reason: str,
        reward: float | None = None,
        test_output: str | None = None,
        reward_calc_time: float | None = None,
    ) -> Trajectory:
        """Pack the accumulated steps into a :class:`Trajectory`.

        Budget fields come from the active :class:`Scaffold` (step/token) and
        the ``Agent.run`` wall-time constants; ``exit_reason`` should be one of
        the R2E-Gym vocabulary (``agent`` / ``max_step_limit`` /
        ``token_limit`` / ``traj_time_limit`` / ...). ``ds`` is required for the
        ``swebench_*`` properties and ``create_swebench_submission``.
        """
        return Trajectory(
            trajectory_steps=self.steps,
            problem_statement=self.problem_statement,
            docker_image=self.docker_image,
            exp_name=self.exp_name,
            ds=self.ds,
            max_steps=DEFAULT_MAX_STEPS,
            max_steps_absolute=self.scaffold.max_steps_absolute,
            max_token_limit=DEFAULT_MAX_TOKEN_LIMIT,
            max_llm_time=DEFAULT_MAX_LLM_TIME,
            max_exec_time=DEFAULT_MAX_EXEC_TIME,
            max_total_time=DEFAULT_MAX_TOTAL_TIME,
            exit_reason=exit_reason,
            output_patch=output_patch,
            reward=reward,
            test_output=test_output,
            reward_calc_time=reward_calc_time,
        )


def to_metadata(traj: Trajectory) -> dict[str, Any]:
    """Serialise a ``Trajectory`` for ``sample.metadata["trajectory"]``.

    Uses pydantic ``model_dump`` (JSON-safe) so ``visualize_rollouts.py`` and
    offline analysis can round-trip it via ``Trajectory.load_from_model_dump_json``.
    """
    return traj.model_dump(mode="json")


__all__ = ["Trajectory", "TrajectoryStep", "Recorder", "to_metadata"]
