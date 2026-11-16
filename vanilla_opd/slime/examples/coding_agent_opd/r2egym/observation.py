"""Observation role: thin adapter over R2E-Gym's ``Observation``.

R2E-Gym's ``env.step`` wraps every tool result in
``Observation(bash_output, error_code, action)`` and the model sees
``str(observation)``. That ``__str__`` encodes several harness-critical
behaviours we must reproduce exactly for train/eval consistency:

  * empty function call -> ``CONTINUE_MSG`` ("You forgot to use a function
    call ...")
  * submit / finish -> ``"<<< Finished >>>"``
  * execute_bash -> ``"Exit code: N\\nExecution output of [...]:\\n<body>"``
    with **16000-char middle truncation**
  * any other tool -> ``"Execution output of [...]:\\n<body>"``
  * ``scaffold``-specific variants (``scale_swe`` uses different messages)

We import ``Observation`` and use its ``__str__`` verbatim -- no reformatting.
The only slime-side plumbing is setting ``observation.scaffold`` from the
active :class:`Scaffold` so the right branch fires, and a named
``continue_message`` for the loop's "no valid call" path (which short-circuits
``env.step`` but must still show the byte-identical continue text).
"""

from __future__ import annotations

from r2egym.agenthub.observation.observation import Observation

from .action import Action
from .scaffold import Scaffold


def format(
    bash_output: str,
    error_code: int,
    action: Action,
    scaffold: Scaffold | None = None,
    num_lines: int = 40,
) -> str:
    """Return the model-facing observation text via ``Observation.__str__``.

    ``scaffold.name`` selects the ``__str__`` branch (``sweagent`` and the
    default empty scaffold both take the standard branch; ``scale_swe`` differs).
    Truncation / exit-code framing all live inside the vendored ``__str__``.
    """
    obs = Observation(bash_output, error_code, action, num_lines=num_lines)
    obs.scaffold = scaffold.name if scaffold is not None else ""
    return str(obs)


def continue_message(scaffold: Scaffold | None = None) -> str:
    """The observation shown when the model emits no valid function call.

    Mirrors R2E-Gym: an empty ``Action`` fed to ``Observation`` yields
    ``CONTINUE_MSG`` (or ``ScaleSWE_CONTINUE_MSG`` for the scale_swe scaffold).
    The token loop uses this instead of running an empty ``env.step``.
    """
    return format("", 0, Action("", {}), scaffold=scaffold)


__all__ = ["Observation", "format", "continue_message"]
