"""Scaffold: the single source of truth for prompts / tools / parser / budget.

R2E-Gym's ``config/scale_swe/<variant>.yaml`` (loaded by ``AgentArgs.from_yaml``)
binds together everything that must stay identical between training and
evaluation. We support both scale_swe variants:

  * ``edit_non_fn_calling`` -- ``system_prompt`` describes the 3 tools + the XML
    ``<function=...>`` call format inline; observations come back as *user*
    messages; ``response_parser=parse_xml_response`` parses the model's raw text.
  * ``edit_fn_calling`` -- ``system_prompt`` does *not* describe the tools; the
    chat template injects a ``# Tools`` block from the ``tools=`` JSON schemas we
    pass, the model emits Qwen-native ``<tool_call>{json}</tool_call>``, and
    observations come back as *tool*-role messages. ``response_parser`` is
    ``parse_tool_call_response`` (used by the message/chat transport, which reads
    structured ``tool_calls``); the token-in transport parses the native
    ``<tool_call>`` text instead (see ``action.parse``).

Common to both:

  * ``instance_prompt`` -- wraps the PR/problem statement (bakes ``/testbed``)
  * ``command_files`` -- the tool CLIs to install (the 3-tool set)
  * budget -- ``max_steps_absolute`` (the hard turn cap; lives in ``Agent.run``'s
    signature, not the yaml)

The variant is chosen by ``SLIME_AGENT_SCAFFOLD_VARIANT`` (default
``edit_fn_calling``). The training token-in loop, the eval token-in loop, and the
eval OpenAI-chat loop all obtain prompts / tools / budget from a single
``Scaffold`` so the harness the policy is trained against and the harness it is
scored against are byte-for-byte the same.

We deliberately do **not** import ``r2egym.agenthub.agent.agent.AgentArgs``:
that module pulls in litellm / openai / anthropic / ``response_helper`` at
import time. ``from_yaml`` is trivial, so we replicate its (only) behaviour --
the ``response_parser`` default rule -- here instead.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import VENDORED_R2EGYM_ROOT

# Default scaffold family + variant. The variant is env-overridable so a single
# run can flip between the Qwen-native fn-calling transport and the inline-XML
# non-fn-calling transport without a code change.
DEFAULT_SCAFFOLD_NAME = "scale_swe"
DEFAULT_SCAFFOLD_VARIANT = os.environ.get(
    "SLIME_AGENT_SCAFFOLD_VARIANT", "edit_fn_calling"
).strip()

# --------------------------------------------------------------------------
# R2E-Gym Agent.run() budget defaults (see agent/agent.py::Agent.run signature:
# max_steps=10, max_steps_absolute=50, max_token_limit=65536). Only
# ``max_steps_absolute`` (the hard turn cap) is enforced by the train/eval loops;
# ``max_steps`` (soft "steps remaining" reminder) and ``max_token_limit`` are not
# consumed by either live loop, so they are NOT Scaffold fields. The two constants
# below are kept solely so ``trajectory.py``'s Recorder can stamp the vendored
# ``Trajectory`` metadata with the canonical R2E-Gym values.
# --------------------------------------------------------------------------
DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_STEPS_ABSOLUTE = 100
DEFAULT_MAX_TOKEN_LIMIT = 1024*128

# The sweagent instance_prompt bakes this workdir literally. render_instance()
# only rewrites it when the caller's workdir differs, preserving byte-identity
# with the vendored harness for the common /testbed case.
DEFAULT_WORKDIR = "/testbed"

_CONFIG_DIR = VENDORED_R2EGYM_ROOT / "src" / "r2egym" / "agenthub" / "config"


@dataclass(frozen=True)
class Scaffold:
    """Immutable bundle of the prompt/tool/parser/budget contract."""

    name: str  # scaffold family, e.g. "scale_swe"
    variant: str  # yaml stem, e.g. "edit_fn_calling"
    system_prompt: str
    instance_prompt: str
    command_files: tuple[Path, ...]
    response_parser: str
    fn_calling: bool = False
    # OpenAI-style tool JSON schemas for the fn-calling transport (``None`` for
    # non-fn-calling). Passed as ``tools=`` to ``apply_chat_template`` (token-in)
    # and to the ``/v1/chat/completions`` request (chat transport) so the model
    # is advertised the same tools the eval harness uses.
    tools: tuple[dict, ...] | None = None
    max_steps_absolute: int = DEFAULT_MAX_STEPS_ABSOLUTE

    @property
    def allowed_tools(self) -> frozenset[str]:
        """Tool-name whitelist = the stems of the scaffold's command_files.

        For ``scale_swe`` this is ``{str_replace_editor, execute_bash, finish}``.
        """
        return frozenset(p.stem for p in self.command_files)

    def render_instance(
        self, problem_statement: str, workdir: str = DEFAULT_WORKDIR
    ) -> str:
        """Render the user (instance) prompt for a problem.

        Mirrors ``Agent.run``: ``instance_prompt.format(problem_statement=...)``.
        Extra ``.format`` kwargs the vendored harness passes (gt_patch,
        working_dir, ...) are unused by the sweagent template, so we only bind
        ``problem_statement``. When ``workdir`` differs from the baked
        ``/testbed`` we rewrite it afterwards so the model is pointed at the
        real repo path.
        """
        text = self.instance_prompt.format(problem_statement=problem_statement or "")
        if workdir and workdir != DEFAULT_WORKDIR:
            text = text.replace(DEFAULT_WORKDIR, workdir)
        return text


def _resolve_command_files(rel_paths: list[str]) -> tuple[Path, ...]:
    """Resolve yaml ``command_files`` (relative to the R2E-Gym root) to
    absolute vendored paths, asserting each exists."""
    resolved: list[Path] = []
    for rel in rel_paths:
        p = (VENDORED_R2EGYM_ROOT / rel.lstrip("./")).resolve()
        if not p.is_file():
            raise FileNotFoundError(
                f"scaffold command_file not found: {rel!r} -> {p}"
            )
        resolved.append(p)
    return tuple(resolved)


# Which vendored schema each scaffold family advertises, mirroring the reference
# harness's hardcoded branch (R2E-Gym-benchmark ``agent.py:463-474``:
# ``elif "scale_swe" in self.scaffold: tools = [scale_swe_*_tool]``).
#
# An explicit table is REQUIRED, not a stylistic preference: schema dicts are
# keyed by ``function.name``, and the scale_swe trio collides with the generic
# trio on all three names (``str_replace_editor`` / ``execute_bash`` / ``finish``
# each exist twice in ``r2egym.agenthub.tools``). A ``dir()``-order scan therefore
# picks a winner alphabetically, i.e. by accident. The schemas are rendered into
# the system prompt's ``# Tools`` block by the chat template, so picking the wrong
# one silently ships a different prompt: the model is handed another scaffold's
# manual for the same in-pod CLI (scale_swe's ``execute_bash`` advertises
# ``required: ["command"]`` plus a ``timeout`` knob; the generic one advertises
# ``required: []`` and no timeout).
_SCAFFOLD_TOOL_ATTRS: dict[str, dict[str, str]] = {
    "scale_swe": {
        "str_replace_editor": "scale_swe_str_replace_editor_tool",
        "execute_bash": "scale_swe_execute_bash_tool",
        "finish": "scale_swe_finish_tool",
    },
}

# Fallback for scaffold families with no entry above: ``execute_bash`` has TWO
# generic vendored schemas -- ``execute_bash_tool`` (parameter ``command``) and
# ``r2egym_bash_execute_tool`` (parameter ``cmd``) -- and alphabetical dir() order
# lets the ``cmd`` one win. See the contract check in _build_tools for what that
# costs.
_COLLIDING_SCHEMAS = {"execute_bash": "execute_bash_tool"}

# Terminal tools are exempt from the ``command``-property contract check below:
# ``Action.to_bashcmd`` short-circuits ``finish``/``submit`` to
# ``echo '<<<Finished>>>'`` (action.py:131-132) and never builds an argv for them,
# and ``RepoEnv.step``'s done-detection keys on ``action.function_name`` (env.py:233),
# not on any parameter. The reference's ``scale_swe_finish_tool`` accordingly
# advertises ``properties: {}``, so requiring ``command`` there would reject the
# very schema we are aligning to.
TERMINAL_TOOLS = frozenset({"finish", "submit"})


def _build_tools(name: str, command_files: tuple[Path, ...]) -> tuple[dict, ...]:
    """Map the scaffold's command_files to the vendored OpenAI tool schemas.

    Single source of truth: rather than hand-copy the JSON schemas, we reuse the
    schema dicts defined in ``r2egym.agenthub.tools`` (the same ones the vendored
    fn-calling harness sends to litellm). ``name`` selects the family's schema set
    via :data:`_SCAFFOLD_TOOL_ATTRS`; families with no entry fall back to a
    ``function.name``-keyed scan. This keeps the ``# Tools`` block the student sees
    byte-identical to the eval harness. Imported lazily so the (chardet-pulling)
    tools package is only loaded on the fn-calling path.
    """
    from r2egym.agenthub import tools as _vendored_tools

    explicit = _SCAFFOLD_TOOL_ATTRS.get(name)
    if explicit is not None:
        registry: dict[str, dict] = {
            fn_name: getattr(_vendored_tools, attr)
            for fn_name, attr in explicit.items()
        }
    else:
        registry = {}
        for attr in dir(_vendored_tools):
            obj = getattr(_vendored_tools, attr)
            if isinstance(obj, dict) and obj.get("type") == "function":
                fn_name = (obj.get("function") or {}).get("name")
                if fn_name:
                    registry[fn_name] = obj
        for fn_name, attr in _COLLIDING_SCHEMAS.items():
            registry[fn_name] = getattr(_vendored_tools, attr)

    selected: list[dict] = []
    for p in command_files:
        stem = p.stem
        if stem not in registry:
            raise KeyError(
                f"no vendored tool schema found for command_file {stem!r}; "
                f"available: {sorted(registry)}"
            )
        selected.append(registry[stem])

    # Contract check, at load time -- before any pod is booted or any token is spent.
    # Every advertised NON-TERMINAL tool MUST expose a ``command`` property:
    # ``Action.to_bashcmd`` routes ONLY ``parameters["command"]`` into the positional
    # slot each tool CLI reads from ``argv[1]`` (tools/scale_swe/execute_bash.py:212).
    # A schema that advertises some other name makes every call of that tool degrade
    # into ``<tool> --<key> '...'`` -> ``argv[1] == "--<key>"`` -> the pod answers
    # ``/bin/sh: 0: Illegal option --`` (exit 2). Nothing downstream rejects that, so
    # it reads as an ordinary failed command: the ``cmd`` variant of execute_bash cost
    # iter_0000059 93% of its trajectories before anyone noticed. Raise (not assert --
    # asserts vanish under python -O) so a schema/scaffold mismatch is impossible to
    # ship silently. Terminal tools are skipped (see :data:`TERMINAL_TOOLS`): they
    # never reach the argv builder, so the check does not apply to them.
    for tool in selected:
        fn = tool["function"]
        if fn["name"] in TERMINAL_TOOLS:
            continue
        props = (fn.get("parameters") or {}).get("properties") or {}
        if "command" not in props:
            raise ValueError(
                f"tool schema {fn['name']!r} advertises parameters {sorted(props)} but "
                "not 'command'. Action.to_bashcmd would emit it as a --flag and the "
                "pod's CLI (argv[1]) would fail with \"/bin/sh: 0: Illegal option --\". "
                "Pick the command-shaped schema in _SCAFFOLD_TOOL_ATTRS."
            )
    return tuple(selected)


def load_scaffold(
    name: str = DEFAULT_SCAFFOLD_NAME,
    variant: str = DEFAULT_SCAFFOLD_VARIANT,
    *,
    max_steps_absolute: int = DEFAULT_MAX_STEPS_ABSOLUTE,
) -> Scaffold:
    """Load the vendored scaffold yaml into a :class:`Scaffold`.

    ``name``/``variant`` select ``config/<name>/<variant>.yaml``; ``variant``
    defaults to ``$SLIME_AGENT_SCAFFOLD_VARIANT`` (``edit_fn_calling`` if unset).
    ``max_steps_absolute`` (the hard turn cap enforced by both loops) is the single
    budget knob and defaults to the R2E-Gym ``Agent.run`` value. When the variant
    is fn-calling we also resolve the tool JSON schemas so the chat template / chat
    request can advertise them.
    """
    yaml_path = _CONFIG_DIR / name / f"{variant}.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"scaffold yaml not found: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text())

    # Replicate AgentArgs.from_yaml's only nontrivial behaviour: the parser
    # default depends on the yaml filename (fn-calling vs non-fn-calling).
    fn_calling = yaml_path.name.endswith("edit_fn_calling.yaml")
    if fn_calling:
        response_parser = config.get("response_parser", "parse_tool_call_response")
    else:
        response_parser = config.get("response_parser", "parse_xml_response")

    command_files = _resolve_command_files(config.get("command_files", []))
    tools = _build_tools(name, command_files) if fn_calling else None

    return Scaffold(
        name=name,
        variant=variant,
        system_prompt=config["system_prompt"],
        instance_prompt=config["instance_prompt"],
        command_files=command_files,
        response_parser=response_parser,
        fn_calling=fn_calling,
        tools=tools,
        max_steps_absolute=max_steps_absolute,
    )
