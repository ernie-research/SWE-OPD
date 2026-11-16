"""Action role: thin adapter over R2E-Gym's ``Action`` + parser registry.

We import the vendored ``Action`` and ``load_parser`` directly -- no hand-rolled
prompt text or regex (the previous ``actions.py`` drifted from them). The only
slime-side additions are:

  * ``parse`` -- run the scaffold's ``response_parser`` and normalise the
    "no call" case (the vendored parsers return ``Action(function_name="")``)
    into ``None`` so callers have one thing to check.
  * ``is_terminal`` -- submit/finish end the loop.
  * ``guard_bashcmd`` -- enforce the scaffold's 3-tool whitelist (which the
    vendored ``sweagent`` observation path does *not* validate) before mapping
    to a shell command via ``Action.to_bashcmd()``.
"""

from __future__ import annotations

import json
import re

from r2egym.agenthub.action import Action
from r2egym.agenthub.agent.parsers import load_parser

from slime.agent.parsing import parse_model_output

from .scaffold import Scaffold

# submit (sweagent) / finish (r2egym) both end the interaction; the loop exits
# and captures the git diff instead of shelling out.
_TERMINAL_FUNCTIONS = frozenset({"submit", "finish"})

_DEFAULT_PARSER = "parse_xml_response"

# The Qwen3-Coder chat template makes the model emit tool calls as XML --
# ``<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n</function>\n</tool_call>``
# -- NOT Hermes JSON. On the token-in path there is no server-side tool parsing
# (that lives in the OpenAI /v1/chat/completions layer, not /generate), so we must
# parse the raw text ourselves. We reuse the SAME primitive coding_agent_rl uses
# (``slime.agent.parsing.parse_model_output``): SGLang's ``FunctionCallParser``
# driven by the ROLLOUT server's ``--tool-call-parser`` (e.g. ``qwen3_coder``),
# with an XML ``<function=..>`` fallback. This produces the same ``Action`` the
# vendored parsers do, so downstream (``guard_bashcmd`` / ``Action.to_bashcmd``)
# is identical across both transports.

# Legacy Hermes-JSON fallback: older checkpoints wrapped calls as
# ``<tool_call>\n{json}\n</tool_call>``. Kept as a last resort so switching back
# to a JSON-emitting checkpoint keeps parsing with this same function.
_JSON_TOOL_CALL_RE = re.compile(r"(?s)<tool_call>\s*(\{.*?\})\s*</tool_call>")


def parse_qwen_tool_call(
    text: str,
    *,
    tools: tuple[dict, ...] | None = None,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
) -> tuple[str, Action]:
    """Parse the ROLLOUT model's tool-call text into ``(thought, Action)``.

    Delegates the format-specific work to ``slime.agent.parsing.parse_model_output``
    (SGLang ``FunctionCallParser`` + XML ``<function=..>`` fallback) -- the same
    parsing coding_agent_rl uses -- and folds the first tool use into an ``Action``.

    IMPORTANT: ``tool_call_parser`` / ``reasoning_parser`` must be the *rollout*
    (student) server's parsers, NOT the teacher's. The text parsed here is the
    student's own generation; the teacher server (a different model / architecture)
    is only queried for token logprobs and never for tool-call parsing. The XML
    fallback needs ``tools`` (the scaffold's OpenAI tool schemas); when it is
    ``None`` no tool parsing runs.

    On a missing / malformed call we return ``Action("", {})`` (empty function
    name), which :func:`parse` normalises to ``None`` -- exactly how the vendored
    parsers signal "no well-formed call".
    """
    text = text or ""
    parsed = parse_model_output(
        text,
        tools_schema=list(tools) if tools else None,
        tool_parser_name=tool_call_parser,
        reasoning_parser_name=reasoning_parser,
    )
    if parsed.tool_uses:
        tu = parsed.tool_uses[0]
        args = tu.get("input")
        thought = (parsed.reasoning or parsed.text or "").strip()
        return thought, Action(
            function_name=tu.get("name") or "",
            parameters=args if isinstance(args, dict) else {},
        )

    # Last-resort legacy Hermes-JSON fallback.
    m = _JSON_TOOL_CALL_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1))
        except Exception:
            obj = None
        if isinstance(obj, dict):
            args = obj.get("arguments")
            return text[: m.start()].strip(), Action(
                function_name=obj.get("name") or "",
                parameters=args if isinstance(args, dict) else {},
            )

    return (parsed.reasoning or parsed.text or text).strip(), Action(
        function_name="", parameters={}
    )


def parse(
    text: str,
    scaffold: Scaffold | None = None,
    *,
    tool_call_parser: str | None = None,
    reasoning_parser: str | None = None,
) -> tuple[str, Action | None]:
    """Parse model text into ``(thought, action)`` using the scaffold's parser.

    On the fn-calling token-in path we parse the model's tool-call text ourselves
    (:func:`parse_qwen_tool_call`) -- the scaffold's ``parse_tool_call_response``
    only works on OpenAI SDK objects. ``tool_call_parser`` / ``reasoning_parser``
    are the *rollout* server's parsers (``args.sglang_tool_call_parser`` /
    ``args.sglang_reasoning_parser``); pass ``None`` to rely on the XML fallback.
    Otherwise we run the scaffold's text parser (``parse_xml_response``). Returns
    ``action=None`` when no well-formed function call is present (empty
    ``function_name``); callers then surface the scaffold's continue-message.
    ``thought`` is retained for trajectory recording (S6).
    """
    if scaffold is not None and scaffold.fn_calling:
        thought, action = parse_qwen_tool_call(
            text or "",
            tools=scaffold.tools,
            tool_call_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
        )
    else:
        parser_name = (
            scaffold.response_parser if scaffold is not None else _DEFAULT_PARSER
        )
        thought, action = load_parser(parser_name)(text or "")
    if not isinstance(action, Action) or not action.function_name:
        return thought, None
    return thought, action


def is_terminal(action: Action | None) -> bool:
    """True when the action ends the loop (``submit`` / ``finish``)."""
    return action is not None and action.function_name in _TERMINAL_FUNCTIONS


def guard_bashcmd(action: Action, scaffold: Scaffold) -> str:
    """Whitelist-check ``action`` then map it to a shell command.

    Raises ``ValueError`` (surfaced to the model as an observation) when the
    function is empty or not in the scaffold's ``allowed_tools``. Terminal
    actions are handled by the loop before dispatch, so they should not reach
    here; if one does, ``Action.to_bashcmd`` maps it to a harmless echo.
    """
    name = action.function_name
    if not name:
        raise ValueError("no function name in action")
    if name not in scaffold.allowed_tools:
        available = ", ".join(sorted(scaffold.allowed_tools))
        raise ValueError(
            f"'{name}' is not a valid function. Available functions: {available}"
        )
    return action.to_bashcmd()


__all__ = [
    "Action",
    "parse",
    "parse_qwen_tool_call",
    "is_terminal",
    "guard_bashcmd",
]
