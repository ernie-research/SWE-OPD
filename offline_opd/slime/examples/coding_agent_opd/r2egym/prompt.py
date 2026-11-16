"""Prompt role: reuse ``Agent.run``'s message construction, render via chat template.

R2E-Gym has **no token/ChatML code of its own**: ``Agent.run`` is entirely
message-level (``list[{role, content}]`` dicts) fed to litellm, and the model
*server* applies the chat template. The non-fn-calling ``sweagent`` scaffold we
use appends observations as plain ``{"role": "user", "content": str(obs)}``
messages (``agent.py:647-648``) -- *not* tool-role ``<tool_response>`` wrapping
(that is the fn-calling branch, ``agent.py:618-633``).

So "reuse the r2e-gym code block" means: replicate ``Agent.run``'s message-dict
construction here (:func:`initial_messages` / :func:`observation_message`,
mirroring ``agent.py:502-505`` + ``:522-531`` + ``:645-648``) and render it with
the tokenizer's official ``apply_chat_template`` -- the same renderer the model
server would run. We never hand-roll ChatML markup.

Both the training loop (``generate.py``) and the eval loop
(``eval/eval_chat.py``) import this module, so the harness the policy is
trained against and the one it is scored against are byte-for-byte identical.

TITO bridge (the one manual step): a chat template starts the next turn with a
boundary token, while our verbatim token-out stream may *already* end in that
same token. Qwen uses ``<|im_end|>`` for both; GLM uses the next role token
(``<|user|>`` or ``<|observation|>``). :func:`render_turn_delta` therefore drops
the delta's leading boundary only when it equals the last token already emitted
by the model, so the boundary occurs exactly once. Everything else is produced
by the official template, not by string surgery.

Why we never re-render the assistant content: the model (Qwen3) emits
``<think>...</think>`` blocks, and Qwen3.5's template *auto-opens* ``<think>\\n``
at every generation prompt while rendering a *historical* assistant turn
differently (empty ``<think>`` block before a ``tool`` response, or the block
stripped entirely before a ``user`` query). So round-tripping the real assistant
text through the template would not be verbatim, and the generation-prompt tail
is never a prefix of the historical render. We do not need it to be: the
assistant's tokens are already in ``sample.tokens`` verbatim (that is the point
of TITO). :func:`render_turn_delta` therefore renders a think-free sentinel in
the assistant slot and slices off *everything up to and including* the sentinel,
keeping only the *structural* connector (assistant close + observation turn +
next generation prompt). That tail depends only on message roles, not on
assistant content, so the empty-vs-real ``<think>`` difference is irrelevant.
"""

from __future__ import annotations

from typing import Any

from .scaffold import Scaffold

# Probe the connector once per process. This flag flips after the first
# successful probe so the hot path pays nothing thereafter.
_eos_alignment_checked = False


def ensure_tito_eos_alignment(tokenizer: Any) -> None:
    """Validate that the template yields a tokenized turn connector.

    :func:`render_turn_delta` derives the turn connector by rendering a sentinel
    assistant turn and slicing off everything up to and including the sentinel;
    what remains must contain the assistant-to-observation boundary. Its first
    token is dropped only when the verbatim stream already ends in that exact
    token. This covers Qwen's shared ``<|im_end|>`` separator and GLM's
    role-specific ``<|user|>`` / ``<|observation|>`` separators.

    We probe both observation roles the scaffold can use -- ``user`` (non-fn) and
    ``tool`` (fn-calling) -- because Qwen's template renders the historical
    assistant turn differently in each (a following user query resets
    ``last_query_index`` and strips the turn's ``<think>`` block, a following tool
    response does not). In both cases the connector must be non-empty; otherwise
    there is no safe token-level delta to append.
    """
    global _eos_alignment_checked
    if _eos_alignment_checked:
        return
    for obs_role in ("user", "tool"):
        full = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": _ASSISTANT_SENTINEL},
                {"role": obs_role, "content": "o"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        idx = full.find(_ASSISTANT_SENTINEL)
        if idx == -1:
            raise ValueError(
                "chat template did not render the assistant turn verbatim "
                f"(sentinel missing, obs_role={obs_role}); cannot bridge TITO"
            )
        connector = full[idx + len(_ASSISTANT_SENTINEL):]
        ids = tokenizer(connector, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(
                "chat template produced an empty assistant-to-observation "
                f"connector (obs_role={obs_role}); cannot bridge TITO"
            )
    _eos_alignment_checked = True


def initial_messages(
    scaffold: Scaffold, problem_statement: str, workdir: str
) -> list[dict]:
    """Build the turn-0 message history (``agent.py:502-505``).

    ``[system, user(instance)]``. No stepcount reminder is baked in (the vendored
    ``Steps Remaining:`` postpend is deliberately dropped from our scaffold).
    """
    instance = scaffold.render_instance(problem_statement, workdir)
    return [
        {"role": "system", "content": scaffold.system_prompt},
        {"role": "user", "content": instance},
    ]


def observation_message(scaffold: Scaffold, obs_text: str) -> dict:
    """Build the per-step observation turn.

    * fn-calling: a ``tool``-role message (Qwen's template wraps the content in
      ``<tool_response>...</tool_response>``), matching the fn-calling harness.
    * non-fn-calling: a plain ``user``-role message, matching the sweagent path.

    No stepcount reminder is postpended (deliberately dropped from our scaffold).
    """
    role = "tool" if scaffold.fn_calling else "user"
    return {"role": role, "content": obs_text}


def render_prompt_tokens(
    tokenizer: Any, messages: list[dict], tools: tuple[dict, ...] | None = None
) -> tuple[list[int], str]:
    """Render ``messages`` to ``(token_ids, text)`` via the official chat template.

    ``add_generation_prompt=True`` so the text ends at the assistant open marker,
    ready for the model to continue. When ``tools`` is given (fn-calling), it is
    passed to ``apply_chat_template`` so the template injects the ``# Tools``
    block -- byte-identical to the eval harness. This is the single entry point
    for the turn-0 prompt in both the training and eval loops.
    """
    text = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return list(ids), text


# A think-free sentinel that stands in for the current assistant turn when
# deriving the structural connector (see render_turn_delta). It must contain no
# <think> block (so Qwen3's template does not strip it) and no chat-template
# markup (so it round-trips as a single opaque span we can slice off).
_ASSISTANT_SENTINEL = "\x00ASSISTANT_TURN\x00"


def render_turn_delta(
    tokenizer: Any,
    prompt_messages: list[dict],
    obs_message: dict,
    *,
    last_token_id: int | None,
    tools: tuple[dict, ...] | None = None,
) -> list[int]:
    """Return the token delta to append after the model's verbatim assistant tokens.

    ``prompt_messages`` is the history *before* this turn's assistant reply (the
    same messages that produced the current ``sample.tokens`` prefix). We render::

        full = apply_chat_template(
            prompt_messages + [assistant(SENTINEL), obs], add_generation_prompt=True)

    and take everything *after* the :data:`_ASSISTANT_SENTINEL` occurrence. That
    tail is the connector -- the assistant-closing separator, the observation
    turn, and the next generation prompt (``<|im_start|>assistant\\n<think>\\n``)
    -- and it is content-independent: the ``<tool_response>`` grouping and turn
    separators depend only on message *roles*, never on the assistant's text.

    Anchoring on the sentinel (rather than diffing against a ``base`` render that
    ends in the generation prompt) is what makes this work on Qwen3.5, whose
    template auto-opens ``<think>\\n`` at the generation prompt but renders a
    historical assistant turn differently: with an (empty) ``<think>\\n\\n</think>``
    block when the next message is a ``tool`` response, or with the ``<think>``
    block stripped entirely when the next message is a ``user`` query. Neither
    shape is a prefix of the generation-prompt tail, so the old ``base``-prefix
    diff was unusable; slicing after the sentinel sidesteps it because the real
    (verbatim) assistant tokens are already in ``sample.tokens`` and only the
    structural connector is needed here.

    The lone TITO fix-up: if the verbatim stream already ended in the same token
    that opens the connector, drop the connector copy so the boundary occurs
    exactly once. For Qwen it is ``<|im_end|>``; for GLM it is a role marker.

    Raises ``ValueError`` if the sentinel is not rendered or the connector is
    empty; the caller converts that to an aborted+retried sample.
    """
    full_messages = prompt_messages + [
        {"role": "assistant", "content": _ASSISTANT_SENTINEL},
        obs_message,
    ]
    full_text = tokenizer.apply_chat_template(
        full_messages, tools=tools, tokenize=False, add_generation_prompt=True
    )
    idx = full_text.find(_ASSISTANT_SENTINEL)
    if idx == -1:
        raise ValueError(
            "chat template did not render the assistant turn where expected "
            "(sentinel missing)"
        )
    connector_text = full_text[idx + len(_ASSISTANT_SENTINEL):]
    ids = tokenizer(connector_text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(
            "chat template produced an empty assistant-to-observation connector; "
            "render_turn_delta cannot bridge TITO"
        )
    if last_token_id is not None and ids[0] == last_token_id:
        # The verbatim token-out stream already emitted the connector's first
        # boundary token; keep the remainder but drop the duplicate.
        ids = ids[1:]
    return list(ids)


__all__ = [
    "ensure_tito_eos_alignment",
    "initial_messages",
    "observation_message",
    "render_prompt_tokens",
    "render_turn_delta",
]
