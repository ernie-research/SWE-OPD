"""Chat-completions SWE eval for coding_agent_opd, driving generate.py's harness.

Scores an OpenAI-compatible ``/v1/chat/completions`` endpoint on a SWE dataset
(e.g. ``data/swebench-latest.jsonl``) by driving the *same* sweagent tool loop
the training rollout uses: identical scaffold prompts, budget, stepcount, action
parsing (``r2egym.action``), observation formatting (``r2egym.observation``) and
the interactive / grading pod (``utils.rollout._boot_env`` + ``env.compute_reward``).

Transport -- why chat, not token-in (TITO):
  The training rollout is token-in/token-out: it chat-templates the running
  sequence *locally* (with the trained model's own tokenizer) and POSTs the token
  ids to sglang. That is only byte-identical when the eval tokenizer matches the
  served model. Here the served endpoint is an *external* model (Qwen3.5-9B, vocab
  ~248k) whose tokenizer we do not have -- feeding it locally-built Qwen3-8B token
  ids produced garbage. So this eval sends **messages** and lets the *server* apply
  its own chat template with its own tokenizer, exactly like R2E-Gym's own eval.
  Everything above the transport (scaffold, parser, observation, grading) is still
  the shared harness, so the interaction protocol is identical; only tokenization
  moves server-side.

Usage: see ``run_eval.sh``. Reward >= 1.0 counts as resolved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
import traceback
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import aiohttp

# Import the r2egym glue package first (runs the sys.path shim) via utils.rollout.
from examples.coding_agent_opd.utils.rollout import _boot_env, _missing_ds_field
from examples.coding_agent_opd.r2egym import action as _action
from examples.coding_agent_opd.r2egym import observation as _observation
from examples.coding_agent_opd.r2egym import prompt as _prompt
from examples.coding_agent_opd.r2egym.env import grading_failure_reason
from examples.coding_agent_opd.r2egym.scaffold import DEFAULT_WORKDIR, Scaffold, load_scaffold

from r2egym.agenthub.runtime.docker import PodUnavailableError  # noqa: E402

logger = logging.getLogger("coding_agent_opd.eval")


@dataclass
class EvalConfig:
    """Everything the chat eval needs. Sampling defaults follow the upstream
    R2E-Gym full eval (temperature=0.6 / top_p=0.95 top-level; top_k / min_p /
    presence_penalty / repetition_penalty in the request body); the ``.sh``
    wrapper and CLI override them. No local tokenizer: the server tokenizes."""

    url: str  # OpenAI-compatible /v1/chat/completions endpoint
    model: str
    # sampling (top-level params)
    temperature: float = 0.6
    top_p: float = 0.95
    # Per-turn generation cap sent as ``max_tokens``. **0 (the default) omits the
    # key**, so sglang falls back to its own ceiling
    # ``max_req_len - input_len - 1`` == ``context_len - input_len - 2``
    # (``scheduler.init_req_max_new_tokens``): every turn gets ALL the remaining
    # context instead of a fixed slice. A fixed slice was the single largest
    # source of lost score -- 22% of SWE-bench-verified died to one runaway turn
    # hitting 16384 -- and an overlong turn no longer ends the trajectory anyway
    # (see ``_run_chat_loop``). Set a positive value only to reproduce the old
    # fixed-cap metric.
    max_tokens: int = 0
    # context window: the loop stops once the server-reported total token usage
    # reaches it (the server enforces its own limit; this is a soft early stop).
    max_model_len: int = 131072
    # loop budget: the hard turn cap enforced by the chat loop (the upstream
    # R2E-Gym full eval uses 300 for full SWE eval, far above the training default of 50).
    max_steps_absolute: int = 300
    # extra non-top-level sampling params merged into the request body
    # (the upstream R2E-Gym eval's EXTRA_SAMPLING_PARAMS): top_k / min_p / penalties.
    extra_sampling: dict[str, Any] = field(default_factory=dict)
    # HTTP: this endpoint is a slow reasoning model, so a generous per-request
    # timeout + retries; concurrency caps in-flight instances.
    request_timeout: int = 1200
    retry: int = 3
    # per-trajectory wall-clock ceiling (seconds); the chat loop aborts a
    # trajectory once its total elapsed exceeds this, capping the worst-case
    # duration a single instance can consume even under a retry storm. 0 disables.
    max_total_time: int = 5400
    concurrent: int = 16
    workdir: str = DEFAULT_WORKDIR
    # observability: per-turn interaction logging (think / tool calls / observation,
    # mirroring visualize_rollouts.py) + how much of each field to show. The chat
    # transport already returns the assistant turn as text; observations are text
    # out of env.step -- nothing is detokenized.
    log_turns: bool = False
    log_max_chars: int = 400
    # heartbeat: every N seconds log aggregate progress + in-flight instances so a
    # long-running sweep is visibly alive (0 disables).
    heartbeat_interval: int = 60


# --------------------------------------------------------------------------
# Per-turn interaction logging (parsing mirrors visualize_rollouts.py so the
# live eval log reads the same as an offline rollout dump). We already hold the
# assistant text + observation text at each turn, so nothing is detokenized.
# --------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)
_CHATML_RE = re.compile(r"<\|im_(?:start|end)\|>")


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return f"{text[:limit]}  … [+{len(text) - limit} chars]"


def _inline_reasoning(reasoning: str, content: str) -> str:
    """Fold the model's chain-of-thought back INTO the assistant ``content`` as
    ``{reasoning}\\n</think>\\n\\n{content}`` -- the exact on-the-wire shape the
    R2E-Gym golden trajectory stores (reasoning inline, terminated by a lone
    ``</think>`` whose opening tag the generation prompt supplied).

    The running ``messages`` list is BOTH the replayable transcript AND the body
    re-posted each turn. Re-feeding the reasoning this way (rather than dropping
    it) matches training TITO, which carries the raw generation -- ``<think>``
    included -- forward in the token stream. It is also what makes a thinking
    model keep reasoning on EVERY turn: the server template splits ``content`` on
    ``</think>`` and re-wraps prior turns in ``<think>`` blocks, so the model sees
    a history that reasons each step and continues to do so. Dropping the
    reasoning (the earlier plan A) suppressed thinking after turn 0 AND diverged
    from the training distribution; sending it via a separate ``reasoning_content``
    field does NOT work on this endpoint (verified) -- only the inline form does.
    """
    if not reasoning:
        return content
    return f"{reasoning}\n</think>\n\n{content}"


def _parse_model_turn(text: str) -> tuple[str | None, list[tuple[str, dict]], str]:
    """Split a model turn into (think, [(fn, params)...], leftover) -- see
    visualize_rollouts._parse_model_turn. Qwen3 thinking models emit a leading
    ``<think>`` whose opening tag the server strips, so also treat a lone
    ``</think>`` prefix as a thought delimiter for display."""
    think_m = _THINK_RE.search(text)
    if think_m:
        think = think_m.group(1).strip()
    elif "</think>" in text:
        think = text.split("</think>", 1)[0].strip()
    else:
        think = None
    calls: list[tuple[str, dict]] = []
    for fm in _FUNC_RE.finditer(text):
        fn = fm.group(1).strip()
        params = {k.strip(): v for k, v in _PARAM_RE.findall(fm.group(2))}
        calls.append((fn, params))
    leftover = _THINK_RE.sub("", text)
    if "</think>" in leftover:
        leftover = leftover.split("</think>", 1)[1]
    leftover = _FUNC_RE.sub("", leftover)
    leftover = _CHATML_RE.sub("", leftover).strip()
    return think, calls, leftover


def _log_model_turn(
    tag: str, step_num: int, text: str, cfg: EvalConfig, reasoning: str = "",
    tool_calls: list[dict] | None = None,
) -> None:
    think, calls, leftover = _parse_model_turn(text)
    # This deployment returns the chain-of-thought in a separate response field
    # (``reasoning``) with an empty ``content``, so ``_parse_model_turn`` finds no
    # inline <think>; prefer the server-supplied reasoning when present.
    if reasoning:
        think = reasoning
    # fn-calling: the actual tool call arrives as a STRUCTURED ``tool_calls`` list
    # (function.name + JSON arguments), not inline ``<function=..>`` text, so
    # ``_parse_model_turn`` found none -- decode them here so run.log shows the
    # real command (e.g. ``str_replace_editor  command=view  path=/testbed``).
    for tc in tool_calls or []:
        fn_obj = tc.get("function") or {}
        name = fn_obj.get("name") or "?"
        raw = fn_obj.get("arguments")
        try:
            args = json.loads(raw) if isinstance(raw, str) else (raw or {})
        except Exception:
            args = {"<unparsed>": raw}
        if not isinstance(args, dict):
            args = {"<args>": args}
        calls.append((name, args))
    lines = [f"🤖 [{tag}] MODEL turn {step_num}"]
    if think:
        lines.append(f"  💭 think: {_truncate(think, cfg.log_max_chars)}")
    if leftover:
        lines.append(f"  {_truncate(leftover, cfg.log_max_chars)}")
    for fn, params in calls:
        head = "🏁 submit" if fn.strip() in ("submit", "finish") else f"🔧 {fn}"
        # one-line command summary + per-arg detail below it
        summary = "  ".join(f"{k}={_truncate(str(v), 80)}" for k, v in params.items())
        lines.append(f"  {head}  {summary}" if summary else f"  {head}")
        for k, v in params.items():
            lines.append(f"     {k} = {_truncate(str(v), cfg.log_max_chars)}")
    if not (think or leftover or calls):
        lines.append(f"  {_truncate(text.strip(), cfg.log_max_chars)}")
    logger.info("\n".join(lines))


def _log_observation(tag: str, obs_text: str, done: bool, cfg: EvalConfig) -> None:
    marker = "🏁 done (submit/finish)" if done else "🖥️  OBSERVATION"
    logger.info("  %s [%s]\n%s", marker, tag, _truncate(obs_text or "", cfg.log_max_chars))


def _log_step_header(tag: str, step_num: int, ctx_tokens: int) -> None:
    """Emit a STEP separator between turns (mirrors the R2E-Gym native
    interaction log's ``===== STEP N =====`` divider), so a multi-turn
    trajectory is visually segmented per step in run.log."""
    bar = "=" * 25
    logger.info("%s STEP %d %s  (ctx≈%d tok)", bar, step_num, bar, ctx_tokens)


def _log_initial_prompt(tag: str, messages: list[dict], cfg: EvalConfig) -> None:
    """Log the turn-0 SYSTEM + USER instruction the model receives for this
    instance (mirrors the reference log's ``SYSTEM PROMPT`` / ``USER PROMPT``
    sections). The system prompt is scaffold-wide (already banner-logged once),
    but the USER instruction is per-instance and was otherwise never in run.log.
    """
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        label = "🧩 SYSTEM PROMPT" if role == "system" else (
            "📝 USER PROMPT" if role == "user" else f"[{role}]"
        )
        logger.info("%s [%s]\n%s\n%s\n%s",
                    label, tag, "─" * 78, _truncate(content, cfg.log_max_chars), "─" * 78)


@dataclass
class _Progress:
    """Shared, single-thread-mutated eval progress for the heartbeat + START/END
    logs. ``running`` maps in-flight instance_id -> start timestamp."""

    total: int
    done: int = 0
    resolved: int = 0
    submitted: int = 0
    errors: int = 0
    running: dict[str, float] = field(default_factory=dict)
    status_counts: Counter = field(default_factory=Counter)


async def _heartbeat(progress: _Progress, interval: int) -> None:
    """Periodically log aggregate progress + the oldest in-flight instance, so a
    long sweep is visibly alive even while every worker is mid-generation."""
    t0 = time.time()
    while True:
        await asyncio.sleep(interval)
        running = progress.running
        detail = ""
        if running:
            now = time.time()
            iid, ts = max(running.items(), key=lambda kv: now - kv[1])
            detail = f"  oldest={iid}({now - ts:.0f}s)"
        logger.info(
            "💓 [heartbeat] elapsed=%.0fs  done=%d/%d  resolved=%d (%.2f) submitted=%d "
            "err=%d  in-flight=%d%s",
            time.time() - t0, progress.done, progress.total, progress.resolved, 
            progress.resolved / max(progress.done, 1),
            progress.submitted, progress.errors, len(running), detail,
        )


async def _chat_completion_once(
    session: aiohttp.ClientSession,
    cfg: EvalConfig,
    messages: list[dict],
    tools: tuple[dict, ...] | None = None,
    timeout_override: float | None = None,
) -> tuple[str, list[dict], str, dict, str]:
    """POST ``messages`` to /v1/chat/completions; return (content, tool_calls, finish, usage, reasoning).

    Retries transient failures up to ``cfg.retry`` times. Reads the assistant
    turn from ``choices[0].message.content`` / ``.tool_calls`` and
    ``choices[0].finish_reason`` (defaulting to ``"stop"`` when the server omits
    it), plus the ``usage`` block for token-budget accounting. When ``tools`` is
    given (fn-calling), it is sent so the server can return structured
    ``tool_calls``; ``tool_calls`` is ``[]`` on the non-fn-calling path.

    Thinking models return the chain-of-thought in one of two shapes; both are
    handled here and surfaced as ``reasoning`` so the caller can fold it back into
    the assistant ``content`` (:func:`_inline_reasoning`) for the transcript:
      * a SEPARATE response field (this deployment uses ``reasoning``;
        Qwen/vLLM builds use ``reasoning_content``) with a clean ``content``; or
      * INLINE in ``content`` terminated by ``</think>`` -- split off the head
        as ``reasoning`` and keep the tail as ``content`` (Qwen-template style).
    The reasoning is re-fed inline each turn (matches training TITO / the golden
    trajectory), NOT dropped -- see :func:`_inline_reasoning`.
    """
    body = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "top_p": cfg.top_p,
        **cfg.extra_sampling,
    }
    # ``max_tokens <= 0`` -> omit the key so the server grants the whole remaining
    # window (context_len - input_len - 2) for this turn; see EvalConfig.max_tokens.
    if cfg.max_tokens > 0:
        body["max_tokens"] = cfg.max_tokens
    if tools is not None:
        body["tools"] = list(tools)
    last_err: Exception | None = None
    req_timeout = timeout_override if timeout_override else cfg.request_timeout
    for attempt in range(cfg.retry):
        try:
            async with session.post(
                cfg.url,
                json=body,
                timeout=aiohttp.ClientTimeout(total=req_timeout),
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
            choice = data["choices"][0]
            msg = choice.get("message") or {}
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
            # Two deployments surface the chain-of-thought differently:
            #   (a) a SEPARATE ``reasoning``/``reasoning_content`` field
            #       -- handled above; ``content`` is the clean answer.
            #   (b) INLINE in ``content`` terminated by ``</think>``
            #       -- ``reasoning`` is null and ``content`` holds "<cot></think>\n\nanswer"
            #       (the open ``<think>`` is elided by the server). Split on ``</think>``
            #       exactly like the Qwen chat_template: reasoning = text before it,
            #       content = text after. Only do this when field (a) was empty so the
            #       two paths never double-count.
            if not reasoning and "</think>" in text:
                head, _, tail = text.partition("</think>")
                reasoning = head.rsplit("<think>", 1)[-1].strip("\n")
                text = tail.lstrip("\n")
            tool_calls = msg.get("tool_calls") or []
            finish = choice.get("finish_reason") or "stop"
            usage = data.get("usage") or {}
            return text, tool_calls, finish, usage, reasoning
        except Exception as e:  # noqa: BLE001 (retry any transport/HTTP error)
            last_err = e
            # A deterministic HTTP 400 will not succeed on replay: e.g. when a
            # degenerate tool-call loop grew the prompt past ``max_model_len``
            # (``prompt + max_tokens`` exceeds the context window) the server
            # rejects it identically every retry. Fail fast and surface it rather
            # than burning ``cfg.retry`` backoff cycles. Narrowed to exactly 400
            # (not the whole 4xx range) so other 4xx — 408 request-timeout, 409,
            # 429 rate-limit, transient 5xx, timeouts, transport errors — all stay
            # retryable below. ``resp.raise_for_status()`` above raises
            # ``aiohttp.ClientResponseError`` carrying the HTTP ``.status``.
            if isinstance(e, aiohttp.ClientResponseError) and e.status == 400:
                raise
            # asyncio/aiohttp ``TimeoutError`` (and a few transport errors) carry
            # NO message, so ``str(e)`` is an empty string -- logging it verbatim
            # produces a blank tail ("... failed: TimeoutError: "). Spell out that
            # the message is empty AND why: a bare TimeoutError here means the POST
            # exceeded ``aiohttp.ClientTimeout(total=cfg.request_timeout)`` (the
            # server took too long to return the completion), not a connect failure.
            detail = str(e).strip()
            if not detail:
                if isinstance(e, asyncio.TimeoutError):
                    detail = (
                        f"<empty exception message> request exceeded "
                        f"ClientTimeout(total={cfg.request_timeout}s): server took "
                        f">{cfg.request_timeout}s to return the completion "
                        f"(slow/overloaded endpoint under concurrency, not a "
                        f"connect error)"
                    )
                else:
                    detail = f"<empty exception message> repr={e!r}"
            logger.warning(
                "[eval] chat attempt %d/%d failed: %s: %s",
                attempt + 1, cfg.retry, type(e).__name__, detail[:300],
            )
            await asyncio.sleep(2 * (attempt + 1))
    assert last_err is not None
    raise last_err


class FnCallingUnsupportedError(RuntimeError):
    """The configured endpoint rejected a request carrying ``tools=``.

    Raised by :func:`_probe_fn_calling_support` so an ``edit_fn_calling`` sweep
    against a model server with no tool-calling support fails fast with a clear
    message instead of every instance erroring mid-loop.
    """


async def _probe_fn_calling_support(
    session: aiohttp.ClientSession, cfg: EvalConfig, scaffold: Scaffold
) -> None:
    """Fail fast if the endpoint does not accept the fn-calling ``tools`` field.

    A no-op for non-fn-calling scaffolds. Otherwise sends one minimal chat
    request carrying ``tools=scaffold.tools``; if the server rejects it (HTTP
    error / transport failure) we raise :class:`FnCallingUnsupportedError` so the
    caller can exit with a clear message rather than let every instance fail
    deep inside the loop.
    """
    if not scaffold.fn_calling or not scaffold.tools:
        return
    body = {
        "model": cfg.model,
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "tools": list(scaffold.tools),
    }
    try:
        async with session.post(
            cfg.url,
            json=body,
            timeout=aiohttp.ClientTimeout(total=cfg.request_timeout),
        ) as resp:
            if resp.status >= 400:
                text = (await resp.text())[:300]
                raise FnCallingUnsupportedError(
                    f"endpoint {cfg.url!r} (model {cfg.model!r}) rejected a "
                    f"request with tools= (HTTP {resp.status}): {text}. The "
                    f"deployed model does not support function calling; use "
                    f"SLIME_AGENT_SCAFFOLD_VARIANT=edit_non_fn_calling or point "
                    f"at a fn-calling-capable endpoint."
                )
    except FnCallingUnsupportedError:
        raise
    except Exception as e:  # noqa: BLE001 (transport failure => treat as unsupported)
        raise FnCallingUnsupportedError(
            f"endpoint {cfg.url!r} (model {cfg.model!r}) failed a fn-calling "
            f"probe request with tools=: {type(e).__name__}: {str(e)[:200]}. "
            f"The deployed model likely does not support function calling; use "
            f"SLIME_AGENT_SCAFFOLD_VARIANT=edit_non_fn_calling or point at a "
            f"fn-calling-capable endpoint."
        ) from e


# Stand-in for the content of a turn cut off by the generation ceiling: the raw
# text is dropped from ``messages`` (it would otherwise occupy the whole remaining
# context) and replaced by this note, which also tells the model why its turn
# vanished. The observation that follows is the scaffold's usual "no valid tool
# execution" continue message.
_OVERLONG_ELIDED = (
    "[This turn was cut off by the generation limit after {tokens} tokens without "
    "producing a complete tool call. Its content has been discarded to free up "
    "context. Be concise and issue a single, complete tool call now.]"
)


def _chat_observation_message(
    scaffold: Scaffold,
    obs_text: str,
    *,
    tool_call_id: str | None = None,
    tool_name: str | None = None,
) -> dict:
    """Build a protocol-valid observation for the chat-completions transport.

    A ``tool`` message is only legal when it answers a preceding structured
    assistant ``tool_calls`` entry and carries that entry's ``tool_call_id``.
    The model can instead emit no call, a malformed call, or an inline XML call
    that the local fallback parser handles.  None of those cases has a server
    supplied call id, so representing its observation as ``role=tool`` creates
    an orphan tool message; the next OpenAI request is then rejected with HTTP
    400.  Continue those fallback paths with a normal user message instead.

    The token-in training loop deliberately keeps using
    ``prompt.observation_message`` directly; this compatibility rule is specific
    to the validated message protocol of ``/v1/chat/completions``.
    """
    if not scaffold.fn_calling or not tool_call_id:
        return {"role": "user", "content": obs_text}

    msg = _prompt.observation_message(scaffold, obs_text)
    msg["tool_call_id"] = tool_call_id
    if tool_name:
        msg["name"] = tool_name
    return msg


async def _run_chat_loop(
    session: aiohttp.ClientSession,
    cfg: EvalConfig,
    scaffold: Scaffold,
    env: Any,
    messages: list[dict],
    tag: str = "?",
) -> dict:
    """Drive the model through the sweagent tool loop over chat messages.

    Mirrors ``generate._run_token_loop``'s turn/step accounting and finish
    handling, but message-based: each turn POSTs the running ``messages`` to the
    chat endpoint, parses the assistant text into an ``Action`` (shared
    ``r2egym.action``), runs it through ``env.step``, and appends the assistant +
    observation turns (shared ``r2egym.prompt.observation_message``, with the same
    per-step stepcount reminder). ``messages`` is extended in place.

    ``messages`` accumulates the *entire* interaction verbatim -- system prompt,
    each raw assistant response, and each observation (as a user message with the
    stepcount reminder) -- so the returned ``messages`` alone is a faithful,
    replayable transcript of the whole model<->env exchange. The one exception is
    a turn cut off by the generation ceiling: its content is replaced by
    ``_OVERLONG_ELIDED`` (the raw text survives in ``response``) so the runaway
    does not consume the rest of the context window.
    """
    step_count = 0
    submitted = False
    status = "completed"
    total_tokens = 0
    completion_tokens = 0
    overlong_turns = 0
    transcript: list[str] = []
    loop_t0 = time.time()

    for _turn in range(scaffold.max_steps_absolute):
        # if cfg.max_model_len and total_tokens >= cfg.max_model_len:
        #     status = "truncated"
        #     break

        # Per-trajectory wall-clock ceiling: abort once total elapsed exceeds
        # ``cfg.max_total_time``, and clamp the next request's timeout to the
        # remaining budget so an in-flight POST cannot overshoot the deadline
        # (making max_total_time a hard ceiling). Stop early if too little time
        # remains to make a meaningful request.
        if cfg.max_total_time:
            remaining = cfg.max_total_time - (time.time() - loop_t0)
            if remaining <= 30:
                status = "timeout"
                break
            turn_timeout = min(cfg.request_timeout, remaining)
        else:
            turn_timeout = cfg.request_timeout

        if cfg.log_turns:
            _log_step_header(tag, _turn + 1, total_tokens)
        try:
            text, tool_calls, finish, usage, reasoning = await _chat_completion_once(
                session, cfg, messages, tools=scaffold.tools,
                timeout_override=turn_timeout,
            )
        except aiohttp.ClientResponseError as e:
            # The window is full: once the prompt reaches the served context length
            # the server rejects it outright (HTTP 400 from tokenizer_manager's
            # input-length validation) and every replay is rejected identically.
            # That is context exhaustion, the same condition the loop-top budget
            # check catches, so end the trajectory as ``truncated`` -- and, crucially,
            # RETURN, so run_instance still calls get_patch()/compute_reward(). If
            # the 400 propagated instead it would land in run_instance's
            # ``except Exception`` -> _error_result, which grades nothing and throws
            # away a possibly passing patch.
            # Narrow on purpose: a non-400 here means _chat_completion_once exhausted
            # its retries on a transport/5xx failure (a real infra error), and a 400
            # on the very first turn cannot be prompt growth -- it is a malformed
            # request -- so both stay visible as ``error``.
            if e.status != 400 or _turn == 0:
                raise
            logger.warning(
                "[eval] %s: server rejected the prompt at turn %d (HTTP 400: %s); "
                "treating as context exhaustion", tag, _turn + 1, str(e)[:200],
            )
            status = "truncated"
            break
        total_tokens = int(usage.get("total_tokens") or total_tokens)
        turn_completion = int(usage.get("completion_tokens") or 0)
        completion_tokens += turn_completion
        transcript.append(text)
        if cfg.log_turns:
            _log_model_turn(tag, _turn + 1, text, cfg, reasoning=reasoning,
                            tool_calls=tool_calls)

        # Record the assistant turn. Fold the model's reasoning back INTO the
        # content as ``{reasoning}\n</think>\n\n{content}`` (the golden / training
        # TITO on-the-wire shape) so the running transcript -- which is also the
        # body re-posted next turn -- carries the reasoning inline. This is what
        # keeps a thinking model reasoning on EVERY turn (the server re-wraps prior
        # turns in <think>) and keeps eval on the same distribution as training.
        # For fn-calling we still echo the structured tool_calls so the call is
        # replayable. See _inline_reasoning for the full rationale.
        assistant_msg: dict = {
            "role": "assistant",
            "content": _inline_reasoning(reasoning, text),
        }
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        # A turn cut off by the generation ceiling (finish=="length", or completion
        # tokens reaching an explicit cfg.max_tokens) carries no usable action --
        # the tool call, if any, is incomplete -- but it is NOT fatal: the cut
        # content is elided from the transcript and the turn is replayed as "no
        # valid tool call", i.e. it burns one step and the loop continues. This is
        # what the fixed-cap version got wrong: it ended 22% of SWE-bench-verified
        # trajectories on a single runaway turn, a fifth of which had already
        # written a passing patch.
        # Eliding is what makes continuing worthwhile -- keeping the cut turn
        # verbatim would hand the entire remaining window to the ramble, so the
        # loop-top ``total_tokens >= max_model_len`` check would end the trajectory
        # on the very next iteration. For the same reason ``total_tokens`` is rolled
        # back to this request's ``prompt_tokens``: the discarded completion is not
        # part of the next prompt, and the budget must describe what is.
        # The full text stays in ``transcript`` (-> the dump's ``response``) so the
        # runaway is still diagnosable offline.
        hit_cap = cfg.max_tokens > 0 and turn_completion >= cfg.max_tokens
        overlong = finish == "length" or hit_cap
        if overlong:
            overlong_turns += 1
            assistant_msg["content"] = _OVERLONG_ELIDED.format(tokens=turn_completion)
            assistant_msg.pop("tool_calls", None)
            total_tokens = int(usage.get("prompt_tokens") or total_tokens)

        # fn-calling: prefer the server's structured tool_calls; fall back to the
        # native <tool_call> text parse if the server returned it inline. non-fn:
        # parse the assistant text via the scaffold's XML parser.
        act = None
        tool_call_id = None
        tool_name = None
        if not overlong:
            if scaffold.fn_calling and tool_calls:
                tc = tool_calls[0]
                tool_call_id = tc.get("id")
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                tool_name = name or None
                raw_args = fn.get("arguments")
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                except Exception:
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                if name:
                    act = _action.Action(function_name=name, parameters=args)
            else:
                _thought, act = _action.parse(text, scaffold)

        if act is None:
            obs_text, done = _observation.continue_message(scaffold), False
        else:
            obs_text, done = await env.step(act)
        step_count += 1
        if cfg.log_turns:
            _log_observation(tag, obs_text, done, cfg)

        if done:
            submitted = True
            status = "completed"
            break

        # A tool-role observation is valid only when it answers a structured
        # tool call with an id. No-call/malformed-call and inline-XML fallback
        # paths have no such id and must continue as a user turn; otherwise the
        # next OpenAI request contains an orphan tool message and gets HTTP 400.
        obs_msg = _chat_observation_message(
            scaffold,
            obs_text,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
        )
        messages.append(obs_msg)
    else:
        status = "max_steps"

    return {
        "submitted": submitted,
        "status": status,
        "num_steps": step_count,
        "overlong_turns": overlong_turns,
        "response": "\n\n———\n\n".join(transcript),
        "response_len": completion_tokens,
        "messages": messages,
    }


# ── grading-infra failure detection ────────────────────────────────────────
# ``env.compute_reward`` grades in-place by running ``bash /run_tests.sh`` on the
# RLE pod via the vendored ``docker.py::_run_rle``. When that grading command
# does NOT complete for an *infra* reason -- the RLE session soft-failing
# (returning None/empty/malformed instead of raising), or the command blowing
# past ``SWE_EVAL_TIMEOUT_SEC`` and being killed -- ``_run_rle`` returns one of
# these literal strings as the command output *without raising* (docker.py
# :1161/1165/1176/1196). ``_calculate_reward`` then finds no passing tests in
# that string and returns reward 0, which is otherwise INDISTINGUISHABLE from a
# legitimate "the model's patch didn't pass". That is the hole that made
# ``errors=0`` untrustworthy: a grading that never ran was booked as a real
# resolved=False. We detect these signatures and surface the instance as an
# error (re-gradeable via --resume) instead of a false negative.
#
# NOTE: the pod-dead case is already covered upstream -- ``_run_rle`` *raises*
# ``PodUnavailableError`` on an SDK/transport exception (docker.py:1219), caught
# in ``run_instance`` as status="error". These signatures are the *soft* returns
# that do NOT raise. The signature table + ``grading_failure_reason`` live in
# ``r2egym.env`` so the training path (``generate.py``) shares the exact same
# detection instead of maintaining a divergent copy.


async def run_instance(
    session: aiohttp.ClientSession,
    cfg: EvalConfig,
    scaffold: Scaffold,
    ds: dict,
    sem: asyncio.Semaphore,
    progress: _Progress,
) -> dict:
    """Evaluate a single dataset row end-to-end: boot pod, run loop, grade.

    Boots the interactive pod with the *same* ``_boot_env`` the trainer uses
    (installs the scaffold tools), builds the turn-0 messages via ``r2egym.prompt``
    exactly like ``generate.generate``, runs the chat loop, captures the git diff,
    then grades in place via ``env.compute_reward`` (the swe_type's
    ``_calculate_reward``). Logs a START/END line per instance and registers itself
    in ``progress.running`` for the heartbeat. Never raises: failures are recorded
    as a result row so one bad instance does not abort the sweep.
    """
    instance_id = ds.get("instance_id") or "?"
    missing = _missing_ds_field(ds)
    if missing:
        logger.warning("⏭️  SKIP  [%s]  missing ds field: %s", instance_id, missing)
        return {
            "instance_id": instance_id, "swe_reward": 0.0, "resolved": False,
            "submitted": False, "status": "ds_missing_field",
            "error": f"missing ds field: {missing}",
        }

    problem_statement = ds.get("problem_statement") or ""
    # The direct chat-eval path does not pass through prepare_dataset.py, so it
    # must derive the repository path from the raw ds row itself.  In
    # particular, SWEBenchProDocker force-mounts the repository at /app; using
    # EvalConfig's generic /testbed default sends the model to a nonexistent
    # directory even though pod boot and grading succeed.
    swe_type = ds.get("swe_type")
    if swe_type == "swebench-pro":
        workdir = "/app"
    elif swe_type == "scale-swe":
        workdir = ds.get("workdir") or cfg.workdir
    else:
        workdir = cfg.workdir
    messages = _prompt.initial_messages(scaffold, problem_statement, workdir)

    t0 = time.time()
    async with sem:
        progress.running[instance_id] = t0
        logger.info(
            "▶ START [%s]  in-flight=%d  done=%d/%d",
            instance_id, len(progress.running), progress.done, progress.total,
        )
        if cfg.log_turns:
            _log_initial_prompt(instance_id, messages, cfg)
        env = None
        try:
            env = await _boot_env(ds, instance_id=instance_id)
            loop_info = await _run_chat_loop(
                session, cfg, scaffold, env, messages, tag=instance_id,
            )
            patch = await env.get_patch()
            reward, test_output = await env.compute_reward()
        except PodUnavailableError as e:
            return _finish(progress, instance_id,
                           _error_result(instance_id, f"pod_unavailable:{str(e)[:120]}", t0,
                                         messages=messages, ds_repro=_ds_repro(ds)))
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[eval] %s failed: %s\n%s", instance_id, e, traceback.format_exc()
            )
            return _finish(progress, instance_id,
                           _error_result(instance_id, f"exception:{type(e).__name__}", t0,
                                         messages=messages, ds_repro=_ds_repro(ds)))
        finally:
            if env is not None:
                await env.close()

    reward = float(reward)
    # A 0 reward whose grading command never actually completed (RLE soft-failure
    # or grading-timeout kill) is NOT a real resolved=False -- surface it as an
    # error so it is counted honestly and re-graded on --resume, rather than
    # silently depressing the score. See grading_failure_reason (r2egym.env).
    grading_reason = grading_failure_reason(reward, test_output)
    if grading_reason is not None:
        logger.warning(
            "[eval] %s grading did not complete (%s); recording status="
            "grading_error instead of a false resolved=False (re-gradeable via "
            "--resume)", instance_id, grading_reason,
        )
    status = "grading_error" if grading_reason is not None else loop_info["status"]
    result = {
        "instance_id": instance_id,
        "swe_type": ds.get("swe_type"),
        "swe_reward": reward,
        "resolved": reward >= 1.0,
        "submitted": loop_info["submitted"],
        "status": status,
        "num_steps": loop_info["num_steps"],
        "overlong_turns": loop_info["overlong_turns"],
        "patch_len": len(patch or ""),
        "response_len": loop_info["response_len"],
        "patch": patch or "",
        "messages": loop_info["messages"],
        "response": loop_info["response"],
        "ds_repro": _ds_repro(ds),
        "elapsed": round(time.time() - t0, 1),
    }
    if grading_reason is not None:
        result["error"] = f"grading_incomplete:{grading_reason}"
    return _finish(progress, instance_id, result)


def _ds_repro(ds: dict) -> dict:
    """Extract the ds fields needed to recreate the interactive/grading pod.

    ``docker_image`` + ``base_commit`` pin the environment; ``repo`` +
    ``problem_statement`` describe the task the model was solving. Together with
    the dumped ``messages`` these make a single dump file self-contained: which
    model + which pod + what task produced this transcript.
    """
    return {
        "repo": ds.get("repo"),
        "base_commit": ds.get("base_commit"),
        "docker_image": ds.get("docker_image"),
        "swe_type": ds.get("swe_type"),
        "problem_statement": ds.get("problem_statement") or "",
    }


def _finish(progress: _Progress, instance_id: str, result: dict) -> dict:
    """Deregister the instance from the heartbeat + log its END line."""
    progress.running.pop(instance_id, None)
    logger.info(
        "■ END   [%s]  reward=%.2f resolved=%s status=%s steps=%s (%.1fs)  in-flight=%d",
        instance_id, result.get("swe_reward", 0.0), result.get("resolved"),
        result.get("status"), result.get("num_steps", "-"),
        result.get("elapsed", 0.0), len(progress.running),
    )
    return result


def _error_result(
    instance_id: str, reason: str, t0: float, *,
    messages: list[dict] | None = None, ds_repro: dict | None = None,
) -> dict:
    """Build a failed-instance result row.

    ``messages`` / ``ds_repro`` carry whatever partial trajectory + env-repro
    context existed at the moment of failure so the dump of an errored run is NOT
    an empty shell: ``_run_chat_loop`` extends ``messages`` in place, so on a
    mid-loop TimeoutError/PodUnavailableError the turns completed before the
    failure are preserved (the failing turn's assistant message is not yet
    appended). Both stay OUT of ``_LEAN_KEYS`` so results.jsonl stays compact;
    they only flow into the per-instance dump.
    """
    return {
        "instance_id": instance_id,
        "swe_type": (ds_repro or {}).get("swe_type"),
        "swe_reward": 0.0, "resolved": False,
        "submitted": False, "status": "error", "error": reason,
        "elapsed": round(time.time() - t0, 1),
        "messages": messages or [],
        "ds_repro": ds_repro or {},
    }


def _load_dataset(path: str, limit: int) -> list[dict]:
    """Read a SWE dataset jsonl; keep the first ``limit`` rows (0 == all)."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def _load_prior_results(results_path: Path) -> dict[str, dict]:
    """Read an existing results.jsonl into ``{instance_id: lean_row}`` (last wins).

    Used by ``--resume``: rows whose ``error`` is empty are treated as done and
    skipped on the re-run; rows WITH an ``error`` (exception / pod_unavailable /
    ds_missing_field) are re-attempted. Malformed lines are ignored so a partially
    written file (killed mid-flush) still resumes.
    """
    prior: dict[str, dict] = {}
    if not results_path.exists():
        return prior
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            iid = row.get("instance_id")
            if iid:
                prior[iid] = row
    return prior


# Compact per-row fields flushed to results.jsonl; the bulky messages/response/patch
# are only kept in the optional per-instance traj/ dumps (--save-traj).
_LEAN_KEYS = (
    "instance_id", "swe_type", "swe_reward", "resolved", "submitted", "status",
    "num_steps", "overlong_turns", "patch_len", "response_len", "elapsed", "error",
)


def _print_banner(
    cfg: EvalConfig, scaffold: Scaffold, data_path: str, out_dir: str, total: int
) -> None:
    """Print R2E-Gym-style startup info: params, model, scaffold + the system
    prompt (so the exact instruction the model receives is visible up front)."""
    logger.info("📥 加载数据集 %s  (rows=%d, limit-applied)", data_path, total)
    logger.info(
        "使用参数 EvalConfig(url=%r, model=%r, concurrent=%d, "
        "max_steps_absolute=%d, temperature=%s, top_p=%s, max_tokens=%d, "
        "max_model_len=%d, extra_sampling=%s, request_timeout=%d, retry=%d, "
        "max_total_time=%d, scaffold=%r/%r)",
        cfg.url, cfg.model, cfg.concurrent, cfg.max_steps_absolute,
        cfg.temperature, cfg.top_p, cfg.max_tokens, cfg.max_model_len,
        cfg.extra_sampling, cfg.request_timeout, cfg.retry, cfg.max_total_time,
        scaffold.name, scaffold.variant,
    )
    logger.info("🚀 开始 SWE chat 评测")
    logger.info("📊 数据集: %s", data_path)
    logger.info("📁 输出目录: %s", out_dir)
    logger.info("🤖 模型: %s", cfg.model)
    logger.info("🌐 端点: %s", cfg.url)
    logger.info("🔧 Scaffold: %s/%s  (parser=%s, tools=%s)",
                scaffold.name, scaffold.variant, scaffold.response_parser,
                ", ".join(sorted(scaffold.allowed_tools)))
    logger.info("🌡️  温度: %s / top_p: %s / max_tokens: %d",
                cfg.temperature, cfg.top_p, cfg.max_tokens)
    logger.info("📈 最大步数绝对上限: %d", cfg.max_steps_absolute)
    logger.info("💓 heartbeat: %ds   log_turns=%s", cfg.heartbeat_interval, cfg.log_turns)
    logger.info(
        "🧩 SYSTEM PROMPT (scaffold %s/%s)\n%s\n%s\n%s",
        scaffold.name, scaffold.variant, "─" * 78, scaffold.system_prompt, "─" * 78,
    )


def _build_dump(cfg: EvalConfig, scaffold: Scaffold, res: dict) -> dict:
    """Assemble a single self-contained, replayable trajectory dump for ``res``.

    The whole model<->env interaction lives in ``res["messages"]`` verbatim
    (system prompt -> raw assistant responses -> observations, each with the
    stepcount reminder); we wrap it with the run context (model / endpoint /
    sampling / scaffold), the env repro fields (docker image + commit + task),
    the outcome (reward / status / patch), so a reader can reproduce exactly which
    model + pod + task produced this transcript and re-parse actions offline with
    the same scaffold parser. Compact per-row fields stay in results.jsonl.
    """
    return {
        "instance_id": res.get("instance_id"),
        "model": cfg.model,
        "url": cfg.url,
        "sampling": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            **cfg.extra_sampling,
        },
        "scaffold": {"name": scaffold.name, "variant": scaffold.variant},
        "env": res.get("ds_repro") or {},
        "result": {
            "swe_reward": res.get("swe_reward"),
            "resolved": res.get("resolved"),
            "submitted": res.get("submitted"),
            "status": res.get("status"),
            "num_steps": res.get("num_steps"),
            "patch_len": res.get("patch_len"),
            "response_len": res.get("response_len"),
            "elapsed": res.get("elapsed"),
            "error": res.get("error"),
        },
        "patch": res.get("patch") or "",
        # the full replayable transcript: system + every turn + every observation
        "messages": res.get("messages") or [],
    }


async def run_eval(
    cfg: EvalConfig, data_path: str, out_dir: str, *, limit: int,
    save_traj: bool, dump_dir: str | None = None, resume: bool = False,
    benchmark: str | None = None,
) -> dict:
    """Run the chat eval over the dataset and write results/summary to ``out_dir``.

    Boots the shared scaffold once (the single train/eval prompt+budget source),
    fans out ``run_instance`` under a concurrency semaphore, streams a lean result
    row per instance to ``results.jsonl`` as each finishes, and writes
    ``summary.json``. Two optional per-instance dumps: ``--save-traj`` writes the
    raw ``res`` dict to ``out/traj/<id>.json`` (legacy); ``dump_dir`` (when set)
    writes the self-contained replayable trajectory (``_build_dump``: run context +
    env repro fields + full messages transcript) to ``<dump_dir>/<id>.json``.

    ``resume``: when set, prior ``out/results.jsonl`` rows without an ``error`` are
    kept as-is and their instances are skipped; only errored / never-run instances
    are re-attempted (i.e. point ``--out-dir`` at a finished run to top up its
    failures). Off (default) => the file is overwritten and every row is run.
    """
    scaffold = load_scaffold(max_steps_absolute=cfg.max_steps_absolute)
    rows = _load_dataset(data_path, limit)
    if benchmark:
        # Host-side preflight of the ACTUAL rows about to run (fields + vendored
        # spec coverage); a FAIL here means whole swathes of rows will error at
        # boot, so surface it loudly but let the user run anyway.
        try:
            from .benchmarks import check_rows

            host = check_rows(rows)
            if not host["ok"]:
                logger.warning(
                    "⚠️  benchmark=%s 数据源预检有 FAIL:\n%s",
                    benchmark,
                    "\n".join(
                        f"    [{c['check']}] {c['detail']}" for c in host["checks"]
                    ),
                )
            else:
                logger.info(
                    "✅ benchmark=%s 数据源预检通过: %s",
                    benchmark,
                    " | ".join(c["detail"] for c in host["checks"]),
                )
        except Exception:  # noqa: BLE001 -- preflight must never block the run
            logger.warning("⚠️  数据源预检执行失败（忽略，继续跑）", exc_info=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    traj_dir = out / "traj"
    if save_traj:
        traj_dir.mkdir(exist_ok=True)
    dump_path = Path(dump_dir) if dump_dir else None
    if dump_path is not None:
        dump_path.mkdir(parents=True, exist_ok=True)
        logger.info("📦 dump: 完整可复现轨迹 -> %s/<instance_id>.json", dump_path)

    results_path = out / "results.jsonl"
    # --resume: keep prior successful rows (no error), re-run everything else.
    dataset_ids = {ds.get("instance_id") for ds in rows}
    preserved: list[dict] = []
    if resume:
        prior = _load_prior_results(results_path)
        preserved = [
            row for iid, row in prior.items()
            if iid in dataset_ids and not row.get("error")
        ]
        skip_ids = {row["instance_id"] for row in preserved}
        rerun_rows = [ds for ds in rows if ds.get("instance_id") not in skip_ids]
        logger.info(
            "♻️  resume: %d 条已成功 -> 跳过；%d 条失败/未跑 -> 补跑 (共 %d)",
            len(preserved), len(rerun_rows), len(rows),
        )
    else:
        rerun_rows = rows

    _print_banner(cfg, scaffold, data_path, out_dir, len(rows))

    sem = asyncio.Semaphore(cfg.concurrent)
    progress = _Progress(total=len(rows))
    # Seed counters with the preserved (already-successful) rows so heartbeat /
    # summary reflect the whole dataset, not just the re-run subset.
    progress.done = len(preserved)
    progress.resolved = sum(int(bool(r.get("resolved"))) for r in preserved)
    progress.submitted = sum(int(bool(r.get("submitted"))) for r in preserved)
    for r in preserved:
        progress.status_counts[r.get("status") or "unknown"] += 1

    # Per-swe_type outcome counters, seeded from the whole dataset + the preserved
    # (resume) rows so the summary reflects the full run, not just the subset.
    by_type: dict[str, dict[str, int]] = {}
    for ds in rows:
        t = ds.get("swe_type") or "unknown"
        by_type.setdefault(t, {"total": 0, "resolved": 0, "submitted": 0, "errors": 0})["total"] += 1
    for r in preserved:
        row = by_type.setdefault(
            r.get("swe_type") or "unknown",
            {"total": 0, "resolved": 0, "submitted": 0, "errors": 0},
        )
        row["resolved"] += int(bool(r.get("resolved")))
        row["submitted"] += int(bool(r.get("submitted")))

    t_start = time.time()
    hb_task = (
        asyncio.create_task(_heartbeat(progress, cfg.heartbeat_interval))
        if cfg.heartbeat_interval > 0 else None
    )
    async with aiohttp.ClientSession() as session:
        # Fail fast if this is a fn-calling sweep but the endpoint can't take
        # tools= (rather than letting every instance error deep in the loop).
        await _probe_fn_calling_support(session, cfg, scaffold)
        tasks = [
            asyncio.create_task(run_instance(session, cfg, scaffold, ds, sem, progress))
            for ds in rerun_rows
        ]
        try:
            with open(results_path, "w", encoding="utf-8") as rf:
                # Re-emit preserved successful rows first so the file stays complete.
                for row in preserved:
                    rf.write(json.dumps(row, ensure_ascii=False) + "\n")
                rf.flush()
                for fut in asyncio.as_completed(tasks):
                    res = await fut
                    progress.done += 1
                    progress.resolved += int(bool(res.get("resolved")))
                    progress.submitted += int(bool(res.get("submitted")))
                    progress.errors += int(
                        res.get("status") in ("error", "grading_error", "ds_missing_field")
                    )
                    progress.status_counts[res.get("status") or "unknown"] += 1
                    b = by_type.setdefault(
                        res.get("swe_type") or "unknown",
                        {"total": 0, "resolved": 0, "submitted": 0, "errors": 0},
                    )
                    b["resolved"] += int(bool(res.get("resolved")))
                    b["submitted"] += int(bool(res.get("submitted")))
                    b["errors"] += int(
                        res.get("status") in ("error", "grading_error", "ds_missing_field")
                    )

                    iid = str(res.get("instance_id") or progress.done).replace("/", "_")
                    if save_traj:
                        (traj_dir / f"{iid}.json").write_text(
                            json.dumps(res, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    if dump_path is not None:
                        (dump_path / f"{iid}.json").write_text(
                            json.dumps(_build_dump(cfg, scaffold, res),
                                       ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )
                    lean = {k: res[k] for k in _LEAN_KEYS if k in res}
                    rf.write(json.dumps(lean, ensure_ascii=False) + "\n")
                    rf.flush()
        finally:
            if hb_task is not None:
                hb_task.cancel()
                with suppress(asyncio.CancelledError):
                    await hb_task

    total = len(rows)
    summary = {
        "benchmark": benchmark,
        "data": data_path,
        "model": cfg.model,
        "url": cfg.url,
        "total": total,
        "resolved": progress.resolved,
        "submitted": progress.submitted,
        "errors": progress.errors,
        "rows_by_swe_type": {t: v["total"] for t, v in by_type.items()},
        "by_type": dict(sorted(by_type.items())),
        "status_breakdown": dict(sorted(progress.status_counts.items())),
        "resolve_rate": round(progress.resolved / total, 4) if total else 0.0,
        "submit_rate": round(progress.submitted / total, 4) if total else 0.0,
        "sampling": {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "max_tokens": cfg.max_tokens,
            **cfg.extra_sampling,
        },
        "budget": {
            "max_steps_absolute": cfg.max_steps_absolute,
            "max_model_len": cfg.max_model_len,
            "max_total_time": cfg.max_total_time,
        },
        "elapsed": round(time.time() - t_start, 1),
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(
        "✅ eval done: resolved=%d/%d (%.1f%%) submitted=%d errors=%d  elapsed=%.0fs",
        progress.resolved, total, 100 * summary["resolve_rate"],
        progress.submitted, progress.errors, summary["elapsed"],
    )
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data", default=None,
        help="SWE dataset jsonl (default: the file the registry maps --benchmark to)",
    )
    p.add_argument(
        "--benchmark", default=None,
        help="benchmark name from eval/benchmarks.json; recorded in summary/results "
             "and supplies the default --data file. --data, when given, wins.",
    )
    p.add_argument(
        "--check-only", action="store_true",
        help="run the benchmark's host-side preflight checks (required fields / "
             "spec coverage / run scripts) and exit 0/1; no eval, no pod",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="output dir (results/summary/traj); required for an eval run "
             "(not for --check-only)",
    )
    p.add_argument(
        "--url",
        default="http://<MODEL_HOST>:<PORT>/v1/chat/completions",
        help="OpenAI-compatible /v1/chat/completions endpoint (placeholder default; "
             "--url is always set by run_eval.sh's LLM_URL)",
    )
    p.add_argument(
        "--model", default=None,
        help="served model name; required for an eval run (not for --check-only)",
    )
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument(
        "--max-tokens", type=int, default=0,
        help="per-turn generation cap; 0 (default) sends no max_tokens so the "
             "server grants the whole remaining window (context_len - input_len - 2)",
    )
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--max-steps-absolute", type=int, default=300)
    p.add_argument(
        "--extra-sampling", type=str,
        default='{"top_k": 20, "min_p": 0.0, "presence_penalty": 0.0, "repetition_penalty": 1.0}',
        help="JSON dict of non-top-level sampling params merged into the request body",
    )
    p.add_argument("--request-timeout", type=int, default=1200)
    p.add_argument("--retry", type=int, default=3)
    p.add_argument(
        "--max-total-time", type=int, default=5400,
        help="per-trajectory wall-clock ceiling in seconds (0 disables)",
    )
    p.add_argument("--concurrent", type=int, default=16)
    p.add_argument("--limit", type=int, default=0, help="0 == all rows")
    p.add_argument("--save-traj", action="store_true", help="dump full traj/<id>.json")
    p.add_argument(
        "--resume", action="store_true",
        help="reuse --out-dir/results.jsonl: skip instances that already succeeded "
             "(no error) and only re-run failed/missing ones (top up a finished run)",
    )
    p.add_argument(
        "--dump-dir", type=str, default="",
        help="if set, dump the full self-contained replayable trajectory "
             "(run context + env repro fields + complete messages transcript) to "
             "<dump-dir>/<instance_id>.json (one file per instance). Empty == off.",
    )
    p.add_argument(
        "--log-turns", action="store_true",
        help="log every turn's think / tool calls / observation (like visualize_rollouts)",
    )
    p.add_argument(
        "--log-max-chars", type=int, default=400,
        help="per-field truncation for --log-turns (0 == no truncation)",
    )
    p.add_argument(
        "--heartbeat-interval", type=int, default=60,
        help="seconds between heartbeat progress logs (0 disables)",
    )
    return p.parse_args()


def _setup_logging() -> None:
    """Give our logger its own stdout handler that external libraries can't mute.

    Booting the RLE pod imports r2egym / docker / gym, which reconfigure the root
    logger (raising its level to WARNING), silently dropping every INFO we emit
    after boot. Attaching a dedicated handler + ``propagate=False`` + an explicit
    level isolates the eval log from that, so START/END, per-turn interaction and
    heartbeat lines always come through.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def main() -> None:
    _setup_logging()
    args = _parse_args()
    extra_sampling = json.loads(args.extra_sampling) if args.extra_sampling else {}

    # --check-only: host-side preflight, no server, no pod.
    if args.check_only:
        from .benchmarks import check_benchmark, _print_report

        try:
            r = check_benchmark(args.benchmark or "swebench")
        except (KeyError, FileNotFoundError, ValueError) as e:
            logger.error("❌ preflight failed: %s", e)
            raise SystemExit(2) from e
        _print_report(r)
        raise SystemExit(0 if r["ok"] else 1)

    if not args.model or not args.out_dir:
        print(
            "error: --model and --out-dir are required for an eval run "
            "(--check-only needs only --benchmark)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    # --data wins over the benchmark registry; otherwise derive from --benchmark.
    data_path = args.data
    if not data_path:
        if not args.benchmark:
            print(
                "error: one of --data or --benchmark is required "
                "(benchmark names: see eval/benchmarks.json)",
                file=sys.stderr,
            )
            raise SystemExit(2)
        from .benchmarks import resolve_file

        data_path = str(resolve_file(args.benchmark))

    cfg = EvalConfig(
        url=args.url,
        model=args.model,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_steps_absolute=args.max_steps_absolute,
        extra_sampling=extra_sampling,
        request_timeout=args.request_timeout,
        retry=args.retry,
        max_total_time=args.max_total_time,
        concurrent=args.concurrent,
        log_turns=args.log_turns,
        log_max_chars=args.log_max_chars,
        heartbeat_interval=args.heartbeat_interval,
    )
    try:
        summary = asyncio.run(
            run_eval(
                cfg, data_path, args.out_dir,
                limit=args.limit, save_traj=args.save_traj,
                dump_dir=args.dump_dir or None, resume=args.resume,
                benchmark=args.benchmark,
            )
        )
    except FnCallingUnsupportedError as e:
        logger.error("❌ fn-calling 不受支持: %s", e)
        raise SystemExit(2) from e
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
