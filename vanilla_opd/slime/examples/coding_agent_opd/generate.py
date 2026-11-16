"""Coding-Agent OPD: single-Sample, model-in-the-loop SWE generate().

    --custom-generate-function-path examples.coding_agent_opd.generate.generate

Each turn:
  1. POST the running token sequence to the sglang student ``/generate``
     (``return_logprob=True``).
  2. Parse the emitted text into a tool call via the shared scaffold's parser
     (``r2egym.action.parse``) and execute it with ``AsyncRepoEnv.step`` (which
     renders the observation through ``r2egym.observation``).
  3. Render the observation delta with ``r2egym.prompt`` (the official chat
     template, not hand-rolled ChatML) and append it ``trainable=False`` so it
     carries loss_mask=0 (the model is not trained on tool output).

The eval path always sets ``sample.reward`` directly and skips the teacher.
Sandbox backend: RLE only. Any other backend raises ``NotImplementedError``.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time
import traceback
import uuid
from typing import Any
import os

import numpy as np

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

# Import the r2egym glue package first: its __init__ runs the sys.path shim that
# puts the vendored r2egym ahead of any editable install, so the bare
# ``r2egym.agenthub...`` import below resolves to the vendored tree.
from .r2egym import action as _action
from .r2egym import observation as _observation
from .r2egym import prompt as _prompt
from .r2egym.env import AsyncRepoEnv, grading_failure_reason
from .utils.rollout import (
    _ROLLOUT_GUARD_SEC,
    _SCAFFOLD,
    _LoopAbort,
    _abort_sample,
    _abort_sglang_request,
    _boot_env,
    _max_context_length,
    _missing_ds_field,
    _needs_sglang_teacher_fetch,
    _require_fn_calling,
    _require_rle_backend,
    _session_id,
    _trunc,
)

from r2egym.agenthub.runtime.docker import PodUnavailableError  # noqa: E402

logger = logging.getLogger(__name__)
_GENERATE_LOGGER_LEVEL = os.environ.get("GENERATE_LOGGER_LEVEL", "INFO")
logger.setLevel(_GENERATE_LOGGER_LEVEL)
if not any(getattr(h, "_coding_agent_opd", False) for h in logger.handlers):
    _gen_handler = logging.StreamHandler()  # -> stderr; captured by the script's 2>&1
    _gen_handler.setLevel(_GENERATE_LOGGER_LEVEL)
    _gen_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _gen_handler._coding_agent_opd = True  # sentinel: don't re-add on module re-import
    logger.addHandler(_gen_handler)
logger.propagate = False


def _opd_student_topk(args) -> int:
    """K to request from the student rollout engine; 0 unless topk_reverse_kl, the only
    mode that needs the student's own top-k ids (to query the teacher on that support)."""
    if getattr(args, "opd_loss_type", None) != "topk_reverse_kl":
        return 0
    return int(getattr(args, "opd_id_topk", 0) or 0)


def _append_student_topk_ids(sample: Sample, meta_info: dict, n_tokens: int, k: int) -> None:
    """Append this turn's [T, K] student top-k ids as one np.int32 chunk.

    ``generate`` concatenates the chunks ONCE on exit; doing it per turn re-copies the
    whole array every turn (O(N^2) memcpy on the single rollout event-loop core).
    """

    def _ids(pos):
        head = pos[:k]
        # ids_only -> bare ints; legacy -> [logprob, token_id, token_text] tuples.
        if head and isinstance(head[0], (list, tuple)):
            return [entry[1] for entry in head]
        return head

    blob = meta_info.get("output_top_logprobs_ids_b64")
    if blob is not None:
        # astype copies: frombuffer is read-only and the array outlives this response.
        arr = np.frombuffer(base64.b64decode(blob["vals"]), dtype=np.dtype(blob["dtype"]))
        rows = arr.reshape(blob["shape"])[:, :k].astype(np.int32)
        n_positions = rows.shape[0]
    else:
        top = meta_info.get("output_top_logprobs") or []
        rows = np.array([_ids(pos) for pos in top], dtype=np.int32) if top else None
        n_positions = len(top)

    if n_positions != n_tokens:
        raise _LoopAbort(
            f"topk_reverse_kl needs the student top-k ids aligned to generated tokens "
            f"(got {n_positions} vs {n_tokens}); "
            "check the rollout engine returns top_logprobs_num."
        )

    if rows is not None and rows.shape[1] != k:
        raise _LoopAbort(
            f"topk_reverse_kl needs exactly k={k} student top-k ids per position, "
            f"got {rows.shape[1]}; the rollout engine truncated top_logprobs_num."
        )

    if sample.student_topk_ids is None:
        sample.student_topk_ids = [rows]
    else:
        sample.student_topk_ids.append(rows)



async def _run_token_loop(
    args,
    sample: Sample,
    sampling_params: dict[str, Any],
    *,
    env: AsyncRepoEnv,
    state: GenerateState,
    url: str,
    max_context_length: int,
    messages: list[dict],
    loop_stats: dict[str, int],
) -> bool:
    """Drive the student model through the sweagent tool loop.

    The tool loop (training-owned; shares the scaffold/parser/observation/prompt
    fragile bottom with the eval harness, but keeps its own orchestration).

    Assumes ``sample.tokens`` already holds the turn-0 prompt and ``messages``
    is its matching history (see ``generate``). Appends model tokens
    (trainable=True) and observation tokens (trainable=False) in place, sets
    ``sample.status`` on exit, and returns True iff the model called
    submit/finish. Raises :class:`_LoopAbort` on an unrecoverable generation
    state (aborted generation / missing logprobs / template misalignment), which
    the caller converts to an ABORTED sample so the group is retried.

    A turn cut off by the per-turn generation cap is NOT terminal: it is replayed
    as "no valid tool call" so the model can retry. Only the context budget ends
    a trajectory early (TRUNCATED).
    """
    tokenizer = state.tokenizer
    eos_id = tokenizer.eos_token_id
    scaffold = _SCAFFOLD
    tag = sample.session_id or "?"

    for turn in range(scaffold.max_steps_absolute):
        if len(sample.tokens) >= max_context_length:
            sample.status = Sample.Status.TRUNCATED
            return False

        remaining = max_context_length - len(sample.tokens)
        per_turn = dict(sampling_params)
        # Per-turn generation cap, aligned with the eval harness's ``--max-tokens``:
        # a non-positive value (``--rollout-max-response-len 0``, the default) means
        # "no per-turn cap", i.e. the whole remaining context budget is granted to
        # this turn
        per_turn_cap = int(sampling_params.get("max_new_tokens") or 0)
        per_turn["max_new_tokens"] = min(per_turn_cap, remaining) if per_turn_cap > 0 else remaining

        # rid lets a cancelled/aborted turn (e.g. the rollout wall-clock guard firing)
        # free the KV slot eagerly via /abort_request instead of letting the orphaned
        # generation run to its length cap.
        rid = uuid.uuid4().hex
        req_body = {
            "rid": rid,
            "input_ids": sample.tokens,
            "sampling_params": per_turn,
            "return_logprob": True,
        }
        student_topk = _opd_student_topk(args)
        if student_topk > 0:
            req_body["top_logprobs_num"] = student_topk
            # Bare ids instead of [logprob, id, text] tuples, and one base64 int32 blob
            # instead of a [T, K] JSON list
            req_body["top_logprobs_ids_only"] = True
            req_body["return_top_logprobs_ids_as_bytes"] = True
        _decode_t0 = time.perf_counter()
        try:
            output = await post(url, req_body)
        except asyncio.CancelledError:
            _abort_sglang_request(url, rid)
            raise
        _decode_dt = time.perf_counter() - _decode_t0
        loop_stats["model_decode_times"].append(_decode_dt)
        finish = output["meta_info"]["finish_reason"]["type"]
        if finish == "abort":
            raise _LoopAbort("sglang returned abort")

        otl = output["meta_info"].get("output_token_logprobs")
        if not otl:
            # No per-token logprobs -> cannot align rollout_log_probs. Abort so
            # the group is retried rather than poisoning the trainer.
            raise _LoopAbort("missing output_token_logprobs structure for cur_tokens")

        cur_tokens = [item[1] for item in otl]
        cur_log_probs = [item[0] for item in otl]
        cur_text = tokenizer.decode(cur_tokens)
        last_token_is_eos = bool(cur_tokens) and cur_tokens[-1] == eos_id
        eos_text = tokenizer.eos_token
        if last_token_is_eos and cur_text.endswith(eos_text):
            assistant_content = cur_text[: -len(eos_text)]
        else:
            # Defensive: fall back to token-level decode if the eos text isn't a
            # clean suffix (e.g. a non-eos stop token ended the turn).
            assistant_content = (
                tokenizer.decode(cur_tokens[:-1]) if last_token_is_eos else cur_text
            )
        sample.append_response_tokens(
            args,
            tokens=cur_tokens,
            log_probs=cur_log_probs,
            trainable=True,
            meta_info=output["meta_info"],
            text=cur_text,
        )
        # Capture the student's per-position top-k ids for these (trainable) tokens.
        # One length-K id list per generated token, appended in response order so
        # reward_func can pair them with the response region for topk_reverse_kl.
        if student_topk > 0:
            _append_student_topk_ids(sample, output["meta_info"], len(cur_tokens), student_topk)
        loop_stats["num_turns"] = turn + 1
        # Build ONE log line per turn: the model-generation part now, the tool /
        # no-call part below, then a single logger.info before the turn ends.
        _turn_msg = (
            "[coding_agent_opd] %s turn=%d model-> (finish=%s, %d tok, decode=%.2fs) %s"
            % (tag, turn, finish, len(cur_tokens), _decode_dt, _trunc(cur_text, 20))
        )

        overlong = finish == "length"
        if overlong:
            loop_stats["overlong_turns"] = loop_stats.get("overlong_turns", 0) + 1

        # Use the ROLLOUT (student) server's tool-call / reasoning parsers -- the
        # text here is the student's generation. The teacher server (--rm-url) is a
        # different model queried only for logprobs, never for tool parsing.
        action = None
        if not overlong:
            _ , action = _action.parse(
                cur_text,
                scaffold,
                tool_call_parser=getattr(args, "sglang_tool_call_parser", None) or None,
                reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
            )
        if action is None:
            obs_text = _observation.continue_message(scaffold)
            done = False
            _turn_msg += (
                " | turn cut by the %d-tok per-turn cap, treated as no tool call"
                % per_turn["max_new_tokens"]
                if overlong
                else " | model emitted no valid tool call"
            )
        else:
            _tool_t0 = time.perf_counter()
            obs_text, done = await env.step(action)
            _tool_dt = time.perf_counter() - _tool_t0
            loop_stats["tool_call_count"] = loop_stats.get("tool_call_count", 0) + 1
            loop_stats["tool_exec_times"].append(_tool_dt)
            _by_name = loop_stats["tool_exec_times_by_name"]
            _bucket = action.function_name if action.function_name in _by_name else "__other__"
            _by_name[_bucket].append(_tool_dt)
            _turn_msg += " | tool=%s exec=%.2fs done=%s obs=%s" % (
                action.function_name,
                _tool_dt,
                done,
                _trunc(obs_text, 20),
            )

        logger.debug(_turn_msg)

        if done:
            # submit/finish ends the loop; the final observation is mask=0
            # trailing text never generated against, so we don't append it.
            sample.status = Sample.Status.COMPLETED
            return True

        # Build the next-turn observation (tool-role for fn-calling, user-role
        # otherwise) then render the connector + observation token delta via the
        # official chat template -- no hand-rolled ChatML, no stepcount reminder.
        obs_message = _prompt.observation_message(scaffold, obs_text)
        if action is None and scaffold.fn_calling:
            # No valid tool call this turn (observation is the scaffold's "no
            # valid tool execution" continue message): a tool-role message would
            # be ORPHANED -- no preceding assistant tool_call to pair its
            # tool_call_id -- violating the messages protocol. Demote it to
            # user-role so the template/next request stays protocol-valid
            # (mirrors eval_chat.py).
            obs_message["role"] = "user"
        try:
            obs_tokens = _prompt.render_turn_delta(
                tokenizer,
                messages,
                obs_message,
                last_token_is_eos=last_token_is_eos,
                tools=scaffold.tools,
            )
        except ValueError as e:
            raise _LoopAbort(f"chat-template delta failed: {e}") from e
        messages.append({"role": "assistant", "content": assistant_content})
        messages.append(obs_message)

        # Trim the observation to the context budget; if it overflows we stop.
        overflow = len(sample.tokens) + len(obs_tokens) - max_context_length
        if overflow > 0:
            obs_tokens = obs_tokens[: max(0, len(obs_tokens) - overflow)]
            if obs_tokens:
                sample.append_response_tokens(args, tokens=obs_tokens, trainable=False)
            sample.status = Sample.Status.TRUNCATED
            return False

        sample.append_response_tokens(args, tokens=obs_tokens, trainable=False)

    # Ran out of turns (max_steps_absolute) without a submit.
    sample.status = Sample.Status.COMPLETED
    return False


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
async def generate(
    args,
    base_sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    """Model-in-the-loop SWE rollout. Returns a SINGLE Sample.

    A single-Sample return is required so the training path routes through
    ``async_rm -> reward_func`` (the teacher-logprob call); a list would route
    through ``batched_async_rm`` and skip the per-sample teacher fetch.
    """
    _require_rle_backend()
    if not evaluation:
        # fn-calling-only is a training constraint; eval supports both variants.
        _require_fn_calling()

    md = base_sample.metadata or {}
    ds = md.get("ds")
    workdir = md.get("workdir")
    instance_id = md.get("instance_id") or (ds or {}).get("instance_id") or "?"
    if not isinstance(ds, dict) or not workdir:
        return _abort_sample(base_sample, "missing_ds_or_workdir", instance_id)
    missing = _missing_ds_field(ds)
    if missing:
        return _abort_sample(base_sample, f"ds_missing_field:{missing}", instance_id)

    # ``base_sample.prompt`` is OVERWRITTEN below with the rendered chat template, so on a
    # retry it no longer holds the dataset's problem statement -- falling back to it would
    # nest the whole turn-0 prompt inside the new one.
    problem_statement = ds.get("problem_statement") or md.get("problem_statement") or base_sample.prompt or ""
    md = {**md, "problem_statement": problem_statement}
    base_sample.metadata = md
    base_sample.session_id = _session_id(base_sample, instance_id)

    # Retried samples arrive with stale rollout state; reset before concat.
    base_sample.rollout_log_probs = None
    base_sample.rollout_top_p_token_ids = None
    base_sample.rollout_top_p_token_offsets = None
    base_sample.response = ""
    base_sample.response_length = 0
    base_sample.loss_mask = []
    base_sample.student_topk_ids = None
    base_sample.remove_sample = False
    base_sample.reward = None
    base_sample.weight_versions = []
    base_sample.prefix_cache_info = Sample.PrefixCacheInfo()
    base_sample.spec_info = Sample.SpecInfo()
    base_sample.teacher_log_probs = None
    base_sample.teacher_topk_logprobs = None
    base_sample.teacher_topk_ids = None
    base_sample.student_support_ids = None
    base_sample.teacher_logprobs_on_student = None
    base_sample.opd_topk_overlap = None

    state = GenerateState(args)
    _prompt.ensure_tito_eos_alignment(state.tokenizer)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    messages = _prompt.initial_messages(_SCAFFOLD, problem_statement, workdir)
    prompt_ids, prompt_text = _prompt.render_prompt_tokens(
        state.tokenizer, messages, tools=_SCAFFOLD.tools
    )
    base_sample.prompt = prompt_text
    base_sample.tokens = list(prompt_ids)
    max_context_length = _max_context_length(args)

    logger.info(
        "[coding_agent_opd] %s: workdir=%r | turn-0 prompt (%d tok):%s",
        instance_id,
        workdir,
        len(prompt_ids),
        _trunc(prompt_text, 20),
    )

    t0 = time.time()
    loop_stats: dict = {
        "num_turns": 0,
        "tool_call_count": 0,
        "overlong_turns": 0,
        "model_decode_times": [],
        "tool_exec_times": [],
        "tool_exec_times_by_name": {
            **{name: [] for name in _SCAFFOLD.allowed_tools},
            "__other__": [],
        },
    }
    try:
        async with asyncio.timeout(_ROLLOUT_GUARD_SEC):
            env = await _boot_env(ds, instance_id=instance_id)
            try:
                try:
                    submitted = await _run_token_loop(
                        args,
                        base_sample,
                        sampling_params,
                        env=env,
                        state=state,
                        url=url,
                        max_context_length=max_context_length,
                        messages=messages,
                        loop_stats=loop_stats,
                    )
                except _LoopAbort as e:
                    return _abort_sample(
                        base_sample, f"loop_abort:{e}", instance_id, turns=loop_stats["num_turns"]
                    )

                # Capture the model's diff BEFORE grading: compute_reward resets
                # tests + applies the dataset test patch on the same pod.
                patch = await env.get_patch()
                # In-place grading via R2E-Gym's reward zoo. 
                reward, test_output = await env.compute_reward()
            finally:
                await env.close()
    except asyncio.TimeoutError:
        return _abort_sample(
            base_sample, "wall_clock_timeout", instance_id, turns=loop_stats["num_turns"]
        )
    except PodUnavailableError as e:
        return _abort_sample(
            base_sample, f"pod_unavailable:{str(e)[:120]}", instance_id, turns=loop_stats["num_turns"]
        )
    except Exception as e:
        logger.warning(
            "[coding_agent_opd] %s: rollout failed: %s\n%s",
            instance_id,
            e,
            traceback.format_exc(),
        )
        return _abort_sample(
            base_sample, f"exception:{type(e).__name__}", instance_id, turns=loop_stats["num_turns"]
        )

    if base_sample.response_length == 0:
        return _abort_sample(base_sample, "empty_response", instance_id, turns=loop_stats["num_turns"])

    if isinstance(base_sample.student_topk_ids, list):
        base_sample.student_topk_ids = (
            np.concatenate(base_sample.student_topk_ids) if base_sample.student_topk_ids else None
        )

    # A 0 reward whose grading command never actually ran (RLE soft-failure or a
    # grading-timeout kill) is a grading artifact, NOT a real resolved=False.
    grading_reason = grading_failure_reason(float(reward), test_output)
    if grading_reason is not None and evaluation:
        # Eval path only: the SWE scalar IS the eval result, so a grading artifact
        # must not be counted as resolved=False -- surface it as an aborted sample
        # (status="grading_error") exactly as before. (unchanged eval behaviour)
        return _abort_sample(
            base_sample, f"grading_error:{grading_reason}", instance_id, turns=loop_stats["num_turns"]
        )

    reward = float(reward)
    base_sample.metadata = {
        **md,
        "instance_id": instance_id,
        "swe_reward": reward,
        "grading_solved": reward >= 1.0,
        "grading_failed": grading_reason,
        "submitted": bool(submitted),
        "round_number": loop_stats["num_turns"],
        "tool_call_count": loop_stats["tool_call_count"],
        "overlong_turns": loop_stats["overlong_turns"],
        "model_decode_times": loop_stats["model_decode_times"],
        "tool_exec_times": loop_stats["tool_exec_times"],
        "tool_exec_times_by_name": loop_stats["tool_exec_times_by_name"],
    }
    _dec = loop_stats["model_decode_times"]
    _tex = loop_stats["tool_exec_times"]
    logger.info(
        "[coding_agent_opd] %s: reward=%.2f submitted=%s status=%s patch_len=%d "
        "resp_len=%d turns=%d overlong=%d elapsed=%.1fs decode=%.1fs(n=%d) tool=%.1fs(n=%d)%s",
        instance_id,
        reward,
        bool(submitted),
        base_sample.status.value,
        len(patch or ""),
        base_sample.response_length,
        loop_stats["num_turns"],
        loop_stats["overlong_turns"],
        time.time() - t0,
        sum(_dec),
        len(_dec),
        sum(_tex),
        len(_tex),
        " (eval-only)" if evaluation else "",
    )

    if evaluation:
        # Eval path: scalar reward directly, skip the teacher entirely.
        base_sample.reward = reward
        return base_sample

    if _needs_sglang_teacher_fetch(args):
        # sglang OPD: leave reward None so async_rm -> reward_func fetches teacher
        # logprobs; opd.post_process_rewards promotes swe_reward to the scalar.
        base_sample.reward = None
    else:
        # megatron OPD (or no OPD): teacher logprobs come from the in-backend
        # teacher forward, so set the SWE scalar directly and skip async_rm.
        base_sample.reward = reward
    return base_sample

