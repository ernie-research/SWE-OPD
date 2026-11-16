import asyncio
import importlib.util
import logging
from pathlib import Path

import aiohttp
import torch

from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _get_teacher_http_state(args):
    loop = asyncio.get_running_loop()
    loop_key = id(loop)
    state_by_loop = getattr(args, "_opd_teacher_http_state", None)
    if state_by_loop is None:
        state_by_loop = {}
        setattr(args, "_opd_teacher_http_state", state_by_loop)

    timeout = float(getattr(args, "opd_teacher_timeout", 300.0))
    concurrency = int(getattr(args, "opd_teacher_concurrency", 0))
    state = state_by_loop.get(loop_key)
    if state is None or state["session"].closed:
        state = {
            "session": aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout)),
            "semaphore": asyncio.Semaphore(concurrency) if concurrency > 0 else None,
        }
        state_by_loop[loop_key] = state
    return state


def _teacher_payload_summary(payload):
    if "input_ids" in payload:
        return f"input_tokens={len(payload['input_ids'])}"
    if "text" in payload:
        return f"text_chars={len(payload['text'])}"
    if "prompt" in payload:
        return f"prompt_chars={len(payload['prompt'])}"
    return "payload_size=unknown"


async def _post_teacher_once(session, teacher_url, payload, headers=None):
    async with session.post(teacher_url, json=payload, headers=headers) as resp:
        if resp.status >= 400:
            body = (await resp.text())[:500]
            raise aiohttp.ClientResponseError(
                request_info=resp.request_info,
                history=resp.history,
                status=resp.status,
                message=f"{resp.reason}; body={body}",
                headers=resp.headers,
            )
        return await resp.json()


async def _post_teacher_json(args, teacher_url, payload, sample, warning_prefix, headers=None):
    retries = max(0, int(getattr(args, "opd_teacher_retries", 2)))
    last_exc = None

    for attempt in range(retries + 1):
        try:
            state = _get_teacher_http_state(args)
            session = state["session"]
            semaphore = state["semaphore"]
            if semaphore is None:
                return await _post_teacher_once(session, teacher_url, payload, headers=headers)
            async with semaphore:
                return await _post_teacher_once(session, teacher_url, payload, headers=headers)
        except (TimeoutError, asyncio.TimeoutError, aiohttp.ClientError) as exc:
            last_exc = exc
            if attempt < retries:
                await asyncio.sleep(min(0.2 * (2**attempt), 2.0))
                continue

    logger.warning(
        "%s failed after %s attempt(s); using student rollout logprobs as teacher logprobs "
        "for sample index=%s (%s): %s: %s",
        warning_prefix,
        retries + 1,
        getattr(sample, "index", None),
        _teacher_payload_summary(payload),
        type(last_exc).__name__,
        last_exc,
    )
    return _teacher_fallback_reward(type(last_exc).__name__)


def _teacher_fallback_reward(reason: str) -> dict:
    return {"_opd_teacher_fallback": True, "_opd_teacher_fallback_reason": reason}


def _use_student_log_probs_for_teacher(sample: Sample, reason: str) -> None:
    resp_len = sample.response_length
    if resp_len == 0:
        values = torch.empty(0, dtype=torch.float32)
    else:
        rollout_lps = sample.rollout_log_probs
        if rollout_lps is None:
            values = torch.zeros(resp_len, dtype=torch.float32)
        else:
            values = torch.as_tensor(rollout_lps, dtype=torch.float32).flatten()
            if values.numel() >= resp_len:
                values = values[-resp_len:]
            else:
                values = torch.cat([torch.zeros(resp_len - values.numel(), dtype=torch.float32), values])
    sample.teacher_log_probs = values

    sample.opd_teacher_fallback = True
    sample.opd_teacher_fallback_reason = reason
    sample.metadata["opd_teacher_fallback"] = True
    sample.metadata["opd_teacher_fallback_reason"] = reason


def _load_student_tokenizer(args):
    if not hasattr(args, "_opd_student_tok"):
        from transformers import AutoTokenizer

        args._opd_student_tok = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    return args._opd_student_tok


def _get_opd_mask_token_sequences(args) -> list[list[int]]:
    token_strings = getattr(args, "opd_mask_teacher_logprob_tokens", None)
    if not token_strings:
        return []

    cached = getattr(args, "_opd_mask_teacher_logprob_token_ids", None)
    if cached is not None:
        return cached

    tokenizer = _load_student_tokenizer(args)
    sequences = []
    for token_string in token_strings:
        token_ids = tokenizer.encode(token_string, add_special_tokens=False)
        if not token_ids:
            logger.warning("Skipping empty OPD teacher-logprob mask token string: %r", token_string)
            continue
        sequences.append(token_ids)

    args._opd_mask_teacher_logprob_token_ids = sequences
    return sequences


def _iter_matching_token_positions(tokens: list[int], sequences: list[list[int]]):
    for sequence in sequences:
        seq_len = len(sequence)
        if seq_len == 0 or seq_len > len(tokens):
            continue
        for start in range(0, len(tokens) - seq_len + 1):
            if tokens[start : start + seq_len] == sequence:
                yield from range(start, start + seq_len)


def _ensure_opd_logprob_ignore_mask(sample: Sample) -> list[float]:
    resp_len = sample.response_length
    mask = getattr(sample, "opd_logprob_ignore_mask", None)
    if mask is None:
        mask = [0.0] * resp_len
    else:
        mask = list(mask)[-resp_len:]
        if len(mask) < resp_len:
            mask = [0.0] * (resp_len - len(mask)) + mask
    sample.opd_logprob_ignore_mask = mask
    return mask


def _mark_opd_logprob_ignore_positions(sample: Sample, positions, context: str) -> None:
    mask = _ensure_opd_logprob_ignore_mask(sample)
    marked = 0
    for pos in positions:
        if 0 <= pos < len(mask):
            if mask[pos] == 0.0:
                marked += 1
            mask[pos] = 1.0
    if marked:
        sample.metadata["opd_logprob_ignore_masked"] = sample.metadata.get("opd_logprob_ignore_masked", 0) + marked
        sample.metadata["opd_logprob_ignore_context"] = context


def _is_unmatched_logprob_source_missing(args, sample: Sample) -> bool:
    return getattr(args, "opd_unmatched_logprob_source", "rollout") == "logprobs" and not sample.logprobs


def _get_unmatched_log_probs(args, sample: Sample, *, zero_fill_if_missing: bool, context: str):
    source = getattr(args, "opd_unmatched_logprob_source", "rollout")
    if source not in {"rollout", "logprobs"}:
        raise ValueError(f"Unsupported --opd-unmatched-logprob-source: {source!r}")

    source_attr = "logprobs" if source == "logprobs" else "rollout_log_probs"
    values = getattr(sample, source_attr, None)

    resp_len = sample.response_length
    if not values:
        if source != "logprobs":
            logger.warning(
                "Cannot get OPD unmatched replacement log-probs from %s for sample index=%s during %s.",
                source_attr,
                getattr(sample, "index", None),
                context,
            )
        return [0.0] * resp_len if zero_fill_if_missing else None

    values = list(values)[-resp_len:]
    if len(values) < resp_len:
        values = [0.0] * (resp_len - len(values)) + values
    return values


def _mask_teacher_logprobs_with_student(args, sample: Sample, teacher_log_probs):
    sequences = _get_opd_mask_token_sequences(args)
    if not sequences or sample.response_length == 0:
        return teacher_log_probs

    resp_len = sample.response_length
    prompt_len = len(sample.tokens) - resp_len
    response_tokens = sample.tokens[prompt_len:]
    masked_positions = sorted(set(_iter_matching_token_positions(response_tokens, sequences)))
    if not masked_positions:
        return teacher_log_probs

    replacement_log_probs = _get_unmatched_log_probs(
        args,
        sample,
        zero_fill_if_missing=False,
        context="teacher-logprob token masking",
    )
    if replacement_log_probs is None:
        if _is_unmatched_logprob_source_missing(args, sample):
            _mark_opd_logprob_ignore_positions(sample, masked_positions, "teacher-logprob token masking")
        return teacher_log_probs

    if isinstance(teacher_log_probs, torch.Tensor):
        masked = teacher_log_probs.clone()
        for pos in masked_positions:
            if pos < masked.numel() and pos < len(replacement_log_probs):
                masked[pos] = float(replacement_log_probs[pos])
        return masked

    masked = list(teacher_log_probs)
    for pos in masked_positions:
        if pos < len(masked) and pos < len(replacement_log_probs):
            masked[pos] = float(replacement_log_probs[pos])
    return masked


async def reward_func(args, sample, **kwargs):
    if kwargs.get("evaluation"):
        from slime.rollout.rm_hub import async_rm as default_async_rm

        saved = args.custom_rm_path
        args.custom_rm_path = None
        try:
            return await default_async_rm(args, sample, **kwargs)
        finally:
            args.custom_rm_path = saved

    teacher_url = getattr(args, "teacher_url", None) or args.rm_url
    payload = {
        # "text": sample.prompt + sample.response,
        "input_ids": sample.tokens,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }
    return await _post_teacher_json(args, teacher_url, payload, sample, "OPD teacher request")


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    raw_rewards = [
        sample.reward
        if isinstance(sample.reward, dict) and sample.reward.get("_opd_teacher_fallback")
        else sample.get_reward_value(args)
        for sample in samples
    ]
    response_lengths = [sample.response_length for sample in samples]

    for sample, reward in zip(samples, raw_rewards, strict=False):
        if isinstance(reward, dict) and reward.get("_opd_teacher_fallback"):
            _use_student_log_probs_for_teacher(sample, reward.get("_opd_teacher_fallback_reason", "teacher_failed"))

    # Extract teacher log-probs from the sglang response
    teacher_log_probs = [
        torch.tensor([item[0] for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for sample, reward in zip(samples, raw_rewards, strict=False)
        if not (isinstance(reward, dict) and reward.get("_opd_teacher_fallback"))
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(
            teacher_log_probs,
            [
                length
                for reward, length in zip(raw_rewards, response_lengths, strict=False)
                if not (isinstance(reward, dict) and reward.get("_opd_teacher_fallback"))
            ],
            strict=False,
        )
    ]

    samples_with_teacher = [
        sample
        for sample, reward in zip(samples, raw_rewards, strict=False)
        if not (isinstance(reward, dict) and reward.get("_opd_teacher_fallback"))
    ]
    for sample, t_log_probs in zip(samples_with_teacher, teacher_log_probs, strict=False):
        t_log_probs = _mask_teacher_logprobs_with_student(args, sample, t_log_probs)
        sample.teacher_log_probs = t_log_probs

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards


# ═══════════════════════════════════════════════════════════════════════════
#  Cross-vocabulary OPD support
# ═══════════════════════════════════════════════════════════════════════════


def _load_cross_vocab_tokenizers(args):
    """Lazy-load teacher and student tokenizers, caching them on args."""
    if not hasattr(args, "_cross_vocab_student_tok"):
        from transformers import AutoTokenizer

        teacher_path = getattr(args, "teacher_tokenizer_path", None)
        if not teacher_path:
            raise ValueError(
                "--teacher-tokenizer-path is required for cross-vocabulary OPD. "
                "It must point to the teacher model's tokenizer directory."
            )

        args._cross_vocab_student_tok = AutoTokenizer.from_pretrained(
            args.hf_checkpoint, trust_remote_code=True
        )
        args._cross_vocab_teacher_tok = AutoTokenizer.from_pretrained(
            teacher_path, trust_remote_code=True
        )


def _build_cross_vocab_teacher_input(args, teacher_tok, sample: Sample) -> tuple[str, list[int]]:
    """Build exactly the text scored by both cross-vocabulary teacher APIs."""
    messages = (
        sample.origin_prompt
        if isinstance(sample.origin_prompt, list)
        else [{"role": "user", "content": sample.origin_prompt}]
    )
    if getattr(teacher_tok, "chat_template", None):
        teacher_prompt = teacher_tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        encode_messages = getattr(args, "_cross_vocab_teacher_encode_messages", None)
        if encode_messages is None:
            encoder_path = Path(args.teacher_tokenizer_path) / "encoding" / "encoding_dsv4.py"
            if not encoder_path.is_file():
                raise ValueError(
                    "The teacher tokenizer has no chat template and no DeepSeek-V4 encoding/encoding_dsv4.py."
                )
            spec = importlib.util.spec_from_file_location("_slime_deepseek_v4_encoding", encoder_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load DeepSeek-V4 encoder from {encoder_path}.")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            encode_messages = module.encode_messages
            args._cross_vocab_teacher_encode_messages = encode_messages
        teacher_prompt = encode_messages(messages, thinking_mode="thinking")
    teacher_prompt_ids = teacher_tok.encode(teacher_prompt, add_special_tokens=True)
    return teacher_prompt + sample.response, teacher_prompt_ids


def _align_common_tokens_1to1(tea_texts, tea_lps, stu_texts, fallback_lps):
    """Align teacher logprobs to student positions using 1:1 token matching.

    Cumulative-text greedy matching: both sides walk their token sequences
    simultaneously, accumulating decoded text character-by-character.  A
    teacher logprob is substituted at student position j when:

    1.  The accumulated text on both sides is identical (``hist_tea == hist_stu``),
        meaning they have consumed the same raw text up to this point.
    2.  The current single-token decoded texts are the same
        (``t_text == s_text``), meaning the teacher and student tokenizers
        happen to produce the same token boundary at this position.

    When the accumulated texts differ (different token boundaries), the
    shorter side advances alone until it catches up or both sides diverge
    again (no match for those positions).

    Non-matching student positions keep their fallback log-probs,
    which makes the downstream reverse-KL penalty effectively zero at those
    positions when fallback log-probs match the student log-probs used by OPD.

    Args:
        tea_texts:  Per-token decoded text from teacher (list[str])
        tea_lps:    Teacher logprobs aligned with ``tea_texts`` (list[float])
        stu_texts:  Per-token decoded text from student (list[str])
        fallback_lps: Log-probs used when no 1:1 teacher token match exists,
                      same length as ``stu_texts``.

    Returns:
        (aligned, num_matched, matched_mask):
            aligned     — list[float] of length ``len(stu_texts)``, where
                          position j is either the teacher logprob (if matched)
                          or the student rollout logprob (fallback).
            num_matched — int, number of student positions where teacher
                          logprob was substituted.
            matched_mask — list[bool], True where teacher logprob was substituted.
    """
    n_stu = len(stu_texts)
    aligned = list(fallback_lps)  # initialize all positions with fallback values
    matched_mask = [False] * n_stu

    i = j = 0               # indices into teacher / student token lists
    hist_tea = ""           # accumulated decoded text on teacher side
    hist_stu = ""           # accumulated decoded text on student side
    num_matched = 0

    while i < len(tea_texts) and j < n_stu:
        t_text = tea_texts[i] or ""     # teacher token text (defensive: None → "")
        s_text = stu_texts[j] or ""     # student token text

        if hist_tea == hist_stu and t_text == s_text:
            # Same accumulated text + same single-token text = 1:1 match
            aligned[j] = tea_lps[i]
            matched_mask[j] = True
            num_matched += 1
            hist_tea += t_text
            hist_stu += s_text
            i += 1
            j += 1
        elif len(hist_tea) > len(hist_stu):
            # Teacher has consumed more text; advance student to catch up
            hist_stu += s_text
            j += 1
        elif len(hist_tea) < len(hist_stu):
            # Student has consumed more text; advance teacher to catch up
            hist_tea += t_text
            i += 1
        else:
            # Same accumulated text length but single-token texts differ
            # (different token boundaries).  Advance both — the accumulations
            # will diverge, and the shorter side will catch up in later steps.
            hist_tea += t_text
            hist_stu += s_text
            i += 1
            j += 1

    return aligned, num_matched, matched_mask


async def reward_func_cross_vocab(args, sample: Sample, **kwargs):
    """Send decoded text to teacher SGLang for cross-vocabulary OPD.

    Unlike ``reward_func`` (which sends student token IDs and requires
    identical vocabularies), this decodes ``sample.tokens`` to text so
    the teacher can tokenize with its own vocabulary.  Appends
    ``_cross_meta`` with ``teacher_prompt_len`` for post-processing.

    Requires ``--teacher-tokenizer-path`` to point to the teacher
    tokenizer directory (defaults to ``args.hf_checkpoint``).
    """
    if kwargs.get("evaluation"):
        from slime.rollout.rm_hub import async_rm as default_async_rm

        saved = args.custom_rm_path
        args.custom_rm_path = None
        try:
            return await default_async_rm(args, sample, **kwargs)
        finally:
            args.custom_rm_path = saved

    teacher_url = getattr(args, "teacher_url", None) or args.rm_url
    _load_cross_vocab_tokenizers(args)
    teacher_tok = args._cross_vocab_teacher_tok

    full_text, teacher_prompt_ids = _build_cross_vocab_teacher_input(args, teacher_tok, sample)

    payload = {
        "text": full_text,
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": 0,
    }

    result = await _post_teacher_json(args, teacher_url, payload, sample, "Cross-vocab OPD teacher request")
    if isinstance(result, dict) and result.get("_opd_teacher_fallback"):
        return result

    result["_cross_meta"] = {"teacher_prompt_len": len(teacher_prompt_ids)}
    return result


def _normalize_completion_prompt_logprobs(result: dict) -> list[list]:
    try:
        logprobs = result["choices"][0]["logprobs"]
        tokens = logprobs["tokens"]
        token_logprobs = logprobs["token_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("Teacher Completions response has no echoed token logprobs.") from exc
    if not isinstance(tokens, list) or not isinstance(token_logprobs, list) or len(tokens) != len(token_logprobs):
        raise ValueError("Teacher Completions response has invalid tokens/token_logprobs arrays.")

    normalized = []
    for index, (token, logprob) in enumerate(zip(tokens, token_logprobs, strict=True)):
        if not isinstance(token, str) or not token.startswith("token_id:"):
            raise ValueError(f"Teacher token {index} is not encoded as token_id:<id>.")
        try:
            token_id = int(token.removeprefix("token_id:"))
        except ValueError as exc:
            raise ValueError(f"Teacher token {index} has an invalid token id.") from exc
        if logprob is not None and (not isinstance(logprob, (int, float)) or isinstance(logprob, bool)):
            raise ValueError(f"Teacher token {index} has an invalid logprob.")
        normalized.append([None if logprob is None else float(logprob), token_id])
    return normalized


async def reward_func_cross_vocab_chat_completions(args, sample: Sample, **kwargs):
    """Score ``teacher_chat_template(prompt) + sample.response`` via vLLM."""
    if kwargs.get("evaluation"):
        from slime.rollout.rm_hub import async_rm as default_async_rm

        saved = args.custom_rm_path
        args.custom_rm_path = None
        try:
            return await default_async_rm(args, sample, **kwargs)
        finally:
            args.custom_rm_path = saved

    teacher_base_url = getattr(args, "teacher_url", None) or getattr(args, "rm_url", None)
    if not teacher_base_url:
        raise ValueError("--teacher-url or --rm-url is required for OpenAI Completions OPD.")
    teacher_model_name = getattr(args, "teacher_model_name", None)
    if not teacher_model_name:
        raise ValueError("--teacher-model-name is required for OpenAI Completions OPD.")

    _load_cross_vocab_tokenizers(args)
    full_text, teacher_prompt_ids = _build_cross_vocab_teacher_input(args, args._cross_vocab_teacher_tok, sample)
    payload = {
        "model": teacher_model_name,
        "prompt": full_text,
        "temperature": 0,
        "max_tokens": 0,
        "echo": True,
        "logprobs": 1,
        "return_tokens_as_token_ids": True,
        "skip_special_tokens": False,
    }
    teacher_api_key = getattr(args, "teacher_api_key", None)
    headers = {"Authorization": f"Bearer {teacher_api_key}"} if teacher_api_key else None
    result = await _post_teacher_json(
        args,
        teacher_base_url,
        payload,
        sample,
        "Cross-vocab OPD Completions teacher request",
        headers=headers,
    )
    if isinstance(result, dict) and result.get("_opd_teacher_fallback"):
        return result

    input_token_logprobs = _normalize_completion_prompt_logprobs(result)
    returned_prompt_ids = [item[1] for item in input_token_logprobs[: len(teacher_prompt_ids)]]
    if returned_prompt_ids != teacher_prompt_ids:
        raise ValueError("Teacher API tokenization does not match --teacher-tokenizer-path.")
    result["meta_info"] = {"input_token_logprobs": input_token_logprobs}
    result["_cross_meta"] = {"teacher_prompt_len": len(teacher_prompt_ids)}
    return result


# Keep the originally documented singular name as a compatibility alias.
reward_func_cross_vocab_chat_completion = reward_func_cross_vocab_chat_completions


def post_process_rewards_cross_vocab(args, samples: list[Sample], **kwargs):
    """Align teacher logprobs to student positions for cross-vocab OPD.

    Only OPD on positions where teacher and student have identical
    decoded single tokens at the same text position (1:1 match).
    Non-matching positions rely on --opd-unmatched-logprob-source fallback
    log-probs.

    Requires the same ``--teacher-tokenizer-path`` as
    ``reward_func_cross_vocab``.
    """
    raw_rewards = [
        sample.reward
        if isinstance(sample.reward, dict) and sample.reward.get("_opd_teacher_fallback")
        else sample.get_reward_value(args)
        for sample in samples
    ]
    has_teacher_rewards = any(
        not (isinstance(reward, dict) and reward.get("_opd_teacher_fallback")) for reward in raw_rewards
    )
    if has_teacher_rewards:
        _load_cross_vocab_tokenizers(args)
        student_tok = args._cross_vocab_student_tok
        teacher_tok = args._cross_vocab_teacher_tok

    for sample, reward in zip(samples, raw_rewards, strict=True):
        resp_len = sample.response_length
        teacher_success = not (isinstance(reward, dict) and reward.get("_opd_teacher_fallback"))
        sample.cross_vocab_teacher_success = teacher_success
        if resp_len == 0:
            sample.teacher_log_probs = torch.empty(0, dtype=torch.float32)
            continue

        if not teacher_success:
            _use_student_log_probs_for_teacher(sample, reward.get("_opd_teacher_fallback_reason", "teacher_failed"))
            continue

        all_info = reward["meta_info"]["input_token_logprobs"]

        meta = reward["_cross_meta"]
        t_prompt_len = meta["teacher_prompt_len"]
        t_resp_info = all_info[t_prompt_len:]
        teacher_lps = [item[0] for item in t_resp_info]
        teacher_texts = [teacher_tok.convert_ids_to_tokens([item[1]])[0] or "" for item in t_resp_info]

        if not teacher_lps:
            sample.teacher_log_probs = torch.empty(0, dtype=torch.float32)
            continue

        prompt_len = len(sample.tokens) - resp_len
        s_resp_ids = sample.tokens[prompt_len:]
        stu_texts = [student_tok.convert_ids_to_tokens([tid])[0] for tid in s_resp_ids]

        fallback_lps = _get_unmatched_log_probs(
            args,
            sample,
            zero_fill_if_missing=True,
            context="cross-vocab OPD unmatched tokens",
        )

        aligned_lps, num_matched, matched_mask = _align_common_tokens_1to1(
            teacher_texts, teacher_lps, stu_texts, fallback_lps
        )
        if _is_unmatched_logprob_source_missing(args, sample):
            _mark_opd_logprob_ignore_positions(
                sample,
                [pos for pos, matched in enumerate(matched_mask) if not matched],
                "cross-vocab OPD unmatched tokens",
            )
        aligned_lps = _mask_teacher_logprobs_with_student(args, sample, aligned_lps)

        sample.teacher_log_probs = torch.tensor(aligned_lps, dtype=torch.float32)
        sample.cross_vocab_token_overlap = num_matched / resp_len if resp_len > 0 else 0.0

    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards


def post_process_rewards_cross_vocab_chat_completion(args, samples: list[Sample], **kwargs):
    """Reuse the existing cross-vocabulary response-token alignment."""
    return post_process_rewards_cross_vocab(args, samples, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
#  Sequence-level OPD
# ═══════════════════════════════════════════════════════════════════════════


def _compute_seq_rewards(samples: list[Sample]) -> list[float]:
    """Compute per-sample sequence-level mean reverse-KL as scalar reward.

    reward_i = mean(teacher_log_probs[i] - rollout_log_probs[i])
    """
    seq_rewards = []
    for sample in samples:
        resp_len = sample.response_length
        if resp_len == 0:
            seq_rewards.append(0.0)
            continue
        teacher_t = sample.teacher_log_probs
        if isinstance(teacher_t, list):
            teacher_t = torch.tensor(teacher_t, dtype=torch.float32)
        student_raw = sample.rollout_log_probs or [0.0] * resp_len
        student_t = torch.as_tensor(list(student_raw)[-resp_len:], dtype=torch.float32)
        delta = teacher_t - student_t
        ignore_mask = getattr(sample, "opd_logprob_ignore_mask", None)
        if ignore_mask is not None:
            keep_mask = 1.0 - torch.as_tensor(list(ignore_mask)[-resp_len:], dtype=torch.float32)
            if keep_mask.numel() < delta.numel():
                keep_mask = torch.cat([torch.ones(delta.numel() - keep_mask.numel()), keep_mask])
            keep_mask = keep_mask[: delta.numel()]
            keep_count = keep_mask.sum()
            seq_rewards.append(float((delta * keep_mask).sum() / keep_count) if keep_count > 0 else 0.0)
        else:
            seq_rewards.append(float(delta.mean()))
    return seq_rewards


def post_process_rewards_sequence_opd(args, samples: list[Sample], **kwargs):
    """Sequence-level OPD post-process: mean reverse-KL per sequence as GRPO reward.

    Reuses the original ``post_process_rewards`` or ``post_process_rewards_cross_vocab``
    for teacher-logprob extraction, then replaces the scalar zero rewards with the
    sequence-level mean(teacher_log_probs − rollout_log_probs) and applies GRPO
    group-normalization.

    Compatible with both ``reward_func`` (same-vocab) and ``reward_func_cross_vocab``
    (cross-vocab).  Auto-detects cross-vocab mode from ``_cross_meta`` in reward dicts.

    Pipeline:
        --custom-rm-path slime.rollout.on_policy_distillation.reward_func
        --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards_sequence_opd

    or for cross-vocab:
        --custom-rm-path slime.rollout.on_policy_distillation.reward_func_cross_vocab
        --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards_sequence_opd
    """
    is_cross_vocab = any(
        not (isinstance(s.reward, dict) and s.reward.get("_opd_teacher_fallback"))
        and isinstance(s.get_reward_value(args), dict)
        and s.get_reward_value(args).get("_cross_meta") is not None
        for s in samples
    )

    if is_cross_vocab:
        post_process_rewards_cross_vocab(args, samples, **kwargs)
    else:
        post_process_rewards(args, samples, **kwargs)

    raw_rewards = _compute_seq_rewards(samples)

    should_normalize = (
        args.rewards_normalization
        and args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
    )
    if not should_normalize:
        return raw_rewards, raw_rewards

    rewards = torch.tensor(raw_rewards, dtype=torch.float)
    if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
        rewards = rewards.reshape(-1, args.n_samples_per_prompt)
    else:
        rewards = rewards.view(-1, rewards.shape[-1])
    mean = rewards.mean(dim=-1, keepdim=True)
    rewards = rewards - mean

    if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
        std = rewards.std(dim=-1, keepdim=True)
        rewards = rewards / (std + 1e-6)

    return raw_rewards, rewards.flatten().tolist()