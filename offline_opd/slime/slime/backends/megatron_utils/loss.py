import logging
import math
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_cispo_loss,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
    vocab_parallel_logprobs_and_entropy_on_ids,
    vocab_parallel_logprobs_on_ids,
    vocab_parallel_topk_ids,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_concat_loss_mask,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)

logger = logging.getLogger(__name__)

ROLLOUT_TOP_P_TOKEN_KEYS = (
    "rollout_top_p_token_ids",
    "rollout_top_p_token_offsets",
)

# Prefix marking a reported-loss key that must be reduced by MAX (not the default
# additive SUM-then-divide). ``loss_function`` strips this prefix and routes the
# value into a separate max channel that ``reduce_train_step_metrics`` all-reduces
# with op=MAX. Inner losses using ``custom_loss_function_path`` need not touch it:
# without the prefix a metric just flows through the normal additive channel.
MAX_METRIC_PREFIX = "__max__"

# Clamp bound (in log space) for the χ² off-policy diagnostic, mirroring
# uni-agent/verl's rollout_corr_helper.SAFETY_BOUND: exp(20) ~= 4.85e8 caps the
# squared IS weight so a few extreme tokens can't overflow the running sum.
OFFPOLICY_SAFETY_BOUND = 20.0

# Finite stand-in for a -inf teacher log-prob in the top-k OPD losses. exp(-100)
# underflows to 0 in float32, so substituting it changes no probability-space quantity
# while keeping the log-ratio (and therefore the gradient) finite.
_LOGPROB_FLOOR = -100.0

# Widths for the overlap@m head-agreement diagnostics of the top-k OPD losses (see
# _opd_topk_kl_loss). The full-width opd_topk_overlap (m = K) is a TAIL-set metric: with
# K = 1024 most positions are near-deterministic, their ranks 2..K are numerical noise, and
# the loss weights them by ~0 -- so it sits around 0.55 and barely moves while the head is
# already matched. These small m measure the part of the distribution the loss actually has
# gradient on.
_OVERLAP_AT_M = (1, 2, 4, 8,16,32,64)
_STUDENT_ANCHOR_REASON_NAMES = {
    0: "tokenizer_or_template_mismatch",
    1: "reasoning_prefix",
    2: "structural_newline",
    3: "observation_boundary",
    4: "reasoning_first_token",
    5: "reasoning_end",
    6: "post_reasoning_transition",
    7: "tool_protocol_token",
    8: "stop_token",
}


def get_rollout_top_p_logprob_kwargs(args: Namespace, batch: dict[str, Any]) -> dict[str, Any]:
    if args.rollout_top_p == 1.0:
        return {}

    top_p_token_ids = batch.get("rollout_top_p_token_ids")
    top_p_token_offsets = batch.get("rollout_top_p_token_offsets")
    if top_p_token_ids is None or top_p_token_offsets is None:
        raise ValueError("rollout_top_p != 1.0 requires rollout_top_p_token_ids and rollout_top_p_token_offsets.")
    return {
        "top_p_token_ids": top_p_token_ids,
        "top_p_token_offsets": top_p_token_offsets,
    }


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    apply_temperature: bool = True,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and optionally applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for optional scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.
        apply_temperature: Whether to divide outputs by `rollout_temperature`.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"
    assert logits.size(0) == 1, f"{logits.shape}"
    logits = logits.squeeze(0)

    if apply_temperature and args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for tokens, total_length, response_length in zip(unconcat_tokens, total_lengths, response_lengths, strict=False):
        if cp_size == 1:
            end += total_length
            start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits_local_len: int,
    total_lengths: list[int],
    response_lengths: list[int],
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits_local_len: Local sequence length on this rank.
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Skip keys where all values are None (e.g. entropy when not computed)
        if all(v is None for v in values):
            continue

        # Determine reference dtype/device from first non-None value
        ref_value = next(v for v in values if v is not None)
        ref_dtype = ref_value.dtype
        ref_device = ref_value.device

        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if value is None or e <= s:
                # This rank has no response logprobs for this sample
                full_resp = torch.zeros(
                    response_length,
                    dtype=ref_dtype,
                    device=ref_device,
                    requires_grad=ref_value.requires_grad,
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                full_resp = F.pad(value, (resp_start, response_length - resp_end))

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks.
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for full_resp, total_length, response_length in zip(
            all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False
        ):
            new_values.append(slice_log_prob_with_cp(full_resp, total_length, response_length))

        res[key] = new_values


def _build_shifted_tokens(
    T: int,
    device: torch.device,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    allgather_cp: bool,
) -> torch.Tensor:
    """Build shifted target tokens for the full packed/padded logits."""
    cp_size = mpu.get_context_parallel_world_size()

    # --- zigzag CP: completely different layout ---
    if cp_size > 1 and not allgather_cp:
        full_tokens = torch.zeros(T, dtype=torch.long, device=device)
        end = 0
        for tokens, total_length, response_length in zip(
            unconcat_tokens, total_lengths, response_lengths, strict=False
        ):
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )
            for half, base in ((0, end), (1, end + chunk_size_cp)):
                lo = logits_offset[half][0] - chunks_offset[half][0]
                hi = logits_offset[half][1] - chunks_offset[half][0]
                full_tokens[base + lo : base + hi] = tokens[tokens_offset[half][0] : tokens_offset[half][1]]
            end += 2 * chunk_size_cp
        return full_tokens

    # --- cp1 and allgather-CP both build global shifted tokens the same way ---
    T_global = sum(total_lengths) if allgather_cp else T
    full_tokens = torch.zeros(T_global, dtype=torch.long, device=device)

    offset = 0
    for tokens, total_length in zip(unconcat_tokens, total_lengths, strict=False):
        full_tokens[offset : offset + total_length - 1] = tokens[1:total_length]
        offset += total_length

    # allgather-CP: slice to local chunk
    if allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        chunk_start = cp_rank * T
        chunk_end = chunk_start + T
        if chunk_end <= T_global:
            return full_tokens[chunk_start:chunk_end].contiguous()
        local = torch.zeros(T, dtype=torch.long, device=device)
        valid = T_global - chunk_start
        if valid > 0:
            local[:valid] = full_tokens[chunk_start:]
        return local

    return full_tokens


def _fill_topp_mask_rows(
    keep: torch.Tensor,
    ids: list[int],
    offsets: list[int],
    response_start: int,
    local_start: int,
    length: int,
    vocab_start: int,
    vocab_end: int,
) -> None:
    end = min(response_start + length, max(len(offsets) - 1, 0))
    for response_idx in range(response_start, end):
        local_ids = [
            token_id - vocab_start
            for token_id in ids[offsets[response_idx] : offsets[response_idx + 1]]
            if vocab_start <= token_id < vocab_end
        ]
        row = local_start + response_idx - response_start
        keep[row].fill_(False)
        if local_ids:
            keep[row, torch.tensor(local_ids, device=keep.device, dtype=torch.long)] = True


def _build_topp_keep_mask(
    T: int,
    vocab_local: int,
    device: torch.device,
    top_p_token_ids: list[list[int]],
    top_p_token_offsets: list[list[int]],
    total_lengths: list[int],
    response_lengths: list[int],
    allgather_cp: bool,
) -> torch.Tensor:
    """Build a ``[T, vocab_local]`` boolean keep-mask aligned to local logits.

    For response token ``r`` of a sample, the rollout top-p nucleus is
    ``ids[offsets[r]:offsets[r + 1]]``. Rows without a recorded nucleus stay
    all-True, so only response rows with replay data are masked.
    """
    cp_size = mpu.get_context_parallel_world_size()
    tp_rank = mpu.get_tensor_model_parallel_rank()
    vocab_start = tp_rank * vocab_local
    vocab_end = vocab_start + vocab_local

    # Normalize ragged payloads (may arrive as CPU int32 tensors) to python lists.
    top_p_token_ids = [t.tolist() if torch.is_tensor(t) else list(t) for t in top_p_token_ids]
    top_p_token_offsets = [t.tolist() if torch.is_tensor(t) else list(t) for t in top_p_token_offsets]

    keep = torch.ones((T, vocab_local), dtype=torch.bool, device=device)

    if cp_size > 1 and not allgather_cp:
        local_base = 0
        for ids, offsets, total_length, response_length in zip(
            top_p_token_ids, top_p_token_offsets, total_lengths, response_lengths, strict=False
        ):
            prompt_length = total_length - response_length
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )
            for half, base in ((0, local_base), (1, local_base + chunk_size_cp)):
                local_start = base + logits_offset[half][0] - chunks_offset[half][0]
                length = logits_offset[half][1] - logits_offset[half][0]
                response_start = tokens_offset[half][0] - prompt_length
                _fill_topp_mask_rows(keep, ids, offsets, response_start, local_start, length, vocab_start, vocab_end)
            local_base += 2 * chunk_size_cp
        return keep

    if allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        chunk_start = cp_rank * T
        chunk_end = chunk_start + T
        seq_start = 0
        for ids, offsets, total_length, response_length in zip(
            top_p_token_ids, top_p_token_offsets, total_lengths, response_lengths, strict=False
        ):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1
            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e > s:
                _fill_topp_mask_rows(
                    keep,
                    ids,
                    offsets,
                    s - logit_global_start,
                    s - chunk_start,
                    e - s,
                    vocab_start,
                    vocab_end,
                )
            seq_start += total_length
        return keep

    offset = 0
    for ids, offsets, total_length, response_length in zip(
        top_p_token_ids, top_p_token_offsets, total_lengths, response_lengths, strict=False
    ):
        end = offset + total_length
        start = end - response_length
        _fill_topp_mask_rows(keep, ids, offsets, 0, start - 1, response_length, vocab_start, vocab_end)
        offset += total_length

    return keep


def _extract_per_sample(
    log_prob_full: torch.Tensor,
    entropy_full: torch.Tensor | None,
    total_lengths: list[int],
    response_lengths: list[int],
    allgather_cp: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Slice per-sample response log-probs/entropy from full-length 1-D tensors."""
    cp_size = mpu.get_context_parallel_world_size()
    log_probs_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor] = []

    if cp_size > 1 and not allgather_cp:
        # zigzag CP
        pos = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            chunk_size_cp, chunks_offset, logits_offset, _tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )
            lo0 = logits_offset[0][0] - chunks_offset[0][0]
            hi0 = logits_offset[0][1] - chunks_offset[0][0]
            lo1 = logits_offset[1][0] - chunks_offset[1][0]
            hi1 = logits_offset[1][1] - chunks_offset[1][0]

            lp = torch.cat(
                [
                    log_prob_full[pos + lo0 : pos + hi0],
                    log_prob_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                ],
                dim=0,
            )
            log_probs_list.append(lp)
            if entropy_full is not None:
                ent = torch.cat(
                    [
                        entropy_full[pos + lo0 : pos + hi0],
                        entropy_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                    ],
                    dim=0,
                )
                entropy_list.append(ent)
            pos += 2 * chunk_size_cp

    elif allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        local_len = log_prob_full.size(0)
        chunk_start = cp_rank * local_len
        chunk_end = chunk_start + local_len

        seq_start = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                log_probs_list.append(log_prob_full[:0])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[:0])
            else:
                log_probs_list.append(log_prob_full[s - chunk_start : e - chunk_start])
                if entropy_full is not None:
                    entropy_list.append(entropy_full[s - chunk_start : e - chunk_start])
            seq_start += total_length

    else:
        # cp1
        offset = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            end = offset + total_length
            start = end - response_length
            log_probs_list.append(log_prob_full[start - 1 : end - 1])
            if entropy_full is not None:
                entropy_list.append(entropy_full[start - 1 : end - 1])
            offset += total_length

    return log_probs_list, entropy_list


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    top_p_token_ids: list[list[int]] | None = None,
    top_p_token_offsets: list[list[int]] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    Computes on the **full** logits ``[T, V]`` tensor at once (instead of
    per-sample slicing) so backward traverses ``[T, V]`` only once, then
    extracts per-sample response portions.

    If rollout top-p replay is provided, the keep-mask is applied only to
    log-probabilities; entropy is always computed from the unmasked logits.
    """
    assert non_loss_data
    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"
    assert logits.size(0) == 1, f"{logits.shape}"
    logits = logits.squeeze(0)

    # Apply rollout temperature scaling to logits to match rollout-time log-probs.
    rollout_temperature = getattr(args, "rollout_temperature", 1.0)
    if rollout_temperature != 1.0:
        logits = logits / rollout_temperature
    logits = logits.contiguous()
    T = logits.size(0)
    device = logits.device
    tp_group = mpu.get_tensor_model_parallel_group()
    chunk_size = args.log_probs_chunk_size
    # Keep entropy metrics, but skip saving entropy-backward activations when
    # the entropy term cannot affect the loss.
    with_entropy_grad = with_entropy and getattr(args, "entropy_coef", 0.0) != 0

    # --- build full shifted-token target tensor ---
    full_tokens = _build_shifted_tokens(T, device, unconcat_tokens, total_lengths, response_lengths, args.allgather_cp)

    # --- build top-p nucleus keep-mask (logprob only; entropy stays unmasked) ---
    top_p_keep_mask = None
    if top_p_token_ids is not None and top_p_token_offsets is not None:
        top_p_keep_mask = _build_topp_keep_mask(
            T,
            logits.size(-1),
            device,
            top_p_token_ids,
            top_p_token_offsets,
            total_lengths,
            response_lengths,
            args.allgather_cp,
        )

    # --- compute on full [T,V] logits at once via calculate_log_probs_and_entropy ---
    log_prob_full, entropy_full = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        with_entropy_grad=with_entropy_grad,
        chunk_size=chunk_size,
        log_prob_keep_mask=top_p_keep_mask,
    )
    log_prob_full = log_prob_full.squeeze(-1)  # [T, 1] -> [T]

    # --- extract per-sample response portions ---
    log_probs_list, entropy_list = _extract_per_sample(
        log_prob_full,
        entropy_full,
        total_lengths,
        response_lengths,
        args.allgather_cp,
    )

    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=T,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        )

    return torch.empty((0,), device=device), res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration passed to `get_responses`; temperature scaling is
            disabled for value outputs.
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        apply_temperature=False,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=logits.size(1),
            total_lengths=total_lengths,
            response_lengths=response_lengths,
        )

    return torch.empty((0,), device=logits.device), res


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `use_opd_adjust_advantage` and `opd_kl_coef`.
        rollout_data: Dict containing "teacher_log_probs".
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of student log-probability tensors.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """

    if student_log_probs is None:
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")

    device = student_log_probs[0].device
    teacher_log_probs = [t.to(device=device) for t in teacher_log_probs]

    reverse_kls = []
    for i, adv in enumerate(advantages):
        reverse_kl = student_log_probs[i] - teacher_log_probs[i]
        advantages[i] = adv - args.opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)

    # Store reverse KL for logging
    rollout_data["opd_reverse_kl"] = reverse_kls


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "cispo", "ppo",
    "reinforce_plus_plus", and "reinforce_plus_plus_baseline". When
    `args.normalize_advantages` is True, advantages are whitened across the
    data-parallel group using masked statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    If ``args.custom_advantage_function_path`` is set, it is called after KL computation
    and must populate ``rollout_data["advantages"]`` and
    ``rollout_data["returns"]``.

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
    """
    rollout_log_probs: list[torch.Tensor] | None = rollout_data.get("rollout_log_probs")
    log_probs: list[torch.Tensor] | None = (
        rollout_log_probs if args.use_rollout_logprobs else rollout_data.get("log_probs")
    )
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    # return when not the last pp stage.
    if not mpu.is_pipeline_last_stage():
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs or rollout_log_probs or values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]
    rollout_data["kl"] = kl

    if args.custom_advantage_function_path is not None:
        custom_adv_fn = load_function(args.custom_advantage_function_path)
        custom_adv_fn(args, rollout_data)
        advantages, returns = rollout_data["advantages"], rollout_data["returns"]

    elif args.advantage_estimator in ["grpo", "gspo", "cispo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # TODO: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        old_rewards = rewards
        rewards = []
        kl_coef = -args.kl_coef
        cp_rank = mpu.get_context_parallel_rank()
        for reward, k in zip(old_rewards, kl, strict=False):
            k *= kl_coef
            if cp_rank == 0:
                k[-1] += reward
            rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, rewards, args.gamma, args.lambd
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # Apply on-policy distillation KL penalty to advantages (orthogonal to advantage estimator)
    if args.use_opd_adjust_advantage:
        apply_opd_kl_to_advantages(
            args=args,
            rollout_data=rollout_data,
            advantages=advantages,
            student_log_probs=log_probs,
        )

    # TODO: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(total_len, response_len)

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = mpu.get_data_parallel_group()

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        **get_rollout_top_p_logprob_kwargs(args, batch),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    if not args.use_rollout_logprobs and not old_log_probs:
        old_log_probs = [log_prob.detach() for log_prob in log_probs]
    train_log_probs_for_tis = batch.get("log_probs")
    if not train_log_probs_for_tis:
        train_log_probs_for_tis = [log_prob.detach() for log_prob in log_probs]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        ppo_kl = old_log_probs - log_probs

    if args.advantage_estimator == "cispo":
        pg_loss, pg_clipfrac = compute_cispo_loss(ppo_kl, log_probs, advantages, args.eps_clip, args.eps_clip_high)
    else:
        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": train_log_probs_for_tis,
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with
        # modified_response_masks for numerator correction (rejected tokens
        # zeroed in pg_loss). Denominators stay the precomputed per-rollout
        # totals from ``rollout_mask_sums`` (based on original loss_masks) —
        # same normalizer as the outer reducer, so pg_loss and the rest of the
        # reported metrics live in the same per-rollout-mean space.
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            batch["rollout_mask_sums"],
            args.calculate_per_token_loss,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            importance_ratio = torch.exp(log_probs - old_log_probs)
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    rollout_probs_diff_accum = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        log_probs_to_compare = log_probs if args.use_rollout_logprobs else old_log_probs
        train_rollout_logprob_abs_diff = sum_of_sample_mean((log_probs_to_compare - rollout_log_probs).abs())
        # verl-aligned rollout-vs-train prob-space diagnostics, all on the
        # CONCATENATED, CP-aware, MASKED response tokens (observation/pad carry
        # rollout_log_prob 0.0 => exp 1.0 and are zeroed by the mask):
        #   * rollout_probs_diff_{mean,max,std}: |exp(logp_train)-exp(logp_rollout)|
        #     (verl training/rollout_probs_diff). mean/std ride the additive
        #     SUM-then-divide channel via sum/sqsum/count accumulators (the shared
        #     divisor cancels in the sum/count & sqsum/count ratios, so std is
        #     exact in both reduction modes); max rides the MAX channel.
        #   * rollout_actor_probs_pearson_corr: Pearson corr between the two
        #     engines' token probabilities (x=rollout, y=train). Every moment
        #     accumulator (sx, sy, sxx, syy, sxy, count) is additive, so corr is
        #     assembled from them in the reducer -- same divisor-cancels trick.
        #   * rollout_probs_diff_valid: 1 if any in-mask token exists (=> stats are
        #     computable), else 0 and the stats above report NaN.
        concat_mask = get_concat_loss_mask(
            batch["total_lengths"], batch["response_lengths"], batch["loss_masks"]
        )
        rollout_prob = rollout_log_probs.exp() * concat_mask  # x
        train_prob = log_probs_to_compare.exp() * concat_mask  # y
        prob_diff = (train_prob - rollout_prob).abs()
        local_max = (
            prob_diff.max() if prob_diff.numel() > 0 else prob_diff.new_tensor(float("-inf"))
        )
        # verl rollout_corr/* off-policy diagnostics in LOG space (uni-agent
        # compute_offpolicy_metrics). log_ratio = train - rollout on the SAME
        # concatenated tokens; each quantity is a masked sum over in-mask tokens,
        # so it rides the additive channel and shares rollout_probs_diff_count as
        # its divisor (the shared cp_factor/divisor cancels in sum/count):
        #   * rollout_offpolicy_kl   = mean(rollout - train) = KL(pi_rollout||pi_train)
        #   * rollout_offpolicy_k3_kl = mean(exp(dr) - dr - 1), dr = train - rollout
        #   * rollout_offpolicy_chi2_token = mean(rho^2) - 1, rho = exp(clamp(dr))
        # Masked-out tokens carry rollout_log_prob 0.0 and train logp <= 0, so
        # log_ratio <= 0 there and exp() can't overflow before the mask zeroes it;
        # chi2 additionally clamps to +-OFFPOLICY_SAFETY_BOUND like verl.
        log_ratio = log_probs_to_compare - rollout_log_probs  # train - rollout
        chi2_rho_sq = torch.exp(
            2.0 * log_ratio.clamp(-OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND)
        )
        rollout_probs_diff_accum = {
            "rollout_probs_diff_sum": prob_diff.sum(),
            "rollout_probs_diff_sqsum": (prob_diff * prob_diff).sum(),
            "rollout_probs_diff_count": concat_mask.sum(),
            "rollout_probs_pearson_sx": rollout_prob.sum(),
            "rollout_probs_pearson_sy": train_prob.sum(),
            "rollout_probs_pearson_sxx": (rollout_prob * rollout_prob).sum(),
            "rollout_probs_pearson_syy": (train_prob * train_prob).sum(),
            "rollout_probs_pearson_sxy": (rollout_prob * train_prob).sum(),
            "rollout_offpolicy_kl_sum": ((-log_ratio) * concat_mask).sum(),
            "rollout_offpolicy_k3_sum": (
                (torch.exp(log_ratio) - log_ratio - 1.0) * concat_mask
            ).sum(),
            "rollout_offpolicy_chi2_sqsum": (chi2_rho_sq * concat_mask).sum(),
            f"{MAX_METRIC_PREFIX}rollout_probs_diff_max": local_max,
        }

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()

    if rollout_probs_diff_accum is not None:
        for accum_key, accum_value in rollout_probs_diff_accum.items():
            reported_loss[accum_key] = accum_value.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    # Add OPD metric if available (advantage-adjustment path)
    if "opd_reverse_kl" in batch:
        opd_reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(opd_reverse_kl).clone().detach()

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def _opd_topk_kl_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    *,
    direction: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Truncated per-position KL OPD loss over a top-k support.

    Per response token, given a top-k support S (ids + teacher logprobs from the
    sglang teacher) and the student distribution pi_s (evaluated from the model logits
    at those same ids, TP-aware and differentiable). The WEIGHT is renormalized over S
    -- q(v) = pi(v) / sum_{u in S} pi(u) -- so the weights sum to 1 at every position,
    while the log-ratio keeps the raw full-vocab log-probs:

      * direction="forward"  (opd_loss_type=topk_forward_kl):
            S = the TEACHER's own top-k;
            L = sum_{v in S} q_t(v) * (log pi_t(v) - log pi_s(v))
        A weighted cross-entropy on the teacher's top-k. Un-normalized, the per-position
        gradient would scale with sum_{v in S} pi_t(v), silently down-weighting exactly
        the high-entropy positions whose top-k is least tight.

      * direction="reverse"  (opd_loss_type=topk_reverse_kl):
            S = the STUDENT's per-position top-k (queried on the teacher via
                per-position token_ids_logprob, so teacher_logprobs_on_student holds
                the teacher's logprob AT the student ids);
            L = sum_{v in S} q_s(v) * (log pi_s(v) - log pi_t(v))
        q_s is the training-time student, so it is differentiable and this form keeps a
        d/dtheta[sum_S pi_s] term that the full-vocab reverse KL cancels through
        d/dtheta[sum_v pi_s] = 0: at pi_s == pi_t the loss is 0 but the gradient is not,
        and descent shaves mass off the support. Its size tracks the off-support mass
        1 - sum_S pi_s, so it stays small while the top-k is tight and grows on
        high-entropy positions. Using log q_s instead of log pi_s inside the difference
        (or detaching the difference) removes it, at the price of no longer matching the
        reference implementations this form mirrors.

    Both directions are 0 at pi_s == pi_t on S but are not sign-constrained (a student
    that out-covers the teacher on every support id makes the sum negative), so read the
    logged values as signed distances rather than as KLs.

    Forward reads ``batch["teacher_topk_ids"]`` / ``["teacher_topk_logprobs"]``;
    reverse reads ``batch["student_support_ids"]`` / ``["teacher_logprobs_on_student"]``
    (separate keys so each name matches its content). Both are per-sample ``[R_i, K]``
    tensors. Contrast with the default token-level reverse-KL, a single-sample estimate
    on the sampled token only.
    """
    if direction == "forward":
        ids_key, lp_key = "teacher_topk_ids", "teacher_topk_logprobs"
    else:
        ids_key, lp_key = "student_support_ids", "teacher_logprobs_on_student"
    topk_ids = batch.get(ids_key)
    topk_teacher_lp = batch.get(lp_key)
    if not topk_ids or not topk_teacher_lp:
        raise ValueError(
            f"opd_loss_type='topk_{direction}_kl' requires '{ids_key}' and '{lp_key}' "
            "on the batch. Set --opd-id-topk > 0 so the teacher reward request fetches "
            "per-position top-k logprobs."
        )

    tp_group = mpu.get_tensor_model_parallel_group()
    per_sample_kl: list[torch.Tensor] = []
    per_sample_student_mass: list[torch.Tensor] = []
    per_sample_teacher_mass: list[torch.Tensor] = []
    per_sample_overlap_at: dict[int, list[torch.Tensor]] = {m: [] for m in _OVERLAP_AT_M}
    # Offline-only drift diagnostics. The rollout-student values and teacher values
    # are both fixed Phase-1/2 artifacts; only ``student_lp`` comes from this forward.
    rollout_student_lp = batch.get("rollout_student_logprobs_on_support")
    rollout_sample_lp = batch.get("rollout_log_probs")
    collect_offline_metrics = direction == "reverse" and bool(rollout_student_lp)
    collect_offline_overlap = collect_offline_metrics and bool(
        getattr(args, "offline_opd_topk_overlap", False)
    )
    if collect_offline_metrics and len(rollout_student_lp) != len(topk_ids):
        raise ValueError(
            "offline OPD rollout-student tensors must have one entry per sample: "
            f"old={len(rollout_student_lp)}, support={len(topk_ids)}"
        )
    if collect_offline_metrics and rollout_sample_lp and len(rollout_sample_lp) != len(
        topk_ids
    ):
        raise ValueError(
            "offline OPD rollout token logprobs must have one entry per sample: "
            f"rollout={len(rollout_sample_lp)}, support={len(topk_ids)}"
        )
    per_sample_current_vs_rollout_kl: list[torch.Tensor] = []
    per_sample_support_retention: list[torch.Tensor] = []
    per_sample_current_vs_teacher_kl: list[torch.Tensor] = []
    per_sample_teacher_improvement: list[torch.Tensor] = []
    per_sample_importance_weight: list[torch.Tensor] = []
    per_sample_importance_log_ratio: list[torch.Tensor] = []
    per_sample_current_vs_rollout_overlap: dict[int, list[torch.Tensor]] = {
        m: [] for m in _OVERLAP_AT_M
    }
    per_sample_current_vs_rollout_overlap_full: list[torch.Tensor] = []
    offline_train_mask = None
    offline_mask_offset = 0
    if collect_offline_overlap:
        offline_train_mask = get_concat_loss_mask(
            batch["total_lengths"],
            batch["response_lengths"],
            batch["loss_masks"],
        ).to(device=logits.device) > 0
    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        apply_temperature=False,
    )

    def local_response_positions(sample_idx: int, local_length: int) -> torch.Tensor:
        """Full-response positions aligned to this CP rank's response slice."""
        response_length = batch["response_lengths"][sample_idx]
        total_length = batch["total_lengths"][sample_idx]
        prompt_length = total_length - response_length
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            positions = torch.arange(response_length, device=logits.device)
        elif args.allgather_cp:
            seq_start = sum(batch["total_lengths"][:sample_idx])
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1
            chunk_start = mpu.get_context_parallel_rank() * logits.size(1)
            chunk_end = chunk_start + logits.size(1)
            start = max(logit_global_start, chunk_start)
            end = min(logit_global_end, chunk_end)
            positions = torch.arange(start, end, device=logits.device) - logit_global_start
        else:
            _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                total_length, response_length
            )
            chunks = [
                torch.arange(start - prompt_length, end - prompt_length, device=logits.device)
                for start, end in token_offsets
                if end > start
            ]
            positions = (
                torch.cat(chunks, dim=0)
                if chunks
                else torch.empty(0, dtype=torch.long, device=logits.device)
            )
        if positions.numel() != local_length:
            raise ValueError(
                "offline OPD response-position mapping is misaligned: "
                f"positions={positions.numel()}, local_response={local_length}"
            )
        return positions

    for sample_idx, ((logits_chunk, tokens_chunk), ids, teacher_lp) in enumerate(
        zip(responses, topk_ids, topk_teacher_lp, strict=False)
    ):
        ids = ids.to(device=logits_chunk.device, dtype=torch.long)
        teacher_lp = teacher_lp.to(device=logits_chunk.device, dtype=torch.float32)
        # Guard the teacher log-probs BEFORE they enter the log-ratio, not just the weight
        # below. A -inf entry (the teacher server's non-base64 top-k path serializes an
        # unscored id as "-Infinity") makes log_ratio +-inf, and `weight * inf` is inf
        # where weight > 0 and NaN where weight == 0 -- either one propagates through the
        # sum into every parameter's gradient for the whole micro-batch. Clamping to
        # _LOGPROB_FLOOR is diagnostics-neutral: exp(-100) underflows to 0 in float32, so
        # the renormalized weights and the *_mass series are bit-identical to before.
        teacher_lp = torch.nan_to_num(
            teacher_lp, nan=_LOGPROB_FLOOR, posinf=0.0, neginf=_LOGPROB_FLOOR
        )
        # Student log-probs at the support ids, TP-aware and differentiable.
        # When the offline ESS metric is available, append the actually sampled token
        # to the gather. This reuses the same full-vocab normalization and avoids a
        # second logits forward/logsumexp merely to recover its current log-prob.
        collect_sample_importance = collect_offline_metrics and bool(rollout_sample_lp)
        query_ids = (
            torch.cat([ids, tokens_chunk.to(ids.device).reshape(-1, 1)], dim=-1)
            if collect_sample_importance
            else ids
        )
        gathered_student_lp = vocab_parallel_logprobs_on_ids(
            logits_chunk,
            query_ids,
            tp_group,
            chunk_size=getattr(args, "log_probs_chunk_size", -1),
        )
        student_lp = gathered_student_lp[:, : ids.shape[1]]  # [R, K]
        current_sample_lp = (
            gathered_student_lp[:, -1].detach() if collect_sample_importance else None
        )
        # Only the WEIGHT is renormalized over S; the log-ratio keeps the raw log-probs.
        # Subtracting the logsumexp is the renormalization pi(v) / sum_{u in S} pi(u), done
        # in log space so a support of very small probs cannot underflow.
        if direction == "forward":
            log_weight, log_ratio = teacher_lp, teacher_lp - student_lp
        else:
            log_weight, log_ratio = student_lp, student_lp - teacher_lp
        norm_log_weight = log_weight - torch.logsumexp(log_weight, dim=-1, keepdim=True)
        weight = torch.exp(norm_log_weight)  # q(v), sums to 1 over K
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        kl = (weight * log_ratio).sum(dim=-1)  # [R]
        per_sample_kl.append(kl)
        # Diagnostics (detached): probability mass each engine puts on the support S.
        # exp is safe -- both are log-probs (<= 0). forward -> S = teacher top-k, so
        # student_mass is the student's mass on the teacher's top-k (GKD's student_mass);
        # reverse -> S = student top-k and the two roles swap.
        per_sample_student_mass.append(student_lp.detach().exp().sum(dim=-1))  # [R]
        per_sample_teacher_mass.append(teacher_lp.exp().sum(dim=-1))  # [R]

        if collect_offline_metrics:
            old_lp = rollout_student_lp[sample_idx].to(
                device=logits_chunk.device, dtype=torch.float32
            )
            if old_lp.shape != student_lp.shape:
                raise ValueError(
                    "offline OPD rollout-student logprobs must align with support ids: "
                    f"old={tuple(old_lp.shape)}, current={tuple(student_lp.shape)}"
                )
            old_lp = torch.nan_to_num(
                old_lp, nan=_LOGPROB_FLOOR, posinf=0.0, neginf=_LOGPROB_FLOOR
            )
            current_lp_detached = student_lp.detach()
            old_log_z = torch.logsumexp(old_lp, dim=-1, keepdim=True)
            current_log_z = torch.logsumexp(
                current_lp_detached, dim=-1, keepdim=True
            )
            teacher_log_z = torch.logsumexp(teacher_lp, dim=-1, keepdim=True)
            old_log_q = old_lp - old_log_z
            current_log_q = current_lp_detached - current_log_z
            teacher_log_q = teacher_lp - teacher_log_z
            old_q = old_log_q.exp()
            current_q = current_log_q.exp()

            current_vs_rollout = (
                old_q * (old_log_q - current_log_q)
            ).sum(dim=-1)
            current_vs_teacher = (
                current_q * (current_log_q - teacher_log_q)
            ).sum(dim=-1)
            rollout_vs_teacher = (
                old_q * (old_log_q - teacher_log_q)
            ).sum(dim=-1)
            per_sample_current_vs_rollout_kl.append(current_vs_rollout)
            per_sample_current_vs_teacher_kl.append(current_vs_teacher)
            per_sample_teacher_improvement.append(
                rollout_vs_teacher - current_vs_teacher
            )
            per_sample_support_retention.append(
                torch.exp((current_log_z - old_log_z).squeeze(-1))
            )

            if collect_offline_overlap:
                # True full-vocabulary current-vs-rollout top-k overlap. Unlike the
                # existing opd_topk_overlap_at* diagnostic below, this computes the
                # current model's own global top-k rather than re-ranking only inside
                # the rollout support (which would make overlap@K identically one).
                sample_train_mask = offline_train_mask[
                    offline_mask_offset : offline_mask_offset + ids.shape[0]
                ]
                offline_mask_offset += ids.shape[0]
                if sample_train_mask.numel() != ids.shape[0]:
                    raise ValueError(
                        "offline OPD train mask is not response-aligned: "
                        f"mask={sample_train_mask.numel()}, response={ids.shape[0]}"
                    )
                trainable_ids = ids[sample_train_mask]
                current_topk_ids = vocab_parallel_topk_ids(
                    logits_chunk[sample_train_mask],
                    ids.shape[1],
                    tp_group,
                    chunk_size=getattr(args, "log_probs_chunk_size", -1),
                    global_vocab_size=getattr(args, "vocab_size", None),
                )

                def current_rollout_overlap(width: int) -> torch.Tensor:
                    rollout_ids = torch.sort(trainable_ids[:, :width], dim=-1).values
                    current_ids = current_topk_ids[:, :width].contiguous()
                    positions = torch.searchsorted(rollout_ids, current_ids)
                    valid = positions < width
                    matched = valid & torch.gather(
                        rollout_ids, 1, positions.clamp_max(width - 1)
                    ).eq(current_ids)
                    trainable_overlap = matched.sum(dim=-1).float() / width
                    response_overlap = trainable_overlap.new_zeros(ids.shape[0])
                    response_overlap[sample_train_mask] = trainable_overlap
                    return response_overlap

                for width in _OVERLAP_AT_M:
                    if width <= ids.shape[1]:
                        per_sample_current_vs_rollout_overlap[width].append(
                            current_rollout_overlap(width)
                        )
                per_sample_current_vs_rollout_overlap_full.append(
                    current_rollout_overlap(ids.shape[1])
                )

            if current_sample_lp is not None:
                old_sample_lp = rollout_sample_lp[sample_idx].to(
                    device=logits_chunk.device, dtype=torch.float32
                )

                if old_sample_lp.shape != current_sample_lp.shape:
                    raise ValueError(
                        "offline OPD rollout token logprobs are not response-aligned: "
                        f"old={tuple(old_sample_lp.shape)}, "
                        f"current={tuple(current_sample_lp.shape)}"
                    )
                importance_log_ratio = current_sample_lp - old_sample_lp
                per_sample_importance_log_ratio.append(importance_log_ratio)
                sample_importance_weight = torch.exp(
                    importance_log_ratio.clamp(
                        -OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND
                    )
                )
                per_sample_importance_weight.append(sample_importance_weight)

                # Keep the ordinary metric channel compact, but emit exact payloads
                # for rare tail events while the explicit debug switch is enabled.
                # TP ranks see identical gathered values, so only TP0 logs them;
                # DP/CP rank and stable Phase-2 sample id make each event traceable
                # back to its sparse sample artifact.
                if (
                    getattr(args, "offline_opd_importance_debug", False)
                    and getattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)() == 0
                    and getattr(mpu, "is_pipeline_last_stage", lambda: True)()
                ):
                    local_mask = slice_log_prob_with_cp(
                        batch["loss_masks"][sample_idx],
                        batch["total_lengths"][sample_idx],
                        batch["response_lengths"][sample_idx],
                    ).to(device=sample_importance_weight.device).bool()
                    outlier_indices = torch.nonzero(
                        local_mask & (sample_importance_weight > 1_000.0),
                        as_tuple=False,
                    ).flatten()
                    if outlier_indices.numel() > 32:
                        _, top_positions = torch.topk(
                            sample_importance_weight[outlier_indices], 32
                        )
                        outlier_indices = outlier_indices[top_positions]
                    response_positions = local_response_positions(
                        sample_idx, sample_importance_weight.numel()
                    )
                    sample_ids = batch.get("sample_indices")
                    sample_id = (
                        sample_ids[sample_idx]
                        if sample_ids is not None
                        else -1
                    )
                    for local_idx in outlier_indices.tolist():
                        old_value = old_sample_lp[local_idx]
                        current_value = current_sample_lp[local_idx]
                        ratio_value = importance_log_ratio[local_idx]
                        logger.warning(
                            "offline OPD importance outlier: sample_id=%s "
                            "response_pos=%d token_id=%d rollout_log_prob=%.9g "
                            "current_log_prob=%.9g rollout_prob=%.9g "
                            "current_prob=%.9g raw_log_ratio=%.9g weight=%.9g "
                            "dp_rank=%d cp_rank=%d",
                            sample_id,
                            int(response_positions[local_idx].item()),
                            int(tokens_chunk[local_idx].item()),
                            float(old_value.item()),
                            float(current_value.item()),
                            float(old_value.exp().item()),
                            float(current_value.exp().item()),
                            float(ratio_value.item()),
                            float(sample_importance_weight[local_idx].item()),
                            mpu.get_data_parallel_rank(with_context_parallel=False),
                            mpu.get_context_parallel_rank(),
                        )
        # overlap@m: |TopM_own ∩ TopM_other| / m at each position, detached.
        # S arrives sorted by the OWN model's probability -- descending top-k out of the
        # rollout engine for reverse, out of the teacher for forward, the same ordering the
        # full-width opd_topk_overlap column is computed against -- so S's first m columns
        # ARE the own top-m, and topk(other_lp, m) (sorted, hence nested across m) gives the
        # other model's top-m as column indices. The intersection size is then just how many
        # of those indices land in [0, m).
        # Caveat: the other model is ranked INSIDE S, so this is exact only while its mass on
        # S is ~1 -- read it next to student_mass / teacher_mass (both ~0.9999 in practice).
        other_lp = teacher_lp if direction == "reverse" else student_lp.detach()
        widths = [m for m in _OVERLAP_AT_M if m <= ids.shape[1]]
        if widths:
            other_top = torch.topk(other_lp, max(widths), dim=-1).indices  # [R, max_m]
            for m in widths:
                per_sample_overlap_at[m].append((other_top[:, :m] < m).sum(dim=-1).float() / m)

    if collect_offline_overlap and offline_mask_offset != offline_train_mask.numel():
        raise ValueError(
            "offline OPD train mask has unconsumed response positions: "
            f"consumed={offline_mask_offset}, mask={offline_train_mask.numel()}"
        )

    kl_cat = torch.cat(per_sample_kl, dim=0)
    student_mass_cat = torch.cat(per_sample_student_mass, dim=0)
    teacher_mass_cat = torch.cat(per_sample_teacher_mass, dim=0)
    opd_loss = sum_of_sample_mean(kl_cat)
    loss = opd_loss

    # make sure the gradient could backprop correctly.
    if kl_cat.numel() == 0:
        loss = loss + 0 * logits.sum()

    metric_key = "opd_forward_kl" if direction == "forward" else "opd_reverse_kl"
    log = {
        "loss": loss.clone().detach(),
        metric_key: opd_loss.clone().detach(),
        # verl actor/distillation/* diagnostics. Means ride the same additive
        # sum_of_sample_mean channel as the loss (per-rollout masked mean); the *_max
        # variants ride the MAX channel (element-wise max across microbatches then
        "student_mass": sum_of_sample_mean(student_mass_cat).clone().detach(),
        "teacher_mass": sum_of_sample_mean(teacher_mass_cat).clone().detach(),
    }

    if collect_offline_metrics:
        current_vs_rollout_cat = torch.cat(
            per_sample_current_vs_rollout_kl, dim=0
        )
        support_retention_cat = torch.cat(per_sample_support_retention, dim=0)
        current_vs_teacher_cat = torch.cat(
            per_sample_current_vs_teacher_kl, dim=0
        )
        teacher_improvement_cat = torch.cat(
            per_sample_teacher_improvement, dim=0
        )
        log.update(
            {
                "offline_opd/current_vs_rollout_sparse_kl": sum_of_sample_mean(
                    current_vs_rollout_cat
                ).clone().detach(),
                "offline_opd/current_mass_on_rollout_topk": sum_of_sample_mean(
                    student_mass_cat
                ).clone().detach(),
                "offline_opd/support_mass_retention": sum_of_sample_mean(
                    support_retention_cat
                ).clone().detach(),
                "offline_opd/current_vs_cached_teacher_sparse_kl": sum_of_sample_mean(
                    current_vs_teacher_cat
                ).clone().detach(),
                "offline_opd/teacher_kl_improvement": sum_of_sample_mean(
                    teacher_improvement_cat
                ).clone().detach(),
                "offline_opd/teacher_mass_on_rollout_topk": sum_of_sample_mean(
                    teacher_mass_cat
                ).clone().detach(),
            }
        )
        if collect_offline_overlap:
            for width, values in per_sample_current_vs_rollout_overlap.items():
                if values:
                    log[f"offline_opd/current_vs_rollout_topk_overlap_at{width}"] = (
                        sum_of_sample_mean(torch.cat(values, dim=0)).clone().detach()
                    )
            log["offline_opd/current_vs_rollout_topk_overlap"] = sum_of_sample_mean(
                torch.cat(per_sample_current_vs_rollout_overlap_full, dim=0)
            ).clone().detach()

    # Batch-global masked max over in-mask (trainable) response tokens; the mask is
    # response-token aligned exactly like the one sum_of_sample_mean consumes. Empty
    # support -> -inf sentinel, which all-reduce(MAX) ignores (mirrors rollout_probs_diff).
    # ``kl_cat`` and the two mass tensors are response-token aligned *after* CP
    # slicing.  Use the same CP-aware mask layout as ``sum_of_sample_mean``;
    # concatenating the original full-response masks makes the lengths differ
    # under CP and would silently include masked tool/observation positions in
    # the diagnostic maxima (often surfacing as an impossible mass of K).
    mask_cat = get_concat_loss_mask(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
    ).to(device=kl_cat.device) > 0

    def _masked_max(x: torch.Tensor) -> torch.Tensor:
        vals = x[mask_cat] if x.numel() == mask_cat.numel() else x
        return vals.max() if vals.numel() > 0 else x.new_tensor(float("-inf"))

    if per_sample_importance_weight:
        importance_weight = torch.cat(per_sample_importance_weight, dim=0)
        if importance_weight.numel() != mask_cat.numel():
            raise ValueError(
                "offline OPD importance weights are not aligned with loss_mask: "
                f"weights={importance_weight.numel()}, mask={mask_cat.numel()}"
            )
        # Build one observation per sequence, rather than treating every token as
        # an equally weighted observation.  A CP rank sees only its two sequence
        # chunks, so first all-reduce each sequence's local token sum/count and
        # log-ratio sum across the CP group.  Every CP rank then reports the same
        # complete sequence moments; the later DP*CP reduction duplicates sum,
        # sqsum and count by the same CP factor, which cancels in mean/std/ESS.
        per_sequence_local_stats: list[torch.Tensor] = []
        for sample_idx, (sample_weight, sample_log_ratio) in enumerate(
            zip(
                per_sample_importance_weight,
                per_sample_importance_log_ratio,
                strict=True,
            )
        ):
            sample_mask = slice_log_prob_with_cp(
                batch["loss_masks"][sample_idx],
                batch["total_lengths"][sample_idx],
                batch["response_lengths"][sample_idx],
            ).to(device=sample_weight.device, dtype=sample_weight.dtype)
            if sample_mask.numel() != sample_weight.numel():
                raise ValueError(
                    "offline OPD per-sequence importance mask is not response-aligned: "
                    f"mask={sample_mask.numel()}, weights={sample_weight.numel()}"
                )
            per_sequence_local_stats.append(
                torch.stack(
                    [
                        (sample_weight * sample_mask).sum(),
                        sample_mask.sum(),
                        (sample_log_ratio * sample_mask).sum(),
                    ]
                )
            )
        sequence_stats = torch.stack(per_sequence_local_stats).detach()
        if mpu.get_context_parallel_world_size() > 1:
            dist.all_reduce(sequence_stats, group=mpu.get_context_parallel_group())

        valid_sequence = sequence_stats[:, 1] > 0
        token_mean_weight = sequence_stats[valid_sequence, 0] / sequence_stats[
            valid_sequence, 1
        ]
        # Product(current token probabilities) / product(rollout token
        # probabilities), evaluated stably in log space.  Clamp only the final
        # sequence log-ratio; multiplying probabilities directly would underflow
        # for normal long responses.
        sequence_weight = torch.exp(
            sequence_stats[valid_sequence, 2].clamp(
                -OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND
            )
        )
        sequence_count = valid_sequence.sum().to(dtype=importance_weight.dtype)
        log["offline_importance_weight_sum"] = token_mean_weight.sum()
        log["offline_importance_weight_sqsum"] = token_mean_weight.square().sum()
        log["offline_importance_weight_count"] = sequence_count
        log["offline_sequence_importance_weight_sum"] = sequence_weight.sum()
        log["offline_sequence_importance_weight_sqsum"] = sequence_weight.square().sum()
        log["offline_sequence_importance_weight_count"] = sequence_count.clone()

        if getattr(args, "offline_opd_importance_debug", False):
            importance_log_ratio = torch.cat(per_sample_importance_log_ratio, dim=0)
            if importance_log_ratio.numel() != mask_cat.numel():
                raise ValueError(
                    "offline OPD importance log-ratios are not aligned with loss_mask: "
                    f"ratios={importance_log_ratio.numel()}, mask={mask_cat.numel()}"
                )
            valid_weight = importance_weight[mask_cat]
            valid_log_ratio = importance_log_ratio[mask_cat]
            # Debug tail counts remain token-level; keep their denominator
            # separate now that the main importance count is sequence-level.
            log["offline_importance_token_count"] = mask_cat.sum().to(
                dtype=importance_weight.dtype
            ).detach()
            if valid_weight.numel() > 0:
                # MAX reduction provides exact global extrema across microbatches,
                # DP and CP. Minima use max(-x) and are restored after reduction.
                log[f"{MAX_METRIC_PREFIX}offline_importance_weight_max"] = (
                    valid_weight.max().detach()
                )
                log[f"{MAX_METRIC_PREFIX}offline_importance_weight_neg_min"] = (
                    -valid_weight.min().detach()
                )
                log[f"{MAX_METRIC_PREFIX}offline_importance_log_ratio_max"] = (
                    valid_log_ratio.max().detach()
                )
                log[f"{MAX_METRIC_PREFIX}offline_importance_log_ratio_neg_min"] = (
                    -valid_log_ratio.min().detach()
                )

                # Additive counts are divided by the already-global valid-token
                # count in reduce_train_step_metrics. These tail fractions make a
                # rare single-token spike distinguishable from broad policy drift.
                for label, threshold in (
                    ("2", 2.0),
                    ("10", 10.0),
                    ("100", 100.0),
                    ("1k", 1_000.0),
                    ("10k", 10_000.0),
                ):
                    log[f"offline_importance_weight_gt_{label}_count"] = (
                        (valid_weight > threshold).sum().detach()
                    )
                for label, threshold in (
                    ("0_5", 0.5),
                    ("0_1", 0.1),
                    ("0_01", 0.01),
                ):
                    log[f"offline_importance_weight_lt_{label}_count"] = (
                        (valid_weight < threshold).sum().detach()
                    )
                log["offline_importance_log_ratio_clipped_high_count"] = (
                    (valid_log_ratio > OFFPOLICY_SAFETY_BOUND).sum().detach()
                )
                log["offline_importance_log_ratio_clipped_low_count"] = (
                    (valid_log_ratio < -OFFPOLICY_SAFETY_BOUND).sum().detach()
                )

    log[f"{MAX_METRIC_PREFIX}{metric_key}_max"] = _masked_max(kl_cat.detach())
    log[f"{MAX_METRIC_PREFIX}student_mass_max"] = _masked_max(student_mass_cat)
    log[f"{MAX_METRIC_PREFIX}teacher_mass_max"] = _masked_max(teacher_mass_cat)

    # Head-agreement series, same per-rollout masked-mean space as the loss.
    for m, vals in per_sample_overlap_at.items():
        if vals:
            log[f"opd_topk_overlap_at{m}"] = sum_of_sample_mean(torch.cat(vals, dim=0)).clone().detach()

    # Teacher-side diagnostic, present only when the teacher server was launched with
    # SGLANG_OPD_TOPK_OVERLAP. Reduced with the same sum_of_sample_mean as the loss so it
    # sits in the same per-rollout mean space as opd_reverse_kl in wandb.
    #
    # batch["opd_topk_overlap"] is ALREADY the per-position RATIO in [0, 1], not a count:
    # extract_teacher_topk_overlap (slime/rollout/on_policy_distillation.py) divides the
    # teacher's |TopK_teacher ∩ ids| column by K before storing it on the sample. Dividing by
    # topk_width here as well made overlap_ratio a factor of K too small (at K=1024 it read
    # ~5.7e-4 instead of ~0.59), which looked like the student and teacher top-k sets barely
    # intersected while teacher_mass was ~0.9999 -- self-contradictory. opd_topk_overlap keeps
    # its ratio scale so the wandb series stays continuous.
    if batch.get("opd_topk_overlap"):
        overlap = torch.cat(batch["opd_topk_overlap"], dim=0)
        log["opd_topk_overlap"] = sum_of_sample_mean(overlap).clone().detach()
        log["overlap_ratio"] = log["opd_topk_overlap"]
        log[f"{MAX_METRIC_PREFIX}overlap_ratio_max"] = _masked_max(overlap)
    return loss, log


def _dynamic_local_response_positions(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sample_idx: int,
    local_length: int,
) -> torch.Tensor:
    """Full-response positions aligned with one sample's local CP logits."""
    response_length = batch["response_lengths"][sample_idx]
    total_length = batch["total_lengths"][sample_idx]
    prompt_length = total_length - response_length
    cp_size = mpu.get_context_parallel_world_size()
    if cp_size == 1:
        positions = torch.arange(response_length, device=logits.device)
    elif args.allgather_cp:
        seq_start = sum(batch["total_lengths"][:sample_idx])
        logit_global_start = seq_start + prompt_length - 1
        logit_global_end = seq_start + total_length - 1
        chunk_start = mpu.get_context_parallel_rank() * logits.size(1)
        chunk_end = chunk_start + logits.size(1)
        start = max(logit_global_start, chunk_start)
        end = min(logit_global_end, chunk_end)
        positions = torch.arange(start, end, device=logits.device) - logit_global_start
    else:
        _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
            total_length, response_length
        )
        chunks = [
            torch.arange(start - prompt_length, end - prompt_length, device=logits.device)
            for start, end in token_offsets
            if end > start
        ]
        positions = (
            torch.cat(chunks)
            if chunks
            else torch.empty(0, dtype=torch.long, device=logits.device)
        )
    if positions.numel() != local_length:
        raise ValueError(
            "dynamic offline OPD response-position mapping is misaligned: "
            f"positions={positions.numel()}, local_response={local_length}"
        )
    return positions


def _lookup_dynamic_teacher_cache(
    current_ids: torch.Tensor,
    row_indices: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logprobs: torch.Tensor,
    cutoff_logprobs: torch.Tensor,
    teacher_head_ids: torch.Tensor,
    teacher_cache_mass: torch.Tensor,
    *,
    rows_per_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Lookup current ids in CPU-resident sparse teacher rows in bounded chunks.

    Returns imputed teacher log-probs, hit mask, probability-ranked teacher head,
    cached Top-K mass and the exact matched teacher mass. No ``[R,K]`` response
    expansion or dense vocabulary table is created.
    """
    if rows_per_chunk <= 0:
        raise ValueError(f"opd_teacher_cache_h2d_rows must be positive, got {rows_per_chunk}")
    outputs: list[list[torch.Tensor]] = [[], [], [], [], []]
    for start in range(0, current_ids.size(0), rows_per_chunk):
        end = min(start + rows_per_chunk, current_ids.size(0))
        cpu_rows = row_indices[start:end].detach().to(
            device=candidate_ids.device, dtype=torch.long
        )

        def stage(source: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            selected = source.index_select(0, cpu_rows).contiguous()
            # Pin only the bounded transfer chunk. Pinning the complete cache on
            # every TP/CP process would multiply locked host memory by model-parallel
            # world size and defeat the sparse design.
            if current_ids.is_cuda and selected.device.type == "cpu" and not selected.is_pinned():
                selected = selected.pin_memory()
            return selected.to(
                device=current_ids.device, dtype=dtype, non_blocking=current_ids.is_cuda
            )

        ids = stage(candidate_ids, torch.long)
        teacher_lp = stage(candidate_logprobs, torch.float32)
        cutoff = stage(cutoff_logprobs, torch.float32)
        head = stage(teacher_head_ids, torch.long)
        cache_mass = stage(teacher_cache_mass, torch.float32)
        query = current_ids[start:end]
        columns = torch.searchsorted(ids, query)
        safe_columns = columns.clamp_max(ids.size(1) - 1)
        hit = columns.lt(ids.size(1)) & torch.gather(ids, 1, safe_columns).eq(query)
        matched_lp = torch.gather(teacher_lp, 1, safe_columns)
        imputed_lp = torch.where(hit, matched_lp, cutoff[:, None])
        outputs[0].append(imputed_lp)
        outputs[1].append(hit)
        outputs[2].append(head)
        outputs[3].append(cache_mass)
        outputs[4].append(torch.where(hit, matched_lp.exp(), 0.0).sum(dim=-1))
    if not outputs[0]:
        n, k = current_ids.shape
        empty_lp = torch.empty((n, k), dtype=torch.float32, device=current_ids.device)
        empty_hit = torch.empty((n, k), dtype=torch.bool, device=current_ids.device)
        empty_head = torch.empty((n, teacher_head_ids.shape[1]), dtype=torch.long, device=current_ids.device)
        empty_row = torch.empty(n, dtype=torch.float32, device=current_ids.device)
        return empty_lp, empty_hit, empty_head, empty_row, empty_row
    return tuple(torch.cat(parts, dim=0) for parts in outputs)  # type: ignore[return-value]


def _lookup_cached_teacher_logprobs_on_ids(
    query_ids: torch.Tensor,
    row_indices: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logprobs: torch.Tensor,
    *,
    rows_per_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up one sampled-token id per cached Teacher row.

    Unlike :func:`_lookup_dynamic_teacher_cache`, this path never substitutes the
    cache cutoff for a miss: a sampled-token probability must be exact to define
    its OPD advantage.  Callers can consequently anchor or mask misses explicitly.
    """
    if query_ids.ndim != 1:
        raise ValueError(f"sampled-token query ids must be 1D, got {tuple(query_ids.shape)}")
    if rows_per_chunk <= 0:
        raise ValueError(f"opd_teacher_cache_h2d_rows must be positive, got {rows_per_chunk}")
    output_lp: list[torch.Tensor] = []
    output_hit: list[torch.Tensor] = []
    for start in range(0, query_ids.numel(), rows_per_chunk):
        end = min(start + rows_per_chunk, query_ids.numel())
        cpu_rows = row_indices[start:end].detach().to(
            device=candidate_ids.device, dtype=torch.long
        )

        def stage(source: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            selected = source.index_select(0, cpu_rows).contiguous()
            if query_ids.is_cuda and selected.device.type == "cpu" and not selected.is_pinned():
                selected = selected.pin_memory()
            return selected.to(
                device=query_ids.device, dtype=dtype, non_blocking=query_ids.is_cuda
            )

        ids = stage(candidate_ids, torch.long)
        teacher_lp = stage(candidate_logprobs, torch.float32)
        query = query_ids[start:end, None]
        columns = torch.searchsorted(ids, query)
        safe_columns = columns.clamp_max(ids.size(1) - 1)
        hit = columns.lt(ids.size(1)) & torch.gather(ids, 1, safe_columns).eq(query)
        matched_lp = torch.gather(teacher_lp, 1, safe_columns).squeeze(1)
        output_lp.append(matched_lp)
        output_hit.append(hit.squeeze(1))
    if not output_lp:
        return (
            torch.empty(0, dtype=torch.float32, device=query_ids.device),
            torch.empty(0, dtype=torch.bool, device=query_ids.device),
        )
    return torch.cat(output_lp), torch.cat(output_hit)


def _select_cached_teacher_topk_for_student(
    row_indices: torch.Tensor,
    candidate_ids: torch.Tensor,
    candidate_logprobs: torch.Tensor,
    teacher_head_ids: torch.Tensor,
    *,
    support_k: int,
    student_valid_vocab_size: int,
    token_id_bridge_path: str | None,
    output_device: torch.device,
    rows_per_chunk: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select the Teacher's first K Student-representable cached tokens.

    Candidate rows are stored sorted by Teacher token id.  ``teacher_head_ids``
    retains the original probability-ranked first K ids.  We force every
    representable head id into the result and fill holes from the highest
    log-probability representable tail entries.  Thus a Teacher-only token is
    skipped and the ranking continues instead of shrinking the support.

    Selection stays on CPU/TP0.  Only the resulting ``[N,K]`` Student ids and
    Teacher log-probs are transferred to the accelerator.
    """
    if rows_per_chunk <= 0:
        raise ValueError(
            f"opd_teacher_cache_h2d_rows must be positive, got {rows_per_chunk}"
        )
    if candidate_ids.device.type != "cpu" or candidate_logprobs.device.type != "cpu":
        raise ValueError("cached Teacher candidates must remain CPU-resident")
    if candidate_ids.size(1) < support_k or teacher_head_ids.size(1) < support_k:
        raise ValueError(
            "Teacher cache/head is narrower than requested support: "
            f"cache={candidate_ids.size(1)}, head={teacher_head_ids.size(1)}, K={support_k}"
        )

    reverse = None
    if token_id_bridge_path:
        from slime.utils.token_id_bridge import load_token_id_bridge

        bridge = load_token_id_bridge(token_id_bridge_path)
        reverse = torch.from_numpy(bridge.teacher_to_student)

    selected_ids: list[torch.Tensor] = []
    selected_lp: list[torch.Tensor] = []
    skipped_head: list[torch.Tensor] = []
    for start in range(0, row_indices.numel(), rows_per_chunk):
        end = min(start + rows_per_chunk, row_indices.numel())
        cpu_rows = row_indices[start:end].detach().to(
            device=candidate_ids.device, dtype=torch.long
        )
        teacher_ids = candidate_ids.index_select(0, cpu_rows).long()
        teacher_lp = candidate_logprobs.index_select(0, cpu_rows).float()
        head = teacher_head_ids.index_select(0, cpu_rows)[:, :support_k].long()

        if reverse is None:
            mapped = teacher_ids.clone()
            mapped[teacher_ids.ge(student_valid_vocab_size)] = -1
            mapped_head = head.clone()
            mapped_head[head.ge(student_valid_vocab_size)] = -1
        else:
            if teacher_ids.numel() and (
                int(teacher_ids.min()) < 0 or int(teacher_ids.max()) >= reverse.numel()
            ):
                raise ValueError("Teacher candidate id is outside token bridge vocabulary")
            if head.numel() and (int(head.min()) < 0 or int(head.max()) >= reverse.numel()):
                raise ValueError("Teacher head id is outside token bridge vocabulary")
            mapped = reverse[teacher_ids]
            mapped_head = reverse[head]

        valid = mapped.ge(0) & mapped.lt(student_valid_vocab_size)
        valid_count = valid.sum(dim=-1)
        if bool(valid_count.lt(support_k).any()):
            bad = torch.nonzero(valid_count.lt(support_k), as_tuple=False).flatten()
            first = int(bad[0])
            raise ValueError(
                "Teacher cache cannot provide enough Student-representable tokens: "
                f"row={int(cpu_rows[first])}, mapped={int(valid_count[first])}, K={support_k}"
            )

        # Candidate ids are id-sorted, so locate the exact probability-ranked
        # head entries with searchsorted.  A large bonus preserves all valid
        # original head entries when fp16 cache log-probs tie at the boundary.
        head_columns = torch.searchsorted(teacher_ids, head)
        safe_head_columns = head_columns.clamp_max(teacher_ids.size(1) - 1)
        if bool(
            (
                head_columns.ge(teacher_ids.size(1))
                | torch.gather(teacher_ids, 1, safe_head_columns).ne(head)
            ).any()
        ):
            raise ValueError("Teacher probability-ranked head is absent from candidate cache")
        valid_head = mapped_head.ge(0) & mapped_head.lt(student_valid_vocab_size)
        bonus = torch.zeros_like(teacher_lp)
        bonus.scatter_(1, safe_head_columns, valid_head.to(teacher_lp.dtype) * 1_000_000.0)
        selection_score = (teacher_lp + bonus).masked_fill(~valid, float("-inf"))
        columns = torch.topk(selection_score, support_k, dim=-1).indices
        ids = torch.gather(mapped, 1, columns)
        lp = torch.gather(teacher_lp, 1, columns)
        # The forward-KL support must remain Teacher probability-ranked so
        # top-M metrics are nested prefixes. Exact head membership is already
        # protected above; only fp16-equal entries may exchange order.
        order = torch.argsort(lp, dim=-1, descending=True, stable=True)
        ids = torch.gather(ids, 1, order)
        lp = torch.gather(lp, 1, order)

        selected_ids.append(ids.to(device=output_device, dtype=torch.long, non_blocking=True))
        selected_lp.append(lp.to(device=output_device, dtype=torch.float32, non_blocking=True))
        skipped_head.append(
            (~valid_head).sum(dim=-1).to(
                device=output_device, dtype=torch.float32, non_blocking=True
            )
        )

    if not selected_ids:
        return (
            torch.empty((0, support_k), dtype=torch.long, device=output_device),
            torch.empty((0, support_k), dtype=torch.float32, device=output_device),
            torch.empty(0, dtype=torch.float32, device=output_device),
        )
    return (
        torch.cat(selected_ids, dim=0),
        torch.cat(selected_lp, dim=0),
        torch.cat(skipped_head, dim=0),
    )


def _opd_cached_teacher_topk_forward_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward-KL on a Teacher-owned Top-K derived from a larger sparse cache.

    Cross-tokenizer Teacher-only ids are skipped and replaced by the next
    probability-ranked cached id that maps into the Student vocabulary.  The
    fixed Teacher cache is never expanded to ``[response_length, cache_K]`` on
    GPU; TP0 selects on CPU and broadcasts only ``[trainable_rows, support_K]``.
    """
    required = (
        "teacher_candidate_positions",
        "teacher_candidate_ids",
        "teacher_candidate_logprobs",
        "teacher_top128_ids",
        "teacher_top4096_mass",
    )
    missing = [key for key in required if not batch.get(key)]
    if missing:
        raise ValueError(
            f"cached Teacher forward-KL batch is missing: {', '.join(missing)}"
        )
    support_k = int(getattr(args, "opd_dynamic_support_topk", 128))
    cache_k = int(getattr(args, "opd_teacher_cache_topk", 4096))
    if not 0 < support_k <= cache_k:
        raise ValueError(
            f"require 0 < Teacher support K <= cache K, got {support_k}/{cache_k}"
        )
    token_id_bridge_path = getattr(args, "opd_token_id_bridge", None)
    valid_student_vocab_size = getattr(args, "opd_student_valid_vocab_size", None)
    if valid_student_vocab_size is None:
        valid_student_vocab_size = getattr(args, "vocab_size", None)
    if valid_student_vocab_size is None:
        raise ValueError("Student valid vocabulary size is required for cached forward-KL")
    valid_student_vocab_size = int(valid_student_vocab_size)
    if token_id_bridge_path:
        from slime.utils.token_id_bridge import load_token_id_bridge

        bridge = load_token_id_bridge(token_id_bridge_path)
        if valid_student_vocab_size != bridge.student_tokenizer_size:
            raise ValueError(
                "cached forward-KL valid Student vocabulary does not match bridge: "
                f"{valid_student_vocab_size} != {bridge.student_tokenizer_size}"
            )

    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        apply_temperature=False,
    )
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world_size = getattr(mpu, "get_tensor_model_parallel_world_size", lambda: 1)()
    tp_rank = getattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)()
    rows_per_chunk = int(getattr(args, "opd_teacher_cache_h2d_rows", 512))
    collect_overlap = bool(getattr(args, "offline_opd_topk_overlap", False))
    rollout_ids_all = batch.get("rollout_student_topk_ids_sparse")
    rollout_lp_all = batch.get("rollout_student_topk_logprobs_sparse")
    rollout_sample_lp_all = batch.get("rollout_log_probs")
    widths = tuple(width for width in _OVERLAP_AT_M if width <= support_k)

    per_sample_kl: list[torch.Tensor] = []
    per_sample: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            "student_mass",
            "teacher_mass",
            "teacher_vocab_skip_count",
            "teacher_cache_mass",
            *(f"student_mass_at{width}" for width in widths),
            *(f"teacher_mass_at{width}" for width in widths),
        )
    }
    overlap_at: dict[int, list[torch.Tensor]] = {width: [] for width in widths}
    teacher_overlap_full: list[torch.Tensor] = []
    rollout_overlap_at: dict[int, list[torch.Tensor]] = {width: [] for width in widths}
    rollout_overlap_full: list[torch.Tensor] = []
    current_vs_rollout_kl: list[torch.Tensor] = []
    current_mass_on_rollout: list[torch.Tensor] = []
    support_mass_retention: list[torch.Tensor] = []
    importance_weights: list[torch.Tensor] = []
    importance_log_ratios: list[torch.Tensor] = []

    def exact_overlap(left: torch.Tensor, right: torch.Tensor, width: int) -> torch.Tensor:
        left_sorted = torch.sort(left[:, :width], dim=-1).values
        query = right[:, :width].contiguous()
        pos = torch.searchsorted(left_sorted, query)
        valid = pos.lt(width)
        matched = valid & torch.gather(
            left_sorted, 1, pos.clamp_max(width - 1)
        ).eq(query)
        return matched.float().sum(dim=-1) / width

    for sample_idx, (logits_chunk, tokens_chunk) in enumerate(responses):
        local_positions = _dynamic_local_response_positions(
            args, batch, logits, sample_idx, logits_chunk.size(0)
        )
        cache_positions = batch["teacher_candidate_positions"][sample_idx].detach().cpu().long()
        candidate_ids_cpu = batch["teacher_candidate_ids"][sample_idx]
        candidate_lp_cpu = batch["teacher_candidate_logprobs"][sample_idx]
        teacher_head_cpu = batch["teacher_top128_ids"][sample_idx]
        cache_mass_cpu = batch["teacher_top4096_mass"][sample_idx]
        n_cache_rows = cache_positions.numel()
        if (
            tuple(candidate_ids_cpu.shape) != (n_cache_rows, cache_k)
            or tuple(candidate_lp_cpu.shape) != (n_cache_rows, cache_k)
            or teacher_head_cpu.shape[0] != n_cache_rows
            or teacher_head_cpu.shape[1] < support_k
            or tuple(cache_mass_cpu.shape) != (n_cache_rows,)
        ):
            raise ValueError(
                "cached Teacher forward-KL shape mismatch: "
                f"positions={n_cache_rows}, ids={tuple(candidate_ids_cpu.shape)}, "
                f"lp={tuple(candidate_lp_cpu.shape)}, head={tuple(teacher_head_cpu.shape)}, "
                f"mass={tuple(cache_mass_cpu.shape)}, configured={cache_k}/{support_k}"
            )

        columns = torch.searchsorted(cache_positions, local_positions.detach().cpu())
        safe = columns.clamp_max(max(cache_positions.numel() - 1, 0))
        selected = columns.lt(cache_positions.numel())
        if cache_positions.numel():
            selected &= cache_positions[safe].eq(local_positions.detach().cpu())
        local_row_indices = torch.nonzero(selected, as_tuple=False).flatten().to(
            logits_chunk.device
        )
        cache_row_indices = columns[selected]
        selected_logits = logits_chunk.index_select(0, local_row_indices)
        selected_tokens = tokens_chunk.index_select(0, local_row_indices)

        if tp_world_size == 1 or tp_rank == 0:
            support_ids, teacher_lp, skipped_head = _select_cached_teacher_topk_for_student(
                cache_row_indices,
                candidate_ids_cpu,
                candidate_lp_cpu,
                teacher_head_cpu,
                support_k=support_k,
                student_valid_vocab_size=valid_student_vocab_size,
                token_id_bridge_path=token_id_bridge_path,
                output_device=selected_logits.device,
                rows_per_chunk=rows_per_chunk,
            )
        else:
            n_rows = local_row_indices.numel()
            support_ids = torch.empty(
                (n_rows, support_k), dtype=torch.long, device=selected_logits.device
            )
            teacher_lp = torch.empty(
                (n_rows, support_k), dtype=torch.float32, device=selected_logits.device
            )
            skipped_head = torch.empty(
                n_rows, dtype=torch.float32, device=selected_logits.device
            )
        if tp_world_size > 1:
            tp_src_rank = mpu.get_tensor_model_parallel_src_rank()
            for tensor in (support_ids, teacher_lp, skipped_head):
                dist.broadcast(tensor, src=tp_src_rank, group=tp_group)
        teacher_lp = torch.nan_to_num(
            teacher_lp, nan=_LOGPROB_FLOOR, posinf=0.0, neginf=_LOGPROB_FLOOR
        )

        rollout_ids = None
        query_parts = [support_ids]
        if rollout_ids_all:
            rollout_ids = rollout_ids_all[sample_idx].index_select(
                0, cache_row_indices
            ).to(device=selected_logits.device, dtype=torch.long, non_blocking=True)
            query_parts.append(rollout_ids)
        if rollout_sample_lp_all:
            query_parts.append(selected_tokens.reshape(-1, 1))
        gathered_lp = vocab_parallel_logprobs_on_ids(
            selected_logits,
            torch.cat(query_parts, dim=-1),
            tp_group,
            chunk_size=getattr(args, "log_probs_chunk_size", -1),
        )
        student_lp = gathered_lp[:, :support_k]
        offset = support_k
        current_on_rollout_lp = None
        if rollout_ids is not None:
            current_on_rollout_lp = gathered_lp[:, offset : offset + rollout_ids.size(1)]
            offset += rollout_ids.size(1)
        current_sample_lp = (
            gathered_lp[:, offset].detach() if rollout_sample_lp_all else None
        )

        teacher_log_q = teacher_lp - torch.logsumexp(
            teacher_lp, dim=-1, keepdim=True
        )
        teacher_q = teacher_log_q.exp()
        kl_rows = (teacher_q * (teacher_lp - student_lp)).sum(dim=-1)
        teacher_mass = teacher_lp.exp().sum(dim=-1)
        student_mass = student_lp.detach().exp().sum(dim=-1)

        def response_aligned(rows: torch.Tensor) -> torch.Tensor:
            out = rows.new_zeros(logits_chunk.size(0))
            out[local_row_indices] = rows
            return out

        per_sample_kl.append(response_aligned(kl_rows))
        values = {
            "student_mass": student_mass,
            "teacher_mass": teacher_mass,
            "teacher_vocab_skip_count": skipped_head,
            "teacher_cache_mass": cache_mass_cpu.index_select(
                0, cache_row_indices
            ).to(device=selected_logits.device, dtype=torch.float32, non_blocking=True),
        }
        student_prefix = student_lp.detach().exp().cumsum(dim=-1)
        teacher_prefix = teacher_lp.exp().cumsum(dim=-1)
        for width in widths:
            values[f"student_mass_at{width}"] = student_prefix[:, width - 1]
            values[f"teacher_mass_at{width}"] = teacher_prefix[:, width - 1]
        for name, rows in values.items():
            per_sample[name].append(response_aligned(rows))

        current_topk_ids = None
        if collect_overlap:
            current_topk_ids = vocab_parallel_topk_ids(
                selected_logits,
                support_k,
                tp_group,
                chunk_size=getattr(args, "log_probs_chunk_size", -1),
                global_vocab_size=valid_student_vocab_size,
            )
            for width in widths:
                overlap_at[width].append(
                    response_aligned(exact_overlap(support_ids, current_topk_ids, width))
                )
            teacher_overlap_full.append(
                response_aligned(
                    exact_overlap(support_ids, current_topk_ids, support_k)
                )
            )

        if rollout_ids is not None and current_on_rollout_lp is not None:
            if collect_overlap and current_topk_ids is not None:
                for width in widths:
                    if width <= rollout_ids.size(1):
                        rollout_overlap_at[width].append(
                            response_aligned(
                                exact_overlap(rollout_ids, current_topk_ids, width)
                            )
                        )
                full_width = min(support_k, rollout_ids.size(1))
                rollout_overlap_full.append(
                    response_aligned(
                        exact_overlap(rollout_ids, current_topk_ids, full_width)
                    )
                )
            if rollout_lp_all:
                old_lp = rollout_lp_all[sample_idx].index_select(
                    0, cache_row_indices
                ).to(device=selected_logits.device, dtype=torch.float32, non_blocking=True)
                old_log_q = old_lp - torch.logsumexp(old_lp, dim=-1, keepdim=True)
                current_log_q = current_on_rollout_lp.detach() - torch.logsumexp(
                    current_on_rollout_lp.detach(), dim=-1, keepdim=True
                )
                current_vs_rollout_kl.append(
                    response_aligned(
                        (old_log_q.exp() * (old_log_q - current_log_q)).sum(dim=-1)
                    )
                )
                old_mass = old_lp.exp().sum(dim=-1)
                new_mass = current_on_rollout_lp.detach().exp().sum(dim=-1)
                current_mass_on_rollout.append(response_aligned(new_mass))
                support_mass_retention.append(
                    response_aligned(new_mass / old_mass.clamp_min(1e-30))
                )

        if current_sample_lp is not None:
            old_sample_lp = rollout_sample_lp_all[sample_idx].to(
                device=selected_logits.device, dtype=torch.float32, non_blocking=True
            ).index_select(0, local_row_indices)
            log_ratio = current_sample_lp - old_sample_lp
            importance_log_ratios.append(log_ratio)
            importance_weights.append(
                torch.exp(
                    log_ratio.clamp(
                        -OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND
                    )
                )
            )

    kl_cat = torch.cat(per_sample_kl)
    opd_loss = sum_of_sample_mean(kl_cat)
    loss = opd_loss if kl_cat.numel() else opd_loss + 0 * logits.sum()
    reduced = {
        name: sum_of_sample_mean(torch.cat(rows)).detach().clone()
        for name, rows in per_sample.items()
    }
    log: dict[str, torch.Tensor] = {
        "loss": loss.detach().clone(),
        "opd_forward_kl": opd_loss.detach().clone(),
        "student_mass": reduced["student_mass"],
        "teacher_mass": reduced["teacher_mass"],
        "offline_opd_forward/teacher_vocab_skip_count": reduced[
            "teacher_vocab_skip_count"
        ],
        "offline_opd_forward/teacher_vocab_skip_ratio": reduced[
            "teacher_vocab_skip_count"
        ]
        / support_k,
        f"offline_opd_forward/teacher_top{cache_k}_mass": reduced[
            "teacher_cache_mass"
        ],
    }
    for width in widths:
        log[f"student_mass_at{width}"] = reduced[f"student_mass_at{width}"]
        log[f"teacher_mass_at{width}"] = reduced[f"teacher_mass_at{width}"]
        if overlap_at[width]:
            log[f"opd_topk_overlap_at{width}"] = sum_of_sample_mean(
                torch.cat(overlap_at[width])
            ).detach().clone()
    if teacher_overlap_full:
        full_overlap = sum_of_sample_mean(
            torch.cat(teacher_overlap_full)
        ).detach().clone()
        log["opd_topk_overlap"] = full_overlap
        log["overlap_ratio"] = full_overlap
    if rollout_overlap_full:
        log["offline_opd/current_vs_rollout_topk_overlap"] = sum_of_sample_mean(
            torch.cat(rollout_overlap_full)
        ).detach().clone()
    for width, rows in rollout_overlap_at.items():
        if rows:
            log[f"offline_opd/current_vs_rollout_topk_overlap_at{width}"] = (
                sum_of_sample_mean(torch.cat(rows)).detach().clone()
            )
    if current_vs_rollout_kl:
        log["offline_opd/current_vs_rollout_sparse_kl"] = sum_of_sample_mean(
            torch.cat(current_vs_rollout_kl)
        ).detach().clone()
        log["offline_opd/current_mass_on_rollout_topk"] = sum_of_sample_mean(
            torch.cat(current_mass_on_rollout)
        ).detach().clone()
        log["offline_opd/support_mass_retention"] = sum_of_sample_mean(
            torch.cat(support_mass_retention)
        ).detach().clone()

    if importance_weights:
        sequence_stats = torch.stack(
            [
                torch.stack((weights.sum(), weights.new_tensor(weights.numel()), ratios.sum()))
                for weights, ratios in zip(
                    importance_weights, importance_log_ratios, strict=True
                )
            ]
        ).detach()
        if mpu.get_context_parallel_world_size() > 1:
            dist.all_reduce(sequence_stats, group=mpu.get_context_parallel_group())
        valid = sequence_stats[:, 1] > 0
        token_mean = sequence_stats[valid, 0] / sequence_stats[valid, 1]
        sequence_weight = torch.exp(
            sequence_stats[valid, 2].clamp(
                -OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND
            )
        )
        count = valid.sum().to(dtype=logits.dtype)
        log["offline_importance_weight_sum"] = token_mean.sum()
        log["offline_importance_weight_sqsum"] = token_mean.square().sum()
        log["offline_importance_weight_count"] = count
        log["offline_sequence_importance_weight_sum"] = sequence_weight.sum()
        log["offline_sequence_importance_weight_sqsum"] = sequence_weight.square().sum()
        log["offline_sequence_importance_weight_count"] = count.clone()
    return loss, log


def _resolve_opd_bidirectional_kl_coefficients(
    args: Namespace,
) -> tuple[float, float]:
    """Return fixed or complementary linearly annealed Teacher-KL weights."""
    forward_coef = float(getattr(args, "opd_forward_kl_coef", 0.5))
    reverse_coef = float(getattr(args, "opd_reverse_kl_coef", 0.5))
    anneal_steps = int(getattr(args, "opd_forward_kl_anneal_steps", 0))
    if forward_coef < 0 or reverse_coef < 0:
        raise ValueError(
            "bidirectional OPD KL coefficients must be non-negative, got "
            f"forward={forward_coef}, reverse={reverse_coef}"
        )
    if forward_coef + reverse_coef <= 0:
        raise ValueError("at least one bidirectional OPD KL coefficient must be positive")
    if anneal_steps < 0:
        raise ValueError(
            f"opd_forward_kl_anneal_steps must be non-negative, got {anneal_steps}"
        )
    if anneal_steps:
        step = max(int(getattr(args, "opd_kl_schedule_step", 0)), 0)
        progress = min(step / anneal_steps, 1.0)
        transferred = forward_coef * progress
        forward_coef -= transferred
        reverse_coef += transferred
    return forward_coef, reverse_coef


def _coarsened_reverse_kl_with_tail(
    current_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
) -> torch.Tensor:
    """KL(current || teacher) on sparse ids plus one omitted-vocabulary bucket.

    The sparse ids are individual categories and every token outside that support
    is represented by a final tail category.  Unlike the legacy truncated OPD
    estimator, this objective cannot be reduced by moving Student probability mass
    out of the selected support.

    Persisted Teacher log-probs may be float16 and cache-miss values are cutoff
    estimates.  Roundoff can therefore make a sparse head sum slightly exceed one;
    scale only those invalid rows so both explicit tail probabilities remain valid.
    """
    if current_logprobs.shape != teacher_logprobs.shape:
        raise ValueError(
            "tail-bucket reverse-KL requires aligned Student/Teacher tensors, got "
            f"{tuple(current_logprobs.shape)} and {tuple(teacher_logprobs.shape)}"
        )
    eps = torch.finfo(torch.float32).eps

    def head_and_tail(logprobs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        head = logprobs.float().exp()
        mass = head.sum(dim=-1)
        scale = ((1.0 - eps) / mass.clamp_min(eps)).clamp_max(1.0)
        head = head * scale.unsqueeze(-1)
        tail = (1.0 - head.sum(dim=-1)).clamp_min(eps)
        return head, tail

    current_head, current_tail = head_and_tail(current_logprobs)
    teacher_head, teacher_tail = head_and_tail(teacher_logprobs)
    head_kl = current_head * (
        current_head.clamp_min(eps).log()
        - teacher_head.clamp_min(eps).log()
    )
    tail_kl = current_tail * (current_tail.log() - teacher_tail.log())
    return head_kl.sum(dim=-1) + tail_kl


def _opd_dynamic_cached_teacher_topk_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
    *,
    include_forward: bool = False,
    include_student_anchor: bool = False,
    let_me_forward_else_reverse: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Reverse-KL, optionally mixed with Teacher-owned forward-KL.

    The current support is recomputed from the exact TP-sharded full vocabulary on
    every forward. Teacher candidates remain sparse on CPU; only trainable local CP
    rows are copied to the GPU, and cache misses use the teacher cache cutoff value.

    With ``include_forward=True``, the same sparse cache also supplies the first K
    Student-representable tokens from the Teacher's probability-ranked head.  Both
    directions share one sparse Student log-prob gather and one exact entropy
    calculation; the existing reverse-only mode follows its original path.
    """
    required = (
        "teacher_candidate_positions",
        "teacher_candidate_ids",
        "teacher_candidate_logprobs",
        "teacher_top128_ids",
        "teacher_cutoff_logprobs",
        "teacher_top4096_mass",
    )
    missing = [key for key in required if not batch.get(key)]
    if missing:
        raise ValueError(f"dynamic offline OPD batch is missing: {', '.join(missing)}")
    if let_me_forward_else_reverse and not (include_forward and include_student_anchor):
        raise ValueError(
            "Let-me conditional KL requires both Teacher forward-KL and Student anchors"
        )
    if include_student_anchor:
        anchor_required = (
            "student_anchor_positions",
            "student_anchor_topk_ids",
            "student_anchor_topk_logprobs",
            "student_anchor_reason",
        )
        anchor_missing = [key for key in anchor_required if not batch.get(key)]
        if anchor_missing:
            raise ValueError(
                "cross-tokenizer Student-anchor batch is missing: "
                + ", ".join(anchor_missing)
            )
    if getattr(args, "opd_teacher_cache_miss_policy", "cutoff") != "cutoff":
        raise ValueError("only opd_teacher_cache_miss_policy='cutoff' is implemented")
    support_k = int(getattr(args, "opd_dynamic_support_topk", 128))
    cache_k = int(getattr(args, "opd_teacher_cache_topk", 4096))
    if not 0 < support_k <= cache_k:
        raise ValueError(f"require 0 < dynamic support K <= cache K, got {support_k}/{cache_k}")
    if include_forward:
        forward_coef, reverse_coef = _resolve_opd_bidirectional_kl_coefficients(args)
    else:
        forward_coef = 0.0
        reverse_coef = 1.0
    anchor_coef = float(getattr(args, "opd_student_anchor_kl_coef", 1.0))
    if include_student_anchor and anchor_coef < 0:
        raise ValueError(
            f"opd_student_anchor_kl_coef must be non-negative, got {anchor_coef}"
        )
    # --opd-nm-rows-sampled-token: an anchor row that sits inside an n:m chunk of the
    # artifact's chunk map still has a teacher value -- the chunk's log-prob share -- just
    # no teacher distribution over the Student vocabulary. Such a row leaves the anchor
    # group and trains the chunk loss's sampled-token PG term instead. The three groups
    # are then each averaged over their own rows and added with weights 1, 1, lambda:
    #     mean_T(KL) + mean_nm(PG) + lambda * mean_A(anchor KL)
    # Which n:m kinds count follows --opd-chunk-tier minus its 1:1 kinds.
    nm_sampled_token = bool(getattr(args, "opd_nm_rows_sampled_token", False))
    nm_kinds: list[int] = []
    nm_pg_options: dict[str, Any] = {}
    if nm_sampled_token:
        if not include_student_anchor:
            raise ValueError(
                "--opd-nm-rows-sampled-token needs a *_with_student_anchor loss"
            )
        nm_missing = [
            key
            for key in (
                "teacher_chunk_logprob",
                "teacher_chunk_mask",
                "teacher_chunk_kind",
                "rollout_log_probs",
            )
            if not batch.get(key)
        ]
        if nm_missing:
            raise ValueError(
                "--opd-nm-rows-sampled-token needs the chunk artifact fields "
                "(build_sampled_token_set.py --emit-topk-cache); missing: "
                + ", ".join(nm_missing)
            )
        nm_tier = getattr(args, "opd_chunk_tier", "struct")
        if nm_tier not in CHUNK_KIND_TIERS:
            raise ValueError(
                f"opd_chunk_tier must be one of {sorted(CHUNK_KIND_TIERS)}, got {nm_tier!r}"
            )
        nm_kinds = sorted(CHUNK_KIND_TIERS[nm_tier] - CHUNK_KIND_TIERS["exact"])
        nm_pg_options = _sampled_token_pg_options(args)
        # The group weight is applied when the three means are added; every n:m row has
        # a teacher value by construction.
        nm_pg_options.update(teacher_coef=1.0, miss_policy="anchor")
    let_token_ids: set[int] = set()
    me_token_ids: set[int] = set()
    if let_me_forward_else_reverse:
        def parse_token_ids(value: object, argument: str) -> set[int]:
            if isinstance(value, str):
                pieces = value.split(",")
            elif isinstance(value, (list, tuple, set)):
                pieces = list(value)
            else:
                pieces = []
            try:
                ids = {int(piece) for piece in pieces if str(piece).strip()}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{argument} must be a comma-separated list of token ids") from exc
            if not ids or min(ids) < 0:
                raise ValueError(f"{argument} must contain non-negative token ids")
            return ids

        let_token_ids = parse_token_ids(
            getattr(args, "opd_let_token_ids", ""), "--opd-let-token-ids"
        )
        me_token_ids = parse_token_ids(
            getattr(args, "opd_me_token_ids", ""), "--opd-me-token-ids"
        )
        # This mode selects one direction per exact Teacher row rather than
        # mixing both directions on every row.  Coefficients therefore remain
        # one by definition; Student-anchor rows retain their independent coef.
        forward_coef = 1.0
        reverse_coef = 1.0

    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        apply_temperature=False,
    )
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world_size = getattr(
        mpu, "get_tensor_model_parallel_world_size", lambda: 1
    )()
    tp_rank = getattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)()
    token_id_bridge_path = getattr(args, "opd_token_id_bridge", None)
    valid_student_vocab_size = getattr(args, "opd_student_valid_vocab_size", None)
    if valid_student_vocab_size is None:
        valid_student_vocab_size = getattr(args, "vocab_size", None)
    if include_forward and valid_student_vocab_size is None:
        raise ValueError(
            "Student valid vocabulary size is required for bidirectional cached OPD"
        )
    if token_id_bridge_path:
        from slime.utils.token_id_bridge import load_token_id_bridge

        bridge = load_token_id_bridge(token_id_bridge_path)
        if int(valid_student_vocab_size) != bridge.student_tokenizer_size:
            raise ValueError(
                "dynamic offline OPD valid student vocabulary does not match token-id bridge: "
                f"{valid_student_vocab_size} != {bridge.student_tokenizer_size}"
            )
    rows_per_chunk = int(getattr(args, "opd_teacher_cache_h2d_rows", 512))
    rollout_ids_all = batch.get("rollout_student_topk_ids_sparse")
    rollout_lp_all = batch.get("rollout_student_topk_logprobs_sparse")
    rollout_sample_lp_all = batch.get("rollout_log_probs")
    mass_at_widths = tuple(width for width in _OVERLAP_AT_M if width <= support_k)

    metric_names = (
        "student_mass",
        "teacher_mass",
        "cache_hit_ratio",
        "cache_miss_count",
        "cache_miss_student_mass",
        "loss_weight_on_imputed_tokens",
        "teacher_mass_lower",
        "teacher_mass_upper",
        "teacher_cache_mass",
        "teacher_estimated_kl",
        "teacher_overlap",
        *(name for name in (
            "entropy",
            "forward_student_mass",
            "forward_teacher_mass",
            "forward_teacher_vocab_skip_count",
        ) if include_forward),
        *(f"student_mass_at{width}" for width in mass_at_widths),
        *(f"teacher_mass_at{width}" for width in mass_at_widths),
    )
    per_sample: dict[str, list[torch.Tensor]] = {name: [] for name in metric_names}
    overlap_at: dict[int, list[torch.Tensor]] = {m: [] for m in _OVERLAP_AT_M}
    rollout_overlap_at: dict[int, list[torch.Tensor]] = {m: [] for m in _OVERLAP_AT_M}
    rollout_overlap_full: list[torch.Tensor] = []
    current_vs_rollout_kl: list[torch.Tensor] = []
    current_mass_on_rollout: list[torch.Tensor] = []
    support_mass_retention: list[torch.Tensor] = []
    per_sample_reverse_kl: list[torch.Tensor] = []
    per_sample_forward_kl: list[torch.Tensor] = []
    rollout_topk_entropy: list[torch.Tensor] = []
    current_on_rollout_topk_entropy: list[torch.Tensor] = []
    importance_weights: list[torch.Tensor] = []
    importance_log_ratios: list[torch.Tensor] = []
    per_sample_anchor_kl: list[torch.Tensor] = []
    per_sample_anchor_old_mass: list[torch.Tensor] = []
    per_sample_anchor_current_mass: list[torch.Tensor] = []
    per_sample_anchor_fraction: list[torch.Tensor] = []
    per_sample_let_me_forward_fraction: list[torch.Tensor] = []
    per_sample_anchor_reason_fraction: dict[int, list[torch.Tensor]] = {
        reason: [] for reason in _STUDENT_ANCHOR_REASON_NAMES
    }
    per_sample_nm_pg: list[torch.Tensor] = []
    per_sample_nm_fraction: list[torch.Tensor] = []
    per_sample_nm_advantage: list[torch.Tensor] = []
    per_sample_nm_clipfrac: list[torch.Tensor] = []
    nm_local_counts: list[int] = []

    for sample_idx, (logits_chunk, tokens_chunk) in enumerate(responses):
        local_positions = _dynamic_local_response_positions(
            args, batch, logits, sample_idx, logits_chunk.size(0)
        )
        cache_positions = batch["teacher_candidate_positions"][sample_idx].detach().cpu().long()
        if cache_positions.ndim != 1:
            raise ValueError("teacher_candidate_positions must be one-dimensional")
        candidate_ids_cpu = batch["teacher_candidate_ids"][sample_idx]
        candidate_lp_cpu = batch["teacher_candidate_logprobs"][sample_idx]
        teacher_head_cpu = batch["teacher_top128_ids"][sample_idx]
        cutoff_cpu = batch["teacher_cutoff_logprobs"][sample_idx]
        cache_mass_cpu = batch["teacher_top4096_mass"][sample_idx]
        n_cache_rows = cache_positions.numel()
        if (
            tuple(candidate_ids_cpu.shape) != (n_cache_rows, cache_k)
            or tuple(candidate_lp_cpu.shape) != (n_cache_rows, cache_k)
            or tuple(teacher_head_cpu.shape) != (n_cache_rows, support_k)
            or tuple(cutoff_cpu.shape) != (n_cache_rows,)
            or tuple(cache_mass_cpu.shape) != (n_cache_rows,)
        ):
            raise ValueError(
                "dynamic offline OPD cache shape mismatch: "
                f"positions={n_cache_rows}, ids={tuple(candidate_ids_cpu.shape)}, "
                f"lp={tuple(candidate_lp_cpu.shape)}, head={tuple(teacher_head_cpu.shape)}, "
                f"cutoff={tuple(cutoff_cpu.shape)}, mass={tuple(cache_mass_cpu.shape)}, "
                f"configured cache/support={cache_k}/{support_k}"
            )
        columns = torch.searchsorted(cache_positions, local_positions.detach().cpu())
        safe = columns.clamp_max(max(cache_positions.numel() - 1, 0))
        selected = columns.lt(cache_positions.numel())
        if cache_positions.numel():
            selected &= cache_positions[safe].eq(local_positions.detach().cpu())
        local_row_indices = torch.nonzero(selected, as_tuple=False).flatten().to(logits_chunk.device)
        cache_row_indices = columns[selected]
        selected_logits = logits_chunk.index_select(0, local_row_indices)
        selected_tokens = tokens_chunk.index_select(0, local_row_indices)

        let_me_forward_rows = None
        if let_me_forward_else_reverse:
            response_length = int(batch["response_lengths"][sample_idx])
            response_tokens = batch["unconcat_tokens"][sample_idx][
                -response_length:
            ].detach().cpu().long()
            exact_teacher_row = torch.zeros(response_length, dtype=torch.bool)
            if cache_positions.numel():
                if cache_positions.min() < 0 or cache_positions.max() >= response_length:
                    raise ValueError("teacher_candidate_positions is outside the response")
                exact_teacher_row[cache_positions] = True
            phrase_rows = torch.zeros(response_length, dtype=torch.bool)
            if response_length >= 3:
                let_starts = torch.zeros(response_length - 2, dtype=torch.bool)
                me_seconds = torch.zeros_like(let_starts)
                for token_id in let_token_ids:
                    let_starts |= response_tokens[:-2].eq(token_id)
                for token_id in me_token_ids:
                    me_seconds |= response_tokens[1:-1].eq(token_id)
                starts = torch.nonzero(
                    let_starts
                    & me_seconds
                    & exact_teacher_row[:-2]
                    & exact_teacher_row[1:-1]
                    & exact_teacher_row[2:],
                    as_tuple=False,
                ).flatten()
                phrase_rows[starts] = True
                phrase_rows[starts + 1] = True
                phrase_rows[starts + 2] = True
            selected_response_positions = local_positions.detach().cpu()[selected]
            let_me_forward_rows = phrase_rows.index_select(
                0, selected_response_positions
            ).to(device=selected_logits.device)

        forward_ids = None
        forward_teacher_lp = None
        forward_skipped_head = None
        if include_forward:
            if tp_world_size == 1 or tp_rank == 0:
                forward_ids, forward_teacher_lp, forward_skipped_head = (
                    _select_cached_teacher_topk_for_student(
                        cache_row_indices,
                        candidate_ids_cpu,
                        candidate_lp_cpu,
                        teacher_head_cpu,
                        support_k=support_k,
                        student_valid_vocab_size=int(valid_student_vocab_size),
                        token_id_bridge_path=token_id_bridge_path,
                        output_device=selected_logits.device,
                        rows_per_chunk=rows_per_chunk,
                    )
                )
            else:
                n_rows = local_row_indices.numel()
                forward_ids = torch.empty(
                    (n_rows, support_k), dtype=torch.long, device=selected_logits.device
                )
                forward_teacher_lp = torch.empty(
                    (n_rows, support_k), dtype=torch.float32, device=selected_logits.device
                )
                forward_skipped_head = torch.empty(
                    n_rows, dtype=torch.float32, device=selected_logits.device
                )
            if tp_world_size > 1:
                tp_src_rank = mpu.get_tensor_model_parallel_src_rank()
                for tensor in (forward_ids, forward_teacher_lp, forward_skipped_head):
                    dist.broadcast(tensor, src=tp_src_rank, group=tp_group)
            forward_teacher_lp = torch.nan_to_num(
                forward_teacher_lp,
                nan=_LOGPROB_FLOOR,
                posinf=0.0,
                neginf=_LOGPROB_FLOOR,
            )

        current_ids = vocab_parallel_topk_ids(
            selected_logits,
            support_k,
            tp_group,
            chunk_size=getattr(args, "log_probs_chunk_size", -1),
            global_vocab_size=valid_student_vocab_size,
        )
        query_parts = [current_ids]
        if forward_ids is not None:
            query_parts.append(forward_ids)
        rollout_ids = None
        if rollout_ids_all:
            rollout_ids = rollout_ids_all[sample_idx].index_select(
                0, cache_row_indices
            ).to(device=selected_logits.device, dtype=torch.long, non_blocking=True)
            if rollout_ids.size(1) != support_k:
                raise ValueError(
                    "rollout student Top-K width does not match dynamic support: "
                    f"rollout={rollout_ids.size(1)}, dynamic={support_k}"
                )
            query_parts.append(rollout_ids)
        if rollout_sample_lp_all:
            query_parts.append(selected_tokens.reshape(-1, 1))
        query_ids = torch.cat(query_parts, dim=-1)
        if include_forward:
            gathered_lp, current_entropy = vocab_parallel_logprobs_and_entropy_on_ids(
                selected_logits,
                query_ids,
                tp_group,
                chunk_size=getattr(args, "log_probs_chunk_size", -1),
            )
        else:
            gathered_lp = vocab_parallel_logprobs_on_ids(
                selected_logits,
                query_ids,
                tp_group,
                chunk_size=getattr(args, "log_probs_chunk_size", -1),
            )
            current_entropy = None
        current_lp = gathered_lp[:, :support_k]
        offset = support_k
        forward_student_lp = None
        if forward_ids is not None:
            forward_student_lp = gathered_lp[:, offset : offset + support_k]
            offset += support_k
        current_on_rollout_lp = None
        if rollout_ids is not None:
            rollout_width = rollout_ids.size(1)
            current_on_rollout_lp = gathered_lp[:, offset : offset + rollout_width]
            offset += rollout_width
        current_sample_lp = gathered_lp[:, offset].detach() if rollout_sample_lp_all else None

        teacher_query_ids = current_ids
        if token_id_bridge_path:
            from slime.utils.token_id_bridge import map_token_ids_torch

            teacher_query_ids = map_token_ids_torch(
                current_ids,
                token_id_bridge_path,
                context="dynamic current-student Top-K",
                allow_unmapped=True,
            )
        if tp_world_size == 1 or tp_rank == 0:
            teacher_lp, hit, teacher_head, cache_mass, matched_mass = _lookup_dynamic_teacher_cache(
                teacher_query_ids,
                cache_row_indices,
                candidate_ids_cpu,
                candidate_lp_cpu,
                cutoff_cpu,
                teacher_head_cpu,
                cache_mass_cpu,
                rows_per_chunk=rows_per_chunk,
            )
        else:
            n_rows = current_ids.size(0)
            teacher_lp = torch.empty(
                (n_rows, support_k), dtype=torch.float32, device=current_ids.device
            )
            hit = torch.empty(
                (n_rows, support_k), dtype=torch.bool, device=current_ids.device
            )
            teacher_head = torch.empty(
                (n_rows, support_k), dtype=torch.long, device=current_ids.device
            )
            cache_mass = torch.empty(n_rows, dtype=torch.float32, device=current_ids.device)
            matched_mass = torch.empty(n_rows, dtype=torch.float32, device=current_ids.device)
        if tp_world_size > 1:
            # Only TP0 stages/searches [N, cache_K]. Broadcasts are bounded by
            # [N, support_K], preventing TP8 from repeating the large H2D copy.
            tp_src_rank = mpu.get_tensor_model_parallel_src_rank()
            for tensor in (teacher_lp, hit, teacher_head, cache_mass, matched_mass):
                dist.broadcast(tensor, src=tp_src_rank, group=tp_group)
        teacher_lp = torch.nan_to_num(
            teacher_lp, nan=_LOGPROB_FLOOR, posinf=0.0, neginf=_LOGPROB_FLOOR
        )
        log_q = current_lp - torch.logsumexp(current_lp, dim=-1, keepdim=True)
        q = log_q.exp()
        if getattr(args, "opd_reverse_kl_tail_bucket", False):
            reverse_kl_rows = _coarsened_reverse_kl_with_tail(
                current_lp, teacher_lp
            )
        else:
            # Legacy sparse estimator: renormalize only the Student weights while
            # retaining raw full-vocabulary log-probs in the ratio.  This is kept
            # unchanged for reproducibility; at very small K it can reward moving
            # mass outside the support and therefore become negative.
            reverse_kl_rows = (q * (current_lp - teacher_lp)).sum(dim=-1)
        forward_kl_rows = None
        if include_forward:
            assert forward_teacher_lp is not None
            assert forward_student_lp is not None
            forward_log_q = forward_teacher_lp - torch.logsumexp(
                forward_teacher_lp, dim=-1, keepdim=True
            )
            forward_kl_rows = (
                forward_log_q.exp() * (forward_teacher_lp - forward_student_lp)
            ).sum(dim=-1)
        if let_me_forward_rows is not None:
            assert forward_kl_rows is not None
            reverse_kl_rows = reverse_kl_rows * (~let_me_forward_rows)
            forward_kl_rows = forward_kl_rows * let_me_forward_rows
        current_mass = current_lp.detach().exp().sum(dim=-1)
        cutoff_prob = batch["teacher_cutoff_logprobs"][sample_idx].index_select(
            0, cache_row_indices
        ).to(device=selected_logits.device, dtype=torch.float32).exp()
        missing = (~hit).sum(dim=-1)
        teacher_upper = (matched_mass + missing * cutoff_prob).clamp_max(1.0)

        def response_aligned(rows: torch.Tensor) -> torch.Tensor:
            out = rows.new_zeros(logits_chunk.size(0))
            out[local_row_indices] = rows
            return out

        per_sample_reverse_kl.append(response_aligned(reverse_kl_rows))
        if forward_kl_rows is not None:
            per_sample_forward_kl.append(response_aligned(forward_kl_rows))
        if let_me_forward_rows is not None:
            per_sample_let_me_forward_fraction.append(
                response_aligned(let_me_forward_rows.float())
            )
        values = {
            "student_mass": current_mass,
            "teacher_mass": teacher_upper,
            "cache_hit_ratio": hit.float().mean(dim=-1),
            "cache_miss_count": missing.float(),
            "cache_miss_student_mass": (current_lp.detach().exp() * (~hit)).sum(dim=-1),
            "loss_weight_on_imputed_tokens": (q.detach() * (~hit)).sum(dim=-1),
            "teacher_mass_lower": matched_mass,
            "teacher_mass_upper": teacher_upper,
            "teacher_cache_mass": cache_mass,
            "teacher_estimated_kl": reverse_kl_rows.detach(),
        }
        if include_forward:
            assert current_entropy is not None
            assert forward_student_lp is not None
            assert forward_teacher_lp is not None
            assert forward_skipped_head is not None
            values.update(
                {
                    "entropy": current_entropy.detach(),
                    "forward_student_mass": forward_student_lp.detach().exp().sum(dim=-1),
                    "forward_teacher_mass": forward_teacher_lp.exp().sum(dim=-1),
                    "forward_teacher_vocab_skip_count": forward_skipped_head,
                }
            )
        # current_ids/current_lp are ordered by the current student's exact
        # full-vocabulary rank.  Report cumulative probability mass on the same
        # Top-M prefixes used by overlap@M.  Teacher cache misses retain the loss
        # path's cutoff imputation, so teacher_mass_atM is an upper estimate just
        # like the existing full-width teacher_mass metric.
        student_mass_prefix = current_lp.detach().exp().cumsum(dim=-1)
        teacher_mass_prefix = teacher_lp.exp().cumsum(dim=-1).clamp_max(1.0)
        for width in mass_at_widths:
            values[f"student_mass_at{width}"] = student_mass_prefix[:, width - 1]
            values[f"teacher_mass_at{width}"] = teacher_mass_prefix[:, width - 1]

        def exact_overlap(left: torch.Tensor, right: torch.Tensor, width: int) -> torch.Tensor:
            left_sorted = torch.sort(left[:, :width], dim=-1).values
            query = right[:, :width].contiguous()
            pos = torch.searchsorted(left_sorted, query)
            valid = pos.lt(width)
            matched = valid & torch.gather(left_sorted, 1, pos.clamp_max(width - 1)).eq(query)
            return matched.float().sum(dim=-1) / width

        values["teacher_overlap"] = exact_overlap(
            teacher_head, teacher_query_ids, support_k
        )
        for name, rows in values.items():
            per_sample[name].append(response_aligned(rows))
        for width in _OVERLAP_AT_M:
            if width <= support_k:
                overlap_at[width].append(
                    response_aligned(
                        exact_overlap(teacher_head, teacher_query_ids, width)
                    )
                )

        if rollout_ids is not None:
            rollout_width = rollout_ids.size(1)
            for width in _OVERLAP_AT_M:
                if width <= min(support_k, rollout_width):
                    rollout_overlap_at[width].append(
                        response_aligned(exact_overlap(rollout_ids, current_ids, width))
                    )
            full_width = min(support_k, rollout_width)
            rollout_overlap_full.append(
                response_aligned(exact_overlap(rollout_ids, current_ids, full_width))
            )
            if rollout_lp_all and current_on_rollout_lp is not None:
                old_lp = rollout_lp_all[sample_idx].index_select(
                    0, cache_row_indices
                ).to(device=selected_logits.device, dtype=torch.float32, non_blocking=True)
                if old_lp.shape != current_on_rollout_lp.shape:
                    raise ValueError(
                        "rollout student sparse log-probs do not align with ids: "
                        f"old={tuple(old_lp.shape)}, current={tuple(current_on_rollout_lp.shape)}"
                    )
                old_log_q = old_lp - torch.logsumexp(old_lp, dim=-1, keepdim=True)
                current_rollout_log_q = current_on_rollout_lp.detach() - torch.logsumexp(
                    current_on_rollout_lp.detach(), dim=-1, keepdim=True
                )
                current_vs_rollout_kl.append(
                    response_aligned((old_log_q.exp() * (old_log_q - current_rollout_log_q)).sum(dim=-1))
                )
                if include_forward:
                    rollout_topk_entropy.append(
                        response_aligned(
                            -(old_log_q.exp() * old_log_q).sum(dim=-1)
                        )
                    )
                    current_on_rollout_topk_entropy.append(
                        response_aligned(
                            -(
                                current_rollout_log_q.exp()
                                * current_rollout_log_q
                            ).sum(dim=-1)
                        )
                    )
                old_mass = old_lp.exp().sum(dim=-1)
                new_mass = current_on_rollout_lp.detach().exp().sum(dim=-1)
                current_mass_on_rollout.append(response_aligned(new_mass))
                support_mass_retention.append(response_aligned(new_mass / old_mass.clamp_min(1e-30)))

        if current_sample_lp is not None:
            old_sample_lp = rollout_sample_lp_all[sample_idx].to(
                device=selected_logits.device, dtype=torch.float32, non_blocking=True
            ).index_select(0, local_row_indices)
            log_ratio = current_sample_lp - old_sample_lp
            importance_log_ratios.append(log_ratio)
            importance_weights.append(torch.exp(log_ratio.clamp(-OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND)))

        if include_student_anchor:
            anchor_positions = batch["student_anchor_positions"][sample_idx].detach().cpu().long()
            anchor_ids_cpu = batch["student_anchor_topk_ids"][sample_idx]
            anchor_lp_cpu = batch["student_anchor_topk_logprobs"][sample_idx]
            anchor_reason_cpu = batch["student_anchor_reason"][sample_idx]
            n_anchor = anchor_positions.numel()
            if (
                anchor_positions.ndim != 1
                or tuple(anchor_ids_cpu.shape) != (n_anchor, support_k)
                or tuple(anchor_lp_cpu.shape) != (n_anchor, support_k)
                or tuple(anchor_reason_cpu.shape) != (n_anchor,)
            ):
                raise ValueError(
                    "cross-tokenizer Student-anchor shape mismatch: "
                    f"positions={tuple(anchor_positions.shape)}, "
                    f"ids={tuple(anchor_ids_cpu.shape)}, "
                    f"logprobs={tuple(anchor_lp_cpu.shape)}, "
                    f"reason={tuple(anchor_reason_cpu.shape)}, K={support_k}"
                )
            anchor_columns = torch.searchsorted(
                anchor_positions, local_positions.detach().cpu()
            )
            anchor_safe = anchor_columns.clamp_max(max(n_anchor - 1, 0))
            anchor_selected = anchor_columns.lt(n_anchor)
            if n_anchor:
                anchor_selected &= anchor_positions[anchor_safe].eq(
                    local_positions.detach().cpu()
                )
            anchor_local_rows = torch.nonzero(
                anchor_selected, as_tuple=False
            ).flatten().to(logits_chunk.device)
            anchor_cache_rows = anchor_columns[anchor_selected]
            if anchor_local_rows.numel():
                anchor_logits = logits_chunk.index_select(0, anchor_local_rows)
                anchor_ids = anchor_ids_cpu.index_select(
                    0, anchor_cache_rows
                ).to(
                    device=anchor_logits.device,
                    dtype=torch.long,
                    non_blocking=True,
                )
                anchor_old_lp = anchor_lp_cpu.index_select(
                    0, anchor_cache_rows
                ).to(
                    device=anchor_logits.device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                anchor_current_lp = vocab_parallel_logprobs_on_ids(
                    anchor_logits,
                    anchor_ids,
                    tp_group,
                    chunk_size=getattr(args, "log_probs_chunk_size", -1),
                )
                eps = torch.finfo(torch.float32).eps
                old_prob = anchor_old_lp.exp()
                old_mass_raw = old_prob.sum(dim=-1)
                # Float16 persisted log-probs can round a nearly-complete head
                # slightly above one.  Scale only those pathological rows so
                # the explicit tail remains a valid probability bucket.
                scale = ((1.0 - eps) / old_mass_raw.clamp_min(eps)).clamp_max(1.0)
                old_prob = old_prob * scale[:, None]
                old_mass = old_prob.sum(dim=-1)
                current_prob = anchor_current_lp.exp()
                current_mass = current_prob.sum(dim=-1).clamp_max(1.0 - eps)
                old_tail = (1.0 - old_mass).clamp_min(eps)
                current_tail = (1.0 - current_mass).clamp_min(eps)
                anchor_rows = (
                    old_prob
                    * (old_prob.clamp_min(eps).log() - anchor_current_lp)
                ).sum(dim=-1)
                anchor_rows = anchor_rows + old_tail * (
                    old_tail.log() - current_tail.log()
                )

                def anchor_response_aligned(rows: torch.Tensor) -> torch.Tensor:
                    out = rows.new_zeros(logits_chunk.size(0))
                    out[anchor_local_rows] = rows
                    return out

                # Anchor rows that the n:m sampled-token term takes over (all False
                # without --opd-nm-rows-sampled-token, so nothing below changes then).
                keep_anchor = torch.ones_like(anchor_rows, dtype=torch.bool)
                if nm_sampled_token:
                    chunk_kind = batch["teacher_chunk_kind"][sample_idx].to(
                        device=anchor_logits.device, non_blocking=True
                    )
                    chunk_mask = batch["teacher_chunk_mask"][sample_idx].to(
                        device=anchor_logits.device, non_blocking=True
                    )
                    if chunk_kind.numel() != logits_chunk.size(0) or chunk_mask.numel() != logits_chunk.size(0):
                        raise ValueError(
                            "n:m sampled-token chunk fields do not align with the local "
                            f"response: kind={chunk_kind.numel()}, mask={chunk_mask.numel()}, "
                            f"local={logits_chunk.size(0)}"
                        )
                    is_nm = chunk_mask.bool() & torch.isin(
                        chunk_kind.long(),
                        torch.tensor(nm_kinds, dtype=torch.long, device=chunk_kind.device),
                    )
                    keep_anchor = ~is_nm.index_select(0, anchor_local_rows)
                    anchor_rows = torch.where(keep_anchor, anchor_rows, torch.zeros_like(anchor_rows))
                    nm_local_rows = anchor_local_rows[~keep_anchor]
                    nm_local_counts.append(int(nm_local_rows.numel()))
                    nm_pg = logits_chunk.new_zeros(logits_chunk.size(0))
                    nm_advantage = nm_pg.clone()
                    nm_clipfrac = nm_pg.clone()
                    nm_fraction = nm_pg.clone()
                    if nm_local_rows.numel():
                        nm_logits = logits_chunk.index_select(0, nm_local_rows)
                        # The chunk loss scores its sampled token at the rollout temperature
                        # (get_responses(apply_temperature=True)); match it, as the PPO ratio
                        # is taken against the rollout log-prob.
                        if args.rollout_temperature != 1.0:
                            nm_logits = nm_logits.div(args.rollout_temperature)
                        nm_current_lp = vocab_parallel_logprobs_on_ids(
                            nm_logits,
                            tokens_chunk.index_select(0, nm_local_rows).reshape(-1, 1),
                            tp_group,
                            chunk_size=getattr(args, "log_probs_chunk_size", -1),
                        )[:, 0]
                        nm_rollout_lp = rollout_sample_lp_all[sample_idx].to(
                            device=nm_current_lp.device, dtype=torch.float32, non_blocking=True
                        ).index_select(0, nm_local_rows)
                        nm_target_lp = batch["teacher_chunk_logprob"][sample_idx].to(
                            device=nm_current_lp.device, dtype=torch.float32, non_blocking=True
                        ).index_select(0, nm_local_rows)
                        rows_pg, rows_advantage, rows_clipfrac, _ = _sampled_token_pg_rows(
                            nm_current_lp,
                            nm_rollout_lp,
                            nm_target_lp,
                            torch.ones_like(nm_current_lp, dtype=torch.bool),
                            **nm_pg_options,
                        )
                        nm_pg = nm_pg.index_put((nm_local_rows,), rows_pg)
                        nm_advantage[nm_local_rows] = rows_advantage.detach()
                        nm_clipfrac[nm_local_rows] = rows_clipfrac.detach().to(nm_clipfrac.dtype)
                        nm_fraction[nm_local_rows] = 1.0
                    per_sample_nm_pg.append(nm_pg)
                    per_sample_nm_advantage.append(nm_advantage)
                    per_sample_nm_clipfrac.append(nm_clipfrac)
                    per_sample_nm_fraction.append(nm_fraction)
                keep = keep_anchor.to(anchor_rows.dtype)

                per_sample_anchor_kl.append(anchor_response_aligned(anchor_rows))
                per_sample_anchor_old_mass.append(
                    anchor_response_aligned(old_mass.detach() * keep)
                )
                per_sample_anchor_current_mass.append(
                    anchor_response_aligned(current_mass.detach() * keep)
                )
                per_sample_anchor_fraction.append(
                    anchor_response_aligned(keep)
                )
                selected_reasons = anchor_reason_cpu.index_select(
                    0, anchor_cache_rows
                ).to(device=anchor_logits.device, dtype=torch.int8)
                for reason in _STUDENT_ANCHOR_REASON_NAMES:
                    per_sample_anchor_reason_fraction[reason].append(
                        anchor_response_aligned(selected_reasons.eq(reason).float() * keep)
                    )
            else:
                zero_rows = logits_chunk.new_zeros(logits_chunk.size(0))
                per_sample_anchor_kl.append(zero_rows)
                per_sample_anchor_old_mass.append(zero_rows.clone())
                per_sample_anchor_current_mass.append(zero_rows.clone())
                per_sample_anchor_fraction.append(zero_rows.clone())
                for reason in _STUDENT_ANCHOR_REASON_NAMES:
                    per_sample_anchor_reason_fraction[reason].append(zero_rows.clone())
                if nm_sampled_token:
                    nm_local_counts.append(0)
                    per_sample_nm_pg.append(zero_rows.clone())
                    per_sample_nm_advantage.append(zero_rows.clone())
                    per_sample_nm_clipfrac.append(zero_rows.clone())
                    per_sample_nm_fraction.append(zero_rows.clone())

    reverse_kl_cat = torch.cat(per_sample_reverse_kl)
    reverse_loss = sum_of_sample_mean(reverse_kl_cat)
    if include_forward:
        forward_kl_cat = torch.cat(per_sample_forward_kl)
        forward_loss = sum_of_sample_mean(forward_kl_cat)
        loss = forward_coef * forward_loss + reverse_coef * reverse_loss
    else:
        forward_loss = None
        loss = reverse_loss
    anchor_loss = None
    teacher_metric_scales = [1.0] * len(per_sample_reverse_kl)
    anchor_metric_scales = [1.0] * len(per_sample_reverse_kl)
    nm_metric_scales = [0.0] * len(per_sample_reverse_kl)
    nm_counts = [0] * len(per_sample_reverse_kl)
    if nm_sampled_token:
        # The denominator is per SAMPLE and must agree on every CP rank, while each rank
        # only sees its own slice of the chunk fields: sum the local n:m counts.
        counts = torch.tensor(nm_local_counts, dtype=torch.long, device=logits.device)
        if mpu.get_context_parallel_world_size() > 1:
            dist.all_reduce(counts, group=mpu.get_context_parallel_group())
        nm_counts = [int(c) for c in counts.tolist()]
    if include_student_anchor:
        anchor_loss = sum_of_sample_mean(torch.cat(per_sample_anchor_kl))
        combined_rows: list[torch.Tensor] = []
        for sample_idx, anchor_rows in enumerate(per_sample_anchor_kl):
            if include_forward:
                teacher_rows = (
                    forward_coef * per_sample_forward_kl[sample_idx]
                    + reverse_coef * per_sample_reverse_kl[sample_idx]
                )
            else:
                teacher_rows = per_sample_reverse_kl[sample_idx]
            teacher_count = int(
                batch["teacher_candidate_positions"][sample_idx].numel()
            )
            total_anchor_count = int(batch["student_anchor_positions"][sample_idx].numel())
            nm_count = nm_counts[sample_idx]
            anchor_count = total_anchor_count - nm_count
            weighted_count = teacher_count + nm_count + anchor_coef * anchor_count
            if weighted_count <= 0:
                raise ValueError(
                    "cross-tokenizer Student-anchor sample has zero effective rows"
                )
            # sum_of_sample_mean divides by the original number of trainable
            # tokens.  This factor changes that denominator to
            # |Teacher| + lambda_anchor * |Anchor| without ever materializing
            # a dense [response_length, K] distribution.
            all_rows = teacher_count + total_anchor_count
            denominator_scale = all_rows / weighted_count
            teacher_metric_scales[sample_idx] = (
                all_rows / teacher_count
                if teacher_count
                else 0.0
            )
            anchor_metric_scales[sample_idx] = (
                all_rows / anchor_count
                if anchor_count
                else 0.0
            )
            if nm_sampled_token:
                nm_metric_scales[sample_idx] = all_rows / nm_count if nm_count else 0.0
                # Each group averaged over its own rows, weights 1, 1, lambda: the
                # per-group scales turn sum_of_sample_mean's 1/|T + A| into 1/|group|.
                # An empty group contributes nothing.
                combined_rows.append(
                    teacher_metric_scales[sample_idx] * teacher_rows
                    + nm_metric_scales[sample_idx] * per_sample_nm_pg[sample_idx]
                    + anchor_coef * anchor_metric_scales[sample_idx] * anchor_rows
                )
            else:
                combined_rows.append(
                    denominator_scale * (teacher_rows + anchor_coef * anchor_rows)
                )
        loss = sum_of_sample_mean(torch.cat(combined_rows))
    if not reverse_kl_cat.numel():
        loss = loss + 0 * logits.sum()

    def scaled_metric(
        rows: list[torch.Tensor], scales: list[float]
    ) -> torch.Tensor:
        return sum_of_sample_mean(
            torch.cat([row * scale for row, scale in zip(rows, scales, strict=True)])
        ).detach().clone()

    reported_reverse_loss = (
        scaled_metric(per_sample_reverse_kl, teacher_metric_scales)
        if include_student_anchor
        else reverse_loss.detach().clone()
    )
    reported_forward_loss = (
        scaled_metric(per_sample_forward_kl, teacher_metric_scales)
        if include_student_anchor and forward_loss is not None
        else (forward_loss.detach().clone() if forward_loss is not None else None)
    )
    log: dict[str, torch.Tensor] = {
        "loss": loss.detach().clone(),
        "opd_reverse_kl": reported_reverse_loss,
    }
    if forward_loss is not None:
        assert reported_forward_loss is not None
        log.update(
            {
                "opd_forward_kl": reported_forward_loss,
                "opd_forward_kl_weighted": (
                    forward_coef * reported_forward_loss
                ).clone(),
                "opd_reverse_kl_weighted": (
                    reverse_coef * reported_reverse_loss
                ).clone(),
                f"{MAX_METRIC_PREFIX}opd_forward_kl_coef": reported_forward_loss.new_tensor(
                    forward_coef
                ),
                f"{MAX_METRIC_PREFIX}opd_reverse_kl_coef": reported_reverse_loss.new_tensor(
                    reverse_coef
                ),
            }
        )
    if anchor_loss is not None:
        reported_anchor_loss = scaled_metric(
            per_sample_anchor_kl, anchor_metric_scales
        )
        log.update(
            {
                "offline_opd/student_anchor_kl": reported_anchor_loss,
                "offline_opd/student_anchor_kl_weighted": (
                    anchor_coef * reported_anchor_loss
                ).clone(),
                "offline_opd/student_anchor_rollout_topk_mass": scaled_metric(
                    per_sample_anchor_old_mass, anchor_metric_scales
                ),
                "offline_opd/student_anchor_current_mass_on_rollout_topk": scaled_metric(
                    per_sample_anchor_current_mass, anchor_metric_scales
                ),
                "offline_opd/student_anchor_row_fraction": sum_of_sample_mean(
                    torch.cat(per_sample_anchor_fraction)
                ).detach().clone(),
            }
        )
        for reason, name in _STUDENT_ANCHOR_REASON_NAMES.items():
            log[f"offline_opd/student_anchor_{name}_fraction"] = sum_of_sample_mean(
                torch.cat(per_sample_anchor_reason_fraction[reason])
            ).detach().clone()
    if nm_sampled_token:
        log.update(
            {
                "offline_opd/nm_sampled_token_row_fraction": sum_of_sample_mean(
                    torch.cat(per_sample_nm_fraction)
                ).detach().clone(),
                "offline_opd/nm_sampled_token_pg_loss": scaled_metric(
                    [rows.detach() for rows in per_sample_nm_pg], nm_metric_scales
                ),
                "offline_opd/nm_sampled_token_advantage_mean": scaled_metric(
                    per_sample_nm_advantage, nm_metric_scales
                ),
                "offline_opd/nm_sampled_token_ppo_clipfrac": scaled_metric(
                    per_sample_nm_clipfrac, nm_metric_scales
                ),
            }
        )
    if let_me_forward_else_reverse:
        log["offline_opd/let_me_forward_row_fraction"] = scaled_metric(
            per_sample_let_me_forward_fraction, teacher_metric_scales
        )
    reduced = {
        name: (
            scaled_metric(rows, teacher_metric_scales)
            if include_student_anchor
            else sum_of_sample_mean(torch.cat(rows)).detach().clone()
        )
        for name, rows in per_sample.items()
    }
    log["student_mass"] = reduced["student_mass"]
    log["teacher_mass"] = reduced["teacher_mass"]
    for width in mass_at_widths:
        log[f"student_mass_at{width}"] = reduced[f"student_mass_at{width}"]
        log[f"teacher_mass_at{width}"] = reduced[f"teacher_mass_at{width}"]
    log["opd_topk_overlap"] = reduced["teacher_overlap"]
    log["overlap_ratio"] = reduced["teacher_overlap"]
    if include_forward:
        log.update(
            {
                "entropy": reduced["entropy"],
                "offline_opd/bidirectional_forward_student_mass": reduced[
                    "forward_student_mass"
                ],
                "offline_opd/bidirectional_forward_teacher_mass": reduced[
                    "forward_teacher_mass"
                ],
                "offline_opd/bidirectional_forward_teacher_vocab_skip_count": reduced[
                    "forward_teacher_vocab_skip_count"
                ],
                "offline_opd/bidirectional_forward_teacher_vocab_skip_ratio": reduced[
                    "forward_teacher_vocab_skip_count"
                ]
                / support_k,
            }
        )
    log.update(
        {
            "offline_opd_dynamic/teacher_cache_hit_ratio": reduced["cache_hit_ratio"],
            "offline_opd_dynamic/teacher_cache_miss_count": reduced["cache_miss_count"],
            "offline_opd_dynamic/teacher_cache_miss_student_mass": reduced["cache_miss_student_mass"],
            "offline_opd_dynamic/loss_weight_on_imputed_tokens": reduced["loss_weight_on_imputed_tokens"],
            "offline_opd_dynamic/teacher_mass_cached_lower_bound": reduced["teacher_mass_lower"],
            "offline_opd_dynamic/teacher_mass_cached_upper_bound": reduced["teacher_mass_upper"],
            # Keep the historical name for the original 4096-wide cache while
            # reporting the actual configured cache width for derived caches.
            f"offline_opd_dynamic/teacher_top{cache_k}_mass": reduced["teacher_cache_mass"],
            "offline_opd_dynamic/current_vs_cached_teacher_sparse_kl_estimated": reduced["teacher_estimated_kl"],
        }
    )
    for width, rows in overlap_at.items():
        if rows:
            log[f"opd_topk_overlap_at{width}"] = (
                scaled_metric(rows, teacher_metric_scales)
                if include_student_anchor
                else sum_of_sample_mean(torch.cat(rows)).detach().clone()
            )
    if rollout_overlap_full:
        log["offline_opd/current_vs_rollout_topk_overlap"] = (
            scaled_metric(rollout_overlap_full, teacher_metric_scales)
            if include_student_anchor
            else sum_of_sample_mean(torch.cat(rollout_overlap_full)).detach().clone()
        )
    for width, rows in rollout_overlap_at.items():
        if rows:
            log[f"offline_opd/current_vs_rollout_topk_overlap_at{width}"] = (
                scaled_metric(rows, teacher_metric_scales)
                if include_student_anchor
                else sum_of_sample_mean(torch.cat(rows)).detach().clone()
            )
    if current_vs_rollout_kl:
        reduce_teacher_metric = lambda rows: (
            scaled_metric(rows, teacher_metric_scales)
            if include_student_anchor
            else sum_of_sample_mean(torch.cat(rows)).detach().clone()
        )
        log["offline_opd/current_vs_rollout_sparse_kl"] = reduce_teacher_metric(
            current_vs_rollout_kl
        )
        log["offline_opd/current_mass_on_rollout_topk"] = reduce_teacher_metric(
            current_mass_on_rollout
        )
        log["offline_opd/support_mass_retention"] = reduce_teacher_metric(
            support_mass_retention
        )
        if include_forward and rollout_topk_entropy:
            rollout_entropy = reduce_teacher_metric(rollout_topk_entropy)
            current_fixed_support_entropy = reduce_teacher_metric(
                current_on_rollout_topk_entropy
            )
            log["rollout_topk_entropy"] = rollout_entropy
            log["current_on_rollout_topk_entropy"] = current_fixed_support_entropy
            log["entropy_change_vs_rollout_topk"] = (
                current_fixed_support_entropy - rollout_entropy
            )

    # Match the existing offline metric reducer: one token-ratio mean per sequence,
    # then moments across sequences; sequence ratios use the product in log space.
    if importance_weights:
        sequence_stats = torch.stack(
            [
                torch.stack((weights.sum(), weights.new_tensor(weights.numel()), ratios.sum()))
                for weights, ratios in zip(importance_weights, importance_log_ratios, strict=True)
            ]
        ).detach()
        if mpu.get_context_parallel_world_size() > 1:
            dist.all_reduce(sequence_stats, group=mpu.get_context_parallel_group())
        valid = sequence_stats[:, 1] > 0
        token_mean = sequence_stats[valid, 0] / sequence_stats[valid, 1]
        sequence_weight = torch.exp(
            sequence_stats[valid, 2].clamp(-OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND)
        )
        count = valid.sum().to(dtype=logits.dtype)
        log["offline_importance_weight_sum"] = token_mean.sum()
        log["offline_importance_weight_sqsum"] = token_mean.square().sum()
        log["offline_importance_weight_count"] = count
        log["offline_sequence_importance_weight_sum"] = sequence_weight.sum()
        log["offline_sequence_importance_weight_sqsum"] = sequence_weight.square().sum()
        log["offline_sequence_importance_weight_count"] = count.clone()
    return loss, log


CHUNK_KIND_TIERS: dict[str, frozenset[int]] = {
    # xvocab_chunk.KIND_CODES: 1 exact, 2 struct_1to1, 3 struct_n, 4 text_multi.
    "exact": frozenset({1, 2}),
    "struct": frozenset({1, 2, 3}),
    "chunk": frozenset({1, 2, 3, 4}),
}


def _append_offline_importance_weight_stats(
    log: dict[str, torch.Tensor],
    per_sample_ratio: list[torch.Tensor],
    per_sample_ratio_log: list[torch.Tensor],
    local_masks,
    dtype: torch.dtype,
) -> None:
    """Sequence-balanced importance-weight aggregation shared by the PG losses.

    Each observation is one sequence's mean token ratio; the sequence metric is
    the product ratio accumulated in log space.
    """
    sequence_stats = torch.stack(
        [
            torch.stack(((weights * mask).sum(), mask.sum(), (ratios * mask).sum()))
            for weights, ratios, mask in zip(
                per_sample_ratio, per_sample_ratio_log, local_masks, strict=True
            )
        ]
    ).detach()
    if mpu.get_context_parallel_world_size() > 1:
        dist.all_reduce(sequence_stats, group=mpu.get_context_parallel_group())
    valid = sequence_stats[:, 1] > 0
    token_mean = sequence_stats[valid, 0] / sequence_stats[valid, 1]
    sequence_weight = torch.exp(
        sequence_stats[valid, 2].clamp(-OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND)
    )
    count = valid.sum().to(dtype=dtype)
    log["offline_importance_weight_sum"] = token_mean.sum()
    log["offline_importance_weight_sqsum"] = token_mean.square().sum()
    log["offline_importance_weight_count"] = count
    log["offline_sequence_importance_weight_sum"] = sequence_weight.sum()
    log["offline_sequence_importance_weight_sqsum"] = sequence_weight.square().sum()
    log["offline_sequence_importance_weight_count"] = count.clone()


def _sampled_token_pg_options(args: Namespace) -> dict[str, Any]:
    """Validated sampled-token PG knobs shared by every loss that uses ``_sampled_token_pg_rows``."""
    options = {
        "teacher_coef": float(getattr(args, "opd_kl_coef", 1.0)),
        "anchor_coef": float(getattr(args, "opd_student_anchor_kl_coef", 1.0)),
        "miss_policy": getattr(args, "opd_sampled_token_cache_miss_policy", "anchor"),
        "clip_reference": getattr(args, "opd_sampled_token_clip_reference", "rollout"),
        "use_sdar_gate": bool(getattr(args, "opd_sampled_token_sdar_gate", False)),
        "sdar_beta": float(getattr(args, "opd_sdar_gate_beta", 5.0)),
        "sdar_gate_scale": float(getattr(args, "opd_sdar_gate_scale", 1.0)),
    }
    if options["teacher_coef"] < 0 or options["anchor_coef"] < 0:
        raise ValueError(
            "sampled-token OPD coefficients must be non-negative, got "
            f"teacher={options['teacher_coef']}, anchor={options['anchor_coef']}"
        )
    if options["miss_policy"] not in {"anchor", "mask"}:
        raise ValueError(
            "opd_sampled_token_cache_miss_policy must be 'anchor' or 'mask', "
            f"got {options['miss_policy']!r}"
        )
    if options["clip_reference"] not in {"rollout", "current"}:
        raise ValueError(
            "opd_sampled_token_clip_reference must be 'rollout' or 'current', "
            f"got {options['clip_reference']!r}"
        )
    eps_clip = float(getattr(args, "eps_clip", 0.2))
    eps_clip_high_value = getattr(args, "eps_clip_high", None)
    options["eps_clip"] = eps_clip
    options["eps_clip_high"] = eps_clip if eps_clip_high_value is None else float(eps_clip_high_value)
    return options


def _sampled_token_pg_rows(
    current_lp: torch.Tensor,
    rollout_lp: torch.Tensor,
    target_lp: torch.Tensor,
    teacher_hit_rows: torch.Tensor,
    *,
    teacher_coef: float,
    anchor_coef: float,
    miss_policy: str,
    clip_reference: str,
    use_sdar_gate: bool,
    sdar_beta: float,
    sdar_gate_scale: float,
    eps_clip: float,
    eps_clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Per-row sampled-token PG surrogate: ``(pg_rows, advantage, clipfrac, sdar_gate)``.

    Detached advantage ``coef * (target - current)`` (teacher rows at ``teacher_coef``,
    the rest at ``anchor_coef``, or 0 under miss policy ``mask``), PPO ratio against the
    rollout (or current) log-prob. Shared by the chunk sampled-token loss and the n:m rows
    of the anchored top-K loss, so the two cannot drift apart.
    """
    fallback_rows = ~teacher_hit_rows
    row_coef = torch.full_like(current_lp, anchor_coef)
    row_coef[teacher_hit_rows] = teacher_coef
    if miss_policy == "mask":
        row_coef[fallback_rows] = 0.0

    raw_advantage = target_lp - current_lp.detach()
    sdar_gate = None
    if use_sdar_gate:
        sdar_gate = torch.ones_like(raw_advantage)
        sdar_gate[teacher_hit_rows] = (
            sdar_gate_scale * torch.sigmoid(sdar_beta * raw_advantage[teacher_hit_rows])
        ).detach()
        advantage = row_coef * torch.where(
            teacher_hit_rows, sdar_gate * raw_advantage, raw_advantage
        )
    else:
        advantage = row_coef * raw_advantage
    clip_reference_lp = (
        rollout_lp.detach() if clip_reference == "rollout" else current_lp.detach()
    )
    pg_rows, clipfrac = compute_policy_loss(
        clip_reference_lp - current_lp, advantage, eps_clip, eps_clip_high
    )
    return pg_rows, advantage, clipfrac, sdar_gate


def _opd_chunk_teacher_sampled_token_pg_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sampled-token OPD whose Teacher value comes from a byte-chunk alignment.

    Same estimator as ``_opd_cached_teacher_sampled_token_pg_loss`` -- detached
    advantage, Student-anchor fallback, PPO ratio against the rollout -- but the
    Teacher term is read straight off a response-aligned chunk artifact instead
    of looked up in a sparse Top-K cache through a token-id bridge.  There is
    therefore no cache-miss category: a row either has a chunk share of the
    kind this tier accepts, or it anchors on the rollout.

    For an n:m chunk every covered Student row carries ``chunk_total / n``, so
    the advantage SUM over a chunk is the Teacher's assessment of the whole
    span even though the per-row split is arbitrary.
    """
    required = ("teacher_chunk_logprob", "teacher_chunk_mask", "teacher_chunk_kind", "rollout_log_probs")
    missing = [key for key in required if not batch.get(key)]
    if missing:
        raise ValueError(
            "chunk-teacher sampled-token offline OPD batch is missing: " + ", ".join(missing)
        )
    tier = getattr(args, "opd_chunk_tier", "exact")
    if tier not in CHUNK_KIND_TIERS:
        raise ValueError(
            f"opd_chunk_tier must be one of {sorted(CHUNK_KIND_TIERS)}, got {tier!r}"
        )
    allowed_kinds = sorted(CHUNK_KIND_TIERS[tier])
    pg_options = _sampled_token_pg_options(args)
    use_sdar_gate = pg_options["use_sdar_gate"]
    clip_reference = pg_options["clip_reference"]

    per_sample_pg: list[torch.Tensor] = []
    per_sample_entropy: list[torch.Tensor] = []
    per_sample_advantage: list[torch.Tensor] = []
    per_sample_log_ratio_to_target: list[torch.Tensor] = []
    per_sample_ratio: list[torch.Tensor] = []
    per_sample_ratio_log: list[torch.Tensor] = []
    per_sample_clipfrac: list[torch.Tensor] = []
    per_sample_teacher_hit: list[torch.Tensor] = []
    per_sample_covered: list[torch.Tensor] = []
    per_sample_fallback: list[torch.Tensor] = []
    per_sample_sdar_gate: list[torch.Tensor] = []

    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        apply_temperature=True,
    )
    for sample_idx, (logits_chunk, tokens_chunk) in enumerate(responses):
        current_lp_rows, current_entropy = vocab_parallel_logprobs_and_entropy_on_ids(
            logits_chunk,
            tokens_chunk.reshape(-1, 1),
            mpu.get_tensor_model_parallel_group(),
            chunk_size=getattr(args, "log_probs_chunk_size", -1),
        )
        current_lp = current_lp_rows[:, 0]
        rollout_lp = batch["rollout_log_probs"][sample_idx].to(
            device=current_lp.device, dtype=torch.float32, non_blocking=True
        )
        chunk_lp = batch["teacher_chunk_logprob"][sample_idx].to(
            device=current_lp.device, dtype=torch.float32, non_blocking=True
        )
        chunk_mask = batch["teacher_chunk_mask"][sample_idx].to(
            device=current_lp.device, non_blocking=True
        )
        chunk_kind = batch["teacher_chunk_kind"][sample_idx].to(
            device=current_lp.device, non_blocking=True
        )
        for name, value in (
            ("rollout_log_probs", rollout_lp),
            ("teacher_chunk_logprob", chunk_lp),
            ("teacher_chunk_mask", chunk_mask),
            ("teacher_chunk_kind", chunk_kind),
        ):
            if value.numel() != current_lp.numel():
                raise ValueError(
                    f"chunk-teacher sampled-token length mismatch on {name}: "
                    f"{value.numel()} != current={current_lp.numel()}"
                )
        covered = chunk_mask.bool()
        in_tier = torch.zeros_like(covered)
        for code in allowed_kinds:
            in_tier |= chunk_kind.eq(code)
        teacher_hit_rows = covered & in_tier

        target_lp = rollout_lp.detach().clone()
        target_lp = torch.where(teacher_hit_rows, chunk_lp, target_lp)
        fallback_rows = ~teacher_hit_rows
        pg_rows, advantage, clipfrac, sdar_gate = _sampled_token_pg_rows(
            current_lp, rollout_lp, target_lp, teacher_hit_rows, **pg_options
        )
        if use_sdar_gate:
            per_sample_sdar_gate.append(torch.where(teacher_hit_rows, sdar_gate, 0.0))
        log_ratio = current_lp.detach() - rollout_lp.detach()

        per_sample_pg.append(pg_rows)
        per_sample_entropy.append(current_entropy.detach())
        per_sample_advantage.append(advantage)
        per_sample_log_ratio_to_target.append(current_lp.detach() - target_lp)
        per_sample_ratio.append(
            torch.exp(log_ratio.clamp(-OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND))
        )
        per_sample_ratio_log.append(log_ratio)
        per_sample_clipfrac.append(clipfrac)
        per_sample_teacher_hit.append(teacher_hit_rows.float())
        per_sample_covered.append(covered.float())
        per_sample_fallback.append(fallback_rows.float())

    pg_cat = torch.cat(per_sample_pg)
    loss = sum_of_sample_mean(pg_cat)
    if not pg_cat.numel():
        loss = loss + 0 * logits.sum()

    log: dict[str, torch.Tensor] = {
        "loss": loss.detach().clone(),
        "opd_sampled_token_pg_loss": loss.detach().clone(),
        "entropy": sum_of_sample_mean(torch.cat(per_sample_entropy)).detach().clone(),
        "offline_opd/sampled_token_advantage_mean": sum_of_sample_mean(
            torch.cat(per_sample_advantage)
        ).detach().clone(),
        "offline_opd/sampled_token_log_ratio_to_target": sum_of_sample_mean(
            torch.cat(per_sample_log_ratio_to_target)
        ).detach().clone(),
        "offline_opd/sampled_token_ppo_clipfrac": sum_of_sample_mean(
            torch.cat(per_sample_clipfrac)
        ).detach().clone(),
        "offline_opd/sampled_token_teacher_hit_fraction": sum_of_sample_mean(
            torch.cat(per_sample_teacher_hit)
        ).detach().clone(),
        # Rows the artifact covers at ANY kind: the gap to the hit fraction is
        # exactly what this tier throws away.
        "offline_opd/chunk_covered_fraction": sum_of_sample_mean(
            torch.cat(per_sample_covered)
        ).detach().clone(),
        "offline_opd/sampled_token_rollout_anchor_fraction": sum_of_sample_mean(
            torch.cat(per_sample_fallback)
        ).detach().clone(),
        f"{MAX_METRIC_PREFIX}offline_opd/chunk_tier": loss.new_tensor(
            float(sorted(CHUNK_KIND_TIERS).index(tier))
        ),
        f"{MAX_METRIC_PREFIX}offline_opd/sampled_token_clip_reference_current": loss.new_tensor(
            float(clip_reference == "current")
        ),
    }
    if use_sdar_gate:
        log["offline_opd/sdar_gate_mean"] = (
            sum_of_sample_mean(torch.cat(per_sample_sdar_gate)).detach().clone()
        )

    local_masks = get_concat_loss_mask(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"]
    ).split([x.numel() for x in per_sample_ratio])
    _append_offline_importance_weight_stats(
        log, per_sample_ratio, per_sample_ratio_log, local_masks, logits.dtype
    )
    return loss, log


def _opd_cached_teacher_sampled_token_pg_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Offline sampled-token OPD using a detached advantage and PPO ratio.

    For an exact cross-tokenizer row whose sampled Student token is present in
    the sparse Teacher cache, define

        delta = log p_teacher(a) - log p_current(a)
        gate = stopgrad(sigmoid(beta * delta))
        advantage = stopgrad(gate * delta).

    ``beta`` controls only the sigmoid slope; it is not an additional loss
    multiplier.  There is no reward/RL advantage in this offline objective.

    By default the policy-gradient surrogate uses
    ``p_current(a) / p_rollout(a)`` (with the normal PPO clip).  The isolated
    ``current`` clip-reference mode instead uses
    ``p_current(a) / stopgrad(p_current(a))``: its numerical ratio is one, but
    the numerator still supplies the sampled-token policy gradient.  Unaligned
    rows and exact cache misses use the rollout Student sampled-token
    probability as an anchor by default.
    """
    required = (
        "teacher_candidate_positions",
        "teacher_candidate_ids",
        "teacher_candidate_logprobs",
        "rollout_log_probs",
    )
    missing = [key for key in required if not batch.get(key)]
    if missing:
        raise ValueError(
            "cached sampled-token offline OPD batch is missing: "
            + ", ".join(missing)
        )

    collect_topk_metrics = bool(
        getattr(args, "offline_opd_topk_overlap", False)
    )
    metric_topk = 8
    metric_widths = (1, 2, 4, 8)
    if collect_topk_metrics:
        metric_required = (
            "teacher_top128_ids",
            "teacher_cutoff_logprobs",
            "teacher_top4096_mass",
        )
        metric_missing = [key for key in metric_required if not batch.get(key)]
        if metric_missing:
            raise ValueError(
                "cached sampled-token Top-K metrics batch is missing: "
                + ", ".join(metric_missing)
            )

    cache_k = int(getattr(args, "opd_teacher_cache_topk", 4096))
    rows_per_chunk = int(getattr(args, "opd_teacher_cache_h2d_rows", 512))
    teacher_coef = float(getattr(args, "opd_kl_coef", 1.0))
    anchor_coef = float(getattr(args, "opd_student_anchor_kl_coef", 1.0))
    miss_policy = getattr(args, "opd_sampled_token_cache_miss_policy", "anchor")
    clip_reference = getattr(
        args, "opd_sampled_token_clip_reference", "rollout"
    )
    use_sdar_gate = bool(getattr(args, "opd_sampled_token_sdar_gate", False))
    sdar_beta = float(getattr(args, "opd_sdar_gate_beta", 5.0))
    sdar_gate_scale = float(getattr(args, "opd_sdar_gate_scale", 1.0))
    if teacher_coef < 0 or anchor_coef < 0:
        raise ValueError(
            "sampled-token OPD coefficients must be non-negative, got "
            f"teacher={teacher_coef}, anchor={anchor_coef}"
        )
    if miss_policy not in {"anchor", "mask"}:
        raise ValueError(
            "opd_sampled_token_cache_miss_policy must be 'anchor' or 'mask', "
            f"got {miss_policy!r}"
        )
    if clip_reference not in {"rollout", "current"}:
        raise ValueError(
            "opd_sampled_token_clip_reference must be 'rollout' or 'current', "
            f"got {clip_reference!r}"
        )
    if use_sdar_gate and not math.isfinite(sdar_beta):
        raise ValueError(f"opd_sdar_gate_beta must be finite, got {sdar_beta}")
    if use_sdar_gate and sdar_beta <= 0:
        raise ValueError(f"opd_sdar_gate_beta must be positive, got {sdar_beta}")
    if use_sdar_gate and (
        not math.isfinite(sdar_gate_scale) or sdar_gate_scale <= 0
    ):
        raise ValueError(
            "opd_sdar_gate_scale must be finite and positive, got "
            f"{sdar_gate_scale}"
        )

    token_id_bridge_path = getattr(args, "opd_token_id_bridge", None)
    valid_student_vocab_size = getattr(args, "opd_student_valid_vocab_size", None)
    if valid_student_vocab_size is None:
        valid_student_vocab_size = getattr(args, "vocab_size", None)
    if collect_topk_metrics and valid_student_vocab_size is None:
        raise ValueError(
            "Student valid vocabulary size is required for sampled-token Top-K metrics"
        )
    if valid_student_vocab_size is not None:
        valid_student_vocab_size = int(valid_student_vocab_size)
    if collect_topk_metrics and token_id_bridge_path:
        from slime.utils.token_id_bridge import load_token_id_bridge

        bridge = load_token_id_bridge(token_id_bridge_path)
        if valid_student_vocab_size != bridge.student_tokenizer_size:
            raise ValueError(
                "sampled-token Top-K metric vocabulary does not match token-id bridge: "
                f"{valid_student_vocab_size} != {bridge.student_tokenizer_size}"
            )
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world_size = getattr(
        mpu, "get_tensor_model_parallel_world_size", lambda: 1
    )()
    tp_rank = getattr(mpu, "get_tensor_model_parallel_rank", lambda: 0)()
    eps_clip = float(getattr(args, "eps_clip", 0.2))
    eps_clip_high_value = getattr(args, "eps_clip_high", None)
    eps_clip_high = (
        eps_clip if eps_clip_high_value is None else float(eps_clip_high_value)
    )

    per_sample_pg: list[torch.Tensor] = []
    per_sample_entropy: list[torch.Tensor] = []
    per_sample_advantage: list[torch.Tensor] = []
    per_sample_log_ratio_to_target: list[torch.Tensor] = []
    per_sample_ratio: list[torch.Tensor] = []
    per_sample_ratio_log: list[torch.Tensor] = []
    per_sample_clipfrac: list[torch.Tensor] = []
    per_sample_exact: list[torch.Tensor] = []
    per_sample_teacher_hit: list[torch.Tensor] = []
    per_sample_exact_miss: list[torch.Tensor] = []
    per_sample_fallback: list[torch.Tensor] = []
    per_sample_sdar_gate: list[torch.Tensor] = []
    per_sample_sdar_positive: list[torch.Tensor] = []
    per_sample_sdar_negative: list[torch.Tensor] = []
    per_sample_sdar_raw_positive: list[torch.Tensor] = []
    per_sample_sdar_raw_negative_abs: list[torch.Tensor] = []
    per_sample_sdar_gated_positive: list[torch.Tensor] = []
    per_sample_sdar_gated_negative_abs: list[torch.Tensor] = []
    per_sample_topk_metrics: dict[str, list[torch.Tensor]] = {
        name: []
        for name in (
            *(f"opd_topk_overlap_at{width}" for width in metric_widths),
            *(f"student_mass_at{width}" for width in metric_widths),
            *(f"teacher_mass_at{width}" for width in metric_widths),
        )
    }
    topk_metric_scales: list[float] = []

    def exact_overlap(
        left: torch.Tensor, right: torch.Tensor, width: int
    ) -> torch.Tensor:
        left_sorted = torch.sort(left[:, :width], dim=-1).values
        query = right[:, :width].contiguous()
        columns = torch.searchsorted(left_sorted, query)
        valid = columns.lt(width)
        matched = valid & torch.gather(
            left_sorted, 1, columns.clamp_max(width - 1)
        ).eq(query)
        return matched.float().sum(dim=-1) / width

    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        apply_temperature=True,
    )
    for sample_idx, (logits_chunk, tokens_chunk) in enumerate(responses):
        local_positions = _dynamic_local_response_positions(
            args, batch, logits, sample_idx, logits_chunk.size(0)
        )
        current_topk_ids = None
        query_ids = tokens_chunk.reshape(-1, 1)
        if collect_topk_metrics:
            current_topk_ids = vocab_parallel_topk_ids(
                logits_chunk,
                metric_topk,
                tp_group,
                chunk_size=getattr(args, "log_probs_chunk_size", -1),
                global_vocab_size=valid_student_vocab_size,
            )
            query_ids = torch.cat((query_ids, current_topk_ids), dim=-1)
        current_lp_rows, current_entropy = vocab_parallel_logprobs_and_entropy_on_ids(
            logits_chunk,
            query_ids,
            tp_group,
            chunk_size=getattr(args, "log_probs_chunk_size", -1),
        )
        current_lp = current_lp_rows[:, 0]
        current_topk_lp = (
            current_lp_rows[:, 1:].detach() if collect_topk_metrics else None
        )
        rollout_lp = batch["rollout_log_probs"][sample_idx].to(
            device=current_lp.device, dtype=torch.float32, non_blocking=True
        )
        if rollout_lp.numel() != current_lp.numel():
            raise ValueError(
                "sampled-token rollout/current length mismatch: "
                f"rollout={rollout_lp.numel()}, current={current_lp.numel()}"
            )

        target_lp = rollout_lp.detach().clone()
        exact_rows = torch.zeros_like(current_lp, dtype=torch.bool)
        teacher_hit_rows = torch.zeros_like(current_lp, dtype=torch.bool)

        cache_positions = batch["teacher_candidate_positions"][sample_idx].detach().cpu().long()
        candidate_ids_cpu = batch["teacher_candidate_ids"][sample_idx]
        candidate_lp_cpu = batch["teacher_candidate_logprobs"][sample_idx]
        teacher_head_cpu = (
            batch["teacher_top128_ids"][sample_idx]
            if collect_topk_metrics
            else None
        )
        cutoff_cpu = (
            batch["teacher_cutoff_logprobs"][sample_idx]
            if collect_topk_metrics
            else None
        )
        cache_mass_cpu = (
            batch["teacher_top4096_mass"][sample_idx]
            if collect_topk_metrics
            else None
        )
        n_cache_rows = cache_positions.numel()
        if (
            tuple(candidate_ids_cpu.shape) != (n_cache_rows, cache_k)
            or tuple(candidate_lp_cpu.shape) != (n_cache_rows, cache_k)
        ):
            raise ValueError(
                "cached sampled-token Teacher shape mismatch: "
                f"positions={n_cache_rows}, ids={tuple(candidate_ids_cpu.shape)}, "
                f"lp={tuple(candidate_lp_cpu.shape)}, cache_k={cache_k}"
            )
        if collect_topk_metrics and (
            teacher_head_cpu is None
            or teacher_head_cpu.shape[0] != n_cache_rows
            or teacher_head_cpu.shape[1] < metric_topk
            or cutoff_cpu is None
            or tuple(cutoff_cpu.shape) != (n_cache_rows,)
            or cache_mass_cpu is None
            or tuple(cache_mass_cpu.shape) != (n_cache_rows,)
        ):
            raise ValueError(
                "cached sampled-token Top-K metric shape mismatch: "
                f"positions={n_cache_rows}, "
                f"head={None if teacher_head_cpu is None else tuple(teacher_head_cpu.shape)}, "
                f"cutoff={None if cutoff_cpu is None else tuple(cutoff_cpu.shape)}, "
                f"mass={None if cache_mass_cpu is None else tuple(cache_mass_cpu.shape)}, "
                f"metric_k={metric_topk}"
            )

        columns = torch.searchsorted(
            cache_positions, local_positions.detach().cpu()
        )
        safe = columns.clamp_max(max(n_cache_rows - 1, 0))
        selected = columns.lt(n_cache_rows)
        if n_cache_rows:
            selected &= cache_positions[safe].eq(local_positions.detach().cpu())
        local_row_indices = torch.nonzero(
            selected, as_tuple=False
        ).flatten().to(current_lp.device)
        cache_row_indices = columns[selected]
        exact_rows[local_row_indices] = True

        if collect_topk_metrics:
            assert current_topk_ids is not None
            assert current_topk_lp is not None
            assert teacher_head_cpu is not None
            assert cutoff_cpu is not None
            assert cache_mass_cpu is not None
            selected_current_ids = current_topk_ids.index_select(
                0, local_row_indices
            )
            selected_current_lp = current_topk_lp.index_select(
                0, local_row_indices
            )
            if tp_world_size == 1 or tp_rank == 0:
                teacher_support_ids, _, _ = _select_cached_teacher_topk_for_student(
                    cache_row_indices,
                    candidate_ids_cpu,
                    candidate_lp_cpu,
                    teacher_head_cpu,
                    support_k=metric_topk,
                    student_valid_vocab_size=valid_student_vocab_size,
                    token_id_bridge_path=token_id_bridge_path,
                    output_device=current_lp.device,
                    rows_per_chunk=rows_per_chunk,
                )
            else:
                teacher_support_ids = torch.empty(
                    (local_row_indices.numel(), metric_topk),
                    dtype=torch.long,
                    device=current_lp.device,
                )
            teacher_query_ids = selected_current_ids
            if token_id_bridge_path:
                from slime.utils.token_id_bridge import map_token_ids_torch

                teacher_query_ids = map_token_ids_torch(
                    teacher_query_ids,
                    token_id_bridge_path,
                    context="sampled-token current-Student Top-8 metrics",
                    allow_unmapped=True,
                )
            if tp_world_size == 1 or tp_rank == 0:
                (
                    teacher_on_current_lp,
                    _,
                    _,
                    _,
                    _,
                ) = _lookup_dynamic_teacher_cache(
                    teacher_query_ids,
                    cache_row_indices,
                    candidate_ids_cpu,
                    candidate_lp_cpu,
                    cutoff_cpu,
                    teacher_head_cpu,
                    cache_mass_cpu,
                    rows_per_chunk=rows_per_chunk,
                )
            else:
                teacher_on_current_lp = torch.empty(
                    (local_row_indices.numel(), metric_topk),
                    dtype=torch.float32,
                    device=current_lp.device,
                )
            if tp_world_size > 1:
                tp_src_rank = mpu.get_tensor_model_parallel_src_rank()
                dist.broadcast(
                    teacher_support_ids, src=tp_src_rank, group=tp_group
                )
                dist.broadcast(
                    teacher_on_current_lp, src=tp_src_rank, group=tp_group
                )
            teacher_on_current_lp = torch.nan_to_num(
                teacher_on_current_lp,
                nan=_LOGPROB_FLOOR,
                posinf=0.0,
                neginf=_LOGPROB_FLOOR,
            )

            def response_aligned(rows: torch.Tensor) -> torch.Tensor:
                output = rows.new_zeros(logits_chunk.size(0))
                output[local_row_indices] = rows
                return output

            student_mass_prefix = selected_current_lp.exp().cumsum(dim=-1)
            teacher_mass_prefix = teacher_on_current_lp.exp().cumsum(dim=-1)
            for width in metric_widths:
                per_sample_topk_metrics[f"opd_topk_overlap_at{width}"].append(
                    response_aligned(
                        exact_overlap(
                            teacher_support_ids, selected_current_ids, width
                        )
                    )
                )
                per_sample_topk_metrics[f"student_mass_at{width}"].append(
                    response_aligned(student_mass_prefix[:, width - 1])
                )
                per_sample_topk_metrics[f"teacher_mass_at{width}"].append(
                    response_aligned(teacher_mass_prefix[:, width - 1])
                )
            full_mask = batch["loss_masks"][sample_idx].detach().float().cpu()
            exact_mask_count = full_mask.index_select(
                0, cache_positions
            ).sum().item()
            full_mask_count = full_mask.sum().item()
            topk_metric_scales.append(
                full_mask_count / exact_mask_count if exact_mask_count else 0.0
            )

        sampled_teacher_ids = tokens_chunk.index_select(0, local_row_indices)
        if token_id_bridge_path:
            from slime.utils.token_id_bridge import map_token_ids_torch

            sampled_teacher_ids = map_token_ids_torch(
                sampled_teacher_ids,
                token_id_bridge_path,
                context="cached sampled-token Student actions",
                allow_unmapped=True,
            )
        if tp_world_size == 1 or tp_rank == 0:
            sampled_teacher_lp, sampled_teacher_hit = (
                _lookup_cached_teacher_logprobs_on_ids(
                    sampled_teacher_ids,
                    cache_row_indices,
                    candidate_ids_cpu,
                    candidate_lp_cpu,
                    rows_per_chunk=rows_per_chunk,
                )
            )
        else:
            sampled_teacher_lp = torch.empty(
                local_row_indices.numel(),
                dtype=torch.float32,
                device=current_lp.device,
            )
            sampled_teacher_hit = torch.empty(
                local_row_indices.numel(),
                dtype=torch.bool,
                device=current_lp.device,
            )
        if tp_world_size > 1:
            tp_src_rank = mpu.get_tensor_model_parallel_src_rank()
            dist.broadcast(sampled_teacher_lp, src=tp_src_rank, group=tp_group)
            dist.broadcast(sampled_teacher_hit, src=tp_src_rank, group=tp_group)
        sampled_teacher_lp = torch.nan_to_num(
            sampled_teacher_lp,
            nan=_LOGPROB_FLOOR,
            posinf=0.0,
            neginf=_LOGPROB_FLOOR,
        )
        hit_local_rows = local_row_indices[sampled_teacher_hit]
        target_lp[hit_local_rows] = sampled_teacher_lp[sampled_teacher_hit]
        teacher_hit_rows[hit_local_rows] = True

        fallback_rows = ~teacher_hit_rows
        row_coef = torch.full_like(current_lp, anchor_coef)
        row_coef[teacher_hit_rows] = teacher_coef
        if miss_policy == "mask":
            row_coef[fallback_rows] = 0.0

        raw_advantage = target_lp - current_lp.detach()
        if use_sdar_gate:
            # SDAR defines the token-level privileged-vs-ordinary Student gap
            # against the CURRENT Student.  The gate is explicitly detached so
            # the Student cannot improve the objective merely by manipulating
            # its gate.  The policy-gradient advantage below remains detached,
            # matching the existing sampled-token OPD estimator.
            sdar_gate = torch.ones_like(raw_advantage)
            teacher_gap = raw_advantage[teacher_hit_rows]
            sdar_gate[teacher_hit_rows] = (
                sdar_gate_scale * torch.sigmoid(sdar_beta * teacher_gap)
            ).detach()
            gated_teacher_advantage = sdar_gate * raw_advantage
            advantage = row_coef * torch.where(
                teacher_hit_rows,
                gated_teacher_advantage,
                raw_advantage,
            )
            positive_rows = teacher_hit_rows & raw_advantage.gt(0)
            negative_rows = teacher_hit_rows & raw_advantage.lt(0)
            per_sample_sdar_gate.append(
                torch.where(teacher_hit_rows, sdar_gate, 0.0)
            )
            per_sample_sdar_positive.append(positive_rows.float())
            per_sample_sdar_negative.append(negative_rows.float())
            per_sample_sdar_raw_positive.append(
                torch.where(positive_rows, raw_advantage, 0.0)
            )
            per_sample_sdar_raw_negative_abs.append(
                torch.where(negative_rows, -raw_advantage, 0.0)
            )
            per_sample_sdar_gated_positive.append(
                torch.where(positive_rows, gated_teacher_advantage, 0.0)
            )
            per_sample_sdar_gated_negative_abs.append(
                torch.where(negative_rows, -gated_teacher_advantage, 0.0)
            )
        else:
            advantage = row_coef * raw_advantage
        # ``current`` deliberately snapshots this same forward with detach.
        # Numerically ratio=1 and clipfrac=0, while autograd still sees
        # exp(current_lp - stopgrad(current_lp)) and therefore keeps the
        # sampled-token policy-gradient derivative.  This is distinct from
        # replacing the whole ratio with a literal constant one.
        clip_reference_lp = (
            rollout_lp.detach()
            if clip_reference == "rollout"
            else current_lp.detach()
        )
        reference_minus_current = clip_reference_lp - current_lp
        pg_rows, clipfrac = compute_policy_loss(
            reference_minus_current,
            advantage,
            eps_clip,
            eps_clip_high,
        )
        log_ratio = current_lp.detach() - rollout_lp.detach()
        ratio = torch.exp(
            log_ratio.clamp(-OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND)
        )

        per_sample_pg.append(pg_rows)
        per_sample_entropy.append(current_entropy.detach())
        per_sample_advantage.append(advantage)
        per_sample_log_ratio_to_target.append(current_lp.detach() - target_lp)
        per_sample_ratio.append(ratio)
        per_sample_ratio_log.append(log_ratio)
        per_sample_clipfrac.append(clipfrac)
        per_sample_exact.append(exact_rows.float())
        per_sample_teacher_hit.append(teacher_hit_rows.float())
        per_sample_exact_miss.append((exact_rows & ~teacher_hit_rows).float())
        per_sample_fallback.append(fallback_rows.float())

    pg_cat = torch.cat(per_sample_pg)
    loss = sum_of_sample_mean(pg_cat)
    if not pg_cat.numel():
        loss = loss + 0 * logits.sum()

    log: dict[str, torch.Tensor] = {
        "loss": loss.detach().clone(),
        "opd_sampled_token_pg_loss": loss.detach().clone(),
        # Exact full-vocabulary current-Student entropy.  The sparse gather
        # already constructs the full softmax required by the sampled-token
        # gradient, so this adds only a detached weighted reduction per row.
        # ``sum_of_sample_mean`` preserves the offline metric convention:
        # mean over trainable tokens within each sequence, then over sequences.
        "entropy": sum_of_sample_mean(
            torch.cat(per_sample_entropy)
        ).detach().clone(),
        "offline_opd/sampled_token_advantage_mean": sum_of_sample_mean(
            torch.cat(per_sample_advantage)
        ).detach().clone(),
        "offline_opd/sampled_token_log_ratio_to_target": sum_of_sample_mean(
            torch.cat(per_sample_log_ratio_to_target)
        ).detach().clone(),
        "offline_opd/sampled_token_ppo_clipfrac": sum_of_sample_mean(
            torch.cat(per_sample_clipfrac)
        ).detach().clone(),
        "offline_opd/sampled_token_exact_row_fraction": sum_of_sample_mean(
            torch.cat(per_sample_exact)
        ).detach().clone(),
        "offline_opd/sampled_token_teacher_hit_fraction": sum_of_sample_mean(
            torch.cat(per_sample_teacher_hit)
        ).detach().clone(),
        "offline_opd/sampled_token_teacher_cache_miss_fraction": sum_of_sample_mean(
            torch.cat(per_sample_exact_miss)
        ).detach().clone(),
        "offline_opd/sampled_token_rollout_anchor_fraction": sum_of_sample_mean(
            torch.cat(per_sample_fallback)
        ).detach().clone(),
        f"{MAX_METRIC_PREFIX}offline_opd/sampled_token_clip_reference_current": loss.new_tensor(
            float(clip_reference == "current")
        ),
    }
    if collect_topk_metrics:
        reduced_topk_metrics = {
            name: sum_of_sample_mean(
                torch.cat(
                    [
                        rows * scale
                        for rows, scale in zip(
                            per_sample_rows,
                            topk_metric_scales,
                            strict=True,
                        )
                    ]
                )
            ).detach().clone()
            for name, per_sample_rows in per_sample_topk_metrics.items()
        }
        log.update(reduced_topk_metrics)
        log["opd_topk_overlap"] = reduced_topk_metrics[
            f"opd_topk_overlap_at{metric_topk}"
        ]
        log["overlap_ratio"] = log["opd_topk_overlap"]
        log["student_mass"] = reduced_topk_metrics[
            f"student_mass_at{metric_topk}"
        ]
        log["teacher_mass"] = reduced_topk_metrics[
            f"teacher_mass_at{metric_topk}"
        ]
    if use_sdar_gate:
        log.update(
            {
                "offline_opd/sdar_gate_mass_per_train_token": sum_of_sample_mean(
                    torch.cat(per_sample_sdar_gate)
                ).detach().clone(),
                "offline_opd/sdar_positive_row_fraction": sum_of_sample_mean(
                    torch.cat(per_sample_sdar_positive)
                ).detach().clone(),
                "offline_opd/sdar_negative_row_fraction": sum_of_sample_mean(
                    torch.cat(per_sample_sdar_negative)
                ).detach().clone(),
                "offline_opd/sdar_raw_positive_advantage_mean": sum_of_sample_mean(
                    torch.cat(per_sample_sdar_raw_positive)
                ).detach().clone(),
                "offline_opd/sdar_raw_negative_advantage_abs_mean": sum_of_sample_mean(
                    torch.cat(per_sample_sdar_raw_negative_abs)
                ).detach().clone(),
                "offline_opd/sdar_gated_positive_advantage_mean": sum_of_sample_mean(
                    torch.cat(per_sample_sdar_gated_positive)
                ).detach().clone(),
                "offline_opd/sdar_gated_negative_advantage_abs_mean": sum_of_sample_mean(
                    torch.cat(per_sample_sdar_gated_negative_abs)
                ).detach().clone(),
                f"{MAX_METRIC_PREFIX}offline_opd/sdar_gate_beta": loss.new_tensor(
                    sdar_beta
                ),
                f"{MAX_METRIC_PREFIX}offline_opd/sdar_gate_scale": loss.new_tensor(
                    sdar_gate_scale
                ),
            }
        )

    # Preserve the existing sequence-balanced importance-weight aggregation.
    # Each observation below is one sequence's mean token ratio, while the
    # sequence metric is the product ratio accumulated safely in log space.
    local_masks = get_concat_loss_mask(
        batch["total_lengths"], batch["response_lengths"], batch["loss_masks"]
    ).split([x.numel() for x in per_sample_ratio])
    sequence_stats = torch.stack(
        [
            torch.stack(
                (
                    (weights * mask).sum(),
                    mask.sum(),
                    (ratios * mask).sum(),
                )
            )
            for weights, ratios, mask in zip(
                per_sample_ratio, per_sample_ratio_log, local_masks, strict=True
            )
        ]
    ).detach()
    if mpu.get_context_parallel_world_size() > 1:
        dist.all_reduce(sequence_stats, group=mpu.get_context_parallel_group())
    valid = sequence_stats[:, 1] > 0
    token_mean = sequence_stats[valid, 0] / sequence_stats[valid, 1]
    sequence_weight = torch.exp(
        sequence_stats[valid, 2].clamp(
            -OFFPOLICY_SAFETY_BOUND, OFFPOLICY_SAFETY_BOUND
        )
    )
    count = valid.sum().to(dtype=logits.dtype)
    log["offline_importance_weight_sum"] = token_mean.sum()
    log["offline_importance_weight_sqsum"] = token_mean.square().sum()
    log["offline_importance_weight_count"] = count
    log["offline_sequence_importance_weight_sum"] = sequence_weight.sum()
    log["offline_sequence_importance_weight_sqsum"] = sequence_weight.square().sum()
    log["offline_sequence_importance_weight_count"] = count.clone()
    return loss, log


def _opd_cached_teacher_bidirectional_topk_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted Teacher-owned forward-KL plus current-Student reverse-KL."""
    return _opd_dynamic_cached_teacher_topk_loss(
        args,
        batch,
        logits,
        sum_of_sample_mean,
        include_forward=True,
    )


def _opd_cross_tokenizer_anchored_bidirectional_topk_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Teacher bidirectional KL on exact rows plus rollout-Student anchor KL."""
    return _opd_dynamic_cached_teacher_topk_loss(
        args,
        batch,
        logits,
        sum_of_sample_mean,
        include_forward=True,
        include_student_anchor=True,
    )


def _opd_cross_tokenizer_anchored_reverse_topk_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Current-Student reverse-KL plus rollout-Student KL on unaligned rows."""
    return _opd_dynamic_cached_teacher_topk_loss(
        args,
        batch,
        logits,
        sum_of_sample_mean,
        include_forward=False,
        include_student_anchor=True,
    )


def _opd_cross_tokenizer_let_me_forward_else_reverse_topk_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Forward-KL on exact ``Let/let me X`` triples, reverse-KL elsewhere.

    Cross-tokenizer positions that cannot be aligned exactly remain supervised
    by the rollout-Student sparse+tail anchor, matching the isolated canonical
    template artifact schema.
    """
    return _opd_dynamic_cached_teacher_topk_loss(
        args,
        batch,
        logits,
        sum_of_sample_mean,
        include_forward=True,
        include_student_anchor=True,
        let_me_forward_else_reverse=True,
    )


def opd_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pure on-policy distillation (OPD) loss, with no policy-gradient, entropy or
    reference-KL terms. Wire it in with ``--loss-type custom_loss
    --custom-loss-function-path slime.backends.megatron_utils.loss.opd_loss_function``.

    ``--opd-loss-type`` selects the estimator: the two ``topk_*`` modes go to
    ``_opd_topk_kl_loss``, the default ``token_reverse_kl`` is the token-level
    (student - teacher) reverse KL over response tokens computed below.

    Args:
        args: Configuration containing `opd_loss_type`.
        batch: Mini-batch with "unconcat_tokens", "response_lengths", "total_lengths",
            and the teacher fields the selected mode reads.
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)`; `metrics` holds detached "loss" plus the mode's KL
        term ("opd_loss", or "opd_forward_kl"/"opd_reverse_kl" for the topk modes, which
        also add "opd_topk_overlap" when the teacher sent that diagnostic column).
    """
    opd_loss_type = getattr(args, "opd_loss_type", "token_reverse_kl")
    if opd_loss_type == "cached_teacher_topk_forward_kl":
        return _opd_cached_teacher_topk_forward_loss(
            args, batch, logits, sum_of_sample_mean
        )
    if opd_loss_type == "cached_teacher_dynamic_topk_reverse_kl":
        return _opd_dynamic_cached_teacher_topk_loss(
            args, batch, logits, sum_of_sample_mean
        )
    if opd_loss_type == "cached_teacher_bidirectional_topk_kl":
        return _opd_cached_teacher_bidirectional_topk_loss(
            args, batch, logits, sum_of_sample_mean
        )
    if opd_loss_type == "cached_teacher_bidirectional_topk_kl_with_student_anchor":
        return _opd_cross_tokenizer_anchored_bidirectional_topk_loss(
            args, batch, logits, sum_of_sample_mean
        )
    if opd_loss_type == "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor":
        return _opd_cross_tokenizer_anchored_reverse_topk_loss(
            args, batch, logits, sum_of_sample_mean
        )
    if opd_loss_type == "cached_teacher_sampled_token_pg_with_student_anchor":
        return _opd_cached_teacher_sampled_token_pg_loss(
            args, batch, logits, sum_of_sample_mean
        )
    if opd_loss_type == "chunk_teacher_sampled_token_pg_with_student_anchor":
        return _opd_chunk_teacher_sampled_token_pg_loss(
            args, batch, logits, sum_of_sample_mean
        )
    if opd_loss_type == "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor":
        return _opd_cross_tokenizer_let_me_forward_else_reverse_topk_loss(
            args, batch, logits, sum_of_sample_mean
        )
    if opd_loss_type == "topk_forward_kl":
        return _opd_topk_kl_loss(args, batch, logits, sum_of_sample_mean, direction="forward")
    if opd_loss_type == "topk_reverse_kl":
        return _opd_topk_kl_loss(args, batch, logits, sum_of_sample_mean, direction="reverse")

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        **get_rollout_top_p_logprob_kwargs(args, batch),
    )
    log_probs = torch.cat(log_probs_and_entropy["log_probs"], dim=0)

    teacher_log_probs = batch.get("teacher_log_probs")
    if not teacher_log_probs:
        raise ValueError(
            f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing."
        )
    teacher_log_probs = torch.cat(teacher_log_probs, dim=0).to(device=log_probs.device)

    opd_reverse_kl = log_probs - teacher_log_probs
    opd_loss = sum_of_sample_mean(opd_reverse_kl)
    loss = opd_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    log = {
        "loss": loss.clone().detach(),
        "opd_loss": opd_loss.clone().detach(),
    }

    # Hybrid offline sampled-token OPD may carry the rollout student's cached
    # Top-K support plus the teacher values on that support. They are diagnostics
    # only: run the existing Top-K metric implementation under no_grad, merge its
    # logs, and keep `loss` above as the sole backward objective. This restores
    # train/opd_reverse_kl, mass/overlap and offline_opd/* drift series without
    # changing token_reverse_kl training semantics.
    if batch.get("student_support_ids") and batch.get("teacher_logprobs_on_student"):
        with torch.no_grad():
            _, diagnostic_log = _opd_topk_kl_loss(
                args,
                batch,
                logits,
                sum_of_sample_mean,
                direction="reverse",
            )
        diagnostic_log.pop("loss", None)
        log.update(diagnostic_log)

    return loss, log


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    step_global_batch_size: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        step_global_batch_size: Sample count for the current training step
            (total across DP). Replaces the legacy ``args.global_batch_size``
            fallback so the train side stops depending on "every DP rank holds
            the same N samples".
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        batch["rollout_mask_sums"],
        args.calculate_per_token_loss,
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    if not args.calculate_per_token_loss:
        loss = (
            loss
            * num_microbatches
            / step_global_batch_size
            * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    # Split reported metrics into the additive channel (SUM-then-divide) and the
    # MAX channel (all-reduce MAX, no divide). Keys prefixed with MAX_METRIC_PREFIX
    # go to the latter; the prefix is stripped so the reported name is clean.
    additive_log = {k: v for k, v in log.items() if not k.startswith(MAX_METRIC_PREFIX)}
    max_log = {k[len(MAX_METRIC_PREFIX) :]: v for k, v in log.items() if k.startswith(MAX_METRIC_PREFIX)}

    reduction_dict: dict[str, list[str] | torch.Tensor] = {
        "keys": list(additive_log.keys()),
        # values[0] is the consumer's reporting denominator after
        # all-reduce. For per-token-loss it must equal step total tokens
        # (only known by summing per-mb num_tokens across mbs / DP). For
        # per-rollout-mean it is a constant — ``step_global_batch_size`` —
        # so we leave a 0 placeholder here and let ``train_one_step``
        # substitute the constant directly, instead of routing it through
        # per-mb fractions.
        "values": torch.tensor(
            [
                num_tokens if args.calculate_per_token_loss else 0,
            ]
            + list(additive_log.values()),
            device=logits.device,
        ),
    }
    if max_log:
        reduction_dict["max_keys"] = list(max_log.keys())
        reduction_dict["max_values"] = torch.stack([v.detach() for v in max_log.values()])

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        reduction_dict,
    )
