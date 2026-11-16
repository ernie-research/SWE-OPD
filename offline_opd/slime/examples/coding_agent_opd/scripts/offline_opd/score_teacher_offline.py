#!/usr/bin/env python3
"""OFFLINE OPD — PHASE 2b: bake teacher top-K logprobs into dumped student trajectories.

Reads Phase-1 dumps ``rollout_{id}.pt`` (student trajectories with ``student_topk_ids``
but no teacher fields), queries a standalone sglang teacher (served by serve_teacher.sh)
for the teacher logprobs on the student's per-position top-K support, and writes the
baked samples to ``<out_dir>/rollout_{id}.pt`` for Phase-3 pure-trainer replay.

Reuses slime's existing teacher request and extraction hooks:
  * ``slime.rollout.on_policy_distillation.reward_func`` — POSTs to the teacher and
    returns the raw JSON. Depends only on ``args.rm_url`` + the sample fields
    (tokens/response_length/student_topk_ids/loss_mask); no GenerateState / student engine.
  * ``examples.coding_agent_opd.opd.post_process_rewards`` — under opd_type='sglang' +
    topk_reverse_kl, initially extracts dense ``student_support_ids [R,K]`` and
    ``teacher_logprobs_on_student [R,K]`` from that JSON.

After baking we reset ``sample.reward`` back to the SWE scalar (``metadata['swe_reward']``)
and drop the raw teacher JSON. Before writing, the scorer keeps only trainable teacher rows
``[N,K]`` and drops the duplicate support tensor (the same ids already live in
``student_topk_ids``). Phase 3 expands these rows back to the existing dense loss schema
immediately after loading; the loss and CP implementation remain unchanged.

Usage:
  python3 score_teacher_offline.py \
      --in-dir  /path/to/phase1_dump \
      --out-dir /path/to/phase1_dump/scored \
      --rm-url  http://<teacher_ip>:13141/generate \
      --opd-id-topk 1024 \
      [--concurrency 8] [--shard-index 0 --num-shards 4] [--overwrite]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import torch

# slime imports (run under the slime venv; PYTHONPATH must include the repo root).
from slime.utils.types import Sample
from slime.rollout.on_policy_distillation import reward_func, _decode_b64_matrix
from examples.coding_agent_opd.opd import post_process_rewards
from examples.coding_agent_opd.scripts.offline_opd.offline_utils import (
    compact_baked_teacher_for_disk,
    compact_dynamic_teacher_cache_for_disk,
    compact_token_teacher_for_disk,
    discover_rollout_files,
    load_rollout_file,
    validate_dump_sample,
    validate_cross_tokenizer_anchor_scored_sample,
    validate_dynamic_scored_sample,
    validate_scored_sample,
    validate_token_dump_sample,
    validate_token_scored_sample,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("score_teacher_offline")

_DYNAMIC_ARTIFACT_LOSSES = {
    "cached_teacher_dynamic_topk_reverse_kl",
    "cached_teacher_bidirectional_topk_kl_with_student_anchor",
}
_CROSS_TOKENIZER_ANCHOR_LOSS = (
    "cached_teacher_bidirectional_topk_kl_with_student_anchor"
)


def _dense_teacher_topk() -> bool:
    """Teacher-indexed dense Top-K dump: no Student correspondence is written.

    Phase 2 scores every Teacher-generated position instead of the 1:1-alignable
    subset, and keys the rows by Teacher position.  Mapping onto Student tokens
    becomes a separate offline pass over the artifact.
    """
    return os.environ.get("OFFLINE_OPD_TEACHER_DENSE_TOPK") == "1"


def _write_dense_teacher_topk(sample, reward, topk: int, *, context: str) -> int:
    """Move the Teacher's own Top-K rows onto Teacher-indexed fields.

    Read straight off the teacher response rather than through
    ``post_process_rewards``: its ``[-response_length:]`` trim assumes one row
    per Student response token, while this request has one row per
    Teacher-generated position -- which is the larger of the two whenever a
    trajectory is mostly assistant text.

    ``topk == 0`` is the sampled-token artifact: no Top-K was requested, and only
    the scored sequence is kept (the caller adds ``teacher_token_logprob``).
    """
    positions = sample.__dict__.pop("opd_teacher_dense_positions", None)
    scored_ids = sample.__dict__.pop("opd_teacher_dense_input_ids", None)
    prompt_len = sample.__dict__.pop("opd_teacher_dense_prompt_length", None)
    aligned_positions = sample.__dict__.pop("opd_teacher_dense_aligned_positions", None)
    if positions is None or scored_ids is None or prompt_len is None:
        raise ValueError(
            f"{context}: dense Teacher Top-K requested but reward_func did not stash "
            "the Teacher position grid (canonical template mode is required)"
        )
    # Teacher side of the canonical 1:1 pairs, completion-relative and row for row with
    # opd_teacher_student_scored_positions (used by build_sampled_token_set --aligner
    # canonical). Absent on artifacts written before it was recorded.
    if aligned_positions is not None:
        sample.teacher_aligned_positions = torch.as_tensor(
            list(aligned_positions), dtype=torch.int32
        )
    expected = (int(len(positions)), int(topk))
    if topk > 0:
        meta = (reward or {}).get("meta_info", {})
        vals_b = meta.get("input_top_logprobs_b64")
        ids_b = meta.get("input_top_logprobs_ids_b64")
        if vals_b is None or ids_b is None:
            raise ValueError(
                f"{context}: teacher did not return base64 Top-K blobs; the dense dump "
                "needs return_input_top_logprobs_as_bytes"
            )
        logprobs = torch.from_numpy(_decode_b64_matrix(vals_b).astype("float32", copy=False))
        ids = torch.from_numpy(_decode_b64_matrix(ids_b).astype("int64", copy=False))
        if tuple(ids.shape) != expected or tuple(logprobs.shape) != expected:
            raise ValueError(
                f"{context}: teacher Top-K is {tuple(ids.shape)} / "
                f"{tuple(logprobs.shape)} for {expected} requested rows"
            )
        if not torch.isfinite(logprobs).all():
            raise ValueError(f"{context}: teacher Top-K contains NaN/Inf")
        sample.teacher_dense_topk_ids = ids.to(torch.int32).contiguous()
        sample.teacher_dense_topk_logprobs = logprobs.to(torch.float16).contiguous()
    else:
        sample.teacher_dense_topk_ids = None
        sample.teacher_dense_topk_logprobs = None
    sample.teacher_dense_positions = torch.as_tensor(positions, dtype=torch.int32)
    sample.teacher_input_ids = torch.tensor(scored_ids, dtype=torch.int32)
    sample.teacher_prompt_length = int(prompt_len)
    # The Top-K rows are the artifact; keeping the response-aligned extraction
    # format would duplicate ~100MB per sample.
    sample.teacher_topk_ids = None
    sample.teacher_topk_logprobs = None
    sample.metadata = dict(sample.metadata or {})
    # opd_teacher_dense marks the Teacher-indexed schema; opd_teacher_dense_topk is its
    # width and is 0 for the sampled-token artifact, so it cannot double as the marker.
    sample.metadata["opd_teacher_dense"] = True
    sample.metadata["opd_teacher_dense_topk"] = int(topk)
    sample.metadata["opd_teacher_dense_stats"] = {
        "teacher_tokens": int(len(scored_ids)) - int(prompt_len),
        "teacher_assistant_tokens": expected[0],
        "teacher_prompt_tokens": int(prompt_len),
    }
    return expected[0]


def _validate_dense_teacher_topk(sample, topk: int, *, context: str) -> None:
    """Reject a dense artifact that Phase 3 (or the mapping pass) could misread.

    ``topk == 0`` is the sampled-token artifact and must carry no Top-K at all.
    """
    positions = getattr(sample, "teacher_dense_positions", None)
    ids = getattr(sample, "teacher_dense_topk_ids", None)
    logprobs = getattr(sample, "teacher_dense_topk_logprobs", None)
    teacher_ids = getattr(sample, "teacher_input_ids", None)
    token_logprob = getattr(sample, "teacher_token_logprob", None)
    prompt_len = getattr(sample, "teacher_prompt_length", None)
    required = [
        ("teacher_dense_positions", positions),
        ("teacher_input_ids", teacher_ids),
        ("teacher_token_logprob", token_logprob),
        ("teacher_prompt_length", prompt_len),
    ]
    if topk > 0:
        required += [
            ("teacher_dense_topk_ids", ids),
            ("teacher_dense_topk_logprobs", logprobs),
        ]
    missing = [name for name, value in required if value is None]
    if missing:
        raise ValueError(f"{context}: dense artifact is missing {missing}")
    rows = int(positions.numel())
    if topk > 0:
        if tuple(ids.shape) != (rows, topk) or tuple(logprobs.shape) != (rows, topk):
            raise ValueError(
                f"{context}: dense Top-K is {tuple(ids.shape)} / {tuple(logprobs.shape)} "
                f"for {rows} positions and K={topk}"
            )
    elif ids is not None or logprobs is not None:
        raise ValueError(f"{context}: sampled-token dense artifact (K=0) carries a Top-K")
    if int(teacher_ids.numel()) != int(token_logprob.numel()):
        raise ValueError(
            f"{context}: {int(teacher_ids.numel())} teacher tokens but "
            f"{int(token_logprob.numel())} own-token log-probs"
        )
    response_tokens = int(teacher_ids.numel()) - int(prompt_len)
    if rows == 0 or response_tokens <= 0:
        raise ValueError(f"{context}: dense artifact has no scored teacher response")
    if bool((positions[1:] <= positions[:-1]).any()):
        raise ValueError(f"{context}: dense positions are not strictly ascending")
    if int(positions[0]) < 0 or int(positions[-1]) >= response_tokens:
        raise ValueError(
            f"{context}: dense positions span [{int(positions[0])}, "
            f"{int(positions[-1])}] outside a {response_tokens}-token response"
        )
    aligned = getattr(sample, "teacher_aligned_positions", None)
    if aligned is not None:
        student_rows = getattr(sample, "opd_teacher_student_scored_positions", None)
        if student_rows is None or int(aligned.numel()) != len(student_rows):
            raise ValueError(
                f"{context}: {int(aligned.numel())} teacher aligned positions do not pair "
                "with opd_teacher_student_scored_positions"
            )
        if aligned.numel() and (int(aligned.min()) < 0 or int(aligned.max()) >= response_tokens):
            raise ValueError(f"{context}: teacher aligned positions outside the teacher completion")
    # The Student-indexed cache is a different artifact; mixing the two in one run
    # would let Phase 3 pick up half a distribution.
    stale = [
        name
        for name in (
            "teacher_candidate_ids",
            "teacher_candidate_logprobs",
            "teacher_topk_ids",
            "teacher_topk_logprobs",
            "student_anchor_topk_ids",
        )
        if getattr(sample, name, None) is not None
    ]
    if stale:
        raise ValueError(f"{context}: dense artifact still carries {stale}")


def _mkdir_shared(path: str | Path, *, max_attempts: int = 8) -> Path:
    """Create a directory robustly on an eventually consistent shared FS.

    Some PFS implementations can return EEXIST from concurrent ``mkdir`` even
    when ``exist_ok=True`` and can briefly delay making the new directory
    visible to another client.  Re-checking and retrying makes ordinary shared
    output directories safe without weakening the atomic claim-directory
    protocol used by dynamic sharding below.
    """
    directory = Path(path)
    for attempt in range(1, max_attempts + 1):
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            if directory.is_dir():
                return directory
            if attempt == max_attempts:
                raise
            time.sleep(min(0.05 * (2 ** (attempt - 1)), 1.0))
            continue
        if directory.is_dir():
            return directory
        if attempt == max_attempts:
            raise OSError(f"mkdir returned successfully but directory is not visible: {directory}")
        time.sleep(min(0.05 * (2 ** (attempt - 1)), 1.0))
    raise AssertionError("unreachable")


def _build_args(
    rm_url: str,
    opd_id_topk: int,
    opd_loss_type: str,
    token_id_bridge: str | None = None,
    teacher_template_model: str | None = None,
    teacher_native_reasoning_prefix: bool = False,
    teacher_canonical_template: bool = False,
    teacher_mask_student_format_tokens: bool = False,
    teacher_normalize_student_tool_template: bool = False,
    teacher_mask_student_pre_tool_content: bool = False,
    teacher_reasoning_effort: str | None = None,
) -> SimpleNamespace:
    """Minimal args namespace the two reused hooks read.

    reward_func reads: rm_url, opd_id_topk, opd_loss_type.
    post_process_rewards reads: opd_type, opd_loss_type, opd_id_topk, reward_key.
    """
    request_loss_type = (
        "topk_forward_kl"
        if opd_loss_type in _DYNAMIC_ARTIFACT_LOSSES
        else opd_loss_type
    )
    return SimpleNamespace(
        rm_url=rm_url,
        opd_type="sglang",
        opd_loss_type=request_loss_type,
        artifact_loss_type=opd_loss_type,
        # token_reverse_kl needs only the log-prob of each sampled token. Keeping
        # K at zero prevents reward_func from requesting an unused teacher top-k.
        opd_id_topk=0 if opd_loss_type == "token_reverse_kl" else int(opd_id_topk),
        opd_token_id_bridge=token_id_bridge,
        opd_teacher_template_model=teacher_template_model,
        opd_teacher_native_reasoning_prefix=teacher_native_reasoning_prefix,
        opd_teacher_canonical_template=teacher_canonical_template,
        opd_teacher_mask_student_format_tokens=teacher_mask_student_format_tokens,
        opd_teacher_normalize_student_tool_template=teacher_normalize_student_tool_template,
        opd_teacher_mask_student_pre_tool_content=teacher_mask_student_pre_tool_content,
        opd_teacher_reasoning_effort=teacher_reasoning_effort,
        opd_cross_tokenizer_student_anchor=(
            opd_loss_type == _CROSS_TOKENIZER_ANCHOR_LOSS
        ),
        reward_key=None,
    )


def _swe_scalar(sample: Sample) -> float:
    return float((sample.metadata or {}).get("swe_reward", 0.0))


async def _score_one(args, sample: Sample, sem: asyncio.Semaphore, max_attempts: int) -> None:
    """Fetch one teacher response, retrying transient HTTP failures."""
    async with sem:
        for attempt in range(1, max_attempts + 1):
            try:
                sample.reward = await reward_func(args, sample)
                return
            except Exception as error:  # noqa: BLE001
                if attempt == max_attempts:
                    raise RuntimeError(f"teacher fetch failed after {max_attempts} attempts for sample index={sample.index}") from error
                logger.warning(
                    "teacher fetch failed for sample %s (attempt %d/%d): %r",
                    sample.index,
                    attempt,
                    max_attempts,
                    error,
                )
                await asyncio.sleep(min(2 ** (attempt - 1), 10))


async def _process_file(
    args,
    in_path: str,
    out_path: str,
    concurrency: int,
    max_attempts: int,
    require_student_topk_logprobs: bool,
    sample_output: bool = False,
    token_topk_diagnostics: bool = False,
    teacher_sem: asyncio.Semaphore | None = None,
    rollout_id_override: int | None = None,
) -> None:
    samples, rollout_id = load_rollout_file(in_path)
    if rollout_id_override is not None:
        rollout_id = rollout_id_override
    logger.info("[%s] loaded %d samples (rollout_id=%s)", os.path.basename(in_path), len(samples), rollout_id)

    for offset, sample in enumerate(samples):
        context = f"{os.path.basename(in_path)}:sample[{offset}]"
        artifact_loss_type = getattr(args, "artifact_loss_type", args.opd_loss_type)
        if artifact_loss_type == "token_reverse_kl":
            validate_token_dump_sample(sample, context=context)
            if token_topk_diagnostics:
                validate_dump_sample(
                    sample,
                    args.requested_topk,
                    context=context,
                    require_student_topk_logprobs=True,
                )
        elif artifact_loss_type in _DYNAMIC_ARTIFACT_LOSSES:
            validate_dump_sample(
                sample,
                args.dynamic_support_topk,
                context=context,
                require_student_topk_logprobs=require_student_topk_logprobs,
            )
        else:
            validate_dump_sample(
                sample,
                args.opd_id_topk,
                context=context,
                require_student_topk_logprobs=require_student_topk_logprobs,
            )

    # Dynamic sample-output runs commonly persist one sample per source file.
    # In that layout multiple files are processed concurrently, so all file
    # tasks must share one semaphore; otherwise ``--concurrency`` silently
    # degenerates to one request per node (or is multiplied per file).
    sem = teacher_sem if teacher_sem is not None else asyncio.Semaphore(concurrency)
    await asyncio.gather(*(_score_one(args, sample, sem, max_attempts) for sample in samples))
    logger.info("[%s] teacher-scored %d samples", os.path.basename(in_path), len(samples))

    # Extract the sampled-token teacher log-probs first.  The dense K=0 artifact skips
    # this: with no Top-K, post_process_rewards would take its token-level path and
    # trim the Teacher-grid log-probs to the Student response length, which is
    # meaningless across tokenizers.  teacher_token_logprob below is the real value.
    if _dense_teacher_topk() and args.opd_id_topk == 0:
        for sample in samples:
            sample.teacher_log_probs = None
    else:
        post_process_rewards(args, samples)

    if _dense_teacher_topk():
        from examples.coding_agent_opd.scripts.offline_opd.xvocab_chunk import (
            teacher_token_logprobs,
        )

        dense_rows = 0
        for offset, sample in enumerate(samples):
            context = f"{os.path.basename(in_path)}:sample[{offset}]"
            scored_len = len(sample.opd_teacher_dense_input_ids)
            prompt_len = int(sample.opd_teacher_dense_prompt_length)
            reward = sample.get_reward_value(args)
            # Own-token log-prob for EVERY teacher token, from the same response:
            # sglang returns input_token_logprobs regardless of top_logprobs_num.
            logprobs = teacher_token_logprobs(
                reward,
                teacher_ids_len=scored_len,
                teacher_prompt_len=prompt_len,
            )
            dense_rows += _write_dense_teacher_topk(
                sample, reward, args.opd_id_topk, context=context
            )
            sample.teacher_token_logprob = torch.tensor(logprobs, dtype=torch.float32)
        logger.info(
            "[%s] wrote dense Teacher Top-%d on %d positions across %d samples",
            os.path.basename(in_path),
            args.opd_id_topk,
            dense_rows,
            len(samples),
        )

    token_teacher_logprobs = None
    if token_topk_diagnostics:
        if args.opd_loss_type != "token_reverse_kl":
            raise ValueError("--token-topk-diagnostics requires token_reverse_kl")
        token_teacher_logprobs = [
            torch.as_tensor(s.teacher_log_probs, dtype=torch.float32).clone()
            for s in samples
        ]
        # A second request caches teacher values over the original rollout
        # student's Top-K support. These values are diagnostics only; Phase 3
        # still backpropagates the sampled-token loss above.
        topk_args = _build_args(
            args.rm_url,
            args.requested_topk,
            "topk_reverse_kl",
            getattr(args, "opd_token_id_bridge", None),
            getattr(args, "opd_teacher_template_model", None),
            getattr(args, "opd_teacher_native_reasoning_prefix", False),
            getattr(args, "opd_teacher_canonical_template", False),
            getattr(args, "opd_teacher_mask_student_format_tokens", False),
            getattr(args, "opd_teacher_normalize_student_tool_template", False),
            getattr(args, "opd_teacher_mask_student_pre_tool_content", False),
            getattr(args, "opd_teacher_reasoning_effort", None),
        )
        await asyncio.gather(
            *(_score_one(topk_args, sample, sem, max_attempts) for sample in samples)
        )
        logger.info(
            "[%s] teacher-scored Top-%d diagnostic support on %d samples",
            os.path.basename(in_path),
            args.requested_topk,
            len(samples),
        )
        post_process_rewards(topk_args, samples)
        for sample, token_lp in zip(samples, token_teacher_logprobs, strict=True):
            sample.teacher_log_probs = token_lp

    if not _dense_teacher_topk() and (
        getattr(args, "opd_teacher_native_reasoning_prefix", False)
        or getattr(args, "opd_teacher_canonical_template", False)
        or getattr(args, "opd_teacher_mask_student_format_tokens", False)
    ):
        from examples.coding_agent_opd.scripts.offline_opd.teacher_trajectory import (
            finalize_cross_tokenizer_anchor_artifact,
            finalize_teacher_position_alignment_artifact,
        )

        skipped = 0
        for offset, sample in enumerate(samples):
            finalize = (
                finalize_cross_tokenizer_anchor_artifact
                if getattr(args, "artifact_loss_type", None)
                == _CROSS_TOKENIZER_ANCHOR_LOSS
                else finalize_teacher_position_alignment_artifact
            )
            skipped += finalize(
                sample, context=f"{os.path.basename(in_path)}:sample[{offset}]"
            )
        if getattr(args, "artifact_loss_type", None) == _CROSS_TOKENIZER_ANCHOR_LOSS:
            alignment_label = "Student-anchor positions"
            action = "partitioned"
        elif getattr(args, "opd_teacher_canonical_template", False):
            alignment_label = "unaligned Student tokens"
            action = "masked"
        elif getattr(args, "opd_teacher_mask_student_format_tokens", False):
            alignment_label = "Student template/format tokens"
            action = "masked"
        else:
            alignment_label = "Student-only reasoning-prefix tokens"
            action = "masked"
        logger.info(
            "[%s] %s %d %s across %d samples",
            os.path.basename(in_path),
            action,
            skipped,
            alignment_label,
            len(samples),
        )

    if os.environ.get("OFFLINE_OPD_CHUNK_ARTIFACT") == "1":
        # Chunk artifact from the same scoring pass: own-token log-probs come from
        # input_token_logprobs, which sglang returns regardless of top_logprobs_num.
        from examples.coding_agent_opd.scripts.offline_opd.xvocab_chunk import (
            build_chunk_map,
            chunk_logprobs,
            make_aligner,
            teacher_token_logprobs,
        )
        from examples.coding_agent_opd.r2egym.scaffold import load_scaffold
        from examples.coding_agent_opd.scripts.offline_opd.teacher_trajectory import _load_tokenizer

        assets = os.environ["XVOCAB_ASSETS"]
        # chunk_logprobs only fills teacher_chunk_mask for the kinds this mode accepts, and
        # Phase 3's --opd-chunk-tier can only narrow it. "chunk" is the superset, so every
        # tier (exact / struct / chunk) stays selectable at train time.
        mode = os.environ.get("OFFLINE_OPD_CHUNK_MODE", "chunk")
        effort = getattr(args, "opd_teacher_reasoning_effort", None)
        aligner = make_aligner(assets, effort)
        scaffold = load_scaffold()
        teacher_tokenizer = _load_tokenizer(getattr(args, "opd_teacher_template_model", None))
        chunk_failures = 0
        for offset, sample in enumerate(samples):
            context = f"{os.path.basename(in_path)}:sample[{offset}]"
            scored_ids = sample.__dict__.pop("opd_chunk_scored_input_ids", None)
            scored_prompt_len = sample.__dict__.pop("opd_chunk_scored_prompt_length", None)
            if scored_ids is None or scored_prompt_len is None:
                raise ValueError(
                    f"{context}: chunk artifact requested but reward_func did not stash the "
                    "scored teacher sequence (canonical template mode is required)"
                )
            chunk_map = build_chunk_map(
                aligner,
                tokens=list(sample.tokens),
                loss_mask=list(sample.loss_mask),
                teacher_ids=scored_ids,
                teacher_prompt_len=int(scored_prompt_len),
                scaffold=scaffold,
                teacher_tokenizer=teacher_tokenizer,
                reasoning_effort=effort,
            )
            if chunk_map.teacher_completion_start < int(scored_prompt_len):
                raise ValueError(
                    f"{context}: first chunked teacher position "
                    f"{chunk_map.teacher_completion_start} is inside the prompt "
                    f"(length {scored_prompt_len})"
                )
            logprobs = teacher_token_logprobs(
                sample.get_reward_value(args),
                teacher_ids_len=len(chunk_map.teacher_ids),
                teacher_prompt_len=int(scored_prompt_len),
            )
            share, mask, kind = chunk_logprobs(chunk_map, logprobs, mode)
            if len(share) != sample.response_length:
                raise ValueError(
                    f"{context}: chunk artifact is {len(share)} rows for a "
                    f"{sample.response_length}-token response"
                )
            sample.teacher_chunk_logprob = torch.tensor(share, dtype=torch.float32)
            sample.teacher_chunk_mask = torch.tensor(mask, dtype=torch.uint8)
            sample.teacher_chunk_kind = torch.tensor(kind, dtype=torch.uint8)
            # Keep the scored sequence and its per-token log-probs (~170KB/sample), so an
            # alignment change is a CPU pass instead of a rescore.
            sample.teacher_input_ids = torch.tensor(scored_ids, dtype=torch.int32)
            sample.teacher_token_logprob = torch.tensor(logprobs, dtype=torch.float32)
            sample.teacher_prompt_length = int(scored_prompt_len)
            sample.metadata = dict(sample.metadata or {})
            sample.metadata["opd_chunk_mode"] = mode
            sample.metadata["opd_chunk_stats"] = dict(chunk_map.stats)
            sample.metadata["opd_chunk_failed_turns"] = len(chunk_map.failed_turns)
            chunk_failures += len(chunk_map.failed_turns)
        logger.info(
            "[%s] wrote chunk artifacts (mode=%s) for %d samples, %d unalignable turns",
            os.path.basename(in_path),
            mode,
            len(samples),
            chunk_failures,
        )

    # Keep fp32 teacher values, but persist only TRAINABLE rows. student_topk_ids is
    # already the identical sparse support, so writing dense student_support_ids would
    # duplicate it. Phase 3 reconstructs both dense tensors immediately after loading.
    for offset, s in enumerate(samples):
        context = f"{os.path.basename(in_path)}:sample[{offset}]"
        s.metadata = dict(s.metadata or {})
        s.metadata["opd_teacher_reasoning_effort"] = getattr(
            args, "opd_teacher_reasoning_effort", None
        )
        artifact_loss_type = getattr(args, "artifact_loss_type", args.opd_loss_type)
        if artifact_loss_type == "token_reverse_kl":
            if token_topk_diagnostics:
                if s.teacher_logprobs_on_student is not None:
                    s.teacher_logprobs_on_student = torch.as_tensor(
                        s.teacher_logprobs_on_student, dtype=torch.float32
                    )
                compact_baked_teacher_for_disk(s, args.requested_topk, context=context)
            compact_token_teacher_for_disk(
                s,
                context=context,
                keep_topk_diagnostics=token_topk_diagnostics,
            )
        elif artifact_loss_type in _DYNAMIC_ARTIFACT_LOSSES:
            if not _dense_teacher_topk():
                compact_dynamic_teacher_cache_for_disk(
                    s,
                    args.opd_id_topk,
                    args.dynamic_support_topk,
                    context=context,
                )
            if getattr(args, "opd_token_id_bridge", None):
                from slime.utils.token_id_bridge import load_token_id_bridge

                bridge = load_token_id_bridge(args.opd_token_id_bridge)
                s.metadata = dict(s.metadata or {})
                s.metadata.update(
                    {
                        "opd_token_id_bridge_sha256": bridge.mapping_sha256,
                        "opd_student_tokenizer_size": bridge.student_tokenizer_size,
                        "opd_teacher_tokenizer_size": bridge.teacher_tokenizer_size,
                    }
                )
        else:
            if s.teacher_logprobs_on_student is not None:
                s.teacher_logprobs_on_student = torch.as_tensor(
                    s.teacher_logprobs_on_student, dtype=torch.float32
                )
            compact_baked_teacher_for_disk(s, args.opd_id_topk, context=context)

    # Reset reward to the SWE scalar and drop the raw teacher JSON (huge) before re-saving.
    for s in samples:
        s.reward = _swe_scalar(s)

    for offset, sample in enumerate(samples):
        context = f"{os.path.basename(in_path)}:sample[{offset}]"
        artifact_loss_type = getattr(args, "artifact_loss_type", args.opd_loss_type)
        if artifact_loss_type == "token_reverse_kl":
            validate_token_scored_sample(
                sample,
                context=context,
                allow_topk_diagnostics=token_topk_diagnostics,
            )
            if token_topk_diagnostics:
                validate_scored_sample(
                    sample,
                    args.requested_topk,
                    context=context,
                    require_student_topk_logprobs=True,
                    require_sparse_teacher=True,
                )
        elif artifact_loss_type in _DYNAMIC_ARTIFACT_LOSSES:
            if _dense_teacher_topk():
                _validate_dense_teacher_topk(
                    sample, args.opd_id_topk, context=context
                )
                continue
            validator = (
                validate_cross_tokenizer_anchor_scored_sample
                if artifact_loss_type == _CROSS_TOKENIZER_ANCHOR_LOSS
                else validate_dynamic_scored_sample
            )
            validator(
                sample, args.opd_id_topk, args.dynamic_support_topk, context=context
            )
        else:
            validate_scored_sample(
                sample,
                args.opd_id_topk,
                context=context,
                require_student_topk_logprobs=require_student_topk_logprobs,
                require_sparse_teacher=True,
            )
    logger.info("[%s] validated baked teacher tensors on all %d samples", os.path.basename(in_path), len(samples))

    _mkdir_shared(os.path.dirname(out_path))
    if sample_output:
        sample_dir = Path(os.path.dirname(out_path)) / "samples"
        source_dir = Path(os.path.dirname(out_path)) / "source_manifests"
        _mkdir_shared(sample_dir)
        _mkdir_shared(source_dir)
        manifest_rows = []
        for sample in samples:
            if not isinstance(sample.index, int) or sample.index < 0:
                raise ValueError(f"{in_path}: sample has invalid index={sample.index!r}")
            sample_path = sample_dir / f"sample_{sample.index:06d}.pt"
            temporary_path = sample_path.with_name(f".{sample_path.name}.tmp.{os.getpid()}")
            try:
                torch.save(
                    {
                        "format": "offline_opd_scored_sample_v1",
                        "index": sample.index,
                        "sample": sample.to_dict(),
                    },
                    temporary_path,
                )
                os.replace(temporary_path, sample_path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()
            manifest_rows.append(
                {
                    "index": sample.index,
                    "path": str(sample_path.relative_to(sample_dir.parent)),
                    "trainable_tokens": max(sum(bool(v) for v in sample.loss_mask), 1),
                }
            )

        source_path = source_dir / f"rollout_{rollout_id}.json"
        temporary_source = source_path.with_name(f".{source_path.name}.tmp.{os.getpid()}")
        temporary_source.write_text(
            json.dumps(
                {
                    "format": "offline_opd_scored_source_v1",
                    "source_rollout_id": rollout_id,
                    "opd_teacher_native_reasoning_prefix": bool(
                        getattr(args, "opd_teacher_native_reasoning_prefix", False)
                    ),
                    "opd_teacher_canonical_template": bool(
                        getattr(args, "opd_teacher_canonical_template", False)
                    ),
                    "opd_teacher_template_student_tokens": bool(
                        getattr(args, "opd_teacher_mask_student_format_tokens", False)
                    ),
                    "opd_teacher_normalize_student_tool_template": bool(
                        getattr(args, "opd_teacher_normalize_student_tool_template", False)
                    ),
                    "opd_teacher_mask_student_pre_tool_content": bool(
                        getattr(args, "opd_teacher_mask_student_pre_tool_content", False)
                    ),
                    "opd_teacher_reasoning_effort": getattr(
                        args, "opd_teacher_reasoning_effort", None
                    ),
                    "offline_opd_chunk_artifact": os.environ.get(
                        "OFFLINE_OPD_CHUNK_ARTIFACT"
                    ) == "1",
                    "offline_opd_chunk_raw_teacher": os.environ.get(
                        "OFFLINE_OPD_CHUNK_ARTIFACT"
                    ) == "1",
                    "offline_opd_teacher_dense_topk": _dense_teacher_topk(),
                    "offline_opd_cross_tokenizer_anchor_sparse": bool(
                        getattr(args, "artifact_loss_type", None)
                        == _CROSS_TOKENIZER_ANCHOR_LOSS
                    ),
                    "samples": manifest_rows,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_source, source_path)
        logger.info(
            "[%s] -> %d per-sample artifacts + %s",
            os.path.basename(in_path),
            len(samples),
            source_path,
        )
    else:
        out_blob = dict(rollout_id=rollout_id, samples=[s.to_dict() for s in samples])
        temporary_path = f"{out_path}.tmp.{os.getpid()}"
        try:
            torch.save(out_blob, temporary_path)
            os.replace(temporary_path, out_path)
        finally:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
        logger.info("[%s] -> %s", os.path.basename(in_path), out_path)


def _already_scored(
    out_path: str,
    k: int,
    require_student_topk_logprobs: bool,
    opd_loss_type: str,
    token_topk_diagnostics: bool = False,
    dynamic_support_topk: int = 128,
    teacher_native_reasoning_prefix: bool = False,
    teacher_canonical_template: bool = False,
    teacher_mask_student_format_tokens: bool = False,
    teacher_normalize_student_tool_template: bool = False,
    teacher_mask_student_pre_tool_content: bool = False,
    teacher_reasoning_effort: str | None = None,
) -> bool:
    if not os.path.exists(out_path):
        return False
    try:
        samples, _ = load_rollout_file(out_path)
        for offset, sample in enumerate(samples):
            context = f"{os.path.basename(out_path)}:sample[{offset}]"
            artifact_is_native = bool(
                (sample.metadata or {}).get("opd_teacher_native_reasoning_prefix", False)
            )
            if artifact_is_native != teacher_native_reasoning_prefix:
                return False
            artifact_is_canonical = bool(
                (sample.metadata or {}).get("opd_teacher_canonical_template", False)
            )
            if artifact_is_canonical != teacher_canonical_template:
                return False
            artifact_is_student_tokens = bool(
                (sample.metadata or {}).get(
                    "opd_teacher_template_student_tokens", False
                )
            )
            if artifact_is_student_tokens != teacher_mask_student_format_tokens:
                return False
            metadata = sample.metadata or {}
            if metadata.get("opd_teacher_reasoning_effort") != teacher_reasoning_effort:
                return False
            if bool(metadata.get("opd_teacher_normalize_student_tool_template", False)) != bool(
                teacher_normalize_student_tool_template
            ):
                return False
            if bool(metadata.get("opd_teacher_mask_student_pre_tool_content", False)) != bool(
                teacher_mask_student_pre_tool_content
            ):
                return False
            # A run must not mix samples scored with and without the chunk artifact:
            # Phase 3 would silently train on a partially populated field.
            if bool(metadata.get("opd_chunk_mode")) != (
                os.environ.get("OFFLINE_OPD_CHUNK_ARTIFACT") == "1"
            ):
                return False
            # Same rule for the Teacher-indexed dense dump: it has no Student
            # correspondence, so it must never be mixed into a Student-indexed run.
            # opd_teacher_dense_topk is 0 for the sampled-token artifact, so it only
            # stands in for the marker on artifacts written before opd_teacher_dense.
            artifact_is_dense = bool(
                metadata.get("opd_teacher_dense", metadata.get("opd_teacher_dense_topk"))
            )
            if artifact_is_dense != _dense_teacher_topk():
                return False
            if opd_loss_type == "token_reverse_kl":
                validate_token_scored_sample(
                    sample,
                    context=context,
                    allow_topk_diagnostics=token_topk_diagnostics,
                )
                if token_topk_diagnostics:
                    validate_scored_sample(
                        sample,
                        k,
                        context=context,
                        require_student_topk_logprobs=True,
                        require_sparse_teacher=True,
                    )
            elif opd_loss_type in _DYNAMIC_ARTIFACT_LOSSES:
                validator = (
                    validate_cross_tokenizer_anchor_scored_sample
                    if opd_loss_type == _CROSS_TOKENIZER_ANCHOR_LOSS
                    else validate_dynamic_scored_sample
                )
                validator(sample, k, dynamic_support_topk, context=context)
            else:
                validate_scored_sample(
                    sample,
                    k,
                    context=context,
                    require_student_topk_logprobs=require_student_topk_logprobs,
                    require_sparse_teacher=True,
                )
        return True
    except Exception:  # noqa: BLE001
        return False


def _already_scored_as_samples(
    out_dir: str,
    rollout_id: str,
    opd_loss_type: str = "topk_reverse_kl",
    teacher_native_reasoning_prefix: bool = False,
    teacher_canonical_template: bool = False,
    teacher_mask_student_format_tokens: bool = False,
    teacher_normalize_student_tool_template: bool = False,
    teacher_mask_student_pre_tool_content: bool = False,
    teacher_reasoning_effort: str | None = None,
) -> bool:
    """Cheap resume check; the source manifest is published only after all samples."""
    source_path = Path(out_dir) / "source_manifests" / f"rollout_{rollout_id}.json"
    try:
        value = json.loads(source_path.read_text(encoding="utf-8"))
        if value.get("format") != "offline_opd_scored_source_v1":
            return False
        if bool(value.get("opd_teacher_native_reasoning_prefix", False)) != bool(
            teacher_native_reasoning_prefix
        ):
            return False
        if bool(value.get("opd_teacher_canonical_template", False)) != bool(
            teacher_canonical_template
        ):
            return False
        if bool(value.get("opd_teacher_template_student_tokens", False)) != bool(
            teacher_mask_student_format_tokens
        ):
            return False
        if bool(value.get("opd_teacher_normalize_student_tool_template", False)) != bool(
            teacher_normalize_student_tool_template
        ):
            return False
        if bool(value.get("opd_teacher_mask_student_pre_tool_content", False)) != bool(
            teacher_mask_student_pre_tool_content
        ):
            return False
        if value.get("opd_teacher_reasoning_effort") != teacher_reasoning_effort:
            return False
        if bool(value.get("offline_opd_cross_tokenizer_anchor_sparse", False)) != (
            opd_loss_type == _CROSS_TOKENIZER_ANCHOR_LOSS
        ):
            return False
        # The chunk artifact and the raw Teacher values are written by the same pass, so a
        # manifest produced before either existed must not be treated as complete.
        want_chunk = os.environ.get("OFFLINE_OPD_CHUNK_ARTIFACT") == "1"
        if bool(value.get("offline_opd_chunk_artifact", False)) != want_chunk:
            return False
        if want_chunk and not bool(value.get("offline_opd_chunk_raw_teacher", False)):
            return False
        if bool(value.get("offline_opd_teacher_dense_topk", False)) != _dense_teacher_topk():
            return False
        rows = value.get("samples")
        if not isinstance(rows, list) or not rows:
            return False
        for row in rows:
            relative = row.get("path") if isinstance(row, dict) else None
            if not isinstance(relative, str):
                return False
            path = Path(out_dir) / relative
            if not path.is_file() or path.stat().st_size <= 0:
                return False
        return True
    except (OSError, ValueError, json.JSONDecodeError):
        return False


async def _main_async(a) -> None:
    # Keep direct/unit-test callers compatible with the historical minimal
    # namespace; argparse supplies all of these in production.
    defaults = {
        "opd_loss_type": "topk_reverse_kl",
        "opd_teacher_cache_topk": 4096,
        "opd_dynamic_support_topk": 128,
        "opd_token_id_bridge": None,
        "opd_teacher_template_model": None,
        "opd_teacher_native_reasoning_prefix": False,
        "opd_teacher_canonical_template": False,
        "opd_teacher_mask_student_format_tokens": False,
        "opd_teacher_normalize_student_tool_template": False,
        "opd_teacher_mask_student_pre_tool_content": False,
        "opd_teacher_reasoning_effort": None,
        "diagnostic_topk": None,
        "token_topk_diagnostics": False,
        "sample_output": False,
        "file_concurrency": 1,
        "rollout_id_from_filename": False,
    }
    for name, value in defaults.items():
        if not hasattr(a, name):
            setattr(a, name, value)
    request_topk = (
        a.opd_teacher_cache_topk
        if a.opd_loss_type in _DYNAMIC_ARTIFACT_LOSSES
        else a.opd_id_topk
    )
    args = _build_args(
        a.rm_url,
        request_topk,
        a.opd_loss_type,
        a.opd_token_id_bridge,
        a.opd_teacher_template_model,
        a.opd_teacher_native_reasoning_prefix,
        a.opd_teacher_canonical_template,
        a.opd_teacher_mask_student_format_tokens,
        a.opd_teacher_normalize_student_tool_template,
        a.opd_teacher_mask_student_pre_tool_content,
        a.opd_teacher_reasoning_effort,
    )
    args.dynamic_support_topk = int(a.opd_dynamic_support_topk)
    # Sampled-token OPD intentionally keeps opd_id_topk=0 for its loss request.
    # Hybrid diagnostics, however, reuse a real Top-K snapshot from Phase 1 and
    # therefore need an independent width rather than inheriting that zero.
    args.requested_topk = int(
        a.diagnostic_topk if a.token_topk_diagnostics else a.opd_id_topk
    )
    all_files = discover_rollout_files(a.in_dir)
    teacher_sem = asyncio.Semaphore(a.concurrency)

    async def process_path(in_path: Path) -> None:
        rid = in_path.stem.removeprefix("rollout_")
        out_path = os.path.join(a.out_dir, f"rollout_{rid}.pt")
        already_scored = (
            _already_scored_as_samples(
                a.out_dir,
                rid,
                a.opd_loss_type,
                a.opd_teacher_native_reasoning_prefix,
                a.opd_teacher_canonical_template,
                a.opd_teacher_mask_student_format_tokens,
                a.opd_teacher_normalize_student_tool_template,
                a.opd_teacher_mask_student_pre_tool_content,
                a.opd_teacher_reasoning_effort,
            )
            if a.sample_output
            else _already_scored(
                out_path,
                request_topk,
                a.require_student_topk_logprobs,
                a.opd_loss_type,
                a.token_topk_diagnostics,
                a.opd_dynamic_support_topk,
                a.opd_teacher_native_reasoning_prefix,
                a.opd_teacher_canonical_template,
                a.opd_teacher_mask_student_format_tokens,
                a.opd_teacher_normalize_student_tool_template,
                a.opd_teacher_mask_student_pre_tool_content,
                a.opd_teacher_reasoning_effort,
            )
        )
        if not a.overwrite and already_scored:
            logger.info("skip rollout_%s (already scored at %s)", rid, out_path)
            return
        await _process_file(
            args,
            str(in_path),
            out_path,
            a.concurrency,
            a.max_attempts,
            a.require_student_topk_logprobs,
            a.sample_output,
            a.token_topk_diagnostics,
            teacher_sem,
            int(rid) if a.rollout_id_from_filename else None,
        )

    if not a.dynamic_sharding:
        in_files = [
            path
            for index, path in enumerate(all_files)
            if index % a.num_shards == a.shard_index
        ]
        logger.info(
            "static shard %d/%d owns %d of %d rollout files under %s",
            a.shard_index,
            a.num_shards,
            len(in_files),
            len(all_files),
            a.in_dir,
        )
        for in_path in in_files:
            await process_path(in_path)
        return

    if not a.coord_dir:
        raise ValueError("--coord-dir is required with --dynamic-sharding")
    coord_dir = Path(a.coord_dir)
    _mkdir_shared(coord_dir)
    # Rotate each rank's scan start so simultaneous workers naturally claim
    # different files. Each worker claims only one file at a time, preventing
    # a fast launcher from reserving the entire dataset while still allowing
    # one-sample source files to use the intended teacher concurrency.
    start = a.shard_index % len(all_files)
    scan_order = all_files[start:] + all_files[:start]
    claimed = 0

    def claim_one() -> Path | None:
        nonlocal claimed
        selected: Path | None = None
        for in_path in scan_order:
            rid = in_path.stem.removeprefix("rollout_")
            claim_dir = coord_dir / f"rollout_{rid}.claim"
            try:
                claim_dir.mkdir()
            except FileExistsError:
                continue
            (claim_dir / "owner.txt").write_text(
                f"rank={a.shard_index} host={socket.gethostname()} pid={os.getpid()}\n",
                encoding="utf-8",
            )
            selected = in_path
            claimed += 1
            logger.info(
                "dynamic shard %d/%d claimed %s (claim #%d)",
                a.shard_index,
                a.num_shards,
                in_path.name,
                claimed,
            )
            break
        return selected

    # One bad source sample must not abort mpirun (and the healthy sibling teacher
    # group): quarantine it, but only for data/template errors (ValueError in the cause
    # chain) and up to a cap. A dead teacher is a client/timeout error and still fails.
    quarantine_dir = Path(a.out_dir) / ".quarantine"
    max_quarantine = int(os.environ.get("SCORE_MAX_QUARANTINE", "64"))
    quarantine_lock = asyncio.Lock()
    quarantined = 0

    def _is_source_data_error(error: BaseException) -> bool:
        seen: set[int] = set()
        cur: BaseException | None = error
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            if isinstance(cur, ValueError):
                return True
            cur = cur.__cause__ or cur.__context__
        return False

    async def file_worker() -> None:
        nonlocal quarantined
        while True:
            selected = claim_one()
            if selected is None:
                return
            try:
                await process_path(selected)
            except Exception as error:  # noqa: BLE001
                if not _is_source_data_error(error):
                    raise
                async with quarantine_lock:
                    quarantined += 1
                    if quarantined > max_quarantine:
                        raise
                    _mkdir_shared(quarantine_dir)
                    (quarantine_dir / f"{selected.stem}.txt").write_text(
                        f"{type(error).__name__}: {error}\n", encoding="utf-8"
                    )
                logger.error(
                    "quarantined %s (%d of %d allowed): %s",
                    selected.name,
                    quarantined,
                    max_quarantine,
                    error,
                )

    file_concurrency = min(a.file_concurrency, len(all_files))
    logger.info(
        "dynamic shard %d/%d starts %d file workers with teacher concurrency %d",
        a.shard_index,
        a.num_shards,
        file_concurrency,
        a.concurrency,
    )
    await asyncio.gather(*(file_worker() for _ in range(file_concurrency)))
    logger.info(
        "dynamic shard %d/%d finished after claiming %d of %d files",
        a.shard_index,
        a.num_shards,
        claimed,
        len(all_files),
    )
    if quarantined:
        logger.error(
            "dynamic shard %d/%d quarantined %d files; see %s",
            a.shard_index,
            a.num_shards,
            quarantined,
            quarantine_dir,
        )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", required=True, help="dir with Phase-1 rollout_{id}.pt")
    p.add_argument("--out-dir", required=True, help="dir to write scored rollout_{id}.pt")
    p.add_argument(
        "--rollout-id-from-filename",
        action="store_true",
        help=(
            "use the contiguous rollout_N filename as the output/source-manifest "
            "id instead of the id stored inside the Phase-1 artifact; intended "
            "for read-only symlink subsets"
        ),
    )
    p.add_argument("--rm-url", required=True, help="teacher /generate endpoint")
    p.add_argument("--opd-id-topk", type=int, default=1024, help="K (must match Phase-1 dump)")
    p.add_argument(
        "--opd-loss-type",
        choices=[
            "topk_reverse_kl",
            "token_reverse_kl",
            "cached_teacher_dynamic_topk_reverse_kl",
            "cached_teacher_bidirectional_topk_kl_with_student_anchor",
        ],
        default="topk_reverse_kl",
        help="offline teacher artifact schema; defaults to the existing Top-K path",
    )
    p.add_argument(
        "--opd-teacher-cache-topk",
        type=int,
        default=4096,
        help="teacher Top-K width persisted by the dynamic sparse schema",
    )
    p.add_argument(
        "--opd-dynamic-support-topk",
        type=int,
        default=128,
        help="Phase-1 rollout snapshot width and Phase-3 current support width",
    )
    p.add_argument(
        "--opd-token-id-bridge",
        default=None,
        help="optional student->teacher token-id bridge JSON for cross-tokenizer scoring",
    )
    p.add_argument(
        "--opd-teacher-template-model",
        default=None,
        help=(
            "optional teacher tokenizer/model path; rebuild masked trajectory "
            "connectors with this chat template before sparse teacher scoring"
        ),
    )
    p.add_argument(
        "--opd-teacher-native-reasoning-prefix",
        action="store_true",
        help=(
            "keep the teacher template's generation-time <think> prefix and mask "
            "the corresponding Student-only \\n<think> tokens in the scored artifact"
        ),
    )
    p.add_argument(
        "--opd-teacher-canonical-template",
        action="store_true",
        help=(
            "re-render the complete structured trajectory with the official Teacher "
            "template and score only exact one-to-one Student token alignments"
        ),
    )
    p.add_argument(
        "--opd-teacher-mask-student-format-tokens",
        action="store_true",
        help=(
            "with the native Teacher generation prefix, keep Student semantic "
            "tokens verbatim but exclude Student format tags from Teacher rows/loss"
        ),
    )
    p.add_argument(
        "--opd-teacher-normalize-student-tool-template",
        action="store_true",
        help=(
            "omit GLM-4.5-only structural newlines inside valid tool calls from "
            "the Teacher context and from the Student loss rows"
        ),
    )
    p.add_argument(
        "--opd-teacher-mask-student-pre-tool-content",
        action="store_true",
        help=(
            "exclude Student tokens strictly between </think> and a valid final "
            "<tool_call> from OPD loss while retaining them as context"
        ),
    )
    p.add_argument(
        "--opd-teacher-reasoning-effort",
        choices=["high", "max"],
        default=None,
        help=(
            "optional reasoning_effort passed to the official Teacher chat "
            "template; omitted preserves the template's historical default"
        ),
    )
    p.add_argument("--concurrency", type=int, default=8, help="outer cap on concurrent teacher calls")
    p.add_argument(
        "--file-concurrency",
        type=int,
        default=1,
        help="concurrent source files per dynamic shard (teacher calls still share --concurrency)",
    )
    p.add_argument("--max-attempts", type=int, default=3, help="teacher HTTP attempts per sample")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument(
        "--dynamic-sharding",
        action="store_true",
        help="claim one shared-filesystem work item at a time instead of index modulo sharding",
    )
    p.add_argument(
        "--coord-dir",
        default=None,
        help="run-specific shared claim directory used by --dynamic-sharding",
    )
    p.add_argument("--overwrite", action="store_true", help="re-score even if out file already baked")
    p.add_argument(
        "--sample-output",
        action="store_true",
        help="write one scored artifact per sample plus a lightweight source manifest",
    )
    p.add_argument(
        "--require-student-topk-logprobs",
        action="store_true",
        help="require and preserve the Phase-1 float16 sparse student logprob snapshot",
    )
    p.add_argument(
        "--token-topk-diagnostics",
        action="store_true",
        help=(
            "with token_reverse_kl, also cache teacher values on the Phase-1 "
            "student Top-K support for training-time diagnostics only"
        ),
    )
    p.add_argument(
        "--diagnostic-topk",
        type=int,
        default=None,
        help=(
            "Top-K width of the reused Phase-1 student snapshot when "
            "--token-topk-diagnostics is enabled; independent of the "
            "sampled-token loss's --opd-id-topk=0"
        ),
    )
    a = p.parse_args()
    if a.concurrency <= 0 or a.file_concurrency <= 0 or a.max_attempts <= 0:
        p.error("--concurrency, --file-concurrency, and --max-attempts must be positive")
    if a.num_shards <= 0 or not 0 <= a.shard_index < a.num_shards:
        p.error("require 0 <= --shard-index < --num-shards")
    if a.token_topk_diagnostics and a.opd_loss_type != "token_reverse_kl":
        p.error("--token-topk-diagnostics requires --opd-loss-type token_reverse_kl")
    if a.token_topk_diagnostics and (a.diagnostic_topk is None or a.diagnostic_topk <= 0):
        p.error("--token-topk-diagnostics requires positive --diagnostic-topk")
    if a.opd_loss_type in _DYNAMIC_ARTIFACT_LOSSES:
        if _dense_teacher_topk() and a.opd_teacher_cache_topk == 0:
            # Sampled-token artifact: no teacher Top-K; the dynamic support width is
            # still the Phase-1 Student snapshot the dump is validated against.
            if a.opd_dynamic_support_topk <= 0:
                p.error("dynamic support top-k must be positive")
        elif not 0 < a.opd_dynamic_support_topk <= a.opd_teacher_cache_topk:
            p.error(
                "dynamic support top-k must be positive and no larger than teacher cache "
                "top-k (teacher cache top-k 0 is only valid with "
                "OFFLINE_OPD_TEACHER_DENSE_TOPK=1)"
            )
    if a.opd_loss_type == _CROSS_TOKENIZER_ANCHOR_LOSS:
        if not a.opd_teacher_canonical_template:
            p.error(
                "cross-tokenizer Student anchors require --opd-teacher-canonical-template"
            )
        if not a.require_student_topk_logprobs:
            p.error(
                "cross-tokenizer Student anchors require --require-student-topk-logprobs"
            )
    if a.opd_teacher_template_model and not a.opd_token_id_bridge:
        p.error("--opd-teacher-template-model requires --opd-token-id-bridge")
    if a.opd_teacher_native_reasoning_prefix and not a.opd_teacher_template_model:
        p.error(
            "--opd-teacher-native-reasoning-prefix requires "
            "--opd-teacher-template-model"
        )
    if a.opd_teacher_canonical_template and not a.opd_teacher_template_model:
        p.error(
            "--opd-teacher-canonical-template requires --opd-teacher-template-model"
        )
    if a.opd_teacher_reasoning_effort and not a.opd_teacher_canonical_template:
        p.error(
            "--opd-teacher-reasoning-effort currently requires "
            "--opd-teacher-canonical-template"
        )
    if a.opd_teacher_canonical_template and a.opd_teacher_native_reasoning_prefix:
        p.error(
            "--opd-teacher-canonical-template and "
            "--opd-teacher-native-reasoning-prefix are mutually exclusive"
        )
    if a.opd_teacher_mask_student_format_tokens and not (
        a.opd_teacher_template_model and a.opd_teacher_native_reasoning_prefix
    ):
        p.error(
            "--opd-teacher-mask-student-format-tokens requires both "
            "--opd-teacher-template-model and "
            "--opd-teacher-native-reasoning-prefix"
        )
    if (
        a.opd_teacher_normalize_student_tool_template
        or a.opd_teacher_mask_student_pre_tool_content
    ) and not a.opd_teacher_mask_student_format_tokens:
        p.error(
            "tool-template normalization and pre-tool masking require "
            "--opd-teacher-mask-student-format-tokens"
        )
    asyncio.run(_main_async(a))


if __name__ == "__main__":
    main()
