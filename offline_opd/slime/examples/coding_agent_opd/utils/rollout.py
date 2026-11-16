"""Auxiliary helpers for the coding_agent_opd training rollout.

This module owns the env-driven configuration and the supporting helpers
(backend/transport guards, ds validation, pod boot with retries, sample + abort
bookkeeping, KV-slot release, and small formatting utilities). ``generate.py``
keeps only the core model-in-the-loop orchestration (``_run_token_loop`` +
``generate``) and imports the rest from here.

The eval harness (``eval/eval_chat.py``) reuses ``_boot_env`` and
``_missing_ds_field`` from this module so train and eval share the exact same
pod-boot + ds-validation path.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import secrets

import aiohttp

from slime.utils.types import Sample

# Import the r2egym glue package first: its __init__ runs the sys.path shim that
# puts the vendored r2egym ahead of any editable install, so the bare
# ``r2egym.agenthub...`` imports elsewhere resolve to the vendored tree.
from ..r2egym.env import AsyncRepoEnv
from ..r2egym.scaffold import DEFAULT_MAX_STEPS_ABSOLUTE, load_scaffold

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Env-driven configuration
# --------------------------------------------------------------------------
# Sandbox backend. Only "rle" is implemented; the E2B path is intentionally
# left unimplemented for this recipe (grading + tool loop are RLE-only).
_SANDBOX_BACKEND = os.environ.get("SLIME_AGENT_SANDBOX_BACKEND", "rle").strip().lower()

# The shared scaffold (prompt / tools / parser / budget). Loaded once: the
# training loop, the eval token-in loop and the eval chat loop all read prompts
# and budgets from this single source so train and eval stay byte-identical.
# The hard turn cap (``max_steps_absolute``) is the one budget knob and is
# overridable via ``SLIME_AGENT_MAX_STEPS`` (default = R2E-Gym's Agent.run 50)
# so a run can widen/narrow the training budget without a code change.
_MAX_STEPS_ABSOLUTE = int(
    os.environ.get("SLIME_AGENT_MAX_STEPS", str(DEFAULT_MAX_STEPS_ABSOLUTE))
)
_SCAFFOLD = load_scaffold(max_steps_absolute=_MAX_STEPS_ABSOLUTE)

# Loop / budget knobs. The turn cap is the scaffold's ``max_steps_absolute``;
# timeouts stay env-overridable for per-image tuning.
_TOOL_TIMEOUT_SEC = int(os.environ.get("SLIME_AGENT_TOOL_TIMEOUT_SEC", "180"))
_EVAL_TIMEOUT_SEC = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "600"))
_BOOT_CONCURRENCY = int(os.environ.get("SWE_BOOT_CONCURRENCY", "16"))
_BOOT_RETRIES = int(os.environ.get("SWE_BOOT_RETRIES", "2"))
_ROLLOUT_GUARD_SEC = int(os.environ.get("SWE_ROLLOUT_GUARD_SEC", "0") or 0) or (
    _SCAFFOLD.max_steps_absolute * _TOOL_TIMEOUT_SEC + _EVAL_TIMEOUT_SEC + 180
)

# ds fields the swe_type's Docker runtime reads; a row missing any grades to 0 /
# raises in setup_env, so we abort the sample up front rather than boot a doomed
# pod. FAIL_TO_PASS / PASS_TO_PASS are spelled uppercase by the canonical
# swebench schema (``make_test_spec`` reads ``ds["FAIL_TO_PASS"]``) but lowercase
# by swebench-pro's ``_calculate_reward``; accept whichever case the row carries.
_REQUIRED_DS_FIELDS = (
    "instance_id",
    "docker_image",
    "test_patch",
)
_REQUIRED_DS_FIELD_PAIRS = (
    ("fail_to_pass", "FAIL_TO_PASS"),
    ("pass_to_pass", "PASS_TO_PASS"),
)
_ALLOW_EMPTY_DS_FIELD_PAIRS = frozenset(_REQUIRED_DS_FIELD_PAIRS)

# scale-swe rows have no test_patch: the graded tests live in ``f2p_patch`` (a diff
# over the repo's own test files) and/or ``f2p_script`` (a standalone pytest file),
# and the repo sits at ``ds["workdir"]`` instead of /testbed. ``ScaleSWEDocker``
# (vendored docker.py) reads exactly these, so the row check follows its schema.
_SCALE_SWE_TYPE = "scale-swe"
_REQUIRED_SCALE_SWE_DS_FIELDS = (
    "instance_id",
    "docker_image",
    "workdir",
)
_REQUIRED_SCALE_SWE_DS_FIELD_PAIRS = (
    ("f2p_patch", "f2p_script"),
    ("fail_to_pass", "FAIL_TO_PASS"),
    ("pass_to_pass", "PASS_TO_PASS"),
)

# YiYan SWE factory images install ``eval_script`` as /run_tests.sh and grade
# its OMNIGRIL_EXIT_CODE. They do not consume SWE-bench's
# FAIL_TO_PASS/PASS_TO_PASS lists. These are the fields read directly by
# YiYanSWEFactoryImagesDocker.setup_env (``repo`` is used by its git cleanup).
_YIYAN_FACTORY_TYPE = "yiyan-sweimages-factory"
_REQUIRED_YIYAN_FACTORY_DS_FIELDS = (
    "instance_id",
    "docker_image",
    "repo",
    "base_commit",
    "patch",
    "test_patch",
    "eval_script",
    "language",
)

_BOOT_SEM = asyncio.Semaphore(_BOOT_CONCURRENCY)


class _LoopAbort(RuntimeError):
    """Internal signal: the token loop hit an unrecoverable generation state."""


# --------------------------------------------------------------------------
# Backend / transport guards
# --------------------------------------------------------------------------
def _require_rle_backend() -> None:
    if _SANDBOX_BACKEND != "rle":
        raise NotImplementedError(
            f"SLIME_AGENT_SANDBOX_BACKEND={_SANDBOX_BACKEND!r} is not supported by "
            "coding_agent_opd; only the RLE backend is implemented for the "
            "model-in-the-loop SWE + OPD recipe."
        )


def _require_fn_calling() -> None:
    """Training supports only the fn-calling transport.

    The token-in training loop appends observations as ``tool``-role messages and
    passes ``tools=`` to the chat template (Qwen-native ``<tool_call>`` protocol).
    The non-fn-calling variant (inline-XML ``<function=...>`` + bare user-role
    observations) has a different history/think-retention distribution and is NOT
    validated for training, so we hard-fail rather than silently train on a
    mismatched harness. Eval keeps both variants; this guard is training-only,
    which is why ``generate()`` calls it and not at module import (eval_chat.py
    imports helpers from this module under either variant).
    """
    if not _SCAFFOLD.fn_calling:
        raise RuntimeError(
            "coding_agent_opd training supports only the fn-calling scaffold, but "
            f"SLIME_AGENT_SCAFFOLD_VARIANT={_SCAFFOLD.variant!r} is non-fn-calling. "
            "Set SLIME_AGENT_SCAFFOLD_VARIANT=edit_fn_calling for training "
            "(edit_non_fn_calling is eval-only)."
        )


# --------------------------------------------------------------------------
# Dataset validation
# --------------------------------------------------------------------------
def _missing_ds_field(ds: dict) -> str | None:
    """Return the name of the first required ds field that is absent/empty."""
    swe_type = ds.get("swe_type")
    if swe_type == _SCALE_SWE_TYPE:
        fields = _REQUIRED_SCALE_SWE_DS_FIELDS
        pairs = _REQUIRED_SCALE_SWE_DS_FIELD_PAIRS
    elif swe_type == _YIYAN_FACTORY_TYPE:
        fields = _REQUIRED_YIYAN_FACTORY_DS_FIELDS
        pairs = ()
    else:
        fields = _REQUIRED_DS_FIELDS
        pairs = _REQUIRED_DS_FIELD_PAIRS
    for f in fields:
        if not ds.get(f):
            return f
    for first, second in pairs:
        if (first, second) in _ALLOW_EMPTY_DS_FIELD_PAIRS:
            # An empty test list is valid SWE-bench data.  The harness explicitly
            # treats a zero-length F2P/P2P group as fully passing, so validation
            # must distinguish an absent value from a present-but-empty list.
            first_present = first in ds and ds[first] is not None and ds[first] != ""
            second_present = second in ds and ds[second] is not None and ds[second] != ""
            if not first_present and not second_present:
                return first
        elif not ds.get(first) and not ds.get(second):
            return first
    return None


# --------------------------------------------------------------------------
# Pod boot
# --------------------------------------------------------------------------
async def _boot_env(ds: dict, *, instance_id: str) -> AsyncRepoEnv:
    """Boot the interactive RepoEnv pod and install the scaffold tools, with retries.

    We must NOT pass ``rle_session_id`` here: the RLE session code is assigned by
    the server inside ``create_image_session`` (docker.py:552-570) and returned to
    us. Supplying a caller-invented id instead takes docker.py's *reuse* branch
    (docker.py:857-862 ``self.container = rle_env_session_id``), which skips pod
    creation entirely -- the first ``session_exec`` then fails with "session not
    found" (code 3007/33605). Leaving it None lets RLE create a fresh pod and pick
    the real session id.

    Transient pod-unavailable / boot failures are retried up to ``_BOOT_RETRIES``
    times. The caller owns teardown (``env.close()`` in a finally). The boot is
    guarded by ``_BOOT_SEM`` so a large rollout does not stampede the RLE pool.
    """
    last_err: Exception | None = None
    for attempt in range(_BOOT_RETRIES):
        env: AsyncRepoEnv | None = None
        try:
            async with _BOOT_SEM:
                env = await AsyncRepoEnv.create(
                    ds,
                    scaffold=_SCAFFOLD,
                    backend="rle",
                    step_timeout=_TOOL_TIMEOUT_SEC,
                    reward_timeout=_EVAL_TIMEOUT_SEC,
                )
                await env.add_tools([str(p) for p in _SCAFFOLD.command_files])
            return env
        except Exception as e:
            last_err = e
            # If create() already returned a live pod but a later step (add_tools)
            # failed, the pod is orphaned unless we close it here before retrying.
            if env is not None:
                try:
                    await env.close()
                except Exception:
                    pass
            logger.warning(
                "[coding_agent_opd] %s: boot attempt %d/%d failed: %s: %s",
                instance_id,
                attempt + 1,
                _BOOT_RETRIES,
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt + random.random())
    assert last_err is not None
    raise last_err


# --------------------------------------------------------------------------
# Context / reward-routing helpers
# --------------------------------------------------------------------------
def _trunc(text: str, limit: int = 20) -> str:
    """Single-line-safe truncation for INFO logs of prompts/observations."""
    text = text or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit].replace('\n', '\\n')}... [truncated {len(text) - limit} chars]"


def _max_context_length(args) -> int:
    if getattr(args, "rollout_max_context_len", None):
        return int(args.rollout_max_context_len)
    return int(args.context_parallel_size) * int(args.max_tokens_per_gpu)


def _needs_sglang_teacher_fetch(args) -> bool:
    """True iff the training path must fetch teacher logprobs via the reward hook.

    Only the sglang OPD teacher rides the per-sample reward hook: leaving
    ``sample.reward`` None routes through ``async_rm -> reward_func`` which POSTs
    to the external teacher server. In megatron mode (or with OPD disabled) the
    teacher logprobs come from an in-backend forward, so we set the scalar reward
    directly and let ``async_rm`` be skipped.
    """
    return getattr(args, "opd_type", None) == "sglang"


def _abort_sglang_request(generate_url: str, rid: str) -> None:
    """Fire-and-forget ``POST /abort_request`` to free a cancelled turn's KV slot.

    Mirrors ``slime.agent.adapters.common.call_sglang_generate``: when the awaiting
    task is cancelled (rollout wall-clock guard, shutdown) the in-flight sglang
    generation would otherwise keep occupying KV until its own length cap. We
    schedule a best-effort short-timeout abort on the same event loop and never
    await it (we are unwinding a CancelledError), swallowing any error.
    """
    abort_url = generate_url.rsplit("/generate", 1)[0] + "/abort_request"

    async def _do_abort() -> None:
        try:
            timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=timeout) as sess:
                await sess.post(abort_url, json={"rid": rid})
        except Exception:  # noqa: BLE001 (best-effort slot release)
            pass

    try:
        asyncio.ensure_future(_do_abort())
    except RuntimeError:
        # No running loop to schedule on (unlikely on this async path); skip.
        pass


# --------------------------------------------------------------------------
# Sample / abort bookkeeping
# --------------------------------------------------------------------------
def _session_id(sample: Sample, instance_id: str) -> str:
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"cagent-opd-{instance_id}-{sample.index}-{sample.group_index}"
    return f"cagent-opd-{instance_id}-{secrets.token_hex(8)}"


def _abort_sample(sample: Sample, reason: str, instance_id: str) -> Sample:
    """Mark ``sample`` aborted in place and return it (single-Sample shape).

    Abort is an unrecoverable harness state, never valid training data. The
    ``drop_aborted_groups`` dynamic sampling filter
    (``--dynamic-sampling-filter-path``) discards any group containing an ABORTED
    sample before it reaches reward post-processing or training -- over-sampling
    then refills the batch with a fresh rollout, and persistent aborts halt the
    run (see ``utils/filters.py``). ``remove_sample`` is kept as a belt-and-suspenders
    mask in case a run is launched without the filter wired.

    The stub must be SELF-CONSISTENT, not merely masked: everything response-shaped is
    either resized to the 1-token stub or cleared outright. Leaving e.g. an [N, K]
    ``student_topk_ids`` next to ``response_length = 1`` produced a stub that only
    survived because ``generate()``'s reset block happens to clean it up on the retry --
    the same latent coupling that made the ``reward`` / ``remove_sample`` leaks possible.
    """
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    # Response-shaped payloads of the discarded attempt. None (not empty) because that is
    # the "absent" encoding every consumer checks; the top-p pair must stay both-None
    # together or Sample._validate_response_metadata_lengths rejects it.
    sample.student_topk_ids = None
    sample.student_topk_logprobs = None
    sample.rollout_student_logprobs_on_support = None
    sample.rollout_top_p_token_ids = None
    sample.rollout_top_p_token_offsets = None
    sample.rollout_routed_experts = None
    # Teacher-derived tensors of the discarded attempt. Nothing downstream validates these
    # against response_length, so a leftover block would silently outlive the stub.
    sample.teacher_log_probs = None
    sample.offline_opd_token_teacher_sparse = False
    sample.teacher_topk_logprobs = None
    sample.teacher_topk_ids = None
    sample.student_support_ids = None
    sample.teacher_logprobs_on_student = None
    sample.offline_opd_teacher_sparse = False
    sample.opd_topk_overlap = None
    # Per-turn accumulators: their counts describe the discarded attempt's turns, which
    # no longer exist on this 1-token stub.
    sample.prefix_cache_info = Sample.PrefixCacheInfo()
    sample.spec_info = Sample.SpecInfo()
    sample.weight_versions = []
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {
        **(sample.metadata or {}),
        "abort_reason": reason,
        "instance_id": instance_id,
    }
    logger.warning("[coding_agent_opd] %s aborted: %s", instance_id, reason)
    return sample
