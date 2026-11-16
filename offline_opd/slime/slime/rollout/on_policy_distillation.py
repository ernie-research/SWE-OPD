import asyncio
import base64
import os

import aiohttp
import numpy as np
import torch

from slime.utils.http_utils import _JSON_CONTENT_TYPE, _json_dumps, _json_loads
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


def _opd_loss_type(args) -> str:
    return getattr(args, "opd_loss_type", "token_reverse_kl")


def _token_id_bridge(args):
    """Return the optional cross-tokenizer bridge (default OPD stays a no-op)."""
    path = getattr(args, "opd_token_id_bridge", None)
    if not path:
        return None
    from slime.utils.token_id_bridge import load_token_id_bridge

    return load_token_id_bridge(path)


# Concurrent teacher scoring calls allowed per event loop. slime's rollout semaphore
# (``sglang_rollout.generate_and_rm``) only wraps the GENERATE call, so without a cap here
# every finished trajectory POSTs its ~R*K*4B base64 body at once: that starves the teacher
# instead of feeding it (scheduler queue at 0, client-side disconnects in its log) and holds
# GBs of body bytes alive. A one-shot-prefill teacher admits one request per forward, so a
# small multiple of its replica count is enough.
_TEACHER_CONCURRENCY = int(os.environ.get("OPD_TEACHER_CONCURRENCY", "4"))
# Per event loop: the fully-async rollout worker drives its own loop on a background
# thread, and an asyncio.Semaphore may only be awaited from the loop it was first used
# on. Populated from inside the loop, so no cross-thread mutation of one entry.
_TEACHER_SEMAPHORES: dict[asyncio.AbstractEventLoop, asyncio.Semaphore] = {}


def _teacher_semaphore() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _TEACHER_SEMAPHORES.get(loop)
    if sem is None:
        sem = _TEACHER_SEMAPHORES[loop] = asyncio.Semaphore(_TEACHER_CONCURRENCY)
    return sem


def build_student_per_position_ids(sample: Sample) -> tuple[np.ndarray, np.ndarray] | None:
    """Student top-k ids for the TRAINABLE response positions, and those positions.

    Returns ``(ids, positions)``: ``ids`` is the [N, K] int32 matrix straight out of
    ``sample.student_topk_ids`` (one row per trainable response token, captured at rollout)
    and ``positions`` is the [N] int32 array of response-relative rows those id-lists belong
    to, normally ``flatnonzero(loss_mask)``. During native-prefix cross-template Phase 2 the
    staged Student-position subset omits structural reasoning-prefix rows from both outputs.
    None if the student top-k was not captured.

    Non-trainable (observation) positions are absent: the loss multiplies them by
    ``loss_mask`` = 0 (see ``get_sum_of_sample_mean``), so scoring them only inflates the
    teacher request/response -- at K=1024 by the bulk of an agent trajectory. The teacher
    skips positions it gets no id-list for and returns one row per position in ``positions``.

    ndarrays, NOT lists: at K=1024 a ``tolist()`` here would materialize N*K Python ints on
    the rollout event loop, and both callers feed the result back into numpy/torch anyway.
    """
    if sample.student_topk_ids is None or len(sample.student_topk_ids) == 0:
        return None
    # asarray keeps legacy list-of-lists samples (e.g. reloaded debug dumps) working.
    ids = np.asarray(sample.student_topk_ids, dtype=np.int32)
    R = sample.response_length
    loss_mask = (
        np.asarray(sample.loss_mask[:R], dtype=bool) if sample.loss_mask is not None else np.ones(R, dtype=bool)
    )
    positions = np.flatnonzero(loss_mask).astype(np.int32)
    rows = np.ascontiguousarray(ids[: len(positions)])
    selected = sample.opd_teacher_student_scored_positions
    if selected is not None:
        selected = np.asarray(selected, dtype=np.int32)
        row_indices = np.searchsorted(positions, selected)
        if (
            selected.ndim != 1
            or selected.size == 0
            or np.any(row_indices >= positions.size)
            or not np.array_equal(positions[row_indices], selected)
        ):
            raise ValueError(
                "native teacher Student positions are not a non-empty loss_mask subset"
            )
        rows = np.ascontiguousarray(rows[row_indices])
        positions = selected
    return rows, positions


async def reward_func(args, sample, **kwargs):
    # Gate held across the payload build too, not just the POST: the base64 blob is the
    # bulk of the memory, so admitting N callers caps live bodies at N instead of one per
    # in-flight trajectory.
    async with _teacher_semaphore():
        bridge = _token_id_bridge(args)
        loss_type = _opd_loss_type(args)
        input_ids = sample.tokens
        teacher_prompt_len = len(sample.tokens) - sample.response_length
        teacher_scored_positions = None
        teacher_dense_positions = None
        sample.opd_teacher_student_scored_positions = None
        sample.opd_teacher_student_anchor_positions = None
        sample.opd_teacher_student_anchor_reasons = None
        sample.opd_teacher_alignment_mode = None
        sample.opd_teacher_alignment_stats = None
        teacher_template_model = getattr(args, "opd_teacher_template_model", None)
        if teacher_template_model:
            if bridge is None:
                raise ValueError(
                    "--opd-teacher-template-model requires --opd-token-id-bridge"
                )
            if loss_type not in {"topk_reverse_kl", "topk_forward_kl"}:
                raise ValueError(
                    "teacher-template trajectory rebuilding currently requires a "
                    "sparse per-position Top-K request"
                )
            from examples.coding_agent_opd.scripts.offline_opd.teacher_trajectory import (
                rebuild_teacher_trajectory,
            )

            rebuilt = rebuild_teacher_trajectory(
                sample,
                bridge,
                teacher_template_model,
                context=f"sample index={sample.index}",
                preserve_teacher_generation_prefix=bool(
                    getattr(args, "opd_teacher_native_reasoning_prefix", False)
                ),
                canonicalize_teacher_template=bool(
                    getattr(args, "opd_teacher_canonical_template", False)
                ),
                mask_student_format_tokens=bool(
                    getattr(args, "opd_teacher_mask_student_format_tokens", False)
                ),
                normalize_student_tool_template=bool(
                    getattr(args, "opd_teacher_normalize_student_tool_template", False)
                ),
                mask_student_pre_tool_content=bool(
                    getattr(args, "opd_teacher_mask_student_pre_tool_content", False)
                ),
                partition_student_anchors=bool(
                    getattr(args, "opd_cross_tokenizer_student_anchor", False)
                ),
                teacher_reasoning_effort=getattr(
                    args, "opd_teacher_reasoning_effort", None
                ),
            )
            input_ids = rebuilt.input_ids
            teacher_prompt_len = rebuilt.prompt_length
            teacher_scored_positions = rebuilt.scored_positions
            if os.environ.get("OFFLINE_OPD_TEACHER_DENSE_TOPK") == "1":
                # Teacher-indexed dense Top-K dump: the artifact is keyed by Teacher
                # position and carries no Student correspondence, so stash the grid
                # and the scored sequence for the writer.
                if rebuilt.teacher_assistant_positions is None:
                    raise ValueError(
                        "dense Teacher Top-K requires the canonical template rebuild"
                    )
                teacher_dense_positions = rebuilt.teacher_assistant_positions
                sample.opd_teacher_dense_positions = teacher_dense_positions
                sample.opd_teacher_dense_input_ids = list(input_ids)
                sample.opd_teacher_dense_prompt_length = int(teacher_prompt_len)
                # Teacher side of the canonical 1:1 pairs (completion-relative), paired
                # row for row with opd_teacher_student_scored_positions below; the
                # canonical chunk aligner turns them into EXACT chunks.
                sample.opd_teacher_dense_aligned_positions = teacher_scored_positions
            if os.environ.get("OFFLINE_OPD_CHUNK_ARTIFACT") == "1":
                # The chunk writer indexes into exactly this scored sequence; stash it
                # for score_teacher_offline, which consumes and drops it.
                sample.opd_chunk_scored_input_ids = list(input_ids)
                sample.opd_chunk_scored_prompt_length = int(teacher_prompt_len)
            if getattr(args, "opd_teacher_canonical_template", False):
                sample.opd_teacher_student_scored_positions = (
                    rebuilt.student_scored_positions
                )
                sample.opd_teacher_alignment_mode = "canonical_template_v1"
                sample.opd_teacher_alignment_stats = rebuilt.alignment_stats
                if getattr(args, "opd_cross_tokenizer_student_anchor", False):
                    sample.opd_teacher_student_anchor_positions = (
                        rebuilt.student_anchor_positions
                    )
                    sample.opd_teacher_student_anchor_reasons = (
                        rebuilt.student_anchor_reasons
                    )
            elif getattr(args, "opd_teacher_mask_student_format_tokens", False):
                sample.opd_teacher_student_scored_positions = (
                    rebuilt.student_scored_positions
                )
                sample.opd_teacher_alignment_mode = (
                    "teacher_template_student_tokens_v1"
                )
                sample.opd_teacher_alignment_stats = rebuilt.alignment_stats
            elif getattr(args, "opd_teacher_native_reasoning_prefix", False):
                sample.opd_teacher_student_scored_positions = (
                    rebuilt.student_scored_positions
                )
                sample.opd_teacher_alignment_mode = "native_reasoning_prefix_v1"
                sample.opd_teacher_alignment_stats = rebuilt.alignment_stats
        elif bridge is not None:
            input_ids = bridge.map_numpy(
                sample.tokens, context="teacher trajectory input_ids"
            ).tolist()
        payload = {
            # "text": sample.prompt + sample.response,
            "input_ids": input_ids,  # flat prompt + response token ids (teacher id space)
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 0,
                "skip_special_tokens": False,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        }

        teacher_topk = int(getattr(args, "opd_id_topk", 0) or 0)

        if loss_type == "topk_reverse_kl" and teacher_topk > 0:
            # Query the teacher on the STUDENT's per-position top-k ids (reverse-KL
            # support). Set logprob_start_len = prompt_len - 1 so the teacher scores the
            # response region: sglang then returns pruned_len = response_length + 1
            # positions (a leading boundary/null position + R real ones). The teacher's
            # distribution at gather row r predicts response token r, so an id-row at
            # ``positions[i]`` scores response token ``positions[i]``; positions we send
            # no id-row for are skipped server-side and absent from the response.
            per_pos = build_student_per_position_ids(sample)
            if per_pos is not None:
                rows, positions = per_pos
                if teacher_scored_positions is not None:
                    if len(teacher_scored_positions) != len(positions):
                        raise ValueError(
                            "teacher-template position count does not match Student Top-K rows"
                        )
                    positions = teacher_scored_positions
                teacher_rows = rows
                if bridge is not None:
                    teacher_rows = bridge.map_numpy(
                        rows, context="teacher reverse-KL query ids"
                    ).astype(np.int32, copy=False)
                payload["logprob_start_len"] = teacher_prompt_len - 1
                # Ship the [N, K] student ids as a base64 int32 blob instead of a ~N*K JSON
                # list: it skips JSON serialize here and JSON parse on the teacher. Only
                # trainable rows are sent (see build_student_per_position_ids).
                payload["token_ids_logprob_b64"] = {
                    "vals": base64.b64encode(teacher_rows.tobytes()).decode("ascii"),
                    "shape": [int(teacher_rows.shape[0]), int(teacher_rows.shape[1])],
                    "dtype": "int32",
                    "positions": positions.tolist(),
                }
                # Response comes back as a base64 raw-float32 blob too (only vals are
                # needed; the ids are the student's own), keeping the ~N*K numbers out of
                # JSON in both directions.
                payload["return_token_ids_logprob_as_bytes"] = True
            elif teacher_scored_positions is not None:
                raise ValueError("teacher-template scoring requires Student Top-K rows")
        elif teacher_topk > 0:
            # topk_forward_kl: ask the teacher for its OWN per-position top-k.
            payload["top_logprobs_num"] = teacher_topk
            if loss_type in {
                "topk_forward_kl",
                "cached_teacher_dynamic_topk_reverse_kl",
                "cached_teacher_bidirectional_topk_kl_with_student_anchor",
            }:
                # Only the response region is consumed (extract_teacher_topk keeps the
                # trailing R rows), so score from prompt_len-1 instead of 0: pruned_len drops
                # from len(tokens) to R+1, cutting the top-k gather, the host copy and the
                # response body by the prompt share of the trajectory. The dynamic cached-
                # teacher loss consumes the same teacher-owned Top-K artifact and therefore
                # must use this path too; otherwise Top-4096 is computed and returned for the
                # complete prompt as a giant JSON response.
                payload["logprob_start_len"] = teacher_prompt_len - 1
                # Logprobs + ids come back as two base64 blobs instead of nested
                # [logprob, token_id, None] triples.
                payload["return_input_top_logprobs_as_bytes"] = True

                # Dynamic offline teacher cache: the loss consumes only trainable
                # response rows.  Carry those positions through SGLang using its
                # compact per-position token-id request (one harmless id per row).
                # A dedicated teacher env flag uses the positions to run Top-K only
                # on these rows; the complete prompt+response remains the attention
                # context.  Standard topk_forward_kl does not enter this branch.
                if (
                    getattr(args, "artifact_loss_type", loss_type)
                    in {
                        "cached_teacher_dynamic_topk_reverse_kl",
                        "cached_teacher_bidirectional_topk_kl_with_student_anchor",
                    }
                ):
                    per_pos = build_student_per_position_ids(sample)
                    if per_pos is not None:
                        rows, positions = per_pos
                        if teacher_dense_positions is not None:
                            # Dense dump: every Teacher-generated position. The carrier
                            # only transports positions, so a constant id is enough.
                            positions = teacher_dense_positions
                            carrier = np.zeros((positions.size, 1), dtype=np.int32)
                        else:
                            if teacher_scored_positions is not None:
                                if len(teacher_scored_positions) != len(positions):
                                    raise ValueError(
                                        "teacher-template position count does not match "
                                        "Student Top-K rows"
                                    )
                                positions = teacher_scored_positions
                            carrier = np.ascontiguousarray(rows[:, :1], dtype=np.int32)
                            if bridge is not None:
                                # The carrier only transports positions (the loss reads
                                # the teacher's own Top-K there), so an unmapped Student
                                # id is replaced by any valid teacher id, not an error.
                                mapped = bridge.map_numpy(
                                    carrier,
                                    context="teacher dynamic-cache carrier ids",
                                    allow_unmapped=True,
                                )
                                carrier = np.ascontiguousarray(
                                    np.where(mapped < 0, 0, mapped), dtype=np.int32
                                )
                        payload["token_ids_logprob_b64"] = {
                            "vals": base64.b64encode(carrier.tobytes()).decode("ascii"),
                            "shape": [int(carrier.shape[0]), 1],
                            "dtype": "int32",
                            "positions": positions.tolist(),
                        }
                        payload["return_token_ids_logprob_as_bytes"] = True
                    elif teacher_scored_positions is not None:
                        raise ValueError(
                            "teacher-template scoring requires Student Top-K rows"
                        )

        if teacher_dense_positions is not None and teacher_topk == 0:
            # Dense dump without Top-K (sampled-token OPD): only the teacher's own-token
            # log-probs are stored, so request no Top-K but keep the completion window --
            # teacher_token_logprobs expects one entry per completion token after the
            # boundary row.
            payload["logprob_start_len"] = teacher_prompt_len - 1

        if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
            image_data = sample.multimodal_inputs["images"]
            payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

        # Teacher scoring is one long one-shot prefill (chunked prefill is off to keep the
        # per-position token_ids_logprob matrix aligned), well past aiohttp's 300s default.
        timeout_sec = float(os.environ.get("OPD_TEACHER_TIMEOUT_SEC", "7200"))
        # A saturated teacher stops accepting connections, so this (not `total`) is what
        # fires as ConnectionTimeoutError and drops the rollout group.
        connect_timeout_sec = float(os.environ.get("OPD_TEACHER_CONNECT_TIMEOUT_SEC", "30"))
        session_kwargs = {"timeout": aiohttp.ClientTimeout(total=timeout_sec, sock_connect=connect_timeout_sec)}
        # orjson instead of aiohttp's stdlib-json helpers: the whole round trip runs ON the
        # single rollout event loop without releasing the GIL, so it stalls every other
        # in-flight trajectory (at ~300MiB bodies json.dumps costs 1257ms vs 155ms, and
        # resp.json() wastes another 167ms decoding to str first). _json_loads keeps a
        # stdlib fallback for the non-b64 teacher, whose masked logprobs come back as
        # non-standard ``-Infinity`` that orjson rejects.
        body = _json_dumps(payload)
        async with aiohttp.ClientSession(**session_kwargs) as session:
            # data= (not json=) because the body is already encoded; without the explicit
            # header aiohttp would label it application/octet-stream.
            async with session.post(args.rm_url, data=body, headers=_JSON_CONTENT_TYPE) as resp:
                resp.raise_for_status()
                return _json_loads(await resp.read())


def _logprob_of(item) -> float:
    """Scalar logprob from one sglang input_token_logprobs entry.

    Each entry is either a bare float or a [logprob, token_id, token_text] tuple,
    depending on the sglang build.
    """
    if isinstance(item, (list, tuple)):
        return float(item[0])
    return float(item)


def _decode_b64_matrix(blob: dict) -> np.ndarray:
    """[M, K] matrix from a ``{"vals","shape","dtype"}`` base64 raw-bytes blob.

    frombuffer is zero-copy but read-only, so copy before handing it to torch.
    """
    arr = np.frombuffer(base64.b64decode(blob["vals"]), dtype=np.dtype(blob["dtype"]))
    return arr.reshape(blob["shape"]).copy()


def _teacher_value_rows(reward: dict) -> torch.Tensor | None:
    """The teacher's per-position value rows for a ``token_ids_logprob`` request.

    One row per SCORED position, in position order; empty (boundary / unscored) rows are
    already dropped, server-side on the b64 path and here on the JSON path. Width is K, or
    K+1 when the teacher runs with ``SGLANG_OPD_TOPK_OVERLAP`` and appends the overlap count
    (see ``extract_teacher_topk_overlap``) -- callers slice what they need. None if the field
    is absent.
    """
    mi = reward.get("meta_info", {})
    # Fast path (return_token_ids_logprob_as_bytes): one base64 float32 blob instead of
    # N*K JSON numbers.
    b = mi.get("input_token_ids_logprobs_b64")
    if b is not None:
        return torch.from_numpy(_decode_b64_matrix(b).astype(np.float32, copy=False))
    raw = mi.get("input_token_ids_logprobs")
    if not raw:
        return None
    return torch.tensor(
        [[_logprob_of(e) for e in pos] for pos in raw if pos], dtype=torch.float32
    )


def extract_teacher_logprobs_on_student_ids(
    reward: dict, response_length: int, k: int, positions: np.ndarray
) -> torch.Tensor | None:
    """Teacher per-position logprobs at the student ids ([R, K]), for topk_reverse_kl.

    From a per-position ``token_ids_logprob`` request: sglang returns one entry per
    scored position, and only the ``positions`` we sent an id-row for are non-empty
    (boundary and observation positions come back null/empty). Those ``len(positions)``
    rows are scattered back to the [R, K] the loss indexes by response position.

    Rows we never scored stay 0.0, and they MUST stay finite: ``sum_of_sample_mean``
    multiplies them by loss_mask = 0, and ``0 * inf`` would be NaN and poison the loss
    instead of dropping the row. None if the field is absent.
    """
    rows = _teacher_value_rows(reward)
    if rows is None:
        return None
    if rows.shape[0] != len(positions):
        raise ValueError(
            f"teacher returned {rows.shape[0]} scored rows for {len(positions)} requested "
            "positions; the per-position token_ids_logprob window is misaligned."
        )
    out = torch.zeros((response_length, k), dtype=torch.float32)
    out[torch.from_numpy(positions).long()] = rows[:, :k]
    return out


def extract_teacher_topk(reward: dict, response_length: int) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Teacher's own per-position top-k over the response region, for topk_forward_kl.

    From ``meta_info["input_top_logprobs"]``: one list per scored input position --
    None for boundary positions (no preceding-token distribution), else
    ``[logprob, token_id, token_text]`` entries.

    Returns ``(topk_logprobs, topk_ids)`` each ``[response_length, K]``, or None if the
    teacher returned no top-k (top_logprobs_num was 0).
    """
    mi = reward.get("meta_info", {})
    # Fast path (return_input_top_logprobs_as_bytes): two base64 blobs instead of ~3*M*K
    # JSON elements. Boundary rows are dropped server-side from both alike.
    vals_b = mi.get("input_top_logprobs_b64")
    ids_b = mi.get("input_top_logprobs_ids_b64")
    if vals_b is not None and ids_b is not None:
        vals = torch.from_numpy(_decode_b64_matrix(vals_b).astype(np.float32, copy=False))
        ids = torch.from_numpy(_decode_b64_matrix(ids_b).astype(np.int64, copy=False))
        return vals[-response_length:], ids[-response_length:]

    top = mi.get("input_top_logprobs")
    if not top:
        return None
    # Boundary (None) rows: BOS at logprob_start_len=0, or the leading position of a
    # narrowed window.
    top = [pos for pos in top if pos]
    if not top:
        return None
    vals = torch.tensor([[_logprob_of(e) for e in pos] for pos in top], dtype=torch.float32)
    ids = torch.tensor([[int(e[1]) for e in pos] for pos in top], dtype=torch.long)
    return vals[-response_length:], ids[-response_length:]


def extract_teacher_topk_overlap(
    reward: dict, response_length: int, k: int, positions: np.ndarray
) -> torch.Tensor | None:
    """Top-k overlap ratio as a [R] tensor: the ratio at ``positions``, 0 elsewhere.

    Needs the teacher launched with ``SGLANG_OPD_TOPK_OVERLAP=1``, which appends ONE
    trailing column to each per-position value row: ``|TopK_teacher ∩ ids|`` at that
    position, i.e. how many of the student's top-k ids the teacher also ranks top-k.
    Dividing by k gives the per-position overlap ratio.

    Returned per-token rather than as a scalar so it reduces through the same
    ``sum_of_sample_mean`` as ``opd_reverse_kl``. None when the teacher did not send the
    column.
    """
    rows = _teacher_value_rows(reward)
    if rows is None or rows.shape[1] <= k:
        return None
    if rows.shape[0] != len(positions):
        raise ValueError(
            f"teacher returned {rows.shape[0]} scored rows for {len(positions)} requested "
            "positions; the per-position token_ids_logprob window is misaligned."
        )
    out = torch.zeros(response_length, dtype=torch.float32)
    out[torch.from_numpy(positions).long()] = rows[:, k] / k
    return out


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Attach the teacher's logprobs from the reward response onto each sample.

    Per --opd-loss-type (with --opd-id-topk > 0 for the topk_* modes):
      * token_reverse_kl: token-level ``sample.teacher_log_probs``, trimmed to the
        response region.
      * topk_forward_kl: ``teacher_topk_logprobs`` / ``teacher_topk_ids``.
      * topk_reverse_kl: ``student_support_ids`` / ``teacher_logprobs_on_student``.

    Returns 0.0 scalar rewards (GRPO/PPO-compatible) for pure distillation; the learning
    signal comes from the OPD loss / KL penalty, so add task rewards here if you have any.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]
    loss_type = _opd_loss_type(args)
    teacher_topk = int(getattr(args, "opd_id_topk", 0) or 0)

    # Clear every teacher-derived field before filling in the ones this loss type uses.
    # The branches below skip a sample whose extraction returns None, so without this a
    # failed fetch (or a rollout that requeues the SAME Sample object -- data_buffer
    # appends the group as-is and pop_first slices without copying) leaves the previous
    # values in place. None of these is checked against response_length by
    # Sample._validate_response_metadata_lengths, so stale tensors of a matching length
    # would reach the loss and pair an old teacher distribution with new student logits.
    for sample in samples:
        sample.teacher_log_probs = None
        sample.offline_opd_token_teacher_sparse = False
        sample.teacher_topk_logprobs = None
        sample.teacher_topk_ids = None
        sample.student_support_ids = None
        sample.teacher_logprobs_on_student = None
        sample.offline_opd_teacher_sparse = False
        sample.rollout_student_logprobs_on_support = None
        sample.opd_topk_overlap = None
        sample.teacher_candidate_positions = None
        sample.teacher_candidate_ids = None
        sample.teacher_candidate_logprobs = None
        sample.teacher_top128_ids = None
        sample.teacher_cutoff_logprobs = None
        sample.teacher_top4096_mass = None
        sample.offline_opd_dynamic_teacher_sparse = False
        sample.student_anchor_positions = None
        sample.student_anchor_topk_ids = None
        sample.student_anchor_topk_logprobs = None
        sample.student_anchor_reason = None
        sample.offline_opd_cross_tokenizer_anchor_sparse = False

    if loss_type == "topk_reverse_kl" and teacher_topk > 0:
        # Teacher was queried on the student's per-position top-k ids at the trainable
        # positions only: scatter both the ids and the teacher logprobs back to [R, K],
        # which is what the loss indexes by response position.
        for sample, reward, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
            per_pos = build_student_per_position_ids(sample)
            if per_pos is None:
                continue
            rows, positions = per_pos
            teacher_lp = extract_teacher_logprobs_on_student_ids(
                reward, response_length, teacher_topk, positions
            )
            if teacher_lp is None:
                continue
            row_idx = torch.from_numpy(positions).long()
            support_ids = torch.zeros((response_length, teacher_topk), dtype=torch.long)
            support_ids[row_idx] = torch.from_numpy(rows).long()
            sample.student_support_ids = support_ids
            sample.teacher_logprobs_on_student = teacher_lp
            sample.opd_topk_overlap = extract_teacher_topk_overlap(
                reward, response_length, teacher_topk, positions
            )
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    if loss_type == "topk_forward_kl" and teacher_topk > 0:
        # Attach the teacher's own top-k and return early: it scored the response region
        # only (reward_func narrows logprob_start_len), so the token-level path below no
        # longer spans the sequence -- and _opd_topk_kl_loss never reads
        # sample.teacher_log_probs.
        for sample, reward, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
            topk = extract_teacher_topk(reward, response_length)
            if topk is not None:
                sample.teacher_topk_logprobs, sample.teacher_topk_ids = topk
        scalar_rewards = [0.0] * len(samples)
        return scalar_rewards, scalar_rewards

    # Extract teacher log-probs from the sglang response (token-level path)
    teacher_log_probs = [
        torch.tensor([_logprob_of(item) for item in reward["meta_info"]["input_token_logprobs"][1:]], dtype=torch.float32)
        for reward in raw_rewards
    ]
    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    if teacher_topk > 0:
        # token_reverse_kl with --opd-id-topk > 0: reward_func still asks for the teacher's
        # own top-k (full window, JSON list); the token-level loss ignores it, kept for
        # diagnostics.
        for sample, reward, response_length in zip(samples, raw_rewards, response_lengths, strict=False):
            topk = extract_teacher_topk(reward, response_length)
            if topk is not None:
                sample.teacher_topk_logprobs, sample.teacher_topk_ids = topk

    scalar_rewards = [0.0] * len(samples)

    return scalar_rewards, scalar_rewards
