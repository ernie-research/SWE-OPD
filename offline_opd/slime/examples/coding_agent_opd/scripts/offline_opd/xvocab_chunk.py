"""Chunk-aligned teacher log-probs for cross-vocabulary sampled-token OPD.

Phase 2 scores the teacher once on the full canonical rendering of a trajectory. Verified
on real rollouts: every turn's ``encode_messages(ctx) + concat(piece.ds)`` is an exact
prefix of that final rendering, so a single request covers every turn and chunks can be
expressed directly in global teacher token positions.

Chunk maps: ``build_chunk_map`` (GLM <-> DSV4 byte chunks), ``build_pair_chunk_map``
(1:1 pairs decided elsewhere, e.g. the GLM-5.2 canonical alignment) and
``build_identity_chunk_map`` (same tokenizer). None needs teacher output, so all are
testable without GPUs; ``chunk_logprobs`` then turns the teacher's own-token log-probs into
the three ``[response_length]`` arrays Phase 3 reads.
"""

from __future__ import annotations

import bisect
import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from glm_ds_align import (
    DS_EOS,
    EXACT,
    GLM_ONLY,
    MASKED,
    MISMATCH,
    MISMATCH,
    MODES,
    SAME,
    STRUCT,
    Piece,
    STRUCT_1TO1,
    STRUCT_N,
    TEXT_MULTI,
    AlignmentError,
    _to_ds_messages,
    parse_glm_completion,
)

# Persisted as uint8 so Phase 3 can switch tier (exact / struct / chunk) without rescoring.
KIND_CODES = {MASKED: 0, EXACT: 1, STRUCT_1TO1: 2, STRUCT_N: 3, TEXT_MULTI: 4}

DS_EOS = "<\uff5cend\u2581of\u2581sentence\uff5c>"
STOP_TOKENS = ("<|observation|>", "<|user|>", "<|endoftext|>")
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\n(.*?)\n*</tool_response>", re.S)


class ChunkRenderMismatch(RuntimeError):
    """Our rendering of the trajectory is not the one Phase 2 actually scored."""


@dataclass
class Chunk:
    student: tuple[int, int]  # [start, end) rows in the response region
    teacher: tuple[int, int]  # [start, end) positions in ChunkMap.teacher_ids
    kind: str


@dataclass
class ChunkMap:
    teacher_ids: list[int]
    teacher_completion_start: int  # first teacher position any chunk may reference
    chunks: list[Chunk]
    response_length: int
    failed_turns: list[tuple[int, str]] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)


def mask_runs(loss_mask):
    """Consecutive equal-value runs of the Phase-1 loss mask: (is_assistant, start, end)."""
    runs, current, start = [], loss_mask[0], 0
    for i, value in enumerate(loss_mask[1:], 1):
        if value != current:
            runs.append((bool(current), start, i))
            current, start = value, i
    runs.append((bool(current), start, len(loss_mask)))
    return runs


def tool_content(text):
    m = _TOOL_RESPONSE_RE.search(text)
    return m.group(1) if m else text


def fallback_message(text):
    """Best-effort assistant message for a turn the parser cannot handle.

    The turn stays unusable for training, but replaying it keeps every later turn of the
    trajectory alignable instead of dropping the rest of the sample.
    """
    wo_eos = True
    for stop in STOP_TOKENS:
        if text.endswith(stop):
            text, wo_eos = text[: -len(stop)], False
            break
    reasoning, open_at = "", text.find("<think>")
    if open_at >= 0:
        rest = text[open_at + len("<think>") :]
        close_at = rest.find("</think>")
        reasoning, text = (rest, "") if close_at < 0 else (rest[:close_at], rest[close_at + len("</think>") :])
    message = {"role": "assistant", "reasoning_content": reasoning, "content": text.strip()}
    if wo_eos:
        message["wo_eos"] = True
    return message


def _student_bytes(aligner, ids):
    return b"".join(aligner.glm.token_bytes(t) for t in ids)


def _decode(aligner, ids):
    return _student_bytes(aligner, ids).decode("utf-8")


def _decode_lenient(aligner, ids):
    """Observations only ever enter the context as text, and a tool can emit raw bytes."""
    return _student_bytes(aligner, ids).decode("utf-8", "replace")


def _turn_chunks(pieces, s_ends, t_ends, t_base):
    """Common byte boundaries -> chunks, in local byte space on both sides.

    ``t_ends`` are global teacher byte ends rebased by ``t_base``; a turn whose teacher
    text does not start on a global token boundary yields no chunks and is masked.
    """
    s_set, t_set = {0, *s_ends}, {0, *t_ends}
    s_sorted = [0, *s_ends]
    points = [(0, 0)]
    g = d = 0
    spans = []
    for p in pieces:
        gl, dl = len(p.glm.encode()), len(p.ds.encode())
        spans.append((g, g + gl, d, d + dl, p.kind))
        if p.kind == SAME:
            lo = bisect.bisect_right(s_sorted, g)
            hi = bisect.bisect_right(s_sorted, g + gl)
            candidates = [(b, d + b - g) for b in s_sorted[lo:hi]]
        else:
            candidates = [(g + gl, d + dl)]
        for sb, tb in candidates:
            if sb in s_set and tb in t_set and (sb, tb) != points[-1] and sb != points[-1][0]:
                points.append((sb, tb))
        g, d = g + gl, d + dl
    return points, spans


def canonical_turn(glm_text, *, scaffold, teacher_tokenizer, reasoning_effort=None, context="turn"):
    """Student turn -> (text without stop token, canonical message, Teacher text).

    Both halves come from teacher_trajectory, so the Teacher text is exactly what Phase 2
    assembles; nothing about either template is reproduced here.
    """
    from examples.coding_agent_opd.scripts.offline_opd.teacher_trajectory import (
        _canonical_assistant_message,
        _template_turn_deltas,
    )

    stripped = glm_text
    for stop in STOP_TOKENS:
        if stripped.endswith(stop):
            stripped = stripped[: -len(stop)]
            break
    message = _canonical_assistant_message(stripped, scaffold, context=context)
    _, ds_text, _, _, _ = _template_turn_deltas(
        teacher_tokenizer, message, None, context=context, reasoning_effort=reasoning_effort
    )
    if ds_text.endswith(DS_EOS):
        ds_text = ds_text[: -len(DS_EOS)]
    return stripped, message, ds_text


def build_chunk_map(aligner, *, tokens, loss_mask, teacher_ids, teacher_prompt_len,
                    scaffold, teacher_tokenizer, reasoning_effort=None):
    """Map every Student response row onto a position range of the scored Teacher sequence.

    ``teacher_ids`` must be the sequence Phase 2 actually scored: each turn's Teacher text
    is located inside it by byte search, so no assumption is made about how Phase 2
    assembles turns. A turn that cannot be parsed or located is left entirely masked and
    costs only itself.
    """
    response_length = len(loss_mask)
    prompt_len = len(tokens) - response_length
    turns, failed = [], []
    for turn, (is_assistant, start, end) in enumerate(mask_runs(loss_mask)):
        if not is_assistant:
            continue
        ids = tokens[prompt_len + start : prompt_len + end]
        try:
            text = _decode(aligner, ids)
        except UnicodeDecodeError:
            failed.append((start, "student turn is not valid utf-8"))
            continue
        try:
            stripped, message, ds_text = canonical_turn(
                text, scaffold=scaffold, teacher_tokenizer=teacher_tokenizer,
                reasoning_effort=reasoning_effort, context=f"turn[{start}]",
            )
            pieces = pieces_from_canonical(stripped, ds_text, message)
        except (AlignmentError, ValueError) as exc:
            failed.append((start, str(exc)))
            continue
        if "".join(p.glm for p in pieces) != stripped or "".join(p.ds for p in pieces) != ds_text:
            failed.append((start, "pieces do not reproduce both renderings"))
            continue
        turns.append((start, ids, pieces))

    teacher_ids = list(teacher_ids)
    t_ends = aligner.ds.byte_ends(teacher_ids)
    boundaries = set(t_ends)
    teacher_bytes = b"".join(aligner.ds.token_bytes(t) for t in teacher_ids)
    cursor = t_ends[teacher_prompt_len - 1] if teacher_prompt_len else 0
    chunks, stats, first_teacher = [], {}, len(teacher_ids)
    for start, ids, pieces in turns:
        completion = "".join(p.ds for p in pieces).encode()
        at = teacher_bytes.find(completion, cursor)
        if at < 0:
            failed.append((start, "turn rendering not found in the scored teacher sequence"))
            continue
        cursor = at + len(completion)
        if at and at not in boundaries:
            failed.append((start, "teacher turn start is not a token boundary"))
            continue
        lo = bisect.bisect_right(t_ends, at)
        hi = bisect.bisect_right(t_ends, at + len(completion))
        local_t_ends = [t_ends[i] - at for i in range(lo, hi)]
        s_ends = aligner.glm.byte_ends(ids)
        points, spans = _turn_chunks(pieces, s_ends, local_t_ends, at)
        first_teacher = min(first_teacher, lo)
        for (g0, d0), (g1, d1) in zip(points, points[1:]):
            s0, s1 = bisect.bisect_right(s_ends, g0), bisect.bisect_right(s_ends, g1)
            t0, t1 = bisect.bisect_right(local_t_ends, d0), bisect.bisect_right(local_t_ends, d1)
            kinds = {k for x, y, c, e, k in spans if (x < g1 and y > g0) or (c < d1 and e > d0)}
            if t0 == t1 or MISMATCH in kinds or kinds == {GLM_ONLY}:
                kind = MASKED
            elif kinds <= {SAME}:
                kind = EXACT if (s1 - s0, t1 - t0) == (1, 1) else TEXT_MULTI
            else:
                kind = STRUCT_1TO1 if (s1 - s0, t1 - t0) == (1, 1) else STRUCT_N
            chunks.append(Chunk((start + s0, start + s1), (lo + t0, lo + t1), kind))
            stats[kind] = stats.get(kind, 0) + (s1 - s0)
    return ChunkMap(teacher_ids, first_teacher, chunks, response_length, failed, stats)


def build_pair_chunk_map(*, teacher_ids, teacher_completion_start, loss_mask, pairs):
    """Chunk map of 1:1 EXACT pairs ``(student response row, global teacher position)``.

    Every other trainable row is MASKED, which Phase 3 sends to the Student anchor. Used
    for alignments decided elsewhere: the canonical-template 1:1 pairs of
    teacher_trajectory.py (GLM-4.5-Air <- GLM-5.2), or the identity map below.
    """
    response_length = len(loss_mask)
    chunks, paired = [], set()
    for s, t in pairs:
        s, t = int(s), int(t)
        if not 0 <= s < response_length or not loss_mask[s]:
            raise ValueError(f"aligned Student row {s} is not a trainable response row")
        if not teacher_completion_start <= t < len(teacher_ids):
            raise ValueError(f"aligned teacher position {t} is outside the teacher completion")
        if s in paired:
            raise ValueError(f"Student row {s} is paired twice")
        paired.add(s)
        chunks.append(Chunk((s, s + 1), (t, t + 1), EXACT))
    trainable = int(sum(1 for m in loss_mask if m))
    stats = {EXACT: len(chunks)}
    if trainable > len(chunks):
        stats[MASKED] = trainable - len(chunks)
    return ChunkMap(list(teacher_ids), teacher_completion_start, chunks, response_length, [], stats)


def build_identity_chunk_map(*, tokens, loss_mask, scored_rows=None):
    """Chunk map for a teacher that shares the Student's tokenizer and template (e.g. Qwen).

    The teacher scores the Student's own sequence, so response row ``r`` is teacher
    position ``prompt_len + r`` and every chunk is a 1:1 EXACT pair. ``scored_rows`` (bool
    ``[R]``) restricts the pairs to rows that actually carry a teacher value -- e.g. a
    sampled token that missed a cached Top-K -- and leaves the rest MASKED, which Phase 3
    sends to the Student anchor exactly like the cached sampled-token loss's miss policy.
    """
    response_length = len(loss_mask)
    prompt_len = len(tokens) - response_length
    if prompt_len < 0:
        raise ValueError("loss_mask is longer than the token sequence")
    pairs = [
        (r, prompt_len + r)
        for r, trainable in enumerate(loss_mask)
        if trainable and (scored_rows is None or scored_rows[r])
    ]
    return build_pair_chunk_map(
        teacher_ids=tokens, teacher_completion_start=prompt_len, loss_mask=loss_mask, pairs=pairs
    )


def chunk_logprobs(chunk_map, teacher_logprobs, mode="struct"):
    """Dense [R] teacher log-prob share, trainable mask and chunk kind for one sample.

    ``teacher_logprobs[p]`` is the teacher's log-prob of its own token at global position
    ``p`` of ``chunk_map.teacher_ids``. Rows outside any used chunk stay (0.0, 0) and must
    be finite: Phase 3 multiplies them by a zero mask, and ``0 * inf`` would be NaN.
    """
    use = MODES[mode]
    logprob = [0.0] * chunk_map.response_length
    mask = [0] * chunk_map.response_length
    kind = [0] * chunk_map.response_length
    for chunk in chunk_map.chunks:
        (s0, s1), (t0, t1) = chunk.student, chunk.teacher
        for i in range(s0, s1):
            kind[i] = KIND_CODES[chunk.kind]
        if chunk.kind not in use:
            continue
        share = sum(teacher_logprobs[t0:t1]) / (s1 - s0)
        for i in range(s0, s1):
            logprob[i] = share
            mask[i] = 1
    return logprob, mask, kind


def teacher_token_logprobs(reward, *, teacher_ids_len, teacher_prompt_len):
    """Dense ``[teacher_ids_len]`` log-probs of the teacher's own tokens, 0.0 outside the window.

    sglang returns one ``input_token_logprobs`` entry per scored position; the first is the
    boundary position that has no preceding-token distribution, because ``reward_func``
    narrows the window with ``logprob_start_len = teacher_prompt_len - 1``. Entries are a
    bare float or ``[logprob, token_id, token_text]`` depending on the sglang build.
    """
    entries = reward["meta_info"]["input_token_logprobs"][1:]
    expected = teacher_ids_len - teacher_prompt_len
    if len(entries) != expected:
        raise ValueError(
            f"teacher returned {len(entries)} scored positions for a {expected}-token "
            "completion; the logprob window is misaligned with the rebuilt sequence"
        )
    out = [0.0] * teacher_ids_len
    for j, item in enumerate(entries):
        value = item[0] if isinstance(item, (list, tuple)) else item
        out[teacher_prompt_len + j] = float(value)
    return out


_TOOLS_RE = re.compile(r"<tools>\n(.*?)\n</tools>\n*", re.S)


def parse_student_prompt(prompt):
    """Split the rendered GLM prompt into (system, user, tools).

    The tool schemas the student saw are rendered as one JSON object per line inside the
    system block, so this is the most faithful source available; it also keeps the writer
    independent of the agent scaffold.
    """
    import json

    body = prompt
    for marker in ("[gMASK]", "<sop>"):
        if body.startswith(marker):
            body = body[len(marker) :]
    if not body.startswith("<|system|>\n"):
        raise ValueError("prompt does not start with <|system|>")
    body = body[len("<|system|>\n") :]
    cut = body.index("<|user|>\n")
    system, rest = body[:cut], body[cut + len("<|user|>\n") :]
    if not rest.endswith("<|assistant|>"):
        raise ValueError("prompt does not end with <|assistant|>")
    user = rest[: -len("<|assistant|>")]
    m = _TOOLS_RE.search(system)
    if not m:
        raise ValueError("no <tools> block in system prompt")
    tools = [json.loads(line) for line in m.group(1).splitlines() if line.strip()]
    return (system[: m.start()] + system[m.end() :]).strip(), user, tools


def seed_messages_of(prompt):
    system, user, tools = parse_student_prompt(prompt)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}], tools


@lru_cache(maxsize=1)
def make_aligner(assets_dir, reasoning_effort=None):
    """GlmToDsAligner over <assets>/{glm_tokenizer.json,ds_tokenizer.json,encoding_dsv4.py}."""
    from pathlib import Path

    from glm_ds_align import GlmToDsAligner

    root = Path(assets_dir)
    return GlmToDsAligner(
        root / "glm_tokenizer.json",
        root / "ds_tokenizer.json",
        root / "encoding_dsv4.py",
        reasoning_effort=reasoning_effort,
    )


def seed_messages_from_scaffold(metadata):
    """Rebuild the turn-0 context the way teacher_trajectory.py does.

    Parsing the rendered GLM prompt is not enough: GLM wraps the real system prompt in its
    own tool-call instructions ("# Tools ... <tool_call>{function-name}"), which must not
    reach the teacher. teacher_trajectory.py rebuilds from the scaffold instead, and the
    chunk map has to index the same teacher sequence token for token.
    """
    from examples.coding_agent_opd.r2egym import prompt as agent_prompt
    from examples.coding_agent_opd.r2egym.scaffold import load_scaffold

    scaffold = load_scaffold()
    problem = metadata.get("problem_statement")
    if not isinstance(problem, str):
        problem = (metadata.get("ds") or {}).get("problem_statement")
    if not isinstance(problem, str):
        raise ValueError("metadata has no problem_statement")
    workdir = metadata.get("workdir") or "/testbed"
    return agent_prompt.initial_messages(scaffold, problem, workdir), scaffold.tools


def _split_ws(text):
    """(leading whitespace, core, trailing whitespace)."""
    core = text.strip()
    if not core:
        return text, "", ""
    at = text.index(core[0])
    return text[:at], core, text[at + len(core) :]


def _anchors_of(message):
    """Strings that must appear verbatim in both the Student text and the Teacher text."""
    out = []
    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        out.append(reasoning)
    content = message.get("content") or ""
    if content:
        out.append(content)
    for call in message.get("tool_calls") or []:
        fn = call.get("function", call)
        out.append(fn.get("name"))
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (TypeError, ValueError):
                args = {}
        for key, value in (args or {}).items():
            out.append(key)
            if isinstance(value, str) and value.strip():
                out.append(value)
    return [a for a in out if a]


_GLM_ARG = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S)
_DS_PARAM = re.compile(r'<\uff5cDSML\uff5cparameter name="(.*?)" string="(?:true|false)">(.*?)</\uff5cDSML\uff5cparameter>', re.S)
_DS_INVOKE = re.compile(r'<\uff5cDSML\uff5cinvoke name="(.*?)">')


def _emit_pair(pieces, glm_text, ds_text, g, d, g_span, d_span):
    """Structural prefix, then the paired span with its boundary whitespace split off.

    Boundary whitespace stays structural because on the Teacher side it sits against
    template-inserted bytes and BPE merges the two, so it can never align 1:1.
    """
    (g0, g1), (d0, d1) = g_span, d_span
    glead, gcore, _ = _split_ws(glm_text[g0:g1])
    dlead, dcore, _ = _split_ws(ds_text[d0:d1])
    pieces.append(Piece(glm_text[g:g0] + glead, ds_text[d:d0] + dlead, STRUCT))
    if gcore or dcore:
        pieces.append(Piece(gcore, dcore, SAME if gcore == dcore else MISMATCH))
    return g0 + len(glead) + len(gcore), d0 + len(dlead) + len(dcore)


def pieces_from_canonical(glm_text, ds_text, message):
    """Pieces for one turn, pairing spans by STRUCTURE rather than by content search.

    Values are sliced out of each side's own markup and never searched for: Student
    tool arguments are routinely substrings of one another (``old_str`` / ``new_str``),
    so a content search silently lands on the wrong occurrence. A pair whose two sides
    differ is emitted as MISMATCH, because the rollout parser may normalise an argument.
    """
    pieces, g, d = [], 0, 0
    g_close, d_close = glm_text.find("</think>"), ds_text.find("</think>")
    if (g_close < 0) != (d_close < 0):
        raise AlignmentError("</think> present on only one side")
    if g_close >= 0:
        g_open = glm_text.find("<think>")
        g = 0
        start = g_open + len("<think>") if 0 <= g_open < g_close else 0
        g, d = _emit_pair(pieces, glm_text, ds_text, g, d, (start, g_close), (0, d_close))
        g, d = g_close + len("</think>"), d_close + len("</think>")
        pieces.append(Piece("</think>", "</think>", STRUCT))

    invoke = _DS_INVOKE.search(ds_text)
    if bool(message.get("tool_calls")) != bool(invoke):
        raise AlignmentError("tool-call block on only one side")
    if invoke is None:
        g, d = _emit_pair(pieces, glm_text, ds_text, g, d, (g, len(glm_text)), (d, len(ds_text)))
        pieces.append(Piece(glm_text[g:], ds_text[d:], STRUCT))
        return [p for p in pieces if p.glm or p.ds]

    g_tc = glm_text.rfind("<tool_call>")
    if g_tc < g:
        raise AlignmentError("<tool_call> missing on the Student side")
    g, d = _emit_pair(pieces, glm_text, ds_text, g, d, (g, g_tc), (d, invoke.start()))
    g_name_end = min(
        x for x in (glm_text.find("<arg_key>", g_tc), glm_text.find("</tool_call>", g_tc), len(glm_text)) if x >= 0
    )
    g, d = _emit_pair(
        pieces, glm_text, ds_text, g, d,
        (g_tc + len("<tool_call>"), g_name_end), (invoke.start(1), invoke.end(1)),
    )
    g_args = list(_GLM_ARG.finditer(glm_text, g_tc))
    d_args = list(_DS_PARAM.finditer(ds_text, invoke.end()))
    if len(g_args) != len(d_args):
        raise AlignmentError(f"argument count differs: student {len(g_args)} vs teacher {len(d_args)}")
    for ga, da in zip(g_args, d_args):
        for group in (1, 2):
            g, d = _emit_pair(pieces, glm_text, ds_text, g, d, ga.span(group), da.span(group))
    pieces.append(Piece(glm_text[g:], ds_text[d:], STRUCT))
    return [p for p in pieces if p.glm or p.ds]
