"""GLM-4.5-Air (student) <-> DeepSeek-V4-Flash-0731 (teacher) cross-vocab alignment for sampled-token OPD.

Pipeline:
  1. Parse the student's sampled GLM completion into "pieces": (glm_text, ds_text, kind).
     - SAME:     identical text on both sides (reasoning, content, tool name, arg key, string arg value)
     - STRUCT:   format markers mapped semantically (</think>, <tool_call>, <arg_key>, stop token, ...)
     - GLM_ONLY: student-only whitespace / tokens with no teacher counterpart (masked)
     - MISMATCH: same value but different serialization (e.g. non-string JSON args) (masked)
  2. Render the teacher text with DeepSeek's official encode_messages() and check that
     prompt + concat(ds pieces) reproduces it exactly.
  3. Find byte positions that are token boundaries on BOTH sides; consecutive common
     boundaries delimit chunks (student tokens [a, b) <-> teacher tokens [c, d)).
  4. Reward for a used chunk with n student tokens: r_i = sum(teacher_lp[c:d]) / n - student_lp_i.
     For 1:1 chunks this is the usual log p_T - log p_S of the sampled token.
"""

from __future__ import annotations

import bisect
import copy
import importlib.util
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

from tokenizers import Tokenizer

SAME, STRUCT, GLM_ONLY, MISMATCH = "same", "struct", "glm_only", "mismatch"

# Chunk kinds produced by the aligner.
EXACT = "exact"  # 1 student token <-> 1 teacher token, identical bytes
STRUCT_1TO1 = "struct_1to1"  # 1 <-> 1 mapped special token, e.g. </think> <-> </think>
STRUCT_N = "struct"  # format marker macro, n <-> m tokens
TEXT_MULTI = "text_multi"  # identical text but n <-> m tokens
MASKED = "masked"  # no teacher counterpart or serialization mismatch

MODES: dict[str, frozenset[str]] = {
    # Current practice: identical-string tokens + 1:1 special token mapping.
    "exact": frozenset({EXACT, STRUCT_1TO1}),
    # + tool-call / stop format markers as macro tokens (recommended).
    "struct": frozenset({EXACT, STRUCT_1TO1, STRUCT_N}),
    # + n:m chunks inside identical text.
    "chunk": frozenset({EXACT, STRUCT_1TO1, STRUCT_N, TEXT_MULTI}),
}

# GLM-4.5 markers.
GLM_THINK_OPEN, GLM_THINK_CLOSE = "<think>", "</think>"
GLM_TC_OPEN, GLM_TC_CLOSE = "<tool_call>", "</tool_call>"
GLM_STOP_TOKENS = ("<|observation|>", "<|user|>", "<|endoftext|>")
_GLM_STOP_RE = re.compile("|".join(re.escape(t) for t in GLM_STOP_TOKENS))
_GLM_ARG_RE = re.compile(r"(\s*)<arg_key>(.*?)</arg_key>(\s*)<arg_value>(.*?)</arg_value>", re.S)

# DeepSeek-V4 DSML markers (must match encoding_dsv4.py templates).
DS_EOS = "<｜end▁of▁sentence｜>"
_D = "｜DSML｜"
DS_TC_OPEN_FIRST = f'\n\n<{_D}tool_calls>\n<{_D}invoke name="'
DS_TC_OPEN_NEXT = f'\n<{_D}invoke name="'
DS_ARG_OPEN_FIRST = f'">\n<{_D}parameter name="'
DS_ARG_OPEN_NEXT = f'\n<{_D}parameter name="'
DS_ARG_MID = '" string="{}">'
DS_ARG_CLOSE = f"</{_D}parameter>"
DS_INVOKE_CLOSE = f"\n</{_D}invoke>"
DS_INVOKE_CLOSE_NOARGS = f'">\n\n</{_D}invoke>'
DS_TC_CLOSE = f"\n</{_D}tool_calls>"


class AlignmentError(ValueError):
    """Completion cannot be aligned (truncated / malformed); mask the whole sample."""


def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTE_DECODER = _byte_decoder()


class ByteLevelTokenizer:
    """Thin wrapper giving exact byte strings per token id (byte-level BPE)."""

    def __init__(self, tokenizer_json: str | Path):
        self.tok = Tokenizer.from_file(str(tokenizer_json))
        data = json.loads(Path(tokenizer_json).read_text())
        self.added = {a["id"]: a["content"].encode() for a in data["added_tokens"]}

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text, add_special_tokens=False).ids

    def token_bytes(self, tid: int) -> bytes:
        if tid in self.added:
            return self.added[tid]
        return bytes(_BYTE_DECODER[c] for c in self.tok.id_to_token(tid))

    def byte_ends(self, ids: list[int]) -> list[int]:
        ends, pos = [], 0
        for t in ids:
            pos += len(self.token_bytes(t))
            ends.append(pos)
        return ends


@dataclass
class Piece:
    glm: str
    ds: str
    kind: str


@dataclass
class Chunk:
    student: tuple[int, int]  # [start, end) into student completion tokens
    teacher: tuple[int, int]  # [start, end) into teacher completion tokens
    kind: str


@dataclass
class Alignment:
    teacher_input_ids: list[int]  # full teacher sequence (prompt + completion)
    teacher_completion_start: int  # index of first teacher completion token
    chunks: list[Chunk]
    n_student: int
    n_teacher: int
    assistant_message: dict[str, Any]  # parsed structured message (reuse for later turns)
    stats: dict[str, int] = field(default_factory=dict)


def load_ds_encoding(path: str | Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("encoding_dsv4", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_json(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def _arg_is_string(tool_name: str, key: str, raw: str, schemas: dict[str, dict]) -> tuple[bool, Any]:
    """Mimic the GLM-4.5 tool parser: string-typed params stay raw, others are JSON-decoded."""
    ptype = schemas.get(tool_name, {}).get("parameters", {}).get("properties", {}).get(key, {}).get("type")
    if ptype == "string":
        return True, raw
    try:
        v = json.loads(raw)
    except json.JSONDecodeError:
        return True, raw
    return isinstance(v, str), v


def parse_glm_completion(text: str, tools: list[dict] | None = None) -> tuple[list[Piece], dict[str, Any]]:
    """Split a sampled GLM-4.5 completion (text after `<|assistant|>`) into aligned pieces.

    Returns the pieces (glm sides concatenate to `text`) and the structured assistant message.
    """
    schemas = {t["function"]["name"]: t["function"] for t in (tools or [])}
    pieces: list[Piece] = []

    def add(glm: str, ds: str, kind: str) -> None:
        if glm or ds:
            pieces.append(Piece(glm, ds, kind))

    m = re.compile(r"\s*" + re.escape(GLM_THINK_OPEN)).match(text)
    if not m:
        raise AlignmentError("completion does not start with <think>")
    add(m.group(), "", GLM_ONLY)  # teacher prompt already ends with <think>
    i = m.end()
    j = text.find(GLM_THINK_CLOSE, i)
    if j < 0:
        raise AlignmentError("truncated inside reasoning")
    reasoning = text[i:j]
    add(reasoning, reasoning, SAME)
    add(GLM_THINK_CLOSE, "</think>", STRUCT)
    i = j + len(GLM_THINK_CLOSE)

    stop = _GLM_STOP_RE.search(text, i)
    end = stop.start() if stop else len(text)
    if stop and stop.end() != len(text):
        raise AlignmentError("text after stop token")

    k = text.find(GLM_TC_OPEN, i, end)
    body_end = k if k >= 0 else end
    raw = text[i:body_end]
    content = raw.strip()
    lead = len(raw) - len(raw.lstrip())
    add(raw[:lead], "", GLM_ONLY)
    add(content, content, SAME)
    pending = raw[lead + len(content) :]  # whitespace, merged into the next marker
    i = body_end

    tool_calls: list[dict[str, Any]] = []
    while i < end and text.startswith(GLM_TC_OPEN, i):
        close = text.find(GLM_TC_CLOSE, i, end)
        if close < 0:
            raise AlignmentError("truncated inside <tool_call>")
        body = text[i + len(GLM_TC_OPEN) : close]
        nm = re.match(r"(\s*)([^<]*?)(\s*)(?=<arg_key>|$)", body, re.S)
        if nm is None or not nm.group(2):
            raise AlignmentError(f"bad tool call body: {body[:80]!r}")
        name = nm.group(2)
        add(pending + GLM_TC_OPEN + nm.group(1), DS_TC_OPEN_NEXT if tool_calls else DS_TC_OPEN_FIRST, STRUCT)
        add(name, name, SAME)

        pos, args, pre_ws = nm.end(), {}, nm.group(3)
        for a in _GLM_ARG_RE.finditer(body, pos):
            if a.start() != pos:
                raise AlignmentError(f"unexpected text in tool call: {body[pos : a.start()]!r}")
            ws1, key, ws2, raw_val = a.groups()
            add(pre_ws + ws1 + "<arg_key>", DS_ARG_OPEN_NEXT if args else DS_ARG_OPEN_FIRST, STRUCT)
            add(key, key, SAME)
            is_str, val = _arg_is_string(name, key, raw_val, schemas)
            add("</arg_key>" + ws2 + "<arg_value>", DS_ARG_MID.format("true" if is_str else "false"), STRUCT)
            ds_val = val if is_str else _to_json(val)
            add(raw_val, ds_val, SAME if ds_val == raw_val else MISMATCH)
            add("</arg_value>", DS_ARG_CLOSE, STRUCT)
            if key in args:
                # The teacher renders one parameter per key (last value wins), so a
                # repeated <arg_key> can never be reproduced piece by piece.
                raise AlignmentError(f"duplicate argument key {key!r}")
            args[key] = val
            pos, pre_ws = a.end(), ""
        tail = body[pos:]
        if tail.strip():
            raise AlignmentError(f"unexpected text in tool call: {tail!r}")
        add(pre_ws + tail + GLM_TC_CLOSE, DS_INVOKE_CLOSE if args else DS_INVOKE_CLOSE_NOARGS, STRUCT)
        tool_calls.append(
            {
                "id": f"call_{len(tool_calls)}",
                "type": "function",
                "function": {"name": name, "arguments": _to_json(args)},
            }
        )
        i = close + len(GLM_TC_CLOSE)
        ws = re.match(r"\s*", text[i:end]).group()
        pending, i = ws, i + len(ws)

    if i != end:
        raise AlignmentError(f"unexpected text after tool calls: {text[i:end][:80]!r}")
    if stop is None:
        if tool_calls:
            raise AlignmentError("truncated after tool calls (no stop token)")
        add(pending, "", GLM_ONLY)
    else:
        add(pending + stop.group(), (DS_TC_CLOSE if tool_calls else "") + DS_EOS, STRUCT)

    msg: dict[str, Any] = {"role": "assistant", "reasoning_content": reasoning, "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    if stop is None:
        msg["wo_eos"] = True
    return pieces, msg


def _to_ds_messages(messages: list[dict], tools: list[dict] | None) -> list[dict]:
    msgs = copy.deepcopy(messages)
    if tools:
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["tools"] = tools
        else:
            msgs.insert(0, {"role": "system", "content": "", "tools": tools})
    return msgs


def _common_points(pieces: list[Piece], s_ends: list[int], t_ends: list[int]) -> list[tuple[int, int]]:
    """(student_byte, teacher_byte) positions that are token boundaries on both sides."""
    s_set, t_set = {0, *s_ends}, {0, *t_ends}
    s_sorted = [0, *s_ends]
    points = [(0, 0)]
    g = d = 0
    for p in pieces:
        gl, dl = len(p.glm.encode()), len(p.ds.encode())
        if p.kind == SAME:
            lo = bisect.bisect_right(s_sorted, g)
            hi = bisect.bisect_right(s_sorted, g + gl)
            cands = [(b, d + b - g) for b in s_sorted[lo:hi]]
        else:
            cands = [(g + gl, d + dl)]
        for sb, tb in cands:
            if sb in s_set and tb in t_set and (sb, tb) != points[-1]:
                if sb == points[-1][0]:
                    raise RuntimeError(f"teacher-only span at student byte {sb}")
                points.append((sb, tb))
        g, d = g + gl, d + dl
    return points


class GlmToDsAligner:
    def __init__(
        self,
        glm_tokenizer_json: str | Path,
        ds_tokenizer_json: str | Path,
        ds_encoding_py: str | Path,
        reasoning_effort: str | None = None,
    ):
        self.glm = ByteLevelTokenizer(glm_tokenizer_json)
        self.ds = ByteLevelTokenizer(ds_tokenizer_json)
        self.enc = load_ds_encoding(ds_encoding_py)
        self.reasoning_effort = reasoning_effort

    def _encode(self, msgs: list[dict]) -> str:
        return self.enc.encode_messages(msgs, thinking_mode="thinking", reasoning_effort=self.reasoning_effort)

    def align(self, messages: list[dict], tools: list[dict] | None, student_ids: list[int]) -> Alignment:
        """messages: OpenAI-format context the student saw (system/user/assistant/tool),
        tools: OpenAI-format tool list, student_ids: sampled GLM completion ids (incl. stop token)."""
        s_bytes = b"".join(self.glm.token_bytes(t) for t in student_ids)
        try:
            text = s_bytes.decode("utf-8")
        except UnicodeDecodeError as e:
            raise AlignmentError("student completion is not valid utf-8 (truncated?)") from e
        pieces, msg = parse_glm_completion(text, tools)
        if "".join(p.glm for p in pieces) != text:
            raise RuntimeError("parser lost student text")

        ctx = _to_ds_messages(messages, tools)
        prompt = self._encode(ctx)
        completion = "".join(p.ds for p in pieces)
        full = self._encode(ctx + [msg])
        if full != prompt + completion:
            raise RuntimeError("DS rendering drift: pieces do not reproduce encode_messages output")

        prompt_ids = self.ds.encode(prompt)
        full_ids = self.ds.encode(full)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("teacher prompt tokens are not a prefix of full tokens")
        t_ids = full_ids[len(prompt_ids) :]
        if b"".join(self.ds.token_bytes(t) for t in t_ids) != completion.encode():
            raise RuntimeError("teacher completion bytes mismatch")

        s_ends, t_ends = self.glm.byte_ends(student_ids), self.ds.byte_ends(t_ids)
        points = _common_points(pieces, s_ends, t_ends)

        spans, g, d = [], 0, 0  # piece spans in (student, teacher) byte space
        for p in pieces:
            gl, dl = len(p.glm.encode()), len(p.ds.encode())
            spans.append((g, g + gl, d, d + dl, p.kind))
            g, d = g + gl, d + dl

        chunks, stats = [], {}
        for (g0, d0), (g1, d1) in zip(points, points[1:]):
            s0, s1 = bisect.bisect_right(s_ends, g0), bisect.bisect_right(s_ends, g1)
            t0, t1 = bisect.bisect_right(t_ends, d0), bisect.bisect_right(t_ends, d1)
            kinds = {k for a, b, c, e, k in spans if (a < g1 and b > g0) or (c < d1 and e > d0)}
            if t0 == t1 or MISMATCH in kinds or kinds == {GLM_ONLY}:
                kind = MASKED
            elif kinds <= {SAME}:
                kind = EXACT if (s1 - s0, t1 - t0) == (1, 1) else TEXT_MULTI
            else:
                kind = STRUCT_1TO1 if (s1 - s0, t1 - t0) == (1, 1) else STRUCT_N
            chunks.append(Chunk((s0, s1), (t0, t1), kind))
            stats[kind] = stats.get(kind, 0) + (s1 - s0)

        return Alignment(full_ids, len(prompt_ids), chunks, len(student_ids), len(t_ids), msg, stats)


def token_rewards(
    alignment: Alignment,
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    mode: str = "struct",
) -> tuple[list[float], list[int]]:
    """Per-student-token OPD reward and loss mask.

    student_logprobs: log p_S of each sampled student completion token.
    teacher_logprobs: log p_T of each teacher completion token, i.e. computed on
        alignment.teacher_input_ids at positions [teacher_completion_start:].
    """
    if len(student_logprobs) != alignment.n_student or len(teacher_logprobs) != alignment.n_teacher:
        raise ValueError("logprob lengths do not match alignment")
    use = MODES[mode]
    rewards = [0.0] * alignment.n_student
    mask = [0] * alignment.n_student
    for c in alignment.chunks:
        if c.kind not in use:
            continue
        (s0, s1), (t0, t1) = c.student, c.teacher
        t_share = sum(teacher_logprobs[t0:t1]) / (s1 - s0)
        for i in range(s0, s1):
            rewards[i] = t_share - student_logprobs[i]
            mask[i] = 1
    return rewards, mask
