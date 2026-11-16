"""Run: XVOCAB_ASSETS=<dir with glm_tokenizer.json, ds_tokenizer.json, encoding_dsv4.py> pytest -q"""

import json
import os
import random
from pathlib import Path

import pytest

from glm_ds_align import (
    EXACT,
    MASKED,
    MODES,
    STRUCT_1TO1,
    STRUCT_N,
    AlignmentError,
    GlmToDsAligner,
    token_rewards,
)

ASSETS = Path(os.environ.get("XVOCAB_ASSETS", "/nonexistent"))
pytestmark = pytest.mark.skipif(not (ASSETS / "encoding_dsv4.py").exists(), reason="XVOCAB_ASSETS not set")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "read a file",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}, "limit": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "run",
            "parameters": {"type": "object", "properties": {"command": {"type": "string"}, "ids": {"type": "array"}}},
        },
    },
]
CTX = [
    {"role": "system", "content": "You are a coding agent."},
    {"role": "user", "content": "修一下 a.py 的 bug"},
    {
        "role": "assistant",
        "reasoning_content": "先看文件",
        "content": "",
        "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "Read", "arguments": '{"file_path": "/a.py"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "c0", "content": "print(1)"},
]
TOOL_COMPLETION = (
    "\n<think>文件很短，再读一下 b.py and run tests.</think>\n我先读一下。\n"
    "<tool_call>Read\n<arg_key>file_path</arg_key>\n<arg_value>/b.py</arg_value>\n"
    "<arg_key>limit</arg_key>\n<arg_value>5</arg_value>\n</tool_call>\n"
    "<tool_call>Bash\n<arg_key>command</arg_key>\n<arg_value>pytest -q tests/</arg_value>\n</tool_call><|observation|>"
)


@pytest.fixture(scope="module")
def aligner():
    return GlmToDsAligner(ASSETS / "glm_tokenizer.json", ASSETS / "ds_tokenizer.json", ASSETS / "encoding_dsv4.py")


def _check_partition(al):
    s_pos = t_pos = 0
    for c in al.chunks:
        assert c.student[0] == s_pos and c.teacher[0] == t_pos
        s_pos, t_pos = c.student[1], c.teacher[1]
    assert (s_pos, t_pos) == (al.n_student, al.n_teacher)


def _kind_of(al, aligner, student_ids, piece_text):
    """Kind of the chunk ending with `piece_text` (DS BPE may merge neighbouring markers into one chunk)."""
    for c in al.chunks:
        if b"".join(aligner.glm.token_bytes(t) for t in student_ids[c.student[0] : c.student[1]]).endswith(
            piece_text.encode()
        ):
            return c.kind
    return None


def test_tool_call_alignment(aligner):
    ids = aligner.glm.encode(TOOL_COMPLETION)
    al = aligner.align(CTX, TOOLS, ids)
    _check_partition(al)
    full_text = b"".join(aligner.ds.token_bytes(t) for t in al.teacher_input_ids).decode()
    assert full_text.endswith("</｜DSML｜tool_calls><｜end▁of▁sentence｜>")
    assert al.assistant_message["tool_calls"][0]["function"]["arguments"] == json.dumps(
        {"file_path": "/b.py", "limit": 5}
    )
    assert _kind_of(al, aligner, ids, "</think>") == STRUCT_1TO1
    assert _kind_of(al, aligner, ids, "<|observation|>") == STRUCT_N
    assert al.chunks[0].kind == MASKED  # "\n<think>": teacher prompt already ends with <think>
    assert al.stats[EXACT] > 10


def test_final_answer_and_rewards(aligner):
    ids = aligner.glm.encode("\n<think>done</think>\n修好了，a.py 第 1 行。<|user|>")
    al = aligner.align(CTX, TOOLS, ids)
    _check_partition(al)
    assert _kind_of(al, aligner, ids, "<|user|>") == STRUCT_1TO1  # <|user|> <-> <｜end▁of▁sentence｜>

    rng = random.Random(0)
    s_lp = [-rng.random() for _ in range(al.n_student)]
    t_lp = [-rng.random() for _ in range(al.n_teacher)]
    masks = {m: token_rewards(al, s_lp, t_lp, m)[1] for m in MODES}
    assert all(a <= b <= c for a, b, c in zip(masks["exact"], masks["struct"], masks["chunk"]))
    r, mask = token_rewards(al, s_lp, t_lp, "chunk")
    for c in al.chunks:  # chunk reward sums to sum(log p_T) - sum(log p_S)
        if c.kind != MASKED:
            got = sum(r[c.student[0] : c.student[1]])
            want = sum(t_lp[c.teacher[0] : c.teacher[1]]) - sum(s_lp[c.student[0] : c.student[1]])
            assert abs(got - want) < 1e-9


def test_non_string_arg_serialization_mismatch_is_masked(aligner):
    comp = (
        "<think>x</think><tool_call>Bash\n<arg_key>command</arg_key>\n<arg_value>ls</arg_value>\n"
        "<arg_key>ids</arg_key>\n<arg_value>[1,2]</arg_value>\n</tool_call><|observation|>"
    )
    ids = aligner.glm.encode(comp)
    al = aligner.align(CTX, TOOLS, ids)
    _check_partition(al)
    assert _kind_of(al, aligner, ids, "[") in (MASKED, None)
    text = b"".join(aligner.ds.token_bytes(t) for t in al.teacher_input_ids).decode()
    assert 'string="false">[1, 2]</｜DSML｜parameter>' in text


def test_truncated_tool_call_raises(aligner):
    ids = aligner.glm.encode("\n<think>a</think><tool_call>Read\n<arg_key>file_path</arg_key>\n<arg_value>/a")
    with pytest.raises(AlignmentError):
        aligner.align(CTX, TOOLS, ids)
