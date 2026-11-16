from examples.coding_agent_opd.r2egym import prompt


class _BoundaryTokenizer:
    eos_token_id = 9

    def __init__(self, user_ids, tool_ids):
        self._ids = {"user": user_ids, "tool": tool_ids}

    def apply_chat_template(
        self, messages, *, tools=None, tokenize=False, add_generation_prompt=True
    ):
        del tools, tokenize, add_generation_prompt
        return "prefix" + prompt._ASSISTANT_SENTINEL + messages[-1]["role"]

    def __call__(self, text, *, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": list(self._ids[text])}


def _delta(tokenizer, role, last_token_id):
    return prompt.render_turn_delta(
        tokenizer,
        [],
        {"role": role, "content": "observation"},
        last_token_id=last_token_id,
    )


def test_qwen_shared_eos_boundary_is_deduplicated():
    tokenizer = _BoundaryTokenizer(user_ids=[9, 101], tool_ids=[9, 102])

    assert _delta(tokenizer, "user", last_token_id=9) == [101]
    assert _delta(tokenizer, "tool", last_token_id=9) == [102]


def test_glm_role_specific_boundary_is_deduplicated():
    tokenizer = _BoundaryTokenizer(user_ids=[11, 101], tool_ids=[12, 102])

    assert _delta(tokenizer, "user", last_token_id=11) == [101]
    assert _delta(tokenizer, "tool", last_token_id=12) == [102]


def test_glm_role_boundary_is_kept_when_model_ended_with_regular_eos():
    tokenizer = _BoundaryTokenizer(user_ids=[11, 101], tool_ids=[12, 102])

    assert _delta(tokenizer, "user", last_token_id=tokenizer.eos_token_id) == [11, 101]
    assert _delta(tokenizer, "tool", last_token_id=None) == [12, 102]


def test_alignment_probe_accepts_role_specific_boundaries():
    tokenizer = _BoundaryTokenizer(user_ids=[11, 101], tool_ids=[12, 102])
    prompt._eos_alignment_checked = False

    prompt.ensure_tito_eos_alignment(tokenizer)

    assert prompt._eos_alignment_checked
    prompt._eos_alignment_checked = False
