from types import SimpleNamespace

from examples.coding_agent_opd.eval.eval_chat import _chat_observation_message


def test_fn_calling_observation_without_call_id_uses_user_role():
    scaffold = SimpleNamespace(fn_calling=True)

    msg = _chat_observation_message(
        scaffold,
        "The previous step did not result in a valid tool execution.",
    )

    assert msg == {
        "role": "user",
        "content": "The previous step did not result in a valid tool execution.",
    }


def test_inline_tool_fallback_observation_uses_user_role():
    scaffold = SimpleNamespace(fn_calling=True)

    msg = _chat_observation_message(
        scaffold,
        "command output",
        tool_name="execute_bash",
    )

    assert msg == {"role": "user", "content": "command output"}


def test_structured_tool_observation_keeps_tool_call_id():
    scaffold = SimpleNamespace(fn_calling=True)

    msg = _chat_observation_message(
        scaffold,
        "command output",
        tool_call_id="call_123",
        tool_name="execute_bash",
    )

    assert msg == {
        "role": "tool",
        "content": "command output",
        "tool_call_id": "call_123",
        "name": "execute_bash",
    }


def test_non_fn_calling_observation_remains_user_role():
    scaffold = SimpleNamespace(fn_calling=False)

    msg = _chat_observation_message(scaffold, "command output")

    assert msg == {"role": "user", "content": "command output"}
