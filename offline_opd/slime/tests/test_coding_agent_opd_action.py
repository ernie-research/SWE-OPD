from pathlib import Path

from examples.coding_agent_opd.r2egym.action import parse
from examples.coding_agent_opd.r2egym.scaffold import Scaffold


def _fn_calling_scaffold():
    return Scaffold(
        name="test",
        variant="edit_fn_calling",
        system_prompt="",
        instance_prompt="{problem_statement}",
        command_files=(Path("execute_bash"),),
        response_parser="parse_tool_call_response",
        fn_calling=True,
        tools=(
            {
                "type": "function",
                "function": {
                    "name": "execute_bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            },
        ),
    )


def test_glm45_native_tool_call_is_parsed_with_rollout_parser():
    text = (
        "<think>inspect the repository</think>\n"
        "<tool_call>execute_bash\n"
        "<arg_key>command</arg_key>\n"
        "<arg_value>ls -la /testbed</arg_value>\n"
        "</tool_call>"
    )

    thought, action = parse(
        text,
        _fn_calling_scaffold(),
        tool_call_parser="glm45",
        reasoning_parser="glm45",
    )

    assert action is not None
    assert action.function_name == "execute_bash"
    assert action.parameters == {"command": "ls -la /testbed"}
    assert thought == "inspect the repository"


def test_glm45_native_tool_call_requires_the_rollout_parser_name():
    text = (
        "<tool_call>execute_bash\n"
        "<arg_key>command</arg_key>\n"
        "<arg_value>pwd</arg_value>\n"
        "</tool_call>"
    )

    _, action = parse(text, _fn_calling_scaffold(), tool_call_parser=None)

    assert action is None
