"""
constants and config for different agents
"""

# supported repos
SUPPORTED_REPOS = [
    "sympy",
    "pandas",
    "pillow",
    "numpy",
    "tornado",
    "coveragepy",
    "aiohttp",
    "pyramid",
    "datalad",
    "scrapy",
    "orange3",
]

# hidden / excluded files: to be hidden from the agent
SKIP_FILES = [
    "run_tests.sh",
    "syn_issue.json",
    "expected_test_output.json",
    "execution_result.json",
    "parsed_commit.json",
    "modified_files.json",
    "modified_entities.json",
    "r2e_tests",
]

SKIP_FILES_NEW = [
    "run_tests.sh",
    "r2e_tests",
]

# # continue msg for agent run loop (in case on null action)
# CONTINUE_MSG = """Please continue working on the task on whatever approach you think is suitable.
# If you think you have solved the task, please first send your answer to user through message and then finish the interaction.
# IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
# IMPORTANT: each response must have both a reasoning and function call. Again each response must have a function call.
# """

CONTINUE_MSG = """
You forgot to use a function call in your response. 
YOU MUST USE A FUNCTION CALL IN EACH RESPONSE.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
"""
ScaleSWE_CONTINUE_MSG = """The previous step did not result in a valid tool execution. This may happen because:
- No tool was called.
- The tool call failed due to a parsing or argument error.

Please carefully review your next action and make sure that:
- You call an appropriate tool if one is needed.
- The tool name is correct.
- All required arguments are provided.
- The arguments follow the expected format and schema.

After fixing the issue, try the tool call again.

If you think you have solved the task, please finish the interaction using the finish tool.

IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.
If you want to give up, use the "finish" tool to end the interaction."""

# timeout for bash commands
CMD_TIMEOUT = 120  # seconds: 5 minutes
