##############################################################################
# tool definitions
##############################################################################

# Import allowed commands from the editor module
from .str_replace_editor import ALLOWED_STR_REPLACE_EDITOR_COMMANDS

# V1 File Editor
_FILE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated (preserving the beginning and end)
* The `undo_edit` command will revert the last edit made to the file at `path`
* For Python files longer than 110 lines, `view` without `view_range` automatically uses a condensed skeletal view (equivalent to `concise=True`). Use `view_range` to see the full content of specific line ranges.

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

file_editor = {
    "type": "function",
    "function": {
        "name": "file_editor",
        "description": _FILE_EDITOR_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": "The command to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
                    "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
                    "type": "string",
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    "type": "string",
                },
                "file_text": {
                    "description": "Required for the `create` command, contains the content of the file to be created.",
                    "type": "string",
                },
                "old_str": {
                    "description": "Required for the `str_replace` command, specifies the string in `path` to replace.",
                    "type": "string",
                },
                "new_str": {
                    "description": "Optional for the `str_replace` command to specify the replacement string. Required for the `insert` command to specify the string to insert.",
                    "type": "string",
                },
                "insert_line": {
                    "description": "Required for the `insert` command. The `new_str` will be inserted AFTER the line specified.",
                    "type": "integer",
                },
                "view_range": {
                    "description": "Optional for the `view` command when `path` points to a file. Specifies the line range to view. E.g., [11, 12] shows lines 11 and 12. Indexing starts at 1. Use [start_line, -1] to show all lines from `start_line` to the end.",
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "concise": {
                    "description": "Optional for the `view` command. Defaults to `False`. If `True`, displays a condensed skeletal view of Python files, eliding large function bodies. For Python files longer than 110 lines, this is automatically enabled when no `view_range` is specified. To see the full content, use `view_range` to view specific line ranges.",
                    "type": "boolean",
                },
                "python_only": {
                    "description": "Optional for the `view` command. If `True`, only list `.py` files when viewing a directory, and reject viewing non-`.py` files. Defaults to `False`.",
                    "type": "boolean",
                },
            },
            "required": ["command", "path"],
        },
    },
}


_STR_REPLACE_EDITOR_DESCRIPTION = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

str_replace_editor_tool = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": _STR_REPLACE_EDITOR_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": f"The command to run. Allowed options are: {', '.join(f'`{cmd}`' for cmd in ALLOWED_STR_REPLACE_EDITOR_COMMANDS)}.",
                    "enum": ALLOWED_STR_REPLACE_EDITOR_COMMANDS,
                    "type": "string",
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    "type": "string",
                },
                "file_text": {
                    "description": "Required for the `create` command, contains the content of the file to be created.",
                    "type": "string",
                },
                "old_str": {
                    "description": "Required for the `str_replace` command, specifies the string in `path` to replace.",
                    "type": "string",
                },
                "new_str": {
                    "description": "Optional for the `str_replace` command to specify the replacement string. Required for the `insert` command to specify the string to insert.",
                    "type": "string",
                },
                "insert_line": {
                    "description": "Required for the `insert` command. The `new_str` will be inserted AFTER the line specified.",
                    "type": "integer",
                },
                "view_range": {
                    "description": "Optional for the `view` command when `path` points to a file. Specifies the line range to view. E.g., [11, 12] shows lines 11 and 12. Indexing starts at 1. Use [start_line, -1] to show all lines from `start_line` to the end.",
                    "type": "array",
                    "items": {"type": "integer"},
                },
            },
            "required": ["command", "path"],
        },
    },
}


_R2EGYM_BASH_EXECUTE_DESCRIPTION = """Execute a bash command in the terminal.
* Non-persistent session: Each command runs in a fresh shell starting from the working directory. Environment variables, virtual environments, and directory changes do NOT persist between commands. If you need to change directory before running a command, chain them together (e.g., `cd /path && command`).
* If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
"""

r2egym_bash_execute_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": _R2EGYM_BASH_EXECUTE_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "cmd": {
                    "type": "string",
                    "description": "The bash command to execute.",
                }
            },
            "required": ["cmd"],
        },
    },
}



_BASH_DESCRIPTION = """
Description: Execute a bash command in the terminal.

Parameters:
  (1) command (string, optional): The bash command to execute. For example: `python my_script.py`. If not provided, will show help.
"""

execute_bash_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": _BASH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The command (and optional arguments) to execute. For example: 'python my_script.py'",
                }
            },
            "required": [],
        },
    },
}


_SEARCH_DESCRIPTION = """Search for a term in either a directory or a single file.
* If `path` points to a directory (default is `.`), we recursively search all non-hidden files and directories.
* If `path` points to a file, we run `grep -n` on that file to find line numbers containing the search term.
* If more than 100 files match (directory search scenario), the tool will stop listing and inform you to narrow your search.
* If no files are found that match your search term, the tool will inform you of that as well.
"""

search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": _SEARCH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "search_term": {
                    "description": "The term to search for in files.",
                    "type": "string",
                },
                "path": {
                    "description": "The file or directory to search in. Defaults to `.` if not specified.",
                    "type": "string",
                },
                "python_only": {
                    "description": "If True, only search .py files when searching a directory. Defaults to False.",
                    "type": "boolean",
                },
            },
            "required": ["search_term"],
        },
    },
}

# V1 Finish
_FINISH_DESCRIPTION = (
    "A simple finish tool with a 'submit' command.\n\n"
    "Notes about the `submit` command:\n"
    "* When invoked with `result`, the provided string is used for submitting required task results.\n"
    "* If no `result` is provided, it defaults to an empty string.\n"
)
finish_tool = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": _FINISH_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": "The command to run. Currently only `submit` is supported.",
                    "type": "string",
                    "enum": ["submit"],
                },
                "result": {
                    "description": "Optional. The result text to submit. Defaults to an empty string if not provided.",
                    "type": "string",
                },
            },
            "required": ["command"],
        },
        # "cache_control": {"type": "ephemeral"},
    },
}

# V2 Submit
_SUBMIT_DESCRIPTION = """
A simple submit tool to finish tasks.

This tool signals completion of a task or submission of results.
No parameters required - simply call to indicate task completion.
"""

submit_tool = {
    "type": "function",
    "function": {
        "name": "submit",
        "description": _SUBMIT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


_AIDER_SEARCH_REPLACE_EDIT_DESCRIPTION = (
    "Apply SEARCH/REPLACE operations to one or more files.\n\n"
    "Rules:\n"
    "1. Use `diffs` to describe per-file edits. Each diff is an object with:\n"
    "   {\"path\": <file_path>, \"search\": <search_block>, \"replace\": <replace_block>}.\n"
    "2. Each `search` block must **exactly match** a contiguous portion of the file content,\n"
    "   including whitespace, comments, docstrings, and any literal container markup.\n"
    "3. Only the **first occurrence** of each `search` block will be replaced.\n"
    "4. If `search` is empty, the entire file will be replaced by the `replace` content (creates a new file).\n"
    "5. Keep edits minimal; include just enough context for uniqueness.\n"
    "6. Diffs are applied sequentially in the given order."
)

aider_search_replace_edit = {
    "type": "function",
    "function": {
        "name": "aider_search_replace_edit",
        "description": _AIDER_SEARCH_REPLACE_EDIT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "diffs": {
                    "description": "List of file edit operations. Each diff contains {path, search, replace}.",
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Absolute file path, e.g. '/workspace/file.py'."
                            },
                            "search": {
                                "type": "string",
                                "description": "Exact contiguous text block to match. Empty means full file replace."
                            },
                            "replace": {
                                "type": "string",
                                "description": "Replacement text written in place of the matched block."
                            }
                        },
                        "required": ["path", "search", "replace"]
                    }
                }
            },
            "required": ["diffs"]
        }
    }
}


_AIDER_WHOLE_EDIT_DESCRIPTION = (
    "Write the complete content of one or more files.\n\n"
    "Rules:\n"
    "1. Each file entry specifies the target path and the complete content it should contain.\n"
    "2. The `file_content` must include the entire file content of the file.\n"
    "3. Use this for writing files in their entirety, not for partial edits."
)

aider_whole_edit = {
    "type": "function",
    "function": {
        "name": "aider_whole_edit",
        "description": _AIDER_WHOLE_EDIT_DESCRIPTION,
        "parameters": {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "description": "List of files to write. Each item defines a file path and its complete content.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Absolute file path, e.g. '/workspace/file.py'."
                            },
                            "file_content": {
                                "type": "string",
                                "description": "Full text content that the file should contain."
                            }
                        },
                        "required": ["path", "file_content"]
                    }
                }
            },
            "required": ["files"]
        }
    }
}


##############################################################################
# scale_swe tool definitions
#
# Ported VERBATIM from the reference benchmark tree
# (other_infra/R2E-Gym-benchmark/src/r2egym/agenthub/tools/__init__.py:442-541,
# selected there by agent.py:471-472 `elif "scale_swe" in self.scaffold`).
#
# These are the schemas the reference eval ADVERTISES to the model for the
# scale_swe scaffold, and they are NOT interchangeable with the generic
# str_replace_editor_tool / execute_bash_tool / finish_tool above: their
# `function.name`s collide with those, but the property sets, `required` lists
# and descriptions differ. Since the `tools=` array is rendered into the
# system prompt's `# Tools` block by the chat template, a mismatch here is a
# PROMPT mismatch -- the model reads a different manual for the same CLI.
#
# The in-pod CLIs (tools/scale_swe/*.py) already accept exactly this interface
# (`execute_bash <command> [--timeout N]`), so nothing on the execution path
# changes; only what the model is told.
#
# Byte-identity with the reference is asserted by
# tests/test_agent/test_scaffold_tools.py against the committed golden copy of
# other_infra/R2E-Gym-benchmark/scale_swe-tools.json. Do not "clean up" the
# wording below.
##############################################################################

_STR_REPLACE_EDITOR_DESCRIPTION_ScaleSWE = """Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique.
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""
scale_swe_str_replace_editor_tool = {
    "type": "function",
    "function": {
        "name": "str_replace_editor",
        "description": _STR_REPLACE_EDITOR_DESCRIPTION_ScaleSWE,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`.",
                    "enum": ["view", "create", "str_replace", "insert"],
                    "type": "string"
                },
                "path": {
                    "description": "Absolute path to file or directory, e.g. `/testbed/file.py` or `/testbed`.",
                    "type": "string"
                },
                "file_text": {
                    "description": "Required parameter of `create` command, with the content of the file to be created.",
                    "type": "string"
                },
                "old_str": {
                    "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
                    "type": "string"
                },
                "new_str": {
                    "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
                    "type": "string"
                },
                "insert_line": {
                    "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
                    "type": "integer"
                },
                "view_range": {
                    "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
                    "type": "array",
                    "items": {"type": "integer"}
                }
            },
            "required": ["command", "path"]
        }
    }
}
_BASH_DESCRIPTION_ScaleSWE = """Execute a bash command in the terminal.
* One command at a time: You can only execute one bash command at a time. If you need to run multiple commands sequentially, use `&&` or `;` to chain them together.
* Non-persistent session: Each command runs in a fresh shell starting from the working directory. Environment variables, virtual environments, and directory changes do NOT persist between commands. If you need to change directory before running a command, chain them together (e.g., `cd /path && command`).
* Timeout: Commands have a hard timeout. If a command exceeds the time limit, it will be terminated and any partial output will be returned.
* Output may be truncated: If the command output is too long, it will be truncated. Use targeted commands to get the information you need.
"""

scale_swe_execute_bash_tool = {
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": _BASH_DESCRIPTION_ScaleSWE,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                        "type": "string",
                        "description": "The bash command to execute."
                    },
                "timeout": {
                        "type": "integer",
                        "description": "Optional. Set a shorter timeout in seconds for this command. Useful for commands that should fail fast (e.g., quick checks or tests)."
                    }
            },
            "required": [
                "command"
            ],
        },
    },
}
_FINISH_DESCRIPTION_ScaleSWE = """Finish the interaction when the task is complete OR if you cannot proceed further with the task.

No parameters are required for this function.
"""
scale_swe_finish_tool = {
    "type": "function",
    "function": {
        "name": "finish",
        "description": _FINISH_DESCRIPTION_ScaleSWE,
        "parameters": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
}
