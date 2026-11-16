import re
from typing import Dict
from r2egym.agenthub.action import Action
from r2egym.agenthub import CONTINUE_MSG, ScaleSWE_CONTINUE_MSG

_SCAFFOLD_VALID_FUNCTIONS = {
    "r2egym": {"file_editor", "search", "execute_bash", "finish"},
    "r2egym_old": {"file_editor", "search", "execute_bash", "finish"},
    "sweagent": {"execute_bash", "str_replace_editor", "submit"},
    "openhands": {"execute_bash", "str_replace_editor", "submit"},
    "scale_swe": {"execute_bash", "str_replace_editor", "finish"},
}


class Observation:
    def __init__(self, bash_output, error_code, action: Action, num_lines: int=40):
        self.bash_output = bash_output
        self.error_code = error_code
        self.action = action
        self.num_lines = num_lines
        self.scaffold = ""

    def __str__(self):
        # empty or no function call
        if self.scaffold in ['scale_swe']:
            valid_functions = _SCAFFOLD_VALID_FUNCTIONS.get(self.scaffold, None)
            if not self.action.function_name:
                return ScaleSWE_CONTINUE_MSG
            elif valid_functions and self.action.function_name not in valid_functions:
                available = ", ".join(sorted(valid_functions))
                return f"ERROR: '{self.action.function_name}' is not a valid function. " + \
                    f"Available functions: {available}"
            elif self.action.function_name == "finish" or self.action.function_name == "submit":
                return "<<< Finished >>>"
            else:
                return self.bash_output
        else:
            if not self.action.function_name:
                return CONTINUE_MSG
            elif self.action.function_name == "finish" or self.action.function_name == "submit":
                return "<<< Finished >>>"
            else:
                if self.action.function_name == "execute_bash" or self.action.function_name == "bash":
                    if len(self.bash_output) > 16000:
                        prefix = self.bash_output[:8000]
                        suffix = self.bash_output[-8000:]
                        truncated_output = prefix + "\n<Observation truncated in middle for saving context>\n" + suffix
                    else:
                        truncated_output = self.bash_output

                    output = (
                        f"Exit code: {self.error_code}\n"
                        f"Execution output of [{self.action.function_name}]:\n"
                        f"{truncated_output}"
                    )
                    # output = f"Exit code: {self.error_code}\nExecution output of [{self.action.function_name}]:\n{self.bash_output}"
                else:
                    output = f"Execution output of [{self.action.function_name}]:\n{self.bash_output}"
                return output
