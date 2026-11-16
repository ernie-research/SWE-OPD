import os
import re
import copy
import yaml
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

import litellm
from openai import OpenAI

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.agent.parsers import load_parser
from r2egym.agenthub.environment.env import RepoEnv
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from r2egym.agenthub.config import SCAFFOLD_NAMES
from anthropic import Anthropic, AnthropicVertex  # Add Anthropic Vertex import
from r2egym.agenthub.tools import (
    r2egym_bash_execute_tool,
    search_tool,
    file_editor,
    finish_tool,
    str_replace_editor_tool,
    execute_bash_tool,
    submit_tool,
    aider_search_replace_edit,
    aider_whole_edit
)
from response_helper import OpenAIResponseHandler
import traceback
logger = get_logger(__name__)  # Logger for this module
# MAX_CONTEXT_TOKENS = 65536

##############################################################################
# AgentArgs Dataclass
##############################################################################
@dataclass
class AgentArgs:
    system_prompt: str
    instance_prompt: str
    command_files: List[Path]
    llm_name: str
    response_parser: str  # to handle the response parsing
    llm_base_url: Optional[str] = "http://localhost:8000/v1"  # None
    demo_file: Optional[Path] = None
    use_demo: Optional[bool] = False
    other_args: Optional[Dict[str, Any]] = None  # To handle extra configurations

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AgentArgs":
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        llm_base_url_env = os.environ.get("LLM_BASE_URL", None)
        if llm_base_url_env is not None:
            config["llm_base_url"] = llm_base_url_env
        
        if yaml_path.name.endswith("edit_fn_calling.yaml"):
            config["response_parser"] = config.get("response_parser", "parse_tool_call_response")
        else:
            config["response_parser"] = config.get("response_parser", "parse_xml_response")

        return cls(**config)


##############################################################################
# MultiLanguageAgentArgs Dataclass
##############################################################################
@dataclass
class MultiLanguageAgentArgs:
    system_prompt: str
    # Store all language-specific instance prompts
    python_instance_prompt: Optional[str] = None
    java_instance_prompt: Optional[str] = None
    go_instance_prompt: Optional[str] = None
    c_instance_prompt: Optional[str] = None
    cpp_instance_prompt: Optional[str] = None
    javascript_instance_prompt: Optional[str] = None
    typescript_instance_prompt: Optional[str] = None
    rust_instance_prompt: Optional[str] = None
    # Fallback instance prompt for unsupported languages
    instance_prompt: Optional[str] = None
    
    command_files: List[Path] = None
    llm_name: str = "gpt-4o"
    llm_base_url: Optional[str] = "http://localhost:8000/v1"
    demo_file: Optional[Path] = None
    use_demo: Optional[bool] = False
    other_args: Optional[Dict[str, Any]] = None

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "MultiLanguageAgentArgs":
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)

        llm_base_url_env = os.environ.get("LLM_BASE_URL")
        if llm_base_url_env is not None:
            config["llm_base_url"] = llm_base_url_env

        return cls(**config)
    
    def get_instance_prompt_for_language(self, language: str) -> str:
        """
        Get the appropriate instance prompt based on the language field.
        
        Args:
            language: The programming language (e.g., 'python', 'java', 'go', etc.)
            
        Returns:
            The appropriate instance prompt for the given language
        """
        # Convert language to lowercase for case-insensitive matching
        lang = language.lower().strip()
        

        # Language mapping
        language_prompt_map = {
            'python': {
                "reproduce_issue_description": "Create a script at '{working_dir}/reproduce_issue.py' that demonstrates the error.",
                "how_to_run_test": """Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
         - `python -m pytest -xvs sympy/physics/units/tests/test_dimensions_transcendental.py`
         - `python -m pytest tests/test_domain_py.py::test_pymethod_options`
         - `./tests/runtests.py constraints.tests.CheckConstraintTests -v 2`""",
                "how_to_run_test_edge_case": "Create additional test cases in a separate file '{working_dir}/edge_case_tests.py'."
            },
            'java': {
                "reproduce_issue_description": "Create a Java class at '{working_dir}/ReproduceIssue.java' that demonstrates the error. Compile the class using `javac ReproduceIssue.java` and execute it with `java ReproduceIssue` via the execute_bash to confirm the error behavior.",
                "how_to_run_test": """Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
         - `mvn test -Dtest=DimensionsTranscendentalTest`
         - `./gradlew test --tests DomainOptionsTest`
         - `ant test -Dtestcase=CheckConstraintTests`""",
                "how_to_run_test_edge_case": "Create additional test cases in a separate file '{working_dir}/EdgeCaseTests.java'."
            },
            'go': {
                "reproduce_issue_description": "Create a Go script at '{working_dir}/reproduce_issue.go' that demonstrates the error. Execute this script using `go run reproduce_issue.go` via the execute_bash to confirm the error behavior.",
                "how_to_run_test": """Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
         - `go test -run TestDimensionsTranscendental`
         - `go test ./pkg/domain -run TestDomainOptions`
         - `go test -run TestCheckConstraints ./internal/...`""",
                "how_to_run_test_edge_case": "Create additional test cases in a separate file '{working_dir}/edge_case_test.go'."
            },
            'c': {
                "reproduce_issue_description": "Create a C script at '{working_dir}/reproduce_issue.c' that demonstrates the error. Compile the script using `gcc reproduce_issue.c -o reproduce_issue` and execute it using the execute_bash to confirm the error behavior.",
                "how_to_run_test": """Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
         - `go test -run TestDimensionsTranscendental`
         - `go test ./pkg/domain -run TestDomainOptions`
         - `go test -run TestCheckConstraints ./internal/...`""",
                "how_to_run_test_edge_case": "Create additional test cases in a separate file '{working_dir}/edge_case_test.c'."
            },
            'cpp': {
                "reproduce_issue_description": "Create or adapt a small executable (e.g., a main file or a test driver) at '{working_dir}/reproduce_issue.cpp' that demonstrates the error. Build and run it (for example, by using `g++ -o reproduce_issue reproduce_issue.cpp && ./reproduce_issue` via the execute_bash) to confirm the error behavior.",
                "how_to_run_test": """Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
         - `g++ -o reproduce_issue reproduce_issue.cpp && ./reproduce_issue`
         - `make test`""",
                "how_to_run_test_edge_case": "Create additional test cases in a separate file '{working_dir}/edge_case_tests.cpp'."
            },
            'javascript': {
                "reproduce_issue_description": "Create a script at '{working_dir}/reproduce_issue.js' that demonstrates the error. Execute this script using `node reproduce_issue.js` to confirm the error behavior.",
                "how_to_run_test": """Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
         - `npm test`
         - `npx jest`
         - `node test/my_test_file.js`""",
                "how_to_run_test_edge_case": "Create additional test cases in a separate file '{working_dir}/edge_case_tests.js'."
            },
            'typescript': {
                "reproduce_issue_description": "Create a script at '{working_dir}/reproduce_issue.ts' that demonstrates the error. Execute it using `ts-node {working_dir}/reproduce_issue.ts` to confirm the error behavior.",
                "how_to_run_test": """Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
         - `ts-node {working_dir}/reproduce_issue.ts`""",
                "how_to_run_test_edge_case": "Create additional test cases in a separate file '{working_dir}/edge_case_tests.ts'."
            },
            'rust': {
                "reproduce_issue_description": "Create a script at '{working_dir}/reproduce_issue.rs' that demonstrates the error. Execute it using `cargo run --bin reproduce_issue` to confirm the error behavior.",
                "how_to_run_test": """Use the existing test runner to run the unit tests you identify as relevant to the changes you made. For example:
         - `cargo test`""",
                "how_to_run_test_edge_case": "Create additional test cases in a separate file '{working_dir}/edge_case_tests.rs'."
            },
        }
        language_prompt_map["c++"] = language_prompt_map["cpp"]
        
        # Try to get language-specific prompt
        language_variables = language_prompt_map.get(lang, None)
        if language_variables is None:
            raise ValueError(f"No instance prompt found for language '{lang}'")
        language_variables["language"] = lang

        instance_prompt = self.instance_prompt
        for key, value in language_variables.items():
            instance_prompt = instance_prompt.replace("{" + key + "}", value)

        return instance_prompt
    
    def to_agent_args(self, language: str) -> AgentArgs:
        """
        Convert MultiLanguageAgentArgs to AgentArgs for the specified language.
        
        Args:
            language: The programming language to get the prompt for
            
        Returns:
            AgentArgs instance with the appropriate instance prompt
        """
        instance_prompt = self.get_instance_prompt_for_language(language)
        
        return AgentArgs(
            system_prompt=self.system_prompt,
            instance_prompt=instance_prompt,
            command_files=self.command_files,
            llm_name=self.llm_name,
            llm_base_url=self.llm_base_url,
            demo_file=self.demo_file,
            use_demo=self.use_demo,
            other_args=self.other_args
        )


##############################################################################
# Agent Class
##############################################################################
class Agent:
    """Agent handles the behavior of the model and how it interacts with the environment."""

    def __init__(self, name: str, args: AgentArgs, logger=None):
        self.name = name
        self.args = args
        self.trajectory_steps: List[TrajectoryStep] = []
        if logger is None:
            self.logger = get_logger(name)  # initialize logger from the agent name
        else:
            self.logger = logger
        self.llm_name = args.llm_name
        self.llm_base_url = args.llm_base_url
        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt
        self.command_files = args.command_files
        self.other_args = args.other_args or {}
        self.logger.info(f"Initialized Agent: {name} with LLM: {args.llm_name}")
        self.max_retries = 10000    # forever retry for model
        self.llm_timeout = self.other_args.get("timeout", 3000)

    def prepare_system_message(
        self, problem_statement: str, structure: str, command_docs: str, demo: str
    ) -> str:
        """Prepare the system prompt by filling in placeholders."""
        system_prompt = self.system_prompt_template.format(
            # problem_statement=problem_statement,
            # structure=structure,
            command_docs=command_docs,
            demo=demo,
        )
        return system_prompt

    def prepare_instance_prompt(
        self, agent_history: str, command_docs: str, steps_remaining: int
    ) -> str:
        """Prepare the instance prompt by filling in placeholders."""
        instance_prompt = self.instance_prompt_template.format(
            agent_history=agent_history,
            command_docs=command_docs,
        )
        # self.logger.info(isinstance(steps_remaining, int))
        # Add steps remaining message
        if steps_remaining > 0:
            stepcount_message = f"Steps Remaining: {steps_remaining}"
        else:
            stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
        instance_prompt += f"\n{stepcount_message}"
        self.logger.info(stepcount_message)  # Log the steps remaining message
        return instance_prompt

    def prepare_history_message(self, include_all_obs=False) -> str:
        """Prepare the agent's message history as a string."""
        history = ""
        for idx, step in enumerate(self.trajectory_steps):
            thought = step.thought
            action = step.action
            observation = step.observation
            # history += f'THOUGHT:\n```\n{thought}\n```\n'
            # history += f'ACTION:\n```\n{action}\n```\n'
            action_template = """
            {thought}
            ```
            {action}
            ```
            """
            history += action_template.format(thought=thought, action=action)
            if idx == len(self.trajectory_steps) - 1 or include_all_obs:
                history += f"\nOBSERVATION:\n```\n{observation}\n```\n"
            # add a separator
            history += "-" * 50 + "\n"
        return history

    def reset(self):
        """Reset the agent's trajectory."""
        self.trajectory_steps = []
        self.history = []

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the tokens for a list of messages using the litellm library.
        Adjust as needed depending on the model and library.
        """
        token_count = litellm.token_counter(model=self.llm_name, messages=messages)
        self.logger.info(f"Total tokens in conversation: {token_count}")
        return token_count

    def model_query(
        self, messages: List[Dict[str, str]], temperature: float = 0, top_p: float = 1.0, max_token_limit: int = 65536) -> Dict[str, Any]:
        """Query the LLM with the messages and measure execution time."""
        response = None
        retries = 0
        tools = None

        if self.use_fn_calling:
            if self.scaffold in ["r2egym", "r2egym-claude"]:
                tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
            elif self.scaffold == "aider_diff":
                tools = [aider_search_replace_edit]
            elif self.scaffold == "aider_whole":
                tools = [aider_whole_edit]
            elif self.scaffold in ["openhands", "sweagent", "multi_swe_bench"]:
                tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]
            if "vertex" not in self.llm_name.lower():
                self.logger.warning(f"using prompt caching for {self.llm_name}")
                # vertex is not supported yet: https://cloud.google.com/vertex-ai/generative-ai/docs/partner-models/claude-prompt-caching
                # litellm might need dev install with vertex: https://github.com/BerriAI/litellm/issues/6898
                # add prompt caching for anthropic
                tools[-1]["function"]["cache_control"] = {"type": "ephemeral"}
                breakpoints_remaining = 3  # remaining 1 for system/tool (above)
                for message in reversed(messages):
                    if message["role"] in ("user", "tool"):
                        if breakpoints_remaining > 0:
                            message["cache_control"] = {"type": "ephemeral"}
                            breakpoints_remaining -= 1
                        else:
                            break

        # Start timer
        start_time = time.time()
        # check if using locally hosted models
        using_local = "openai/" in self.llm_name or "hosted" in self.llm_name
        if using_local:
            litellm.api_key = None

        messages_ = copy.deepcopy(messages)
        total_tokens = self._count_tokens(messages_)
        if total_tokens > max_token_limit:
            logger.warning(f"Total tokens: {total_tokens} > {max_token_limit}")
            raise ValueError(f"Total tokens: {total_tokens} > {max_token_limit}")
        # query the model with retries
        while retries < self.max_retries:
            try:
                kwargs = {
                    # "tool_choice": "none",
                    "function_call": None,
                }
                if tools:
                    kwargs = {}
                if "o3" not in self.llm_name and "o4" not in self.llm_name:
                    kwargs["temperature"] = temperature
                    kwargs["top_p"] = top_p
                payload = dict(
                    model=self.llm_name,
                    tools=tools,
                    messages=messages_,
                    timeout=self.llm_timeout,
                    api_base=self.llm_base_url,
                    # max_tokens=max_tokens,
                    **kwargs,
                )
                response = litellm.completion(**payload)

                # validate the response
                assert len(response.choices) > 0, "No choices returned from LLM"
                assert response.usage is not None, "No usage returned from LLM"

                response_handler = OpenAIResponseHandler()
                response_handler.add_response({
                    "payload": payload,
                    "response": response.model_dump(),
                })
                self.logger.warning(f"Querying LLM complete")
                break
            except Exception as e:
                if "ContextWindowExceededError" in str(e):
                    raise

                if "generator error: Input text is too long" in str(e) and "fastdeploy" in str(e):
                    # raise ContextWindowExceededError
                    raise ValueError("ContextWindowExceededError from fastdeploy")

                if "Input is too long for requested model" in str(e):
                    # raise ContextWindowExceededError
                    raise ValueError("ContextWindowExceededError")
                
                if "Requested token count exceeds the model's maximum context length" in str(e):
                    raise ValueError("ContextWindowExceededError from fastdeploy")

                self.logger.error(f"LLM<{self.llm_base_url}> query failed @ {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    self.logger.error(f"Sleeping for 60 seconds for rate limiting ...")
                    time.sleep(60)
                else:
                    self.logger.error(f"Sleeping for 30 seconds ...")
                    time.sleep(30)
                if retries >= self.max_retries:
                    raise e

        # End timer, calculate total execution time, and include in response
        exec_time = time.time() - start_time
        return response, exec_time


    def run(
        self,
        env: "RepoEnv",  # env: RepoEnv
        use_fn_calling: bool = True,
        # step limits TODO: maybe add these limits in the agent args
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        # token limits
        max_token_limit: int = 65536,  # 64k tokens
        # time limits
        max_exec_time: int = 90,  # 5 mins per env execution
        max_total_time: int = 50000,  # 20 minutes overall agent run limit
        max_llm_time: int = 7200,  # 2 mins per LLM timeout (note this is per query exlcuding retries | not enforcing hard limit since llm might hit rate limits etc)
        # temperature
        temperature=0,
        top_p=1.0,
        # additional metadata e.g. for hints / additional inputs etc
        metadata: Optional[Dict[str, Any]] = {},
        scaffold: str = "r2egym",
    ):

        assert scaffold in SCAFFOLD_NAMES, f"Scaffold must be one of {SCAFFOLD_NAMES}, but received {scaffold}"
        self.scaffold = scaffold
        # get the start time
        start_time = time.time()
        self.llm_timeout = max_llm_time

        self.use_fn_calling = use_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")

        # Log the environment and agent
        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Reset the environment and the agent
        env.reset()
        env.add_commands(self.command_files)
        self.reset()

        # Prepare problem_statement and structure from the environment
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")

        if "fix_patch" in env.runtime.ds:
            gt_patch = env.runtime.ds["fix_patch"]
        elif "patch" in env.runtime.ds:
            gt_patch = env.runtime.ds["patch"]
        elif hasattr(env.runtime, "commit"):
            gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)
        else:
            gt_patch = ""

        # get system and instance prompts
        system_prompt = self.system_prompt_template
        if env.runtime.__class__.__name__ == "AiderDockerRuntime":
            user_prompt = self.instance_prompt_template.format(
                problem_statement=problem_statement,
                repo_path=env.runtime.ds['repo_path'],
                file_name=env.runtime.ds['file_name'],
                file_content=env.runtime.ds['initial_file_content'],
            )
        else:
            user_prompt = self.instance_prompt_template.format(
                problem_statement=problem_statement,
                gt_patch=gt_patch,
                working_dir=env.runtime.repo_path,
                # base_commit=env.runtime.ds['base_commit'],
                test_patch_hint=metadata.get("test_patch_hint", ""),
                candidate_patch=metadata.get("candidate_patch", ""),
                candidate_patch_correctness=(
                    "correct"
                    if metadata.get("candidate_patch_correctness", False)
                    else "incorrect"
                ),
            )
        self.logger.info(f"User Prompt: {user_prompt}")

        if self.args.use_demo:
            with open(self.args.demo_file, "r") as file:
                demo = file.read()
            user_prompt = f"{demo}\n\n{user_prompt}"
        self.logger.info(f"User Prompt with demo: {user_prompt}")

        # initialize the history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.history_to_save = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # initialize the parameters
        obs = None
        done = False
        step_count = 0
        total_time_traj = 0
        self.trajectory_steps: List[TrajectoryStep] = []

        # agent loop
        while not done:
            # Prepare the agent's message history
            # self.logger.info(isinstance(steps_remaining, int))
            # Add steps remaining message
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                stepcount_message = f"Steps Remaining: {steps_remaining}"
            else:
                stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
            self.history[-1][
                "content"
            ] += f"\n{stepcount_message}"  # postpend stepcount message
            self.logger.info(stepcount_message)
            self.history_to_save[-1]["content"] += f"\n{stepcount_message}"

            # Query the LLM
            messages = copy.deepcopy(self.history)
            try:
                response, llm_exec_time = self.model_query(messages, temperature, top_p, max_token_limit)
            except Exception as e:
                self.logger.error(f"Error querying LLM: {e}")
                self.logger.error(f"Error querying LLM: {traceback.format_exc()}")
                done = True
                exit_reason = "llm_query_error"
                if "ContextWindowExceededError" in str(e):
                    exit_reason = "token_limit"

                break
            
            # Log total tokens in the response
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

                prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                self.logger.warning(f"Prompt Token Details: {prompt_tokens_details}")
                self.logger.info(
                    f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                )
            else:

                completion_tokens = -1
                prompt_tokens = -1
                total_tokens = -1
                total_tokens =  self._count_tokens(messages)
                self.logger.warning(
                    "No token usage information available in the response."
                )
            # Parse the LLM response to get 'thought' and 'action'
            self.response = response  # for debugging
            assistant_message = response.choices[0].message.content
            self.logger.info(f"Assistant's message:\n{assistant_message}\n")
            self.logger.info(f"Assistant's response:\n{response}\n")

            if self.use_fn_calling:
                thought, action = load_parser(self.args.response_parser)(response)
            elif self.scaffold in ['mini_swe_agent']:
                thought, action = load_parser(self.args.response_parser)(assistant_message)
            else:
                thought, action = load_parser(self.args.response_parser)(assistant_message)

            action_str = None
            try:
                action_str = action.to_xml_string()
            except:
                done = True
                exit_reason = "finish error"
                break
            self.logger.info(f"THOUGHT:\n{thought}\n")
            self.logger.info(f"ACTION:\n{action.to_bashcmd()}\n")

            # Send the action to the environment
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
                # env.runtime.commit_after_step(step_count)
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")

            env_exec_time = info["total_time"]
            total_step_time = llm_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1  # Increment the step count

            if self.use_fn_calling:
                assistant_response = response.choices[0].message.dict()
                if assistant_response.get("tool_calls", None):
                    assistant_response["tool_calls"] = assistant_response["tool_calls"][
                        :1
                    ]  # only keep the first tool call
                self.history.append(assistant_response)
                self.history_to_save.append(assistant_response)
                # add tool response / user response to history
                try:
                    function_name = (
                        response.choices[0].message.tool_calls[0].function.name
                    )
                    function_id = response.choices[0].message.tool_calls[0].id
                    self.history.append(
                        {
                            "role": "tool",
                            "content": str(obs),
                            "name": function_name,
                            "tool_call_id": function_id,
                        }
                    )
                    self.history_to_save.append(
                        {
                            "role": "tool",
                            "content": str(obs),
                            "name": function_name,
                            "tool_call_id": function_id,
                        }
                    )
                    self.logger.warning("logging fn response as a tool call")
                    self.logger.warning(
                        f"number of fn calls: {len(response.choices[0].message.tool_calls)}"
                    )
                except Exception as e:
                    self.logger.error(f"Error logging tool response: {e}")
                    self.logger.warning("fallback: logging fn response as a tool call")
                    self.history.append({"role": "user", "content": str(obs)})
                    self.history_to_save.append({"role": "user", "content": str(obs)})
            else:
                self.logger.warning("logging fn response as a user message")
                assistant_message = f"{thought}\n\n{action.to_xml_string()}"
                # assistant_message = f"{thought}\n\n{original_xml_str}"
                self.history.append({"role": "assistant", "content": assistant_message})
                self.history.append({"role": "user", "content": str(obs)})
                self.history_to_save.append({"role": "assistant", "content": self.response.choices[0].message.content, "reasoning_content": thought, "action": action.to_xml_string()})
                self.history_to_save.append({"role": "user", "content": str(obs)})

            # Log the thought, action, and observation
            self.logger.info(f"OBSERVATION:\n{obs}\n")
            self.logger.info("-" * 50)

            # Check if the agent has reached limits or done
            # check if agent has finished naturally i.e. the agent uses the finish tool
            if done:
                if steps_remaining > 0:
                    self.logger.info(
                        f"Agent has finished naturally before step limit. current step count: {step_count}. max steps: {max_steps}."
                    )
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    self.logger.info(
                        f"Agent finised on reaching the maximum number of steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "max_step_limit"
                else:
                    self.logger.info(
                        f"Agent has finished after continuing past the max steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "agent_max_step_limit"
            # check for token limit
            elif total_tokens >= max_token_limit:
                self.logger.info(
                    f"Agent reached max tokens: {max_token_limit}. Current token count: {total_tokens}. Exiting."
                )
                exit_reason = "token_limit"
                done = True
            # check for absolute step limit | note that the max steps is just indicative but the absolute step limit is the hard limit
            elif step_count >= max_steps_absolute:
                self.logger.info(
                    f"Agent reached max steps: {max_steps_absolute}. Exiting."
                )
                exit_reason = "abs_step_limit"
                done = True

            elif total_time_traj >= max_total_time:
                self.logger.info(f"Agent reached max time: {max_total_time}. Exiting.")
                exit_reason = "traj_time_limit"
                done = True

            # Create a TrajectoryStep object and append to the list
            trajectory_step = TrajectoryStep(
                # key parts
                step_idx=step_count - 1,
                thought=thought,
                action=action.to_xml_string(),
                observation=str(obs),
                done=done,
                info=info,  # also store the info to be safe
                # tokens
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_total=total_tokens,
                # metadata (current step stats)
                llm_exec_time=llm_exec_time,
                env_exec_time=env_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
            )
            self.trajectory_steps.append(trajectory_step)

        # get the output patch
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD")
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD -- . ':(exclude)pyproject.toml'")
        # env.runtime.soft_git_reset()

        # compute output patch cummulatively from the start using git diff from the initial commit
        output_patch = env.runtime.get_patch()

        # Create a Trajectory object
        self.trajectory = Trajectory(
            trajectory_steps=[
                traj_step.model_dump() for traj_step in self.trajectory_steps
            ],
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            agent_args=asdict(self.args),
            env_args=asdict(env.args),
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason,  # reason for exiting. must be one of the [agent, max_step_limit, agent_max_step_limit, abs_step_limit, token_limit, traj_time_limit, llm_query_error]
            output_patch=output_patch,
        )

        self.logger.info(f"Agent completed in {time.time() - start_time} seconds.")
        # return self.trajectory
        return self.trajectory, self.history_to_save

    def run_with_parallel_function_call(
        self,
        env: "RepoEnv",  # env: RepoEnv
        use_fn_calling: bool = True,
        # step limits TODO: maybe add these limits in the agent args
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        # token limits
        max_token_limit: int = 65536,  # 64k tokens
        # time limits
        max_exec_time: int = 90,  # 5 mins per env execution
        max_total_time: int = 50000,  # 20 minutes overall agent run limit
        max_llm_time: int = 7200,  # 2 mins per LLM timeout (note this is per query exlcuding retries | not enforcing hard limit since llm might hit rate limits etc)
        # temperature
        temperature=0,
        top_p=1.0,
        # additional metadata e.g. for hints / additional inputs etc
        metadata: Optional[Dict[str, Any]] = {},
        scaffold: str = "r2egym",
    ):
        """
        Similar to run(), but supports parallel function calls.
        In this mode, the model can return multiple actions in a single response,
        and they will be executed sequentially, with results combined.
        """
        assert scaffold in SCAFFOLD_NAMES, f"Scaffold must be one of {SCAFFOLD_NAMES}, but received {scaffold}"
        self.scaffold = scaffold
        # get the start time
        start_time = time.time()
        self.llm_timeout = max_llm_time

        self.use_fn_calling = use_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")
        self.logger.warning(f"Running with parallel function call support")

        # Log the environment and agent
        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Reset the environment and the agent
        # env.reset()
        env.add_commands(self.command_files)
        self.reset()

        # Prepare problem_statement and structure from the environment
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")

        if "fix_patch" in env.runtime.ds:
            gt_patch = env.runtime.ds["fix_patch"]
        elif "patch" in env.runtime.ds:
            gt_patch = env.runtime.ds["patch"]
        elif hasattr(env.runtime, "commit"):
            gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)
        else:
            gt_patch = ""

        # get system and instance prompts
        system_prompt = self.system_prompt_template
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            gt_patch=gt_patch,
            working_dir=env.runtime.repo_path,
            # base_commit=env.runtime.ds['base_commit'],
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct"
                if metadata.get("candidate_patch_correctness", False)
                else "incorrect"
            ),
        )
        self.logger.info(f"User Prompt: {user_prompt}")

        if self.args.use_demo:
            with open(self.args.demo_file, "r") as file:
                demo = file.read()
            user_prompt = f"{demo}\n\n{user_prompt}"
        self.logger.info(f"User Prompt with demo: {user_prompt}")

        # initialize the history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        self.history_to_save = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        # initialize the parameters
        obs = None
        done = False
        step_count = 0
        total_time_traj = 0
        max_tokens = min(max_token_limit, 32768)
        self.trajectory_steps: List[TrajectoryStep] = []

        # agent loop
        while not done:
            # Prepare the agent's message history
            # Add steps remaining message
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                stepcount_message = f"Steps Remaining: {steps_remaining}"
            else:
                stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
            self.history[-1][
                "content"
            ] += f"\n{stepcount_message}"  # postpend stepcount message
            self.logger.info(stepcount_message)
            self.history_to_save[-1]["content"] += f"\n{stepcount_message}"

            # Query the LLM
            messages = copy.deepcopy(self.history)
            try:
                response, llm_exec_time = self.model_query(messages, temperature, top_p, max_tokens)
            except Exception as e:
                self.logger.error(f"Error querying LLM: {e}")
                self.logger.error(f"Error querying LLM: {traceback.format_exc()}")
                done = True
                exit_reason = "llm_query_error"
                if "ContextWindowExceededError" in str(e):
                    exit_reason = "token_limit"
                break
            
            finish_reason = getattr(response.choices[0], "finish_reason", "length")
            if finish_reason == "length":
                exit_reason = "token_limit"
                done = True
                break
            prompt_tokens = getattr(response.usage, "prompt_tokens", 1024 * 1024)
            if prompt_tokens >= max_token_limit:
                exit_reason = "token_limit"
                done = True
                break
            total_tokens = getattr(response.usage, "total_tokens", 1024 * 1024)
            if total_tokens >= max_token_limit:
                exit_reason = "token_limit"
                done = True
                break
            # Log total tokens in the response
            total_tokens = 0
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 1024 * 1024)

                prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                self.logger.warning(f"Prompt Token Details: {prompt_tokens_details}")
                self.logger.info(
                    f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                )
            else:
                completion_tokens = -1
                prompt_tokens = -1
                total_tokens = 1024 * 1024
                self.logger.error(
                    "No token usage information available in the response."
                )
            max_tokens = max(min(max_token_limit - total_tokens, 32768), 0)
            
            # Parse the LLM response to get 'thought' and 'actions' (multiple)
            self.response = response  # for debugging
            assistant_message = response.choices[0].message.content
            self.logger.info(f"Assistant's message:\n{assistant_message}\n")

            # Parse response to get multiple actions
            if self.use_fn_calling:
                thought, actions = load_parser(self.args.response_parser)(response)
            else:
                thought, actions = load_parser(self.args.response_parser)(assistant_message)

            self.logger.info(f"THOUGHT:\n{thought}\n")
            self.logger.info(f"Number of actions parsed: {len(actions)}")
            
            # Execute all actions sequentially and collect observations
            all_observations = []
            total_env_exec_time = 0
            
            for action_idx, action in enumerate(actions):
                self.logger.info(f"ACTION [{action_idx + 1}/{len(actions)}]:\n{action.to_bashcmd()}\n")

                # Send the action to the environment
                try:
                    obs, reward, action_done, info = env.step(action, timeout=max_exec_time)
                except Exception as e:
                    obs = str(e)
                    info = {"total_time": 0}
                    action_done = False
                    self.logger.error(f"Error during environment step: {obs}")

                all_observations.append({
                    "action_idx": action_idx,
                    "action": action.to_xml_string(),
                    "observation": str(obs),
                })
                total_env_exec_time += info.get("total_time", 0)
                
                self.logger.info(f"OBSERVATION [{action_idx + 1}/{len(actions)}]:\n{obs}\n")
                
                # Check if this action causes termination (e.g., finish/submit)
                if action_done:
                    done = True
                    break
            
            # Combine all observations into a single response
            combined_obs_parts = []
            for obs_item in all_observations:
                combined_obs_parts.append(
                    f"[Action {obs_item['action_idx'] + 1}]\n{obs_item['observation']}"
                )
            combined_obs = "\n\n---\n\n".join(combined_obs_parts)

            env_exec_time = total_env_exec_time
            total_step_time = llm_exec_time + env_exec_time
            total_time_traj += total_step_time
            step_count += 1  # Increment the step count

            # Update history
            if self.use_fn_calling:
                assistant_response = response.choices[0].message.dict()
                # Keep all tool calls for parallel execution
                self.history.append(assistant_response)
                self.history_to_save.append(assistant_response)
                
                # Add tool responses for each action
                try:
                    tool_calls = response.choices[0].message.tool_calls
                    if tool_calls:
                        for action_idx, (tool_call, obs_item) in enumerate(zip(tool_calls, all_observations)):
                            function_name = tool_call.function.name
                            function_id = tool_call.id
                            self.history.append(
                                {
                                    "role": "tool",
                                    "content": str(obs_item["observation"]),
                                    "name": function_name,
                                    "tool_call_id": function_id,
                                }
                            )
                            self.history_to_save.append(
                                {
                                    "role": "tool",
                                    "content": str(obs_item["observation"]),
                                    "name": function_name,
                                    "tool_call_id": function_id,
                                }
                            )
                        self.logger.warning(f"logging {len(tool_calls)} fn responses as tool calls")
                except Exception as e:
                    self.logger.error(f"Error logging tool response: {e}")
                    self.logger.warning("fallback: logging fn response as a user message")
                    self.history.append({"role": "user", "content": combined_obs})
                    self.history_to_save.append({"role": "user", "content": combined_obs})
            else:
                self.logger.warning("logging fn response as a user message")
                # Build combined action string for all actions
                combined_action_str = "\n".join([action.to_xml_string() for action in actions])
                assistant_message_content = self.response.choices[0].message.content
                self.history.append({"role": "assistant", "content": assistant_message_content})
                self.history.append({"role": "user", "content": combined_obs})
                self.history_to_save.append({
                    "role": "assistant", 
                    "content": assistant_message_content, 
                    "reasoning_content": thought, 
                    "actions": [action.to_xml_string() for action in actions]
                })
                self.history_to_save.append({"role": "user", "content": combined_obs})

            # Log the combined observation
            self.logger.info(f"COMBINED OBSERVATION:\n{combined_obs}\n")
            self.logger.info("-" * 50)

            # Check if the agent has reached limits or done
            if done:
                if steps_remaining > 0:
                    self.logger.info(
                        f"Agent has finished naturally before step limit. current step count: {step_count}. max steps: {max_steps}."
                    )
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    self.logger.info(
                        f"Agent finised on reaching the maximum number of steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "max_step_limit"
                else:
                    self.logger.info(
                        f"Agent has finished after continuing past the max steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "agent_max_step_limit"
            # check for token limit
            elif total_tokens >= max_token_limit:
                self.logger.info(
                    f"Agent reached max tokens: {max_token_limit}. Current token count: {total_tokens}. Exiting."
                )
                exit_reason = "token_limit"
                done = True
            # check for absolute step limit
            elif step_count >= max_steps_absolute:
                self.logger.info(
                    f"Agent reached max steps: {max_steps_absolute}. Exiting."
                )
                exit_reason = "abs_step_limit"
                done = True
            elif total_time_traj >= max_total_time:
                self.logger.info(f"Agent reached max time: {max_total_time}. Exiting.")
                exit_reason = "traj_time_limit"
                done = True

            # Create a TrajectoryStep object for this turn (may contain multiple actions)
            combined_action_str = "\n".join([action.to_xml_string() for action in actions])
            trajectory_step = TrajectoryStep(
                # key parts
                step_idx=step_count - 1,
                thought=thought,
                action=combined_action_str,
                observation=combined_obs,
                done=done,
                info={"total_time": total_env_exec_time, "num_actions": len(actions)},
                # tokens
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_total=total_tokens,
                # metadata (current step stats)
                llm_exec_time=llm_exec_time,
                env_exec_time=env_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
            )
            self.trajectory_steps.append(trajectory_step)

        # compute output patch cummulatively from the start using git diff from the initial commit
        output_patch = env.runtime.get_patch()

        # Create a Trajectory object
        self.trajectory = Trajectory(
            trajectory_steps=[
                traj_step.model_dump() for traj_step in self.trajectory_steps
            ],
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            agent_args=asdict(self.args),
            env_args=asdict(env.args),
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason,
            output_patch=output_patch,
        )

        self.logger.info(f"Agent completed in {time.time() - start_time} seconds.")
        return self.trajectory, self.history_to_save
