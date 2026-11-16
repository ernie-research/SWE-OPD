import re
import json
from typing import Tuple, List
from r2egym.agenthub.action import Action


def parse_xml_response(response_text: str) -> Tuple[str, Action]:
    """
    Parse the response from the LLM.
    Extracts:
    - thought: first thing in <think>...</think> block
    - action: the entire first <function=...></function> block
    Returns (thought, action).
    """
    # Regex to match (non-greedily) from `<think>` up to the first `</think>`
    pattern_thought = re.compile(r"(?s)(<think>.*?</think>)")
    pattern_action = re.compile(r"(?s)(<function=.*?</function>)")
    match_thought = pattern_thought.search(response_text)
    match_action = pattern_action.search(response_text)

    if match_thought:
        thought = match_thought.group(1)  # The entire <think>...</think> block
    else:
        thought = ""
    if match_action:
        action = match_action.group(1)  # The entire <function=...></function> block
    else:
        action = ""
    # Strip leading/trailing whitespace
    thought = thought.strip()
    action = action.strip()

    # convert action to Action object
    action = Action.from_string(action)

    return thought, action


def parse_xml_response_v2(response_text: str) -> Tuple[str, Action]:
    """
    Extracts:
    - thought: everything before the first <function=...> block
    - action: the entire first <function=...></function> block
    Returns (thought, action).
    """
    # Regex to match (non-greedily) from `<function=` up to the first `</function>`
    pattern = re.compile(r"(?s)(<function=.*?</function>)")
    match = pattern.search(response_text)

    if match:
        action = match.group(1)  # The entire <function=...></function> block
        thought = response_text[: match.start()]  # Everything before the block
    else:
        # If no match, treat entire text as "thought"
        thought = response_text
        action = ""

    # Strip leading/trailing whitespace
    thought = thought.strip()
    action = action.strip()

    # convert action to Action object
    action = Action.from_string(action)

    return thought, action


def parse_tool_call_response(response) -> Tuple[str, Action]:
    """
    Parse response from LLM function calling API.
    Extracts thought from message content and action from tool_calls.
    """
    thought = response.choices[0].message.content
    if not thought:
        thought = ""

    try:
        function_name = response.choices[0].message.tool_calls[0].function.name
        parameters = json.loads(
            response.choices[0].message.tool_calls[0].function.arguments
        )
        action = Action(function_name=function_name, parameters=parameters)
    except:
        action = Action(function_name="", parameters={})

    return thought, action


def parse_response_mini_swe_agent(response: str) -> Tuple[str, Action]:
    """
    Parse response for mini_swe_agent scaffold.
    Extracts:
    - thought: everything before the bash code block (including THOUGHT: prefix if present)
    - action: the bash command from the code block
    Returns (thought, action).
    Example:
    <format_example>
        Your reasoning and analysis here. Explain why you want to perform the action.

        ```bash
        your_command_here
        ```
    </format_example>
    """

    if not isinstance(response, str):
        response = ""
    # Extract thought - everything before the bash code block
    thought_pattern = re.compile(r"(?s)(.*?)(?=```bash)", re.MULTILINE)
    thought_match = thought_pattern.search(response)

    if thought_match:
        thought_raw = thought_match.group(1).strip()
        # Remove THOUGHT: prefix if present
        if thought_raw.startswith("THOUGHT:"):
            thought = thought_raw[len("THOUGHT:"):].strip()
        else:
            thought = thought_raw
    else:
        # If no code block found, treat everything as thought
        thought = response.strip()
        if thought.startswith("THOUGHT:"):
            thought = thought[len("THOUGHT:"):].strip()

    # Extract bash command from code block
    bash_pattern = re.compile(r"```bash\s*\n(.*?)\n```", re.DOTALL)
    bash_match = bash_pattern.search(response)

    if bash_match:
        bash_command = bash_match.group(1).strip()
        # Create a simple Action object for bash command
        action = Action(function_name="execute_bash", parameters={"cmd": bash_command})
    else:
        # No bash block found, return empty action
        action = ""

    return thought, action
    

def parse_parallel_xml_response(response_text: str) -> Tuple[str, List[Action]]:
    """
    Extracts multiple actions from the response text in parallel function call format.
    The format is:
        <functions>
        <function=example_function_name_1>
        <parameter=example_parameter_1>value_1</parameter>
        </function>
        <function=example_function_name_2>
        ...
        </function>
        </functions>
    
    Returns (thought, list_of_actions).
    """
    # Try to find the <functions>...</functions> block
    functions_pattern = re.compile(r"(?s)<functions>(.*?)</functions>")
    functions_match = functions_pattern.search(response_text)
    
    if functions_match:
        functions_block = functions_match.group(1)
        thought = response_text[:functions_match.start()].strip()
    else:
        # Fallback: try to parse individual <function=...></function> blocks
        thought = response_text
        functions_block = response_text
    
    # Extract all <function=...></function> blocks from the functions block
    function_pattern = re.compile(r"(?s)(<function=.*?</function>)")
    function_matches = function_pattern.findall(functions_block)
    
    actions = []
    for func_str in function_matches:
        action = Action.from_string(func_str.strip())
        if action.function_name:  # Only add valid actions
            actions.append(action)
    
    # If no actions found, return empty list
    if not actions:
        actions = [Action(function_name="", parameters={})]
    
    return thought, actions

def parse_parallel_tool_call_response(response) -> Tuple[str, List[Action]]:
    """
    Extracts multiple actions from the response text in parallel function call format.
    The format is:
        <functions>
        <function=example_function_name_1>
        <parameter=example_parameter_1>value_1</parameter>
        </function>
        <function=example_function_name_2>
        ...
        </function>
        </functions>
    
    Returns (thought, list_of_actions).
    """
    pass


def load_parser(parser_name: str = "parse_xml_response"):
    """
    Load a parser function by name.

    Args:
        parser_name: Name of the parser to load. Available options:
            - "parse_xml_response": Parses XML-style responses with <think> and <function> tags
            - "parse_tool_call_response": Parses responses from LLM function calling API (tool_calls format)
            - "parse_response_mini_swe_agent": Parses mini_swe_agent scaffold format (thought + bash code block)

    Returns:
        callable: The requested parser function that takes a response string/object
                  and returns a tuple of (thought: str, action: Action)

    Raises:
        ValueError: If the parser_name is not found in the available parsers

    Example:
        >>> parser = load_parser("parse_xml_response")
        >>> thought, action = parser(response_text)
    """
    parsers = {
        "parse_xml_response": parse_xml_response,
        "parse_xml_response_v2": parse_xml_response_v2,
        "parse_tool_call_response": parse_tool_call_response,
        "parse_response_mini_swe_agent": parse_response_mini_swe_agent,
        "parse_parallel_xml_response": parse_parallel_xml_response,
        "parse_parallel_tool_call_response": parse_parallel_tool_call_response
    }

    if parser_name not in parsers:
        raise ValueError(f"Unknown parser: {parser_name}. Available parsers: {list(parsers.keys())}")

    return parsers[parser_name]
