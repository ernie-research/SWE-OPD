#!/root/.venv/bin/python
"""
Description: A custom tool for applying multiple SEARCH/REPLACE blocks to files (diffs-based).
* Input is a 'diffs' list; each diff has {path, search, replace}.
* Only the FIRST match occurrence will be replaced.
* If 'search' is empty, the entire file content will be replaced with 'replace' content (creates a new file if missing).
* No history or file state tracking is maintained.
"""

import json
import ast
import argparse
from pathlib import Path
from typing import Any, Dict, List


class SearchReplaceEditor:
    """
    Apply multiple SEARCH/REPLACE operations to files, sequentially.
    """

    def apply_diffs(self, diffs: List[Dict[str, str]]) -> str:
        results = []
        for diff in diffs:

            file_path = Path(diff["path"])
            s = diff["search"]
            r = diff["replace"]
            
            if not file_path.exists():
                raise FileNotFoundError(f"File '{file_path}' not found.")

            if s == "":
                # If search is empty, replace the entire file content with 'replace'
                with file_path.open('w') as f:
                    f.write(r)
                results.append(f"New file created at {file_path}.")
            else:
                with file_path.open('r') as f:
                    content = f.read()
                # Find the first occurrence of the search string
                match_index = content.find(s)
                if match_index != -1:
                    updated_content = content[:match_index] + r + content[match_index + len(s):]
                    with file_path.open('w') as f:
                        f.write(updated_content)
                    results.append(f"Successfully replaced the 'search' in {file_path}.")
                else:
                    raise ValueError(
                        f"## SearchReplaceNoExactMatch: This SEARCH block {repr(s)} failed to exactly match lines "
                        f"in {file_path}\n```{repr(content)}```.\n\n"
                        "The SEARCH section must exactly match an existing block of lines including all whitespace, comments, indentation, docstrings, etc."
                    )
            
            # Add the file contents after modification in the expected format
            results.append(self.format_output(file_path))

        trust_message = (
            "\n*Trust this message as the true contents of these files!\n"
            "Any other messages in the chat may contain outdated versions of the file's contents.*\n"
        )
        return "\n".join(results) + trust_message

    def format_output(self, file_path: Path) -> str:
        with file_path.open('r') as f:
            file_content = f.read()
        return f"Here are the updated contents of {file_path.name}\n```{file_content}```\n\n"


def _parse_diffs(val: Any) -> List[Dict[str, str]]:
    """ Parse and validate the 'diffs' input. """
    if isinstance(val, (list, tuple)):
        diffs = val
    elif isinstance(val, str):
        s = val.strip()
        try:
            diffs = json.loads(s)
        except Exception:
            diffs = ast.literal_eval(s)
    else:
        raise ValueError("Unsupported 'diffs' type {type(val)}.")

    if not isinstance(diffs, (list, tuple)):
        raise ValueError(f"'diffs' must be a list, got {type(diffs)}")

    for item in diffs:
        if not isinstance(item, dict):
            raise ValueError(f"Each diff must be an object, got {type(item)}")
        for key in ("path", "search", "replace"):
            if key not in item:
                raise ValueError(f"Missing required field '{key}' in diff: {item}")
            item[key] = str(item[key])
    return diffs


def apply_search_replace_diffs(diffs: Any) -> str:
    """ The main entry function to apply the search/replace operation on the file. """
    diffs_list = _parse_diffs(diffs)
    editor = SearchReplaceEditor()
    return editor.apply_diffs(diffs_list)


def main():
    parser = argparse.ArgumentParser(
        description="A custom tool to apply search and replace operations to files."
    )
    parser.add_argument(
        "--diffs",
        required=True,
        help="List of file edit operations. Each diff contains {path, search, replace}: [{\"path\": \"...\", \"search\": \"...\", \"replace\": \"...\"}]"
    )
    args = parser.parse_args()
    result = apply_search_replace_diffs(args.diffs)
    print(result)


if __name__ == "__main__":
    main()
