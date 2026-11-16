#!/root/.venv/bin/python
"""
Description: A custom tool for writing the complete content of one or more files.
* Input is a 'files' list; each item has {path, file_content}.
* Each file entry specifies the target path and the complete text it should contain.
* No history or file state tracking is maintained.
"""

import json
import ast
import argparse
from pathlib import Path
from typing import Any, Dict, List


class WholeFileEditor:
    """
    Write the complete content of one or more files.
    """

    def apply_files(self, files: List[Dict[str, str]]) -> str:
        results = []
        for item in files:
            file_path = Path(item["path"])
            content = item["file_content"]

            # Ensure parent directories exist
            if file_path.parent and not file_path.parent.exists():
                file_path.parent.mkdir(parents=True, exist_ok=True)

            # Write full content
            with file_path.open("w", encoding="utf-8") as f:
                f.write(content)

            results.append(f"Successfully write the contents in {file_path}.")
            results.append(self.format_output(file_path))

        trust_message = (
            "\n*Trust this message as the true contents of these files!\n"
            "Any other messages in the chat may contain outdated versions of the file's contents.*\n"
        )
        return "\n".join(results) + trust_message

    def format_output(self, file_path: Path) -> str:
        with file_path.open("r", encoding="utf-8") as f:
            file_content = f.read()
        return f"Here are the updated contents of {file_path.name}\n```{file_content}```\n\n"


def _parse_files(val: Any) -> List[Dict[str, str]]:
    """ Parse and validate the 'files' input. """
    if isinstance(val, (list, tuple)):
        files = val
    elif isinstance(val, str):
        s = val.strip()
        try:
            files = json.loads(s)
        except Exception:
            files = ast.literal_eval(s)
    else:
        raise ValueError(f"Unsupported 'files' type {type(val)}.")

    if not isinstance(files, (list, tuple)):
        raise ValueError(f"'files' must be a list, got {type(files)}")

    for item in files:
        if not isinstance(item, dict):
            raise ValueError(f"Each file entry must be an object, got {type(item)}")
        for key in ("path", "file_content"):
            if key not in item:
                raise ValueError(f"Missing required field '{key}' in file entry: {item}")
            item[key] = str(item[key])
    return files


def apply_whole_edits(files: Any) -> str:
    """ The main entry function to write complete file contents. """
    files_list = _parse_files(files)
    editor = WholeFileEditor()
    return editor.apply_files(files_list)


def main():
    parser = argparse.ArgumentParser(
        description="A custom tool to write the complete content of one or more files."
    )
    parser.add_argument(
        "--files",
        required=True,
        help='List of file write operations. Each item contains {path, file_content}: '
             '[{"path": "...", "file_content": "..."}]'
    )
    args = parser.parse_args()
    result = apply_whole_edits(args.files)
    print(result)


if __name__ == "__main__":
    main()