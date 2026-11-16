#!/root/.venv/bin/python
"""
Description: Execute a bash command in the terminal, with Python version compatibility.

Parameters:
  --command (string, optional): The bash command to execute. For example: --command 'python my_script.py'. If not provided, will show help.
"""

import argparse
import subprocess
import sys
import io


def _ensure_utf8_stdio():
    """Force stdout/stderr to UTF-8 encoding. Docker containers with POSIX/C
    locale default to ASCII, causing UnicodeEncodeError on non-ASCII output."""
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name)
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
        elif hasattr(stream, 'buffer'):
            setattr(sys, stream_name, io.TextIOWrapper(
                stream.buffer, encoding='utf-8', errors='replace',
            ))

BLOCKED_BASH_COMMANDS = ["git", "ipython", "jupyter", "nohup"]


class CommandResult:
    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def run_command(cmd):
    try:
        # Try to use the new parameters (Python 3.7+)
        return subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore'
        )
    except TypeError:
        # Fallback for Python 3.5 and 3.6:
        # universal_newlines=True uses locale encoding (often ASCII), which
        # blows up on non-ASCII output. Use bytes mode + manual decode instead.
        if isinstance(cmd, str):
            cmd = cmd.encode('utf-8', errors='surrogateescape')
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout = result.stdout.decode('utf-8', errors='replace') if result.stdout else ''
        stderr = result.stderr.decode('utf-8', errors='replace') if result.stderr else ''
        return CommandResult(result.returncode, stdout, stderr)


def _fix_surrogate_argv():
    """Fix surrogate characters in sys.argv caused by POSIX/C locale in Docker.

    When filesystem encoding is ASCII, Python decodes non-ASCII argv bytes using
    surrogateescape, producing surrogate code points (U+DC80..U+DCFF). These are
    illegal in UTF-8 and cause UnicodeEncodeError in subprocess and print calls.
    Re-encode to recover original bytes, then decode as UTF-8."""
    import os
    cleaned = []
    for arg in sys.argv:
        try:
            arg.encode('utf-8')
            cleaned.append(arg)
        except UnicodeEncodeError:
            cleaned.append(
                os.fsencode(arg).decode('utf-8', errors='replace')
            )
    sys.argv = cleaned


def main():
    _ensure_utf8_stdio()
    _fix_surrogate_argv()

    parser = argparse.ArgumentParser(description="Execute a bash command.")
    parser.add_argument(
        "command",
        type=str,
        help="The command (and optional arguments) to execute. For example: 'python my_script.py'",
    )
    args = parser.parse_args()

    # Check if the command is blocked
    first_token = args.command.strip().split()[0]
    if first_token in BLOCKED_BASH_COMMANDS:
        print(
            "Bash command '{}' is not allowed. "
            "Please use a different command or tool.".format(first_token)
        )
        sys.exit(1)

    result = run_command(args.command)

    if result.returncode != 0:
        print("Error executing command:\n")
        print("[STDOUT]\n")
        print(result.stdout.strip(), "\n")
        print("[STDERR]\n")
        print(result.stderr.strip())
        sys.exit(result.returncode)

    print("[STDOUT]\n")
    print(result.stdout.strip(), "\n")
    print("[STDERR]\n")
    print(result.stderr.strip())


if __name__ == "__main__":
    main()
