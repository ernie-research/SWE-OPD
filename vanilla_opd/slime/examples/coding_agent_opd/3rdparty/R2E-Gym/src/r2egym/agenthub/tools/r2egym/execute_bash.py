#!/root/.venv/bin/python
"""
Description: Execute a bash command in the terminal, with Python version compatibility.

Parameters:
  --cmd (string, required): The bash command to execute. For example: --cmd 'python my_script.py'
"""

import argparse
import os
import re
import subprocess
import sys


def _ensure_utf8_stdio():
    """Force stdout/stderr to UTF-8 encoding. Docker containers with POSIX/C
    locale default to ASCII, causing UnicodeEncodeError on non-ASCII output."""
    for stream_name in ('stdout', 'stderr'):
        stream = getattr(sys, stream_name)
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
        elif hasattr(stream, 'buffer'):
            import io
            setattr(sys, stream_name, io.TextIOWrapper(
                stream.buffer, encoding='utf-8', errors='replace',
            ))

DEFAULT_MAX_OUTPUT_CHARS = 16000


def _get_max_output_chars():
    env_val = os.environ.get("MAX_OUTPUT_CHARS")
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            pass
    return DEFAULT_MAX_OUTPUT_CHARS


def truncate_output(text, max_chars=None):
    """Truncate text keeping head and tail."""
    if max_chars is None:
        max_chars = _get_max_output_chars()
    if not text or len(text) <= max_chars:
        return text
    half = max_chars // 2
    removed = len(text) - max_chars
    return (
        text[:half]
        + "\n\n... [{} characters truncated] ...\n\n".format(removed)
        + text[-half:]
    )


_BLOCKED_PATTERNS = [
    # git: only block dangerous/information-leaking subcommands, allow normal ones
    # (git diff, git status, git log without --all, git show, git blame, etc.)
    r".*\bgit\s+log\b.*--all\b.*",
    r".*\bgit\s+verify-pack\b.*",
    r".*\bgit\s+fsck\b.*",
    r".*\bgit\s+cat-file\b.*",
    r".*\bgit\s+fetch\b.*",
    r".*\bgit\s+pull\b.*",
    r".*\bgit\s+push\b.*",
    r".*\bgit\s+clone\b.*",
    r".*\bgit\s+remote\b.*",
    # block external network access to prevent information leakage
    r".*\bapi\.github\.com\b.*",
    r".*\bgithub\.io\b.*",
    r".*\bgithubusercontent\.com\b.*",
    r".*\braw\.githubusercontent\.com\b.*",
    # block interactive / background tools
    r".*\bipython\b.*",
    r".*\bjupyter\b.*",
    r".*\bnohup\b.*",
]
_BLOCKED_RE = [re.compile(p) for p in _BLOCKED_PATTERNS]


class CommandResult:
    """兼容 subprocess.CompletedProcess 的结果类"""
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
        # 不能用 universal_newlines=True，因为它默认用 ASCII 解码，遇到非 ASCII 字符会报错
        # 改用二进制模式读取，然后手动用 UTF-8 解码
        if isinstance(cmd, str):
            cmd = cmd.encode('utf-8', errors='surrogateescape')
        result = subprocess.run(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        # 手动解码，忽略无法解码的字符
        stdout = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ''
        stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ''
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
        "--cmd",
        required=True,
        help="The command (and optional arguments) to execute. For example: --cmd 'python my_script.py'",
    )

    args = parser.parse_args()

    # Check if the command is blocked
    cmd_str = args.cmd.strip()
    for pattern in _BLOCKED_RE:
        if pattern.match(cmd_str):
            print(
                "Bash command '{}' not allowed, blocked pattern '{}'".format(
                    cmd_str, pattern.pattern)
            )
            sys.exit(1)

    result = run_command(args.cmd)

    stdout = truncate_output(result.stdout.strip()) if result.stdout else ""
    stderr = truncate_output(result.stderr.strip()) if result.stderr else ""

    if stdout:
        print(stdout)
    if stderr:
        if stdout:
            print()
        print("[STDERR]")
        print(stderr)

    sys.exit(result.returncode)


if __name__ == "__main__":
    main()