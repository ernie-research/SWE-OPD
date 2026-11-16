#!/root/.venv/bin/python
"""execute_bash -- run a shell command with structured metadata output.

Output format aligned with Scale-SWE-Agent evaluation trajectories.
Designed to be copied into Docker containers and invoked via CLI.
Compatible with Python 3.5 - 3.11+.

Timeout is read from STEP_TIMEOUT env var (set by docker.py runtime),
falling back to DEFAULT_TIMEOUT if unset.

Output structure:
  Normal (with output):  {output}\n\n[CWD]\n[Elapsed]\n[Exit code]
  Normal (no output):    [CWD]\n[Elapsed]\n[Exit code]
  Timeout (with output): {output}\n\n[CWD]\n[Elapsed]\n[Timed out]\n[Cleanup]\n[Reminder]
  Timeout (no output):   [CWD]\n[Elapsed]\n[Timed out]\n[Cleanup]\n[Reminder]
  Blocklist hit:         Bash command '{cmd}' not allowed, blocked pattern '{pat}'
"""

import os
import re
import signal
import sys
import time
import subprocess


# ---------------------------------------------------------------------------
# Blocklist (AweAgent SearchSWEAgent, enable_search=false mode)
# ---------------------------------------------------------------------------

BLOCKLIST_PATTERNS = [
    r".*git log.*--all.*",
    r".*git verify-pack.*",
    r".*git fsck.*",
    r".*git cat-file.*",
    r".*git fetch.*",
    r".*git pull.*",
    r".*git clone.*",
    r".*api\.github\.com.*",
    r".*github\.io.*",
    r".*githubusercontent.*",
]

DEFAULT_TIMEOUT = 900  # reference R2E-Gym-benchmark value; only a fallback (see _get_timeout)
DEFAULT_MAX_OUTPUT_CHARS = 16000
SIGKILL_WAIT_SECONDS = 3


# ---------------------------------------------------------------------------
# Output truncation (head + tail, character-based)
# ---------------------------------------------------------------------------

def truncate_output(text, max_chars):
    """Truncate *text* to *max_chars* keeping head and tail."""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    removed = len(text) - max_chars
    return (
        text[:half]
        + "\n\n... [{0} characters truncated] ...\n\n".format(removed)
        + text[-half:]
    )


# ---------------------------------------------------------------------------
# Blocklist check
# ---------------------------------------------------------------------------

def check_blocklist(command):
    """Return the matched blocklist pattern, or None."""
    for pattern in BLOCKLIST_PATTERNS:
        if re.match(pattern, command, re.DOTALL):
            return pattern
    return None


# ---------------------------------------------------------------------------
# Command execution (Python 3.5+ compatible)
# ---------------------------------------------------------------------------

def run_command(command, timeout):
    """Execute *command* in a sub-shell.

    Returns (stdout_text, returncode, timed_out, cleanup_status).
    Always uses bytes mode + manual decode for Python 3.5 compatibility.
    stderr is merged into stdout (subprocess.STDOUT).
    """
    # Encode to bytes before passing to subprocess to bypass filesystem
    # encoding (ASCII in POSIX/C locale containers with Python 3.6).
    # subprocess with shell=True and bytes arg creates [b'/bin/sh', b'-c', cmd_bytes],
    # skipping os.fsencode() which would fail on non-ASCII characters.
    if isinstance(command, str):
        command = command.encode('utf-8', errors='surrogateescape')
    proc = None
    try:
        proc = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        stdout, _ = proc.communicate(timeout=timeout)
        stdout = stdout.decode("utf-8", errors="replace") if stdout else ""
        return stdout, proc.returncode, False, None
    except subprocess.TimeoutExpired as exc:
        stdout = exc.output.decode("utf-8", errors="replace") if exc.output else ""
        cleanup_status = "sigkill_unconfirmed"
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
                try:
                    proc.wait(timeout=SIGKILL_WAIT_SECONDS)
                    cleanup_status = "sigkill_confirmed"
                except subprocess.TimeoutExpired:
                    cleanup_status = "sigkill_unconfirmed"
            except ProcessLookupError:
                cleanup_status = "process_group_gone"
            except OSError:
                cleanup_status = "sigkill_failed"
            finally:
                if proc.stdout is not None:
                    try:
                        proc.stdout.close()
                    except Exception:
                        pass
        return stdout, None, True, cleanup_status


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def format_output(raw_output, cwd, elapsed, returncode, timed_out, timeout,
                  cleanup_status=None,
                  max_chars=DEFAULT_MAX_OUTPUT_CHARS):
    """Build the final observation string matching the spec."""
    parts = []

    output = raw_output.rstrip("\n") if raw_output else ""
    output = truncate_output(output, max_chars)

    if output:
        parts.append(output)
        parts.append("")  # blank line separating output from metadata

    parts.append("[Current working directory: {0}]".format(cwd))
    parts.append("[Execution time: {0}]".format(elapsed))

    if timed_out:
        parts.append("[Command timed out after {0} seconds]".format(timeout))
        if cleanup_status == "sigkill_confirmed":
            parts.append("[Cleanup: SIGKILL was sent to this command's process group, and the command shell exited.]")
        elif cleanup_status == "process_group_gone":
            parts.append("[Cleanup: the command process group was already gone when cleanup was attempted.]")
        elif cleanup_status == "sigkill_failed":
            parts.append("[Cleanup: the tool attempted to send SIGKILL to this command's process group, but cleanup failed.]")
        else:
            parts.append("[Cleanup: SIGKILL was sent to this command's process group, but command exit was not confirmed within {0} seconds. Detached, daemonized, or uninterruptible child processes may still remain.]".format(SIGKILL_WAIT_SECONDS))
        parts.append("[Reminder: This command exceeded the hard timeout. It may be a non-returning command such as a foreground server, watcher, interactive command, or tail -f, or it may be a normally returning command that is too broad or slow for the available time, such as a full build, full test suite, or cold compile. Do not repeat the same command unchanged. Use commands that return promptly: narrow the test/build scope, run targeted checks, use fail-fast limits when appropriate, and if you need a long-running service, start it in the background with output redirected so the startup command returns promptly.]")
    else:
        parts.append("[Command finished with exit code {0}]".format(returncode))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

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


def _fix_surrogate_argv():
    """Fix surrogate characters in sys.argv caused by POSIX/C locale in Docker.

    When filesystem encoding is ASCII, Python decodes non-ASCII argv bytes using
    surrogateescape, producing surrogate code points (U+DC80..U+DCFF). These are
    illegal in UTF-8 and cause UnicodeEncodeError in subprocess and print calls.
    Re-encode to recover original bytes, then decode as UTF-8."""
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
    if len(sys.argv) < 2 or not sys.argv[1]:
        print("Usage: execute_bash <command> [--timeout N]")
        return

    command = sys.argv[1]

    # Timeout priority: STEP_TIMEOUT env var > DEFAULT_TIMEOUT.
    # A per-command --timeout may only shorten this step-level hard limit.
    step_timeout = DEFAULT_TIMEOUT
    env_timeout = os.environ.get("STEP_TIMEOUT")
    if env_timeout is not None:
        try:
            step_timeout = int(env_timeout)
        except ValueError:
            pass
    timeout = step_timeout

    # Parse optional --timeout (model can override via <parameter=timeout>)
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--timeout" and i + 1 < len(sys.argv):
            try:
                requested_timeout = int(sys.argv[i + 1])
                timeout = min(requested_timeout, step_timeout)
            except ValueError:
                pass
            i += 2
        else:
            i += 1

    # --- Blocklist ---
    blocked = check_blocklist(command)
    if blocked:
        print("Bash command '{0}' not allowed, blocked pattern '{1}'".format(
            command, blocked))
        return

    # --- Max output chars ---
    max_chars = DEFAULT_MAX_OUTPUT_CHARS
    env_max = os.environ.get("MAX_OUTPUT_CHARS")
    if env_max is not None:
        try:
            max_chars = int(env_max)
        except ValueError:
            pass

    # --- Execute ---
    cwd = os.getcwd()
    start = time.time()
    raw_output, returncode, timed_out, cleanup_status = run_command(command, timeout)
    elapsed = "{0:.2f}s".format(time.time() - start)

    # --- Format and print ---
    observation = format_output(raw_output, cwd, elapsed, returncode, timed_out, timeout,
                                cleanup_status=cleanup_status,
                                max_chars=max_chars)
    print(observation)


if __name__ == "__main__":
    main()
