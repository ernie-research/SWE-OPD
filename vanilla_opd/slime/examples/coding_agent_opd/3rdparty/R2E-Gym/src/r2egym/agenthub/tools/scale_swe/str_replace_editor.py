#!/root/.venv/bin/python
"""str_replace_editor -- single-file editor tool for container execution.

Output format aligned with openhands-aci OHEditor (as observed in Scale-SWE-Agent
evaluation trajectories). Designed to be copied into Docker containers and invoked
via CLI with no dependencies beyond stdlib + chardet.

Compatible with Python 3.5 - 3.11+.

Supported commands: view, create, str_replace, insert.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

# ---------------------------------------------------------------------------
# Constants (from openhands-aci editor/config.py & editor/prompts.py)
# ---------------------------------------------------------------------------

MAX_RESPONSE_LEN_CHAR = int(os.environ.get("MAX_OUTPUT_CHARS", "16000"))
SNIPPET_CONTEXT_WINDOW = 4

TEXT_FILE_CONTENT_TRUNCATED_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of this file '
    'has been shown to you. You should retry this tool after you have searched inside '
    'the file with `grep -n` in order to find the line numbers of what you are '
    'looking for.</NOTE>'
)

DIRECTORY_CONTENT_TRUNCATED_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of this directory '
    'has been shown to you. You should use `ls -la` instead to view large directories '
    'incrementally.</NOTE>'
)

CONTENT_TRUNCATED_NOTICE = (
    '<response clipped><NOTE>Due to the max output limit, only part of the full '
    'response has been shown to you.</NOTE>'
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def maybe_truncate(content, truncate_after=MAX_RESPONSE_LEN_CHAR,
                   truncate_notice=CONTENT_TRUNCATED_NOTICE):
    if not truncate_after or len(content) <= truncate_after:
        return content
    return content[:truncate_after] + truncate_notice


def _detect_encoding(path):
    """UTF-8 first, chardet fallback. Avoids chardet misdetecting UTF-8 files
    as Windows-1252/latin-1 which causes codec errors on non-ASCII content."""
    try:
        with open(str(path), 'r', encoding='utf-8') as f:
            f.read()
        return 'utf-8'
    except (UnicodeDecodeError, IOError):
        pass
    try:
        import chardet
        with open(str(path), 'rb') as f:
            sample = f.read(1024 * 1024)
        result = chardet.detect(sample)
        enc = result.get('encoding') or 'utf-8'
        if enc.lower() == 'ascii':
            enc = 'utf-8'
        return enc
    except Exception:
        return 'utf-8'


def _is_binary(path):
    """Simple binary detection via null-byte check."""
    try:
        with open(str(path), 'rb') as f:
            chunk = f.read(8192)
        return b'\x00' in chunk
    except Exception:
        return False


def _read_file(path):
    enc = _detect_encoding(path)
    try:
        with open(str(path), 'r', encoding=enc) as f:
            return f.read()
    except UnicodeDecodeError:
        with open(str(path), 'r', encoding='utf-8', errors='replace') as f:
            return f.read()


def _write_file(path, content):
    enc = _detect_encoding(path) if os.path.exists(str(path)) else 'utf-8'
    try:
        with open(str(path), 'w', encoding=enc) as f:
            f.write(content)
    except Exception as e:
        raise ToolError('Ran into {} while trying to write to {}'.format(e, path))


def _count_lines(path):
    enc = _detect_encoding(path)
    with open(str(path), 'r', encoding=enc) as f:
        return sum(1 for _ in f)


def _get_timeout():
    val = os.environ.get("STEP_TIMEOUT")
    if val is not None:
        try:
            return int(val)
        except ValueError:
            pass
    return 120


def _run_shell(cmd, truncate_notice=CONTENT_TRUNCATED_NOTICE):
    """Run shell command, return (returncode, stdout, stderr)."""
    if isinstance(cmd, str):
        cmd = cmd.encode('utf-8', errors='surrogateescape')
    try:
        proc = subprocess.Popen(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = proc.communicate(timeout=_get_timeout())
        stdout = stdout_bytes.decode('utf-8', errors='replace') if stdout_bytes else ''
        stderr = stderr_bytes.decode('utf-8', errors='replace') if stderr_bytes else ''
        return (
            proc.returncode or 0,
            maybe_truncate(stdout, truncate_notice=truncate_notice),
            maybe_truncate(stderr, truncate_notice=CONTENT_TRUNCATED_NOTICE),
        )
    except subprocess.TimeoutExpired:
        proc.kill()
        raise ToolError("Command '{}' timed out".format(cmd))


def _make_output(snippet_content, snippet_description, start_line=1):
    """Format file content with line numbers, mimicking `cat -n`."""
    snippet_content = maybe_truncate(
        snippet_content, truncate_notice=TEXT_FILE_CONTENT_TRUNCATED_NOTICE,
    )
    numbered = '\n'.join(
        '{:6}\t{}'.format(i + start_line, line)
        for i, line in enumerate(snippet_content.split('\n'))
    )
    return "Here's the result of running `cat -n` on {}:\n{}\n".format(
        snippet_description, numbered)


# ---------------------------------------------------------------------------
# Exceptions (from openhands-aci editor/exceptions.py)
# ---------------------------------------------------------------------------


class ToolError(Exception):
    def __init__(self, message):
        self.message = message
        super(ToolError, self).__init__(message)


class EditorToolParameterMissingError(ToolError):
    def __init__(self, command, parameter):
        super(EditorToolParameterMissingError, self).__init__(
            'Parameter `{}` is required for command: {}.'.format(parameter, command))


class EditorToolParameterInvalidError(ToolError):
    def __init__(self, parameter, value, hint=None):
        if hint:
            msg = 'Invalid `{}` parameter: {}. {}'.format(parameter, value, hint)
        else:
            msg = 'Invalid `{}` parameter: {}.'.format(parameter, value)
        super(EditorToolParameterInvalidError, self).__init__(msg)


# ---------------------------------------------------------------------------
# Editor
# ---------------------------------------------------------------------------


class StrReplaceEditor:
    """Single-file str_replace_editor aligned with openhands-aci OHEditor output."""

    def __init__(self):
        self._file_history = {}

    def __call__(self, command, path, file_text=None, view_range=None,
                 old_str=None, new_str=None, insert_line=None):
        _path = path if hasattr(path, 'is_absolute') else _Path(path)
        self._validate_path(command, _path)

        if command == 'view':
            return self._view(_path, view_range)
        elif command == 'create':
            return self._create(_path, file_text)
        elif command == 'str_replace':
            return self._str_replace(_path, old_str, new_str)
        elif command == 'insert':
            return self._insert(_path, insert_line, new_str)
        else:
            raise ToolError(
                'Unrecognized command {}. '
                'The allowed commands are: view, create, str_replace, insert'.format(command)
            )

    # -- Validation --------------------------------------------------------

    def _validate_path(self, command, path):
        if not path.is_absolute():
            raise EditorToolParameterInvalidError(
                'path', path,
                'The path should be an absolute path, starting with `/`.',
            )
        if command == 'create' and path.exists():
            raise EditorToolParameterInvalidError(
                'path', path,
                'File already exists at: {}. Cannot overwrite files using command `create`.'.format(path),
            )
        if command != 'create' and not path.exists():
            raise EditorToolParameterInvalidError(
                'path', path,
                'The path {} does not exist. Please provide a valid path.'.format(path),
            )
        if command != 'view' and path.is_dir():
            raise EditorToolParameterInvalidError(
                'path', path,
                'The path {} is a directory and only the `view` command can be used on directories.'.format(path),
            )

    # -- View --------------------------------------------------------------

    def _view(self, path, view_range):
        if path.is_dir():
            if view_range:
                raise EditorToolParameterInvalidError(
                    'view_range', view_range,
                    'The `view_range` parameter is not allowed when `path` points to a directory.',
                )
            return self._view_directory(path)
        return self._view_file(path, view_range)

    def _view_directory(self, path):
        _, hidden_stdout, _ = _run_shell(
            "find -L {} -mindepth 1 -maxdepth 1 -name '.*'".format(path)
        )
        hidden_count = len(hidden_stdout.strip().split('\n')) if hidden_stdout.strip() else 0

        _, stdout, stderr = _run_shell(
            "find -L {p} -maxdepth 2 -not \\( -path '{p}/\\.*' -o -path '{p}/*/\\.*' \\) | sort".format(p=path),
            truncate_notice=DIRECTORY_CONTENT_TRUNCATED_NOTICE,
        )
        if stderr:
            raise ToolError('Error listing directory {}: {}'.format(path, stderr))

        paths = stdout.strip().split('\n') if stdout.strip() else []
        formatted = []
        for p in paths:
            if os.path.isdir(p):
                formatted.append('{}/'.format(p))
            else:
                formatted.append(p)

        msg_parts = [
            "Here's the files and directories up to 2 levels deep in {}, excluding hidden items:\n".format(path)
            + '\n'.join(formatted)
        ]
        if hidden_count > 0:
            msg_parts.append(
                "\n{} hidden files/directories in this directory are excluded. "
                "You can use 'ls -la {}' to see them.".format(hidden_count, path)
            )
        return '\n'.join(msg_parts)

    def _view_file(self, path, view_range):
        if _is_binary(path):
            raise ToolError(
                'File {} appears to be binary and cannot be displayed.'.format(path)
            )

        num_lines = _count_lines(path)

        if not view_range:
            content = _read_file(path)
            return _make_output(content, str(path), 1)

        if len(view_range) != 2 or not all(isinstance(i, int) for i in view_range):
            raise EditorToolParameterInvalidError(
                'view_range', view_range,
                'It should be a list of two integers.',
            )

        start_line, end_line = view_range

        if start_line < 1 or start_line > num_lines:
            raise EditorToolParameterInvalidError(
                'view_range', view_range,
                'Its first element `{}` should be within the range of lines of the file: {}.'.format(
                    start_line, [1, num_lines]),
            )

        warning_message = None
        if end_line == -1:
            end_line = num_lines
        elif end_line > num_lines:
            warning_message = "We only show up to {} since there're only {} lines in this file.".format(
                num_lines, num_lines)
            end_line = num_lines

        if end_line < start_line:
            raise EditorToolParameterInvalidError(
                'view_range', view_range,
                'Its second element `{}` should be greater than or equal to the first element `{}`.'
                .format(end_line, start_line),
            )

        content = _read_file_range(path, start_line, end_line)
        output = _make_output('\n'.join(content.splitlines()), str(path), start_line)

        if warning_message:
            output = 'NOTE: {}\n{}'.format(warning_message, output)

        return output

    # -- Create ------------------------------------------------------------

    def _create(self, path, file_text):
        if file_text is None:
            raise EditorToolParameterMissingError('create', 'file_text')

        parent = os.path.dirname(str(path))
        if parent and not os.path.exists(parent):
            os.makedirs(parent, exist_ok=True)

        _write_file(path, file_text)
        self._file_history.setdefault(str(path), []).append('')
        return 'File created successfully at: {}'.format(path)

    # -- str_replace -------------------------------------------------------

    def _str_replace(self, path, old_str, new_str):
        if old_str is None:
            raise EditorToolParameterMissingError('str_replace', 'old_str')
        if new_str is not None and new_str == old_str:
            raise EditorToolParameterInvalidError(
                'new_str', new_str,
                'No replacement was performed. `new_str` and `old_str` must be different.',
            )

        new_str = new_str or ''
        file_content = _read_file(path)

        pattern = re.escape(old_str)
        occurrences = [
            (
                file_content.count('\n', 0, m.start()) + 1,
                m.group(),
                m.start(),
            )
            for m in re.finditer(pattern, file_content)
        ]

        if not occurrences:
            stripped_old = old_str.strip()
            stripped_new = new_str.strip()
            pattern = re.escape(stripped_old)
            occurrences = [
                (
                    file_content.count('\n', 0, m.start()) + 1,
                    m.group(),
                    m.start(),
                )
                for m in re.finditer(pattern, file_content)
            ]
            if not occurrences:
                raise ToolError(
                    'No replacement was performed, old_str `{}` did not appear verbatim in {}.'.format(
                        old_str, path)
                )
            old_str = stripped_old
            new_str = stripped_new

        if len(occurrences) > 1:
            line_numbers = sorted(set(line for line, _, _ in occurrences))
            raise ToolError(
                'No replacement was performed. Multiple occurrences of old_str '
                '`{}` in lines {}. Please ensure it is unique.'.format(old_str, line_numbers)
            )

        replacement_line, matched_text, idx = occurrences[0]
        new_file_content = (
            file_content[:idx] + new_str + file_content[idx + len(matched_text):]
        )

        self._file_history.setdefault(str(path), []).append(file_content)
        _write_file(path, new_file_content)

        start_line = max(0, replacement_line - 1 - SNIPPET_CONTEXT_WINDOW)
        end_line = replacement_line - 1 + SNIPPET_CONTEXT_WINDOW + new_str.count('\n') + 1
        snippet_lines = new_file_content.split('\n')[start_line:end_line]
        snippet = '\n'.join(snippet_lines)

        msg = 'The file {} has been edited. '.format(path)
        msg += _make_output(snippet, 'a snippet of {}'.format(path), start_line + 1)
        msg += 'Review the changes and make sure they are as expected. Edit the file again if necessary.'
        return msg

    # -- Insert ------------------------------------------------------------

    def _insert(self, path, insert_line, new_str):
        if insert_line is None:
            raise EditorToolParameterMissingError('insert', 'insert_line')
        if new_str is None:
            raise EditorToolParameterMissingError('insert', 'new_str')

        file_content = _read_file(path)
        lines = file_content.split('\n')
        num_lines = len(lines)

        if insert_line < 0 or insert_line > num_lines:
            raise EditorToolParameterInvalidError(
                'insert_line', insert_line,
                'It should be within the range of allowed values: {}'.format([0, num_lines]),
            )

        new_str_lines = new_str.split('\n')

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as tmp:
            enc = _detect_encoding(path)
            with open(str(path), 'r', encoding=enc) as f:
                for i, line in enumerate(f, 1):
                    if i > insert_line:
                        break
                    tmp.write(line)
            for line in new_str_lines:
                tmp.write(line + '\n')
            with open(str(path), 'r', encoding=enc) as f:
                for i, line in enumerate(f, 1):
                    if i <= insert_line:
                        continue
                    tmp.write(line)
            tmp_path = tmp.name

        self._file_history.setdefault(str(path), []).append(file_content)
        shutil.move(tmp_path, str(path))

        new_file_content = _read_file(path)
        new_file_lines = new_file_content.split('\n')
        start = max(0, insert_line - SNIPPET_CONTEXT_WINDOW)
        end = min(len(new_file_lines), insert_line + SNIPPET_CONTEXT_WINDOW + len(new_str_lines))
        snippet_lines = new_file_lines[start:end]
        snippet = '\n'.join(snippet_lines)

        msg = 'The file {} has been edited. '.format(path)
        msg += _make_output(snippet, 'a snippet of the edited file', max(1, start + 1))
        msg += (
            'Review the changes and make sure they are as expected '
            '(correct indentation, no duplicate lines, etc). Edit the file again if necessary.'
        )
        return msg


# ---------------------------------------------------------------------------
# Path compatibility shim for Python 3.5
# ---------------------------------------------------------------------------

try:
    from pathlib import Path as _Path
except ImportError:
    class _Path(object):
        """Minimal Path shim for Python < 3.4 (unlikely but safe)."""
        def __init__(self, p):
            self._p = str(p)
        def __str__(self):
            return self._p
        def is_absolute(self):
            return self._p.startswith('/')
        def exists(self):
            return os.path.exists(self._p)
        def is_dir(self):
            return os.path.isdir(self._p)
        @property
        def parent(self):
            return _Path(os.path.dirname(self._p))
else:
    _Path = _Path  # use real pathlib


# ---------------------------------------------------------------------------
# File range reader
# ---------------------------------------------------------------------------


def _read_file_range(path, start_line, end_line):
    enc = _detect_encoding(path)
    lines = []
    with open(str(path), 'r', encoding=enc) as f:
        for i, line in enumerate(f, 1):
            if i > end_line:
                break
            if i >= start_line:
                lines.append(line)
    return ''.join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_view_range(s):
    s = s.strip().strip('[]() ')
    parts = s.replace(',', ' ').split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Expected two numbers, got: {}'.format(s))
    return [int(parts[0]), int(parts[1])]


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
    parser = argparse.ArgumentParser(description='str_replace_editor tool')
    parser.add_argument('command', type=str, help='view, create, str_replace, insert')
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--file_text', type=str, default=None)
    parser.add_argument('--view_range', type=_parse_view_range, default=None)
    parser.add_argument('--old_str', type=str, default=None)
    parser.add_argument('--new_str', type=str, default=None)
    parser.add_argument('--insert_line', type=int, default=None)

    args = parser.parse_args()
    editor = StrReplaceEditor()

    try:
        result = editor(
            command=args.command,
            path=args.path,
            file_text=args.file_text,
            view_range=args.view_range,
            old_str=args.old_str,
            new_str=args.new_str,
            insert_line=args.insert_line,
        )
        print(result)
    except ToolError as e:
        print('ERROR:\n{}'.format(e.message))
    except Exception as e:
        print('ERROR:\n{}'.format(e))


if __name__ == '__main__':
    main()
