# cd /path/to/workspace/env_run/<user>/slime
export RLE_ENV=test
export RLE_TOOL_TYPE=swe-datagen
export RLE_API_KEY="${RLE_API_KEY:?set RLE_API_KEY}"
export RLE_SESSION_IDS_DIR=/tmp/rle_smoke

python3 <<'PY'
import asyncio
import json

from slime.agent.sandbox_rle import RLESandbox
from slime.agent.sandbox import install_r2egym_tools


image = json.loads(open('data/external_hf_v2_with_docker.jsonl').readline())['docker_image']


async def run(sb, label, cmd, *, timeout=30):
    print(f"\n=== {label} ===")
    print(f"$ {cmd}")
    rc, out, err = await sb.exec(cmd, timeout=timeout)
    print(f"[rc={rc}]")
    if out:
        print(out.rstrip())
    if err.strip():
        print("--- stderr ---")
        print(err.rstrip())


async def main():
    # setup=install_r2egym_tools injects the r2egym tool CLIs at session
    # creation time: shebang symlink + pip install chardet + add_commands
    # (str_replace_editor, execute_bash, search, finish, submit).
    async with RLESandbox(image, setup=install_r2egym_tools) as sb:
        print(f"session_id: {sb.sandbox_id[:16]}...")
        print(f"pod_view  : {sb.pod_view_url[:120]}")

        # 0. Confirm each tool CLI landed in /usr/local/bin.
        await run(
            sb,
            "installed tool binaries",
            "ls -l /usr/local/bin/str_replace_editor "
            "/usr/local/bin/execute_bash /usr/local/bin/search "
            "/usr/local/bin/finish /usr/local/bin/submit",
        )

        # 1. str_replace_editor: create a fresh file
        await run(
            sb,
            "str_replace_editor create",
            "rm -f /tmp/demo.py && "
            "str_replace_editor create --path /tmp/demo.py "
            "--file_text 'def hello():\n    return \"world\"\n'",
        )

        # 2. str_replace_editor: view what we just created
        await run(
            sb,
            "str_replace_editor view",
            "str_replace_editor view --path /tmp/demo.py",
        )

        # 3. str_replace_editor: replace 'world' with 'planet'
        await run(
            sb,
            "str_replace_editor str_replace",
            "str_replace_editor str_replace --path /tmp/demo.py "
            "--old_str world --new_str planet",
        )

        # 4. Re-view to verify the edit landed
        await run(
            sb,
            "str_replace_editor view (after str_replace)",
            "str_replace_editor view --path /tmp/demo.py",
        )

        # 5. str_replace_editor: insert a new first line
        await run(
            sb,
            "str_replace_editor insert",
            "str_replace_editor insert --path /tmp/demo.py "
            "--insert_line 0 --new_str 'import sys'",
        )

        # 6. Final view — should now show `import sys` + hello()->'planet'
        await run(
            sb,
            "str_replace_editor view (final)",
            "str_replace_editor view --path /tmp/demo.py",
        )

        # 7. execute_bash: dispatch a shell fragment through the tool
        await run(
            sb,
            "execute_bash: run the edited file",
            "execute_bash 'python /tmp/demo.py; "
            "python -c \"import demo; print(demo.hello())\" 2>&1 || true; "
            "echo done=$?'",
        )

        # 8. search: locate the 'hello' definition under /tmp
        await run(
            sb,
            "search: grep for hello under /tmp",
            "search --search_term hello --path /tmp",
        )


asyncio.run(main())
PY
