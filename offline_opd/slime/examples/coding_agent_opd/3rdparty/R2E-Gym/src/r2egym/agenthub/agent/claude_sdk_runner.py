"""
Standalone script to run ClaudeSDKClient inside a Docker container.
This script is copied to the container and executed there.

Usage:
    python claude_sdk_runner.py <config_json_file>

The config file should contain:
{
    "prompt": "...",
    "model": "...",
    "api_key": "...",
    "base_url": "...",
    "max_steps": 50,
    "output_file": "/tmp/claude_output.json"
}
"""

import sys
import json
import time
import asyncio
import traceback
from typing import Any, Dict

try:
    from claude_agent_sdk import (
        query,
        ClaudeAgentOptions,
        AssistantMessage,
        UserMessage,
        SystemMessage,
        ResultMessage,
        ToolUseBlock,
        ToolResultBlock,
        TextBlock,
    )
except ImportError:
    print("ERROR: claude_agent_sdk not installed", file=sys.stderr)
    sys.exit(1)


def _serialize_text_block(block: TextBlock) -> Dict[str, Any]:
    return {"type": "text", "text": block.text}

def _serialize_tool_use_block(block: ToolUseBlock) -> Dict[str, Any]:
    return {
        "type": "tool_use",
        "id": getattr(block, "id", None),
        "name": getattr(block, "name", None),
        "input": getattr(block, "input", None),
    }

def _serialize_tool_result_block(block: ToolResultBlock) -> Dict[str, Any]:

    if isinstance(block.content, str):
        content_str = block.content
    elif isinstance(block.content, list):
        content_str = "\n".join(
            str(c.text if hasattr(c, "text") else c)
            for c in block.content
        )
    else:
        content_str = str(block.content)

    return {
        "type": "tool_result",
        "tool_use_id": block.tool_use_id,
        "content": content_str,
        "is_error": block.is_error,
    }

def handle_result_message(message, stats):
    def format_duration_ms(ms: int) -> str:
        if ms is None:
            return "0ms"

        seconds = ms / 1000
        minutes = seconds / 60

        return f"{ms}ms ({seconds:.2f}s, {minutes:.2f}m)"
    
    """ Handle the final ResultMessage from Claude and fill the stats dictionary. """
    print(f"[CLAUDE_COMPLETE] Duration: {format_duration_ms(message.duration_ms)}, Cost: ${message.total_cost_usd}",flush=True)
    print(f"[CLAUDE_COMPLETE] ResultMessage - session_id: {message.session_id}, is_error: {message.is_error}", flush=True,)
    print(f"[CLAUDE_TEXT] Result: {str(message.result)[:1000]}", flush=True)

    msg_data = {
        "type": "ResultMessage",
        "timestamp": time.time(),
        "session_id": message.session_id,
        "is_error": message.is_error,
        "result": str(message.result),
        "duration_ms": message.duration_ms,
        "total_cost_usd": message.total_cost_usd,
    }

    if message.usage:
        usage = message.usage
        print(f"[CLAUDE_COMPLETE] ResultMessage usage = {usage}", flush=True)
        stats["total_prompt_tokens"] = usage.get("input_tokens", 0)
        stats["total_completion_tokens"] = usage.get("output_tokens", 0)

    stats["total_tokens"] = stats["total_prompt_tokens"] + stats["total_completion_tokens"]
    stats["duration_ms"] = message.duration_ms
    stats["cost_usd"] = message.total_cost_usd

    if message.is_error:
        print(f"ERROR: {message.result}", flush=True)

    return msg_data


async def run_claude_code(config: Dict[str, Any]) -> Dict[str, Any]:
    """Run Claude Code SDK and collect all messages for JSON output."""

    system = config["system"]
    prompt = config["prompt"]
    model = config["model"]
    api_key = config.get("api_key", "")
    base_url = config.get("base_url", "")
    max_steps = config.get("max_steps", 50)

    print(f"[CLAUDE_CONFIG] Config parsed - model: {model}, base_url: {base_url}, max_steps: {max_steps}", flush=True)
    print(f"[CLAUDE_CONFIG] Prompt length: {len(prompt)} chars", flush=True)

    # Ensure claude binary is in PATH for the SDK
    import os as os_module
    current_path = os_module.environ.get('PATH', '')
    claude_paths = ['/usr/local/bin', '/root/node-v24.11.0-linux-x64/bin']
    prefix = ":".join(claude_paths)
    os_module.environ['PATH'] = f"{prefix}:{current_path}" if current_path else prefix
    print(f"[CLAUDE_DEBUG] Updated PATH: {os_module.environ['PATH']}", flush=True)

    # === 禁用代理（访问内网 API）===
    proxy_keys = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'no_proxy', 'NO_PROXY']
    removed_proxies = {}
    for key in proxy_keys:
        if key in os_module.environ:
            removed_proxies[key] = os_module.environ[key]
            del os_module.environ[key]
    if removed_proxies:
        print(f"[CLAUDE_CONFIG] Disabled proxies for internal API access: {list(removed_proxies.keys())}", flush=True)
        print(f"[CLAUDE_CONFIG] Original http_proxy: {removed_proxies.get('http_proxy', 'N/A')}", flush=True)

    # === 从 ~/.claude/settings.json 加载 env（包含 ANTHROPIC_MODEL、ANTHROPIC_DEFAULT_*_MODEL） ===
    settings_env: Dict[str, str] = {}
    settings_path = os_module.path.expanduser("~/.claude/settings.json")
    try:
        with open(settings_path, "r", encoding="utf-8") as f:
            settings_data = json.load(f)
        if isinstance(settings_data, dict):
            settings_env = settings_data.get("env", {}) or {}
        print(f"[CLAUDE_CONFIG] Loaded settings env from {settings_path}: {list(settings_env.keys())}", flush=True)
    except Exception as e:
        print(f"[CLAUDE_CONFIG] Failed to load settings.json: {e}", flush=True)
    
    # === 合并 env：settings.json 的 env + 我们显式传入的内容 ===
    merged_env = dict(settings_env)
    merged_env["ANTHROPIC_BASE_URL"] = base_url  # 覆盖 settings.json 中的同名项
    merged_env["ANTHROPIC_API_KEY"] = api_key
    merged_env["IS_SANDBOX"] = "1"
    # === 所有 env value 统一转成字符串，避免 int 等类型导致报错 ===
    safe_env: Dict[str, str] = {}
    for k, v in merged_env.items():
        if v is None:
            continue
        safe_env[str(k)] = str(v)

    # Check if thinking is enabled from env
    max_thinking_tokens = None
    thinking_enabled = safe_env.get("CLAUDE_THINKING_ENABLED", "").lower() == "true"
    if thinking_enabled:
        thinking_budget = safe_env.get("CLAUDE_THINKING_BUDGET_TOKENS", "10000")
        try:
            max_thinking_tokens = int(thinking_budget)
            print(f"[CLAUDE_CONFIG] Extended thinking enabled with {max_thinking_tokens} tokens", flush=True)
        except ValueError:
            print(f"[CLAUDE_CONFIG] Invalid CLAUDE_THINKING_BUDGET_TOKENS: {thinking_budget}, using default", flush=True)
            max_thinking_tokens = 10000

    options = ClaudeAgentOptions(
        system_prompt=system,
        model=model,
        env=safe_env,
        permission_mode="bypassPermissions",
        disallowed_tools=["WebSearch", "WebFetch"],
        max_thinking_tokens=max_thinking_tokens,
    )

    stats = {
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "duration_ms": 0,
        "cost_usd": 0,
    }

    exit_reason = "agent"
    step_count = 0
    message_count = 0

    # prompt = "can you explain to me the structure of this repo?"  # for debug
    all_messages = [{
        "type": "UserMessage",
        "timestamp": time.time(),
        "is_error": False,
        "content": [
            {
                "type": "text",
                "text": prompt,
            }
        ],
    }]

    try:
        print(f"[CLAUDE_DEBUG] Starting query iteration...", flush=True)

        async for message in query(prompt=prompt, options=options):
            message_count += 1
            msg_type = type(message).__name__
            print(f"[CLAUDE_DEBUG] Message #{message_count} ({msg_type})", flush=True)

            msg_data = {
                "type": msg_type,
                "timestamp": time.time(),
            }

            if isinstance(message, AssistantMessage):
                print(f"[CLAUDE_DEBUG] AssistantMessage blocks={len(message.content)}, types={[type(b).__name__ for b in message.content]}", flush=True)
                
                content_blocks = []
                for block in message.content:
                    if isinstance(block, TextBlock):
                        text_data = _serialize_text_block(block)
                        print(f"[CLAUDE_TEXT] {json.dumps(text_data, ensure_ascii=False)[:600]}", flush=True)
                        content_blocks.append(text_data)
                    elif isinstance(block, ToolUseBlock):
                        tool_use_data = _serialize_tool_use_block(block)
                        print(f"[CLAUDE_TOOL_USE] {json.dumps(tool_use_data, ensure_ascii=False)[:600]}", flush=True,)
                        content_blocks.append(tool_use_data)
                        step_count += 1  # 每个 tool_use 计一步
                msg_data["content"] = content_blocks

            elif isinstance(message, UserMessage):
                content_blocks = []
                content = getattr(message, "content", None)

                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolResultBlock):
                            tool_result_data = _serialize_tool_result_block(block)
                            print(f"[CLAUDE_TOOL_RESULT] {json.dumps(tool_result_data, ensure_ascii=False)[:600]}", flush=True)
                            content_blocks.append(tool_result_data)
                        elif isinstance(block, TextBlock):
                            text_data = _serialize_text_block(block)
                            print(f"[CLAUDE_TEXT] {json.dumps(text_data, ensure_ascii=False)[:600]}", flush=True)
                            content_blocks.append(text_data)
                else:
                    # fallback: 把整体当成一段文本
                    text = str(content)
                    print(f"[WARNING] UserMessage text: {text[:600]}", flush=True)
                    content_blocks.append({"type": "text", "text": text})
                
                msg_data["content"] = content_blocks

            elif isinstance(message, SystemMessage):
                text = str(getattr(message, "content", message))
                print(f"[CLAUDE_SYSTEM] SystemMessage text: {text}", flush=True)
                msg_data["content"] = [{"type": "text", "text": text}]

            elif isinstance(message, ResultMessage):
                msg_data = handle_result_message(message, stats)
                all_messages.append(msg_data)
                exit_reason = "llm_query_error" if message.is_error else "agent"
                break

            all_messages.append(msg_data)

            if step_count >= max_steps:
                exit_reason = "abs_step_limit"
                print(f"[WARNING] Hit step limit: {max_steps}", flush=True)
                break

        print(f"[CLAUDE_DEBUG] Finished receiving messages. Total messages: {message_count}", flush=True)
        
    except Exception as e:
        print(f"ERROR: Exception caught: {type(e).__name__}: {e}", flush=True)

        traceback.print_exc()
        exit_reason = "llm_query_error"
        all_messages.append({
            "type": "Error",
            "error": str(e),
            "timestamp": time.time(),
        })

    print(f"[CLAUDE_DEBUG] Returning result - exit_reason: {exit_reason}, messages: {len(all_messages)}, stats: {stats}", flush=True)

    return {
        "messages": all_messages,
        "exit_reason": exit_reason,
        "stats": stats,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python claude_sdk_runner.py <config_json_file>", file=sys.stderr)
        sys.exit(1)

    config_file = sys.argv[1]
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load config file: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[CLAUDE_START] Model: {config.get('model', 'unknown')}, Max steps: {config.get('max_steps', 50)}", flush=True)
    print(f"[CLAUDE_CONFIG] {config.get('api_key', '')[:10]}...{config.get('api_key', '')[-4:]}", flush=True)
    
    # Run Claude Code (with internal API retry configured)
    try:
        result = asyncio.run(run_claude_code(config))
        print(f"[CLAUDE_END] SDK execution completed, writing output...", flush=True)
    except Exception as e:
        print(f"ERROR: Exception during SDK execution: {e}", file=sys.stderr, flush=True)

        traceback.print_exc()
        result = {
            "messages": [{
                "type": "Error",
                "error": str(e),
                "timestamp": time.time(),
            }],
            "exit_reason": "sdk_execution_error",
            "stats": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "total_tokens": 0,
                "duration_ms": 0,
                "cost_usd": 0,
            },
        }

    output_file = config.get("output_file", "/tmp/claude_output.json")
    try:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"SUCCESS: Output written to {output_file}")
    except Exception as e:
        print(f"ERROR: Failed to write output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
