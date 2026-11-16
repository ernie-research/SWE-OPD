#!/usr/bin/env python3
"""Export random offline rollout samples as readable evaluation-style JSON."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from examples.coding_agent_opd.visualize_rollouts import _clean_observation, _reconstruct_turns


ROLE_RE = re.compile(r"<\|(system|user|assistant)\|>\n?")
GLM_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
GLM_ARG_RE = re.compile(r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)
ROLE_MARKERS_RE = re.compile(r"<\|(?:assistant|observation)\|>\n?")


def prompt_messages(prompt: str) -> list[dict[str, Any]]:
    prompt = prompt.removeprefix("[gMASK]<sop>")
    fields = ROLE_RE.split(prompt)
    parsed = []
    for i in range(1, len(fields), 2):
        role = fields[i]
        content = fields[i + 1].strip() if i + 1 < len(fields) else ""
        if role != "assistant" and content:
            parsed.append({"role": role, "content": content})

    # The first GLM system block is generated from the tool schema. Evaluation
    # dumps store the human-authored system message, not that rendered wrapper.
    systems = [m for m in parsed if m["role"] == "system"]
    users = [m for m in parsed if m["role"] == "user"]
    result = []
    if systems:
        result.append(systems[-1])
    result.extend(users)
    return result


def parse_assistant(text: str, rollout_id: int, turn: int) -> tuple[dict[str, Any], list[str]]:
    text = ROLE_MARKERS_RE.sub("", text).strip()
    calls = []
    call_ids = []
    for index, match in enumerate(GLM_TOOL_CALL_RE.finditer(text)):
        body = match.group(1).strip()
        name, _, args_body = body.partition("\n")
        args = {key.strip(): value.strip() for key, value in GLM_ARG_RE.findall(args_body)}
        call_id = f"call_rollout_{rollout_id}_{turn}_{index}"
        call_ids.append(call_id)
        calls.append(
            {
                "id": call_id,
                "index": index,
                "type": "function",
                "function": {
                    "name": name.strip(),
                    "arguments": json.dumps(args, ensure_ascii=False),
                },
            }
        )
    content = GLM_TOOL_CALL_RE.sub("", text).strip()
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = calls
    return message, call_ids


def trajectory_messages(sample: dict[str, Any], tokenizer: Any, rollout_id: int) -> list[dict[str, Any]]:
    messages = prompt_messages(str(sample.get("prompt") or ""))
    pending_call_ids: list[str] = []
    assistant_turn = 0
    for mask, text in _reconstruct_turns(sample, tokenizer):
        if not text.strip():
            continue
        if mask == 1:
            assistant_turn += 1
            message, pending_call_ids = parse_assistant(text, rollout_id, assistant_turn)
            messages.append(message)
        else:
            content = _clean_observation(ROLE_MARKERS_RE.sub("", text))
            if pending_call_ids:
                message = {
                    "role": "tool",
                    "content": content,
                    "tool_call_id": pending_call_ids[-1],
                }
                if messages and messages[-1].get("tool_calls"):
                    message["name"] = messages[-1]["tool_calls"][-1]["function"]["name"]
            else:
                # Parser-error feedback is injected as a user turn and has no
                # preceding tool call, matching the online evaluation dumps.
                message = {"role": "user", "content": content}
            messages.append(message)
            pending_call_ids = []
    return messages


def export_sample(sample: dict[str, Any], tokenizer: Any, source: Path, rollout_id: int) -> dict[str, Any]:
    metadata = sample.get("metadata") or {}
    ds = metadata.get("ds") or {}
    decode_times = metadata.get("model_decode_times") or []
    tool_times = metadata.get("tool_exec_times") or []
    reward = metadata.get("swe_reward", sample.get("reward"))
    return {
        "instance_id": metadata.get("instance_id") or sample.get("label"),
        "messages": trajectory_messages(sample, tokenizer, rollout_id),
        "model": "/path/to/model/<org>/GLM-4.5-Air",
        "url": None,
        "sampling": {
            "temperature": 1.0,
            "top_p": None,
            "max_tokens": 16384,
        },
        "scaffold": {"name": "scale_swe", "variant": "edit_fn_calling"},
        "env": {
            key: ds.get(key)
            for key in ("repo", "base_commit", "docker_image", "swe_type", "problem_statement")
            if ds.get(key) is not None
        },
        "result": {
            "swe_reward": reward,
            "resolved": reward == 1.0,
            "submitted": metadata.get("submitted"),
            "status": sample.get("status"),
            "num_steps": metadata.get("tool_call_count", metadata.get("round_number")),
            "response_len": sample.get("response_length"),
            "elapsed": sum(decode_times) + sum(tool_times),
            "error": None,
        },
        "patch": None,
        "source": {"file": str(source), "rollout_id": rollout_id},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="A rollout .pt file or its dump directory")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--tokenizer",
        default="/path/to/model/<org>/GLM-4.5-Air",
    )
    args = parser.parse_args()

    source_dir = args.source.parent if args.source.is_file() else args.source
    files = sorted(source_dir.glob("rollout_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    if len(files) < args.count:
        raise ValueError(f"requested {args.count} samples, but only found {len(files)} rollout files")
    selected = random.Random(args.seed).sample(files, args.count)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    manifest = []
    for path in selected:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        samples = payload.get("samples") or []
        if len(samples) != 1:
            raise ValueError(f"expected one sample in {path}, found {len(samples)}")
        rollout_id = int(payload.get("rollout_id", path.stem.split("_")[-1]))
        exported = export_sample(samples[0], tokenizer, path, rollout_id)
        instance_id = str(exported["instance_id"]).replace("/", "_")
        output = args.output_dir / f"rollout_{rollout_id:04d}__{instance_id}.json"
        output.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n")
        manifest.append(
            {
                "rollout_id": rollout_id,
                "instance_id": exported["instance_id"],
                "source": str(path),
                "output": str(output),
                "messages": len(exported["messages"]),
                "response_tokens": exported["result"]["response_len"],
            }
        )

    (args.output_dir / "manifest.json").write_text(
        json.dumps({"seed": args.seed, "samples": manifest}, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"output_dir": str(args.output_dir), "seed": args.seed, "samples": manifest}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
