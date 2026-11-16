#!/usr/bin/env python3
"""Pretty-print coding_agent_opd rollout dumps (--dump-details output).

    python examples/coding_agent_opd/visualize_rollouts.py logs/dumps/rollout_data/0.pt
    python examples/coding_agent_opd/visualize_rollouts.py logs/dumps/rollout_data
    python examples/coding_agent_opd/visualize_rollouts.py <dump> --sample 0 --full

Each ``.pt`` file holds ``{"rollout_id", "samples": [sample.to_dict(), ...]}``
saved by ``slime/ray/rollout.py::_save_debug_rollout_data``. Per sample:
  * ``prompt``    : system+tools prompt + the user problem statement (rendered once)
  * ``response``  : ONLY the model-generated (trainable) text -- no observations,
                    no ChatML markers.
  * ``tokens``    : full prompt+response token ids
  * ``loss_mask`` : 1 for model tokens, 0 for tool observations (over the response)
  * ``metadata``  : instance_id, swe_reward, submitted, applied_cleanly, ...
  * ``teacher_log_probs`` : per-token teacher logprobs for the OPD KL penalty

Because ``response`` drops the tool observations, we reconstruct the real turn
sequence by decoding ``tokens[-response_length:]`` and splitting on runs of
equal ``loss_mask`` (1 => model turn, 0 => observation turn).


💡 example:
  cd /path/to/workspace/env_run/<user>/slime
  python examples/coding_agent_opd/visualize_rollouts.py /path/to/workspace/env_run/<user>/slime/logs/run-qwen3.5-9B-coding-opd-fully_async-pdc-multinode-nonuniform_OPD_20260724_233001/dumps/rollout_data --sample 1 --prompt

"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

# --------------------------------------------------------------------------
# ANSI color / emoji
# --------------------------------------------------------------------------
_USE_COLOR = True


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def bold(t):
    return _c("1", t)


def dim(t):
    return _c("2", t)


def red(t):
    return _c("31", t)


def green(t):
    return _c("32", t)


def yellow(t):
    return _c("33", t)


def blue(t):
    return _c("34", t)


def magenta(t):
    return _c("35", t)


def cyan(t):
    return _c("36", t)


def _rule(char: str = "─", width: int = 78) -> str:
    return dim(char * width)


def _truncate(text: str, limit: int) -> str:
    text = text or ""
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit] + dim(f"  … [+{len(text) - limit} chars]")


# --------------------------------------------------------------------------
# Tokenizer discovery
# --------------------------------------------------------------------------
def _find_tokenizer_dir(dump_path: str) -> str | None:
    """Walk up from the dump toward a sibling ``tokenizer/`` directory."""
    d = os.path.dirname(os.path.abspath(dump_path))
    for _ in range(4):
        cand = os.path.join(d, "tokenizer")
        if os.path.isdir(cand):
            return cand
        d = os.path.dirname(d)
    return None


def _load_tokenizer(dump_path: str):
    tok_dir = _find_tokenizer_dir(dump_path)
    if tok_dir is None:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - best effort
        print(dim(f"(tokenizer load failed: {type(e).__name__}: {e})"), file=sys.stderr)
        return None


# --------------------------------------------------------------------------
# Turn reconstruction
# --------------------------------------------------------------------------
def _mask_segments(loss_mask: list[int]) -> list[tuple[int, int, int]]:
    """Split loss_mask into runs -> list of (mask_value, start, end) index spans."""
    segs: list[tuple[int, int, int]] = []
    if not loss_mask:
        return segs
    start = 0
    cur = loss_mask[0]
    for i in range(1, len(loss_mask)):
        if loss_mask[i] != cur:
            segs.append((cur, start, i))
            start = i
            cur = loss_mask[i]
    segs.append((cur, start, len(loss_mask)))
    return segs


def _reconstruct_turns(sample: dict, tokenizer) -> list[tuple[int, str]]:
    """Return [(mask_value, decoded_text), ...] over the response region.

    Falls back to a single model turn (the raw ``response`` string) when no
    tokenizer / loss_mask is available.
    """
    resp_len = int(sample.get("response_length") or 0)
    loss_mask = sample.get("loss_mask") or []
    tokens = sample.get("tokens") or []
    if tokenizer is None or not loss_mask or resp_len <= 0 or len(tokens) < resp_len:
        return [(1, sample.get("response") or "")]

    resp_tokens = tokens[-resp_len:]
    turns: list[tuple[int, str]] = []
    for mask_val, s, e in _mask_segments(loss_mask):
        seg_ids = resp_tokens[s:e]
        if not seg_ids:
            continue
        text = tokenizer.decode(seg_ids, skip_special_tokens=False)
        turns.append((mask_val, text))
    return turns


# ChatML markers we strip when rendering an observation turn.
_CHATML_RE = re.compile(r"<\|im_(?:start|end)\|>")
# Qwen3 wraps a tool-role observation in <tool_response>...</tool_response>.
_TOOL_RESPONSE_RE = re.compile(r"</?tool_response>")


def _clean_observation(text: str) -> str:
    """Strip the ChatML tool-role wrapper from an observation segment.

    The wrapper is
    ``<|im_end|>\\n<|im_start|>user\\n<tool_response>\\n{obs}\\n</tool_response><|im_end|>\\n<|im_start|>assistant\\n``;
    after removing the ChatML markers and the ``<tool_response>`` tags we shed the
    leading ``user`` / trailing ``assistant`` role labels so only the observation
    body remains.
    """
    text = _CHATML_RE.sub("", text)
    text = _TOOL_RESPONSE_RE.sub("", text).strip()
    for lead in ("user\n", "user"):
        if text.startswith(lead):
            text = text[len(lead):]
            break
    for trail in ("\nassistant", "assistant"):
        if text.endswith(trail):
            text = text[: -len(trail)]
            break
    return text.strip()


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_FUNC_RE = re.compile(r"<function=([^>]+)>(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)


def _parse_model_turn(text: str) -> tuple[str | None, list[tuple[str, dict]], str]:
    """Split a model turn into (think, [(fn, params)...], leftover_text)."""
    think_m = _THINK_RE.search(text)
    think = think_m.group(1).strip() if think_m else None

    calls: list[tuple[str, dict]] = []
    for fm in _FUNC_RE.finditer(text):
        fn = fm.group(1).strip()
        params = {k.strip(): v for k, v in _PARAM_RE.findall(fm.group(2))}
        calls.append((fn, params))

    leftover = _THINK_RE.sub("", text)
    leftover = _FUNC_RE.sub("", leftover)
    leftover = _CHATML_RE.sub("", leftover).strip()
    return think, calls, leftover


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
_STATUS_EMOJI = {
    "completed": "✅",
    "truncated": "✂️",
    "aborted": "🛑",
    "pending": "⏳",
}


def _fmt_teacher_stats(tlp) -> str:
    if tlp is None:
        return dim("none")
    try:
        import torch

        if isinstance(tlp, torch.Tensor) and tlp.numel():
            return f"len={tlp.numel()} mean={tlp.float().mean().item():.3f}"
    except Exception:
        pass
    try:
        n = len(tlp)
        return f"len={n} mean={sum(float(x) for x in tlp) / n:.3f}" if n else dim("empty")
    except Exception:
        return dim(str(type(tlp).__name__))


def _render_header(sample: dict, idx: int) -> None:
    md = sample.get("metadata") or {}
    status = str(sample.get("status") or "?")
    emoji = _STATUS_EMOJI.get(status, "❔")
    swe_reward = md.get("swe_reward")
    reward_str = (
        green("PASS ✔ (1.0)") if swe_reward == 1.0
        else red(f"FAIL ✘ ({swe_reward})") if swe_reward is not None
        else dim("n/a")
    )

    print()
    print(bold(cyan("╔" + "═" * 76 + "╗")))
    print(bold(cyan("║")) + bold(f" 🎯 Sample #{idx}  ") + dim(f"(index={sample.get('index')} group={sample.get('group_index')})"))
    print(bold(cyan("╚" + "═" * 76 + "╝")))
    print(f"  {bold('instance')} : {yellow(str(md.get('instance_id', '?')))}")
    print(f"  {bold('status')}   : {emoji} {status}   {bold('reward')}: {reward_str}")
    print(
        f"  {bold('submitted')}: {md.get('submitted')}   "
        f"{bold('applied_cleanly')}: {md.get('applied_cleanly')}   "
        f"{bold('resp_len')}: {sample.get('response_length')}"
    )
    print(f"  {bold('teacher_lp')}: {_fmt_teacher_stats(sample.get('teacher_log_probs'))}   {bold('session')}: {dim(str(sample.get('session_id')))}")


def _render_problem(sample: dict, max_chars: int) -> None:
    md = sample.get("metadata") or {}
    ps = md.get("problem_statement")
    if not ps:
        return
    print()
    print(magenta("📋 PROBLEM STATEMENT"))
    print(_rule())
    print(_truncate(str(ps).strip(), max_chars))


def _render_turns(sample: dict, tokenizer, max_chars: int) -> None:
    turns = _reconstruct_turns(sample, tokenizer)
    print()
    print(blue(f"💬 TRAJECTORY  ({len(turns)} segments)"))
    print(_rule())
    model_turn = 0
    for mask_val, text in turns:
        if not text.strip():
            continue
        if mask_val == 1:
            model_turn += 1
            _render_model_turn(model_turn, text, max_chars)
        else:
            _render_observation(text, max_chars)


def _render_model_turn(n: int, text: str, max_chars: int) -> None:
    think, calls, leftover = _parse_model_turn(text)
    print()
    print(green(f"🤖 MODEL turn {n}"))
    if think:
        print(dim("  💭 think:"))
        for line in _truncate(think, max_chars).splitlines():
            print(dim(f"     {line}"))
    if leftover:
        print(f"  {_truncate(leftover, max_chars)}")
    for fn, params in calls:
        is_submit = fn.strip() == "submit"
        head = ("🏁 " if is_submit else "🔧 ") + bold(magenta(fn))
        print(f"  {head}")
        for k, v in params.items():
            v1 = _truncate(str(v), max_chars)
            if "\n" in v1:
                print(f"     {cyan(k)}:")
                for line in v1.splitlines():
                    print(f"       {line}")
            else:
                print(f"     {cyan(k)} = {v1}")
    if not calls and not think and not leftover:
        print(f"  {dim(_truncate(text.strip(), max_chars))}")


def _render_observation(text: str, max_chars: int) -> None:
    body = _clean_observation(text)
    if not body:
        return
    print(yellow("  🖥️  OBSERVATION"))
    for line in _truncate(body, max_chars).splitlines():
        print(dim(f"     {line}"))


def _render_prompt(sample: dict, max_chars: int) -> None:
    prompt = sample.get("prompt") or ""
    print()
    print(dim("🧩 SYSTEM/USER PROMPT (rendered once)"))
    print(_rule())
    print(dim(_truncate(prompt, max_chars)))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def _resolve_path(path: str) -> str:
    """Resolve ``path`` as given, else relative to this script's directory.

    The docstring examples invoke the tool from the repo root with a bare
    ``logs/dumps``, but the dumps actually live next to this script under
    ``examples/coding_agent_opd/logs/dumps``. Fall back to the script-relative
    location so the documented command works from anywhere.
    """
    if os.path.exists(path):
        return path
    script_rel = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.exists(script_rel):
        return script_rel
    return path


def _iter_dump_files(path: str) -> list[str]:
    path = _resolve_path(path)
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "*.pt")))
        if not files:  # maybe they pointed at logs/dumps
            files = sorted(glob.glob(os.path.join(path, "rollout_data", "*.pt")))
        return files
    return [path] if os.path.isfile(path) else []


def _visualize_file(path: str, args, tokenizer_cache: dict) -> None:
    import torch

    print()
    print(bold(cyan("█" * 78)))
    print(bold(cyan(f"█  DUMP: {os.path.abspath(path)}")))
    print(bold(cyan("█" * 78)))

    data = torch.load(path, weights_only=False)
    rollout_id = data.get("rollout_id")

    # train_data dumps are batched tensors ({"rollout_id","rank","rollout_data"}),
    # not per-sample trajectories -- this tool only renders rollout_data/*.pt.
    if "samples" not in data and "rollout_data" in data:
        print(yellow(
            "⏭️  This looks like a train_data dump (batched tensors, no per-sample "
            "trajectories). Point me at logs/dumps/rollout_data/*.pt instead."
        ))
        return

    samples = data.get("samples") or []
    print(f"{bold('rollout_id')}={rollout_id}   {bold('samples')}={len(samples)}")

    tok_dir = _find_tokenizer_dir(path)
    if tok_dir not in tokenizer_cache:
        tokenizer_cache[tok_dir] = _load_tokenizer(path)
    tokenizer = tokenizer_cache[tok_dir]
    if tokenizer is None:
        print(red("⚠️  no tokenizer found -> observations cannot be reconstructed; "
                  "showing raw model response only."))

    for idx, sample in enumerate(samples):
        if args.sample is not None and idx != args.sample:
            continue
        _render_header(sample, idx)
        if args.prompt:
            _render_prompt(sample, args.max_chars)
        _render_problem(sample, args.max_chars)
        _render_turns(sample, tokenizer, args.max_chars)
        print()


def main() -> None:
    global _USE_COLOR
    p = argparse.ArgumentParser(description="Pretty-print coding_agent_opd rollout dumps.")
    p.add_argument("path", help=".pt dump file, or a directory of dumps (rollout_data/).")
    p.add_argument("--sample", type=int, default=None, help="Only show this sample index.")
    p.add_argument("--full", action="store_true", help="Do not truncate any text.")
    p.add_argument("--max-chars", type=int, default=300,
                   help="Per-field truncation limit (default 300; ignored with --full).")
    p.add_argument("--prompt", action="store_true", help="Also print the initial system/user prompt.")
    p.add_argument("--no-color", action="store_true", help="Disable ANSI color.")
    args = p.parse_args()

    if args.no_color or not sys.stdout.isatty():
        _USE_COLOR = False
    if args.full:
        args.max_chars = 0

    files = _iter_dump_files(args.path)
    if not files:
        print(red(f"No .pt dumps found at {args.path!r}"), file=sys.stderr)
        sys.exit(1)

    tokenizer_cache: dict = {}
    for f in files:
        _visualize_file(f, args, tokenizer_cache)


if __name__ == "__main__":
    main()
