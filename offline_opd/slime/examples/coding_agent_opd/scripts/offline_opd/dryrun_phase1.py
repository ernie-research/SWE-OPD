"""Dry-run the GLM->DS chunk aligner on real Phase-1 rollout dumps (CPU, no torch).

A torch.save archive is a zip whose data.pkl reaches tensors only through
persistent_id, so a stub Unpickler recovers every plain field; tokens / loss_mask /
rollout_log_probs / prompt are all plain lists or str in the Phase-1 dump.

Usage:
  XVOCAB_ASSETS=<assets> PYTHONPATH=/tmp/xvocab_pylibs python3 dryrun_phase1.py <dump_dir> -n 20
"""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing
import os
import pickle
import re
import time
import zipfile
from collections import Counter
from pathlib import Path

from glm_ds_align import MASKED, MODES, AlignmentError, GlmToDsAligner


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        self.state = state


def _factory(module, name):
    def make(*a, **k):
        return _Stub()

    return make


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "builtins":
            return getattr(__import__("builtins"), name)
        return _factory(module, name)

    def persistent_load(self, pid):
        return _Stub()


def read_rollout(path):
    with zipfile.ZipFile(path) as z:
        key = next(n for n in z.namelist() if n.endswith("data.pkl"))
        blob = z.read(key)
    return _Unpickler(io.BytesIO(blob)).load()["samples"]


_TOOLS_RE = re.compile(r"<tools>\n(.*?)\n</tools>\n*", re.S)


def parse_prompt(prompt):
    """Split the rendered GLM prompt into (system, user, tools)."""
    body = prompt
    for marker in ("[gMASK]", "<sop>"):
        if body.startswith(marker):
            body = body[len(marker) :]
    if not body.startswith("<|system|>\n"):
        raise ValueError("prompt does not start with <|system|>")
    body = body[len("<|system|>\n") :]
    cut = body.index("<|user|>\n")
    system, rest = body[:cut], body[cut + len("<|user|>\n") :]
    if not rest.endswith("<|assistant|>"):
        raise ValueError("prompt does not end with <|assistant|>")
    user = rest[: -len("<|assistant|>")]
    m = _TOOLS_RE.search(system)
    if not m:
        raise ValueError("no <tools> block in system prompt")
    tools = [json.loads(line) for line in m.group(1).splitlines() if line.strip()]
    return (system[: m.start()] + system[m.end() :]).strip(), user, tools


def mask_runs(mask):
    runs, cur, start = [], mask[0], 0
    for i, v in enumerate(mask[1:], 1):
        if v != cur:
            runs.append((cur, start, i))
            cur, start = v, i
    runs.append((cur, start, len(mask)))
    return runs


def _assets():
    d = Path(os.environ.get("XVOCAB_ASSETS", ""))
    return d / "glm_tokenizer.json", d / "ds_tokenizer.json", d / "encoding_dsv4.py"


def probe(aligner, sample):
    """Print the mask-run layout of one sample so the turn parser can be checked."""
    tokens, mask = sample["tokens"], sample["loss_mask"]
    prompt_len = len(tokens) - len(mask)
    print(f"prompt_len={prompt_len} response_len={len(mask)} runs={len(mask_runs(mask))}")
    for kind, a, b in mask_runs(mask)[:8]:
        ids = tokens[prompt_len + a : prompt_len + b]
        text = b"".join(aligner.glm.token_bytes(t) for t in ids).decode("utf-8", "replace")
        tag = "ASSISTANT" if kind else "OBSERVATION"
        print(f"\n--- {tag} [{a},{b}) {b - a} tokens")
        print("  head:", repr(text[:160]))
        print("  tail:", repr(text[-80:]))


_AGG_KEYS = ("tokens", "teacher_tokens", "recovered", "exact", "struct", "chunk",
             "masked", "skipped", "turns", "turns_ok")

_WORKER = {}


def _init_worker(opts):
    """One aligner per process: the tokenizers are built after fork, never shared."""
    _WORKER["aligner"] = GlmToDsAligner(*_assets())
    _WORKER["opts"] = opts


def run_file(path):
    """Align every selected sample of one rollout file. Runs inside a pool worker."""
    aligner, opts = _WORKER["aligner"], _WORKER["opts"]
    rows, fails = [], Counter()
    blob = read_rollout(path)
    picked = blob if opts["all_in_file"] else blob[:1]
    if opts["overlong_only"]:
        picked = [s for s in picked if (s.get("metadata") or {}).get("overlong_turns")]
    for sample in picked:
        try:
            rows.append(run_sample(aligner, sample, opts["mode"]))
        except Exception as exc:  # prompt parsing / unexpected shape
            fails[f"sample:{type(exc).__name__}: {exc}"[:90]] += 1
    return rows, fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir")
    ap.add_argument("-n", type=int, default=20)
    ap.add_argument("--mode", default="struct", choices=sorted(MODES))
    ap.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 1))
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--all-in-file", action="store_true",
                    help="use every sample in each rollout file, not just the first")
    ap.add_argument("--overlong-only", action="store_true",
                    help="keep only samples whose rollout hit the per-turn length cap")
    args = ap.parse_args()

    files = sorted(Path(args.dump_dir).glob("rollout_*.pt"))[: args.n]
    if args.probe:
        probe(GlmToDsAligner(*_assets()), read_rollout(files[0])[0])
        return

    opts = dict(mode=args.mode, all_in_file=args.all_in_file, overlong_only=args.overlong_only)
    agg, kinds, fails, per_sample = Counter(), Counter(), Counter(), []
    started = time.monotonic()
    jobs = max(1, min(args.jobs, len(files)))
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(jobs, initializer=_init_worker, initargs=(opts,)) as pool:
        for done, (rows, file_fails) in enumerate(pool.imap_unordered(run_file, files), 1):
            fails.update(file_fails)
            for row in rows:
                per_sample.append(row)
                for key in _AGG_KEYS:
                    agg[key] += row[key]
                kinds.update(row["kinds"])
                fails.update(row["fails"])
            if done % 50 == 0 or done == len(files):
                rate = done / max(time.monotonic() - started, 1e-9)
                print(f"  .. {done}/{len(files)} files, {rate:.2f} files/s", flush=True)

    elapsed = time.monotonic() - started
    print(f"wall {elapsed:.0f}s on {jobs} workers "
          f"({len(per_sample) / max(elapsed, 1e-9):.2f} samples/s)")
    report(agg, kinds, fails, per_sample, args.mode)


_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\n(.*?)\n*</tool_response>", re.S)


def _text(aligner, ids):
    return b"".join(aligner.glm.token_bytes(t) for t in ids).decode("utf-8", "replace")


def _tool_content(text):
    m = _TOOL_RESPONSE_RE.search(text)
    return m.group(1) if m else text


_STOP_TOKENS = ("<|observation|>", "<|user|>", "<|endoftext|>")


def _fallback_message(text):
    """Best-effort assistant message for a turn the aligner cannot parse.

    Keeps the reasoning / content split when `</think>` is present and marks the turn
    unterminated when the student never emitted a stop token, which is what the official
    encoder's wo_eos template expects.
    """
    wo_eos = True
    for stop in _STOP_TOKENS:
        if text.endswith(stop):
            text, wo_eos = text[: -len(stop)], False
            break
    reasoning, open_at = "", text.find("<think>")
    if open_at >= 0:
        rest = text[open_at + len("<think>") :]
        close_at = rest.find("</think>")
        reasoning, text = (rest, "") if close_at < 0 else (rest[:close_at], rest[close_at + len("</think>") :])
    msg = {"role": "assistant", "reasoning_content": reasoning, "content": text.strip()}
    if wo_eos:
        msg["wo_eos"] = True
    return msg


def run_sample(aligner, sample, mode):
    """Walk one trajectory turn by turn, aligning every assistant run."""
    tokens, mask = sample["tokens"], sample["loss_mask"]
    prompt_len = len(tokens) - len(mask)
    system, user, tools = parse_prompt(sample["prompt"])
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    row = dict(tokens=0, teacher_tokens=0, recovered=0, exact=0, struct=0, chunk=0, masked=0, skipped=0,
               turns=0, turns_ok=0, kinds=Counter(), fails=Counter(), index=sample.get("index"))
    pending_ids, stopped = [], False
    for turn, (is_assistant, a, b) in enumerate(mask_runs(mask)):
        ids = tokens[prompt_len + a : prompt_len + b]
        if not is_assistant:
            call_id = pending_ids.pop(0) if pending_ids else f"t{turn}_0"
            messages.append({"role": "tool", "tool_call_id": call_id,
                             "content": _tool_content(_text(aligner, ids))})
            continue
        row["turns"] += 1
        if stopped:
            row["skipped"] += len(ids)
            continue
        try:
            al = aligner.align(messages, tools, ids)
        except AlignmentError as exc:
            # Unalignable turn (truncated, or the student wrote a malformed tool call):
            # mask it, but replay it as a best-effort assistant message so the remaining
            # turns of this trajectory stay usable.
            row["fails"][f"align: {exc}"[:70]] += 1
            row["skipped"] += len(ids)
            row["recovered"] += 1
            messages.append(_fallback_message(_text(aligner, ids)))
            pending_ids = []
            continue
        except RuntimeError as exc:
            # The renderer disagrees with the official encoder: a bug, not a data issue.
            row["fails"][f"render: {exc}"[:70]] += 1
            row["skipped"] += len(ids)
            stopped = True
            continue
        row["turns_ok"] += 1
        row["tokens"] += al.n_student
        row["teacher_tokens"] += al.n_teacher
        row["kinds"].update(al.stats)
        for name in ("exact", "struct", "chunk"):
            row[name] += sum(c.student[1] - c.student[0] for c in al.chunks if c.kind in MODES[name])
        row["masked"] += al.stats.get(MASKED, 0)
        msg = al.assistant_message
        for i, call in enumerate(msg.get("tool_calls", [])):
            call["id"] = f"t{turn}_{i}"
        pending_ids = [c["id"] for c in msg.get("tool_calls", [])]
        messages.append(msg)
    return row


def report(agg, kinds, fails, per_sample, mode):
    total = agg["tokens"] + agg["skipped"]
    print(f"samples={len(per_sample)} turns={agg['turns']} turns_ok={agg['turns_ok']}"
          f" ({agg['turns_ok'] / max(agg['turns'], 1):.1%})")
    print(f"response tokens: aligned {agg['tokens']} + skipped {agg['skipped']} = {total}")
    for name in ("exact", "struct", "chunk"):
        mark = " <- selected" if name == mode else ""
        print(f"  {name:6s} usable {agg[name]:8d}  {agg[name] / max(total, 1):6.1%}{mark}")
    print(f"  masked in aligned turns {agg['masked']:8d}  {agg['masked'] / max(total, 1):6.1%}")
    print(f"  skipped turns           {agg['skipped']:8d}  {agg['skipped'] / max(total, 1):6.1%}"
          f"  (recovered-and-continued turns: {agg['recovered']})")
    print(f"teacher tokens {agg['teacher_tokens']} = {agg['teacher_tokens'] / max(agg['tokens'], 1):.2f}x student")
    print("chunk kinds (student tokens):", dict(kinds))
    worst = sorted(per_sample, key=lambda r: r[mode] / max(r["tokens"] + r["skipped"], 1))[:5]
    print("worst samples (%s coverage):" % mode,
          [f"{r['index']}:{r[mode] / max(r['tokens'] + r['skipped'], 1):.0%}" for r in worst])
    if fails:
        print("failures:")
        for key, count in fails.most_common(12):
            print(f"  {count:4d}  {key}")



if __name__ == "__main__":
    main()
