#!/usr/bin/env python3
"""Export Student/Teacher Top-128 distributions for ``let me <token>`` triples.

The utility consumes Phase-2 ``offline_opd_scored_sample_v1`` artifacts.  It
can reuse the ten trajectories from an existing token audit so that the output
is directly comparable with ``export_teacher_token_loss.py``.

For every matching three-token sequence it writes both:

* the rollout Student's own Top-128 distribution, augmented with the cached
  Teacher score for each mapped Student token (the support used by Phase 3);
* the cached Teacher's own probability-ranked Top-128 distribution, augmented
  with the reverse token-id mapping and the Student score when present.

Probabilities are full-vocabulary probabilities.  A separately named
``normalized_within_top128_probability`` field is included to make it explicit
when a distribution has been renormalized over only the retained support.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoTokenizer

from slime.utils.token_id_bridge import TokenIdBridge, load_token_id_bridge


SAMPLE_FORMAT = "offline_opd_scored_sample_v1"
WORD_TOKEN = re.compile(r"[A-Za-z][A-Za-z'-]*\Z")


def token_text(tokenizer: Any, token_id: int) -> tuple[str, str]:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    text = tokenizer.decode(
        [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return str(token), text


def decode_token_pieces(tokenizer: Any, token_ids: Iterable[int], chunk_size: int = 4096) -> list[str]:
    ids = [int(token_id) for token_id in token_ids]
    pieces: list[str] = []
    for start in range(0, len(ids), chunk_size):
        pieces.extend(
            tokenizer.batch_decode(
                [[token_id] for token_id in ids[start : start + chunk_size]],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    return pieces


def lookup_sorted(
    sorted_ids: torch.Tensor,
    sorted_logprobs: torch.Tensor,
    query_ids: torch.Tensor,
    cutoff_logprob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    columns = torch.searchsorted(sorted_ids, query_ids)
    safe = columns.clamp_max(sorted_ids.numel() - 1)
    hit = columns.lt(sorted_ids.numel()) & sorted_ids[safe].eq(query_ids)
    matched = sorted_logprobs[safe]
    return torch.where(hit, matched, cutoff_logprob), hit


def load_sample(path: Path) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(blob, dict) or blob.get("format") != SAMPLE_FORMAT:
        raise ValueError(f"{path}: unsupported scored sample")
    sample = blob.get("sample")
    if not isinstance(sample, dict):
        raise ValueError(f"{path}: missing sample dict")
    return sample


def source_to_rollout(experiment: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for manifest_path in sorted((experiment / "scored" / "source_manifests").glob("rollout_*.json")):
        payload = json.loads(manifest_path.read_text())
        rollout_id = int(payload["source_rollout_id"])
        for item in payload.get("samples") or []:
            result[str((experiment / "scored" / item["path"]).resolve())] = rollout_id
    return result


def select_sources(
    experiment: Path,
    selection_manifest: Path | None,
    seed: int,
    sample_count: int,
) -> list[tuple[int, Path]]:
    if selection_manifest is not None:
        payload = json.loads(selection_manifest.read_text())
        selected = [(int(item["rollout_id"]), Path(item["source"])) for item in payload["rollouts"]]
        if len(selected) != sample_count:
            raise ValueError(
                f"{selection_manifest}: expected {sample_count} selected trajectories, found {len(selected)}"
            )
        return selected

    files = sorted((experiment / "scored" / "samples").glob("sample_*.pt"))
    if len(files) < sample_count:
        raise ValueError(f"requested {sample_count} samples from only {len(files)} files")
    reverse = source_to_rollout(experiment)
    sampled = random.Random(seed).sample(files, sample_count)
    return [(reverse[str(path.resolve())], path) for path in sampled]


def is_let_me_triple(pieces: list[str], position: int) -> bool:
    if position + 2 >= len(pieces):
        return False
    return (
        pieces[position].strip().casefold() == "let"
        and pieces[position + 1].strip().casefold() == "me"
        and bool(WORD_TOKEN.fullmatch(pieces[position + 2].strip()))
    )


def probability(logprob: float) -> float:
    return float(math.exp(logprob))


def position_distribution(
    *,
    response_position: int,
    response_ids: np.ndarray,
    response_pieces: list[str],
    mask: np.ndarray,
    row_for_position: np.ndarray,
    rollout_logprobs: np.ndarray,
    student_ids: torch.Tensor,
    student_lp: torch.Tensor,
    teacher_ids: torch.Tensor,
    teacher_lp: torch.Tensor,
    cutoff_lp: torch.Tensor,
    bridge: TokenIdBridge,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    support_k: int,
    cache_k: int,
) -> dict[str, Any]:
    sampled_student_id = int(response_ids[response_position])
    sampled_student_token, sampled_student_text = token_text(student_tokenizer, sampled_student_id)
    sampled_student_lp = float(rollout_logprobs[response_position])
    result: dict[str, Any] = {
        "response_position": response_position,
        "student_token_id": sampled_student_id,
        "student_token": sampled_student_token,
        "student_text": sampled_student_text,
        "decoded_piece": response_pieces[response_position],
        "trainable": bool(mask[response_position]),
        "rollout_student_sampled_logprob": sampled_student_lp,
        "rollout_student_sampled_probability": probability(sampled_student_lp),
    }
    row = int(row_for_position[response_position])
    if row < 0:
        result.update(
            {
                "teacher_scored": False,
                "status": "unavailable_masked_position",
                "student_top128": None,
                "teacher_top128": None,
            }
        )
        return result

    sids = student_ids[row]
    slp = student_lp[row]
    sorder = torch.argsort(slp, descending=True)
    sids = sids[sorder]
    slp = slp[sorder]
    mapped_sids_np = bridge.map_numpy(
        sids.numpy(), context=f"response position {response_position} Student Top-{support_k}"
    )
    mapped_sids = torch.from_numpy(mapped_sids_np)

    tids = teacher_ids[row]
    tlp = teacher_lp[row]
    support_teacher_lp, support_hit = lookup_sorted(
        tids, tlp, mapped_sids, cutoff_lp[row]
    )
    student_log_z = torch.logsumexp(slp, dim=-1)
    student_q = (slp - student_log_z).exp()
    contributions = student_q * (slp - support_teacher_lp)

    teacher_order = torch.argsort(tlp, descending=True)[:support_k]
    top_teacher_ids = tids[teacher_order]
    top_teacher_lp = tlp[teacher_order]
    top_teacher_student_ids = bridge.map_teacher_numpy(
        top_teacher_ids.numpy(),
        context=f"response position {response_position} Teacher Top-{support_k}",
        allow_unmapped=True,
    )
    teacher_log_z = torch.logsumexp(top_teacher_lp, dim=-1)
    teacher_q = (top_teacher_lp - teacher_log_z).exp()

    mapped_to_student_rank = {int(tid): rank for rank, tid in enumerate(mapped_sids.tolist())}
    teacher_top_set = set(int(token_id) for token_id in top_teacher_ids.tolist())
    overlap_count = sum(int(token_id) in teacher_top_set for token_id in mapped_sids.tolist())

    student_candidates: list[dict[str, Any]] = []
    for rank in range(support_k):
        sid = int(sids[rank].item())
        tid = int(mapped_sids[rank].item())
        stok, stext = token_text(student_tokenizer, sid)
        ttok, ttext = token_text(teacher_tokenizer, tid)
        s_logprob = float(slp[rank].item())
        t_logprob = float(support_teacher_lp[rank].item())
        student_candidates.append(
            {
                "rank": rank + 1,
                "student_token_id": sid,
                "student_token": stok,
                "student_text": stext,
                "student_logprob": s_logprob,
                "student_probability": probability(s_logprob),
                "normalized_within_top128_probability": float(student_q[rank].item()),
                "mapped_teacher_token_id": tid,
                "mapped_teacher_token": ttok,
                "mapped_teacher_text": ttext,
                "teacher_cache_hit": bool(support_hit[rank].item()),
                "teacher_logprob_on_student_token": t_logprob,
                "teacher_probability_on_student_token": probability(t_logprob),
                "teacher_probability_status": (
                    f"exact_top{cache_k}"
                    if bool(support_hit[rank].item())
                    else "cutoff_imputation_upper_bound"
                ),
                "in_teacher_own_top128": tid in teacher_top_set,
                "reverse_kl_loss_contribution": float(contributions[rank].item()),
            }
        )

    teacher_candidates: list[dict[str, Any]] = []
    for rank in range(support_k):
        tid = int(top_teacher_ids[rank].item())
        sid = int(top_teacher_student_ids[rank])
        ttok, ttext = token_text(teacher_tokenizer, tid)
        t_logprob = float(top_teacher_lp[rank].item())
        candidate: dict[str, Any] = {
            "rank": rank + 1,
            "teacher_token_id": tid,
            "teacher_token": ttok,
            "teacher_text": ttext,
            "teacher_logprob": t_logprob,
            "teacher_probability": probability(t_logprob),
            "normalized_within_top128_probability": float(teacher_q[rank].item()),
            "mapped_student_token_id": sid if sid >= 0 else None,
            "teacher_only_token": sid < 0,
        }
        student_rank = mapped_to_student_rank.get(tid)
        if sid >= 0:
            stok, stext = token_text(student_tokenizer, sid)
            candidate.update({"mapped_student_token": stok, "mapped_student_text": stext})
        else:
            candidate.update({"mapped_student_token": None, "mapped_student_text": None})
        if student_rank is None:
            candidate.update(
                {
                    "in_student_top128": False,
                    "student_rank": None,
                    "student_logprob": None,
                    "student_probability": None,
                }
            )
        else:
            candidate.update(
                {
                    "in_student_top128": True,
                    "student_rank": student_rank + 1,
                    "student_logprob": float(slp[student_rank].item()),
                    "student_probability": probability(float(slp[student_rank].item())),
                }
            )
        teacher_candidates.append(candidate)

    sampled_teacher_id = int(
        bridge.map_numpy(
            np.asarray([sampled_student_id], dtype=np.int64),
            context=f"response position {response_position} sampled token",
        )[0]
    )
    sampled_teacher_lp, sampled_teacher_hit = lookup_sorted(
        tids,
        tlp,
        torch.tensor([sampled_teacher_id], dtype=torch.long),
        cutoff_lp[row],
    )
    sampled_tlp = float(sampled_teacher_lp.item())
    sampled_teacher_token, sampled_teacher_text = token_text(teacher_tokenizer, sampled_teacher_id)
    result.update(
        {
            "teacher_scored": True,
            "status": "ok",
            "mapped_sampled_teacher_token_id": sampled_teacher_id,
            "mapped_sampled_teacher_token": sampled_teacher_token,
            "mapped_sampled_teacher_text": sampled_teacher_text,
            "teacher_sampled_token_cache_hit": bool(sampled_teacher_hit.item()),
            "teacher_sampled_token_logprob": sampled_tlp,
            "teacher_sampled_token_probability": probability(sampled_tlp),
            "teacher_sampled_token_probability_status": (
                f"exact_top{cache_k}"
                if bool(sampled_teacher_hit.item())
                else "cutoff_imputation_upper_bound"
            ),
            "student_top128_mass": float(slp.exp().sum().item()),
            "teacher_top128_mass": float(top_teacher_lp.exp().sum().item()),
            "student_teacher_top128_overlap_count": overlap_count,
            "student_teacher_top128_overlap_ratio": overlap_count / support_k,
            "rollout_snapshot_top128_reverse_kl_loss": float(contributions.sum().item()),
            "student_top128": student_candidates,
            "teacher_top128": teacher_candidates,
        }
    )
    return result


def analyse_one(
    *,
    source: Path,
    rollout_id: int,
    output_dir: Path,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    bridge: TokenIdBridge,
    support_k: int,
    cache_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample = load_sample(source)
    response_length = int(sample["response_length"])
    tokens = np.asarray(sample["tokens"], dtype=np.int64)
    response_ids = tokens[-response_length:]
    response_pieces = decode_token_pieces(student_tokenizer, response_ids)
    if len(response_pieces) != response_length:
        raise ValueError(f"{source}: decoded piece count mismatch")

    mask = np.asarray(sample["loss_mask"], dtype=bool)
    positions = torch.as_tensor(sample["teacher_candidate_positions"], dtype=torch.long)
    row_for_position = np.full(response_length, -1, dtype=np.int64)
    row_for_position[positions.numpy()] = np.arange(positions.numel(), dtype=np.int64)
    student_ids = torch.as_tensor(sample["student_topk_ids"], dtype=torch.long)
    student_lp = torch.as_tensor(sample["student_topk_logprobs"], dtype=torch.float32)
    teacher_ids = torch.as_tensor(sample["teacher_candidate_ids"], dtype=torch.long)
    teacher_lp = torch.as_tensor(sample["teacher_candidate_logprobs"], dtype=torch.float32)
    cutoff_lp = torch.as_tensor(sample["teacher_cutoff_logprobs"], dtype=torch.float32)
    rollout_logprobs = np.asarray(sample["rollout_log_probs"], dtype=np.float64)
    expected = positions.numel()
    if student_ids.shape != (expected, support_k):
        raise ValueError(f"{source}: Student support shape {tuple(student_ids.shape)}")
    if teacher_ids.shape != (expected, cache_k):
        raise ValueError(f"{source}: Teacher cache shape {tuple(teacher_ids.shape)}")

    occurrences: list[dict[str, Any]] = []
    compact: list[dict[str, Any]] = []
    output_path = output_dir / f"rollout_{rollout_id:04d}_let_me_top128.jsonl"
    with output_path.open("w") as stream:
        for position in range(response_length - 2):
            if not is_let_me_triple(response_pieces, position):
                continue
            triple_positions = [position, position + 1, position + 2]
            position_rows = [
                position_distribution(
                    response_position=token_position,
                    response_ids=response_ids,
                    response_pieces=response_pieces,
                    mask=mask,
                    row_for_position=row_for_position,
                    rollout_logprobs=rollout_logprobs,
                    student_ids=student_ids,
                    student_lp=student_lp,
                    teacher_ids=teacher_ids,
                    teacher_lp=teacher_lp,
                    cutoff_lp=cutoff_lp,
                    bridge=bridge,
                    student_tokenizer=student_tokenizer,
                    teacher_tokenizer=teacher_tokenizer,
                    support_k=support_k,
                    cache_k=cache_k,
                )
                for token_position in triple_positions
            ]
            phrase = "".join(response_pieces[token_position] for token_position in triple_positions).strip()
            occurrence = {
                "rollout_id": rollout_id,
                "instance_id": (sample.get("metadata") or {}).get("instance_id") or sample.get("label"),
                "occurrence_index": len(occurrences),
                "phrase": phrase,
                "third_token_text": response_pieces[position + 2].strip(),
                "response_positions": triple_positions,
                "all_three_trainable": all(row["trainable"] for row in position_rows),
                "all_three_teacher_scored": all(row["teacher_scored"] for row in position_rows),
                "all_three_student_and_teacher_probability_gt_0_8": all(
                    row.get("rollout_student_sampled_probability", 0.0) > 0.8
                    and row.get("teacher_sampled_token_probability", 0.0) > 0.8
                    for row in position_rows
                ),
                "all_three_student_and_teacher_probability_gt_0_9": all(
                    row.get("rollout_student_sampled_probability", 0.0) > 0.9
                    and row.get("teacher_sampled_token_probability", 0.0) > 0.9
                    for row in position_rows
                ),
                "context_before": "".join(response_pieces[max(0, position - 12) : position]),
                "context_after": "".join(response_pieces[position + 3 : position + 15]),
                "positions": position_rows,
            }
            stream.write(json.dumps(occurrence, ensure_ascii=False) + "\n")
            occurrences.append(occurrence)
            compact.append(
                {
                    "rollout_id": rollout_id,
                    "instance_id": occurrence["instance_id"],
                    "occurrence_index": occurrence["occurrence_index"],
                    "phrase": phrase,
                    "third_token_text": occurrence["third_token_text"],
                    "response_positions": triple_positions,
                    "all_three_trainable": occurrence["all_three_trainable"],
                    "all_three_student_and_teacher_probability_gt_0_8": occurrence[
                        "all_three_student_and_teacher_probability_gt_0_8"
                    ],
                    "all_three_student_and_teacher_probability_gt_0_9": occurrence[
                        "all_three_student_and_teacher_probability_gt_0_9"
                    ],
                    "student_sampled_probabilities": [
                        row["rollout_student_sampled_probability"] for row in position_rows
                    ],
                    "teacher_sampled_probabilities": [
                        row.get("teacher_sampled_token_probability") for row in position_rows
                    ],
                    "top128_overlap_ratios": [
                        row.get("student_teacher_top128_overlap_ratio") for row in position_rows
                    ],
                    "reverse_kl_losses": [
                        row.get("rollout_snapshot_top128_reverse_kl_loss") for row in position_rows
                    ],
                    "context_before": occurrence["context_before"],
                    "context_after": occurrence["context_after"],
                    "full_distribution_file": str(output_path),
                }
            )

    trainable_occurrences = sum(item["all_three_trainable"] for item in occurrences)
    summary = {
        "rollout_id": rollout_id,
        "instance_id": (sample.get("metadata") or {}).get("instance_id") or sample.get("label"),
        "source": str(source),
        "output": str(output_path),
        "response_tokens": response_length,
        "let_me_triple_count": len(occurrences),
        "all_three_trainable_count": trainable_occurrences,
        "all_three_student_and_teacher_probability_gt_0_8_count": sum(
            item["all_three_student_and_teacher_probability_gt_0_8"] for item in occurrences
        ),
        "all_three_student_and_teacher_probability_gt_0_9_count": sum(
            item["all_three_student_and_teacher_probability_gt_0_9"] for item in occurrences
        ),
        "third_token_counts": dict(Counter(item["third_token_text"] for item in occurrences).most_common()),
    }
    (output_dir / f"rollout_{rollout_id:04d}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary, compact


def write_readme(
    output_dir: Path,
    experiment: Path,
    selection_manifest: Path | None,
    seed: int,
    summaries: list[dict[str, Any]],
    aggregate_third_tokens: Counter[str],
) -> None:
    total = sum(item["let_me_triple_count"] for item in summaries)
    trainable = sum(item["all_three_trainable_count"] for item in summaries)
    gt08 = sum(item["all_three_student_and_teacher_probability_gt_0_8_count"] for item in summaries)
    gt09 = sum(item["all_three_student_and_teacher_probability_gt_0_9_count"] for item in summaries)
    lines = [
        "# `let me <token>` Student/Teacher Top-128 audit",
        "",
        f"- Experiment: `{experiment}`",
        f"- Selection: `{selection_manifest}`" if selection_manifest else f"- Random seed: `{seed}`",
        f"- Trajectories: {len(summaries)}",
        f"- Three-token occurrences: {total}",
        f"- All three positions trainable: {trainable}",
        f"- All three sampled-token probabilities > 0.8 in both models: {gt08}",
        f"- All three sampled-token probabilities > 0.9 in both models: {gt09}",
        "",
        "A match is exactly three consecutive Student tokens whose individually decoded pieces are",
        "case-insensitive `let`, `me`, and one alphabetic token. This intentionally matches the",
        "earlier three-token metric rather than a re-tokenized text regex.",
        "",
        "Each `rollout_*_let_me_top128.jsonl` record contains the three token positions. At every",
        "trainable position it includes (1) Student Top-128 plus Teacher probabilities on that",
        "same support and (2) Teacher's own Top-128 plus reverse mappings into Student vocabulary.",
        "Teacher cache misses on Student support use the same rank-1000 cutoff upper bound as the",
        "Phase-3 loss and are explicitly marked.",
        "",
        "## Per-trajectory counts",
        "",
        "| rollout | instance | occurrences | trainable | both >0.8 | both >0.9 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['rollout_id']} | `{item['instance_id']}` | {item['let_me_triple_count']} | "
            f"{item['all_three_trainable_count']} | "
            f"{item['all_three_student_and_teacher_probability_gt_0_8_count']} | "
            f"{item['all_three_student_and_teacher_probability_gt_0_9_count']} |"
        )
    lines.extend(
        [
            "",
            "## Most common third tokens",
            "",
            "| token | count |",
            "|---|---:|",
        ]
    )
    for token, count in aggregate_third_tokens.most_common(30):
        lines.append(f"| `{token}` | {count} |")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--selection-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--sample-count", type=int, default=10)
    parser.add_argument(
        "--student-tokenizer", default="/path/to/model/<org>/GLM-4.5-Air"
    )
    parser.add_argument(
        "--teacher-tokenizer", default="/path/to/model/<org>/GLM-5.2-FP8"
    )
    parser.add_argument("--token-id-bridge", type=Path)
    parser.add_argument("--support-k", type=int, default=128)
    parser.add_argument("--cache-k", type=int, default=1000)
    args = parser.parse_args()

    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    selected = select_sources(
        args.experiment, args.selection_manifest, args.seed, args.sample_count
    )
    bridge_path = args.token_id_bridge or args.experiment / "glm45air_to_glm52_token_ids.json"
    bridge = load_token_id_bridge(str(bridge_path))
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_tokenizer, trust_remote_code=True)
    args.output_dir.mkdir(parents=True)

    summaries: list[dict[str, Any]] = []
    compact_rows: list[dict[str, Any]] = []
    aggregate_third_tokens: Counter[str] = Counter()
    for index, (rollout_id, source) in enumerate(selected, start=1):
        print(f"[{index}/{len(selected)}] rollout={rollout_id} source={source}", flush=True)
        summary, compact = analyse_one(
            source=source,
            rollout_id=rollout_id,
            output_dir=args.output_dir,
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
            bridge=bridge,
            support_k=args.support_k,
            cache_k=args.cache_k,
        )
        summaries.append(summary)
        compact_rows.extend(compact)
        aggregate_third_tokens.update(summary["third_token_counts"])
        print(f"  found {summary['let_me_triple_count']} occurrences", flush=True)

    compact_path = args.output_dir / "occurrences_compact.jsonl"
    with compact_path.open("w") as stream:
        for row in compact_rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    csv_path = args.output_dir / "occurrences_compact.csv"
    csv_fields = [
        "rollout_id",
        "instance_id",
        "occurrence_index",
        "phrase",
        "third_token_text",
        "response_positions",
        "all_three_trainable",
        "all_three_student_and_teacher_probability_gt_0_8",
        "all_three_student_and_teacher_probability_gt_0_9",
        "student_sampled_probabilities",
        "teacher_sampled_probabilities",
        "top128_overlap_ratios",
        "reverse_kl_losses",
        "context_before",
        "context_after",
        "full_distribution_file",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        for row in compact_rows:
            writer.writerow(
                {
                    key: json.dumps(row[key], ensure_ascii=False)
                    if isinstance(row[key], (list, dict))
                    else row[key]
                    for key in csv_fields
                }
            )

    manifest = {
        "experiment": str(args.experiment),
        "selection_manifest": str(args.selection_manifest) if args.selection_manifest else None,
        "seed": args.seed,
        "sample_count": args.sample_count,
        "student_tokenizer": args.student_tokenizer,
        "teacher_tokenizer": args.teacher_tokenizer,
        "token_id_bridge": str(bridge_path),
        "support_k": args.support_k,
        "teacher_cache_k": args.cache_k,
        "total_occurrences": len(compact_rows),
        "rollouts": summaries,
        "aggregate_third_token_counts": dict(aggregate_third_tokens.most_common()),
        "compact_jsonl": str(compact_path),
        "compact_csv": str(csv_path),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    write_readme(
        args.output_dir,
        args.experiment,
        args.selection_manifest,
        args.seed,
        summaries,
        aggregate_third_tokens,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
