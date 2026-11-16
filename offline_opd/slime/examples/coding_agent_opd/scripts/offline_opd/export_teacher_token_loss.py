#!/usr/bin/env python3
"""Audit cached-Teacher probabilities and Phase-3 Top-K loss per response token.

This utility intentionally uses only a scored Phase-2 artifact.  Consequently
``student_topk_*`` is the rollout Student snapshot, not a later training
checkpoint's dynamically recomputed Top-K.  Apart from that substitution, the
lookup, cutoff imputation, support normalization, and reverse-KL expression are
the same as ``_opd_dynamic_cached_teacher_topk_loss`` in
``slime/backends/megatron_utils/loss.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

from slime.utils.token_id_bridge import load_token_id_bridge


SAMPLE_FORMAT = "offline_opd_scored_sample_v1"


def json_number(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def token_text(tokenizer: Any, token_id: int) -> tuple[str, str]:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    text = tokenizer.decode(
        [int(token_id)], skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    return str(token), text


def lookup_sorted(
    sorted_ids: torch.Tensor,
    sorted_logprobs: torch.Tensor,
    query_ids: torch.Tensor,
    cutoff_logprob: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mirror Phase 3's searchsorted lookup and cutoff miss policy."""
    columns = torch.searchsorted(sorted_ids, query_ids)
    safe = columns.clamp_max(sorted_ids.numel() - 1)
    hit = columns.lt(sorted_ids.numel()) & sorted_ids[safe].eq(query_ids)
    matched = sorted_logprobs[safe]
    return torch.where(hit, matched, cutoff_logprob), hit


def resolve_scored_sample(experiment: Path, rollout_id: int) -> Path:
    manifest_path = experiment / "scored" / "source_manifests" / f"rollout_{rollout_id}.json"
    manifest = json.loads(manifest_path.read_text())
    samples = manifest.get("samples") or []
    if len(samples) != 1:
        raise ValueError(f"{manifest_path}: expected one sample, found {len(samples)}")
    return experiment / "scored" / samples[0]["path"]


def load_sample(path: Path) -> dict[str, Any]:
    blob = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if not isinstance(blob, dict) or blob.get("format") != SAMPLE_FORMAT:
        raise ValueError(f"{path}: unsupported scored sample")
    sample = blob.get("sample")
    if not isinstance(sample, dict):
        raise ValueError(f"{path}: missing sample dict")
    return sample


def validate_shapes(sample: dict[str, Any], support_k: int, cache_k: int) -> None:
    response_length = int(sample["response_length"])
    mask = np.asarray(sample["loss_mask"], dtype=bool)
    positions = torch.as_tensor(sample["teacher_candidate_positions"], dtype=torch.long)
    n = int(positions.numel())
    expected_positions = torch.from_numpy(np.flatnonzero(mask).astype(np.int64))
    checks = {
        "loss_mask": mask.shape == (response_length,),
        "teacher_candidate_positions": torch.equal(positions, expected_positions),
        "student_topk_ids": tuple(np.asarray(sample["student_topk_ids"]).shape) == (n, support_k),
        "student_topk_logprobs": tuple(np.asarray(sample["student_topk_logprobs"]).shape)
        == (n, support_k),
        "teacher_candidate_ids": tuple(torch.as_tensor(sample["teacher_candidate_ids"]).shape)
        == (n, cache_k),
        "teacher_candidate_logprobs": tuple(
            torch.as_tensor(sample["teacher_candidate_logprobs"]).shape
        )
        == (n, cache_k),
        "teacher_cutoff_logprobs": tuple(
            torch.as_tensor(sample["teacher_cutoff_logprobs"]).shape
        )
        == (n,),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(f"invalid scored sample fields: {', '.join(failed)}")


def analyse_one(
    *,
    experiment: Path,
    rollout_id: int,
    output_dir: Path,
    student_tokenizer: Any,
    teacher_tokenizer: Any,
    bridge_path: Path | None,
    identity_token_ids: bool,
    support_k: int,
    cache_k: int,
    component_examples: int,
) -> dict[str, Any]:
    source = resolve_scored_sample(experiment, rollout_id)
    sample = load_sample(source)
    validate_shapes(sample, support_k, cache_k)
    bridge = None if identity_token_ids else load_token_id_bridge(str(bridge_path))

    tokens = np.asarray(sample["tokens"], dtype=np.int64)
    response_length = int(sample["response_length"])
    prompt_length = int(tokens.size - response_length)
    response_ids = tokens[prompt_length:]
    mask = np.asarray(sample["loss_mask"], dtype=bool)
    positions = torch.as_tensor(sample["teacher_candidate_positions"], dtype=torch.long)
    position_to_row = {int(position): row for row, position in enumerate(positions.tolist())}

    student_ids = torch.as_tensor(sample["student_topk_ids"], dtype=torch.long)
    student_lp = torch.as_tensor(sample["student_topk_logprobs"], dtype=torch.float32)
    teacher_ids = torch.as_tensor(sample["teacher_candidate_ids"], dtype=torch.long)
    teacher_lp = torch.as_tensor(sample["teacher_candidate_logprobs"], dtype=torch.float32)
    cutoff_lp = torch.as_tensor(sample["teacher_cutoff_logprobs"], dtype=torch.float32)
    cache_mass = torch.as_tensor(sample["teacher_top4096_mass"], dtype=torch.float32)

    mapped_support_np = (
        student_ids.numpy()
        if bridge is None
        else bridge.map_numpy(
            student_ids.numpy(), context=f"rollout {rollout_id} Student Top-{support_k}"
        )
    )
    mapped_support = torch.from_numpy(mapped_support_np)

    loss_rows: list[float] = []
    hit_ratios: list[float] = []
    sampled_exact = 0
    sampled_miss = 0
    trainable = 0
    component_rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    token_output = output_dir / f"rollout_{rollout_id:04d}_tokens.jsonl"

    with token_output.open("w") as stream:
        for response_position, sampled_student_id in enumerate(response_ids.tolist()):
            student_token, student_text = token_text(student_tokenizer, sampled_student_id)
            record: dict[str, Any] = {
                "response_position": response_position,
                "sequence_position": prompt_length + response_position,
                "student_token_id": int(sampled_student_id),
                "student_token": student_token,
                "student_text": student_text,
                "trainable": bool(mask[response_position]),
            }
            row = position_to_row.get(response_position)
            if row is None:
                record.update(
                    {
                        "teacher_scored": False,
                        "teacher_probability": None,
                        "teacher_probability_status": "unavailable_masked_position",
                        "rollout_snapshot_top128_loss": None,
                    }
                )
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            trainable += 1
            sampled_teacher_id = (
                int(sampled_student_id)
                if bridge is None
                else int(
                    bridge.map_numpy(
                        np.asarray([sampled_student_id], dtype=np.int64),
                        context=f"rollout {rollout_id} response token",
                    )[0]
                )
            )
            teacher_token, teacher_text = token_text(teacher_tokenizer, sampled_teacher_id)
            sampled_lp, sampled_hit = lookup_sorted(
                teacher_ids[row],
                teacher_lp[row],
                torch.tensor([sampled_teacher_id], dtype=torch.long),
                cutoff_lp[row],
            )
            exact = bool(sampled_hit.item())
            sampled_exact += int(exact)
            sampled_miss += int(not exact)

            support_teacher_lp, support_hit = lookup_sorted(
                teacher_ids[row], teacher_lp[row], mapped_support[row], cutoff_lp[row]
            )
            # Exact Phase-3 expression.  student_lp is the full-vocabulary
            # log-probability, while q is normalized only within Student Top-K.
            log_q = student_lp[row] - torch.logsumexp(student_lp[row], dim=-1)
            q = log_q.exp()
            contributions = q * (student_lp[row] - support_teacher_lp)
            loss = float(contributions.sum().item())
            hit_count = int(support_hit.sum().item())
            hit_ratio = hit_count / support_k
            student_mass = float(student_lp[row].exp().sum().item())
            matched_teacher_mass = float(
                torch.where(support_hit, support_teacher_lp.exp(), 0.0).sum().item()
            )
            teacher_mass_upper = min(
                1.0,
                matched_teacher_mass
                + (support_k - hit_count) * float(cutoff_lp[row].exp().item()),
            )
            loss_rows.append(loss)
            hit_ratios.append(hit_ratio)

            record.update(
                {
                    "teacher_scored": True,
                    "teacher_token_id": sampled_teacher_id,
                    "teacher_token": teacher_token,
                    "teacher_text": teacher_text,
                    "teacher_cache_hit": exact,
                    "teacher_logprob": float(sampled_lp.item()),
                    "teacher_probability": float(sampled_lp.exp().item()),
                    "teacher_probability_status": (
                        f"exact_top{cache_k}" if exact else "cutoff_imputation_upper_bound"
                    ),
                    "teacher_cutoff_logprob": float(cutoff_lp[row].item()),
                    "teacher_cutoff_probability": float(cutoff_lp[row].exp().item()),
                    "rollout_student_sampled_logprob": float(sample["rollout_log_probs"][response_position]),
                    "rollout_student_sampled_probability": float(
                        math.exp(float(sample["rollout_log_probs"][response_position]))
                    ),
                    "rollout_snapshot_top128_loss": loss,
                    "top128_teacher_cache_hit_count": hit_count,
                    "top128_teacher_cache_hit_ratio": hit_ratio,
                    "top128_student_mass": student_mass,
                    "top128_teacher_matched_mass_lower": matched_teacher_mass,
                    "top128_teacher_mass_upper": teacher_mass_upper,
                    f"teacher_top{cache_k}_mass": float(cache_mass[row].item()),
                }
            )
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

            if len(component_rows) < component_examples:
                candidates = []
                for rank in range(support_k):
                    sid = int(student_ids[row, rank].item())
                    tid = int(mapped_support[row, rank].item())
                    stok, stext = token_text(student_tokenizer, sid)
                    ttok, ttext = token_text(teacher_tokenizer, tid)
                    candidates.append(
                        {
                            "student_rank": rank + 1,
                            "student_token_id": sid,
                            "student_token": stok,
                            "student_text": stext,
                            "teacher_token_id": tid,
                            "teacher_token": ttok,
                            "teacher_text": ttext,
                            "student_logprob": float(student_lp[row, rank].item()),
                            "student_probability": float(student_lp[row, rank].exp().item()),
                            "q_within_student_top128": float(q[rank].item()),
                            "teacher_cache_hit": bool(support_hit[rank].item()),
                            "teacher_logprob_used": float(support_teacher_lp[rank].item()),
                            "teacher_probability_used": float(support_teacher_lp[rank].exp().item()),
                            "loss_contribution": float(contributions[rank].item()),
                        }
                    )
                component_rows.append(
                    {
                        "response_position": response_position,
                        "sampled_student_token_id": int(sampled_student_id),
                        "rollout_snapshot_top128_loss": loss,
                        "sum_of_components": float(sum(x["loss_contribution"] for x in candidates)),
                        "candidates": candidates,
                    }
                )

    summary = {
        "rollout_id": rollout_id,
        "instance_id": (sample.get("metadata") or {}).get("instance_id") or sample.get("label"),
        "source": str(source),
        "token_output": str(token_output),
        "response_tokens": response_length,
        "trainable_tokens": trainable,
        "masked_tokens": response_length - trainable,
        f"sampled_token_teacher_exact_top{cache_k}_count": sampled_exact,
        "sampled_token_teacher_cutoff_imputed_count": sampled_miss,
        f"sampled_token_teacher_exact_top{cache_k}_ratio": sampled_exact / max(trainable, 1),
        "rollout_snapshot_top128_loss_mean": float(np.mean(loss_rows)),
        "rollout_snapshot_top128_loss_std_over_tokens": float(np.std(loss_rows)),
        "rollout_snapshot_top128_loss_min": float(np.min(loss_rows)),
        "rollout_snapshot_top128_loss_max": float(np.max(loss_rows)),
        "top128_teacher_cache_hit_ratio_mean": float(np.mean(hit_ratios)),
        "component_examples": component_rows,
        "formula": {
            "q_i": "exp(student_logprob_i - logsumexp(student_top128_logprobs))",
            "per_token_loss": "sum_i q_i * (student_logprob_i - teacher_logprob_used_i)",
            "teacher_cache_miss": f"teacher_logprob_used_i = teacher_top{cache_k}_cutoff_logprob",
        },
        "scope_note": (
            "Uses the rollout Student Top-128 snapshot saved in Phase 1. Phase 3 uses "
            "the same formula but recomputes current Student Top-128 each forward."
        ),
    }
    summary_path = output_dir / f"rollout_{rollout_id:04d}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return {key: value for key, value in summary.items() if key != "component_examples"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rollout-ids", nargs="+", type=int, required=True)
    parser.add_argument(
        "--student-tokenizer",
        default="/path/to/model/<org>/GLM-4.5-Air",
    )
    parser.add_argument(
        "--teacher-tokenizer",
        default="/path/to/model/<org>/GLM-5.2-FP8",
    )
    parser.add_argument("--token-id-bridge", type=Path)
    parser.add_argument(
        "--identity-token-ids",
        action="store_true",
        help="Student and Teacher share token IDs; do not load or apply a token bridge.",
    )
    parser.add_argument("--support-k", type=int, default=128)
    parser.add_argument("--cache-k", type=int, default=1000)
    parser.add_argument(
        "--component-examples",
        type=int,
        default=3,
        help="Fully expand Top-K contributions for this many positions per trajectory.",
    )
    args = parser.parse_args()

    if args.identity_token_ids and args.token_id_bridge is not None:
        parser.error("--identity-token-ids and --token-id-bridge are mutually exclusive")
    bridge_path = (
        None
        if args.identity_token_ids
        else args.token_id_bridge or args.experiment / "glm45air_to_glm52_token_ids.json"
    )
    student_tokenizer = AutoTokenizer.from_pretrained(
        args.student_tokenizer, trust_remote_code=True
    )
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_tokenizer, trust_remote_code=True
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    summaries = []
    for rollout_id in args.rollout_ids:
        summaries.append(
            analyse_one(
                experiment=args.experiment,
                rollout_id=rollout_id,
                output_dir=args.output_dir,
                student_tokenizer=student_tokenizer,
                teacher_tokenizer=teacher_tokenizer,
                bridge_path=bridge_path,
                identity_token_ids=args.identity_token_ids,
                support_k=args.support_k,
                cache_k=args.cache_k,
                component_examples=args.component_examples,
            )
        )
    manifest = {
        "experiment": str(args.experiment),
        "student_tokenizer": args.student_tokenizer,
        "teacher_tokenizer": args.teacher_tokenizer,
        "token_id_bridge": "identity" if args.identity_token_ids else str(bridge_path),
        "support_k": args.support_k,
        "teacher_cache_k": args.cache_k,
        "rollouts": summaries,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
