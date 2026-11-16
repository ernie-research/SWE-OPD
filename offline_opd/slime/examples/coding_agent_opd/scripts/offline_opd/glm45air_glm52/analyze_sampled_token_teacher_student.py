#!/usr/bin/env python3
"""Analyze cached Teacher vs rollout Student probability on sampled tokens.

The Phase-2 cache only contains exact Teacher probabilities when the sampled
Student token appears in the cached Teacher Top-K row.  Cache misses are kept as
a separate, censored population: the persisted cutoff is an upper bound, not a
substitute probability.  This script supports independent file shards and a
later merge into a self-contained HTML report.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import pickle
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROB_BINS = np.asarray(
    [
        0.0,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        5e-2,
        1e-1,
        2.5e-1,
        5e-1,
        8e-1,
        9e-1,
        9.5e-1,
        9.9e-1,
        9.99e-1,
        1.0000001,
    ],
    dtype=np.float64,
)
PROB_DELTA_BINS = np.asarray(
    [
        -1.0000001,
        -0.9,
        -0.8,
        -0.5,
        -0.25,
        -0.1,
        -0.05,
        -0.01,
        -1e-3,
        -1e-4,
        -1e-5,
        0.0,
        1e-5,
        1e-4,
        1e-3,
        0.01,
        0.05,
        0.1,
        0.25,
        0.5,
        0.8,
        0.9,
        1.0000001,
    ],
    dtype=np.float64,
)
LOG_DELTA_BINS = np.asarray(
    [
        -np.inf,
        -20,
        -12,
        -8,
        -6,
        -4,
        -3,
        -2,
        -1.5,
        -1,
        -0.75,
        -0.5,
        -0.25,
        -0.1,
        -0.03,
        0.0,
        0.03,
        0.1,
        0.25,
        0.5,
        0.75,
        1,
        1.5,
        2,
        3,
        4,
        6,
        8,
        12,
        20,
        np.inf,
    ],
    dtype=np.float64,
)
RESERVOIR_COLUMNS = (
    "student_prob",
    "teacher_prob",
    "prob_delta",
    "logprob_delta",
)
MISS_RESERVOIR_COLUMNS = ("student_prob", "cutoff_prob")
QUANTILES = (0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0)


class Reservoir:
    """Uniform bounded sample of a streamed row population."""

    def __init__(self, columns: tuple[str, ...], capacity: int, seed: int):
        self.columns = columns
        self.capacity = capacity
        self.seen = 0
        self.values = {column: np.empty(0, dtype=np.float64) for column in columns}
        self.rng = np.random.default_rng(seed)

    def update(self, values: dict[str, np.ndarray]) -> None:
        n = len(values[self.columns[0]])
        if n == 0:
            return
        if any(len(values[column]) != n for column in self.columns):
            raise ValueError("reservoir columns have unequal lengths")
        current = len(self.values[self.columns[0]])
        if self.seen + n <= self.capacity:
            for column in self.columns:
                self.values[column] = np.concatenate(
                    (self.values[column], np.asarray(values[column], dtype=np.float64))
                )
        elif self.seen < self.capacity:
            combined_size = current + n
            selected = self.rng.choice(combined_size, size=self.capacity, replace=False)
            for column in self.columns:
                combined = np.concatenate(
                    (self.values[column], np.asarray(values[column], dtype=np.float64))
                )
                self.values[column] = combined[selected]
        else:
            new_count = int(
                self.rng.hypergeometric(
                    ngood=n,
                    nbad=self.seen,
                    nsample=self.capacity,
                )
            )
            if new_count:
                replacement = self.rng.choice(
                    self.capacity, size=new_count, replace=False
                )
                incoming = self.rng.choice(n, size=new_count, replace=False)
                for column in self.columns:
                    self.values[column][replacement] = np.asarray(
                        values[column], dtype=np.float64
                    )[incoming]
        self.seen += n

    def payload(self) -> dict[str, Any]:
        return {
            "seen": self.seen,
            "capacity": self.capacity,
            "values": self.values,
        }


def rowwise_searchsorted(sorted_ids: np.ndarray, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Binary-search one query in each independently sorted cache row."""
    rows, width = sorted_ids.shape
    if queries.shape != (rows,):
        raise ValueError(f"query/cache row mismatch: {queries.shape} vs {sorted_ids.shape}")
    row_ids = np.arange(rows)
    low = np.zeros(rows, dtype=np.int32)
    high = np.full(rows, width, dtype=np.int32)
    while np.any(low < high):
        middle = (low + high) // 2
        active = low < high
        active_rows = row_ids[active]
        active_middle = middle[active]
        values = sorted_ids[active_rows, active_middle]
        go_right = values < queries[active]
        low[active_rows[go_right]] = active_middle[go_right] + 1
        high[active_rows[~go_right]] = active_middle[~go_right]
    safe = np.minimum(low, width - 1)
    hit = (low < width) & (sorted_ids[row_ids, safe] == queries)
    return safe, hit


def probability_bin_labels() -> list[str]:
    def edge(value: float) -> str:
        if value == 0:
            return "0"
        if value < 0.001:
            return f"{value:.0e}"
        if value < 0.1:
            return f"{value:.3g}"
        return f"{value:.3g}"

    return [
        f"[{edge(left)}, {edge(right)})"
        for left, right in zip(PROB_BINS[:-1], PROB_BINS[1:], strict=True)
    ]


def update_extremes(
    target: list[dict[str, Any]],
    score: np.ndarray,
    *,
    largest: bool,
    limit: int,
    payload: dict[str, np.ndarray],
    sample_index: int,
    label: str,
) -> None:
    if not score.size:
        return
    take = min(limit, score.size)
    if take == score.size:
        columns = np.arange(score.size)
    elif largest:
        columns = np.argpartition(score, -take)[-take:]
    else:
        columns = np.argpartition(score, take - 1)[:take]
    for column in columns:
        target.append(
            {
                "score": float(score[column]),
                "sample_index": sample_index,
                "label": label,
                **{
                    key: value[column].item()
                    for key, value in payload.items()
                },
            }
        )
    target.sort(key=lambda item: item["score"], reverse=largest)
    del target[limit:]


def scan_shard(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).resolve()
    files = sorted(input_dir.glob("sample_*.pt"))
    if args.max_files:
        files = files[: args.max_files]
    selected_files = [
        path for index, path in enumerate(files) if index % args.num_shards == args.shard_index
    ]
    bridge_payload = json.loads(Path(args.bridge).read_text())
    bridge = np.asarray(bridge_payload["student_to_teacher"], dtype=np.int64)

    counts: defaultdict[str, int] = defaultdict(int)
    histograms = {
        "student_prob": np.zeros(len(PROB_BINS) - 1, dtype=np.int64),
        "teacher_prob": np.zeros(len(PROB_BINS) - 1, dtype=np.int64),
        "prob_delta": np.zeros(len(PROB_DELTA_BINS) - 1, dtype=np.int64),
        "logprob_delta": np.zeros(len(LOG_DELTA_BINS) - 1, dtype=np.int64),
        "probability_2d": np.zeros(
            (len(PROB_BINS) - 1, len(PROB_BINS) - 1), dtype=np.int64
        ),
    }
    token_stats: dict[int, np.ndarray] = {}
    sequence_stats: list[dict[str, Any]] = []
    extremes = {
        "teacher_higher": [],
        "student_higher": [],
        "absolute_logprob_delta": [],
    }
    reservoir = Reservoir(RESERVOIR_COLUMNS, args.reservoir_size, 9103 + args.shard_index)
    miss_reservoir = Reservoir(
        MISS_RESERVOIR_COLUMNS, max(args.reservoir_size // 4, 10_000), 19301 + args.shard_index
    )
    errors: list[dict[str, str]] = []
    started = time.time()

    for file_number, path in enumerate(selected_files, start=1):
        try:
            wrapper = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
            sample = wrapper.get("sample", wrapper)
            sample_index = int(wrapper.get("index", sample.get("index", -1)))
            label = str(sample.get("label") or sample.get("metadata", {}).get("instance_id") or "")
            response_length = int(sample["response_length"])
            response_tokens = np.asarray(sample["tokens"][-response_length:], dtype=np.int64)
            student_lp_all = np.asarray(sample["rollout_log_probs"], dtype=np.float64)
            loss_mask = np.asarray(sample["loss_mask"], dtype=np.bool_)
            if response_tokens.shape != (response_length,) or student_lp_all.shape != (
                response_length,
            ) or loss_mask.shape != (response_length,):
                raise ValueError("response token/logprob/loss-mask lengths disagree")

            positions = sample["teacher_candidate_positions"].numpy().astype(np.int64, copy=False)
            candidate_ids = sample["teacher_candidate_ids"].numpy()
            candidate_lp = sample["teacher_candidate_logprobs"].numpy()
            cutoff_lp = sample["teacher_cutoff_logprobs"].numpy().astype(np.float64)
            if candidate_ids.shape != candidate_lp.shape or candidate_ids.shape[0] != len(positions):
                raise ValueError("Teacher candidate cache shapes disagree")
            trainable_exact = loss_mask[positions]
            positions = positions[trainable_exact]
            candidate_ids = candidate_ids[trainable_exact]
            candidate_lp = candidate_lp[trainable_exact]
            cutoff_lp = cutoff_lp[trainable_exact]

            student_ids = response_tokens[positions]
            if student_ids.size and (student_ids.min() < 0 or student_ids.max() >= bridge.size):
                raise ValueError("sampled Student token is outside token bridge")
            teacher_ids = bridge[student_ids]
            if np.any(teacher_ids < 0):
                raise ValueError("trainable sampled Student token is unmapped in bridge")
            columns, hit = rowwise_searchsorted(candidate_ids, teacher_ids)
            hit_rows = np.flatnonzero(hit)
            miss_rows = np.flatnonzero(~hit)
            student_lp = student_lp_all[positions]

            counts["files"] += 1
            counts["response_tokens"] += response_length
            counts["trainable_tokens"] += int(loss_mask.sum())
            counts["exact_trainable_rows"] += len(positions)
            anchor_positions = sample.get("student_anchor_positions")
            if anchor_positions is not None:
                anchor_np = anchor_positions.numpy().astype(np.int64, copy=False)
                counts["anchor_trainable_rows"] += int(loss_mask[anchor_np].sum())
            counts["teacher_cache_hits"] += len(hit_rows)
            counts["teacher_cache_misses"] += len(miss_rows)

            if len(hit_rows):
                teacher_lp = candidate_lp[hit_rows, columns[hit_rows]].astype(np.float64)
                hit_student_lp = student_lp[hit_rows]
                finite = np.isfinite(teacher_lp) & np.isfinite(hit_student_lp)
                counts["nonfinite_hit_rows"] += int((~finite).sum())
                teacher_lp = teacher_lp[finite]
                hit_student_lp = hit_student_lp[finite]
                kept_hit_rows = hit_rows[finite]
                p_teacher = np.exp(teacher_lp)
                p_student = np.exp(hit_student_lp)
                probability_delta = p_teacher - p_student
                logprob_delta = teacher_lp - hit_student_lp

                counts["teacher_probability_higher"] += int((logprob_delta > 0).sum())
                counts["student_probability_higher"] += int((logprob_delta < 0).sum())
                counts["probability_equal"] += int((logprob_delta == 0).sum())
                counts["teacher_materially_higher_10pct"] += int(
                    (logprob_delta > math.log(1.1)).sum()
                )
                counts["student_materially_higher_10pct"] += int(
                    (logprob_delta < -math.log(1.1)).sum()
                )
                counts["within_10pct_ratio"] += int(
                    (np.abs(logprob_delta) <= math.log(1.1)).sum()
                )
                counts["teacher_ge_0.5_student_lt_0.1"] += int(
                    ((p_teacher >= 0.5) & (p_student < 0.1)).sum()
                )
                counts["student_ge_0.5_teacher_lt_0.1"] += int(
                    ((p_student >= 0.5) & (p_teacher < 0.1)).sum()
                )
                counts["teacher_ge_0.8_student_lt_0.1"] += int(
                    ((p_teacher >= 0.8) & (p_student < 0.1)).sum()
                )
                counts["student_ge_0.8_teacher_lt_0.1"] += int(
                    ((p_student >= 0.8) & (p_teacher < 0.1)).sum()
                )
                counts["both_ge_0.5"] += int(
                    ((p_teacher >= 0.5) & (p_student >= 0.5)).sum()
                )
                counts["both_lt_0.1"] += int(
                    ((p_teacher < 0.1) & (p_student < 0.1)).sum()
                )

                histograms["student_prob"] += np.histogram(p_student, PROB_BINS)[0]
                histograms["teacher_prob"] += np.histogram(p_teacher, PROB_BINS)[0]
                histograms["prob_delta"] += np.histogram(probability_delta, PROB_DELTA_BINS)[0]
                histograms["logprob_delta"] += np.histogram(logprob_delta, LOG_DELTA_BINS)[0]
                histograms["probability_2d"] += np.histogram2d(
                    p_student, p_teacher, bins=(PROB_BINS, PROB_BINS)
                )[0].astype(np.int64)
                reservoir.update(
                    {
                        "student_prob": p_student,
                        "teacher_prob": p_teacher,
                        "prob_delta": probability_delta,
                        "logprob_delta": logprob_delta,
                    }
                )

                payload = {
                    "response_position": positions[kept_hit_rows],
                    "student_token_id": student_ids[kept_hit_rows],
                    "teacher_token_id": teacher_ids[kept_hit_rows],
                    "student_prob": p_student,
                    "teacher_prob": p_teacher,
                    "prob_delta": probability_delta,
                    "logprob_delta": logprob_delta,
                }
                update_extremes(
                    extremes["teacher_higher"],
                    logprob_delta,
                    largest=True,
                    limit=args.extreme_count,
                    payload=payload,
                    sample_index=sample_index,
                    label=label,
                )
                update_extremes(
                    extremes["student_higher"],
                    logprob_delta,
                    largest=False,
                    limit=args.extreme_count,
                    payload=payload,
                    sample_index=sample_index,
                    label=label,
                )
                update_extremes(
                    extremes["absolute_logprob_delta"],
                    np.abs(logprob_delta),
                    largest=True,
                    limit=args.extreme_count,
                    payload=payload,
                    sample_index=sample_index,
                    label=label,
                )

                unique_ids, inverse = np.unique(student_ids[kept_hit_rows], return_inverse=True)
                token_count = np.bincount(inverse)
                token_student = np.bincount(inverse, weights=p_student)
                token_teacher = np.bincount(inverse, weights=p_teacher)
                token_delta = np.bincount(inverse, weights=probability_delta)
                token_abs_logdelta = np.bincount(inverse, weights=np.abs(logprob_delta))
                token_teacher_higher = np.bincount(inverse, weights=(logprob_delta > 0))
                for offset, token_id in enumerate(unique_ids.tolist()):
                    values = token_stats.setdefault(token_id, np.zeros(6, dtype=np.float64))
                    values += np.asarray(
                        [
                            token_count[offset],
                            token_student[offset],
                            token_teacher[offset],
                            token_delta[offset],
                            token_abs_logdelta[offset],
                            token_teacher_higher[offset],
                        ]
                    )
            else:
                p_student = np.empty(0)
                p_teacher = np.empty(0)
                probability_delta = np.empty(0)
                logprob_delta = np.empty(0)

            if len(miss_rows):
                miss_student_prob = np.exp(student_lp[miss_rows])
                miss_cutoff_prob = np.exp(cutoff_lp[miss_rows])
                counts["miss_definitely_teacher_lower"] += int(
                    (miss_cutoff_prob < miss_student_prob).sum()
                )
                counts["miss_direction_ambiguous"] += int(
                    (miss_cutoff_prob >= miss_student_prob).sum()
                )
                miss_reservoir.update(
                    {
                        "student_prob": miss_student_prob,
                        "cutoff_prob": miss_cutoff_prob,
                    }
                )

            sequence_stats.append(
                {
                    "sample_index": sample_index,
                    "label": label,
                    "response_tokens": response_length,
                    "trainable_tokens": int(loss_mask.sum()),
                    "exact_rows": len(positions),
                    "cache_hits": len(hit_rows),
                    "cache_misses": len(miss_rows),
                    "mean_student_prob": float(p_student.mean()) if p_student.size else None,
                    "mean_teacher_prob": float(p_teacher.mean()) if p_teacher.size else None,
                    "mean_prob_delta": float(probability_delta.mean()) if probability_delta.size else None,
                    "mean_logprob_delta": float(logprob_delta.mean()) if logprob_delta.size else None,
                    "teacher_higher_fraction": float((logprob_delta > 0).mean()) if logprob_delta.size else None,
                }
            )
        except Exception as exc:  # keep a durable partial shard for diagnosis
            errors.append({"path": str(path), "error": repr(exc)})

        if file_number % args.progress_every == 0 or file_number == len(selected_files):
            elapsed = time.time() - started
            rate = file_number / elapsed if elapsed else 0.0
            remaining = (len(selected_files) - file_number) / rate if rate else math.inf
            print(
                f"shard={args.shard_index}/{args.num_shards} files={file_number}/{len(selected_files)} "
                f"rate={rate:.3f}/s eta={remaining / 60:.1f}m hits={counts['teacher_cache_hits']:,} "
                f"misses={counts['teacher_cache_misses']:,}",
                flush=True,
            )

    output = {
        "format": "sampled_token_teacher_student_shard_v1",
        "input_dir": str(input_dir),
        "bridge": str(Path(args.bridge).resolve()),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "selected_files": len(selected_files),
        "elapsed_seconds": time.time() - started,
        "counts": dict(counts),
        "histograms": histograms,
        "reservoir": reservoir.payload(),
        "miss_reservoir": miss_reservoir.payload(),
        "sequence_stats": sequence_stats,
        "token_stats": token_stats,
        "extremes": extremes,
        "errors": errors,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(output_path)
    print(f"wrote {output_path} errors={len(errors)}", flush=True)
    return 1 if errors else 0


def format_count(value: int | float) -> str:
    return f"{int(value):,}"


def format_pct(numerator: int | float, denominator: int | float) -> str:
    return "—" if not denominator else f"{100 * numerator / denominator:.3f}%"


def format_number(value: float | None, digits: int = 6) -> str:
    if value is None or not math.isfinite(value):
        return "—"
    if value != 0 and abs(value) < 1e-4:
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, quantiles: tuple[float, ...]) -> np.ndarray:
    if not len(values):
        return np.full(len(quantiles), np.nan)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= weights.sum()
    return np.interp(np.asarray(quantiles), cumulative, values)


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if not len(values):
        return math.nan, math.nan
    mean = float(np.average(values, weights=weights))
    variance = float(np.average((values - mean) ** 2, weights=weights))
    return mean, math.sqrt(max(variance, 0.0))


def bin_label(left: float, right: float) -> str:
    def edge(value: float) -> str:
        if np.isneginf(value):
            return "−∞"
        if np.isposinf(value):
            return "+∞"
        if value == 0:
            return "0"
        if abs(value) < 1e-3:
            return f"{value:.0e}"
        return f"{value:g}"

    return f"[{edge(left)}, {edge(right)})"


def histogram_svg(counts: np.ndarray, bins: np.ndarray, title: str, color: str) -> str:
    width, height, left, top, bottom = 1040, 300, 54, 30, 62
    plot_width, plot_height = width - left - 20, height - top - bottom
    maximum = max(int(counts.max()), 1)
    bar_width = plot_width / len(counts)
    bars = []
    for index, count in enumerate(counts.tolist()):
        bar_height = plot_height * count / maximum
        x = left + index * bar_width + 1
        y = top + plot_height - bar_height
        label = bin_label(float(bins[index]), float(bins[index + 1]))
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(bar_width - 2, 1):.2f}" '
            f'height="{bar_height:.2f}" fill="{color}"><title>{html.escape(label)}: '
            f'{count:,} ({100 * count / max(counts.sum(), 1):.3f}%)</title></rect>'
        )
        if index % max(1, len(counts) // 8) == 0:
            bars.append(
                f'<text x="{x:.2f}" y="{height - 34}" font-size="10" '
                f'transform="rotate(35 {x:.2f},{height - 34})">{html.escape(label)}</text>'
            )
    return (
        f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" '
        f'aria-label="{html.escape(title)}"><text x="{left}" y="18" class="chart-title">'
        f'{html.escape(title)}</text><line x1="{left}" y1="{top + plot_height}" '
        f'x2="{left + plot_width}" y2="{top + plot_height}" stroke="#9ca3af"/>'
        + "".join(bars)
        + "</svg>"
    )


def decode_tokenizer(model_path: str | None):
    if not model_path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )


def merge_and_report(args: argparse.Namespace) -> int:
    shard_paths = sorted(Path(args.shard_dir).glob("shard_*.pkl"))
    if not shard_paths:
        raise FileNotFoundError(f"no shard_*.pkl files in {args.shard_dir}")
    shards = []
    for path in shard_paths:
        with path.open("rb") as handle:
            shards.append(pickle.load(handle))
    expected_shards = shards[0]["num_shards"]
    actual_indices = {shard["shard_index"] for shard in shards}
    if actual_indices != set(range(expected_shards)):
        raise ValueError(
            f"incomplete shards: got {sorted(actual_indices)}, expected 0..{expected_shards - 1}"
        )
    errors = [error for shard in shards for error in shard["errors"]]
    if errors:
        raise RuntimeError(f"shard scan errors: {errors[:5]}")

    counts: defaultdict[str, int] = defaultdict(int)
    for shard in shards:
        for key, value in shard["counts"].items():
            counts[key] += int(value)
    histograms = {
        key: sum((shard["histograms"][key] for shard in shards), np.zeros_like(shards[0]["histograms"][key]))
        for key in shards[0]["histograms"]
    }
    sequence_stats = [item for shard in shards for item in shard["sequence_stats"]]
    token_stats: dict[int, np.ndarray] = {}
    for shard in shards:
        for token_id, values in shard["token_stats"].items():
            token_stats.setdefault(token_id, np.zeros(6, dtype=np.float64))[:] += values
    extremes = {}
    for name in shards[0]["extremes"]:
        combined = [item for shard in shards for item in shard["extremes"][name]]
        reverse = name != "student_higher"
        extremes[name] = sorted(combined, key=lambda item: item["score"], reverse=reverse)[: args.extreme_count]

    samples: dict[str, np.ndarray] = {}
    sample_weights: dict[str, np.ndarray] = {}
    for column in RESERVOIR_COLUMNS:
        parts, weights = [], []
        for shard in shards:
            values = shard["reservoir"]["values"][column]
            if len(values):
                parts.append(values)
                weights.append(np.full(len(values), shard["reservoir"]["seen"] / len(values)))
        samples[column] = np.concatenate(parts) if parts else np.empty(0)
        sample_weights[column] = np.concatenate(weights) if weights else np.empty(0)
    miss_samples: dict[str, np.ndarray] = {}
    miss_weights: dict[str, np.ndarray] = {}
    for column in MISS_RESERVOIR_COLUMNS:
        parts, weights = [], []
        for shard in shards:
            values = shard["miss_reservoir"]["values"][column]
            if len(values):
                parts.append(values)
                weights.append(np.full(len(values), shard["miss_reservoir"]["seen"] / len(values)))
        miss_samples[column] = np.concatenate(parts) if parts else np.empty(0)
        miss_weights[column] = np.concatenate(weights) if weights else np.empty(0)

    quantile_values = {
        column: weighted_quantile(samples[column], sample_weights[column], QUANTILES)
        for column in RESERVOIR_COLUMNS
    }
    sampled_mean_std = {
        column: weighted_mean_std(samples[column], sample_weights[column])
        for column in RESERVOIR_COLUMNS
    }
    miss_quantiles = {
        column: weighted_quantile(miss_samples[column], miss_weights[column], QUANTILES)
        for column in MISS_RESERVOIR_COLUMNS
    }
    tokenizer = decode_tokenizer(args.student_model)

    def token_text(token_id: int) -> str:
        if tokenizer is None:
            return ""
        try:
            piece = tokenizer.decode([token_id], skip_special_tokens=False)
            return repr(piece)
        except Exception as exc:
            return f"<decode failed: {exc}>"

    exact_rows = counts["exact_trainable_rows"]
    hits = counts["teacher_cache_hits"] - counts["nonfinite_hit_rows"]
    misses = counts["teacher_cache_misses"]
    token_totals = sum(token_stats.values(), np.zeros(6, dtype=np.float64))
    exact_means = {
        "student_prob": float(token_totals[1] / hits),
        "teacher_prob": float(token_totals[2] / hits),
        "prob_delta": float(token_totals[3] / hits),
        "logprob_delta": float(
            sum(
                item["mean_logprob_delta"] * item["cache_hits"]
                for item in sequence_stats
                if item["mean_logprob_delta"] is not None
            )
            / hits
        ),
        "absolute_logprob_delta": float(token_totals[4] / hits),
    }
    direction_rows = [
        ("Teacher 概率更高", counts["teacher_probability_higher"]),
        ("Student 概率更高", counts["student_probability_higher"]),
        ("数值完全相等", counts["probability_equal"]),
        ("Teacher 至少高 10%（概率比）", counts["teacher_materially_higher_10pct"]),
        ("Student 至少高 10%（概率比）", counts["student_materially_higher_10pct"]),
        ("两者概率比在 ±10% 内", counts["within_10pct_ratio"]),
    ]
    quadrant_rows = [
        ("Teacher ≥ 0.5，Student < 0.1", counts["teacher_ge_0.5_student_lt_0.1"]),
        ("Student ≥ 0.5，Teacher < 0.1", counts["student_ge_0.5_teacher_lt_0.1"]),
        ("Teacher ≥ 0.8，Student < 0.1", counts["teacher_ge_0.8_student_lt_0.1"]),
        ("Student ≥ 0.8，Teacher < 0.1", counts["student_ge_0.8_teacher_lt_0.1"]),
        ("两者均 ≥ 0.5", counts["both_ge_0.5"]),
        ("两者均 < 0.1", counts["both_lt_0.1"]),
    ]

    def rows_html(rows: list[tuple[str, int]], denominator: int) -> str:
        return "".join(
            f"<tr><td>{html.escape(name)}</td><td>{format_count(value)}</td>"
            f"<td>{format_pct(value, denominator)}</td></tr>"
            for name, value in rows
        )

    quantile_headers = "<th>mean</th><th>std（reservoir）</th>" + "".join(
        f"<th>{q:g}</th>" for q in QUANTILES
    )
    quantile_rows = []
    quantile_names = {
        "student_prob": "Student sampled-token probability",
        "teacher_prob": "Teacher sampled-token probability",
        "prob_delta": "p_teacher − p_student",
        "logprob_delta": "log p_teacher − log p_student",
    }
    for column in RESERVOIR_COLUMNS:
        quantile_rows.append(
            f"<tr><td>{html.escape(quantile_names[column])}</td>"
            f"<td>{format_number(exact_means[column])}</td>"
            f"<td>{format_number(sampled_mean_std[column][1])}</td>"
            + "".join(f"<td>{format_number(float(value))}</td>" for value in quantile_values[column])
            + "</tr>"
        )

    probability_labels = probability_bin_labels()
    matrix = histograms["probability_2d"]
    matrix_max = max(int(matrix.max()), 1)
    matrix_header = "<th>Student ↓ / Teacher →</th>" + "".join(
        f"<th>{html.escape(label)}</th>" for label in probability_labels
    )
    matrix_rows = []
    for row_index, label in enumerate(probability_labels):
        cells = []
        for value in matrix[row_index]:
            alpha = 0.04 + 0.86 * math.sqrt(int(value) / matrix_max)
            cells.append(
                f'<td style="background:rgba(37,99,235,{alpha:.3f})" '
                f'title="{int(value):,} ({format_pct(int(value), hits)})">{int(value):,}</td>'
            )
        matrix_rows.append(f"<tr><th>{html.escape(label)}</th>{''.join(cells)}</tr>")

    token_rows = []
    frequent_tokens = sorted(token_stats.items(), key=lambda item: item[1][0], reverse=True)[:50]
    high_disagreement_tokens = sorted(
        (
            (token_id, values)
            for token_id, values in token_stats.items()
            if values[0] >= 1_000
        ),
        key=lambda item: item[1][4] / item[1][0],
        reverse=True,
    )[:50]

    def token_table_rows(items: list[tuple[int, np.ndarray]]) -> str:
        rows = []
        for token_id, values in items:
            count = values[0]
            rows.append(
                "<tr>"
                f"<td>{token_id}</td><td><code>{html.escape(token_text(token_id))}</code></td>"
                f"<td>{format_count(count)}</td><td>{format_number(values[1] / count)}</td>"
                f"<td>{format_number(values[2] / count)}</td><td>{format_number(values[3] / count)}</td>"
                f"<td>{format_number(values[4] / count)}</td><td>{format_pct(values[5], count)}</td>"
                "</tr>"
            )
        return "".join(rows)

    for token_id, values in frequent_tokens:
        count = values[0]
        token_rows.append(
            "<tr>"
            f"<td>{token_id}</td><td><code>{html.escape(token_text(token_id))}</code></td>"
            f"<td>{format_count(count)}</td><td>{format_number(values[1] / count)}</td>"
            f"<td>{format_number(values[2] / count)}</td><td>{format_number(values[3] / count)}</td>"
            f"<td>{format_number(values[4] / count)}</td><td>{format_pct(values[5], count)}</td>"
            "</tr>"
        )
    disagreement_token_rows = token_table_rows(high_disagreement_tokens)

    def extremes_table(items: list[dict[str, Any]]) -> str:
        rows = []
        for item in items[:25]:
            rows.append(
                "<tr>"
                f"<td>{item['sample_index']}</td><td>{html.escape(item['label'])}</td>"
                f"<td>{item['response_position']}</td><td>{item['student_token_id']}</td>"
                f"<td><code>{html.escape(token_text(int(item['student_token_id'])))}</code></td>"
                f"<td>{format_number(item['student_prob'])}</td>"
                f"<td>{format_number(item['teacher_prob'])}</td>"
                f"<td>{format_number(item['prob_delta'])}</td>"
                f"<td>{format_number(item['logprob_delta'])}</td>"
                "</tr>"
            )
        return "".join(rows)

    valid_sequences = [item for item in sequence_stats if item["mean_logprob_delta"] is not None]
    seq_teacher = sorted(valid_sequences, key=lambda item: item["mean_logprob_delta"], reverse=True)[:20]
    seq_student = sorted(valid_sequences, key=lambda item: item["mean_logprob_delta"])[:20]

    def sequence_table(items: list[dict[str, Any]]) -> str:
        return "".join(
            "<tr>"
            f"<td>{item['sample_index']}</td><td>{html.escape(item['label'])}</td>"
            f"<td>{format_count(item['cache_hits'])}</td>"
            f"<td>{format_pct(item['cache_misses'], item['exact_rows'])}</td>"
            f"<td>{format_number(item['mean_student_prob'])}</td>"
            f"<td>{format_number(item['mean_teacher_prob'])}</td>"
            f"<td>{format_number(item['mean_prob_delta'])}</td>"
            f"<td>{format_number(item['mean_logprob_delta'])}</td>"
            "</tr>"
            for item in items
        )

    total_elapsed = sum(float(shard["elapsed_seconds"]) for shard in shards)
    report_generated = datetime.now(timezone.utc).isoformat()
    css = """
body{font-family:Inter,system-ui,-apple-system,sans-serif;margin:0;background:#f6f7fb;color:#172033;line-height:1.55}
main{max-width:1440px;margin:auto;padding:32px}.hero{background:linear-gradient(135deg,#172554,#1d4ed8);color:white;padding:30px;border-radius:18px}
h1{margin:0 0 8px}h2{margin-top:38px;border-bottom:2px solid #dbe4f3;padding-bottom:8px}h3{margin-top:28px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:22px 0}.card{background:white;border:1px solid #e2e8f0;border-radius:13px;padding:17px;box-shadow:0 2px 7px #0f172a0c}.big{font-size:1.55rem;font-weight:700;color:#1d4ed8}.muted{color:#64748b;font-size:.92rem}.warn{background:#fff7ed;border-left:5px solid #ea580c;padding:14px 18px;border-radius:8px}.ok{background:#ecfdf5;border-left:5px solid #059669;padding:14px 18px;border-radius:8px}
table{border-collapse:collapse;width:100%;background:white;font-size:.88rem;display:block;overflow-x:auto}th,td{border:1px solid #dbe3ef;padding:7px 9px;text-align:right;white-space:nowrap}th{background:#eef3fb;font-weight:650}td:first-child,th:first-child{text-align:left}.chart{width:100%;height:auto;background:white;border:1px solid #e2e8f0;border-radius:10px}.chart-title{font-size:14px;font-weight:650}code{white-space:pre-wrap;word-break:break-all}.matrix{font-size:.69rem}.matrix td,.matrix th{padding:4px}.columns{display:grid;grid-template-columns:1fr 1fr;gap:18px}@media(max-width:900px){.columns{grid-template-columns:1fr}main{padding:14px}}
"""
    report = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sampled-token Teacher / Student 概率分析</title><style>{css}</style></head><body><main>
<section class="hero"><h1>Sampled-token Teacher / Student 概率分析</h1><div>GLM-4.5-Air rollout Student × GLM-5.2 Teacher · 全量 Phase-2 sparse Top-1000 cache</div><div class="muted" style="color:#bfdbfe">生成时间 {html.escape(report_generated)}</div></section>
<div class="cards"><div class="card"><div class="big">{format_count(counts['files'])}</div><div>轨迹</div></div><div class="card"><div class="big">{format_count(counts['trainable_tokens'])}</div><div>可训练 token</div></div><div class="card"><div class="big">{format_count(exact_rows)}</div><div>Teacher 精确对齐行</div></div><div class="card"><div class="big">{format_count(hits)}</div><div>sampled token 精确命中</div></div><div class="card"><div class="big">{format_pct(hits, exact_rows)}</div><div>Teacher Top-1000 命中率</div></div><div class="card"><div class="big">{format_count(counts['anchor_trainable_rows'])}</div><div>Student anchor 行</div></div></div>
<p class="warn"><b>口径：</b>Teacher sampled-token 概率只在 sampled token 命中该位置 Teacher Top-1000 cache 时精确可知。{format_count(misses)} 个 miss（{format_pct(misses, exact_rows)}）没有被 cutoff 冒充为真实概率；它们在单独章节作为删失数据分析。所有方向和差值表的分母均为 {format_count(hits)} 个有限精确命中。</p>
<h2>1. 主要结论</h2><div class="cards"><div class="card"><div class="big">{format_pct(counts['student_probability_higher'],hits)}</div><div>Student 概率更高</div></div><div class="card"><div class="big">{format_pct(counts['teacher_probability_higher'],hits)}</div><div>Teacher 概率更高</div></div><div class="card"><div class="big">{format_number(exact_means['logprob_delta'])}</div><div>mean Δlogp</div></div><div class="card"><div class="big">{format_number(exact_means['prob_delta'])}</div><div>mean Δp</div></div><div class="card"><div class="big">{format_number(exact_means['absolute_logprob_delta'])}</div><div>mean |Δlogp|</div></div></div><p><b>总体呈明显的负向不对称：</b>Student sampled token 概率更高的比例约为 Teacher 更高的 {counts['student_probability_higher']/max(counts['teacher_probability_higher'],1):.2f} 倍；“Student ≥ 0.5、Teacher &lt; 0.1”的强分歧有 {format_count(counts['student_ge_0.5_teacher_lt_0.1'])} 个，是反方向强分歧的 {counts['student_ge_0.5_teacher_lt_0.1']/max(counts['teacher_ge_0.5_student_lt_0.1'],1):.2f} 倍。与此同时，中位 Δlogp 仅为 {format_number(float(quantile_values['logprob_delta'][6]))}，说明大量近确定性 token 很接近，主要差异由非对称长尾贡献。</p><div class="columns"><div><h3>概率方向</h3><table><thead><tr><th>类别</th><th>token 数</th><th>占精确命中</th></tr></thead><tbody>{rows_html(direction_rows,hits)}</tbody></table></div><div><h3>高低概率交叉</h3><table><thead><tr><th>类别</th><th>token 数</th><th>占精确命中</th></tr></thead><tbody>{rows_html(quadrant_rows,hits)}</tbody></table></div></div>
<p class="ok"><b>训练含义：</b>初始 sampled-token advantage 正是 <code>A = log p_teacher(a|s) − log p_rollout(a|s)</code>。正值推动 sampled token，负值压低 sampled token；概率差 <code>p_teacher − p_student</code> 更直观，但实际梯度信号采用 log-prob 差。</p>
<h2>2. 分位数</h2><p class="muted">分位数来自按 token 均匀抽取的分片 reservoir；计数和直方图为全量精确值。</p><table><thead><tr><th>变量 / quantile</th>{quantile_headers}</tr></thead><tbody>{''.join(quantile_rows)}</tbody></table>
<h2>3. 正负差值分布</h2>{histogram_svg(histograms['logprob_delta'],LOG_DELTA_BINS,'log p_teacher − log p_student（即初始 advantage）','#2563eb')}{histogram_svg(histograms['prob_delta'],PROB_DELTA_BINS,'p_teacher − p_student','#0f766e')}
<h2>4. Student / Teacher 概率二维分布</h2><p class="muted">行是 Student sampled-token 概率，列是 Teacher 概率；颜色按单元格计数平方根缩放，悬停可查看精确计数与比例。</p><table class="matrix"><thead><tr>{matrix_header}</tr></thead><tbody>{''.join(matrix_rows)}</tbody></table>
<h2>5. Teacher cache miss（删失数据）</h2><div class="cards"><div class="card"><div class="big">{format_count(misses)}</div><div>Top-1000 miss</div></div><div class="card"><div class="big">{format_count(counts['miss_definitely_teacher_lower'])}</div><div>Teacher 必然低于 Student</div><div class="muted">cutoff probability &lt; Student probability</div></div><div class="card"><div class="big">{format_count(counts['miss_direction_ambiguous'])}</div><div>方向无法判定</div></div></div><table><thead><tr><th>删失变量 / quantile</th><th>mean</th><th>std（reservoir）</th>{''.join(f'<th>{q:g}</th>' for q in QUANTILES)}</tr></thead><tbody><tr><td>miss Student probability</td><td>{format_number(weighted_mean_std(miss_samples['student_prob'],miss_weights['student_prob'])[0])}</td><td>{format_number(weighted_mean_std(miss_samples['student_prob'],miss_weights['student_prob'])[1])}</td>{''.join(f'<td>{format_number(float(x))}</td>' for x in miss_quantiles['student_prob'])}</tr><tr><td>Teacher cutoff upper bound</td><td>{format_number(weighted_mean_std(miss_samples['cutoff_prob'],miss_weights['cutoff_prob'])[0])}</td><td>{format_number(weighted_mean_std(miss_samples['cutoff_prob'],miss_weights['cutoff_prob'])[1])}</td>{''.join(f'<td>{format_number(float(x))}</td>' for x in miss_quantiles['cutoff_prob'])}</tr></tbody></table>
<h2>6. sampled token 类型</h2><h3>出现次数最多</h3><table><thead><tr><th>Student token id</th><th>文本</th><th>次数</th><th>mean p_student</th><th>mean p_teacher</th><th>mean Δp</th><th>mean |Δlogp|</th><th>Teacher 更高占比</th></tr></thead><tbody>{''.join(token_rows)}</tbody></table><h3>高频且平均分歧最大（至少出现 1000 次）</h3><table><thead><tr><th>Student token id</th><th>文本</th><th>次数</th><th>mean p_student</th><th>mean p_teacher</th><th>mean Δp</th><th>mean |Δlogp|</th><th>Teacher 更高占比</th></tr></thead><tbody>{disagreement_token_rows}</tbody></table>
<h2>7. 极端 token case</h2><h3>Teacher 相对 Student 最大</h3><table><thead><tr><th>sample</th><th>实例</th><th>response pos</th><th>token id</th><th>token</th><th>p_student</th><th>p_teacher</th><th>Δp</th><th>Δlogp</th></tr></thead><tbody>{extremes_table(extremes['teacher_higher'])}</tbody></table><h3>Student 相对 Teacher 最大</h3><table><thead><tr><th>sample</th><th>实例</th><th>response pos</th><th>token id</th><th>token</th><th>p_student</th><th>p_teacher</th><th>Δp</th><th>Δlogp</th></tr></thead><tbody>{extremes_table(extremes['student_higher'])}</tbody></table>
<h2>8. 轨迹级差异</h2><div class="columns"><div><h3>平均 Δlogp 最偏向 Teacher</h3><table><thead><tr><th>sample</th><th>实例</th><th>hits</th><th>miss率</th><th>mean p_s</th><th>mean p_t</th><th>mean Δp</th><th>mean Δlogp</th></tr></thead><tbody>{sequence_table(seq_teacher)}</tbody></table></div><div><h3>平均 Δlogp 最偏向 Student</h3><table><thead><tr><th>sample</th><th>实例</th><th>hits</th><th>miss率</th><th>mean p_s</th><th>mean p_t</th><th>mean Δp</th><th>mean Δlogp</th></tr></thead><tbody>{sequence_table(seq_student)}</tbody></table></div></div>
<h2>9. 对下一步算法设计的含义</h2><ol><li><b>负 advantage 多并不自动等于实现错误。</b>动作来自 Student 自己的 rollout，因而天然偏向 Student 高概率 token；Teacher 是对这些 Student action 做离策略评价。真正需要关注的是负尾的幅度与方差，而不只是正负 token 数。</li><li><b>当前信号明显由负向长尾主导。</b>中位 Δlogp 接近 0，但均值为 {format_number(exact_means['logprob_delta'])}，1% 分位为 {format_number(float(quantile_values['logprob_delta'][2]))}，而 99% 分位仅为 {format_number(float(quantile_values['logprob_delta'][10]))}。后续可优先比较 advantage clipping、序列内 baseline/centering 或 tail reweighting。</li><li><b>协议与结构 token 应独立审计。</b>极端负向 case 中反复出现换行 token；高频分歧表也包含多种换行组合。这提示模板/结构边界即使通过一对一 token bridge，也可能仍处于不同 Teacher 上下文语义下。</li><li><b>Top-1000 容量不是当前主要误差来源。</b>精确命中率为 {format_pct(hits,exact_rows)}，miss 不足以解释总体负偏。</li><li><b>大量位置几乎无需强更新。</b>{format_pct(counts['within_10pct_ratio'],hits)} 的概率比在 ±10% 内，且 {format_pct(counts['both_ge_0.5'],hits)} 的位置两者都 ≥0.5；可以考虑将计算和梯度预算更多放在有实质分歧的位置。</li></ol>
<h2>10. 方法与可复现信息</h2><ul><li>输入：<code>{html.escape(shards[0]['input_dir'])}</code></li><li>token bridge：<code>{html.escape(shards[0]['bridge'])}</code></li><li>分片：{expected_shards}；累计扫描 CPU 时间：{total_elapsed/3600:.2f} 小时。</li><li>Student 概率来自 Phase-1 <code>rollout_log_probs</code>；Teacher 概率来自 Phase-2 稀疏 cache 对 sampled Teacher token id 的精确查找。</li><li>仅纳入 <code>loss_mask=1</code> 的 Teacher 精确对齐行。Student-anchor 行不具有 Teacher 概率，不进入差值统计。</li><li>Top-1000 miss 的真实 Teacher 概率满足 <code>p_teacher ≤ exp(cutoff_logprob)</code>，因此只在 cutoff 已低于 Student 时能确定 Teacher 更低。</li></ul>
</main></body></html>"""

    output_path = Path(args.report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    summary = {
        "format": "sampled_token_teacher_student_report_v1",
        "generated_at": report_generated,
        "counts": dict(counts),
        "quantiles": {
            key: {str(q): float(value) for q, value in zip(QUANTILES, values, strict=True)}
            for key, values in quantile_values.items()
        },
        "exact_means": exact_means,
        "reservoir_std": {
            key: value[1] for key, value in sampled_mean_std.items()
        },
        "miss_quantiles": {
            key: {str(q): float(value) for q, value in zip(QUANTILES, values, strict=True)}
            for key, values in miss_quantiles.items()
        },
        "direction": {name: value for name, value in direction_rows},
        "quadrants": {name: value for name, value in quadrant_rows},
    }
    output_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"wrote report {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan")
    scan.add_argument("--input-dir", required=True)
    scan.add_argument("--bridge", required=True)
    scan.add_argument("--output", required=True)
    scan.add_argument("--shard-index", type=int, required=True)
    scan.add_argument("--num-shards", type=int, required=True)
    scan.add_argument("--max-files", type=int, default=0)
    scan.add_argument("--reservoir-size", type=int, default=250_000)
    scan.add_argument("--extreme-count", type=int, default=100)
    scan.add_argument("--progress-every", type=int, default=25)
    merge = subparsers.add_parser("merge")
    merge.add_argument("--shard-dir", required=True)
    merge.add_argument("--report", required=True)
    merge.add_argument("--student-model")
    merge.add_argument("--extreme-count", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "scan":
        if not 0 <= args.shard_index < args.num_shards:
            raise ValueError("shard-index must be in [0, num-shards)")
        return scan_shard(args)
    return merge_and_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
