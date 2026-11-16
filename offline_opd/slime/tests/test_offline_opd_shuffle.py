import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from examples.coding_agent_opd.scripts.offline_opd import score_teacher_offline
from examples.coding_agent_opd.scripts.offline_opd.build_scored_manifests import (
    build_manifests,
)
from examples.coding_agent_opd.scripts.offline_opd.shuffle_phase1 import (
    MANIFEST_NAME,
    _balanced_batches,
    shuffle_phase1,
)
from slime.ray.rollout import _load_debug_rollout_samples
from slime.utils.types import Sample


def _sample(index: int, trainable_tokens: int) -> dict:
    return {
        "index": index,
        "tokens": list(range(trainable_tokens + 3)),
        "loss_mask": [1] * trainable_tokens,
        "student_topk_ids": np.zeros((trainable_tokens, 2), dtype=np.int32),
    }


def test_shuffle_rebalances_uneven_sources_into_requested_file_size(tmp_path: Path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    costs = [100, 90, 80, 70, 10, 10, 10, 10]
    samples = [_sample(index, cost) for index, cost in enumerate(costs)]
    # The source layout deliberately differs from the requested final layout.
    torch.save({"rollout_id": 0, "samples": samples[:3]}, input_dir / "rollout_0.pt")
    torch.save({"rollout_id": 1, "samples": samples[3:]}, input_dir / "rollout_1.pt")

    shuffle_phase1(
        input_dir,
        output_dir,
        seed=17,
        samples_per_file=4,
        balance_key="trainable_tokens",
    )

    outputs = [
        torch.load(output_dir / f"rollout_{index}.pt", weights_only=False)
        for index in range(2)
    ]
    assert [len(blob["samples"]) for blob in outputs] == [4, 4]
    assert sorted(sample["index"] for blob in outputs for sample in blob["samples"]) == list(range(8))
    manifest = json.loads((output_dir / MANIFEST_NAME).read_text())
    assert manifest["samples_per_file"] == 4
    assert manifest["balance_key"] == "trainable_tokens"
    assert manifest["batch_costs"] == [190, 190]

    # A matching published output is reusable and must not be rewritten.
    mtimes = [path.stat().st_mtime_ns for path in output_dir.glob("rollout_*.pt")]
    shuffle_phase1(
        input_dir,
        output_dir,
        seed=17,
        samples_per_file=4,
        balance_key="trainable_tokens",
    )
    assert [path.stat().st_mtime_ns for path in output_dir.glob("rollout_*.pt")] == mtimes


def test_shuffle_rejects_partial_training_batch():
    samples = [_sample(index, 1) for index in range(5)]
    with pytest.raises(ValueError, match="not divisible"):
        _balanced_batches(
            samples,
            [1] * len(samples),
            samples_per_file=2,
            seed=1,
            balance_key="trainable_tokens",
        )


def test_dynamic_phase2_claims_each_file_once(tmp_path: Path, monkeypatch):
    files = [tmp_path / f"rollout_{index}.pt" for index in range(10)]
    processed: list[str] = []

    monkeypatch.setattr(score_teacher_offline, "discover_rollout_files", lambda _: files)
    monkeypatch.setattr(score_teacher_offline, "_already_scored", lambda *args, **kwargs: False)

    async def fake_process_file(args, in_path, out_path, *unused):
        await asyncio.sleep(0)
        processed.append(Path(in_path).name)

    monkeypatch.setattr(score_teacher_offline, "_process_file", fake_process_file)

    common = dict(
        rm_url="http://teacher/generate",
        opd_id_topk=1024,
        in_dir=str(tmp_path),
        out_dir=str(tmp_path / "scored"),
        overwrite=False,
        require_student_topk_logprobs=True,
        concurrency=8,
        max_attempts=3,
        num_shards=4,
        dynamic_sharding=True,
        sample_output=False,
        coord_dir=str(tmp_path / "claims"),
    )

    async def run_all():
        await asyncio.gather(
            *(
                score_teacher_offline._main_async(
                    SimpleNamespace(**common, shard_index=rank)
                )
                for rank in range(4)
            )
        )

    asyncio.run(run_all())
    assert sorted(processed) == sorted(path.name for path in files)
    assert len(processed) == len(set(processed)) == 10


def test_scored_sample_manifests_balance_without_rewriting(tmp_path: Path):
    (tmp_path / "samples").mkdir()
    (tmp_path / "source_manifests").mkdir()
    costs = [100, 90, 80, 70, 10, 10, 10, 10]
    source_rows = [[], []]
    for index, cost in enumerate(costs):
        relative = f"samples/sample_{index:06d}.pt"
        sample = Sample(index=index, rollout_id=99, tokens=[1, 2], response_length=1, loss_mask=[1])
        torch.save(
            {
                "format": "offline_opd_scored_sample_v1",
                "index": index,
                "sample": sample.to_dict(),
            },
            tmp_path / relative,
        )
        source_rows[index // 4].append(
            {"index": index, "path": relative, "trainable_tokens": cost}
        )
    for source_id, rows in enumerate(source_rows):
        (tmp_path / "source_manifests" / f"rollout_{source_id}.json").write_text(
            json.dumps(
                {
                    "format": "offline_opd_scored_source_v1",
                    "source_rollout_id": source_id,
                    "samples": rows,
                }
            )
        )

    summary = build_manifests(tmp_path, samples_per_file=4, seed=17)
    assert summary["batch_trainable_tokens"] == [190, 190]
    loaded = []
    for rollout_id in range(2):
        samples = _load_debug_rollout_samples(
            str(tmp_path / f"rollout_{rollout_id}.pt"), rollout_id
        )
        assert len(samples) == 4
        assert [sample.rollout_id for sample in samples] == [sample.index for sample in samples]
        assert len({sample.rollout_id for sample in samples}) == len(samples)
        loaded.extend(sample.index for sample in samples)
    assert sorted(loaded) == list(range(8))


def test_scored_manifests_can_preserve_phase1_batch_layout(tmp_path: Path):
    (tmp_path / "samples").mkdir()
    (tmp_path / "source_manifests").mkdir()
    source_layout = [[2, 0], [3, 1]]
    costs = [10, 20, 30, 40]
    for index, cost in enumerate(costs):
        relative = f"samples/sample_{index:06d}.pt"
        torch.save({"index": index}, tmp_path / relative)
    for source_id, indices in enumerate(source_layout):
        rows = [
            {
                "index": index,
                "path": f"samples/sample_{index:06d}.pt",
                "trainable_tokens": costs[index],
            }
            for index in indices
        ]
        (tmp_path / "source_manifests" / f"rollout_{source_id}.json").write_text(
            json.dumps(
                {
                    "format": "offline_opd_scored_source_v1",
                    "source_rollout_id": source_id,
                    "samples": rows,
                }
            )
        )

    summary = build_manifests(
        tmp_path,
        samples_per_file=2,
        seed=999,
        balance_strategy="phase1",
    )

    assert summary["balance_strategy"] == "phase1"
    assert summary["batch_trainable_tokens"] == [40, 60]
    for rollout_id, expected_indices in enumerate(source_layout):
        manifest = torch.load(
            tmp_path / f"rollout_{rollout_id}.pt", weights_only=False
        )
        assert manifest["sample_indices"] == expected_indices
        assert manifest["sample_files"] == [
            f"samples/sample_{index:06d}.pt" for index in expected_indices
        ]


def test_phase1_manifest_layout_rejects_different_batch_size(tmp_path: Path):
    (tmp_path / "samples").mkdir()
    (tmp_path / "source_manifests").mkdir()
    rows = []
    for index in range(2):
        relative = f"samples/sample_{index:06d}.pt"
        torch.save({"index": index}, tmp_path / relative)
        rows.append({"index": index, "path": relative, "trainable_tokens": 1})
    (tmp_path / "source_manifests" / "rollout_0.json").write_text(
        json.dumps(
            {
                "format": "offline_opd_scored_source_v1",
                "source_rollout_id": 0,
                "samples": rows,
            }
        )
    )

    with pytest.raises(ValueError, match="cannot be preserved"):
        build_manifests(
            tmp_path,
            samples_per_file=1,
            seed=1,
            balance_strategy="phase1",
        )
