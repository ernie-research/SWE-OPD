import importlib.util
from pathlib import Path

import pytest
import torch

from examples.coding_agent_opd.scripts.offline_opd.offline_utils import load_rollout_file
from slime.utils.types import Sample


MODULE_PATH = (
    Path(__file__).parents[1]
    / "examples/coding_agent_opd/scripts/offline_opd/rebatch_scored_epochs.py"
)
SPEC = importlib.util.spec_from_file_location("rebatch_scored_epochs", MODULE_PATH)
rebatch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(rebatch)


def _make_scored_dir(root: Path, num_samples: int = 4, source_batch: int = 2) -> Path:
    scored = root / "scored"
    samples = scored / "samples"
    samples.mkdir(parents=True)
    for index in range(num_samples):
        torch.save(
            {
                "format": "offline_opd_scored_sample_v1",
                "index": index,
                "sample": Sample(index=index).to_dict(),
            },
            samples / f"sample_{index:06d}.pt",
        )
    for rollout_id, start in enumerate(range(0, num_samples, source_batch)):
        indices = list(range(start, start + source_batch))
        torch.save(
            {
                "format": "offline_opd_sample_manifest_v1",
                "rollout_id": rollout_id,
                "sample_files": [f"samples/sample_{index:06d}.pt" for index in indices],
                "sample_indices": indices,
            },
            scored / f"rollout_{rollout_id}.pt",
        )
    return scored


def _manifest_indices(directory: Path) -> list[list[int]]:
    result = []
    rollout_id = 0
    while (path := directory / f"rollout_{rollout_id}.pt").is_file():
        result.append(torch.load(path, weights_only=False)["sample_indices"])
        rollout_id += 1
    return result


def test_divisible_batch_does_not_cross_epoch(tmp_path):
    scored = _make_scored_dir(tmp_path)
    output = scored / "epochs_x2_b2"

    summary = rebatch.build_rebatched_epochs(
        scored, output, epochs=2, batch_size=2, shuffle=False
    )

    assert summary["cross_epoch"] is False
    assert summary["padding_samples"] == 0
    assert _manifest_indices(output) == [[0, 1], [2, 3], [0, 1], [2, 3]]


def test_nondivisible_batch_continues_from_next_epoch(tmp_path):
    scored = _make_scored_dir(tmp_path)
    output = scored / "epochs_x2_b3"

    summary = rebatch.build_rebatched_epochs(
        scored, output, epochs=2, batch_size=3, shuffle=False
    )

    assert summary["cross_epoch"] is True
    assert summary["padding_samples"] == 1
    assert _manifest_indices(output) == [[0, 1, 2], [3, 0, 1], [2, 3, 0]]
    first = torch.load(output / "rollout_0.pt", weights_only=False)
    assert first["format"] == "offline_opd_sample_manifest_v2"
    assert (output / first["sample_root"]).resolve() == scored.resolve()
    samples, rollout_id = load_rollout_file(output / "rollout_1.pt")
    assert rollout_id == 1
    assert [sample.index for sample in samples] == [3, 0, 1]


def test_shuffled_cross_epoch_batch_has_no_duplicate_samples(tmp_path):
    scored = _make_scored_dir(tmp_path)
    output = scored / "epochs_x2_b3_shuffled"

    rebatch.build_rebatched_epochs(
        scored, output, epochs=2, batch_size=3, shuffle=True
    )

    batches = _manifest_indices(output)
    assert all(len(batch) == len(set(batch)) for batch in batches)
    stream = [index for batch in batches for index in batch]
    # Boundary repair may change order, but each complete epoch is still an
    # exact permutation of the source dataset.
    assert set(stream[:4]) == set(range(4))
    assert set(stream[4:8]) == set(range(4))


def test_explicit_rollout_count_caps_partial_final_epoch(tmp_path):
    scored = _make_scored_dir(tmp_path)
    output = scored / "epochs_x3_b3_s3"

    summary = rebatch.build_rebatched_epochs(
        scored, output, epochs=3, batch_size=3, shuffle=False, num_rollouts=3
    )

    assert summary["num_rollouts"] == 3
    assert summary["emitted_samples"] == 9
    assert summary["truncated_samples"] == 3
    assert _manifest_indices(output) == [[0, 1, 2], [3, 0, 1], [2, 3, 0]]


def test_external_sample_symlink_requires_explicit_opt_in(tmp_path):
    scored = _make_scored_dir(tmp_path / "current", num_samples=2, source_batch=1)
    reused = _make_scored_dir(tmp_path / "reused", num_samples=1, source_batch=1)
    logical_sample = scored / "samples/sample_000000.pt"
    logical_sample.unlink()
    logical_sample.symlink_to(reused / "samples/sample_000000.pt")

    with pytest.raises(ValueError, match="sample escapes source root"):
        rebatch.build_rebatched_epochs(
            scored, scored / "epochs_default", epochs=1, batch_size=1, shuffle=False
        )

    output = scored / "epochs_external_symlink"
    summary = rebatch.build_rebatched_epochs(
        scored,
        output,
        epochs=1,
        batch_size=1,
        shuffle=False,
        allow_external_sample_symlinks=True,
    )

    assert summary["allow_external_sample_symlinks"] is True
    first = torch.load(output / "rollout_0.pt", weights_only=False)
    assert first["sample_files"] == ["samples/sample_000000.pt"]
    samples, rollout_id = load_rollout_file(output / "rollout_0.pt")
    assert rollout_id == 0
    assert [sample.index for sample in samples] == [0]
