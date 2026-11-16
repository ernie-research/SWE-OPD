import hashlib
import json

import numpy as np
import pytest
import torch

from slime.utils.token_id_bridge import load_token_id_bridge, map_token_ids_torch


def _write_bridge(path, mapping, *, tokenizer_size=None, teacher_vocab_size=None):
    mapping = np.asarray(mapping, dtype=np.int32)
    tokenizer_size = mapping.size if tokenizer_size is None else tokenizer_size
    teacher_vocab_size = mapping.size if teacher_vocab_size is None else teacher_vocab_size
    digest = hashlib.sha256(np.ascontiguousarray(mapping, dtype="<i4").tobytes()).hexdigest()
    path.write_text(
        json.dumps(
            {
                "format": "offline_opd_token_id_bridge_v1",
                "student_tokenizer_size": tokenizer_size,
                "student_model_vocab_size": mapping.size,
                "teacher_tokenizer_size": teacher_vocab_size,
                "teacher_model_vocab_size": teacher_vocab_size,
                "mapping_sha256": digest,
                "student_to_teacher": mapping.tolist(),
            }
        )
    )
    return str(path)


def test_token_id_bridge_maps_and_rejects_padded_ids(tmp_path):
    path = _write_bridge(
        tmp_path / "bridge.json",
        [3, 1, 4, 0, -1, -1],
        tokenizer_size=4,
        teacher_vocab_size=5,
    )
    bridge = load_token_id_bridge(path)
    np.testing.assert_array_equal(
        bridge.map_numpy(np.array([[0, 3], [2, 1]], dtype=np.int32)),
        np.array([[3, 0], [4, 1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        bridge.map_teacher_numpy(
            np.array([[3, 2], [4, 0]], dtype=np.int32), allow_unmapped=True
        ),
        np.array([[0, -1], [2, 3]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="teacher-only/unmapped"):
        bridge.map_teacher_numpy(np.array([2], dtype=np.int32))
    torch.testing.assert_close(
        map_token_ids_torch(torch.tensor([2, 0]), path), torch.tensor([4, 3])
    )
    with pytest.raises(ValueError, match="unmapped/padded"):
        bridge.map_numpy(np.array([4], dtype=np.int32))


def test_token_id_bridge_rejects_hash_mismatch(tmp_path):
    path = tmp_path / "bridge.json"
    _write_bridge(path, [0, 1, 2])
    payload = json.loads(path.read_text())
    payload["student_to_teacher"][1] = 2
    path.write_text(json.dumps(payload))
    load_token_id_bridge.cache_clear()
    with pytest.raises(ValueError, match="hash mismatch"):
        load_token_id_bridge(str(path))
