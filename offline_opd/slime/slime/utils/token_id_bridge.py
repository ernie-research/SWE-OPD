"""Optional one-to-one token-id bridge for cross-tokenizer OPD.

Ordinary OPD uses one tokenizer and never calls this module; offline runs whose student and
teacher assign different ids to the same token opt in with ``--opd-token-id-bridge``. The
artifact is a small JSON file so it is inspectable and can be hashed into experiment metadata.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch


_FORMAT = "offline_opd_token_id_bridge_v1"


@dataclass(frozen=True)
class TokenIdBridge:
    path: str
    student_to_teacher: np.ndarray
    teacher_to_student: np.ndarray
    student_tokenizer_size: int
    student_model_vocab_size: int
    teacher_tokenizer_size: int
    teacher_model_vocab_size: int
    mapping_sha256: str
    # Number of valid student ids with no teacher surface counterpart. Zero for a
    # same-family pair; nonzero for a genuinely different teacher vocabulary.
    unmapped_student_ids: int = 0

    def map_numpy(
        self,
        token_ids,
        *,
        context: str = "token ids",
        allow_unmapped: bool = False,
    ) -> np.ndarray:
        """Map Student ids to Teacher ids.

        A genuinely different teacher vocabulary cannot cover every student token.
        Callers that only compare the mapped id against one candidate teacher id and
        skip on mismatch (canonical span alignment) ask for ``allow_unmapped``: the
        ``-1`` then simply fails that comparison, and the position is routed to the
        Student-anchor group, whose loss consults no teacher at all.  The strict
        default remains for callers that must have a complete mapping.
        """
        ids = np.asarray(token_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"{context}: expected integer ids, got {ids.dtype}")
        if ids.size:
            low, high = int(ids.min()), int(ids.max())
            if low < 0 or high >= self.student_to_teacher.size:
                raise ValueError(
                    f"{context}: student id range [{low}, {high}] is outside bridge "
                    f"[0, {self.student_to_teacher.size})"
                )
        mapped = self.student_to_teacher[ids]
        if not allow_unmapped and np.any(mapped < 0):
            bad = np.unique(ids[mapped < 0])[:16].tolist()
            raise ValueError(f"{context}: unmapped/padded student token ids: {bad}")
        return np.ascontiguousarray(mapped, dtype=np.int64)

    def map_teacher_numpy(
        self,
        token_ids,
        *,
        context: str = "teacher token ids",
        allow_unmapped: bool = False,
    ) -> np.ndarray:
        """Map Teacher ids back to Student ids.

        Cross-tokenizer forward-KL asks for ``allow_unmapped``: Teacher-only vocabulary
        entries stay ``-1`` so the caller can skip them and continue farther down the cached
        Teacher ranking. Callers that need a complete mapping keep the strict default.
        """
        ids = np.asarray(token_ids)
        if not np.issubdtype(ids.dtype, np.integer):
            raise TypeError(f"{context}: expected integer ids, got {ids.dtype}")
        if ids.size:
            low, high = int(ids.min()), int(ids.max())
            if low < 0 or high >= self.teacher_to_student.size:
                raise ValueError(
                    f"{context}: teacher id range [{low}, {high}] is outside bridge "
                    f"[0, {self.teacher_to_student.size})"
                )
        mapped = self.teacher_to_student[ids]
        if not allow_unmapped and np.any(mapped < 0):
            bad = np.unique(ids[mapped < 0])[:16].tolist()
            raise ValueError(f"{context}: teacher-only/unmapped token ids: {bad}")
        return np.ascontiguousarray(mapped, dtype=np.int64)


def _canonical_mapping_bytes(mapping: np.ndarray) -> bytes:
    return np.ascontiguousarray(mapping, dtype="<i4").tobytes()


@lru_cache(maxsize=8)
def load_token_id_bridge(path: str) -> TokenIdBridge:
    resolved = str(Path(path).expanduser().resolve())
    payload = json.loads(Path(resolved).read_text())
    if payload.get("format") != _FORMAT:
        raise ValueError(
            f"unsupported token-id bridge format in {resolved}: {payload.get('format')!r}"
        )
    mapping = np.asarray(payload.get("student_to_teacher"), dtype=np.int32)
    if mapping.ndim != 1:
        raise ValueError(f"token-id bridge mapping must be 1D, got {mapping.shape}")
    model_vocab = int(payload["student_model_vocab_size"])
    tokenizer_size = int(payload["student_tokenizer_size"])
    if mapping.size != model_vocab or not 0 < tokenizer_size <= model_vocab:
        raise ValueError(
            "token-id bridge size mismatch: "
            f"mapping={mapping.size}, tokenizer={tokenizer_size}, model={model_vocab}"
        )
    unmapped_student_ids = int(np.count_nonzero(mapping[:tokenizer_size] < 0))
    if np.any(mapping[tokenizer_size:] >= 0):
        bad = (np.flatnonzero(mapping[tokenizer_size:] >= 0) + tokenizer_size)[:16].tolist()
        raise ValueError(f"token-id bridge maps padded student ids: {bad}")
    digest = hashlib.sha256(_canonical_mapping_bytes(mapping)).hexdigest()
    expected_digest = payload.get("mapping_sha256")
    if expected_digest and digest != expected_digest:
        raise ValueError(
            f"token-id bridge hash mismatch in {resolved}: {digest} != {expected_digest}"
        )
    # Injectivity and the teacher-range check apply to the *mapped* subset only.
    # -1 is not an id; np.unique would otherwise collapse every unmapped entry into
    # one value and report a spurious collision.
    mapped_valid = mapping[:tokenizer_size]
    mapped_present = mapped_valid[mapped_valid >= 0]
    if np.unique(mapped_present).size != mapped_present.size:
        raise ValueError("token-id bridge must be injective over mapped student tokens")
    teacher_model_vocab = int(payload["teacher_model_vocab_size"])
    if mapped_present.size and int(mapped_present.max()) >= teacher_model_vocab:
        raise ValueError("token-id bridge contains an id outside teacher model vocabulary")
    teacher_to_student = np.full(teacher_model_vocab, -1, dtype=np.int32)
    # Scatter only the mapped entries. Passing the raw slice would let numpy treat
    # every -1 as a negative index and silently overwrite teacher_to_student[-1].
    student_ids = np.arange(tokenizer_size, dtype=np.int32)
    teacher_to_student[mapped_present] = student_ids[mapped_valid >= 0]
    return TokenIdBridge(
        path=resolved,
        student_to_teacher=mapping,
        teacher_to_student=teacher_to_student,
        student_tokenizer_size=tokenizer_size,
        student_model_vocab_size=model_vocab,
        teacher_tokenizer_size=int(payload["teacher_tokenizer_size"]),
        teacher_model_vocab_size=teacher_model_vocab,
        mapping_sha256=digest,
        unmapped_student_ids=unmapped_student_ids,
    )


_DEVICE_MAP_CACHE: dict[tuple[str, str], torch.Tensor] = {}


def map_token_ids_torch(
    token_ids: torch.Tensor,
    path: str,
    *,
    context: str = "token ids",
    allow_unmapped: bool = False,
) -> torch.Tensor:
    """Map a student-id tensor to teacher ids on the tensor's current device.

    ``allow_unmapped`` keeps the ``-1`` sentinel for student ids with no teacher
    surface counterpart.  The sparse teacher-cache lookups locate a query with
    ``searchsorted`` over non-negative, ascending candidate ids and report a
    miss unless the gathered id compares equal, so ``-1`` degrades into exactly
    the same clean miss as an id outside the cached Top-K -- which is the
    truthful statement for a surface string the teacher vocabulary cannot
    express.  Callers that must have a complete mapping keep the strict default.
    """
    if token_ids.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{context}: expected integer ids, got {token_ids.dtype}")
    bridge = load_token_id_bridge(path)
    if token_ids.numel():
        low, high = int(token_ids.min()), int(token_ids.max())
        if low < 0 or high >= bridge.student_model_vocab_size:
            raise ValueError(
                f"{context}: student id range [{low}, {high}] is outside bridge "
                f"[0, {bridge.student_model_vocab_size})"
            )
    key = (bridge.path, str(token_ids.device))
    mapping = _DEVICE_MAP_CACHE.get(key)
    if mapping is None or mapping.device != token_ids.device:
        mapping = torch.from_numpy(bridge.student_to_teacher.astype(np.int64, copy=False)).to(
            token_ids.device
        )
        _DEVICE_MAP_CACHE[key] = mapping
    mapped = mapping[token_ids.long()]
    if not allow_unmapped and bool((mapped < 0).any()):
        bad = torch.unique(token_ids[mapped < 0].detach().cpu())[:16].tolist()
        raise ValueError(f"{context}: unmapped/padded student token ids: {bad}")
    return mapped
