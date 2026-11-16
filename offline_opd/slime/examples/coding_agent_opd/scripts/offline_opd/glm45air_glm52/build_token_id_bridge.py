#!/usr/bin/env python3
"""Build and verify the GLM-4.5-Air -> GLM-5.2 token-id bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoTokenizer


FORMAT = "offline_opd_token_id_bridge_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    student_path = Path(args.student_model).resolve()
    teacher_path = Path(args.teacher_model).resolve()
    output = Path(args.output).resolve()
    student = AutoTokenizer.from_pretrained(student_path, trust_remote_code=True)
    teacher = AutoTokenizer.from_pretrained(teacher_path, trust_remote_code=True)
    student_config = AutoConfig.from_pretrained(student_path, trust_remote_code=True)
    teacher_config = AutoConfig.from_pretrained(teacher_path, trust_remote_code=True)

    student_tokenizer_size = len(student)
    teacher_tokenizer_size = len(teacher)
    student_model_vocab_size = int(student_config.vocab_size)
    teacher_model_vocab_size = int(teacher_config.vocab_size)
    if student_tokenizer_size > student_model_vocab_size:
        raise ValueError("student tokenizer is larger than its model vocabulary")
    if teacher_tokenizer_size > teacher_model_vocab_size:
        raise ValueError("teacher tokenizer is larger than its model vocabulary")

    teacher_vocab = teacher.get_vocab()
    mapping = np.full(student_model_vocab_size, -1, dtype=np.int32)
    missing: list[tuple[int, str]] = []
    for student_id in range(student_tokenizer_size):
        token = student.convert_ids_to_tokens(student_id)
        teacher_id = teacher_vocab.get(token)
        if teacher_id is None:
            missing.append((student_id, token))
        else:
            mapping[student_id] = int(teacher_id)
    if missing:
        raise ValueError(
            f"teacher tokenizer is missing {len(missing)} student tokens; first={missing[:16]}"
        )
    mapped = mapping[:student_tokenizer_size]
    if np.unique(mapped).size != mapped.size:
        raise ValueError("student->teacher token mapping is not injective")

    # For this model pair the base vocabulary is an identical prefix.  Keep this
    # as a hard invariant so a silently changed tokenizer cannot contaminate an
    # expensive offline teacher cache.
    student_base_vocab = int(student.vocab_size)
    identity_prefix = 0
    while identity_prefix < student_base_vocab and mapping[identity_prefix] == identity_prefix:
        identity_prefix += 1
    if identity_prefix != student_base_vocab:
        raise ValueError(
            f"shared GLM base vocabulary stopped being identity-mapped at id {identity_prefix}"
        )

    mapping_sha256 = hashlib.sha256(
        np.ascontiguousarray(mapping, dtype="<i4").tobytes()
    ).hexdigest()
    payload = {
        "format": FORMAT,
        "student_model": str(student_path),
        "teacher_model": str(teacher_path),
        "student_tokenizer_size": student_tokenizer_size,
        "student_base_vocab_size": student_base_vocab,
        "student_model_vocab_size": student_model_vocab_size,
        "teacher_tokenizer_size": teacher_tokenizer_size,
        "teacher_base_vocab_size": int(teacher.vocab_size),
        "teacher_model_vocab_size": teacher_model_vocab_size,
        "identity_prefix_size": identity_prefix,
        "student_tokenizer_sha256": sha256_file(student_path / "tokenizer.json"),
        "teacher_tokenizer_sha256": sha256_file(teacher_path / "tokenizer.json"),
        "mapping_sha256": mapping_sha256,
        "student_to_teacher": mapping.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    os.replace(temporary, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "student_tokenizer_size": student_tokenizer_size,
                "student_padding_ids": student_model_vocab_size - student_tokenizer_size,
                "teacher_tokenizer_size": teacher_tokenizer_size,
                "teacher_only_tokens": teacher_tokenizer_size - student_tokenizer_size,
                "identity_prefix_size": identity_prefix,
                "mapping_sha256": mapping_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

