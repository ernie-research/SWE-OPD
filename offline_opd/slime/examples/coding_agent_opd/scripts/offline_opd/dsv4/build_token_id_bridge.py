#!/usr/bin/env python3
"""GLM-4.5-Air -> DeepSeek-V4-Flash token-id bridge builder.

Same output format as glm45air_glm52/build_token_id_bridge.py, with the two checks that
fail on a genuinely different vocabulary relaxed: Student tokens with no teacher
counterpart stay -1, and the identity-prefix invariant is recorded, not enforced.
Injectivity over the mapped subset and sha256 pinning are kept.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from transformers import AutoConfig, AutoTokenizer

FORMAT = "offline_opd_token_id_bridge_v1"

# Functional (not surface-string) pairs for the protocol tokens, which surface matching
# leaves at -1: each pair is the token the two templates emit at the same decision
# ("start a tool call", "end the turn"). <|observation|> is GLM's generated end-of-turn
# stop; <arg_key>/<arg_value> have no 1:1 counterpart, since DSV4 writes parameters as text.
#
# Each entry: (student token, teacher token, alignment policy), read by
# teacher_trajectory._add_protocol_pairs:
#   auto                        zip occurrences in order when counts match, else pair
#                               first + last
#   student_last_teacher_first  the Student's LAST occurrence in the turn (the call the
#                               parser keeps) with the teacher's FIRST (the block opener)
PROTOCOL_PAIR_SETS = {
    # DeepSeek-V4-Flash-Code-SFT-V0.3.0 (ernie chat format)
    "sft": (
        ("<tool_call>", "<ernie:tool_call>", "auto"),
        ("</tool_call>", "</ernie:tool_call>", "auto"),
        ("<|observation|>", "<|im_end|>", "auto"),
    ),
    # DeepSeek-V4-Flash-0731 (official encoding_dsv4.py): a tool call opens with
    # <|DSML|tool_calls>, so "start a tool call" is the turn's first |DSML| (the id recurs
    # inside the block, hence the policy). The closer has no 1:1 token, so </tool_call>
    # stays unmapped (Student anchor). <|tool_calls_begin|> is never emitted.
    "v0731": (
        ("<tool_call>", "｜DSML｜", "student_last_teacher_first"),
        ("<|observation|>", "<｜end▁of▁sentence｜>", "auto"),
    ),
}
PROTOCOL_PAIRS = PROTOCOL_PAIR_SETS["sft"]


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
    parser.add_argument(
        "--protocol-pairs",
        nargs="?",
        const="sft",
        default=None,
        choices=sorted(PROTOCOL_PAIR_SETS),
        help="also map the GLM protocol tokens to the teacher's functional counterparts "
        "(bare flag = 'sft', kept for the 10-16 SFT bridge)",
    )
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
    missing = 0
    for student_id in range(student_tokenizer_size):
        token = student.convert_ids_to_tokens(student_id)
        teacher_id = teacher_vocab.get(token)
        if teacher_id is None:
            missing += 1
        else:
            mapping[student_id] = int(teacher_id)

    protocol_pairs = []
    if args.protocol_pairs:
        for student_token, teacher_token, policy in PROTOCOL_PAIR_SETS[args.protocol_pairs]:
            student_id = student.convert_tokens_to_ids(student_token)
            teacher_id = teacher_vocab.get(teacher_token)
            if student_id is None or student_id == student.unk_token_id or student_id >= student_tokenizer_size:
                raise ValueError(f"student tokenizer has no {student_token!r}")
            if teacher_id is None:
                raise ValueError(f"teacher tokenizer has no {teacher_token!r}")
            if mapping[student_id] >= 0 and int(mapping[student_id]) != int(teacher_id):
                raise ValueError(
                    f"{student_token!r} already maps to teacher {int(mapping[student_id])}"
                )
            if mapping[student_id] < 0:
                missing -= 1
            mapping[student_id] = int(teacher_id)
            entry = [student_token, int(student_id), teacher_token, int(teacher_id)]
            if policy != "auto":
                # "auto" is left implicit, keeping older "auto"-only bridges byte-identical.
                entry.append(policy)
            protocol_pairs.append(entry)

    mapped = mapping[:student_tokenizer_size]
    mapped_valid = mapped[mapped >= 0]
    if np.unique(mapped_valid).size != mapped_valid.size:
        raise ValueError("student->teacher mapping is not injective over mapped ids")

    student_base_vocab = int(student.vocab_size)
    identity_prefix = 0
    while identity_prefix < student_base_vocab and mapping[identity_prefix] == identity_prefix:
        identity_prefix += 1

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
        "protocol_pairs": protocol_pairs,
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
                "student_unmapped_tokens": missing,
                "student_mapped_tokens": student_tokenizer_size - missing,
                "teacher_tokenizer_size": teacher_tokenizer_size,
                "identity_prefix_size": identity_prefix,
                "mapping_sha256": mapping_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
