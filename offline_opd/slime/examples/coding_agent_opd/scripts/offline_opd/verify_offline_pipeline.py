#!/usr/bin/env python3
"""Validate Phase-1/2 offline OPD artifacts or bake a small live-teacher probe."""

from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace

from examples.coding_agent_opd.scripts.offline_opd.offline_utils import (
    load_rollout_file,
    validate_baked_sample,
    validate_cross_tokenizer_anchor_scored_sample,
    validate_dump_sample,
    validate_dynamic_scored_sample,
    validate_rollout_directory,
    validate_scored_sample,
    validate_token_dump_sample,
    validate_token_scored_sample,
    truncate_dynamic_scored_sample_support,
)


def check_file(
    path: str,
    k: int,
    *,
    require_teacher: bool,
    require_student_topk_logprobs: bool,
    require_sparse_teacher: bool,
    opd_loss_type: str,
    token_topk_diagnostics: bool,
    dynamic_support_topk: int = 128,
    teacher_cache_topk: int = 4096,
    truncate_dynamic_support: bool = False,
) -> None:
    samples, rollout_id = load_rollout_file(path)
    for offset, sample in enumerate(samples):
        context = f"{path}:sample[{offset}]"
        if truncate_dynamic_support and require_teacher:
            truncate_dynamic_scored_sample_support(
                sample, dynamic_support_topk, context=context
            )
        if opd_loss_type == "token_reverse_kl" and require_teacher:
            validate_token_scored_sample(
                sample,
                context=context,
                allow_topk_diagnostics=token_topk_diagnostics,
            )
            if token_topk_diagnostics:
                validate_scored_sample(
                    sample,
                    k,
                    context=context,
                    require_student_topk_logprobs=True,
                    require_sparse_teacher=True,
                )
        elif opd_loss_type == "token_reverse_kl":
            validate_token_dump_sample(sample, context=context)
        elif opd_loss_type in {
            "cached_teacher_dynamic_topk_reverse_kl",
            "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor",
            "cached_teacher_sampled_token_pg_with_student_anchor",
            "cached_teacher_topk_forward_kl",
            "cached_teacher_bidirectional_topk_kl",
            "cached_teacher_bidirectional_topk_kl_with_student_anchor",
            "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor",
        } and require_teacher:
            if opd_loss_type in {
                "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor",
                "cached_teacher_sampled_token_pg_with_student_anchor",
                "cached_teacher_bidirectional_topk_kl_with_student_anchor",
                "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor",
            }:
                validate_cross_tokenizer_anchor_scored_sample(
                    sample, teacher_cache_topk, dynamic_support_topk, context=context
                )
            else:
                validate_dynamic_scored_sample(
                    sample,
                    teacher_cache_topk,
                    dynamic_support_topk,
                    context=context,
                )
        elif opd_loss_type in {
            "cached_teacher_dynamic_topk_reverse_kl",
            "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor",
            "cached_teacher_sampled_token_pg_with_student_anchor",
            "cached_teacher_topk_forward_kl",
            "cached_teacher_bidirectional_topk_kl",
            "cached_teacher_bidirectional_topk_kl_with_student_anchor",
            "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor",
        }:
            validate_dump_sample(
                sample,
                dynamic_support_topk,
                context=context,
                require_student_topk_logprobs=require_student_topk_logprobs,
            )
        elif require_teacher:
            validate_scored_sample(
                sample,
                k,
                context=context,
                require_student_topk_logprobs=require_student_topk_logprobs,
                require_sparse_teacher=require_sparse_teacher,
            )
        else:
            validate_dump_sample(
                sample,
                k,
                context=context,
                require_student_topk_logprobs=require_student_topk_logprobs,
            )
    print(f"validated {path}: rollout_id={rollout_id}, samples={len(samples)}, teacher={require_teacher}")


def check_bake(path: str, k: int, rm_url: str, limit: int | None) -> None:
    from examples.coding_agent_opd.opd import post_process_rewards
    from slime.rollout.on_policy_distillation import reward_func

    samples, _ = load_rollout_file(path)
    samples = samples[:limit] if limit else samples
    for offset, sample in enumerate(samples):
        validate_dump_sample(sample, k, context=f"{path}:sample[{offset}]")
    args = SimpleNamespace(
        rm_url=rm_url,
        opd_type="sglang",
        opd_loss_type="topk_reverse_kl",
        opd_id_topk=k,
        reward_key=None,
    )

    async def collect_responses():
        return await asyncio.gather(*(reward_func(args, sample) for sample in samples))

    responses = asyncio.run(collect_responses())
    for sample, response in zip(samples, responses, strict=True):
        sample.reward = response
    post_process_rewards(args, samples)
    for offset, sample in enumerate(samples):
        validate_baked_sample(sample, k, context=f"{path}:sample[{offset}]")
    print(f"live bake validated for {len(samples)} samples against {rm_url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["dump", "scored", "dump-dir", "scored-dir", "bake"],
    )
    parser.add_argument("--pt")
    parser.add_argument("--dir")
    parser.add_argument("--opd-id-topk", type=int, default=1024)
    parser.add_argument(
        "--opd-loss-type",
        choices=[
            "topk_reverse_kl",
            "token_reverse_kl",
            "cached_teacher_topk_forward_kl",
            "cached_teacher_dynamic_topk_reverse_kl",
            "cached_teacher_dynamic_topk_reverse_kl_with_student_anchor",
            "cached_teacher_sampled_token_pg_with_student_anchor",
            "cached_teacher_bidirectional_topk_kl",
            "cached_teacher_bidirectional_topk_kl_with_student_anchor",
            "cached_teacher_let_me_forward_else_reverse_topk_kl_with_student_anchor",
        ],
        default="topk_reverse_kl",
    )
    parser.add_argument("--token-topk-diagnostics", action="store_true")
    parser.add_argument("--opd-dynamic-support-topk", type=int, default=128)
    parser.add_argument("--opd-teacher-cache-topk", type=int, default=4096)
    parser.add_argument("--truncate-dynamic-support", action="store_true")
    parser.add_argument("--expected-samples", type=int)
    parser.add_argument("--allow-repeated-samples", action="store_true")
    parser.add_argument(
        "--allow-partial-index-set",
        action="store_true",
        help="allow a unique completion-order subset when validating an interrupted Phase 1",
    )
    parser.add_argument("--dataset-size", type=int)
    parser.add_argument("--rm-url")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--require-student-topk-logprobs",
        action="store_true",
        help="require the offline Phase-1 float16 sparse student logprob snapshot",
    )
    parser.add_argument(
        "--require-sparse-teacher",
        action="store_true",
        help="require Phase-2 trainable-row-only teacher tensors on disk",
    )
    args = parser.parse_args()

    if args.mode in {"dump", "scored", "bake"} and not args.pt:
        parser.error(f"--pt is required for --mode {args.mode}")
    if args.mode in {"dump-dir", "scored-dir"} and not args.dir:
        parser.error(f"--dir is required for --mode {args.mode}")

    if args.mode == "dump":
        check_file(
            args.pt,
            args.opd_id_topk,
            require_teacher=False,
            require_student_topk_logprobs=args.require_student_topk_logprobs,
            require_sparse_teacher=args.require_sparse_teacher,
            opd_loss_type=args.opd_loss_type,
            token_topk_diagnostics=args.token_topk_diagnostics,
            dynamic_support_topk=args.opd_dynamic_support_topk,
            teacher_cache_topk=args.opd_teacher_cache_topk,
            truncate_dynamic_support=args.truncate_dynamic_support,
        )
    elif args.mode == "scored":
        check_file(
            args.pt,
            args.opd_id_topk,
            require_teacher=True,
            require_student_topk_logprobs=args.require_student_topk_logprobs,
            require_sparse_teacher=args.require_sparse_teacher,
            opd_loss_type=args.opd_loss_type,
            token_topk_diagnostics=args.token_topk_diagnostics,
            dynamic_support_topk=args.opd_dynamic_support_topk,
            teacher_cache_topk=args.opd_teacher_cache_topk,
            truncate_dynamic_support=args.truncate_dynamic_support,
        )
    elif args.mode in {"dump-dir", "scored-dir"}:
        summary = validate_rollout_directory(
            args.dir,
            args.opd_id_topk,
            require_teacher=args.mode == "scored-dir",
            expected_samples=args.expected_samples,
            require_unique_samples=not args.allow_repeated_samples,
            require_complete_index_set=not args.allow_partial_index_set,
            dataset_size=args.dataset_size,
            require_student_topk_logprobs=args.require_student_topk_logprobs,
            require_sparse_teacher=args.require_sparse_teacher,
            opd_loss_type=args.opd_loss_type,
            token_topk_diagnostics=args.token_topk_diagnostics,
            dynamic_support_topk=args.opd_dynamic_support_topk,
            teacher_cache_topk=args.opd_teacher_cache_topk,
            truncate_dynamic_support=args.truncate_dynamic_support,
        )
        print(f"validated {args.dir}: files={summary.num_files}, samples={summary.num_samples}, samples_per_file={summary.samples_per_file}, teacher={args.mode == 'scored-dir'}")
    else:
        if not args.rm_url:
            parser.error("--rm-url is required for --mode bake")
        check_bake(args.pt, args.opd_id_topk, args.rm_url, args.limit)


if __name__ == "__main__":
    main()
