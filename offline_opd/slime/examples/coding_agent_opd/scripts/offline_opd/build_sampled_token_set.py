"""Phase-2 artifact -> Phase-3 chunk artifact (``teacher_chunk_{logprob,mask,kind}``).

Aligners, one per teacher setup:
  dsv4       GLM-4.5-Air <- DeepSeek-V4-Flash: byte-chunk map over a Teacher-indexed dense
             dump (``teacher_input_ids`` / ``teacher_prompt_length`` / ``teacher_token_logprob``).
  canonical  GLM-4.5-Air <- GLM-5.2: Phase 2's canonical-template 1:1 pairs become EXACT
             chunks, everything it anchored stays MASKED.
  identity   same tokenizer (Qwen3.5-9B <- Qwen3.6-27B): row r is teacher position
             prompt_len + r.
identity/canonical can also reuse an existing Top-K artifact (``--from-teacher-cache``): the
sampled token is looked up in the cached Top-K, and a miss is MASKED (Student anchor), as in
the cached sampled-token loss.

Always ``mode="chunk"`` with the per-row kind kept, so exact / struct / chunk stays a
train-time switch. ``--emit-topk-cache K`` additionally writes the anchored Top-K schema
from a dense dump with a teacher Top-K, so the top-K reverse KL losses read the same
artifact. The batch manifests are copied, keeping the source run's batch layout.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

OFFLINE_OPD_DIR = os.path.dirname(os.path.abspath(__file__))
# <tree>/examples/coding_agent_opd/scripts/offline_opd/this_file
TREE = os.path.abspath(os.path.join(OFFLINE_OPD_DIR, "..", "..", "..", ".."))
sys.path.insert(0, TREE)
sys.path.insert(0, OFFLINE_OPD_DIR)

# What the sampled-token loss and the Phase-3 data path read. The dense Top-K arrays are
# dropped (a 559GB dense artifact becomes ~5GB) unless --emit-topk-cache re-adds a cache.
KEEP = (
    "group_index", "index", "rollout_id", "prompt", "tokens", "response",
    "response_length", "label", "reward", "loss_mask", "weight_versions",
    "rollout_log_probs", "status", "metadata", "session_id",
)


def identity_teacher_logprobs(sample, *, from_teacher_cache=False):
    """``(teacher_logprobs over tokens, scored_rows [R] bool, source)`` for a same-tokenizer teacher."""
    tokens = [int(t) for t in sample["tokens"]]
    loss_mask = np.asarray(sample["loss_mask"], dtype=bool)
    R = len(loss_mask)
    P = len(tokens) - R
    full = np.zeros(len(tokens), dtype=np.float64)
    scored = np.zeros(R, dtype=bool)
    if sample.get("teacher_token_logprob") is not None:
        if sample["teacher_input_ids"].tolist() != tokens or int(sample["teacher_prompt_length"]) != P:
            raise ValueError(
                "identity alignment needs the teacher to have scored the Student's own "
                "sequence; this dense artifact was re-rendered (use --aligner dsv4)"
            )
        full[:] = sample["teacher_token_logprob"].double().numpy()
        scored[:] = loss_mask
        return full, scored, "dense"
    rows = np.flatnonzero(loss_mask)
    if sample.get("offline_opd_token_teacher_sparse") and sample.get("teacher_log_probs") is not None:
        values = torch.as_tensor(sample["teacher_log_probs"]).double().numpy().reshape(-1)
        if len(values) != len(rows):
            raise ValueError(f"{len(values)} token teacher log-probs for {len(rows)} trainable rows")
        full[P + rows] = values
        scored[rows] = True
        return full, scored, "token"
    if from_teacher_cache and sample.get("teacher_candidate_ids") is not None:
        full, scored = _teacher_cache_lookup(sample, tokens, P, R, token_id_bridge=None)
        return full, scored, "teacher_cache"
    raise ValueError(
        "no teacher value for identity alignment: expected a dense artifact, a "
        "token_reverse_kl artifact, or a cached teacher Top-K with --from-teacher-cache"
    )


def _teacher_cache_lookup(sample, tokens, P, R, *, token_id_bridge):
    """Sampled-token value from a Student-indexed cached teacher Top-K, as the cached loss reads it.

    Same steps as _opd_cached_teacher_sampled_token_pg_loss: bridge the sampled Student id
    to the teacher vocabulary (an id with no counterpart becomes -1 and misses), find it in
    that row's cached Top-K, and map NaN/inf like the loss does. A miss leaves the row
    unscored, i.e. MASKED -> Student anchor, matching miss policy ``anchor``.
    """
    positions = sample["teacher_candidate_positions"].long()
    ids = sample["teacher_candidate_ids"].long()
    lps = sample["teacher_candidate_logprobs"].float()
    query = torch.as_tensor(tokens, dtype=torch.long)[P + positions]
    if token_id_bridge:
        from slime.utils.token_id_bridge import map_token_ids_torch

        query = map_token_ids_torch(
            query, token_id_bridge, context="cached sampled-token Student actions",
            allow_unmapped=True,
        )
    hit = ids.eq(query[:, None])
    found = hit.any(1)
    values = torch.nan_to_num((lps * hit).sum(1), nan=-100.0, posinf=0.0, neginf=-100.0)
    full = np.zeros(len(tokens), dtype=np.float64)
    scored = np.zeros(R, dtype=bool)
    rows = positions[found].numpy()
    full[P + rows] = values[found].double().numpy()
    scored[rows] = True
    return full, scored


def canonical_teacher_pairs(sample, *, from_teacher_cache=False, token_id_bridge=None):
    """Teacher log-probs and 1:1 pairs from Phase 2's canonical-template alignment.

    For teachers whose template differs but whose vocabulary bridges 1:1 for most tokens
    (GLM-4.5-Air <- GLM-5.2). teacher_trajectory.py already decided the pairs: a Student
    row pairs with a teacher position only when the bytes are identical and the bridged id
    equals the teacher id, so the teacher's own-token log-prob there IS log q(sampled
    token). Everything it anchored (reasoning prefix, Student-only newline, observation
    stop, N:M splits) stays MASKED.

    Returns ``(teacher_ids, teacher_completion_start, teacher_logprobs, pairs, source)``.
    """
    tokens = [int(t) for t in sample["tokens"]]
    R = len(sample["loss_mask"])
    P = len(tokens) - R
    if sample.get("teacher_token_logprob") is not None:
        student_rows = sample.get("opd_teacher_student_scored_positions")
        teacher_rows = sample.get("teacher_aligned_positions")
        if student_rows is None or teacher_rows is None:
            raise ValueError(
                "dense artifact has no canonical pairs (opd_teacher_student_scored_positions "
                "+ teacher_aligned_positions); rescore it with the current Phase 2"
            )
        Pt = int(sample["teacher_prompt_length"])
        pairs = list(zip(np.asarray(student_rows).tolist(), (Pt + np.asarray(teacher_rows)).tolist()))
        return (sample["teacher_input_ids"].tolist(), Pt,
                sample["teacher_token_logprob"].tolist(), pairs, "dense")
    if from_teacher_cache and sample.get("teacher_candidate_ids") is not None:
        # The cache is already Student-indexed at the aligned rows, so the Student grid
        # doubles as the teacher grid: row r <-> position P + r.
        full, scored = _teacher_cache_lookup(sample, tokens, P, R, token_id_bridge=token_id_bridge)
        pairs = [(r, P + r) for r in np.flatnonzero(scored).tolist()]
        return tokens, P, full.tolist(), pairs, "teacher_cache"
    raise ValueError(
        "no teacher value for canonical alignment: expected a dense artifact with canonical "
        "pairs, or a cached teacher Top-K with --from-teacher-cache"
    )


def anchored_topk_fields(sample, chunk_map, *, cache_topk, dynamic_topk, token_id_bridge, context):
    """Cross-tokenizer anchored Top-K fields, from a Teacher-indexed dense dump with Top-K.

    What the top-K reverse KL losses (``cached_teacher_*_with_student_anchor``) read, built
    on the same chunk map as the sampled-token fields: a Student row is a teacher row when
    its chunk is 1:1 (EXACT / STRUCT_1TO1), the teacher position has a dense Top-K row, and
    the bridged sampled Student id IS the teacher's token there. It then gets the teacher's
    Top-``cache_topk`` at that position; every other trainable row is a Student anchor.
    The split, the id-sorted cache and the checks are Phase 2's own
    (finalize_cross_tokenizer_anchor_artifact, compact_dynamic_teacher_cache_for_disk,
    validate_cross_tokenizer_anchor_scored_sample), so Phase 3 cannot tell the difference.
    """
    from types import SimpleNamespace

    from xvocab_chunk import EXACT, STRUCT_1TO1
    from examples.coding_agent_opd.scripts.offline_opd.offline_utils import (
        compact_dynamic_teacher_cache_for_disk,
        validate_cross_tokenizer_anchor_scored_sample,
    )
    from examples.coding_agent_opd.scripts.offline_opd.teacher_trajectory import (
        finalize_cross_tokenizer_anchor_artifact,
    )

    dense_ids = sample.get("teacher_dense_topk_ids")
    dense_lp = sample.get("teacher_dense_topk_logprobs")
    if dense_ids is None or dense_lp is None:
        raise ValueError(
            f"{context}: --emit-topk-cache needs a dense artifact with a teacher Top-K "
            "(Phase 2 with OPD_PHASE3_OBJECTIVE=topk_kl)"
        )
    if not 0 < dynamic_topk <= cache_topk <= dense_ids.shape[1]:
        raise ValueError(
            f"{context}: need 0 < dynamic support {dynamic_topk} <= cache {cache_topk} <= "
            f"dense Top-K width {dense_ids.shape[1]}"
        )
    tokens = torch.as_tensor([int(t) for t in sample["tokens"]], dtype=torch.long)
    loss_mask = np.asarray(sample["loss_mask"], dtype=bool)
    R = len(loss_mask)
    Ps = len(tokens) - R
    Pt = int(sample["teacher_prompt_length"])
    teacher_ids = sample["teacher_input_ids"].long()
    dense_pos = sample["teacher_dense_positions"].long().numpy()

    pairs = [
        (c.student[0], c.teacher[0]) for c in chunk_map.chunks
        if c.kind in (EXACT, STRUCT_1TO1)
        and c.student[1] - c.student[0] == 1 and c.teacher[1] - c.teacher[0] == 1
    ]
    s = np.asarray([p[0] for p in pairs], dtype=np.int64)
    t = np.asarray([p[1] for p in pairs], dtype=np.int64)
    rows = np.searchsorted(dense_pos, t - Pt)
    in_dense = (rows < len(dense_pos)) & (dense_pos[np.minimum(rows, len(dense_pos) - 1)] == t - Pt)
    query = tokens[Ps + torch.from_numpy(s)]
    if token_id_bridge:
        from slime.utils.token_id_bridge import map_token_ids_torch

        query = map_token_ids_torch(
            query, token_id_bridge, context=f"{context}: 1:1 Student tokens", allow_unmapped=True
        )
    same_token = (query == teacher_ids[torch.from_numpy(t)]).numpy()
    keep = in_dense & same_token
    order = np.argsort(s[keep], kind="stable")
    s_keep, rows_keep = s[keep][order], rows[keep][order]
    anchors = np.setdiff1d(np.flatnonzero(loss_mask), s_keep)

    ns = SimpleNamespace(
        loss_mask=list(sample["loss_mask"]),
        response_length=R,
        student_topk_ids=np.asarray(sample["student_topk_ids"]),
        student_topk_logprobs=np.asarray(sample["student_topk_logprobs"]),
        opd_teacher_alignment_mode="canonical_template_v1",
        opd_teacher_student_scored_positions=s_keep.astype(np.int32),
        opd_teacher_student_anchor_positions=anchors.astype(np.int32),
        # Audit-only: the loss partition never reads it (see the finalize function).
        opd_teacher_student_anchor_reasons=np.zeros(anchors.size, dtype=np.int8),
        opd_teacher_alignment_stats={
            "chunk_one_to_one_pairs": int(len(pairs)),
            "chunk_pairs_outside_dense_rows": int((~in_dense).sum()),
            "chunk_pairs_bridged_id_mismatch": int((in_dense & ~same_token).sum()),
        },
        metadata=dict(sample.get("metadata") or {}),
        teacher_topk_ids=dense_ids[torch.from_numpy(rows_keep)][:, :cache_topk],
        teacher_topk_logprobs=dense_lp[torch.from_numpy(rows_keep)][:, :cache_topk].float(),
        teacher_candidate_positions=None,
        offline_opd_cross_tokenizer_anchor_sparse=False,
        offline_opd_dynamic_teacher_sparse=False,
        offline_opd_teacher_sparse=False,
        student_support_ids=None,
        teacher_logprobs_on_student=None,
        opd_topk_overlap=None,
    )
    finalize_cross_tokenizer_anchor_artifact(ns, context=context)
    compact_dynamic_teacher_cache_for_disk(ns, cache_topk, dynamic_topk, context=context)
    validate_cross_tokenizer_anchor_scored_sample(ns, cache_topk, dynamic_topk, context=context)
    names = (
        "student_topk_ids", "student_topk_logprobs", "student_anchor_positions",
        "student_anchor_topk_ids", "student_anchor_topk_logprobs", "student_anchor_reason",
        "teacher_candidate_positions", "teacher_candidate_ids", "teacher_candidate_logprobs",
        "teacher_top128_ids", "teacher_cutoff_logprobs", "teacher_top4096_mass",
        "offline_opd_cross_tokenizer_anchor_sparse", "offline_opd_dynamic_teacher_sparse",
    )
    fields = {name: getattr(ns, name) for name in names}
    return fields, ns.metadata


def _write(args, path, target, sample, blob, chunk_map, logprob, mask, kind, extra_meta):
    slim = {k: sample[k] for k in KEEP if k in sample}
    meta = dict(slim.get("metadata") or {})
    if args.emit_topk_cache:
        fields, topk_meta = anchored_topk_fields(
            sample, chunk_map, cache_topk=args.emit_topk_cache,
            dynamic_topk=args.dynamic_support_topk, token_id_bridge=args.token_id_bridge,
            context=os.path.basename(path),
        )
        slim.update(fields)
        meta = topk_meta
        meta["opd_chunk_topk_cache"] = int(args.emit_topk_cache)
    slim["teacher_chunk_logprob"] = torch.tensor(logprob, dtype=torch.float32)
    slim["teacher_chunk_mask"] = torch.tensor(mask, dtype=torch.uint8)
    slim["teacher_chunk_kind"] = torch.tensor(kind, dtype=torch.uint8)
    slim["offline_opd_chunk_teacher"] = True
    # Phase 3 checks --opd-chunk-tier against this (slime/ray/rollout.py).
    meta["opd_chunk_mode"] = "chunk"
    meta["opd_chunk_aligner"] = args.aligner
    meta["opd_chunk_stats"] = dict(chunk_map.stats)
    meta["opd_chunk_failed_turns"] = len(chunk_map.failed_turns)
    meta["opd_chunk_source"] = os.path.basename(args.in_dir.rstrip("/"))
    meta.update(extra_meta)
    slim["metadata"] = meta
    tmp = target.with_suffix(f".tmp.{os.getpid()}")
    torch.save({"format": blob.get("format"), "index": blob.get("index"), "sample": slim}, tmp)
    os.replace(tmp, target)
    return {"rows": int(sum(mask)), "failed": len(chunk_map.failed_turns), "stats": dict(chunk_map.stats)}


def worker(args, paths):
    from xvocab_chunk import build_chunk_map, build_identity_chunk_map, build_pair_chunk_map, chunk_logprobs

    if args.aligner == "dsv4":
        from xvocab_chunk import make_aligner
        from examples.coding_agent_opd.r2egym.scaffold import load_scaffold
        from examples.coding_agent_opd.scripts.offline_opd.teacher_trajectory import _load_tokenizer

        aligner = make_aligner(args.assets, args.reasoning_effort)
        scaffold = load_scaffold()
        teacher_tokenizer = _load_tokenizer(args.teacher_template_model)
    out_dir = Path(args.out_dir)
    done = []
    for path in paths:
        name = os.path.basename(path)
        target = out_dir / "samples" / name
        if target.is_file() and target.stat().st_size > 0:
            done.append((name, None))
            continue
        blob = torch.load(path, weights_only=False, map_location="cpu", mmap=True)
        sample = blob["sample"]
        if args.aligner == "dsv4":
            chunk_map = build_chunk_map(
                aligner,
                tokens=list(sample["tokens"]),
                loss_mask=list(sample["loss_mask"]),
                teacher_ids=sample["teacher_input_ids"].tolist(),
                teacher_prompt_len=int(sample["teacher_prompt_length"]),
                scaffold=scaffold,
                teacher_tokenizer=teacher_tokenizer,
                reasoning_effort=args.reasoning_effort,
            )
            teacher_lp, extra = sample["teacher_token_logprob"].tolist(), {}
        elif args.aligner == "canonical":
            teacher_ids, start, teacher_lp, pairs, source = canonical_teacher_pairs(
                sample, from_teacher_cache=args.from_teacher_cache,
                token_id_bridge=args.token_id_bridge,
            )
            chunk_map = build_pair_chunk_map(
                teacher_ids=teacher_ids, teacher_completion_start=start,
                loss_mask=list(sample["loss_mask"]), pairs=pairs,
            )
            extra = {"opd_chunk_teacher_source": source}
        else:
            teacher_lp, scored, source = identity_teacher_logprobs(
                sample, from_teacher_cache=args.from_teacher_cache
            )
            chunk_map = build_identity_chunk_map(
                tokens=list(sample["tokens"]), loss_mask=list(sample["loss_mask"]),
                scored_rows=scored,
            )
            teacher_lp, extra = teacher_lp.tolist(), {"opd_chunk_teacher_source": source}
        logprob, mask, kind = chunk_logprobs(chunk_map, teacher_lp, "chunk")
        done.append((name, _write(args, path, target, sample, blob, chunk_map, logprob, mask, kind, extra)))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True, help="Phase-2 scored/ directory (samples/ layout)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--aligner", choices=["dsv4", "canonical", "identity"], default="dsv4",
                   help="dsv4: byte-chunk GLM<->DSV4; canonical: Phase 2's canonical-template "
                        "1:1 pairs (GLM-4.5-Air <- GLM-5.2); identity: same tokenizer (Qwen)")
    p.add_argument("--assets", default=os.environ.get("XVOCAB_ASSETS"),
                   help="dsv4: dir with glm_tokenizer.json, ds_tokenizer.json, encoding_dsv4.py "
                        "(default: $XVOCAB_ASSETS)")
    p.add_argument("--teacher-template-model", default=None, help="dsv4: teacher_shim dir")
    p.add_argument("--reasoning-effort", default=None)
    p.add_argument("--from-teacher-cache", action="store_true",
                   help="identity/canonical: take the sampled-token value from a cached teacher "
                        "Top-K (a miss becomes a MASKED row), for reusing existing Top-K artifacts")
    p.add_argument("--token-id-bridge", default=os.environ.get("OPD_TOKEN_ID_BRIDGE"),
                   help="canonical --from-teacher-cache: Student->teacher id bridge JSON "
                        "(default: $OPD_TOKEN_ID_BRIDGE)")
    p.add_argument("--emit-topk-cache", type=int, default=0,
                   help="also write the cross-tokenizer anchored Top-K schema with this teacher "
                        "cache width (e.g. 1000) from a dense artifact that stored a teacher "
                        "Top-K, so the top-K reverse KL losses read the same artifact; 0 = off")
    p.add_argument("--dynamic-support-topk", type=int, default=128,
                   help="--emit-topk-cache: Student support width (the Phase-1 Top-K)")
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.emit_topk_cache and a.aligner == "identity":
        raise SystemExit("--emit-topk-cache is for re-rendered dense dumps (dsv4/canonical); "
                         "same-tokenizer runs already store a Student-indexed Top-K cache")
    if a.aligner == "dsv4" and not (a.assets and a.teacher_template_model):
        raise SystemExit("--aligner dsv4 needs --assets (or $XVOCAB_ASSETS) and --teacher-template-model")
    if a.aligner == "canonical" and a.from_teacher_cache and not a.token_id_bridge:
        raise SystemExit("--aligner canonical --from-teacher-cache needs --token-id-bridge")

    samples = sorted(Path(a.in_dir, "samples").glob("sample_*.pt"))
    if a.limit:
        samples = samples[: a.limit]
    if not samples:
        raise SystemExit(f"no samples under {a.in_dir}/samples")
    out = Path(a.out_dir)
    (out / "samples").mkdir(parents=True, exist_ok=True)

    shards = [[str(x) for x in samples[i :: a.workers]] for i in range(a.workers)]
    shards = [s for s in shards if s]
    from concurrent.futures import ProcessPoolExecutor

    written = skipped = failed_turns = 0
    rows = 0
    agg: dict[str, int] = {}
    with ProcessPoolExecutor(len(shards)) as ex:
        futures = [ex.submit(worker, a, shard) for shard in shards]
        for n, future in enumerate(futures, 1):
            for name, info in future.result():
                if info is None:
                    skipped += 1
                    continue
                written += 1
                rows += info["rows"]
                failed_turns += info["failed"]
                for kind, count in info["stats"].items():
                    agg[kind] = agg.get(kind, 0) + count
            print(f"[{n}/{len(shards)}] written={written} skipped={skipped}", flush=True)

    summary = {
        "samples": len(samples), "written": written, "skipped": skipped,
        "chunk_rows": rows, "failed_turns": failed_turns, "kind_rows": agg,
        "source": a.in_dir, "mode": "chunk", "aligner": a.aligner,
    }
    (out / "chunk_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    # Same sample file names and trainable-token costs, so the source manifests and the
    # Phase-3 batch manifests (rollout_N.pt) carry over unchanged -- keeping the batch
    # layout identical to the source run, which is what an A/B against it needs.
    # Skipped under --limit: they would name samples that were not written.
    if not a.limit:
        source = Path(a.in_dir, "source_manifests")
        if source.is_dir():
            shutil.copytree(source, out / "source_manifests", dirs_exist_ok=True)
        for manifest in list(Path(a.in_dir).glob("rollout_*.pt")) + [Path(a.in_dir, ".sample_manifest_summary.json")]:
            if manifest.is_file() and _is_sample_manifest(manifest):
                shutil.copy2(manifest, out / manifest.name)
    print(json.dumps(summary, indent=2))


def _is_sample_manifest(path):
    """Only tiny batch manifests are copied, never an inline rollout file with teacher data."""
    if path.suffix == ".json":
        return True
    if path.stat().st_size > 1 << 20:
        return False
    blob = torch.load(path, weights_only=False, map_location="cpu")
    return isinstance(blob, dict) and blob.get("format") == "offline_opd_sample_manifest_v1"


if __name__ == "__main__":
    main()
