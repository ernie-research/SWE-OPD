# Iterative Top-128 Offline/Online OPD

This directory is an isolated strategy layer. It reuses the validated generic
offline OPD phases from the parent directory, but does not modify or invoke the
existing `offline_opd/top128` wrappers.

## Schedule

The default production schedule is:

```text
cycle 1: base SFT -> fresh rollout -> teacher score -> train epochs 1-2 -> checkpoint -> HF export
cycle 2: cycle 1  -> fresh rollout -> teacher score -> train epochs 3-4 -> checkpoint -> HF export
cycle 3: cycle 2  -> fresh rollout -> teacher score -> train epochs 5-6 -> checkpoint -> HF export
cycle 4: cycle 3  -> fresh rollout -> teacher score -> train epochs 7-8 -> checkpoint -> HF export
cycle 5: cycle 4  -> fresh rollout -> teacher score -> train epochs 9-10 -> checkpoint -> HF export
```

Every cycle owns separate `dump`, `scored`, `hf`, and log directories. The new
experiment owns one cumulative `checkpoints/` root and one lightweight symlink
`replay/` index, producing cycle checkpoints at iterations 19, 39, 59, 79 and
99 with the default 500 samples and GBS 50. The loss is hard-pinned to
`topk_reverse_kl` with `K=128`, so a stale
`OFFLINE_OPD_LOSS_TYPE=token_reverse_kl` environment variable cannot alter this
experiment.

Phase 1 is rollout-only and SGLang loads `HF_CHECKPOINT` directly. Therefore the
Megatron checkpoint produced by each training cycle is converted to HF before
the next cycle starts. Phase 3 uses the shared Megatron checkpoint root, so
model state, optimizer state, W&B step and checkpoint iteration all continue
across cycles.

Defaults match the completed Top-128 experiment apart from refreshing data every
two epochs:

```text
rollout concurrency = 500
rollout/scored batch = 50
rollout port range    = 26000+ (isolated from older jobs)
GBS                  = 50
TP x PP x CP         = 8 x 1 x 2
packing limit        = 128K
learning rate        = 1e-6, constant
epochs               = 5 cycles x 2 = 10
```

The cadence can be changed per new run without modifying the validated default.
For example, `ITERATIVE_CYCLES=10 EPOCHS_PER_CYCLE=1` performs a fresh rollout
and teacher pass after every training epoch while retaining ten total epochs.

## Launch

Sync the new scripts to the worker nodes first, then launch from rank 0:

```bash
bash examples/coding_agent_opd/scripts/multi_nodes/dispatch.sh sync \
  /path/to/workspace/env_run/<user>/slime

NEW_RUN=1 bash \
  examples/coding_agent_opd/scripts/offline_opd/iterative_top128/run-all.sh
```

`NEW_RUN=1` creates a new timestamped experiment under
`/path/to/offline_opd`. Without it, `run-all.sh` resumes the
run stored in `.qwen35-9b-k128-refresh5x2-current`.

The iterative strategy keeps all five cycles' rollout and teacher-scored data
under this large shared PFS mount; cycle data is not automatically deleted.
The cycle orchestrator pre-creates Phase-1/2 shared directories before mpirun
and waits briefly for PFS metadata propagation, avoiding concurrent `mkdir`
races between ranks.

Status:

```bash
bash examples/coding_agent_opd/scripts/offline_opd/iterative_top128/status.sh
```

Each phase and cycle is marker-based and idempotent. Phase 1 and Phase 2 reuse
their existing partial-output resume paths. If Phase 3 has already written a
checkpoint, it resumes inside that cycle; otherwise it restarts that cycle's
two-epoch training from the previous cycle checkpoint.
