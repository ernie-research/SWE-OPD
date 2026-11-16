# Offline OPD Top-128 run

This directory pins the validated four-node offline-OPD recipe while changing
the sparse support from K=1024 to K=128. Run the scripts on rank 0 in order:

```bash
bash examples/coding_agent_opd/scripts/multi_nodes/dispatch.sh sync \
  /path/to/workspace/env_run/<user>/slime/examples/coding_agent_opd/scripts/offline_opd

bash examples/coding_agent_opd/scripts/offline_opd/top128/01-phase1-rollout.sh
bash examples/coding_agent_opd/scripts/offline_opd/top128/02-phase2-teacher.sh
bash examples/coding_agent_opd/scripts/offline_opd/top128/03-phase3-train.sh
```

Or start all three phases as one fail-fast background pipeline:

```bash
bash examples/coding_agent_opd/scripts/offline_opd/top128/04-run-all.sh
```

Each command starts in the background and prints its launcher log. Do not start
the next phase until the previous launcher exits successfully and its
`.phaseN.complete` marker exists. All phases automatically share the run path
stored in:

```text
/path/to/OPD/offline_opd/.qwen35-9b-k128-current
```

Useful controls:

```bash
# Watch the active run.
RUN_ROOT=$(< /path/to/OPD/offline_opd/.qwen35-9b-k128-current)
tail -f "${RUN_ROOT}/logs-phase1/launcher.log"

# Resume a failed phase in the foreground.
bash examples/coding_agent_opd/scripts/offline_opd/top128/01-phase1-rollout.sh --foreground

# Deliberately create another timestamped run rather than resume the current one.
NEW_RUN=1 bash examples/coding_agent_opd/scripts/offline_opd/top128/01-phase1-rollout.sh

# Explicitly select an existing run for Phase 2 or 3.
RUN_ROOT=/path/to/OPD/offline_opd/qwen35-9b-k128-... \
  bash examples/coding_agent_opd/scripts/offline_opd/top128/02-phase2-teacher.sh
```

Pinned defaults: Phase 1 concurrency 500; file batch 50; Phase 2 teacher
TP4 x DP2 per node and scoring concurrency 8; Phase 3 10 shuffled epochs,
GBS 50, TP8 x PP1 x CP2, 128K packing, constant LR 1e-6, and checkpoint every
30 steps. Phase 2 scores the teacher once; epoch fan-out happens only in Phase 3.
Phase 2 manifest load balancing is enabled by default. Set
`PHASE2_BALANCE_MANIFESTS=0` when launching `02-phase2-teacher.sh` to preserve
the exact Phase-1 `rollout_N.pt` batch grouping and in-batch order instead.
