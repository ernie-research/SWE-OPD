# GLM-4.7-Flash student / GLM-5.2 teacher offline OPD

This recipe runs three-stage offline OPD with the following fixed contract:

- Student rollout/training: `GLM-4.7-Flash`
- Teacher: `GLM-5.2-FP8`, dense attention
- Loss: cached Teacher Top-1000 / current Student Top-128 reverse KL only
- Tokenization: byte-identical `tokenizer.json`, direct shared token ids
- Template policy: preserve and replay the exact Student token sequence
- Training positions: every Student-generated token; observations remain masked
- No token-id bridge, Teacher-template rebuild, format mask, forward KL, or anchor

Although the two model directories contain different `chat_template.jinja`
files, Phase 2 never invokes the Teacher template. It forwards the complete
Phase-1 token sequence to GLM-5.2 unchanged, so Teacher rows and Student loss
positions stay exactly aligned.

Default four-node topology (8 H800s per node):

- Phase 1: eight TP4 rollout engines, global agent concurrency 400.
- Phase 2: two TP16 GLM-5.2 replicas, two nodes per replica, dense attention.
- Phase 3: TP4 x PP1 x CP2, EP8, DP4, GBS64, two replay epochs.

Run the detached pipeline with:

```bash
bash examples/coding_agent_opd/scripts/offline_opd/glm47flash_glm52/04-run-all.sh
```

The active run path is recorded in
`/path/to/offline_opd/.glm47flash-glm52-reverse-t1k-s128-current`.

## Resume and monitor Phase 3

For the partial 20-step run, `05-resume-full-from-partial.sh` builds the
deterministic two-epoch stream and resumes model, optimizer, and RNG state from
the latest distributed checkpoint. `06-queue-resume-when-gpus-free.sh` keeps
retrying failed attempts from the latest checkpoint at a 30-minute interval.

```bash
bash examples/coding_agent_opd/scripts/offline_opd/glm47flash_glm52/06-queue-resume-when-gpus-free.sh
```

Use the read-only status command for a concise, secret-safe snapshot:

```bash
bash examples/coding_agent_opd/scripts/offline_opd/glm47flash_glm52/07-status-phase3-resume.sh
```

The companion monitor records this snapshot every 30 minutes and restarts the
resume queue if that queue unexpectedly disappears. It does not kill a live
training job merely because a log is temporarily quiet.

```bash
bash examples/coding_agent_opd/scripts/offline_opd/glm47flash_glm52/08-monitor-phase3-resume.sh
```

After training finishes, run the completion audit. It checks all three phase
artifacts, the fixed experiment contract, tokenizer identity, final step and
checkpoint, W&B resume evidence, and fatal errors from the last attempt.

```bash
bash examples/coding_agent_opd/scripts/offline_opd/glm47flash_glm52/09-audit-completion.sh
```

All resume scripts accept `RUN_ROOT`; `05-resume-full-from-partial.sh` also
accepts `MAIN_RUN`. Their defaults point at the current two-epoch experiment.
