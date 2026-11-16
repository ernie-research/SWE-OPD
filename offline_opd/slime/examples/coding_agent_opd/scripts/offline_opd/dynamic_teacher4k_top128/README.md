# Dynamic current Top-128 with cached teacher Top-4096

This directory is an isolated three-phase recipe. It selects
`cached_teacher_dynamic_topk_reverse_kl`; the existing `topk_reverse_kl` path and
its artifacts are unchanged.

Phase 1 saves the rollout trajectory plus the rollout student's ranked Top-128
ids/log-probs. Phase 2 asks the teacher for its own Top-4096 once and writes only
trainable rows. Candidate ids are token-id sorted and log-probs are float16;
teacher Top-128 ids, the 4096th cutoff log-prob, and cached mass are stored too.

Phase 3 recomputes the current student's exact TP-aware full-vocabulary Top-128.
For each current id, it looks up the teacher value in the cached Top-4096. A miss
uses the cached 4096th log-prob. The cache remains `[N_trainable,4096]` on CPU and
is copied in bounded row chunks; it is never expanded to
`[response_length,4096]`.

Run all phases with:

```bash
bash examples/coding_agent_opd/scripts/offline_opd/dynamic_teacher4k_top128/04-run-all.sh
```

Important metrics are under `offline_opd_dynamic/*`; `train/student_mass`,
`train/teacher_mass`, and `train/opd_topk_overlap` now refer to the current
student's true Top-128 support. `teacher_mass` is the cutoff-imputed upper
estimate, not an exact teacher mass.
