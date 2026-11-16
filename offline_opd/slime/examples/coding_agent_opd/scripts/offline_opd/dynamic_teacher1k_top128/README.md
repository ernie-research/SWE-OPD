# Dynamic current Top-128 with derived Teacher Top-1024

This isolated recipe reuses the scored data from the Teacher Top-4096 run. It
does not rerun rollout or Teacher forward. Each source cache row is reduced to
its probability Top-1024 and then token-id sorted for the native sparse Phase-3
lookup. The current student still recomputes its exact Top-128 each forward.

Run the derivation and two-epoch/128-step Phase 3 pipeline with:

```bash
bash examples/coding_agent_opd/scripts/offline_opd/dynamic_teacher1k_top128/05-run-base-yiyan-ep2-gbs64-reuse-t4k.sh
```

The compatibility field `teacher_top4096_mass` stores the actual derived
Top-1024 mass. Metrics use the configured-width name
`offline_opd_dynamic/teacher_top1024_mass`; the original Top-4096 experiment
continues to report `offline_opd_dynamic/teacher_top4096_mass`.
