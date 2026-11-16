# Multi-node coding-agent OPD (fully-async, external sglang teacher)

Reproduce the multi-node On-Policy Distillation run for the Qwen3.5-9B coding agent.

- **Student** = Qwen3.5-9B (megatron actor + sglang rollout, drives the SWE agent loop)
- **Teacher** = Qwen3.6-27B (standalone sglang server; token logprobs fetched over HTTP)
- Training driver = `train_async.py` (fully-async / disaggregated: rollout(N+1) overlaps training(N))

## Topology (per node = 8 GPUs)

The placement group sorts all ray bundles by `(node_ip, gpu_id)` and hands the first
`actor_num_gpus` to the actor, the rest to rollout. The actor/rollout boundary only lands
on a **physical node** boundary if **every ray node exposes the same GPU count**. So we
hold out 2 GPUs/node (= `TEACHER_TP`) to keep a uniform **6 ray GPUs/node**.

**The actor/rollout split is derived from the node count to fill EVERY node** (no idle
spare node): the first half of the ray nodes go to the actor, the rest to rollout. The
script computes this automatically from `NNODES` (even node counts only):

```
NNODES=2 : actor TP2 x DP3  (1 node, 6 GPU) | rollout 3 engines x tp2 (1 node, 6 GPU)
NNODES=4 : actor TP2 x DP6  (2 nodes,12 GPU)| rollout 6 engines x tp2 (2 nodes,12 GPU)
teacher  : Qwen3.6-27B sglang tp2 (2 GPUs, always on rank0's held-out GPUs 6,7)

per node:  GPUs 0-5 -> ray (actor or rollout)      GPUs 6,7 -> teacher (rank0) / idle (others)
```

`GLOBAL_BATCH_SIZE` must be divisible by the actor DP (= actor_total_gpus / TP). The full
default **48** (= `ROLLOUT_BATCH_SIZE 6 × N_SAMPLES_PER_PROMPT 8`) is divisible by DP=3
(2 nodes) and DP=6 (4 nodes), so it stays valid as the actor scales. Override any of
`ACTOR_NUM_NODES` / `ACTOR_NUM_GPUS_PER_NODE` / `ROLLOUT_NUM_GPUS` to force a custom split.

Cross-node traffic = student weight sync (actor -> rollout) over `bond0`. rollout reaches
the teacher at `${MASTER_ADDR}:13141` (in-cluster, proxy-bypassed).

### GPU utilization / idle GPUs

Because an **in-cluster teacher** forces a uniform 6 ray GPUs/node, the 2 held-out GPUs
on every **non-head** node stay idle — this is the structural cost of Plan A:

| source of idle GPU | present in Plan A? | how to remove |
|--------------------|--------------------|---------------|
| spare nodes (node count > what actor+rollout need) | **no** — split fills every node | keep node count even; the split scales with `NNODES` |
| 2 held-out GPUs on each non-head node (teacher uniformity) | **yes** (e.g. 6 GPUs idle on 4 nodes) | use a **dedicated teacher host** (`LAUNCH_TEACHER=0`), then `RAY_GPUS=0..7 RAY_NUM_GPUS_PER_NODE=8` → all 8/node used |

So on a 4-node cluster Plan A uses **26 / 32 GPUs** (12 actor + 12 rollout + 2 teacher);
only the 6 held-out GPUs on the 3 non-head nodes are idle. For zero idle role GPU, move
the teacher off the role nodes (see the escape hatch in the launch-script header).


## Prerequisites

1. **TE 2.17 venv (REQUIRED).** slime-venv ships TransformerEngine 2.10 which CANNOT run
   Qwen3.5 `head_dim=256` attention on the sm103 (B30Z) GPUs with THD/dynamic-batch — the
   first train step crashes with `No dot product attention backend is available`. Use
   `te216-clone-venv` (a hardlink clone of slime-venv with TE 2.17.0 built for cu12), which
   dispatches head_dim=256 THD/varlen to the FA4 backend. Pass it via `SLIME_VENV`.
2. **Identical paths on all nodes** (PDC assumption): the repo, both venvs, and all
   checkpoints must be at the same absolute paths on every pod.
3. **Checkpoints** (defaults): student `Qwen3.5-9B` (HF) + `Qwen3.5-9B_torch_dist` (ref/load);
   teacher `Qwen3.6-27B` (HF, servable by sglang). Under `/path/to/`.
4. **RLE credentials** (NOT in the repo): every script sources `${SLIME_DIR}/.secret/.key.sh`
   (gitignored) for `RLE_ENV` / `RLE_TOOL_TYPE` / `RLE_API_KEY` and aborts if that file is
   missing, so it must exist on EVERY node (`dispatch.sh sync` copies it). Exporting any of
   the three before launching overrides the file; `RLE_SECRET_FILE` points elsewhere.
5. **GitHub proxy** (baked in): `http://proxy.example.com:8188`. swebench's grading-spec
   builder does an untimed `requests.get` to `raw.githubusercontent.com`, which is NOT
   directly reachable here and hangs the whole rollout without the proxy. In-cluster hosts
   (teacher, all pods, sglang engines, RLE gateways) are in `no_proxy`.

## The reproduce loop: clean -> sync -> run -> clean

Run all of this **on rank0** (`PADDLE_TRAINER_ID=0`). `dispatch.sh` fans out to workers
over ssh, discovering hosts from `PADDLE_TRAINERS` (or `/path/to/workspace/hostfile`).

```bash
cd $SLIME/examples/coding_agent_opd/scripts
# 设定SLIME目录（按实际布局调整；发布包 vanilla_opd 下 `source env/env.sh` 后即已导出）
export SLIME=/path/to/slime
# 设定环境目录
export SLIME_VENV=/path/to/slime-venv   

# 1) CLEAN 每个节点（杀掉 sglang/ray/megatron 残留，清 /tmp/ray）
bash dispatch.sh clean

# 2) SYNC 代码和环境到 worker（仓库排除项：.git、wandb 输出目录、logs、tarball）
bash dispatch.sh sync 

# 3) 运行脚本（以下以 scaleswe_topk128 变体为例，其余 run-* 脚本用法相同）
bash dispatch.sh launch run-qwen3.5-9B-coding-opd-fully_async-multinode-8tr-8te-16ro_scaleswe_topk128.sh
```

## Launching

Under PDC, configure the job to launch the **same script on every trainer pod**; it
branches on `PADDLE_TRAINER_ID` (rank0 = ray head + teacher + submit; rank>0 = ray worker
that blocks). To drive it manually from rank0, start workers detached first, then run the
head in the foreground:

```bash
PADDLE_TRAINERS=<ip0>,<ip1>,<ip2>,<ip3>          # rank0 first
RUN=examples/coding_agent_opd/scripts/run-qwen3.5-9B-coding-opd-fully_async-multinode-8tr-8te-16ro_scaleswe_topk128.sh

# workers (rank 1..N): detached background launch over ssh
for i in 1 2 3; do
  ssh -f -n "${host[$i]}" \
    "nohup setsid env PADDLE_TRAINERS=${PADDLE_TRAINERS} PADDLE_TRAINER_ID=$i \
       MASTER_ADDR=<ip0> SLIME_VENV=${TE_VENV} RUN_TAG=<shared-tag> \
       bash ${SLIME}/${RUN} >/tmp/worker$i.log 2>&1 &"
done

# head (rank0), foreground:
PADDLE_TRAINERS=${PADDLE_TRAINERS} PADDLE_TRAINER_ID=0 \
  MASTER_ADDR=<ip0> SLIME_VENV=${TE_VENV} RUN_TAG=<shared-tag> \
  bash "${SLIME}/${RUN}"
```

Use a **shared `RUN_TAG`** so every rank writes into the same `logs/<EXP>_<RUN_TAG>/` dir
(filenames are per-rank: `run.rank${RANK}.log`).

## Success markers (rank0 log)

- 4-node ray cluster forms; `ray status` shows all nodes (all under the TE 2.17 python).
- Teacher up: `Teacher server is up at <ip>:13141`.
- Node-aligned placement: actor node(s) and rollout node(s) are disjoint (4 nodes → actor
  spans 2 nodes, rollout spans 2 nodes; check the `GPU split:` line the script prints).
- Cross-node `update_weights_from_distributed` completes.
- **First train step runs with NO head_dim=256 crash**: e.g. `step 1: {... loss ...
  grad_norm ...}`; `ref_log_probs` / `log_probs` complete.
- Checkpoint saved at iter 1; `ray job ... succeeded`.
- some SWE trajectories abort (`loop_abort:sglang returned abort`) and
  rewards 0.0 — the base model rarely solves in 50 steps; the plumbing is the target.

## Known gotchas

- **Ray actors inherit the raylet's `sys.executable`, NOT `$PATH`.** A bare `ray start`
  runs `${SLIME_VENV}/bin/ray`, whose shebang is baked at venv-creation; for a hardlink
  clone (te216-clone-venv) that shebang still points at the ORIGINAL slime-venv python, so
  every actor would silently run TE 2.10 and crash at step 1. The launch script pins
  `RAY_LAUNCH=("${SLIME_VENV}/bin/python3" "${SLIME_VENV}/bin/ray")` and starts ray through
  it in both the head and worker branches. Verify with
  `pgrep -af 'bin/ray start'` on each node — must show the te216 python.
- **dispatch.sh venv sync.** `DEFAULT_EXCLUDES` is repo-oriented (excludes `wandb`,
  `outputs`, `logs`, tarballs) and would strip real Python packages from `site-packages`
  (the `wandb` package, `litellm/llms/wandb`, `cleo/io/outputs`, ...) if applied to a venv,
  producing `ModuleNotFoundError` on workers. `cmd_sync` now auto-detects a venv
  (`is_venv_dir`: `pyvenv.cfg` or `bin/python*`) and uses `VENV_EXCLUDES` (only
  `__pycache__`/`*.pyc`); it prints `Detected venv ... using venv-safe excludes`. Pass
  `--link-dest=<worker slime-venv>` for near-zero transfer.
- **CUDNN load order.** The script points `CUDNN_PATH` at the wheel's `nvidia/cudnn` dir so
  TE loads the cudnn dispatcher in the correct DT_NEEDED order (avoids an undefined-symbol
  crash). It is propagated into `RUNTIME_ENV_JSON` for ray workers.
- **External / dedicated teacher host.** To give every node all 8 GPUs, set
  `LAUNCH_TEACHER=0 TEACHER_IP=<host> RAY_GPUS=0,1,2,3,4,5,6,7 RAY_NUM_GPUS_PER_NODE=8
  ACTOR_NUM_GPUS_PER_NODE=8 ROLLOUT_NUM_GPUS=8` (8 ray GPUs/node stays uniform).
