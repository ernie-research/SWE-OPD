# Offline OPD

Offline On-Policy Distillation (OPD) training for SWE coding agents, built on
[slime](https://github.com/THUDM/slime). The training flow has three phases:

1. **Phase 1 (student rollout)** — the student model runs full trajectories in
   SWE sandboxes; the student top-K token ids and log-probs are saved for every
   trainable position.
2. **Phase 2 (teacher scoring)** — the teacher computes log-probs offline on
   the same token ids; results are written per sample, then shuffled and
   load-balanced into a manifest.
3. **Phase 3 (offline training)** — training reads only the Phase 2 cache and
   runs `topk_reverse_kl`; the teacher and the rollout engines are not started.

The formulas and metric definitions are in
[`OFFLINE_OPD_FORMULAS.md`](slime/examples/coding_agent_opd/scripts/offline_opd/OFFLINE_OPD_FORMULAS.md);
the data contract and multi-node flow are in
[`PLAN.md`](slime/examples/coding_agent_opd/scripts/offline_opd/PLAN.md).

## Directory layout

```
offline_opd/
├── slime/          # training framework (fork of THUDM/slime); the offline OPD
│                   # code lives in examples/coding_agent_opd/
├── sglang/         # inference engine (fork of sgl-project/sglang)
├── Megatron-LM/    # training backend (fork of NVIDIA/Megatron-LM)
├── requirements.txt
└── README.md
```

- sglang fork: official `release/v0.5.12` HEAD `5a15cde858ea09b77116212a39356f2fc51b8584`
  ("fix: fix deepseek v4 CP error (#25396) (#26154)", 2026-05-22; verified
  against the official repository)
- Megatron-LM fork: official `1dcf0dafa884ad52ffb243625717a3471643e087`
  ("[dev] ci: Fix merge queue (#3385)", 2026-02-14; verified against the
  official repository)
- slime: `0.3.0`

## Requirements

- Linux x86_64 with NVIDIA GPUs (tested on Hopper and Blackwell)
- CUDA 12.9 (`CUDA_HOME` defaults to `/usr/local/cuda-12.9`)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Build tools: `cmake`, `ninja`, `protobuf-compiler`, and a Rust toolchain
  (needed to compile sglang's `sglang-grpc`)

## Install

```bash
cd offline_opd
uv venv .venv --python 3.12
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"

# 1. Pinned Python dependencies (includes torch 2.11.0+cu129)
uv pip install -r requirements.txt --index-strategy unsafe-best-match

# 2. Dependencies that need build isolation disabled
MAX_JOBS=64 uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.17.0"
uv pip install pip
NVCC_APPEND_FLAGS="--threads 4" pip install --no-build-isolation --no-cache-dir \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
uv pip install --no-deps git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c
uv pip install --no-build-isolation git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a
uv pip install --no-build-isolation git+https://github.com/fzyzcjy/torch_memory_saver.git@6d5bce48c898c64497be2022fbdc08cc7ddb5637

# 3. Local source trees (editable installs; --no-deps keeps the pins above intact)
uv pip install --no-deps -e "sglang/python"
uv pip install --no-deps --no-build-isolation -e Megatron-LM
uv pip install --no-deps -e slime
```

Before running, make the pip-installed cudnn 9.17.1 take precedence over the
system one and add Megatron-LM to `PYTHONPATH`:

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$(pwd)/Megatron-LM:$PYTHONPATH"
```

Verify the install:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.is_available())"
# expected output: 2.11.0+cu129 12.9 91701 True
```

## Run

Launch scripts are in
`slime/examples/coding_agent_opd/scripts/offline_opd/`; they target multi-node
environments (reading `PADDLE_TRAINERS` and `PADDLE_TRAINERS_NUM`) and default
to four nodes.

```bash
cd slime
export SLIME_VENV=../.venv
export MEGATRON_LM_PATH=$(cd ../Megatron-LM && pwd)
export HF_CHECKPOINT=/path/to/student_hf_checkpoint
export LOAD_PATH=/path/to/student_torch_dist_checkpoint
export TEACHER_MODEL=/path/to/teacher_hf_checkpoint

bash examples/coding_agent_opd/scripts/offline_opd/start-offline-opd.sh phase1
bash examples/coding_agent_opd/scripts/offline_opd/start-offline-opd.sh phase2
SCORED_DIR=/path/to/phase2_scored \
  bash examples/coding_agent_opd/scripts/offline_opd/start-offline-opd.sh phase3
```

Common environment variables:

- `OPD_ID_TOPK`: number of tokens kept per position, default `1024`
- `OFFLINE_OPD_LOSS_TYPE`: default `topk_reverse_kl`
- `TRAIN_EPOCHS`: number of epochs Phase 3 replays the Phase 2 cache, default `10`
- `ROLLOUT_BATCH_SIZE`, `SAVE_INTERVAL`

Phase 1 rollout requires a SWE sandbox backend; the data format is described in
[`slime/examples/coding_agent_opd/README.md`](slime/examples/coding_agent_opd/README.md).

## Acknowledgments and licenses

This project derives from the following open-source projects; each subtree
keeps its original license:

- [THUDM/slime](https://github.com/THUDM/slime) (Apache-2.0, see `slime/LICENSE`)
- [sgl-project/sglang](https://github.com/sgl-project/sglang) (Apache-2.0, see
  `sglang/LICENSE`)
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (see
  `Megatron-LM/LICENSE`)