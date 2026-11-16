# Vanilla OPD

Fully-async (online) On-Policy Distillation (OPD) training for SWE coding
agents, built on [slime](https://github.com/THUDM/slime). Where the sibling
package [`../offline_opd`](../offline_opd) runs the offline three-phase flow
(rollout, teacher scoring, training), this package runs an online pipeline
driven by `train_async.py`: the teacher sglang service, the student rollout
engines, and the Megatron training job run in parallel under ray, and the
teacher scores each rollout trajectory on the fly with `topk_reverse_kl`
(`--opd-type sglang`, `OPD_ID_TOPK=128`).

This package shares the same sglang and Megatron-LM source trees as
`offline_opd`; they are supplied through `GOLDEN_OPD_DIR` (default
`../offline_opd`) rather than vendored again.

The method is described in
[`slime/examples/coding_agent_opd/README.md`](slime/examples/coding_agent_opd/README.md);
the full environment setup is in [`env/env.readme.md`](env/env.readme.md).

## Directory layout

```
vanilla_opd/
├── env/                    # environment setup (see below)
│   ├── env.sh
│   ├── env.readme.md       # full install instructions
│   ├── setup.sh            # environment build
│   ├── repos.lock          # source checkouts
│   ├── freeze_final.txt
│   └── example/            # build + verification scripts
└── slime/                  # training framework + OPD agent code
                            # (examples/coding_agent_opd/)
```

> The vendored Dockerfiles under
> `slime/examples/coding_agent_opd/3rdparty/R2E-Gym/third_party/swebench_pro_dockerfiles/`
> build SWE-bench instance images. Their internal proxy settings are
> placeholders; replace them before rebuilding images with the docker runtime.

## Requirements

- Linux x86_64 with NVIDIA GPUs (tested on Hopper and Blackwell)
- CUDA 12.9 (`CUDA_HOME` defaults to `/usr/local/cuda-12.9`)
- Python 3.12 and [uv](https://docs.astral.sh/uv/)
- Build tools: `cmake`, `ninja`, `protobuf-compiler`, and a Rust toolchain
  (needed to compile sglang's `sglang-grpc`)

Environment build and reproduction are documented under `vanilla_opd/env`.

## Build

Once `../offline_opd` (or a custom `GOLDEN_OPD_DIR`) is in place:

```bash
cd vanilla_opd
bash env/setup.sh        # env/example steps 00 → 50; about 1 hour
                         # (flash-attn ~18 min, apex ~20 min)
bash env/example/50_verify.sh   # self-check (already run by setup.sh);
                                # expected output: ENV SMOKE OK
source env/env.sh        # activate in every shell afterwards
```

See [`env/env.readme.md`](env/env.readme.md) for details.

## Run

A full single-node online OPD run (teacher logprobs + rollout + Megatron
training) is driven by the validation script. The 8 GPUs are assigned as
teacher (6,7) / actor (0,1) / rollout (2-5), so no services need to be started
by hand:

```bash
cd slime
bash examples/coding_agent_opd/scripts/run-qwen3.5-9B-coding-opd-fully_async-singlenode-2tr-2te-4ro_scaleswe_topk128.sh
```

The multi-node (TP8) variant and the full parameter list are in
[`slime/examples/coding_agent_opd/scripts/utils/README.md`](slime/examples/coding_agent_opd/scripts/utils/README.md).

The environment-side smoke test starts two sglang instances and serves the
models; the model-evaluation / scoring harness is not shipped with this package
and is provided by [`../offline_opd`](../offline_opd) under
`slime/examples/coding_agent_opd/eval/`.

## Acknowledgments and licenses

This project derives from the following open-source projects; each subtree
keeps its original license:

- [THUDM/slime](https://github.com/THUDM/slime) (Apache-2.0, see `slime/LICENSE`)
- [sgl-project/sglang](https://github.com/sgl-project/sglang) (Apache-2.0, see
  `$GOLDEN_OPD_DIR/sglang/LICENSE`)
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM) (see
  `$GOLDEN_OPD_DIR/Megatron-LM/LICENSE`)