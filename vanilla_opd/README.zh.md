# Vanilla OPD

基于 [slime](https://github.com/THUDM/slime) 的 **fully-async（在线）On-Policy Distillation** 训练代码，
面向 SWE coding agent 场景。与姊妹包 [`../offline_opd`](../offline_opd)（离线三阶段：rollout → teacher
打分 → 训练）不同，本包跑 `train_async.py` 驱动的**在线**流程：teacher sglang 服务、student rollout
引擎与 Megatron 训练在 ray 上并行，teacher 实时为 rollout 轨迹做 `topk_reverse_kl` 打分
（`--opd-type sglang`、`OPD_ID_TOPK=128`）。

**本包与 offline_opd 共享同一份 sglang / Megatron-LM 源码**，

方法说明见 [`slime/examples/coding_agent_opd/README.md`](slime/examples/coding_agent_opd/README.md)；
本环境的完整安装见 [`env/env.readme.md`](env/env.readme.md)。

## 目录结构

```
vanilla_opd/
├── env/                    # 环境配置（见下）
│   ├── env.sh              
│   ├── env.readme.md       # 完整安装
│   ├── setup.sh            # 环境构建
│   ├── repos.lock          # 环境快照
│   ├── freeze_final.txt    
│   └── example/            # 构建 + 验证脚本
└── slime/                  # 训练框架 + OPD agent 代码（examples/coding_agent_opd/）
```


> 注：`slime/examples/coding_agent_opd/3rdparty/R2E-Gym/third_party/swebench_pro_dockerfiles/`
> 是 SWE-bench 实例镜像构建用的 Dockerfile，其中内网代理为占位符
> ；若用 docker runtime 重建镜像需先替换为可用代理。

## 环境要求

- Linux x86_64，NVIDIA GPU（在 Hopper / Blackwell 上使用过）
- CUDA 12.9（`CUDA_HOME` 默认 `/usr/local/cuda-12.9`）
- Python 3.12，[uv](https://docs.astral.sh/uv/)
- 编译依赖：`cmake`、`ninja`、`protobuf-compiler`、Rust 工具链（编译 sglang 的 `sglang-grpc` 时需要）
- 环境构建和复现见 `vanilla_opd/env`

## 构建

`../offline_opd`（或自定义 `GOLDEN_OPD_DIR`）就绪后：

```bash
cd vanilla_opd
bash env/setup.sh        # example/ 的 00 → 50；约 1 小时（flash-attn 约 18min、apex 约 20min）
bash env/example/50_verify.sh   # 自检（setup.sh 已含）：应打印 ENV SMOKE OK
source env/env.sh        # 之后每个 shell 先激活
```


详见 [`env/env.readme.md`](env/env.readme.md)。

## 运行

完整在线 OPD 训练（fully-async：teacher logprob + rollout + megatron 训练）用单节点验证脚本，
8 卡自带 teacher（GPUs 6,7）/ actor（0,1）/ rollout（2-5）分配，无需手动起服务：

```bash
cd slime
bash examples/coding_agent_opd/scripts/run-qwen3.5-9B-coding-opd-fully_async-singlenode-2tr-2te-4ro_scaleswe_topk128.sh
```

多机（TP8）版本与详细参数见
[`slime/examples/coding_agent_opd/scripts/utils/README.md`](slime/examples/coding_agent_opd/scripts/utils/README.md)。

环境侧 smoke（sglang 双实例 serve；模型评估/打分 harness 不随本包提供，随
[`../offline_opd`](../offline_opd) 的 `slime/examples/coding_agent_opd/eval/` 提供）：

## 致谢与许可证

本项目基于以下开源项目修改，各子目录保留其原始许可证：

- [THUDM/slime](https://github.com/THUDM/slime)（Apache-2.0，见 `slime/LICENSE`）
- [sgl-project/sglang](https://github.com/sgl-project/sglang)（Apache-2.0，见 `$GOLDEN_OPD_DIR/sglang/LICENSE`）
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)（见 `$GOLDEN_OPD_DIR/Megatron-LM/LICENSE`）