# Offline OPD

基于 [slime](https://github.com/THUDM/slime) 的离线 On-Policy Distillation（OPD）训练代码，面向 SWE coding agent 场景。
训练流程分三个阶段：

1. **Phase 1（student rollout）**：学生模型在 SWE 沙箱中生成完整轨迹，保存每个可训练位置的 student top-K token id 和 log-prob。
2. **Phase 2（teacher 打分）**：teacher 在同一组 token id 上离线计算 log-prob，结果逐样本落盘，并生成打乱和负载均衡后的 manifest。
3. **Phase 3（离线训练）**：只读取 Phase 2 的缓存做 `topk_reverse_kl` 训练，训练期间不启动 teacher 和 rollout engine。

公式和指标口径见 [`OFFLINE_OPD_FORMULAS.md`](slime/examples/coding_agent_opd/scripts/offline_opd/OFFLINE_OPD_FORMULAS.md)，数据契约和多机流程见 [`PLAN.md`](slime/examples/coding_agent_opd/scripts/offline_opd/PLAN.md)。

## 目录结构

```
offline_opd/
├── slime/          # 训练框架（基于 THUDM/slime 修改），offline OPD 代码在 examples/coding_agent_opd/
├── sglang/         # 推理引擎（基于 sgl-project/sglang 修改）
├── Megatron-LM/    # 训练后端（基于 NVIDIA/Megatron-LM 修改）
├── requirements.txt
└── README.md
```

- sglang fork：官方 `release/v0.5.12` HEAD `5a15cde858ea09b77116212a39356f2fc51b8584`
  （"fix: fix deepseek v4 CP error (#25396) (#26154)"，2026-05-22，已在官方仓库核对）
- Megatron-LM fork：官方 `1dcf0dafa884ad52ffb243625717a3471643e087`
  （"[dev] ci: Fix merge queue (#3385)"，2026-02-14，已在官方仓库核对）
- slime：`0.3.0`

## 环境要求

- Linux x86_64，NVIDIA GPU（在 Hopper / Blackwell 上使用过）
- CUDA 12.9（`CUDA_HOME` 默认 `/usr/local/cuda-12.9`）
- Python 3.12，[uv](https://docs.astral.sh/uv/)
- 编译依赖：`cmake`、`ninja`、`protobuf-compiler`、Rust 工具链（编译 sglang 的 `sglang-grpc` 时需要）

## 安装

```bash
cd offline_opd
uv venv .venv --python 3.12
source .venv/bin/activate
export CUDA_HOME=/usr/local/cuda-12.9
export PATH="$CUDA_HOME/bin:$PATH"

# 1. 固定版本的 Python 依赖（含 torch 2.11.0+cu129）
uv pip install -r requirements.txt --index-strategy unsafe-best-match

# 2. 需要关闭 build isolation 编译的依赖
MAX_JOBS=64 uv pip install flash-attn==2.7.4.post1 --no-build-isolation
uv pip install --no-build-isolation "transformer_engine[pytorch]==2.17.0"
uv pip install pip
NVCC_APPEND_FLAGS="--threads 4" pip install --no-build-isolation --no-cache-dir \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" \
  git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4
uv pip install --no-deps git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c
uv pip install --no-build-isolation git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a
uv pip install --no-build-isolation git+https://github.com/fzyzcjy/torch_memory_saver.git@6d5bce48c898c64497be2022fbdc08cc7ddb5637

# 3. 本地源码（editable 安装，--no-deps 避免覆盖上面固定的版本）
uv pip install --no-deps -e "sglang/python"
uv pip install --no-deps --no-build-isolation -e Megatron-LM
uv pip install --no-deps -e slime
```

运行前需要让 pip 安装的 cudnn 9.17.1 优先于系统自带版本，并把 Megatron-LM 加入 `PYTHONPATH`：

```bash
export LD_LIBRARY_PATH="$VIRTUAL_ENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$(pwd)/Megatron-LM:$PYTHONPATH"
```

检查安装是否成功：

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.cuda.is_available())"
# 期望输出：2.11.0+cu129 12.9 91701 True
```

## 运行

启动脚本在 `slime/examples/coding_agent_opd/scripts/offline_opd/`，按多机环境设计（读取 `PADDLE_TRAINERS`、`PADDLE_TRAINERS_NUM`），默认四机。

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

常用环境变量：

- `OPD_ID_TOPK`：每个位置保留的 token 数，默认 `1024`
- `OFFLINE_OPD_LOSS_TYPE`：默认 `topk_reverse_kl`
- `TRAIN_EPOCHS`：Phase 3 重放 Phase 2 缓存的 epoch 数，默认 `10`
- `ROLLOUT_BATCH_SIZE`、`SAVE_INTERVAL`

Phase 1 的 rollout 需要 SWE 沙箱后端，数据格式见 [`slime/examples/coding_agent_opd/README.md`](slime/examples/coding_agent_opd/README.md)。

## 致谢与许可证

本项目基于以下开源项目修改，各子目录保留其原始许可证：

- [THUDM/slime](https://github.com/THUDM/slime)（Apache-2.0，见 `slime/LICENSE`）
- [sgl-project/sglang](https://github.com/sgl-project/sglang)（Apache-2.0，见 `sglang/LICENSE`）
- [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)（见 `Megatron-LM/LICENSE`）
