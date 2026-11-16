# Coding Agent OPD

SWE agent 强化学习 + On-Policy Distillation。Student 模型在装好真实代码仓库的
沙箱里用工具修 bug，奖励来自单测是否通过，同时向一个更强的 Teacher 模型做蒸馏。

## 方法

每条样本是一个真实的 GitHub issue（SWE-bench 风格）。Student 在一个 sandbox pod
里循环调用三个工具，直到提交或耗尽预算：

| 工具 | 作用 |
| --- | --- |
| `str_replace_editor` | 查看 / 编辑文件 |
| `execute_bash` | 执行命令 |
| `submit` | 提交当前改动 |

轨迹结束后取 `git diff`，在同一个 pod 上应用官方 test patch 跑测试，得到
`swe_reward ∈ {0, 1}`。训练信号由两部分合成：

- **SWE reward** — 测试是否通过，进 GRPO 的 advantage
- **OPD** — Student 生成的每个 token，取 Teacher 在同一位置的 logprob，做 reverse-KL

Teacher 只需与 Student **共享 tokenizer/vocab**，架构可以不同：它是一个独立的
sglang server，只提供 logprob，不参与训练。

评测复用完全相同的 scaffold / 工具 / 打分代码，只把 transport 从 token-in/token-out
换成标准 `/v1/chat/completions`，因此任何 OpenAI 兼容 endpoint 都能直接评测。

## 环境

```bash
source env/env.sh       # 激活 venv + CUDA / PYTHONPATH / 代理 / sandbox 配置（包根下执行）
```

安装过程见 `env/env.readme.md`。另需两项：

- **沙箱凭据** `slime/.secret/.key.sh`（gitignored），导出 `RLE_ENV` /
  `RLE_TOOL_TYPE` / `RLE_API_KEY`。缺这个文件 pod 起不来。
- **代理** — swebench 构建打分 spec 时要访问 `raw.githubusercontent.com`，
  集群内直连不通且没有超时，会挂住整个 rollout。`env/env.sh` 已内置。

自检：

```bash
python -c "import examples.coding_agent_opd.r2egym; print('ok')"
```

## 快速开始：评测

评测 harness（`run_eval.sh` / `eval_chat.py` / `benchmarks.json` / `QUICK_EVAL_GUIDE.md`）
随姊妹包 `../offline_opd` 提供：`../offline_opd/slime/examples/coding_agent_opd/eval/`。
评测复用本包同一套 scaffold / 工具 / 打分代码，只需一个 OpenAI 兼容 endpoint，
用法见该目录的 `QUICK_EVAL_GUIDE.md`。

## 训练

先起 Teacher server，再提交训练作业。示例脚本已把这两步串在一起：

```bash
# 单机 8 卡，fully-async（rollout 与 training 重叠）
bash examples/coding_agent_opd/scripts/run-qwen3.5-9B-coding-opd-fully_async-singlenode-2tr-2te-4ro_scaleswe_topk128.sh

# 多机（TP8），见 scripts/utils/README.md
```

脚本里的模型路径、venv 路径、GPU 划分都是环境变量，按需覆盖：

```bash
SLIME_VENV=$SLIME_VENV \
HF_CHECKPOINT=<student HF dir>  REF_MODEL_PATH=<student torch_dist ckpt> \
TEACHER_MODEL=<teacher HF dir>  TEACHER_GPUS=6,7 \
   bash examples/coding_agent_opd/scripts/run-qwen3.5-9B-coding-opd-fully_async-singlenode-2tr-2te-4ro_scaleswe_topk128.sh
```

关键参数：

| 参数 | 说明 |
| --- | --- |
| `--opd-type sglang` | Teacher 是外部 sglang server，logprob 走 HTTP（`--rm-url`）。**跨架构 Teacher 只能用这个** |
| `--opd-type megatron` | Teacher ckpt 直接 load 进 Student 的模型定义，**要求同架构** |
| `--opd-kl-coef` | 蒸馏项权重 |
| `--opd-loss-type` | `token_reverse_kl`（默认）/ `topk_reverse_kl` / `topk_forward_kl` |
| `--opd-id-topk K` | top-k 变体的 K；Teacher 只在 Student 每个位置的 top-k id 上打分 |
| `SLIME_AGENT_MAX_STEPS` | 单条轨迹的工具步数上限（默认 50） |
| `SWE_BOOT_CONCURRENCY` | 同时启动的 sandbox pod 数 |

> 本 workspace 的 Megatron checkout 未打 slime 的 `megatron.patch`，训练路径尚未验证；
> 已验证的是 serve + rollout + 打分。

## 数据

训练和评测用两种格式：

- **评测** — 每行一个扁平的 SWE 样本（`instance_id` / `docker_image` /
  `test_patch` / `FAIL_TO_PASS` … 都在顶层）
- **训练** — `{prompt, label, metadata}`，完整样本放在 `metadata["ds"]`

评测格式 → 训练格式：

```bash
python examples/coding_agent_opd/prepare_dataset.py \
   --in data/swebench-latest.jsonl --out data/swe_train.jsonl
```

评测数据来源、转换工具与基准注册见 `../offline_opd/slime/examples/coding_agent_opd/eval/`。

## 代码结构

```
generate.py            训练侧 rollout：驱动工具循环、打分、取 Teacher logprob
opd.py                 奖励后处理：SWE reward + Teacher logprob 挂到 sample
utils/rollout.py       两条路径共用的 pod 启动 / 上下文预算 / abort
utils/filters.py       dynamic sampling 过滤器
r2egym/                R2E-Gym 适配层（Action / Observation / Runtime / Env / Trajectory）
3rdparty/R2E-Gym/      vendored R2E-Gym + swebench（不 pip 安装，走 sys.path）
prepare_dataset.py     数据格式转换
scripts/               训练启动脚本（单机 / 多机 run-*）+ utils/ 工具脚本
logs/                  默认输出目录
```

## 常见问题

**评测相关 FAQ（`run_eval.sh` 的 cwd 要求、HTTP 400、sandbox pod 排查、
session 回收）见 `../offline_opd` 的 `eval/QUICK_EVAL_GUIDE.md`。**

**base model 的 smoke 里大量 `swe_reward=0.0` 是正常的** —— 几十步内很难真正解出
issue，smoke 验的是链路而不是效果。

