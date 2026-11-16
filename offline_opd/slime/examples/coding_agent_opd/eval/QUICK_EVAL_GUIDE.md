# Coding Agent OPD 评测流程（Agent 简版）

## 0. 前置准备

1. **评测数据**：`benchmarks.json` 声明的三个数据文件（`slime/data/swebench-latest.jsonl`、
   `slime/data/swe-bench-pro-rebuilt_eval.jsonl`、`slime/data/swebench-multilingual-latest_eval.jsonl`）
   不在仓库中（`data/` 已忽略），需自行准备。每行必须是一个 R2E-Gym `ds` dict
   （含 `swe_type` / `docker_image` / `instance_id` / `test_patch` 等字段，格式见
   `examples/coding_agent_opd/prepare_dataset.py` 与 `eval/benchmarks.py --check-only`），
   可从 R2E-Gym（https://huggingface.co/R2E-Gym、https://r2e-gym.github.io/）或 SWE-bench
   官方数据转换而来。启动前用 `--check-only` 校验：
   ```bash
   PYTHONPATH="$(pwd)/../Megatron-LM:$(pwd)" \
     .slime_venv/bin/python3 -m examples.coding_agent_opd.eval.benchmarks \
     --check-only multilingual
   ```
2. **RLE 密钥**：`run_eval.sh` 强制要求 `RLE_API_KEY`（RLE 平台颁发的 key），
   脚本本身不读取任何密钥文件。本仓库约定 `slime/.secret/.key.sh` 只导出密钥，
   先 `source` 它（或自行 export）再运行：
   ```bash
   source slime/.secret/.key.sh        # 文件只需导出 RLE_API_KEY（可选 RLE_TOOL_TYPE=swe-pdc / RLE_ENV=prod）
   ```
3. **Python 环境**：评测复用训练 venv。`run_eval.sh` 会尝试激活 `slime/.slime_venv`
   （未设置 `VIRTUAL_ENV` 时），因此先在 slime 根按仓库说明创建该 venv；使用其他环境时激活后运行即可。
4. **GitHub 代理**：swebench 数据中 django 等 repo 启动时要访问 raw.githubusercontent.com
   （请求无超时）。`run_eval.sh` 默认的 `GITHUB_PROXY` 是内网代理地址（对公开复现
   不可用），无法直连外网时必须用
   `GITHUB_PROXY=<你的代理> bash examples/coding_agent_opd/eval/run_eval.sh`
   覆盖，否则启动阶段会挂起；无需外网的场景可忽略。

## 1. 入口与数据

在 slime 根目录（`offline_opd/slime/`）执行。主要入口是 `eval/run_eval.sh`，数据注册表是
`eval/benchmarks.json`。

| `BENCHMARK` | 数据量 | 默认数据文件 |
|---|---:|---|
| `swebench` | 500 | `data/swebench-latest.jsonl` |
| `swebench-pro` | 731 | `data/swe-bench-pro-rebuilt_eval.jsonl` |
| `multilingual` | 300 | `data/swebench-multilingual-latest_eval.jsonl` |

Multilingual 必须是完整 300 条。空 `FAIL_TO_PASS` 或 `PASS_TO_PASS` 是合法数据，
不能因此过滤。启动前先检查：

```bash
PYTHONPATH="$(pwd)/../Megatron-LM:$(pwd)" \
  .slime_venv/bin/python3 -m examples.coding_agent_opd.eval.benchmarks \
  --check-only multilingual
```

预期为 `rows      : 300`、`[PASS] required_fields: 300/300 …`、
`[PASS] spec_coverage: … 300/300 …`、`verdict   : PASS — runnable`（exit 0）。

## 2. 启动

先确认 OpenAI 兼容服务可用：

```bash
curl --noproxy '*' -fsS http://MODEL_HOST:PORT/v1/models
```

然后启动单项评测：

```bash
source .secret/.key.sh
EVAL_OUT=examples/coding_agent_opd/logs/EXP_NAME/multilingual
mkdir -p "$EVAL_OUT"
BENCHMARK=multilingual \
LLM_URL=http://MODEL_HOST:PORT/v1/chat/completions \
MODEL_NAME=/path/to/hf/model \
OUT_DIR="$EVAL_OUT" \
CONCURRENT=46 SWE_BOOT_CONCURRENCY=46 \
MAX_STEPS_ABS=150 MAX_MODEL_LEN=131072 MAX_TOKENS=0 \
TEMPERATURE=1.0 TOP_P=0.95 \
REQUEST_TIMEOUT=7200 RETRY=5 MAX_TOTAL_TIME=50000 \
HEARTBEAT=60 LOG_TURNS=0 SAVE_TRAJ=0 \
bash examples/coding_agent_opd/eval/run_eval.sh \
  >"$EVAL_OUT/launcher.log" 2>&1
```

推荐步数：Verified 为 100，Pro/Multilingual 为 150。Pod 启动并发
`SWE_BOOT_CONCURRENCY` 通常与 trajectory 并发 `CONCURRENT` 相同。多个模型/数据集
可以并行，但同一个 `OUT_DIR` 绝不能同时启动两个进程。

## 3. 输出与监控

每个 `OUT_DIR` 主要包含：

- `launcher.log` / `run.log`：运行日志与心跳；
- `results.jsonl`：逐题结果；
- `summary.json`：最终汇总；
- `dump/`：完整轨迹（设置 `DUMP_DIR` 时）。

```bash
tail -f OUT_DIR/launcher.log
wc -l OUT_DIR/results.jsonl
jq '{total,resolved,errors,status_breakdown,resolve_rate}' OUT_DIR/summary.json
```

`num_steps` 是题目使用的 step 数，`response_len` 是所有轮次累计的模型输出
token 数，不含输入 prompt。

## 4. 断点补跑

必须等该目录原进程结束后再补跑。`RESUME=1` 会保留已有非错误结果，仅重跑
`error`、`grading_error`、`ds_missing_field` 或缺失题目：

```bash
source .secret/.key.sh
BENCHMARK=multilingual RESUME=1 \
LLM_URL=http://MODEL_HOST:PORT/v1/chat/completions \
MODEL_NAME=/path/to/hf/model OUT_DIR=原目录 \
CONCURRENT=46 SWE_BOOT_CONCURRENCY=46 \
MAX_STEPS_ABS=150 MAX_MODEL_LEN=131072 MAX_TOKENS=0 \
TEMPERATURE=1.0 TOP_P=0.95 REQUEST_TIMEOUT=7200 RETRY=5 \
MAX_TOTAL_TIME=50000 \
bash examples/coding_agent_opd/eval/run_eval.sh \
  >原目录/launcher.resume1.log 2>&1
```

补跑前应另存旧 `summary.json` 和 `run.log`。HTTP 400 上下文耗尽会记为
`truncated`，属于正常终态；RLE session 丢失、Pod 不可用或重试耗尽才属于应补跑的
基础设施错误。

## 5. 汇总口径

- 成功率：`resolved / 数据集总题数`；
- 平均 step：对有数值的 `num_steps` 求均值；
- 平均输出 token：对有数值的 `response_len` 求均值；
- 最终交付前必须确认题数严格为 500 / 731 / 300、`errors == 0`，且没有残留评测进程。

多端点/多数据集并行评测时，为每个 OUT_DIR 独立启动一个 `run_eval.sh` 进程（env 变量按需覆盖），
且同一个 OUT_DIR 绝不能同时运行两个进程。所有配置均应始终以当前 `benchmarks.json`、
模型服务地址和用户指定并发为准。
