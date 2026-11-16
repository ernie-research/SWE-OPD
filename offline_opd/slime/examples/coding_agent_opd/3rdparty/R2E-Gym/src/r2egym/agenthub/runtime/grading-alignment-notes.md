# docker.py 打分逻辑与 swebench 官方打分逻辑的对齐分析

## 背景

docker.py 中 `get_logs_eval` 方法（line 1423-1478）是从 swebench 官方 `grading.py` 修改而来，但有多处行为差异。官方代码参考：`_debug/SWE-bench-latest/swebench/harness/grading.py` 的 `get_logs_eval`（line 39-91）。

本文件记录当前两者的差异点，供后续修复对齐时参考。

## 当前链路概述

### 官方 swebench 评估流程

1. 在 Docker 容器内执行 `/bin/bash /eval.sh`（eval.sh 由 `test_spec.eval_script` 生成）
2. eval_script 内部用 `set -uxo pipefail` 启动，`-x` 使 bash trace 模式输出每条命令
3. eval_script 在测试命令前后插入 `: '>>>>> Start Test Output'` 和 `: '>>>>> End Test Output'`（bash `:` 是 no-op，但 `-x` trace 会将这些标记打印到 stderr）
4. Docker `exec_run` 默认同时捕获 stdout+stderr，所以日志中包含标记
5. `get_logs_eval` 读日志文件 → 检查 bad_codes → 检查标记是否存在 → 用标记切割出测试输出 → 调用 log_parser 解析

### docker.py 当前流程

1. `_calculate_reward` 执行 `bash /run_tests.sh 2>&1`（`/run_tests.sh` 内容来自 `test_spec.eval_script`，与官方一致）
2. `get_logs_eval` 接收原始输出字符串 → 检查 bad_codes → 跳过标记检查 → 用 `test_cmd` 做 split → 调用 log_parser 解析

## 4 处影响结果的差异

### Diff 1: 标记存在性检查（被注释掉）

位置：docker.py line 1461-1464

```python
# 官方（grading.py:74-76）
elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
    return {}, False

# docker.py（已注释）
# elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
#     self.logger.error("Test patch did not apply")
#     return {}, False
```

影响方向：docker.py 更宽松。如果测试脚本异常退出导致标记不完整，官方返回 `{}, False`（reward=0），docker.py 仍尝试解析。

修复方式：取消注释，恢复标记检查。需确认 `2>&1` 能保证标记出现在捕获的输出中（因为 `set -x` trace 输出到 stderr）。

### Diff 2: 内容切割方式不同

位置：docker.py line 1467-1468

```python
# 官方（grading.py:79）
test_content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]

# docker.py
if self.backend != "rle":
    content = content.split(test_cmd)[-1]
```

差异：
- 官方用 `START_TEST_OUTPUT` / `END_TEST_OUTPUT` 标记精确切割，只取标记之间的测试输出
- docker.py 用 `test_cmd`（测试命令字符串）做 split 取最后一段，范围更大：包含了 END 标记之后的 git checkout 重置输出等

潜在风险：
1. `test_cmd` 字符串可能在输出中多次出现（set -x trace 打印一次、pytest 自身 echo 一次），`split()[-1]` 取最后一次，通常正确但不绝对
2. 切割范围比官方大，尾部多余内容理论上可能被 log_parser 误匹配

修复方式：改用标记切割。注意处理 `self.backend == "rle"` 的分支。

### Diff 3: 解析失败时无 fallback

位置：docker.py line 1475-1478 vs grading.py:82-89

```python
# 官方（grading.py:82-89）
status_map = log_parser(test_content, test_spec)
if not status_map:
    status_map = log_parser(content, test_spec)  # fallback: 解析整个日志

# docker.py：无 fallback，只解析一次
if len(inspect.signature(log_parser).parameters) == 1:
    return log_parser(content), True
return log_parser(content, test_spec), True
```

影响方向：docker.py 可能漏判。如果测试输出在标记区间外（如 pytest 输出到 stderr 而标记只框住 stdout），官方通过 fallback 兜底，docker.py 可能返回空 status_map。

但由于 docker.py 的 test_cmd split 已经取了比标记区间更大的范围，实际触发概率低。

修复方式：对齐后加上 fallback 逻辑。

### Diff 4: log_parser 调用签名处理

位置：docker.py line 1475-1478

```python
# 官方：始终传 2 个参数
status_map = log_parser(test_content, test_spec)

# docker.py：检查签名，根据参数个数决定传 1 还是 2
if len(inspect.signature(log_parser).parameters) == 1:
    return log_parser(content), True
return log_parser(content, test_spec), True
```

影响：如果某个 repo 的 parser 用 test_spec 做额外过滤（如限定测试名范围），docker.py 在 1 参数分支下会丢失这部分逻辑。当前 swebench 的 Python 项目 parser 都接受 2 参数，此差异主要影响旧版兼容场景。

修复方式：统一传 2 参数。如需兼容旧 parser，保留 inspect 检查但加日志 warning。

## 修复优先级

| 优先级 | Diff | 原因 |
|---|---|---|
| P1 | Diff 2 | 切割方式是最核心的逻辑差异，直接影响 status_map 的输入内容 |
| P1 | Diff 1 | 标记检查是官方的第一道防线，注释掉后异常情况下会虚高 |
| P2 | Diff 3 | fallback 是兜底逻辑，改完 Diff 2 后需要配套加上 |
| P3 | Diff 4 | 影响面最小，当前 parser 都是 2 参数 |

## 验证方法

修复后可用 `trajectory_diffs/` 下的 data 和 replay 轨迹的 test_output 做回归验证：
- 7 条打分不一致的轨迹重新计算 reward，确认结果与官方 `grading.py` 一致
- 抽样其他 reward=1 的轨迹确认没有引入 regression
