# 四机离线 OPD

该目录提供与在线 OPD 隔离的三阶段流程。在线 `multi_nodes/run-*.sh` 不需要改动。

## 数据契约

1. Phase 1 保存完整轨迹、rollout student 的 `student_topk_ids [N,K] int32`，以及同一
   support 上的 `student_topk_logprobs [N,K] float16`；`N` 是 `loss_mask=1` 的位置数。
2. Phase 2 在相同 support id 上计算 teacher log-prob，仅落盘
   `teacher_logprobs_on_student [N,K] float32`。不重复保存 dense
   `student_support_ids [R,K]`，overlap 也只保存 N 行。
3. Phase 3 从文件加载后，按照 `loss_mask` 临时展开回现有 loss 所需的
   `student_support_ids/teacher_logprobs_on_student [R,K]`，因此 loss 和 CP 不变；随后
   从当前 student logits 可微地重算固定 id 上的 student log-prob，执行
   `topk_reverse_kl`。训练阶段没有 teacher 或 rollout engine。

## 为什么 Phase 1 不使用 fully-async

在线 fully-async worker 会跨 rollout 边界预取，并按完成顺序返回数据；这适合持续训练，
但不能证明有限数据集里的每条数据恰好落盘一次。Phase 1 使用 `exact_rollout.py`：

- 每个 batch 只从 dataset 取一次；
- 不跨 batch 预取；
- abort 时重试原样本，不用下一条数据替换；
- batch size 必须整除数据集大小。未显式指定时，脚本选择不大于
  `PREFERRED_ROLLOUT_BATCH_SIZE` 的最大除数。当前 500 条数据默认得到 batch=20、25 个文件。

Phase 1 结束后会验证文件连续、总数正确、`sample.index`/`instance_id` 无重复，以及每个
trainable position 都有 K 个 student ids 和对齐的 float16 student log-prob。

Phase 1 按完成顺序落盘，响应长度会因此与文件编号相关。Phase 2 不再为 shuffle 额外
读取和重写全部 Phase-1 数据；teacher 打分本来就必须读取每个源文件，因此它在打分后
直接把每条 scored sample 原子落盘一次，并写轻量长度元数据。四机全部完成后，rank 0
只读取这些小元数据，用固定 seed=42 做样本级随机化和 trainable-token 负载均衡，再生成
每批 50 条的 `rollout_N.pt` 小 manifest。Phase 3 根据 manifest 并行加载对应样本。
大 tensor 不会为了 shuffle 再复制一份；`PHASE2_SAMPLES_PER_FILE` 控制 manifest batch
大小，`OFFLINE_OPD_MANIFEST_LOAD_WORKERS` 控制训练加载并发（默认 8）。

Phase 1 重启时会先验证已有的连续文件前缀，并从下一个 rollout id 续跑。数据源会一次性
跳过已落盘的样本，因此确定性的 epoch-0 shuffle、`sample.index` 和文件边界都保持不变；
已有文件损坏、编号有缺口或显式 `START_ROLLOUT_ID` 与文件前缀不一致时会直接拒绝启动。

## 训练期离线有效性指标

Phase 3 不启动或请求 teacher。下面所有 teacher 值都来自 Phase 2 缓存；当前 student 的
log-prob 复用本次训练 forward。对旧 rollout support `S`，令 `q_old/q_cur/q_teacher` 为
各自 log-prob 在 S 内经过 softmax 重新归一化后的分布：

- `current_vs_rollout_sparse_kl = sum(q_old * (log(q_old) - log(q_cur)))`
- `current_mass_on_rollout_topk = sum(exp(current_logprob_on_S))`
- `support_mass_retention = current_mass_on_rollout_topk / sum(exp(old_logprob_on_S))`
- `current_vs_cached_teacher_sparse_kl = sum(q_cur * (log(q_cur) - log(q_teacher)))`
- `teacher_kl_improvement = KL(q_old || q_teacher) - KL(q_cur || q_teacher)`
- `teacher_mass_on_rollout_topk = sum(exp(cached_teacher_logprob_on_S))`
- 对每条序列先计算 `u_i = mean_t(exp(current_sampled_token_logprob - rollout_logprob))`。
  `importance_weight_mean/std` 是 `u_i` 的跨序列均值/总体标准差；
  `importance_ratio_ess = sum(u_i)^2 / (B * sum(u_i^2))`，因此长短序列等权。
- `sequence_importance_weight_mean/std`：先计算整序列联合概率比
  `v_i = exp(clip(sum_t(current_logprob-rollout_logprob), -20, 20))`，再统计跨序列
  均值/总体标准差。

## 四机执行

在 hostfile 第一台机器上执行：

```bash
# 1. 四机全部作为 student rollout engine
DUMP_DIR=/shared/offline_opd/run1 \
  bash examples/coding_agent_opd/scripts/offline_opd/start-offline-opd.sh phase1

# 2. 每台机器启动一个本地 TP4×DP2 teacher，动态领取源文件；打分结果逐样本
#    落盘，结束后只对小 manifest 做样本级随机和 token 负载均衡
IN_DIR=/shared/offline_opd/run1 \
OUT_DIR=/shared/offline_opd/run1/scored \
  bash examples/coding_agent_opd/scripts/offline_opd/start-offline-opd.sh phase2

# 3. 四机全部训练；默认 TP8×PP2。启动器默认把一次 teacher scored 数据
#    展开成 10 个 epoch（可通过 TRAIN_EPOCHS 覆盖），不会重新请求 teacher。
#    如果 ROLLOUT_BATCH_SIZE 与 Phase-2 batch 相同且整除数据量，仍走原来的
#    整批软链接路径；否则按样本组成连续 epoch 流，再切成固定大小的训练 batch。
SCORED_DIR=/shared/offline_opd/run1/scored \
SAVE_PATH=/shared/offline_opd/run1/checkpoints \
  bash examples/coding_agent_opd/scripts/offline_opd/start-offline-opd.sh phase3

# 例如数据集 500 条：batch=50 的行为不变且不会跨 epoch；batch=64 时，
# epoch 尾部 52 条会与下一 epoch 开头 12 条组成一个完整 batch，下一批从
# 下一 epoch 的第 13 条继续。改变 batch size 时应启动新训练，不要续训旧
# batch size 产生的 checkpoint。
ROLLOUT_BATCH_SIZE=64 \
GLOBAL_BATCH_SIZE=64 \
SCORED_DIR=/shared/offline_opd/run1/scored \
SAVE_PATH=/shared/offline_opd/run1/checkpoints-b64 \
  bash examples/coding_agent_opd/scripts/offline_opd/start-offline-opd.sh phase3
```

三个阶段都支持在最后增加 `--bg`。Phase 2 依赖共享的 `IN_DIR/OUT_DIR`；每个 rank 只写
自己负责的文件，不存在多进程覆盖同一文件。

## 验证和恢复

```bash
python examples/coding_agent_opd/scripts/offline_opd/verify_offline_pipeline.py \
  --mode dump-dir --dir /shared/offline_opd/run1 --expected-samples 500

python examples/coding_agent_opd/scripts/offline_opd/verify_offline_pipeline.py \
  --mode scored-dir --dir /shared/offline_opd/run1/scored
```

Phase 2 的每条 scored sample 和每个源文件完成 manifest 均通过临时文件加原子 rename
落盘。源 manifest 只在该源文件的全部样本完成后发布；重启会跳过已完整发布的源文件，
不完整源文件会重新计算。
每台机器的持久日志分别写入 `RUN_DIR/phase2.teacher.rank<R>.log` 和
`RUN_DIR/phase2.scorer.rank<R>.log`；未显式设置 `RUN_DIR` 时使用 scored 目录同级的 `logs/`。

## 资源默认值

- Phase 1：四机 32 GPU rollout，默认每个 student engine 2 GPU。
- Phase 2：每机一个 teacher，TP4×DP2、8 GPU；配置与当前 H800 在线 teacher 一致。
- Phase 3：四机 32 GPU trainer，TP8×PP2，剩余维度为 DP2。
- K 默认 1024。每个 trainable position 的三个稀疏矩阵合计约占 `10*K` 字节
  （student id 4B + student log-prob 2B + teacher log-prob 4B）；K=1024 时约
  10 KiB/trainable position。Phase 2 仍只计算一次，Phase 3 默认通过软链接复用 10 个 epoch。
