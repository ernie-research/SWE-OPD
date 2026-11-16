# Offline OPD 公式与指标口径

本文档固定记录当前代码中 offline OPD 的训练与监测公式。每条公式都同时提供
LaTeX 和纯文本版本；即使 Markdown 查看器不渲染数学公式，也可以直接阅读代码块。

当前代码位置：

- `slime/backends/megatron_utils/loss.py`
- `slime/backends/megatron_utils/cp_utils.py`
- `slime/rollout/on_policy_distillation.py`

## 1. 符号

- `h_t`：固定离线轨迹在位置 `t` 之前的 token 前缀。
- `a_t`：Phase 1 rollout 在位置 `t` **实际采样得到的 token**，不是概率最大的
  top-1 token。
- `pi_rollout`：生成 Phase 1 轨迹时的学生模型。
- `pi_current`：Phase 3 当前训练中的学生模型。
- `pi_teacher`：Phase 2 打分使用的 teacher；训练期间不重新打分。
- `m_t`：loss mask，只有 `m_t = 1` 的 token 参与以下统计。
- `N_i`：序列 `i` 中满足 `m_{i,t} = 1` 的 sampled token 数。
- `B`：当前全局训练 step 中至少包含一个有效 sampled token 的序列数。

## 2. Sampled-token importance weight

Phase 1 采样过程：

$$
a_t \sim \pi_{\mathrm{rollout}}(\cdot\mid h_t)
$$

纯文本：

```text
a_t ~ pi_rollout(. | h_t)
```

原始 log-ratio：

$$
d_t = \log \pi_{\mathrm{current}}(a_t\mid h_t)
      - \log \pi_{\mathrm{rollout}}(a_t\mid h_t)
$$

纯文本：

```text
d_t = log pi_current(a_t | h_t) - log pi_rollout(a_t | h_t)
```

当前实现为了数值安全，将 log-ratio 截断到 `[-20, 20]` 后计算：

$$
w_t = \exp\!\left(\operatorname{clip}(d_t,-20,20)\right)
$$

纯文本：

```text
w_t = exp(clip(d_t, -20, 20))
    = pi_current(a_t | h_t) / pi_rollout(a_t | h_t)  # 未触发截断时
```

`pi_current` 的 log-prob 在该指标路径中已经 `detach`，因此这些监测指标不参与
反向传播。

注意：虽然概率本身不超过 1，但 ratio 没有上限 1。若 rollout 概率很小，则：

$$
w_t \leq \frac{1}{\pi_{\mathrm{rollout}}(a_t\mid h_t)}
$$

纯文本：

```text
w_t <= 1 / pi_rollout(a_t | h_t)
```

## 3. importance_weight_mean

当前实现先在每条序列内对有效 sampled-token ratio 求均值：

$$
u_i = \frac{1}{N_i}\sum_{t:m_{i,t}=1} w_{i,t}
$$

然后让每条序列等权，计算序列间均值：

$$
\mu_u = \frac{1}{B}\sum_{i=1}^{B}u_i
$$

纯文本：

```text
u_i = mean(w_i_t for valid tokens in sequence i)
importance_weight_mean = mean(u_i across sequences)
```

该结果通过 `sum(u)`、`count` 先跨 micro-batch、DP 和 CP 聚合，再执行除法。
因此它不会受到 micro-batch 划分或 CP 切分的系统性影响。

这是 **sequence-weighted mean**：长短序列各贡献一个 `u_i`，权重相同。

在以 rollout 分布真实采样且没有截断、mask 选择偏差及有限样本误差的理想条件下：

$$
\mathbb{E}_{a\sim\pi_{\mathrm{rollout}}}
\left[\frac{\pi_{\mathrm{current}}(a\mid h)}
{\pi_{\mathrm{rollout}}(a\mid h)}\right]=1
$$

纯文本：

```text
E_{a ~ pi_rollout}[pi_current(a|h) / pi_rollout(a|h)] = 1
```

所以 mean 接近 1 只是重要性权重的归一化检查；它本身不能证明两个策略接近。
mean 可以接近 1，同时 std 很大。

## 4. importance_weight_std

当前实现是真正以观测均值 `mu_u` 为中心的总体标准差：

$$
\sigma_u =
\sqrt{
\frac{1}{B}\sum_{i=1}^{B}u_i^2-\mu_u^2
}
$$

等价形式：

$$
\sigma_u =
\sqrt{
\frac{1}{B}\sum_{i=1}^{B}(u_i-\mu_u)^2
}
$$

纯文本：

```text
importance_weight_std = sqrt(mean(u_i^2) - importance_weight_mean^2)
                      = sqrt(mean((u_i - importance_weight_mean)^2))
```

代码先跨 micro-batch、DP 和 CP 聚合 `sum(u)`、`sum(u^2)`、`count`，最后才开方，
所以它是严格的全局序列间标准差，而不是 token 标准差或各 micro-batch 标准差的平均。

## 5. 与 Lightning-OPD 的 RMS-from-one 指标区别

部分 Lightning-OPD 版本将所谓的 `importance_weight_std` 定义为：

$$
R_1 = \sqrt{\mathbb{E}[(w-1)^2]}
$$

纯文本：

```text
RMS_deviation_from_one = sqrt(mean((w_t - 1)^2))
```

它既不是通常意义上的标准差，观测单位也不同：Lightning-OPD 的公式通常把每个
token 的 `w_t` 作为观测；当前实现把每条序列的 `u_i` 作为观测。因此二者不能只靠
当前的 mean/std 精确互相换算，因为序列内 token 方差已被平均掉。

若另外在相同的序列观测 `u_i` 上定义：

$$
R_{\mathrm{seq},1}=\sqrt{\mathbb{E}_i[(u_i-1)^2]}
$$

此时才有：

$$
R_{\mathrm{seq},1}^2 = \sigma_u^2 + (\mu_u-1)^2
$$

纯文本：

```text
sequence_RMS_deviation_from_one^2
    = importance_weight_std^2 + (importance_weight_mean - 1)^2
```

当 `mean` 接近 1 且 `std` 很大时，两者数值接近；但概念和聚合口径不同。

## 6. importance_ratio_ess

当前实现记录归一化 effective sample size：

$$
\mathrm{ESS}_{\mathrm{norm}}
=\frac{\left(\sum_i u_i\right)^2}
{B\sum_i u_i^2}
$$

使用当前 mean/std 可以写成：

$$
\mathrm{ESS}_{\mathrm{norm}}
=\frac{\mu_u^2}{\mu_u^2+\sigma_u^2}
$$

纯文本：

```text
importance_ratio_ess = sum(u_i)^2 / (B * sum(u_i^2))
                     = mean^2 / (mean^2 + std^2)
```

范围通常为 `(0, 1]`：

- 越接近 1：权重集中在 1 附近，离线数据与当前策略较匹配。
- 越接近 0：少量序列的巨大平均权重主导统计，off-policy mismatch 严重。

## 7. 整序列 importance weight

新增指标使用整条 response 的联合概率比。实现中在 log-space 求和，避免直接相乘下溢：

$$
D_i=\sum_{t:m_{i,t}=1}d_{i,t},\qquad
v_i=\exp\!\left(\operatorname{clip}(D_i,-20,20)\right)
$$

未触发截断时：

$$
v_i=
\frac{\prod_t\pi_{\mathrm{current}}(a_{i,t}\mid h_{i,t})}
{\prod_t\pi_{\mathrm{rollout}}(a_{i,t}\mid h_{i,t})}
$$

跨序列记录总体均值和总体标准差：

```text
offline_opd/sequence_importance_weight_mean = mean(v_i)
offline_opd/sequence_importance_weight_std  = sqrt(mean(v_i^2) - mean(v_i)^2)
```

它比 `u_i` 对连续多个 token 的小偏差敏感得多，长序列也更容易触发 `exp(±20)`
截断，因此应主要作为漂移告警指标解读。

## 8. 当前 sampled-token OPD loss

当前 `token_reverse_kl` 对每个 sampled token 使用：

$$
L_t = \log\pi_{\mathrm{current}}(a_t\mid h_t)
      -\log\pi_{\mathrm{teacher}}(a_t\mid h_t)
$$

纯文本：

```text
L_t = log pi_current(a_t | h_t) - log pi_teacher(a_t | h_t)
```

teacher log-prob 是 Phase 2 保存的常量，因此当前直接梯度为：

$$
\nabla_\theta L_t
=\nabla_\theta\log\pi_{\mathrm{current}}(a_t\mid h_t)
$$

纯文本：

```text
grad(L_t) = grad(log pi_current(a_t | h_t))
```

也就是说，当前形式中的 teacher 项改变 loss 数值但不改变梯度。它不等价于完整的
reverse-KL 梯度；这是分析 sampled-token 实验中策略漂移时必须单独考虑的问题。

## 9. 实际异常示例

在 `iter_0000059` 后的复现实验中，某个位置的 rollout 分布为：

```text
rollout top-1 token : "/run",   probability = 0.999574275
actual sampled token: "/admin", probability = 0.000261147
current probability of "/admin"              = 0.999997854
importance weight                             = 3829.25049
```

计算为：

```text
0.999997854 / 0.000261147 = 3829.25
```

这说明 `a_t` 是实际 sampled token，而不是 rollout 的 top-1 token。

## 10. train/opd_topk_overlap

`train/opd_topk_overlap` 衡量 Phase 1 rollout 学生与 Phase 2 teacher 在同一个
离线 token 前缀上的全词表 Top-K 集合重合度。它不是当前训练模型与 teacher 的
重合度；`train/` 只是日志命名空间。

对序列 `i` 的第 `t` 个有效 response 位置，定义：

$$
S^{\mathrm{rollout}}_{i,t}
=\operatorname{TopK}\!\left(
\pi_{\mathrm{rollout}}(\cdot\mid h_{i,t})
\right)
$$

$$
S^{T}_{i,t}
=\operatorname{TopK}\!\left(
\pi_{\mathrm{teacher}}(\cdot\mid h_{i,t})
\right)
$$

单个位置的 overlap ratio 为：

$$
o_{i,t}
=
\frac{
\left|S^{\mathrm{rollout}}_{i,t}\cap S^T_{i,t}\right|
}{K}
$$

纯文本：

```text
rollout_set = TopK(pi_rollout(. | h_i_t))
teacher_set = TopK(pi_teacher(. | h_i_t))
o_i_t       = size(rollout_set intersect teacher_set) / K
```

例如当前 Top-128 实验中，若两边有 80 个 token 相同：

```text
o_i_t = 80 / 128 = 0.625
```

### 10.1 默认归约口径

当前默认 `calculate_per_token_loss = False`，因此先对每条序列的有效 response token
求均值：

$$
\bar o_i
=
\frac{
\sum_t m_{i,t}o_{i,t}
}{
\sum_t m_{i,t}
}
$$

再让 global batch 中的序列等权平均：

$$
\texttt{train/opd\_topk\_overlap}
=
\frac{1}{B}\sum_{i=1}^{B}\bar o_i
$$

纯文本：

```text
sequence_overlap_i = mean(o_i_t over valid response tokens in sequence i)
train/opd_topk_overlap = mean(sequence_overlap_i across global-batch sequences)
```

因此 Top-128 实验中的 `train/opd_topk_overlap = 0.603` 可以近似理解为：

```text
128 * 0.603 = 77.2
```

即平均每个有效位置约有 77 个 Top-128 token 相同。这里是多个位置、多个序列的
平均值，所以 `77.2` 不是某一个位置的整数交集大小。

若显式启用 `calculate_per_token_loss = True`，归约会改为所有有效 token 等权，而不是
序列等权；长 response 会贡献更多权重。

### 10.2 Offline OPD 中该值为什么不会跟随模型更新

该指标的数据流为：

1. Phase 1 保存 rollout 学生在每个可训练位置的 Top-K token IDs。
2. Phase 2 teacher 在相同前缀上计算自己的全词表 Top-K，并保存两者的交集比例。
3. Phase 3 只加载并记录这个缓存值，不使用 `pi_current` 重新计算。

所以同一条样本跨 epoch 的 `o_i_t` 完全相同。训练曲线的 step 间波动仅表示每个
global batch 包含的样本不同；数据顺序跨 epoch 重复时，曲线也可能呈周期性重复。
该指标已经 detach，只用于监测，不参与 loss 或反向传播。

它实际回答的是：

```text
生成离线数据时，rollout 学生与 teacher 的 Top-K support 有多一致？
```

它不能回答：

```text
当前训练模型相对 rollout 模型已经漂移了多少？
```

后一个问题应读取：

```text
offline_opd/current_vs_rollout_topk_overlap
```

其公式为：

$$
\frac{
\left|
\operatorname{TopK}(\pi_{\mathrm{current}})
\cap
\operatorname{TopK}(\pi_{\mathrm{rollout}})
\right|
}{K}
$$

### 10.3 与 train/opd_topk_overlap_atM 的区别

`train/opd_topk_overlap_at1/2/4/8/16/32/64` 仍比较 rollout 学生与 teacher，仍不涉及
当前训练模型。对于 `M <= K`，rollout Top-M 是 Phase 1 Top-K support 的前 `M` 列；
teacher Top-M 则根据 Phase 2 缓存的 teacher log-prob，在这个 rollout Top-K support
内部重新排序得到：

$$
\text{overlap@M}_{i,t}
=
\frac{
\left|
\operatorname{TopM}(\pi_{\mathrm{rollout}})
\cap
\operatorname{TopM}_{S^{\mathrm{rollout}}}(\pi_{\mathrm{teacher}})
\right|
}{M}
$$

纯文本：

```text
train/opd_topk_overlap      = rollout global Top-K vs teacher global Top-K
train/opd_topk_overlap_atM  = rollout Top-M vs teacher Top-M ranked inside rollout Top-K
offline_opd/current_vs_rollout_topk_overlap
                            = current global Top-K vs rollout global Top-K
```

因此不带后缀的 `train/opd_topk_overlap` 是 Phase 2 直接计算的全词表 Top-K 集合指标；
`atM` 是 support 内的头部排名指标。只有当 teacher 在 rollout Top-K support 上的概率
质量接近 1 时，support 内的 teacher Top-M 才近似等于 teacher 的全词表 Top-M。

## 11. Dynamic current Top-128 / cached teacher Top-4096

该隔离模式使用：

$$
C_t=\operatorname{Top4096}(\pi_T(\cdot\mid s_t)),\qquad
S_t=\operatorname{Top128}(\pi_{\mathrm{current}}(\cdot\mid s_t)).
$$

Phase 2 只缓存 $C_t$。对 current token $v\notin C_t$，不顺延 support，而使用
teacher 第 4096 名的 log-prob：

$$
\widetilde{\ell}_T(v)=
\begin{cases}
\log\pi_T(v\mid s_t),&v\in C_t,\\
\ell_T^{(4096)},&v\notin C_t.
\end{cases}
$$

训练 loss 为：

$$
L_t=\sum_{v\in S_t}q_t(v)
\left[\log\pi_{\mathrm{current}}(v\mid s_t)-\widetilde{\ell}_T(v)\right],
\qquad
q_t(v)=\frac{\pi_{\mathrm{current}}(v\mid s_t)}
{\sum_{u\in S_t}\pi_{\mathrm{current}}(u\mid s_t)}.
$$

动态指标：

$$
\text{cache-hit-ratio}_t=\frac{|S_t\cap C_t|}{128},
$$

$$
\text{miss-student-mass}_t=
\sum_{v\in S_t\setminus C_t}\pi_{\mathrm{current}}(v\mid s_t),
$$

$$
\text{loss-weight-on-imputed}_t=
\sum_{v\in S_t\setminus C_t}q_t(v).
$$

teacher 在 current support 上的缓存下界和 cutoff 上界分别为：

$$
M_t^{\mathrm{lower}}=
\sum_{v\in S_t\cap C_t}\pi_T(v\mid s_t),
$$

$$
M_t^{\mathrm{upper}}=
\min\left(1,
M_t^{\mathrm{lower}}+|S_t\setminus C_t|\exp(\ell_T^{(4096)})
\right).
$$

`train/teacher_mass` 使用这个 cutoff-imputed 上界估计；
`train/opd_topk_overlap` 则是 current 全词表 Top-128 与 teacher 全词表 Top-128 的精确集合重合率。

与 `train/opd_topk_overlap_atM` 相同，dynamic offline OPD 还报告
`train/student_mass_atM` 和 `train/teacher_mass_atM`，其中
`M in {1,2,4,8,16,32,64}`。令 current student 的 Top-M 为
`S_t^(M)`，则：

$$
\text{student-mass-atM}_t=
\sum_{v\in S_t^{(M)}}\pi_{\mathrm{current}}(v\mid s_t),
$$

$$
\text{teacher-mass-atM}_t=
\sum_{v\in S_t^{(M)}}\exp\left(\widetilde{\ell}_T(v)\right).
$$

后者和完整的 `train/teacher_mass` 一样，对 cache miss 使用 cutoff
log-prob，因此是上界估计。不带 `atM` 后缀的指标仍表示完整 Top-128。
Phase 3 的 `[N_trainable,4096]` cache 始终保持 sparse，只将当前 CP rank 的有效行分块搬到 GPU，
不会展开成 `[response_length,4096]`。
