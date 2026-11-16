# GLM-4.5-Air Student / GLM-5.2 Teacher 模板与词表对齐

本文是自包含的复现规范：给出数据契约、模板转换、跨词表映射、位置对齐、稀疏落盘、Phase 3 loss、参数和验收条件。实现者不需要读取其他说明材料即可复现该方法。该能力应作为 opt-in 路径实现；不启用时保持标准单 tokenizer OPD 行为不变。

## 1. 当前方案

- Student：GLM-4.5-Air，Phase 1 使用 Student 自己的 chat template 生成轨迹。
- Teacher：GLM-5.2，Phase 2 将轨迹恢复成结构化 messages，再用 Teacher 官方 chat template 重渲染完整上下文。
- Teacher 推理强度：由 `--opd-teacher-reasoning-effort` 指定，当前全量脚本使用 `high`。
- Teacher cache：每个严格对齐位置保存 Teacher Top-1000；Phase 3 从中选择 Student 可表达的 Top-128。
- 对齐原则：只有文本、字符跨度、UTF-8 bytes 和 bridge 后 token ID 全部相同的一对一位置使用 Teacher KL。
- 其他可训练位置不丢弃，使用 Phase 1 保存的 rollout Student Top-K 分布作为 Student anchor。
- 当前 loss：Teacher 对齐位置使用 `0.5 * forward KL + 0.5 * reverse KL`；anchor 位置使用 rollout Student 到 current Student 的 sparse KL，默认系数为 1.0。

本文固定复现实验使用：Teacher cache K=1000、训练 support K=128、Teacher forward/reverse KL 系数各 0.5、Student anchor 系数 1.0、Teacher `reasoning_effort=high`。

## 2. 数据流

```text
Phase 1: GLM-4.5-Air rollout
  Student prompt/response token IDs
  loss_mask
  sampled-token logprob
  Student sparse Top-K IDs/logprobs
                 |
                 v
Phase 2: rebuild_teacher_trajectory()
  校验并恢复 Student prompt/messages/tool observations
  将 assistant 文本解析为 reasoning/content/tool_calls
  使用 GLM-5.2 官方模板重渲染完整历史
  对 Student/Teacher token 做严格一对一对齐
           /                         \
          v                           v
  精确对齐位置                     未精确对齐位置
  Teacher Top-1000 cache            rollout Student Top-128 anchor
           \                         /
            v                       v
Phase 3: cached_teacher_bidirectional_topk_kl_with_student_anchor
  Teacher-only token 跳过并从 cache 后续候选顺延
  Teacher 行计算双向 KL；anchor 行约束 current Student 不偏离 rollout Student
```

## 3. Phase 2 的模板切换

Phase 2 的入口执行：

1. 检测 `--opd-teacher-template-model`，并要求同时提供 `--opd-token-id-bridge`。
2. 调用下文定义的 `rebuild_teacher_trajectory()`。
3. 将重建后的 Teacher token IDs 和 Teacher 位置发送给打分服务。

canonical 重建执行：

1. 从样本 metadata 中恢复 `problem_statement`、workdir、tools 和初始 messages。
2. 用 Student tokenizer/template 重新渲染初始 prompt，并与 Phase 1 保存的 prompt IDs 做逐 token 校验。不同则直接失败。
3. 将 response 的 `loss_mask` 压缩成连续 run：`True` run 是一轮 Student assistant 输出；其后的 `False` run 是环境 observation connector。首个 run 必须为 `True`，多轮时必须按 `True/False/True` 交替。
4. `_canonical_assistant_message()` 将 GLM-4.5 的原始 assistant 输出解析为 Teacher 模板所需的：
   - `reasoning_content`
   - `content`
   - 结构化 `tool_calls`
5. `_render_template()` 调用 GLM-5.2 tokenizer 的官方 `apply_chat_template()`；`reasoning_effort=high/max` 也在这里传入。
6. `_template_turn_deltas()` 取得每轮 assistant/observation 的 Teacher 原生模板增量。对 assistant 做位置对齐时，候选序列还包含后续 connector 的首个 token，以覆盖“当前 assistant 最后一个边界 token 由下一角色模板提供”的情况。
7. Student observation 文本通过 sentinel 从已保存 connector 中恢复，再用同一 Student history 重渲染；只有 token IDs 完全相同才接受。
8. 将 Teacher assistant delta 和完整 observation delta 依次追加，保持完整多轮因果上下文。
9. 最后对完整 `teacher_messages` 再执行一次官方模板渲染，并断言结果与增量拼接的 Teacher IDs 完全一致。

因此 Teacher 实际看到的是自己的完整原生模板上下文，不是 Student 模板，也不是把 Student 整串 ID 直接映射后回放。

## 4. 跨 tokenizer 的位置对齐

位置对齐分成两步：

- `_canonical_student_offsets()`：将 Student 已采样 IDs 解码后重新 canonical encode，建立“已采样位置 -> canonical Student 位置”。
- `_align_canonical_turn()`：在 Student/Teacher 相同文本区间内寻找 token 级一对一位置。

一个 Student token 只有同时满足以下条件才会成为 Teacher loss 行：

1. 已采样 Student token 能对应到 canonical Student token；
2. Student token 和 Teacher token 覆盖完全相同的字符跨度；
3. 两侧该跨度的 UTF-8 bytes 完全相同；
4. Student ID 经 token-id bridge 映射后等于该 Teacher ID；
5. 所有匹配严格单调且一对一。

例如 Student 的两个或三个 token 被 Teacher 合并成一个 token（2→1/3→1），或 Teacher 将一个 Student token 拆开（1→N），均不会做概率近似、复制或顺移，而是转入 Student anchor。

## 5. Token-ID bridge

bridge 按 tokenizer 的 token 字符串建立：

```text
student token ID -> 相同 token 字符串对应的 teacher token ID
```

构建时强制检查：

- 每个有效 Student token 都必须在 Teacher tokenizer 中存在；
- Student→Teacher 映射必须是 injective；
- GLM 共享 base vocabulary 必须是 identity prefix；
- tokenizer size、model vocab size、tokenizer 文件 hash 和 mapping hash 写入 JSON；
- Student 模型的 padded vocab IDs 保持 `-1`，禁止使用。

加载时还会生成反向表 `teacher_to_student`。Teacher 独有 token 的反向值为 `-1`。

bridge 有两个不同用途：

- Phase 2 请求 Teacher 打分时，把 Student token/support ID 转为 Teacher ID 空间；
- Phase 3 使用 Teacher-owned cache 时，把 Teacher 候选 ID 转回 Student ID 空间。

它只解决“同一个 token 字符串使用不同 ID”的问题；模板和 1→N/N→1 分词差异由 canonical 重渲染与严格位置对齐解决。

## 6. Teacher-only token 的处理

Phase 3 从 Teacher Top-1000 cache 构造训练用 Top-128：

1. 按 Teacher 概率顺序读取候选；
2. 用 `teacher_to_student` 映射到 Student 词表；
3. 映射为 `-1` 的 Teacher-only token 跳过；
4. 从后续概率最高的、Student 可表达的候选顺延，直到凑齐 128 个；
5. 如果整个 cache 仍不足 128 个可映射 token，直接报错；
6. 最终支持保持 Teacher 概率降序。

“顺延”只用于 Teacher Top-K 支持集合，不用于轨迹位置对齐。

## 7. Student anchor

Phase 2 将 Phase 1 的所有 trainable positions 严格划分成两个互补集合：

- Teacher positions：严格一对一对齐，保存 Teacher sparse cache；
- Student anchor positions：其余位置，保存 Phase 1 rollout Student sparse Top-K。

anchor reason 仅用于审计，不改变 loss：

- `0`：普通 tokenizer/template mismatch；
- `1`：Student 生成的 `\n<think>`，在 Teacher canonical 模板中由模板提供；
- `2`：Student-only 工具格式结构性换行；
- `3`：observation boundary。

Phase 3 在 anchor 行计算 rollout Student 分布到 current Student 分布的 sparse KL（含显式 tail bucket），避免这些无法可靠映射的位置被 Teacher 模板或词表差异错误监督。它不是把这些位置简单 mask 掉。

## 8. Phase 2 落盘字段与 Phase 3 消费

Teacher 对齐部分：

- `teacher_candidate_positions`
- `teacher_candidate_ids`
- `teacher_candidate_logprobs`
- `teacher_top128_ids`（历史字段名；实际宽度由 dynamic support K 决定）
- `teacher_cutoff_logprobs`
- `teacher_top4096_mass`（历史字段名；当前 cache 可以是 Top-1000）

Student anchor 部分：

- `student_anchor_positions`
- `student_anchor_topk_ids`
- `student_anchor_topk_logprobs`
- `student_anchor_reason`

这些字段必须随样本从磁盘原样传入 Phase 3；第 12 节给出完整 shape/dtype 契约。

## 9. 当前主开关

```bash
OFFLINE_OPD_LOSS_TYPE=cached_teacher_bidirectional_topk_kl_with_student_anchor
OPD_TOKEN_ID_BRIDGE=/path/to/glm45air_to_glm52_token_ids.json
OPD_TEACHER_CANONICAL_TEMPLATE=1
OPD_TEACHER_REASONING_EFFORT=high
OPD_TEACHER_NATIVE_REASONING_PREFIX=0
OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS=0
OPD_TEACHER_NORMALIZE_STUDENT_TOOL_TEMPLATE=0
OPD_TEACHER_MASK_STUDENT_PRE_TOOL_CONTENT=0
OPD_TEACHER_CACHE_TOPK=1000
OPD_DYNAMIC_SUPPORT_TOPK=128
OPD_FORWARD_KL_COEF=0.5
OPD_REVERSE_KL_COEF=0.5
OPD_STUDENT_ANCHOR_KL_COEF=1.0
```

canonical 模式与旧的 native-reasoning-prefix 模式互斥，启动时必须验证这一点。

## 10. 实现隔离要求

跨模板/跨词表方案只能在同时满足以下条件时启用：

- 提供 token-ID bridge；
- 提供 Teacher tokenizer/model；
- `canonical_template=1`；
- loss 类型为 `cached_teacher_bidirectional_topk_kl_with_student_anchor`；
- Phase 1 已保存 Student sparse Top-K log-prob。

禁止同时启用 `native_reasoning_prefix`。旧式的 Student-token replay、格式 token mask、工具换行 normalization 和 pre-tool-content mask 均不属于本复现方案。未满足 opt-in 条件时，程序不得改变输入 token、mask、缓存格式或标准 OPD loss。

## 11. 核心代码摘录

下面给出复现所需的核心实现。为保持篇幅可读，外围日志和重试代码不在算法范围内；数据校验条件已集中列在第 15 节。

### 11.1 入口：选择 Teacher 模板重建路径

```python
teacher_template_model = getattr(args, "opd_teacher_template_model", None)
if teacher_template_model:
    if bridge is None:
        raise ValueError(
            "--opd-teacher-template-model requires --opd-token-id-bridge"
        )

    rebuilt = rebuild_teacher_trajectory(
        sample,
        bridge,
        teacher_template_model,
        context=f"sample index={sample.index}",
        canonicalize_teacher_template=bool(
            getattr(args, "opd_teacher_canonical_template", False)
        ),
        partition_student_anchors=bool(
            getattr(args, "opd_cross_tokenizer_student_anchor", False)
        ),
        teacher_reasoning_effort=getattr(
            args, "opd_teacher_reasoning_effort", None
        ),
    )
    input_ids = rebuilt.input_ids
    teacher_prompt_len = rebuilt.prompt_length
    teacher_scored_positions = rebuilt.scored_positions

    if getattr(args, "opd_teacher_canonical_template", False):
        sample.opd_teacher_student_scored_positions = (
            rebuilt.student_scored_positions
        )
        sample.opd_teacher_alignment_mode = "canonical_template_v1"
        sample.opd_teacher_alignment_stats = rebuilt.alignment_stats
        if getattr(args, "opd_cross_tokenizer_student_anchor", False):
            sample.opd_teacher_student_anchor_positions = (
                rebuilt.student_anchor_positions
            )
            sample.opd_teacher_student_anchor_reasons = (
                rebuilt.student_anchor_reasons
            )
```

### 11.2 使用 Teacher 官方模板渲染

```python
def _render_template(
    tokenizer,
    messages,
    *,
    tools,
    add_generation_prompt: bool,
    reasoning_effort: str | None = None,
):
    template_kwargs = {}
    if reasoning_effort is not None:
        template_kwargs["reasoning_effort"] = reasoning_effort

    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return list(ids), text
```

Student 原始 assistant 文本先变成 Teacher 模板理解的结构化字段：

```python
def _canonical_assistant_message(assistant_text, scaffold, *, context):
    prefix = "\n<think>"
    close = "</think>"
    has_reasoning_prefix = assistant_text.startswith(prefix)
    reasoning_start = len(prefix) if has_reasoning_prefix else 0
    close_at = assistant_text.find(close, reasoning_start)

    if close_at < 0:
        if not has_reasoning_prefix:
            reasoning = ""
            remainder = assistant_text
        else:
            raw_tool_at = assistant_text.rfind("<tool_call>")
            if raw_tool_at < reasoning_start:
                reasoning = assistant_text[reasoning_start:]
                remainder = ""
            else:
                reasoning = assistant_text[reasoning_start:raw_tool_at]
                remainder = assistant_text[raw_tool_at:]
    else:
        reasoning = assistant_text[reasoning_start:close_at]
        remainder = assistant_text[close_at + len(close):]

    message = {
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": remainder,
    }

    # 若 Phase-1 parser 确认最终 tool call 有效，则转换成结构化调用；
    # malformed/未执行的调用继续作为普通 content，不虚构工具执行。
    tool_at = remainder.rfind("<tool_call>")
    if tool_at >= 0:
        _, action = agent_action.parse(
            assistant_text,
            scaffold,
            tool_call_parser="glm45",
            reasoning_parser="glm45",
        )
        if action is not None and remainder.rstrip().endswith("</tool_call>"):
            message["content"] = remainder[:tool_at]
            message["tool_calls"] = [{
                "type": "function",
                "function": {
                    "name": action.function_name,
                    "arguments": action.parameters,
                },
            }]
    return message
```

还必须再次只解析 `remainder[tool_at:]`，并验证其函数名和参数与完整消息 parser 的结果一致；否则保留完整原文为普通 `content`，不得构造结构化工具调用。

每轮 Teacher 模板增量必须通过“相同前缀历史的两次完整渲染”求差，不能手写特殊 token：

```python
def _template_turn_deltas(
    teacher_tokenizer,
    assistant_message,
    observation_message,
    *,
    reasoning_effort,
):
    probe = [{"role": "user", "content": "SLIME_TEMPLATE_ALIGNMENT_PROBE"}]
    base_ids, base_text = _render_template(
        teacher_tokenizer,
        probe,
        tools=None,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    assistant_ids, assistant_text = _render_template(
        teacher_tokenizer,
        probe + [assistant_message],
        tools=None,
        add_generation_prompt=False,
        reasoning_effort=reasoning_effort,
    )
    if assistant_ids[:len(base_ids)] != base_ids:
        raise ValueError("Teacher assistant render is not prompt-prefixed")

    assistant_delta_ids = assistant_ids[len(base_ids):]
    assistant_delta_text = assistant_text[len(base_text):]
    if observation_message is None:
        return assistant_delta_ids, assistant_delta_text, [], ""

    full_ids, full_text = _render_template(
        teacher_tokenizer,
        probe + [assistant_message, observation_message],
        tools=None,
        add_generation_prompt=True,
        reasoning_effort=reasoning_effort,
    )
    if full_ids[:len(assistant_ids)] != assistant_ids:
        raise ValueError("Teacher observation render is not assistant-prefixed")
    return (
        assistant_delta_ids,
        assistant_delta_text,
        full_ids[len(assistant_ids):],
        full_text[len(assistant_text):],
    )
```

从 Student connector 恢复 observation 时使用唯一 sentinel，而不是按换行或特殊 token 猜边界：

```python
def recover_observation(
    connector_ids,
    *,
    student_tokenizer,
    render_student_observation_delta,
):
    sentinel = "SLIME_OBSERVATION_ALIGNMENT_SENTINEL_7f9c"
    sentinel_ids = render_student_observation_delta(sentinel)
    sentinel_text = student_tokenizer.decode(
        sentinel_ids, skip_special_tokens=False
    )
    marker = sentinel_text.find(sentinel)
    if marker < 0:
        raise ValueError("observation sentinel did not survive rendering")

    prefix = sentinel_text[:marker]
    suffix = sentinel_text[marker + len(sentinel):]
    actual = student_tokenizer.decode(
        connector_ids, skip_special_tokens=False
    )
    if not actual.startswith(prefix) or (suffix and not actual.endswith(suffix)):
        raise ValueError("connector does not match Student observation template")

    end = len(actual) - len(suffix) if suffix else len(actual)
    observation = actual[len(prefix):end]
    if render_student_observation_delta(observation) != connector_ids:
        raise ValueError("recovered observation is not token-exact")
    return observation
```

`render_student_observation_delta(x)` 的定义是：在该轮当时的完整 Student message history、相同 tools 和相同前一 token 条件下，追加一条 observation/tool-result message，返回新增的 token IDs。它在 sentinel 和真实 observation 两次调用中必须使用完全相同的历史。

### 11.3 严格的一对一 token 对齐

先把任意 sampled Student token 序列映射到其 canonical re-encode 序列。模型可能采样出合法但非 canonical 的切分；`SequenceMatcher` 没匹配上的 sampled token 不获得字符跨度，稍后自然进入 anchor：

```python
from difflib import SequenceMatcher


def _canonical_student_offsets(tokenizer, text, stored_ids):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    canonical_ids = list(encoded["input_ids"])
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    if len(offsets) != len(canonical_ids):
        raise ValueError("invalid Student offset mapping")

    stored_to_canonical = {}
    matcher = SequenceMatcher(None, stored_ids, canonical_ids, autojunk=False)
    for stored_start, canonical_start, size in matcher.get_matching_blocks():
        for delta in range(size):
            stored_to_canonical[stored_start + delta] = canonical_start + delta
    return canonical_ids, offsets, stored_to_canonical


def _encode_with_offsets(tokenizer, text, expected_ids):
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    ids = list(encoded["input_ids"])
    if ids != expected_ids:
        raise ValueError("text does not round-trip to expected token IDs")
    offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
    if len(offsets) != len(ids):
        raise ValueError("invalid Teacher offset mapping")
    return offsets
```

```python
def _align_canonical_turn(
    student_tokenizer,
    teacher_tokenizer,
    bridge,
    student_ids,
    teacher_ids,
    teacher_text,
    *,
    context,
):
    student_text = student_tokenizer.decode(
        student_ids, skip_special_tokens=False
    )
    _, student_offsets, stored_to_canonical = _canonical_student_offsets(
        student_tokenizer, student_text, student_ids
    )
    teacher_offsets = _encode_with_offsets(
        teacher_tokenizer,
        teacher_text,
        teacher_ids,
        context=f"{context}: Teacher",
    )

    teacher_by_span = {}
    for index, span in enumerate(teacher_offsets):
        if span[1] > span[0]:
            teacher_by_span.setdefault(span, []).append(index)

    mapped_student_ids = bridge.map_numpy(
        np.asarray(student_ids, dtype=np.int32),
        context=f"{context}: Student ids",
    )
    aligned = []
    used_teacher = set()

    # 先寻找 Student/Teacher 文本完全相同的字符区间。
    matcher = SequenceMatcher(None, student_text, teacher_text, autojunk=False)
    for student_char, teacher_char, size in matcher.get_matching_blocks():
        if size <= 0:
            continue
        student_stop = student_char + size

        for student_index, canonical_index in stored_to_canonical.items():
            start, end = student_offsets[canonical_index]
            if start < student_char or end > student_stop or end <= start:
                continue

            teacher_span = (
                teacher_char + start - student_char,
                teacher_char + end - student_char,
            )
            candidates = teacher_by_span.get(teacher_span, ())
            if len(candidates) != 1:
                continue

            teacher_index = candidates[0]
            if teacher_index in used_teacher:
                continue
            if mapped_student_ids[student_index] != teacher_ids[teacher_index]:
                continue
            if (
                student_text[start:end].encode("utf-8")
                != teacher_text[teacher_span[0]:teacher_span[1]].encode("utf-8")
            ):
                continue

            aligned.append((student_index, teacher_index))
            used_teacher.add(teacher_index)

    aligned.sort()
    if any(
        left[0] >= right[0] or left[1] >= right[1]
        for left, right in zip(aligned, aligned[1:])
    ):
        raise ValueError("canonical token alignment is not strictly monotonic")
    return aligned
```

### 11.4 强制归入 Student anchor 的结构位置

```python
aligned = _align_canonical_turn(
    student_tokenizer,
    teacher_tokenizer,
    bridge,
    run_ids,
    candidate_ids,
    candidate_text,
    context=f"{context}: trainable run {run_index}",
)

if partition_student_anchors:
    # 1: Student 生成、Teacher 模板提供的 reasoning prefix
    reasoning_prefix = student_tokenizer(
        "\n<think>", add_special_tokens=False
    )["input_ids"]
    if run_ids[:len(reasoning_prefix)] == reasoning_prefix:
        for offset in range(len(reasoning_prefix)):
            forced_anchor_reasons[start + offset] = 1

    # 2: Student-only 的工具格式结构换行
    omitted_offsets, _ = _student_tool_call_policy(
        student_tokenizer,
        scaffold,
        run_ids,
        normalize_tool_template=True,
        mask_pre_tool_content=False,
        context=f"{context}: anchor classification run {run_index}",
    )
    for offset in omitted_offsets:
        forced_anchor_reasons[start + offset] = 2

    # 3: observation stop boundary
    if last_is_boundary and run_ids[-1] == observation_id:
        forced_anchor_reasons[end - 1] = 3

    aligned = [
        pair
        for pair in aligned
        if start + pair[0] not in forced_anchor_reasons
    ]
```

完整轨迹完成后还会进行独立的官方模板校验：

```python
official_ids, _ = _render_template(
    teacher_tokenizer,
    teacher_messages,
    tools=scaffold.tools,
    add_generation_prompt=False,
    reasoning_effort=reasoning_effort,
)
if official_ids != teacher_ids:
    raise ValueError(
        "incremental canonical trajectory differs from official Teacher template"
    )
```

### 11.5 构建和加载 Token-ID bridge

```python
from dataclasses import dataclass
import hashlib
import numpy as np


@dataclass(frozen=True)
class TokenIdBridge:
    student_to_teacher: np.ndarray
    teacher_to_student: np.ndarray
    student_tokenizer_size: int
    student_model_vocab_size: int
    teacher_tokenizer_size: int
    teacher_model_vocab_size: int
    mapping_sha256: str

    def map_student(self, ids):
        ids = np.asarray(ids)
        if ids.size and (ids.min() < 0 or ids.max() >= self.student_model_vocab_size):
            raise ValueError("Student token ID outside bridge")
        mapped = self.student_to_teacher[ids]
        if np.any(mapped < 0):
            raise ValueError("unmapped/padded Student token ID")
        return np.ascontiguousarray(mapped, dtype=np.int64)

    def map_teacher(self, ids, *, allow_unmapped=False):
        ids = np.asarray(ids)
        if ids.size and (ids.min() < 0 or ids.max() >= self.teacher_model_vocab_size):
            raise ValueError("Teacher token ID outside bridge")
        mapped = self.teacher_to_student[ids]
        if not allow_unmapped and np.any(mapped < 0):
            raise ValueError("Teacher-only token ID")
        return np.ascontiguousarray(mapped, dtype=np.int64)


teacher_vocab = teacher.get_vocab()
mapping = np.full(student_model_vocab_size, -1, dtype=np.int32)
missing = []

for student_id in range(student_tokenizer_size):
    token = student.convert_ids_to_tokens(student_id)
    teacher_id = teacher_vocab.get(token)
    if teacher_id is None:
        missing.append((student_id, token))
    else:
        mapping[student_id] = int(teacher_id)

if missing:
    raise ValueError(f"teacher tokenizer is missing Student tokens: {missing[:16]}")
if np.unique(mapping[:student_tokenizer_size]).size != student_tokenizer_size:
    raise ValueError("student->teacher token mapping is not injective")

# GLM 共享基础词表必须是完全相同的 ID 前缀。
identity_prefix = 0
while (
    identity_prefix < student.vocab_size
    and mapping[identity_prefix] == identity_prefix
):
    identity_prefix += 1
if identity_prefix != student.vocab_size:
    raise ValueError("shared GLM base vocabulary is not identity-mapped")

mapping_sha256 = hashlib.sha256(
    np.ascontiguousarray(mapping, dtype="<i4").tobytes()
).hexdigest()
```

```python
mapped_valid = mapping[:student_tokenizer_size]
teacher_to_student = np.full(teacher_model_vocab_size, -1, dtype=np.int32)
teacher_to_student[mapped_valid] = np.arange(
    student_tokenizer_size, dtype=np.int32
)

return TokenIdBridge(
    student_to_teacher=mapping,
    teacher_to_student=teacher_to_student,
    student_tokenizer_size=student_tokenizer_size,
    student_model_vocab_size=student_model_vocab_size,
    teacher_tokenizer_size=teacher_tokenizer_size,
    teacher_model_vocab_size=teacher_model_vocab_size,
    mapping_sha256=mapping_sha256,
)
```

### 11.6 Phase 2：拆分 Teacher 行和 Student-anchor 行

```python
trainable_positions = np.flatnonzero(
    np.asarray(sample.loss_mask, dtype=bool)
).astype(np.int32)
teacher_positions = np.asarray(
    sample.opd_teacher_student_scored_positions, dtype=np.int32
)

teacher_rows = np.searchsorted(trainable_positions, teacher_positions)
is_teacher = np.zeros(trainable_positions.size, dtype=bool)
is_teacher[teacher_rows] = True
anchor_rows = np.flatnonzero(~is_teacher)
anchor_positions = trainable_positions[anchor_rows]

# Teacher 行只保留对应的 rollout 诊断支持。
sample.student_topk_ids = np.ascontiguousarray(ids[teacher_rows])
sample.student_topk_logprobs = np.ascontiguousarray(logprobs[teacher_rows])

# 其余行保存成 rollout Student anchor。
sample.student_anchor_positions = torch.from_numpy(anchor_positions.copy())
sample.student_anchor_topk_ids = torch.from_numpy(
    np.ascontiguousarray(ids[anchor_rows], dtype=np.int32)
)
sample.student_anchor_topk_logprobs = torch.from_numpy(
    np.ascontiguousarray(logprobs[anchor_rows], dtype=np.float16)
)
sample.student_anchor_reason = torch.from_numpy(staged_anchor_reasons.copy())
```

### 11.7 Phase 3：跳过 Teacher-only token 并顺延

```python
bridge = load_token_id_bridge(token_id_bridge_path)
reverse = torch.from_numpy(bridge.teacher_to_student)

teacher_ids = candidate_ids.index_select(0, cpu_rows).long()
teacher_lp = candidate_logprobs.index_select(0, cpu_rows).float()
head = teacher_head_ids.index_select(0, cpu_rows)[:, :support_k].long()

mapped = reverse[teacher_ids]
mapped_head = reverse[head]
valid = mapped.ge(0) & mapped.lt(student_valid_vocab_size)

if bool(valid.sum(dim=-1).lt(support_k).any()):
    raise ValueError(
        "Teacher cache cannot provide enough Student-representable tokens"
    )

# 保证 Teacher 原始 Top-K 中可映射的 token 必定保留；空位再从尾部顺延。
head_columns = torch.searchsorted(teacher_ids, head)
safe_head_columns = head_columns.clamp_max(teacher_ids.size(1) - 1)
valid_head = mapped_head.ge(0) & mapped_head.lt(student_valid_vocab_size)

bonus = torch.zeros_like(teacher_lp)
bonus.scatter_(
    1,
    safe_head_columns,
    valid_head.to(teacher_lp.dtype) * 1_000_000.0,
)
selection_score = (teacher_lp + bonus).masked_fill(~valid, float("-inf"))
columns = torch.topk(selection_score, support_k, dim=-1).indices

ids = torch.gather(mapped, 1, columns)
lp = torch.gather(teacher_lp, 1, columns)
order = torch.argsort(lp, dim=-1, descending=True, stable=True)
ids = torch.gather(ids, 1, order)
lp = torch.gather(lp, 1, order)
```

### 11.8 Phase 3：Student-anchor KL

```python
anchor_current_lp = vocab_parallel_logprobs_on_ids(
    anchor_logits,
    anchor_ids,
    tp_group,
    chunk_size=getattr(args, "log_probs_chunk_size", -1),
)

eps = torch.finfo(torch.float32).eps
old_prob = anchor_old_lp.exp()
old_mass_raw = old_prob.sum(dim=-1)

# fp16 落盘可能使 head mass 略大于 1，只修正这种数值异常。
scale = ((1.0 - eps) / old_mass_raw.clamp_min(eps)).clamp_max(1.0)
old_prob = old_prob * scale[:, None]
old_mass = old_prob.sum(dim=-1)

current_prob = anchor_current_lp.exp()
current_mass = current_prob.sum(dim=-1).clamp_max(1.0 - eps)
old_tail = (1.0 - old_mass).clamp_min(eps)
current_tail = (1.0 - current_mass).clamp_min(eps)

# KL(rollout Student || current Student)，含聚合后的 tail bucket。
anchor_rows = (
    old_prob
    * (old_prob.clamp_min(eps).log() - anchor_current_lp)
).sum(dim=-1)
anchor_rows += old_tail * (old_tail.log() - current_tail.log())
```

最终组合 loss：

```python
teacher_rows = (
    forward_coef * per_sample_forward_kl[sample_idx]
    + reverse_coef * per_sample_reverse_kl[sample_idx]
)
combined_rows.append(
    denominator_scale * (teacher_rows + anchor_coef * anchor_rows)
)
loss = sum_of_sample_mean(torch.cat(combined_rows))
```

这里的 `denominator_scale` 将归一化分母调整为：

```text
Teacher 行数 + anchor_coef * Student-anchor 行数
```

因此 Teacher 行和 anchor 行是互补分区，不会对同一个 response token 重复计算两种监督。

## 12. 完整数据契约

### 12.1 坐标定义

设 Student 完整序列为：

```text
[Student prompt: P_s tokens] [Student response: R_s tokens]
```

本文中所有持久化的 `*_positions` 都是 **Student response-relative position**，范围为 `[0, R_s)`。Phase 2 临时使用的 `teacher_scored_positions` 是 **Teacher response-relative position**，范围为 `[0, R_t)`。

位置 `r` 的分布表示：

```text
P(token_at_r | prompt + response_before_r)
```

若直接从 causal LM logits 实现，Student response 位置 `r` 使用完整序列 logits 的行 `P_s + r - 1`，Teacher 同理使用 `P_t + r_t - 1`。首个 response token 使用 prompt 最后一个位置的 logits。

### 12.2 Phase 1 输入样本

每条样本必须包含：

| 字段 | shape / 类型 | 语义 |
|---|---:|---|
| `tokens` | `[P_s + R_s] int` | Student prompt 与完整多轮轨迹 |
| `response_length` | scalar int | `R_s` |
| `loss_mask` | `[R_s] bool` | 需要 OPD 监督的位置 |
| `student_topk_ids` | `[N, K_s] int32` | rollout Student 在 trainable positions 上的 Top-K |
| `student_topk_logprobs` | `[N, K_s] fp16/fp32` | 上述 ID 的完整词表 log-prob |
| `metadata.problem_statement` | string | 重建初始 prompt |
| `metadata.workdir` | string | 默认 `/testbed` |
| tools/scaffold | structured object | 必须与 rollout 时完全一致 |

其中：

```text
trainable_positions = nonzero(loss_mask)
N = len(trainable_positions)
K_s = 128
```

`student_topk_ids[i]` 必须对应 `trainable_positions[i]`。Top-K 必须按 rollout Student 概率降序保存，log-prob 是完整 Student 词表 softmax 后的值，不能先在 Top-K 内重新归一化。

### 12.3 Phase 2 输出样本

令：

```text
T = 严格对齐的 Teacher positions 数
A = Student-anchor positions 数
K_c = 1000
K_s = 128
```

必须满足 `T + A = N`，输出字段为：

| 字段 | shape / 类型 | 排序 |
|---|---:|---|
| `teacher_candidate_positions` | `[T] int32` | Student response position 升序 |
| `teacher_candidate_ids` | `[T,K_c] int32` | 每行按 Teacher token ID 升序 |
| `teacher_candidate_logprobs` | `[T,K_c] fp16` | 与 candidate IDs 同步重排 |
| `teacher_top128_ids` | `[T,K_s] int32` | Teacher 概率降序；字段名不改变 |
| `teacher_cutoff_logprobs` | `[T] fp16` | Teacher 第 `K_c` 名的 log-prob |
| `teacher_top4096_mass` | `[T] fp32` | `sum(exp(Teacher Top-K_c logprob))`；字段名不改变 |
| `student_topk_ids` | `[T,K_s] int32` | Teacher 行对应的 rollout Student Top-K |
| `student_topk_logprobs` | `[T,K_s] fp16` | 对应 rollout Student log-prob |
| `student_anchor_positions` | `[A] int32` | Student response position 升序 |
| `student_anchor_topk_ids` | `[A,K_s] int32` | rollout Student Top-K |
| `student_anchor_topk_logprobs` | `[A,K_s] fp16` | 完整词表 log-prob |
| `student_anchor_reason` | `[A] int8` | 0/1/2/3，仅用于审计 |

`teacher_top128_ids` 和 `teacher_top4096_mass` 是历史字段名。复现时保留名字以兼容数据加载，但其实际宽度分别由 `K_s` 和 `K_c` 决定。

### 12.4 Bridge JSON

```json
{
  "format": "offline_opd_token_id_bridge_v1",
  "student_model": "/abs/path/to/GLM-4.5-Air",
  "teacher_model": "/abs/path/to/GLM-5.2",
  "student_tokenizer_size": 151365,
  "student_model_vocab_size": 151552,
  "teacher_tokenizer_size": 154856,
  "teacher_model_vocab_size": 154880,
  "identity_prefix_size": 151329,
  "student_tokenizer_sha256": "...",
  "teacher_tokenizer_sha256": "...",
  "mapping_sha256": "...",
  "student_to_teacher": [0, 1, 2, "...", -1]
}
```

具体 tokenizer 版本发生变化时，size 和 identity prefix 以实际 tokenizer 为准，不得为了通过检查硬编码旧值。`student_to_teacher` 长度等于 Student model vocab size；有效 tokenizer 范围必须全部可映射，padding 范围必须为 `-1`。

## 13. Phase 2 Teacher 打分与稀疏缓存

### 13.1 Teacher 输入

Teacher 必须接收完整 canonical Teacher 上下文：

```text
teacher_input_ids = teacher_prompt_ids + all_teacher_turn_ids
```

不能只发送当前 assistant turn，也不能用 Student 模板 token 作为 Teacher 上下文。只在严格对齐的 `teacher_scored_positions` 上提取 Teacher Top-1000，但所有前缀 token 都必须参与 attention。

等价的通用实现为：

```python
@torch.no_grad()
def score_teacher_rows(model, teacher_input_ids, teacher_prompt_len,
                       teacher_scored_positions, cache_k=1000):
    x = torch.tensor(teacher_input_ids, device=model.device)[None, :]
    logits = model(input_ids=x).logits[0]  # [sequence, teacher_vocab]
    absolute_logit_rows = torch.as_tensor(
        [teacher_prompt_len + r - 1 for r in teacher_scored_positions],
        device=logits.device,
    )
    selected_logits = logits.index_select(0, absolute_logit_rows)
    log_probs = selected_logits.float().log_softmax(dim=-1)
    top_lp, top_ids = torch.topk(log_probs, cache_k, dim=-1)
    return top_ids.cpu(), top_lp.cpu()
```

可以用支持 selective Top-K 的推理服务替代该实现。服务端优化只能减少 Top-K、CPU copy 和网络返回行数，不能裁掉完整轨迹的 attention context。

### 13.2 Teacher cache 压缩

Teacher 服务返回的是概率降序 Top-`K_c`。落盘前转换为：

```python
def compact_teacher_cache(ranked_ids, ranked_lp, student_positions,
                          support_k=128):
    # ranked_ids/ranked_lp: [T, K_c]，概率降序
    sorted_ids, order = torch.sort(ranked_ids.long(), dim=-1)
    sorted_lp = torch.gather(ranked_lp.float(), 1, order)
    return {
        "teacher_candidate_positions": student_positions.int().contiguous(),
        "teacher_candidate_ids": sorted_ids.int().contiguous(),
        "teacher_candidate_logprobs": sorted_lp.half().contiguous(),
        "teacher_top128_ids": ranked_ids[:, :support_k].int().contiguous(),
        "teacher_cutoff_logprobs": ranked_lp[:, -1].half().contiguous(),
        "teacher_top4096_mass": ranked_lp.exp().sum(-1).float().contiguous(),
    }
```

两种排序都需要保留：

- ID 升序 cache 用于 Phase 3 `searchsorted` 查当前 Student token；
- 概率降序 head 用于 forward KL、Top-M overlap 和 Teacher-only token 顺延。

## 14. Phase 3 双向 KL 的精确定义

### 14.1 Reverse KL：Current Student-owned Top-128

在 Teacher 对齐位置 `t`，从完整 Student logits 取当前模型真实 Top-128：

```text
C_t = TopK_128(p_current)
```

将每个 Student ID 通过 `student_to_teacher` 映射为 Teacher ID，在 Teacher Top-1000 cache 中查询：

```text
l_T(v) = cached Teacher log-prob,                    if cache hit
l_T(v) = Teacher 第 1000 名 token 的 log-prob,       if cache miss
```

设 `l_S(v)` 是 current Student 在完整 Student 词表上的 log-prob，并在 Student Top-128 内定义权重：

```text
q_t(v) = exp(l_S(v)) / sum_{u in C_t} exp(l_S(u))
```

则每行 reverse 项为：

```text
L_reverse(t) = sum_{v in C_t} q_t(v) * (l_S(v) - l_T(v))
```

代码：

```python
current_ids = exact_full_vocab_topk(student_logits, k=128)
current_lp = full_vocab_logprobs_on_ids(student_logits, current_ids)
teacher_query_ids = student_to_teacher[current_ids]
teacher_lp, hit = lookup_id_sorted_cache(
    teacher_query_ids,
    candidate_ids,
    candidate_logprobs,
    miss_value=teacher_cutoff_logprobs,
)
log_q = current_lp - torch.logsumexp(current_lp, dim=-1, keepdim=True)
reverse_rows = (log_q.exp() * (current_lp - teacher_lp)).sum(dim=-1)
```

注意：`q_t` 在 Student Top-128 内归一化，但括号内使用的是两模型完整词表 log-prob，不是将两边都重新归一化后的普通截断 KL。cache miss 的 cutoff 填充值使该项成为稀疏近似。

### 14.2 Forward KL：Teacher-owned、Student 可表达的 Top-128

从 Teacher 概率降序 Top-1000 中选择前 128 个能映射到 Student 词表的 token。Teacher-only token 跳过并继续向后取，得到：

```text
F_t = first 128 Student-representable Teacher candidates
r_t(v) = exp(l_T(v)) / sum_{u in F_t} exp(l_T(u))
```

每行 forward 项为：

```text
L_forward(t) = sum_{v in F_t} r_t(v) * (l_T(v) - l_S(v))
```

代码：

```python
forward_student_ids, forward_teacher_lp = select_first_representable(
    teacher_cache,
    teacher_to_student,
    k=128,
)
forward_student_lp = full_vocab_logprobs_on_ids(
    student_logits, forward_student_ids
)
teacher_log_q = forward_teacher_lp - torch.logsumexp(
    forward_teacher_lp, dim=-1, keepdim=True
)
forward_rows = (
    teacher_log_q.exp() * (forward_teacher_lp - forward_student_lp)
).sum(dim=-1)
```

### 14.3 Student-anchor KL

对无法严格对齐的位置，设 Phase 1 rollout Student Top-128 为 `A_t`：

```text
p_old(v) = exp(l_old(v)), v in A_t
p_cur(v) = exp(l_cur(v)), v in A_t
m_old = sum p_old(v)
m_cur = sum p_cur(v)
tail_old = 1 - m_old
tail_cur = 1 - m_cur
```

Student-anchor loss 为：

```text
L_anchor(t)
  = sum_{v in A_t} p_old(v) * (log p_old(v) - log p_cur(v))
  + tail_old * (log tail_old - log tail_cur)
```

这里不在 Top-128 内重新归一化，而是把 Top-128 外的完整概率质量聚合成一个 tail bucket。因此它是 rollout Student 到 current Student 的稀疏 forward KL 近似。

### 14.4 每条序列及全局归一化

对一条样本，设严格 Teacher 行集合为 `T_i`、anchor 行集合为 `A_i`，则：

```text
L_teacher(t) = 0.5 * L_forward(t) + 0.5 * L_reverse(t)

L_sequence(i)
  = [sum_{t in T_i} L_teacher(t)
     + lambda_anchor * sum_{t in A_i} L_anchor(t)]
    / [|T_i| + lambda_anchor * |A_i|]

lambda_anchor = 1.0
```

global batch loss 是每条有效序列 `L_sequence(i)` 的算术平均；不是把全局所有 token 混在一起平均。长轨迹不会仅因 token 更多而获得更大的样本权重。

## 15. 必须实现的校验和失败条件

### 15.1 模板与轨迹

- metadata 重建出的 Student prompt IDs 必须逐 token 等于 Phase 1 保存的 prompt；
- observation 必须用 sentinel 提取，再经 Student 模板重渲染并逐 token 等于保存的 connector；
- malformed tool call 不得被转换为有效结构化调用；
- 增量构造的完整 Teacher IDs 必须等于对完整 Teacher messages 一次性调用官方模板的结果；
- `reasoning_effort` 必须同时用于 prompt、每轮 delta 和最终完整校验。

### 15.2 位置与分区

- Student positions 和 Teacher positions 分别严格递增；
- 每个 Teacher position 必须位于 `loss_mask=True` 的 Student response token；
- `Teacher positions ∩ anchor positions = ∅`；
- `Teacher positions ∪ anchor positions = nonzero(loss_mask)`；
- N→1、1→N、重复跨度、空字符跨度和非 canonical sampled IDs 一律进入 anchor；
- anchor reason 只做审计，不得改变集合成员或 loss 权重。

### 15.3 词表

- bridge 的 tokenizer hash、mapping hash、vocab size 必须与当前模型一致；
- 所有有效 Student tokenizer IDs 必须可映射且映射 injective；
- Student padded IDs 不得出现在输入、support 或映射结果中；
- Teacher candidate 至少包含 128 个 Student 可表达 token，否则失败，不允许缩小 K；
- Teacher-only token 只在 forward support 选择时跳过，不能错误映射成 UNK。

### 15.4 数值与缓存

- 所有持久化 log-prob 必须有限；
- Teacher cache IDs 必须每行严格按 ID 非降序，head 必须按概率非升序；
- `0 < K_s <= K_c`；当前为 `128 <= 1000`；
- fp16 anchor head mass 若略大于 1，只允许按 `(1-eps)/mass` 向下缩放；
- current Student Top-128 必须来自 TP 分片完整词表的全局 Top-K，不能分别取每个 TP rank 的局部 Top-K；
- 计算 log-prob 时必须使用完整有效 Student 词表做 softmax，并排除 Megatron padding vocab。

## 16. 最小验收测试

实现完成后至少通过以下测试：

1. **相同 token**：两 tokenizer 对同一字符产生相同单 token，bridge 后 ID 相等，应进入 Teacher 集合。
2. **3→1 分词**：Student 三 token、Teacher 一 token，三个 Student 位置全部进入 anchor。
3. **1→N 分词**：Student 一 token、Teacher 多 token，该 Student 位置进入 anchor。
4. **模板提供 `\n<think>`**：Student rollout 中相应 token 的 reason 为 1，不进入 Teacher 集合。
5. **结构性工具换行**：reason 为 2；普通工具参数正文若严格对齐，仍进入 Teacher 集合。
6. **observation boundary**：reason 为 3。
7. **Teacher-only Top-K**：Teacher head 中插入多个 `teacher_to_student=-1` token，输出仍为 128 个不同的、概率最高的可映射 token。
8. **初始 anchor loss**：current Student 等于 rollout Student 时，`L_anchor` 应接近 0（允许 fp16 误差）。
9. **分区守恒**：每条样本 `T+A=N`，无重叠、无遗漏。
10. **模板守恒**：增量 Teacher trajectory 与完整官方模板渲染完全一致。
11. **标准 OPD 隔离**：关闭 opt-in 参数后，输入字段、loss mask 和标准 OPD loss 与改动前逐位一致。

## 17. 端到端复现顺序

```python
# 一次性准备
bridge = build_bridge(student_tokenizer, teacher_tokenizer)
assert_bridge_invariants(bridge)

# Phase 1：Student rollout
for sample in dataset:
    sample.tokens, sample.response_length, sample.loss_mask = rollout_with_student_template(sample)
    sample.student_topk_ids, sample.student_topk_logprobs = save_rollout_student_topk(
        sample, k=128, only_positions=nonzero(sample.loss_mask)
    )
    save_phase1(sample)

# Phase 2：Teacher canonical template + sparse score
for sample in phase1_samples:
    assert_rebuilt_student_prompt_equals_saved_prompt(sample)
    teacher_ids, teacher_prompt_len, pairs, anchor_reason = rebuild_canonical_teacher_history(
        sample,
        teacher_tokenizer,
        reasoning_effort="high",
    )
    # pairs: [(student_response_position, teacher_response_position), ...]
    teacher_positions = [teacher_pos for _, teacher_pos in pairs]
    student_positions = [student_pos for student_pos, _ in pairs]

    ranked_ids, ranked_lp = score_teacher_rows(
        teacher_model,
        teacher_ids,
        teacher_prompt_len,
        teacher_positions,
        cache_k=1000,
    )
    partition_teacher_and_anchor_rows(sample, student_positions, anchor_reason)
    attach_compacted_teacher_cache(sample, ranked_ids, ranked_lp, student_positions, support_k=128)
    validate_phase2_artifact(sample)
    save_phase2(sample)

# Phase 3：重新计算 current Student 分布并训练
for global_batch in shuffled_phase2_samples:
    logits = current_student(global_batch.tokens)
    for sample in global_batch:
        reverse = reverse_on_current_student_top128(sample, logits, bridge)
        forward = forward_on_teacher_representable_top128(sample, logits, bridge)
        anchor = rollout_student_anchor_kl(sample, logits)
        sample_loss = sequence_normalized_loss(
            forward,
            reverse,
            anchor,
            forward_coef=0.5,
            reverse_coef=0.5,
            anchor_coef=1.0,
        )
    loss = mean(sample_loss for sample in global_batch)
    loss.backward()
    optimizer.step()
```

完成上述数据契约、算法、校验和验收测试，即构成与本文方案等价的复现。
