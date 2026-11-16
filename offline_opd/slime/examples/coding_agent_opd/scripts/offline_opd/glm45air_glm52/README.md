# GLM-4.5-Air student / GLM-5.2 teacher offline OPD

This recipe is isolated from the existing Qwen offline OPD experiments. It uses:

- Student/rollout/training: `GLM-4.5-Air`
- Teacher: `GLM-5.2-FP8`
- Dynamic student support: current Top-128
- Fixed teacher cache: raw teacher Top-1000
- Cache miss: teacher's 1000th log-probability (cutoff upper estimate)

The two tokenizers share the first 151329 ordinary tokens but assign different
ids to 36 special tokens. `build_token_id_bridge.py` creates a verified,
injective student-to-teacher mapping. Student ids 151365--151551 are model
padding and are excluded from rollout snapshots, sampled-token validation, and
training-time current Top-K.

Phase 2 maps the complete trajectory into teacher id space before the teacher
forward. Teacher candidate ids stay in teacher space on disk. Phase 3 computes
the true current student Top-128 in student space, maps only those ids for the
cache lookup, and keeps student log-probabilities/gradients in student space.

Default topology:

- Phase 1: eight TP4 rollout engines over four nodes, concurrency 100.
- Phase 2: two independent TP16 teacher replicas, each spanning two nodes.
- Phase 3: TP8 x PP2 x CP2 over 32 GPUs (DP1), GBS64, two replay epochs.

`00-prepare-student.sh` converts the student HF model to a reshardable
`torch_dist` checkpoint. Run the complete pipeline with:

```bash
bash examples/coding_agent_opd/scripts/offline_opd/glm45air_glm52/04-run-all.sh
```

The run is created under `/path/to/offline_opd`; `pipeline.log`
and `logs-phase{1,2,3}/` contain launcher and per-rank logs.

## Native Teacher reasoning prefix

The GLM recipe enables `OPD_TEACHER_NATIVE_REASONING_PREFIX=1` in Phase 2. This
is an isolated, opt-in artifact mode; the generic Phase-2 default remains off.

- GLM-5.2 keeps the `<think>` inserted by its own generation prompt.
- Each GLM-4.5-Air assistant run must begin token-exactly with `\n<think>`.
- Those two Student-only structural tokens are not sent as scoring rows and are
  changed to `loss_mask=0` in the scored artifact.
- The corresponding rows are removed from `student_topk_ids` and
  `student_topk_logprobs`; all response-aligned fields keep their original length.
- Metadata records schema version 1 and the number of skipped tokens.

The mutation happens only after a successful Teacher response, so HTTP retries
always rebuild from the untouched Phase-1 sample. Phase 3 consumes the normal
dynamic sparse schema and needs no special loss path. Artifacts produced by the
old replay mode must not be mixed with this mode; re-run Phase 2 into a new
output directory.

## Other GLM-4.5-Air / GLM-5.2 template differences

The local official tokenizer templates also differ in the following ways. They
are intentional native-Teacher context differences and do not create a sparse
row offset:

- GLM-5.2 prepends a synthetic `<|system|>Reasoning Effort: Max` section.
- GLM-5.2 unwraps OpenAI-style `{"type":"function","function":...}` tool
  schemas, filters `strict` / `defer_loading`, and emits a more compact tool-call
  instruction. GLM-4.5-Air renders the wrapper object directly.
- Role separators and observation connectors differ in whitespace. For example,
  GLM-4.5-Air renders `<|observation|>\n<tool_response>\n...`, while GLM-5.2
  renders `<|observation|><tool_response>...`.
- Historical assistant `<think>` normalization and whitespace stripping differ.
  The trajectory rebuilder does not re-render generated assistant content: it
  maps those tokens verbatim and re-renders only masked connectors. Therefore
  this difference does not reorder scoring rows.
- GLM-4.5-Air typically emits a newline before `<tool_call>` whereas GLM-5.2's
  template formats it without that newline. Generated Student tool-call tokens
  are deliberately replayed verbatim so every retained training token still has
  one well-defined Teacher distribution; this remains a genuine cross-policy
  context difference, not a mapping bug.

With the current fn-calling scaffold, a measured initial prompt was 2939 Student
tokens versus 2911 Teacher tokens. This length difference is expected because
each model uses its own native system/tool template. A real-data smoke check on
12 Phase-1 trajectories rebuilt 485 assistant turns and strictly recognized all
970 `\n<think>` structural tokens.

## Fully canonical GLM-5.2 Teacher template

Status: implemented as an isolated, opt-in Phase-2 artifact mode. Enable it with
`OPD_TEACHER_CANONICAL_TEMPLATE=1`; it is mutually exclusive with
`OPD_TEACHER_NATIVE_REASONING_PREFIX=1`. The generic offline OPD path and the
older GLM native-prefix mode remain unchanged.

The canonical mode makes the complete Teacher input token-exactly equal to the
official GLM-5.2 `apply_chat_template` result while retaining Student-generated
semantic content:

1. Parse every Student assistant turn into reasoning, visible content and
   structured tool calls.
2. Render the complete multi-turn Teacher trajectory with the GLM-5.2 tokenizer
   and its native system, tools, reasoning and observation formatting.
3. Build an explicit monotonic mapping from Student response positions to
   Teacher response positions. Never infer alignment from a constant offset and
   never fill a mismatch by shifting to the next token.
4. Retain a training position only when it has a one-to-one mapping, its decoded
   byte span is identical, and the Teacher token id equals the verified
   Student-to-Teacher bridge result.
5. Keep Teacher-only template tokens as attention context without assigning them
   a Student loss position. Set `loss_mask=0` for Student-only formatting tokens
   and for every ambiguous one-to-many or many-to-one tokenization span.
6. Re-anchor alignment at each structural boundary so one local mismatch cannot
   shift later tokens.
7. Continue to score the canonical trajectory in one Teacher forward. Persist
   only the explicitly aligned sparse rows and use a new artifact schema/flag;
   old Phase-2 artifacts are not reusable.

Required fail-closed validation and diagnostics:

- Assert the final Teacher token sequence exactly equals an independent official
  `apply_chat_template` render.
- Assert both position arrays are strictly increasing and every retained pair
  passes token-id and byte-span equality checks.
- Record total/aligned/unaligned semantic tokens, Student structural tokens
  masked, Teacher-only context tokens and semantic alignment rate.
- Abort Phase 2 instead of silently writing an artifact when parsing is
  incomplete, alignment is ambiguous, or coverage falls below the configured
  threshold (`OPD_TEACHER_MIN_ALIGNMENT_RATE`, default `0.90`).

The scored artifact uses schema flag
`opd_teacher_canonical_template_schema=1`. It stores only aligned sparse rows;
Teacher-only template tokens remain attention context, while unaligned Student
positions are masked before Phase 3. Old/native-prefix Phase-2 artifacts cannot
be resumed as canonical artifacts.

To reuse the completed GLM Phase-1 rollout and run a fresh canonical Phase 2 and
Phase 3 in a new directory:

```bash
SOURCE_RUN=/path/to/offline_opd/glm45air-glm52-dynamic-t1k-s128-20260819_001719 \
  bash examples/coding_agent_opd/scripts/offline_opd/glm45air_glm52/05-run-canonical-from-phase2.sh
```

Tokenization audit on 12 real Phase-1 trajectories (485 assistant turns):

- 465/485 turns were already token-identical when the same assistant text was
  encoded by both tokenizers.
- 137339/137418 Student tokens were one-to-one aligned: 99.9425%.
- 79 Student tokens (0.0575%) belonged to non-one-to-one spans. Observed cases
  include GLM-5.2 representing `🎉` or `🐛` with one new token where GLM-4.5-Air
  uses three tokens, and representing `ubuntu` with one token where the Student
  uses two.

Those non-one-to-one positions have no unique token-level Teacher distribution
under the current OPD loss and must be masked, not approximated. This produces a
slightly smaller training set rather than a shifted or corrupted alignment.

The end-to-end canonical rebuild additionally includes the template-only token
differences. On the same real Phase-1 data, observed retained trainable-token
coverage was approximately 95.3%--97.9%; every rebuilt Teacher sequence passed
an independent full-history official-template token equality assertion.

## Teacher template with verbatim Student tokens

For reusing an existing GLM-4.5-Air Phase-1 rollout without re-tokenizing its
assistant actions, enable both
`OPD_TEACHER_NATIVE_REASONING_PREFIX=1` and
`OPD_TEACHER_MASK_STUDENT_FORMAT_TOKENS=1`. This isolated mode uses:

- the official GLM-5.2 system/tools/user prompt, generation-time `<think>`
  prefix, observation connector, and following assistant prefix;
- the original Student token ids, translated one-to-one through the verified
  vocabulary bridge, for reasoning, code, tool names, keys, and values;
- loss masking for the Student-only `\n<think>` prefix and Student format tags
  (role, think, tool-call/tool-response, and argument tags).

Format tokens remain in the Teacher attention context when they are part of the
actual Student action, but receive no cached Teacher row and no Phase-3 loss.
No assistant text is re-tokenized, so a Unicode character represented by three
Student tokens remains three Teacher-conditioning/scoring tokens even when the
GLM-5.2 tokenizer would canonically encode it as one new token. Missing Student
`\n<think>` prefixes are accepted because the Teacher generation prompt already
supplies its native `<think>`.

Run from the completed Phase-1 data with:

```bash
SOURCE_RUN=/path/to/offline_opd/glm45air-glm52-dynamic-t1k-s128-20260819_001719 \
  bash examples/coding_agent_opd/scripts/offline_opd/glm45air_glm52/06-run-teacher-template-student-tokens-from-phase2.sh
```

Artifacts carry `opd_teacher_template_student_tokens_schema=1` and are not
resume-compatible with either native-prefix-only or fully canonical artifacts.
