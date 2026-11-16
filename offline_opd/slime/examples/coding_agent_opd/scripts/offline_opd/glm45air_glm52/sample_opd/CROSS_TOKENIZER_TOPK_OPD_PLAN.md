# GLM-4.5-Air / GLM-5.2 Cross-Tokenizer Top-K OPD Plan

## Implementation status (2026-08-26)

Implemented as the isolated loss/artifact type:

```text
cached_teacher_bidirectional_topk_kl_with_student_anchor
```

The implementation does not alter the behavior of the existing standard OPD,
dynamic reverse-KL, cached forward-KL, or cached bidirectional-KL modes.  The
end-to-end launcher is:

```text
examples/coding_agent_opd/scripts/offline_opd/glm45air_glm52/
14-run-canonical-anchor-bidirectional-from-phase1.sh
```

The Phase-3 anchor coefficient is exposed as
`--opd-student-anchor-kl-coef` / `OPD_STUDENT_ANCHOR_KL_COEF` and defaults to
1.0. Old dynamic artifacts are rejected by the new loss because they do not
carry the explicit anchor schema marker and complete position partition.

## Goal

Adapt the Simple-OPD fallback principle to the existing offline top-k,
bidirectional-KL pipeline:

- Teacher-compatible positions use the cached Teacher distribution.
- Positions that cannot be aligned reliably use the rollout Student
  distribution as an anchor.
- Environment and padding positions remain excluded from training.

This must be implemented as an isolated artifact schema and loss type so that
existing standard OPD and offline OPD experiments remain unchanged.

## Why the Simple-OPD code cannot be used directly

`teacher_student_align.py` provides two useful ideas:

1. Only use a Teacher log-prob when the Teacher and Student tokens have an
   exact 1-to-1 text alignment.
2. Fall back to the rollout Student log-prob for unmatched positions.

Its request path is not suitable for the coding-agent experiment, however:

- `teacher_prompt + sample.response` duplicates the thinking prefix for this
  model pair: the GLM-5.2 prompt already ends in `<think>`, while the
  GLM-4.5-Air response begins with `\n<think>`.
- `sample.response` contains generated assistant text but not the complete
  interleaved tool observations and assistant connectors. Later turns would
  therefore be scored under the wrong history.
- It aligns only the sampled token's scalar log-prob. The current experiment
  consumes Teacher top-1000 and Student/current top-128 distributions for
  bidirectional KL.

The new path should reuse the fallback principle, not the single-turn request
builder or scalar-log-prob representation.

## Canonical Teacher trajectory

Phase 2 should continue to use the existing canonical trajectory machinery:

1. Recover each Student assistant turn and its real tool observation.
2. Render the complete history with the official GLM-5.2 chat template.
3. Tokenize that history with the GLM-5.2 tokenizer.
4. Align each Student action token with the Teacher sequence using exact token
   offsets and UTF-8 byte spans.
5. Verify that the incrementally assembled Teacher sequence is identical to a
   full official-template render.

`_rebuild_canonical_teacher_trajectory()` and `_align_canonical_turn()` in
`teacher_trajectory.py` already implement most of this. The latter is stricter
than comparing `convert_ids_to_tokens()` strings because it checks canonical
offsets, byte equality, token-id mapping and monotonicity.

## Position ownership

Every original Student-trainable response position must belong to exactly one
of the following sets.

### Teacher KL positions

Use the existing 0.5 forward-KL plus 0.5 reverse-KL objective for:

- reasoning text;
- `</think>`;
- ordinary content after `</think>`;
- `<tool_call>` and `</tool_call>`;
- tool names;
- `<arg_key>`, `</arg_key>`, `<arg_value>` and `</arg_value>`;
- argument names and values;
- semantic newlines inside reasoning, content or argument values;
- other tokens with an exact one-Student-token to one-Teacher-token alignment.

The GLM special tokens above have different numeric IDs but exact vocabulary
bridge mappings. They must not be masked solely because they are special
tokens.

### Rollout Student anchor positions

Use the frozen Phase-1 Student top-128 distribution for:

- the Student-generated `\n<think>` prefix at the start of every assistant
  turn; GLM-5.2 receives `<think>` from its generation template;
- Student-only structural newlines after `</think>`, before visible content,
  before `<tool_call>`, and between tool-call fields outside `arg_value`;
- the Student-generated `<|observation|>` stop boundary, which the Teacher
  receives from its template/agent connector;
- the analogous generated role boundary in a non-function-calling scaffold;
- all Student tokens in a tokenizer 1-to-N or N-to-1 span;
- malformed or uncertain spans for which exact alignment cannot be proven.

Structural newlines must be identified from parsed positions. Globally
matching the token string `"\n"` would incorrectly anchor semantic newlines in
reasoning, code, content and argument values.

For an N-to-1 or 1-to-N tokenizer span, do not copy, average or duplicate a
Teacher token probability across Student positions. Anchor every Student token
in that span, then resume Teacher KL after both tokenizers reach the same UTF-8
boundary.

### Ignored positions

Keep `loss_mask=0` for tokens that are not Student actions:

- system and user prompt tokens;
- template-injected `<|assistant|>` connectors;
- tool-response markers and tool output;
- padding and truncated regions.

## Anchor objective

Phase 1 already stores `student_topk_ids` and `student_topk_logprobs` for every
original trainable position. For an anchor support `S`, use a sparse forward KL
with a collapsed tail bucket:

```text
p_old_tail = 1 - sum(v in S) p_old(v)
p_cur_tail = 1 - sum(v in S) p_cur(v)

L_anchor = sum(v in S) p_old(v) * (log p_old(v) - log p_cur(v))
         + p_old_tail * (log p_old_tail - log p_cur_tail)
```

The tail term preserves top-128 probability mass as well as the conditional
distribution inside the cached support. A normalized top-128-only KL does not
preserve this mass, while sampled-token replacement preserves only one token.

Combine the two position groups without allowing the small structural group to
disappear accidentally:

```text
L = (sum(t in T) L_teacher[t]
     + lambda_anchor * sum(t in A) L_anchor[t])
    / (|T| + lambda_anchor * |A|)
```

Start with `lambda_anchor = 1.0` and expose it as an experiment-local option.

## Sparse artifact schema

Keep the existing Teacher cache fields for Teacher-owned rows and add a
separate sparse anchor block:

```text
teacher_candidate_positions
teacher_candidate_ids
teacher_candidate_logprobs
teacher_top128_ids
teacher_cutoff_logprobs
teacher_topk_mass

student_anchor_positions
student_anchor_topk_ids
student_anchor_topk_logprobs
student_anchor_reason
```

Do not expand either block to `[response_length, K]`. Validate that:

```text
teacher_positions intersect anchor_positions = empty
teacher_positions union anchor_positions = original Student-trainable positions
```

Environment positions are outside both sets.

Use a new artifact schema and a new loss type. Old artifacts must fail schema
validation instead of being silently interpreted under the new rules.

## Candidate-vocabulary mismatch

Trajectory-position alignment and distribution-support mapping are separate
problems:

- A Student/Teacher N-to-M trajectory span selects the Student anchor path.
- At a valid Teacher row, Teacher top-k candidates absent from the Student
  vocabulary are skipped and replaced by lower-ranked representable Teacher
  candidates, as in the current forward-KL implementation.
- Current-Student candidates mapped into the Teacher cache continue to use the
  configured cache-miss policy.

Do not mask an entire valid row merely because one Teacher candidate is absent
from the Student vocabulary.

## Phase changes

### Phase 1

No rerollout is required. Existing Phase-1 artifacts already contain Student
top-128 IDs and log-probs for all original trainable positions.

### Phase 2

Phase 2 must be rerun because previous scored artifacts compacted away the
format and unmatched rows. It should:

1. build the canonical Teacher trajectory;
2. classify every Student-trainable position;
3. request/store Teacher top-1000 only for Teacher positions;
4. copy Phase-1 Student top-128 rows into the sparse anchor block;
5. record per-reason row counts for auditing.

### Phase 3

Phase 3 should:

1. run the existing bidirectional top-k KL on Teacher positions;
2. run sparse Student anchor KL on anchor positions;
3. combine both with the configured anchor coefficient;
4. keep all existing standard/offline OPD paths unchanged.

## Required metrics

Log at least:

- `offline_opd/teacher_position_count`;
- `offline_opd/student_anchor_position_count`;
- anchor counts split by prefix, structural whitespace, boundary and tokenizer
  mismatch;
- `offline_opd/student_anchor_kl`;
- rollout/current top-128 mass on anchor positions;
- `<think>`, `</think>`, `<tool_call>` and `<|observation|>` sampled-token
  probabilities or log-prob deltas;
- generated thinking-open rate, thinking-close rate and non-empty-content rate
  in validation rollouts.

Implemented training metrics are:

```text
offline_opd/student_anchor_kl
offline_opd/student_anchor_kl_weighted
offline_opd/student_anchor_rollout_topk_mass
offline_opd/student_anchor_current_mass_on_rollout_topk
offline_opd/student_anchor_row_fraction
offline_opd/student_anchor_tokenizer_or_template_mismatch_fraction
offline_opd/student_anchor_reasoning_prefix_fraction
offline_opd/student_anchor_structural_newline_fraction
offline_opd/student_anchor_observation_boundary_fraction
```

Exact per-reason row counts are persisted in
`metadata.offline_opd_cross_tokenizer_anchor_alignment`; the W&B fractions use
the original Student-trainable token count as denominator.

Teacher-KL metrics must be reduced only over Teacher positions; anchor metrics
must be reduced only over anchor positions.

## Verification

Before a full run:

1. Test synthetic 1-to-1, 1-to-N and N-to-1 tokenizer spans.
2. Audit real trajectories and assert exact partitioning of every original
   trainable position.
3. Assert that `</think>` and tool/argument tags are Teacher positions.
4. Assert that `\n<think>`, Student-only structural newlines and generated
   `<|observation|>` are anchor positions.
5. Assert that newlines inside reasoning, content and argument values remain
   Teacher positions.
6. Verify the complete Teacher input against an independent official-template
   render.
7. Run one Phase-3 step and confirm anchor KL is near zero at initialization.
8. Run a short checkpoint comparison and verify thinking-open, thinking-close,
   content and tool-call rates before launching the full experiment.

The CPU unit/regression suite covers artifact partitioning, schema rejection,
sparse+tail KL and gradients. A real Phase-1 GLM sample was also audited:
8,471 original trainable positions = 8,201 Teacher positions + 270 Student
anchors (78 generic exact-alignment misses, 52 reasoning-prefix tokens, 114
structural newlines and 26 observation boundaries).
