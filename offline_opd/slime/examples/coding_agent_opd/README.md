# coding_agent_opd — model-in-the-loop SWE + On-Policy Distillation

The **student model itself** drives an explicit sweagent 3-tool loop against an
**RLE sandbox**. Each trajectory is scored by **SWE test pass/fail (scalar
reward) + a reverse-KL penalty toward a teacher model's token logprobs (OPD)**.

This differs from `examples/coding_agent_rl` (which runs Claude Code / Codex as a
black-box CLI inside the sandbox): here the trained model emits tool calls
directly, and its own tokens + logprobs are trained on.

## How it works

```
generate() [examples/coding_agent_opd/generate.py]   -> a SINGLE Sample
  boot AsyncRepoEnv(image, ...) — scaffold tools installed via RepoEnv.add_commands
  prepare workspace (problem statement, git)
  tool loop (manual token concat):
    POST sglang student /generate (return_logprob)  -> student tokens (trainable=1)
    parse_action -> to_bashcmd -> env.step          -> observation (trainable=0)
    stop on submit / max turns / context budget
  get_patch, then grade on a FRESH RLE pod (AsyncRepoEnv.compute_reward)
  training (sglang):   reward=None + metadata["swe_reward"]=r  (async_rm fetches teacher)
  training (megatron): reward=r     + metadata["swe_reward"]=r  (async_rm skipped)
  eval: reward=r

reward_func (reused from slime.rollout.on_policy_distillation) — sglang mode only
  POST the full token sequence to the sglang TEACHER (--rm-url), return logprobs

opd.post_process_rewards
  scalar reward = metadata["swe_reward"]   (NOT hardcoded 0.0)
  sglang mode: also set sample.teacher_log_probs (reverse-KL penalty in the loss)
  megatron mode: teacher_log_probs come from the in-backend forward (actor.py)
```

The 3 tools exposed to the model: `str_replace_editor` (view/create/str_replace/
insert), `execute_bash`, `submit`. Protocol in `r2egym/action.py`.

## Teacher mode

Two teacher modes are supported, selected by `--opd-type`:

- **`sglang`** (external teacher server): the teacher runs as a separate sglang
  server. `reward_func` POSTs the token sequence to it per sample and
  `post_process_rewards` extracts `input_token_logprobs` into
  `sample.teacher_log_probs`. The reward and teacher logprobs both ride the reward
  hook.
- **`megatron`** (in-backend teacher): the teacher is loaded into the training
  backend via `--opd-teacher-load` and forwarded during training, filling
  `rollout_data["teacher_log_probs"]` directly (`actor.py`). **No external server,
  no GPU7, no `--rm-url`, no teacher fetch in the reward hook.

The reverse-KL penalty (`apply_opd_kl_to_advantages`) reads the single key
`rollout_data["teacher_log_probs"]` in **both** modes; in megatron mode
`sample.teacher_log_probs` is never consumed.

### Megatron teacher variant

In megatron mode the teacher weights are restored **in-place** into the live
student model for the forward pass, so the teacher **must have the same
architecture** as the student. `--opd-teacher-load` must point at a valid
megatron dist checkpoint (has `latest_checkpointed_iteration.txt`);
`--opd-type sglang` forbids `--opd-teacher-load`.

`OPD_TEACHER_LOAD` (env) selects the teacher checkpoint and **defaults to the ref
torch_dist** so the smoke runs out-of-box as a self-distillation demo. For a real
run, override it with a **stronger same-architecture** checkpoint (e.g. convert
your stronger HF checkpoint via `slime/tools/convert_hf_to_torch_dist.py`, or point at
your own megatron dist dir).

The `generate.py` reward path is `opd_type`-aware: in sglang mode it leaves
`sample.reward = None` (so `async_rm → reward_func` fetches the teacher), while in
megatron/no-OPD mode it sets the SWE scalar directly (so `async_rm` is skipped).

Because the teacher is loaded alongside the student, the megatron variant needs
**extra actor-GPU memory** for the teacher weight backup + forward.

Some notes
`scripts/offline_opd/`（ `PLAN.md`， `OFFLINE_OPD_FORMULAS.md`）。

## Sandbox backend

RLE only. Any other `SLIME_AGENT_SANDBOX_BACKEND` raises `NotImplementedError`.

## Prerequisites

- A teacher checkpoint servable by sglang (the launcher starts it for you).
- RLE credentials (`RLE_ENV`, `RLE_TOOL_TYPE`, `RLE_API_KEY`, `RLE_SESSION_IDS_DIR`).
- An SWE dataset in slime row schema `{prompt, label, metadata:{...}}` where
  `metadata` carries `image` / `workdir` / `problem_statement` / `instance_id`
  plus grading fields (see `examples/coding_agent_rl/swe.py::get_metadata` for
  the exact scaleswe / swebench field maps).
