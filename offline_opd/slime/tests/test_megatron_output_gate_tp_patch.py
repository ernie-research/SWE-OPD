"""CPU unit tests for slime's Megatron output-gate TP patch.

`megatron_output_gate_tp_patch` fixes gated attention (`--attention-output-gate`) when
`tensor_model_parallel_size > num_query_groups`: upstream narrows `query` to the
num_attention_heads/tp_size heads a rank owns but returns `gate` still covering the whole
query group (num_attention_heads/num_query_groups heads), so `_apply_output_gate`'s
`gate.view(*core_attn_out.shape)` raises. These tests pin the per-rank gate slice without a
real Megatron process group by driving the patch with a stand-in for the upstream method.
"""

import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0

# Qwen3.5-9B full-attention layer at TP8 -- the layout that crashed.
NUM_ATTENTION_HEADS = 16
NUM_QUERY_GROUPS = 4
HEAD_DIM = 256
TP_SIZE = 8
SEQ_LEN = 8
HEADS_PER_GROUP = NUM_ATTENTION_HEADS // NUM_QUERY_GROUPS  # 4: what upstream returns as gate
HEADS_PER_RANK = NUM_ATTENTION_HEADS // TP_SIZE  # 2: what upstream narrows query to

_PATCH_FILE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "slime/backends/megatron_utils/megatron_patch/megatron_output_gate_tp_patch.py"
)


def _load_patch_module():
    """Load the patch against stand-ins for the two Megatron symbols it imports.

    Stubs unconditionally: CI installs no megatron, and a partially working CUDA install
    makes importing the real one raise OSError rather than ImportError. Any pre-existing
    megatron modules are restored so the rest of the session is unaffected.
    """
    stubs = {name: types.ModuleType(name) for name in ("megatron", "megatron.core", "megatron.core.transformer")}
    for module in stubs.values():
        module.__path__ = []
    stubs["megatron.core.parallel_state"] = types.ModuleType("megatron.core.parallel_state")
    stubs["megatron.core.parallel_state"].get_tensor_model_parallel_rank = lambda: 0
    stubs["megatron.core.transformer.attention"] = types.ModuleType("megatron.core.transformer.attention")
    stubs["megatron.core.transformer.attention"].SelfAttention = type(
        "SelfAttention", (), {"get_query_key_value_tensors": lambda self: None}
    )

    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location("_output_gate_patch_under_test", _PATCH_FILE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
    return module


def _attention_stub():
    return SimpleNamespace(world_size=TP_SIZE, config=SimpleNamespace(num_query_groups=NUM_QUERY_GROUPS))


def _heads(tensor):
    """Global head ids carried by a [seq, batch, heads, head_dim] gate/query tensor."""
    return tensor[0, 0, :, 0].tolist()


def _upstream_with_unnarrowed_gate(rank):
    """Stand in for upstream: query narrowed to this rank's heads, gate left whole-group.

    Every element of gate holds its own global head id, so a wrong slice is visible.
    """

    def _call(_self, *_args, **_kwargs):
        group = rank // (TP_SIZE // NUM_QUERY_GROUPS)
        within_group = rank % (TP_SIZE // NUM_QUERY_GROUPS)
        gate = torch.arange(group * HEADS_PER_GROUP, (group + 1) * HEADS_PER_GROUP, dtype=torch.float32)
        gate = gate.view(1, 1, HEADS_PER_GROUP, 1).expand(SEQ_LEN, 1, HEADS_PER_GROUP, HEAD_DIM).contiguous()
        query = gate[:, :, within_group * HEADS_PER_RANK : (within_group + 1) * HEADS_PER_RANK, :].clone()
        return query, "key", "value", gate

    return _call


def test_gate_is_narrowed_to_the_query_heads_of_each_rank():
    module = _load_patch_module()
    attention = _attention_stub()
    seen = []

    for rank in range(TP_SIZE):
        module._orig_get_query_key_value_tensors = _upstream_with_unnarrowed_gate(rank)
        module.get_tensor_model_parallel_rank = lambda r=rank: r

        query, key, value, gate = module.get_query_key_value_tensors(attention, torch.zeros(SEQ_LEN, 1, 8))

        assert (key, value) == ("key", "value")
        assert gate.shape == query.shape, f"rank {rank}: gate {tuple(gate.shape)} vs query {tuple(query.shape)}"
        assert _heads(gate) == _heads(query), f"rank {rank}: gate gates the wrong heads"
        # The view that used to raise in Megatron's _apply_output_gate.
        gate.contiguous().view(SEQ_LEN, 1, HEADS_PER_RANK * HEAD_DIM)
        seen += _heads(gate)

    assert sorted(seen) == list(range(NUM_ATTENTION_HEADS)), "ranks must cover every head exactly once"


def test_gate_already_matching_query_is_untouched():
    """num_query_groups >= tp_size, or a fixed upstream: nothing to narrow."""
    module = _load_patch_module()
    gate = torch.zeros(SEQ_LEN, 1, HEADS_PER_RANK, HEAD_DIM)
    module._orig_get_query_key_value_tensors = lambda *a, **kw: (gate.clone(), "key", "value", gate)
    module.get_tensor_model_parallel_rank = lambda: 0

    assert module.get_query_key_value_tensors(_attention_stub(), torch.zeros(SEQ_LEN, 1, 8))[3] is gate


def test_ungated_returns_are_passed_through():
    module = _load_patch_module()
    module.get_tensor_model_parallel_rank = lambda: 0

    module._orig_get_query_key_value_tensors = lambda *a, **kw: ("query", "key", "value")
    assert module.get_query_key_value_tensors(_attention_stub(), None) == ("query", "key", "value")

    # split_qkv=False returns (mixed_qkv, split_arg_list).
    module._orig_get_query_key_value_tensors = lambda *a, **kw: ("mixed_qkv", [1, 2, 3])
    assert module.get_query_key_value_tensors(_attention_stub(), None) == ("mixed_qkv", [1, 2, 3])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
