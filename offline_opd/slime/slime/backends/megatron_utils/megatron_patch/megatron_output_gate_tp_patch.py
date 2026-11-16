# Patch megatron.core.transformer.attention.SelfAttention.get_query_key_value_tensors so
# gated attention (--attention-output-gate) survives tensor_model_parallel_size >
# num_query_groups.
#
# Upstream (megatron/core/transformer/attention.py, dev @1dcf0dafa and the core_v0.13.0
# line): when num_query_groups < tp_size every rank all-gathers the qkv projection and
# keeps ONE query group, i.e. num_attention_heads/num_query_groups q heads. `query` is
# then narrowed to the num_attention_heads/tp_size heads the rank actually owns
# ("step 4" at attention.py:1470-1480), but `gate`, split out of the very same group, is
# only reshaped and returned un-narrowed. core_attention therefore runs on the narrowed
# query while _apply_output_gate does gate.view(*core_attn_out.shape) and dies:
#   Qwen3.5-9B (num_attention_heads 16, num_query_groups 4, kv_channels 256) at TP8
#   -> core_attn_out [T, 1, 2*256], gate [T, 1, 4, 256]
#   -> RuntimeError: shape '[T, 1, 512]' is invalid for input of size T*1024
# gate head i gates query head i, so the gate needs exactly the slice query got --
# same rank-within-group index, same head count.
#
# No-op whenever the gate already lines up with the query (num_query_groups >= tp_size,
# ungated attention, or a fixed upstream), so it is safe to apply unconditionally.

import logging
import warnings

logger = logging.getLogger(__name__)

try:
    # Same rank source as the query slice this mirrors (attention.py:1474), so gate and
    # query cannot disagree about which heads belong to this rank.
    from megatron.core.parallel_state import get_tensor_model_parallel_rank
    from megatron.core.transformer.attention import SelfAttention

    _orig_get_query_key_value_tensors = SelfAttention.get_query_key_value_tensors

    def get_query_key_value_tensors(self, *args, **kwargs):
        out = _orig_get_query_key_value_tensors(self, *args, **kwargs)
        # (query, key, value, gate) is the only 4-tuple; ungated returns 3 and
        # split_qkv=False returns (mixed_qkv, split_arg_list).
        if len(out) != 4:
            return out

        query, key, value, gate = out
        num_heads = query.shape[2]
        if gate.shape[2] == num_heads:
            return query, key, value, gate

        # Reached only on the num_query_groups < tp_size path, where the rank holds one
        # query group and owns the idx-th chunk of num_heads heads inside it.
        idx = get_tensor_model_parallel_rank() % (self.world_size // self.config.num_query_groups)
        return query, key, value, gate[:, :, idx * num_heads : (idx + 1) * num_heads, :]

    SelfAttention.get_query_key_value_tensors = get_query_key_value_tensors

    logger.info(
        "slime output-gate TP patch applied to "
        "megatron.core.transformer.attention.SelfAttention.get_query_key_value_tensors"
    )

except ImportError as exc:
    warnings.warn(
        f"slime output-gate TP patch not applied — Megatron import failed ({exc!r}). "
        "If this is a Megatron upgrade, the symbol layout may have changed; without this "
        "patch, --attention-output-gate models crash in _apply_output_gate whenever "
        "tensor_model_parallel_size > num_query_groups.",
        stacklevel=2,
    )
