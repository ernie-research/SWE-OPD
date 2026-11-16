from __future__ import annotations

import dataclasses
import os
from enum import Enum, auto
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.eagle_info import EagleVerifyOutput
    from sglang.srt.speculative.ngram_info import NgramVerifyInput


# OPD diagnostic: append |TopK_teacher ∩ ids| at each scored position as ONE EXTRA
# TRAILING COLUMN of the per-position token_ids_logprob rows -- i.e. how many of the ids
# the client asked for are also in the teacher's OWN top-K there (the id rows get a -1 in
# that slot so value/index widths stay equal). Lets the client get the top-k overlap
# without shipping the teacher's own [P, K] top-k back. Off by default: it costs a
# full-vocab topk over the scored window, which this path does not otherwise pay.
# Server-level (like SGLANG_TOP_LOGPROBS_IDS_B64) because the teacher is a dedicated
# process and sgl-router drops per-request flags.
_OPD_TOPK_OVERLAP = os.environ.get("SGLANG_OPD_TOPK_OVERLAP", "0").lower() in (
    "1",
    "true",
)
# Dedicated offline-OPD teacher fast path.  A dynamic teacher-cache request also
# carries a compact PerPositionTokenIds object whose ``positions`` are exactly the
# trainable response rows.  When enabled, compute the teacher's own Top-K only on
# those rows.  Full hidden-state attention still sees the complete trajectory; this
# only skips the vocab projection/topk post-processing for masked observation rows.
# Off by default so the regular SGLang and standard OPD response schemas are unchanged.
_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS = os.environ.get(
    "SGLANG_TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS", "0"
).lower() in ("1", "true")
# Placeholder id for the overlap-count column, so a value row and its id row stay the
# same width for the JSON (detokenize) path.
_OVERLAP_ID = -1


class LogprobStage(Enum):
    PREFILL = auto()
    DECODE = auto()


@dataclasses.dataclass
class InputLogprobsResult:
    input_token_logprobs: torch.Tensor
    input_top_logprobs_val: Optional[List] = None
    input_top_logprobs_idx: Optional[List] = None
    input_token_ids_logprobs_val: Optional[List] = None
    input_token_ids_logprobs_idx: Optional[List] = None


@dataclasses.dataclass
class PerPositionTokenIds:
    """Sparse per-position ``token_ids_logprob`` ids, kept as one ``[N, K]`` array.

    ``rows[i]`` is the id-list for scored-window position ``positions[i]`` (strictly
    increasing); window positions with no row are not scored. Stands in for the
    ``[window][K]`` nested Python list the JSON/broadcast paths index row by row: at
    K=1024 that list is tens of millions of Python ints per request, built inside the
    TokenizerManager event loop and pickled element-wise over ZMQ to the scheduler,
    where an ndarray is a single buffer on both ends.
    """

    rows: np.ndarray  # [N, K] integer ids
    positions: np.ndarray  # [N] int32 indices into the scored window
    window: int
    # Global offset, within ``window``, of the current scheduler-level prefill
    # chunk. Zero for an unchunked request. Logits-level subchunks add their own
    # split offset on top of this value.
    base: int = 0

    def __len__(self) -> int:
        return self.window

    @property
    def k(self) -> int:
        return int(self.rows.shape[1])

    def to_nested_list(self) -> List[List[int]]:
        """Expand into the ``[window][K]`` nested list, unscored rows empty."""
        sparse: List[List[int]] = [[] for _ in range(self.window)]
        for pos, row in zip(self.positions.tolist(), self.rows.tolist()):
            sparse[pos] = row
        return sparse

    def last_position_ids(self) -> List[int]:
        """The ids at the final window position, as one flat next-token id-list.

        The next-token paths gather a single distribution per request, so they collapse a
        per-position request to its last window row -- ``to_nested_list()[-1]``, without
        building the list. Normally empty: the last window position would predict a token
        past the end of the response, so it is never scored.
        """
        if self.positions.size and int(self.positions[-1]) == self.window - 1:
            return self.rows[-1].tolist()
        return []


@dataclasses.dataclass
class SparseLogprobRows:
    """Gathered logprob rows for a :class:`PerPositionTokenIds` request.

    ``vals[i]`` is the gathered row for window position ``positions[i]``, of width K
    (K+1 with the ``SGLANG_OPD_TOPK_OVERLAP`` column). Only scored rows are carried, so
    this replaces a ``[window][K]`` nested list whose unscored entries are empty -- the
    b64 response path drops those anyway, and it is the only path that reads this.
    """

    vals: np.ndarray  # [N, K] float32
    positions: np.ndarray  # [N] int32 indices into the scored window
    window: int

    def __len__(self) -> int:
        return self.window

    def concat(self, other: "SparseLogprobRows") -> "SparseLogprobRows":
        """A later slice of the same window, appended. Returns a new instance.

        The chunked logits path hands one request's window over several calls, in
        increasing position order, so the pieces just append.
        """
        assert self.window == other.window, (
            f"cannot extend across windows {self.window} != {other.window}"
        )
        if other.positions.size == 0:
            return self
        if self.positions.size == 0:
            return other
        return SparseLogprobRows(
            vals=np.concatenate([self.vals, other.vals]),
            positions=np.concatenate([self.positions, other.positions]),
            window=self.window,
        )

    def prepend_boundary(self) -> "SparseLogprobRows":
        """Shift by one, matching the nested-list path's ``[None] + rows`` then pop-last.

        The value gathered at window position p predicts the token at position p+1, so
        the nested path publishes it at output index p+1 and drops the row for the last
        window position to keep the length at ``window``.
        """
        keep = self.positions < self.window - 1
        if bool(keep.all()):
            return SparseLogprobRows(self.vals, self.positions + 1, self.window)
        return SparseLogprobRows(self.vals[keep], self.positions[keep] + 1, self.window)


def compute_temp_top_p_normalized_logprobs(
    last_logits: torch.Tensor,
    logits_metadata: LogitsMetadata,
    top_p: Optional[torch.Tensor] = None,
    temperature: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    compute logprobs for the output token from the given logits.

    Returns:
        torch.Tensor: logprobs from logits
    """
    if top_p is None:
        top_p = logits_metadata.top_p
    if temperature is None:
        temperature = logits_metadata.temperature

    # Scale logits if temperature scaling is enabled
    if logits_metadata.temp_scaled_logprobs:
        last_logits = last_logits / temperature

    # Normalize logprobs if top_p normalization is enabled
    # NOTE: only normalize logprobs when top_p is set and not equal to 1.0
    if logits_metadata.top_p_normalized_logprobs and (top_p != 1.0).any():
        from sglang.srt.layers.sampler import top_p_normalize_probs_torch

        probs = torch.softmax(last_logits, dim=-1)
        del last_logits
        probs = top_p_normalize_probs_torch(probs, top_p)
        return torch.log(probs)
    else:
        return torch.nn.functional.log_softmax(last_logits, dim=-1)


def get_top_logprobs_raw(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    max_k = max(top_logprobs_nums)
    values, indices = logprobs.topk(max_k, dim=-1)
    if not no_copy_to_cpu:
        values = values.tolist()
        indices = indices.tolist()

    top_logprobs_val = []
    top_logprobs_idx = []

    if stage == LogprobStage.DECODE:
        for i, k in enumerate(top_logprobs_nums):
            top_logprobs_val.append(values[i][:k])
            top_logprobs_idx.append(indices[i][:k])
    else:
        pt = 0
        for k, pruned_len in zip(top_logprobs_nums, extend_logprob_pruned_lens_cpu):
            if pruned_len <= 0:
                top_logprobs_val.append([])
                top_logprobs_idx.append([])
                continue

            top_logprobs_val.append([values[pt + j][:k] for j in range(pruned_len)])
            top_logprobs_idx.append([indices[pt + j][:k] for j in range(pruned_len)])
            pt += pruned_len

    return top_logprobs_val, top_logprobs_idx


def get_top_logprobs_prefill(
    all_logprobs: torch.Tensor, logits_metadata: LogitsMetadata
):
    # ``process_input_logprobs_by_chunk`` calls get_top_logprobs_chunk below, but a
    # short scheduler-level prefill slice bypasses that logits-level chunker and
    # lands here.  Keep the dynamic offline-OPD representation sparse in both
    # cases.  In particular, the final scheduler chunk commonly has only a few
    # rows while PerPositionTokenIds still describes the complete scored window.
    token_ids_logprobs = logits_metadata.token_ids_logprobs
    sparse_positions = (
        _TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS
        and token_ids_logprobs is not None
        and len(token_ids_logprobs) == len(logits_metadata.top_logprobs_nums)
        and all(
            isinstance(ids, PerPositionTokenIds)
            for ids, k in zip(
                token_ids_logprobs, logits_metadata.top_logprobs_nums
            )
            if k > 0
        )
    )
    if sparse_positions:
        values: List = []
        indices: List = []
        remaining = get_top_logprobs_chunk(
            all_logprobs,
            logits_metadata,
            logits_metadata.top_logprobs_nums,
            logits_metadata.extend_logprob_pruned_lens_cpu,
            values,
            indices,
            0,
            token_ids_logprobs,
        )
        assert remaining == 0, (
            "short sparse Top-K prefill did not consume all logprob rows: "
            f"remaining offset={remaining}"
        )
        return values, indices
    return get_top_logprobs_raw(
        all_logprobs,
        logits_metadata.top_logprobs_nums,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
    )


def get_top_logprobs(
    logprobs: torch.Tensor,
    top_logprobs_nums: List[int],
    no_copy_to_cpu: bool = False,
):
    return get_top_logprobs_raw(
        logprobs,
        top_logprobs_nums,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def _is_per_position_token_ids(token_ids) -> bool:
    """Whether a request's token_ids_logprob is a per-position id matrix.

    Broadcast mode (legacy): ``token_ids`` is a flat ``List[int]`` applied to every
    scored position. Per-position mode: ``token_ids`` is a ``List[List[int]]`` with
    one length-K id-list per scored position, so each position gathers its own ids.
    Disambiguated structurally by whether the first element is itself a list/tuple.
    """
    if isinstance(token_ids, PerPositionTokenIds):
        return True
    return (
        token_ids is not None
        and len(token_ids) > 0
        and isinstance(token_ids[0], (list, tuple))
    )


def _append_overlap_count(
    row_logprobs: torch.Tensor, row_thresholds: torch.Tensor, k: int
) -> torch.Tensor:
    """Append ``|TopK_teacher ∩ ids|`` as a trailing column to gathered value rows.

    ``row_thresholds`` is the K-th largest teacher logprob per row, so an id counts iff
    its logprob is >= it. Exact ties at the boundary can push the count above K, hence
    the clamp.
    """
    count = (
        (row_logprobs >= row_thresholds)
        .sum(dim=1, keepdim=True)
        .clamp_(max=k)
        .to(row_logprobs.dtype)
    )
    return torch.cat([row_logprobs, count], dim=1)


def _gather_per_position_compact(
    logprobs: torch.Tensor,
    pt: int,
    num_rows: int,
    token_ids: PerPositionTokenIds,
    base: int = 0,
) -> SparseLogprobRows:
    """Gather the scored rows of a compact per-position request, staying in ndarray form.

    ``logprobs[pt : pt + num_rows]`` holds sequence-relative window rows
    ``[base, base + num_rows)``. The unchunked caller passes the whole window
    (``base=0``, ``num_rows=window``); the chunked caller passes whichever slice of it
    landed in the current logits chunk.

    Same math as the sparse branch of ``get_token_ids_logprobs_raw``, but the ids arrive
    as one ``[N, K]`` array and the values leave as one, so neither side pays the ~N*K
    Python objects that a nested list costs to read and to build.
    """
    positions, rows = token_ids.positions, token_ids.rows
    if base or num_rows != token_ids.window:
        sel = (positions >= base) & (positions < base + num_rows)
        positions, rows = positions[sel], rows[sel]
    # torch.tensor (not from_numpy) because the ids may be a read-only frombuffer view;
    # the copy is a C-level int32 -> int64 widening, not a per-element Python read.
    ids_2d = torch.tensor(rows, dtype=torch.long).to(
        logprobs.device, non_blocking=True
    )  # [N, K]
    row_idx = torch.tensor(
        positions if base == 0 else positions - base, dtype=torch.long
    ).to(
        logprobs.device, non_blocking=True
    )  # [N], relative to this slice
    # Advanced-index both dims so only the [N, K] entries are read; selecting the scored
    # rows first would materialize an [N, vocab_size] copy.
    pos_logprobs = logprobs[(pt + row_idx).unsqueeze(1), ids_2d]
    if _OPD_TOPK_OVERLAP and positions.size:
        k = token_ids.k
        # topk on the slice view keeps the intermediate at [num_rows, K].
        thresholds = logprobs[pt : pt + num_rows].topk(k, dim=-1).values[:, -1:]
        pos_logprobs = _append_overlap_count(pos_logprobs, thresholds[row_idx], k)
    return SparseLogprobRows(
        vals=pos_logprobs.float().cpu().numpy(),
        positions=positions,
        window=token_ids.window,
    )


def get_token_ids_logprobs_raw(
    logprobs: torch.Tensor,
    token_ids_logprobs_list: List[Optional[List[int]]],
    stage: LogprobStage,
    extend_logprob_pruned_lens_cpu: Optional[List[int]] = None,
    no_copy_to_cpu: bool = False,
):
    vals, idxs = [], []
    if stage == LogprobStage.DECODE:
        for i, token_ids in enumerate(token_ids_logprobs_list):
            if token_ids is None:
                vals.append([])
                idxs.append([])
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                    logprobs.device, non_blocking=True
                )
                row = logprobs[i, token_ids_tensor]
                vals.append(row if no_copy_to_cpu else row.tolist())
                idxs.append(token_ids)
    else:  # prefill
        pt = 0
        for i, (token_ids, pruned_len) in enumerate(
            zip(token_ids_logprobs_list, extend_logprob_pruned_lens_cpu)
        ):
            if pruned_len <= 0:
                vals.append([])
                idxs.append([])
                continue
            if isinstance(token_ids, PerPositionTokenIds):
                # Compact OPD fast path: ids in, values out, both as one array. The idx
                # entry is the request's own id matrix -- only the b64 response path
                # consumes a compact result, and it never reads the ids back.
                base = int(token_ids.base)
                assert 0 <= base <= token_ids.window, (
                    f"per-position token_ids_logprob base {base} is outside "
                    f"window {token_ids.window}"
                )
                assert base + pruned_len <= token_ids.window, (
                    "per-position token_ids_logprob chunk exceeds window: "
                    f"base={base}, pruned_len={pruned_len}, "
                    f"window={token_ids.window}"
                )
                vals.append(
                    _gather_per_position_compact(
                        logprobs, pt, pruned_len, token_ids, base=base
                    )
                )
                idxs.append(token_ids)
                pt += pruned_len
                continue
            if _is_per_position_token_ids(token_ids):
                # Per-position ids: token_ids is a list of `pruned_len` id-lists (each
                # length K), so each scored position gathers its OWN ids instead of a
                # single id-list broadcast to every position. An EMPTY id-list means
                # "skip this position": the caller only sends id-rows for the positions
                # it consumes, and the emitted row stays empty so downstream length
                # bookkeeping (and the b64 response packer, which drops empty rows) is
                # unchanged.
                assert len(token_ids) == pruned_len, (
                    f"per-position token_ids_logprob length {len(token_ids)} "
                    f"!= pruned_len {pruned_len}"
                )
                scored = [j for j, row in enumerate(token_ids) if row]
                window = logprobs[pt : pt + pruned_len]
                if _OPD_TOPK_OVERLAP and scored:
                    # An id is in the teacher's own top-K iff its logprob is >= the K-th
                    # largest one at that position. topk on the window view keeps the
                    # intermediate at [pruned_len, K]; selecting the scored rows first
                    # would materialize [len(scored), vocab].
                    k = len(token_ids[scored[0]])
                    thresholds = window.topk(k, dim=-1).values[:, -1:]
                else:
                    k = thresholds = None

                if len(scored) == pruned_len:
                    ids_2d = torch.tensor(token_ids, dtype=torch.long).to(
                        logprobs.device, non_blocking=True
                    )  # [pruned_len, K]
                    pos_logprobs = torch.gather(window, 1, ids_2d)
                    if thresholds is not None:
                        pos_logprobs = _append_overlap_count(
                            pos_logprobs, thresholds, k
                        )
                    vals.append(
                        pos_logprobs if no_copy_to_cpu else pos_logprobs.tolist()
                    )
                else:
                    assert not no_copy_to_cpu, (
                        "sparse per-position token_ids_logprob cannot be returned as a "
                        "single tensor: skipped positions have no row"
                    )
                    sparse_vals = [[] for _ in range(pruned_len)]
                    if scored:
                        ids_2d = torch.tensor(
                            [token_ids[j] for j in scored], dtype=torch.long
                        ).to(
                            logprobs.device, non_blocking=True
                        )  # [len(scored), K]
                        row_idx = torch.tensor(
                            scored, dtype=torch.long, device=logprobs.device
                        )
                        # Advanced-index both dims so only the [len(scored), K] entries
                        # are read; selecting rows first would materialize a
                        # [len(scored), vocab_size] copy.
                        pos_logprobs = logprobs[(pt + row_idx).unsqueeze(1), ids_2d]
                        if thresholds is not None:
                            pos_logprobs = _append_overlap_count(
                                pos_logprobs, thresholds[row_idx], k
                            )
                        for j, row in zip(scored, pos_logprobs.tolist()):
                            sparse_vals[j] = row
                    vals.append(sparse_vals)
                if thresholds is None:
                    idxs.append([list(row) for row in token_ids])
                else:
                    # Keep value/index widths equal for the JSON (detokenize) path.
                    idxs.append(
                        [[*row, _OVERLAP_ID] if row else [] for row in token_ids]
                    )
            else:
                token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).to(
                    logprobs.device, non_blocking=True
                )
                pos_logprobs = logprobs[pt : pt + pruned_len, token_ids_tensor]
                vals.append(pos_logprobs if no_copy_to_cpu else pos_logprobs.tolist())
                idxs.append([token_ids for _ in range(pruned_len)])
            pt += pruned_len
    return vals, idxs


def get_token_ids_logprobs_prefill(
    all_logprobs, logits_metadata: LogitsMetadata, no_copy_to_cpu=False
):
    return get_token_ids_logprobs_raw(
        all_logprobs,
        logits_metadata.token_ids_logprobs,
        stage=LogprobStage.PREFILL,
        extend_logprob_pruned_lens_cpu=logits_metadata.extend_logprob_pruned_lens_cpu,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_token_ids_logprobs(logprobs, token_ids_logprobs, no_copy_to_cpu=False):
    return get_token_ids_logprobs_raw(
        logprobs,
        token_ids_logprobs,
        stage=LogprobStage.DECODE,
        no_copy_to_cpu=no_copy_to_cpu,
    )


def get_top_logprobs_chunk(
    logprobs: torch.Tensor,
    logits_metadata: LogitsMetadata,
    top_k_nums: List[int],
    pruned_lens: List[int],
    input_top_logprobs_val: List,
    input_top_logprobs_idx: List,
    split_pruned_len: int,
    token_ids_logprobs: Optional[List] = None,
) -> int:
    """Get top-k logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        logits_metadata: Metadata containing top-k and pruned length info
        top_k_nums: List of top-k numbers for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_top_logprobs_val: List to store top-k logprob values
        input_top_logprobs_idx: List to store top-k token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """
    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

    # Dynamic offline-OPD: token_ids_logprobs is used only as a compact carrier
    # for the trainable positions.  Keep the public result shaped like the legacy
    # scored window (empty rows for masked positions), but each non-empty row is a
    # NumPy view rather than 4096 Python scalars.  Consecutive runs are processed as
    # tensor views, avoiding an [N, vocab] advanced-index copy and bounding the GPU
    # temporary to one trainable run's Top-K output.
    sparse_positions = (
        _TOP_LOGPROBS_ON_TOKEN_ID_POSITIONS
        and token_ids_logprobs is not None
        and len(token_ids_logprobs) == len(top_k_nums)
        and all(
            isinstance(ids, PerPositionTokenIds)
            for ids, k in zip(token_ids_logprobs, top_k_nums)
            if k > 0
        )
    )
    if sparse_positions:
        pt = 0
        next_split_pruned_len = 0
        for n, (k, full_pruned_len, position_ids) in enumerate(
            zip(top_k_nums, pruned_lens, token_ids_logprobs)
        ):
            split_base = split_pruned_len if n == 0 else 0
            remaining = full_pruned_len - split_base
            if remaining <= 0:
                input_top_logprobs_val.append([])
                input_top_logprobs_idx.append([])
                continue

            avail = min(remaining, max(logprobs.shape[0] - pt, 0))
            val_rows = [[] for _ in range(avail)]
            idx_rows = [[] for _ in range(avail)]
            positions = position_ids.positions
            global_base = position_ids.base + split_base
            selected = positions[
                (positions >= global_base) & (positions < global_base + avail)
            ]

            # Split the sorted positions into contiguous runs. torch.topk on each
            # slice is mathematically identical to the legacy all-row topk, while
            # masked observation spans never enter the kernel.
            if selected.size:
                run_starts = np.r_[0, np.flatnonzero(np.diff(selected) != 1) + 1]
                run_ends = np.r_[run_starts[1:], selected.size]
                for start, end in zip(run_starts.tolist(), run_ends.tolist()):
                    first = int(selected[start] - global_base)
                    last = int(selected[end - 1] - global_base) + 1
                    ret = logprobs[pt + first : pt + last].topk(k, dim=1)
                    values = ret.values.float().cpu().numpy()
                    indices = ret.indices.to(torch.int32).cpu().numpy()
                    for offset in range(last - first):
                        val_rows[first + offset] = values[offset]
                        idx_rows[first + offset] = indices[offset]

            if split_base > 0:
                input_top_logprobs_val[-1].extend(val_rows)
                input_top_logprobs_idx[-1].extend(idx_rows)
            else:
                input_top_logprobs_val.append(val_rows)
                input_top_logprobs_idx.append(idx_rows)

            if avail < remaining:
                next_split_pruned_len = split_base + avail
            pt += remaining
        return next_split_pruned_len

    max_k = max(logits_metadata.top_logprobs_nums)
    ret = logprobs.topk(max_k, dim=1)
    values = ret.values.tolist()
    indices = ret.indices.tolist()

    pt = 0
    next_split_pruned_len = 0
    for n, (k, pruned_len) in enumerate(zip(top_k_nums, pruned_lens)):
        if n == 0:
            # For the first sequence, adjust the pruned length
            pruned_len -= split_pruned_len
        else:
            # After the first sequence, no split in the middle
            split_pruned_len = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no top-k logprobs to process
            input_top_logprobs_val.append([])
            input_top_logprobs_idx.append([])
            continue

        # Get the top-k logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= len(values):
                next_split_pruned_len = split_pruned_len + j
                break
            # Append the top-k logprobs
            val.append(values[pt + j][:k])
            idx.append(indices[pt + j][:k])

        # Append or extend based on whether the sequence was split across chunks
        if len(val) > 0:
            if split_pruned_len > 0:
                input_top_logprobs_val[-1].extend(val)
                input_top_logprobs_idx[-1].extend(idx)
            else:
                input_top_logprobs_val.append(val)
                input_top_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def get_token_ids_logprobs_chunk(
    logprobs: torch.Tensor,
    token_ids_logprobs: List[int],
    pruned_lens: List[int],
    input_token_ids_logprobs_val: List,
    input_token_ids_logprobs_idx: List,
    split_pruned_len: int = 0,
):
    """Get token_ids logprobs for each sequence in the chunk.

    Args:
        logprobs: Log probabilities tensor of shape [seq_len, vocab_size]
        logits_metadata: Metadata containing token IDs and pruned length info
        token_ids_logprobs: List of token IDs for each sequence
        pruned_lens: List of pruned lengths for each sequence
        input_token_ids_logprobs_val: List to store token logprob values
        input_token_ids_logprobs_idx: List to store token indices
        split_pruned_len: Length of pruned tokens from previous chunk

    Returns:
        int: Number of remaining tokens to process in next chunk
    """

    # No sequences in the chunk
    if logprobs.shape[0] == 0:
        return 0

    pt = 0
    next_split_pruned_len = 0
    for n, (token_ids, pruned_len) in enumerate(
        zip(
            token_ids_logprobs,
            pruned_lens,
        )
    ):
        # Adjust pruned length for first sequence
        if n == 0:
            pruned_len -= split_pruned_len
            # positions of this request already consumed in a previous chunk; used to
            # index the correct row of a per-position id matrix.
            base = split_pruned_len
        else:
            split_pruned_len = 0
            base = 0

        if pruned_len <= 0:
            # if pruned length is less than or equal to 0,
            # there is no token ids logprobs to process
            input_token_ids_logprobs_val.append([])
            input_token_ids_logprobs_idx.append([])
            continue

        if isinstance(token_ids, PerPositionTokenIds):
            # Compact OPD fast path. The nested-list loop below gathers one row per
            # window position with its own kernel launch and device sync; an agent
            # trajectory has tens of thousands of them per request. Do this chunk's
            # whole slice of the window in one advanced-index instead.
            avail = min(pruned_len, logprobs.shape[0] - pt)
            if avail < pruned_len:
                # Rest of this request's window lands in the next chunk.
                next_split_pruned_len = split_pruned_len + avail
            if avail > 0:
                chunk_rows = _gather_per_position_compact(
                    logprobs, pt, avail, token_ids, base=token_ids.base + base
                )
                if split_pruned_len > 0:
                    input_token_ids_logprobs_val[-1] = input_token_ids_logprobs_val[
                        -1
                    ].concat(chunk_rows)
                else:
                    input_token_ids_logprobs_val.append(chunk_rows)
                    # The idx entry is the request's own id matrix -- only the b64
                    # response path consumes a compact result, and it never reads the
                    # ids back. It covers the whole window, so later chunks add nothing.
                    input_token_ids_logprobs_idx.append(token_ids)
            pt += pruned_len
            continue

        per_position = _is_per_position_token_ids(token_ids)

        # Same trailing overlap column as the unchunked path, over this chunk's slice of
        # the request's scored rows.
        thr_list = None
        if _OPD_TOPK_OVERLAP and per_position:
            first_row = next((row for row in token_ids if row), None)
            end = min(pt + pruned_len, logprobs.shape[0])
            if first_row is not None and end > pt:
                thr_list = (
                    logprobs[pt:end].topk(len(first_row), dim=-1).values[:, -1].tolist()
                )

        # Get the token ids logprobs
        val = []
        idx = []
        for j in range(pruned_len):
            # Handle remaining tokens in next chunk if any
            if pt + j >= logprobs.shape[0]:
                next_split_pruned_len = split_pruned_len + j
                break
            if token_ids is not None:
                # Per-position mode gathers this position's own ids (row base+j of the
                # full [R, K] matrix); broadcast mode reuses the single id-list.
                ids_j = token_ids[base + j] if per_position else token_ids
                row = logprobs[pt + j, ids_j].tolist()
                # An empty id-list means "skip this position": it must stay empty, so it
                # gets no overlap column either.
                if thr_list is not None and len(ids_j) > 0:
                    thr = thr_list[j]
                    row.append(float(sum(1 for v in row if v >= thr)))
                    ids_j = [*ids_j, _OVERLAP_ID]
                val.append(row)
                idx.append(ids_j)

        # Append or extend based on whether the sequence was split across chunks
        if len(val) > 0:
            if split_pruned_len > 0:
                input_token_ids_logprobs_val[-1].extend(val)
                input_token_ids_logprobs_idx[-1].extend(idx)
            else:
                input_token_ids_logprobs_val.append(val)
                input_token_ids_logprobs_idx.append(idx)

        pt += pruned_len
    return next_split_pruned_len


def add_output_logprobs_for_spec_v1(
    batch: ScheduleBatch,
    res: Union[EagleVerifyOutput, NgramVerifyInput],
    logits_output: Optional[LogitsProcessorOutput] = None,
):
    # Extract args
    if logits_output is None:
        logits_output = res.logits_output

    if hasattr(res, "num_correct_drafts_per_req_cpu"):
        num_correct_drafts_per_req_cpu = res.num_correct_drafts_per_req_cpu
    else:
        # FIXME: Get a NgramVerifyOutput class and use that instead of this hack.
        num_correct_drafts_per_req_cpu = res.num_correct_drafts.tolist()

    top_logprobs_nums = batch.top_logprobs_nums
    token_ids_logprobs = batch.token_ids_logprobs
    accept_indices = res.accept_indices
    assert len(accept_indices) == len(logits_output.next_token_logits)

    temperatures = batch.sampling_info.temperatures
    num_draft_tokens = batch.spec_info.draft_token_num
    # acceptance indices are the indices in a "flattened" batch.
    # dividing it to num_draft_tokens will yield the actual batch index.
    temperatures = temperatures[accept_indices // num_draft_tokens]
    if envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits, dim=-1
        )
    else:
        logprobs = torch.nn.functional.log_softmax(
            logits_output.next_token_logits / temperatures, dim=-1
        )
    batch_next_token_ids = res.accept_tokens
    num_tokens_per_req = [accept + 1 for accept in num_correct_drafts_per_req_cpu]

    # We should repeat top_logprobs_nums to match num_tokens_per_req.
    top_logprobs_nums_repeat_interleaved = [
        num
        for num, num_tokens in zip(top_logprobs_nums, num_tokens_per_req)
        for _ in range(num_tokens)
    ]

    token_ids_logprobs_repeat_interleaved = [
        token_ids
        for token_ids, num_tokens in zip(token_ids_logprobs, num_tokens_per_req)
        for _ in range(num_tokens)
    ]

    # Extract logprobs
    should_top_logprobs = any(x > 0 for x in top_logprobs_nums)
    should_token_ids_logprobs = any(x is not None for x in token_ids_logprobs)
    if should_top_logprobs:
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            logprobs,
            top_logprobs_nums_repeat_interleaved,
        )

    if should_token_ids_logprobs:
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            logprobs,
            token_ids_logprobs_repeat_interleaved,
        )

    logits_output.next_token_logprobs = logprobs[
        torch.arange(len(batch_next_token_ids), device=batch.sampling_info.device),
        batch_next_token_ids,
    ]

    # Add output logprobs to the request
    pt = 0
    next_token_logprobs = logits_output.next_token_logprobs.tolist()
    accept_tokens_list = batch_next_token_ids.tolist()
    token_top_logprobs_val = logits_output.next_token_top_logprobs_val
    token_top_logprobs_idx = logits_output.next_token_top_logprobs_idx
    token_ids_logprobs_val = logits_output.next_token_token_ids_logprobs_val
    token_ids_logprobs_idx = logits_output.next_token_token_ids_logprobs_idx
    for req, num_tokens in zip(batch.reqs, num_tokens_per_req, strict=True):
        for _ in range(num_tokens):
            if req.return_logprob:
                req.output_token_logprobs_val.append(next_token_logprobs[pt])
                req.output_token_logprobs_idx.append(accept_tokens_list[pt])
                if req.top_logprobs_num > 0:
                    assert (
                        should_top_logprobs
                    ), "Inconsistent state: should_top_logprobs is False"
                    req.output_top_logprobs_val.append(token_top_logprobs_val[pt])
                    req.output_top_logprobs_idx.append(token_top_logprobs_idx[pt])
                if req.token_ids_logprob is not None and should_token_ids_logprobs:
                    req.output_token_ids_logprobs_val.append(token_ids_logprobs_val[pt])
                    req.output_token_ids_logprobs_idx.append(token_ids_logprobs_idx[pt])
            pt += 1


def compute_spec_v2_logprobs(
    batch,
    logits_output,
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    speculative_num_steps: int,
):
    """Compute logprobs for accepted tokens after spec v2 verify sampling.

    Gathers logits at accepted positions, applies log_softmax (temperature-scaled
    if not greedy), and populates logits_output.next_token_logprobs (plus optional
    top-k / token-ids logprobs) so they flow through copy_to_cpu().
    """
    bs = len(batch.seq_lens)
    max_accept = speculative_num_steps + 1
    device = predict.device

    flat_accept_idx = accept_index.long().reshape(-1)
    gathered_logits = logits_output.next_token_logits[flat_accept_idx]

    if batch.sampling_info.is_all_greedy or envs.SGLANG_RETURN_ORIGINAL_LOGPROB.get():
        gathered_logprobs = torch.nn.functional.log_softmax(gathered_logits, dim=-1)
    else:
        temperatures = torch.repeat_interleave(
            batch.sampling_info.temperatures,
            max_accept,
            dim=0,
        )
        gathered_logprobs = torch.nn.functional.log_softmax(
            gathered_logits / temperatures, dim=-1
        )
    gathered_logprobs.clamp_(min=torch.finfo(gathered_logprobs.dtype).min)

    accepted_token_ids = predict[flat_accept_idx]
    token_logprobs = gathered_logprobs[
        torch.arange(bs * max_accept, device=device),
        accepted_token_ids.long(),
    ]
    logits_output.next_token_logprobs = token_logprobs.reshape(bs, max_accept)

    if batch.top_logprobs_nums and any(x > 0 for x in batch.top_logprobs_nums):
        top_logprobs_nums_expanded = [
            num for num in batch.top_logprobs_nums for _ in range(max_accept)
        ]
        (
            logits_output.next_token_top_logprobs_val,
            logits_output.next_token_top_logprobs_idx,
        ) = get_top_logprobs(
            gathered_logprobs, top_logprobs_nums_expanded, no_copy_to_cpu=True
        )

    if batch.token_ids_logprobs and any(
        x is not None for x in batch.token_ids_logprobs
    ):
        token_ids_logprobs_expanded = [
            ids for ids in batch.token_ids_logprobs for _ in range(max_accept)
        ]
        (
            logits_output.next_token_token_ids_logprobs_val,
            logits_output.next_token_token_ids_logprobs_idx,
        ) = get_token_ids_logprobs(
            gathered_logprobs, token_ids_logprobs_expanded, no_copy_to_cpu=True
        )
