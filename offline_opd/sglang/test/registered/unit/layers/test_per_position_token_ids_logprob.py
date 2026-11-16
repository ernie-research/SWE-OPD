from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=5, stage="stage-b", runner_config="1-gpu-small")
register_amd_ci(est_time=5, suite="stage-b-test-1-gpu-small-amd")

import base64
import unittest

import numpy as np
import torch

from sglang.srt.layers.utils.logprob import (
    _OPD_TOPK_OVERLAP,
    LogprobStage,
    PerPositionTokenIds,
    SparseLogprobRows,
    get_token_ids_logprobs_chunk,
    get_token_ids_logprobs_raw,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.srt.managers.tokenizer_manager import _array_to_b64, _rows_to_b64

VOCAB = 512
PROMPT_LEN = 7
RESPONSE_LEN = 11
K = 4
WINDOW = RESPONSE_LEN + 1  # boundary position + one per response token


def _publish(val, idx):
    """Mirror Scheduler._process_input_token_ids_logprobs for both representations."""
    if isinstance(val, SparseLogprobRows):
        return val.prepend_boundary(), idx
    out_val, out_idx = [None], [None]
    out_val.extend(val)
    out_idx.extend(idx)
    out_val.pop()
    out_idx.pop()
    return out_val, out_idx


class TestPerPositionTokenIdsLogprob(unittest.TestCase):
    """The compact ``token_ids_logprob_b64`` path must match the nested-list path.

    ``return_token_ids_logprob_as_bytes`` keeps the per-position id matrix as arrays from
    request normalization through to the response blob, instead of expanding it into a
    ``[window][K]`` Python list in the TokenizerManager and back again. That is a pure
    performance change, so every observable must stay identical.
    """

    def setUp(self):
        rng = np.random.default_rng(0)
        # Strictly increasing and below WINDOW - 1, as a loss-mask-derived request is.
        self.positions = np.array([0, 1, 3, 4, 7, 9], dtype=np.int32)
        self.rows = rng.integers(0, VOCAB, size=(len(self.positions), K), dtype=np.int32)
        self.blob = {
            "vals": base64.b64encode(self.rows.tobytes()).decode("ascii"),
            "shape": list(self.rows.shape),
            "dtype": "int32",
            "positions": self.positions.tolist(),
        }
        torch.manual_seed(0)
        self.logprobs = torch.log_softmax(torch.randn(WINDOW, VOCAB), dim=-1)

    def _normalized(self, as_bytes: bool):
        req = GenerateReqInput(
            input_ids=list(range(PROMPT_LEN + RESPONSE_LEN)),
            return_logprob=True,
            logprob_start_len=PROMPT_LEN + RESPONSE_LEN - RESPONSE_LEN - 1,
            token_ids_logprob_b64=dict(self.blob),
            return_token_ids_logprob_as_bytes=as_bytes,
        )
        req.normalize_batch_and_arguments()
        return req.token_ids_logprob

    def _gather(self, token_ids):
        return get_token_ids_logprobs_raw(
            self.logprobs,
            [token_ids],
            stage=LogprobStage.PREFILL,
            extend_logprob_pruned_lens_cpu=[WINDOW],
        )

    def test_normalization_representations_agree(self):
        compact = self._normalized(as_bytes=True)
        legacy = self._normalized(as_bytes=False)
        self.assertIsInstance(compact, PerPositionTokenIds)
        self.assertIsInstance(legacy, list)
        self.assertEqual(len(compact), WINDOW)
        self.assertEqual(compact.window, WINDOW)
        self.assertEqual(compact.k, K)
        self.assertEqual(compact.to_nested_list(), legacy)

    def test_response_blob_is_identical(self):
        compact_val, compact_idx = self._gather(self._normalized(as_bytes=True))
        legacy_val, legacy_idx = self._gather(self._normalized(as_bytes=False))
        self.assertIsInstance(compact_val[0], SparseLogprobRows)
        self.assertEqual(len(legacy_val[0]), WINDOW)

        compact_pub, _ = _publish(compact_val[0], compact_idx[0])
        legacy_pub, _ = _publish(legacy_val[0], legacy_idx[0])

        # The length is what add_input_logprob_return_values asserts on.
        self.assertEqual(len(compact_pub), WINDOW)
        self.assertEqual(len(legacy_pub), WINDOW)
        self.assertTrue((compact_pub.positions == self.positions + 1).all())

        blob = _array_to_b64(compact_pub.vals, np.float32)
        self.assertEqual(
            blob, _rows_to_b64([r for r in legacy_pub if r], np.float32)
        )

        # And it is what the OPD client reconstructs: the teacher's logprob at each of
        # the requested ids, one row per requested position, in position order.
        decoded = np.frombuffer(
            base64.b64decode(blob["vals"]), dtype=np.float32
        ).reshape(blob["shape"])
        scored = self.logprobs[torch.from_numpy(self.positions).long()]
        expected = torch.gather(
            scored, 1, torch.from_numpy(self.rows.astype(np.int64))
        ).numpy()
        if _OPD_TOPK_OVERLAP:
            # One trailing column: |TopK_teacher of that position ∩ the requested ids|.
            thresholds = scored.topk(K, dim=-1).values[:, -1:]
            counts = (
                (torch.from_numpy(expected) >= thresholds)
                .sum(dim=1, keepdim=True)
                .clamp_(max=K)
                .float()
                .numpy()
            )
            expected = np.concatenate([expected, counts], axis=1)
        np.testing.assert_allclose(decoded, expected, rtol=0, atol=1e-6)

    def test_chunked_path_matches_unchunked(self):
        """Splitting the window across logits chunks must rebuild the same rows.

        ``process_input_logprobs_by_chunk`` hands one request's scored window to
        ``get_token_ids_logprobs_chunk`` a slice at a time, so the compact result has to
        be accumulated across calls. This is the path the OPD teacher actually takes
        (``SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK=1``).
        """
        whole = self._gather(self._normalized(as_bytes=True))[0][0]
        token_ids = self._normalized(as_bytes=True)

        for chunk in (1, 2, 3, 5, WINDOW - 1, WINDOW, WINDOW + 3):
            with self.subTest(chunk=chunk):
                val, idx, split = [], [], 0
                for start in range(0, WINDOW, chunk):
                    # The caller passes the request's FULL pruned_len every chunk and
                    # lets split_pruned_len carry the offset.
                    split = get_token_ids_logprobs_chunk(
                        self.logprobs[start : start + chunk],
                        [token_ids],
                        [WINDOW],
                        val,
                        idx,
                        split,
                    )
                self.assertEqual(split, 0)
                self.assertEqual(len(val), 1)
                merged = val[0]
                self.assertIsInstance(merged, SparseLogprobRows)
                self.assertEqual(len(merged), WINDOW)
                self.assertIs(idx[0], token_ids)
                np.testing.assert_array_equal(merged.positions, whole.positions)
                np.testing.assert_allclose(merged.vals, whole.vals, rtol=0, atol=0)

    def test_chunked_path_matches_legacy_values(self):
        """The compact chunk result must hold what the nested chunk path built row by row.

        Also runs the nested list through the same split, since the compact branch sits
        in front of that loop.
        """
        legacy_val, _ = self._gather(self._normalized(as_bytes=False))
        nested = self._normalized(as_bytes=False)
        chunk = 5

        compact_out, nested_out = [], []
        compact_idx, nested_idx = [], []
        split_compact = split_nested = 0
        for start in range(0, WINDOW, chunk):
            window = self.logprobs[start : start + chunk]
            split_compact = get_token_ids_logprobs_chunk(
                window, [self._normalized(as_bytes=True)], [WINDOW],
                compact_out, compact_idx, split_compact,
            )
            split_nested = get_token_ids_logprobs_chunk(
                window, [nested], [WINDOW], nested_out, nested_idx, split_nested,
            )

        compact = compact_out[0]
        self.assertEqual(len(nested_out[0]), WINDOW)
        # Every row the legacy path scored is present, and no others.
        expected_positions = [j for j in range(WINDOW) if legacy_val[0][j]]
        self.assertEqual(compact.positions.tolist(), expected_positions)
        for i, p in enumerate(expected_positions):
            for other in (legacy_val[0][p], nested_out[0][p]):
                np.testing.assert_allclose(
                    compact.vals[i],
                    np.asarray(other, dtype=np.float32),
                    rtol=0,
                    atol=1e-6,
                )


    def test_next_token_path_collapses_compact_ids(self):
        """The next-token gather must accept the matrix, not just the nested list.

        ``compute_logprobs_only`` runs on every teacher request and computes one
        next-token distribution per request, so it collapses a per-position request to
        its last window row. The nested path indexes ``tids[-1]``, which a
        ``PerPositionTokenIds`` does not support.
        """
        from sglang.srt.layers.sampler import get_token_ids_logprobs_batch_optimized

        next_logprobs = self.logprobs[:1]  # [batch=1, vocab]
        compact_val, compact_idx = get_token_ids_logprobs_batch_optimized(
            next_logprobs, [self._normalized(as_bytes=True)]
        )
        legacy_val, legacy_idx = get_token_ids_logprobs_batch_optimized(
            next_logprobs, [self._normalized(as_bytes=False)]
        )
        # The last window position predicts past the end of the response, so a
        # loss-mask-derived request never scores it and both sides collapse to empty.
        self.assertEqual(compact_idx, [[]])
        self.assertEqual(compact_idx, legacy_idx)
        np.testing.assert_array_equal(compact_val[0].numpy(), legacy_val[0].numpy())

    def test_next_token_path_keeps_a_scored_last_position(self):
        """If the last window position is scored, its row is what gets collapsed to."""
        from sglang.srt.layers.sampler import get_token_ids_logprobs_batch_optimized

        positions = np.append(self.positions, WINDOW - 1).astype(np.int32)
        rows = np.concatenate([self.rows, self.rows[:1]])
        compact = PerPositionTokenIds(rows=rows, positions=positions, window=WINDOW)
        self.assertEqual(compact.last_position_ids(), rows[-1].tolist())

        next_logprobs = self.logprobs[:1]
        val, idx = get_token_ids_logprobs_batch_optimized(next_logprobs, [compact])
        self.assertEqual(idx, [rows[-1].tolist()])
        np.testing.assert_allclose(
            val[0].numpy(),
            next_logprobs[0, torch.from_numpy(rows[-1].astype(np.int64))].numpy(),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()
