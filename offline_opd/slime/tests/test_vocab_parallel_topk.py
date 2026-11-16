"""Tests for exact top-k ids over tensor-parallel vocabulary shards."""

import socket

import torch
import torch.distributed as dist

from slime.utils.ppo_utils import vocab_parallel_topk_ids


NUM_GPUS = 0


def test_vocab_parallel_topk_ids_chunks_and_excludes_padded_vocab():
    logits = torch.tensor(
        [
            [0.0, 4.0, 1.0, 3.0, 2.0, -1.0, 100.0, 99.0],
            [5.0, 0.0, 3.0, 2.0, 4.0, 1.0, 101.0, 98.0],
        ]
    )
    actual = vocab_parallel_topk_ids(
        logits, 4, None, chunk_size=1, global_vocab_size=6
    )
    expected = logits[:, :6].topk(4, dim=-1).indices
    torch.testing.assert_close(actual, expected)


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def _distributed_topk_worker(rank: int, world_size: int, port: int) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        full_logits = torch.tensor(
            [
                [0.0, 7.0, 1.0, 5.0, 2.0, 3.0, 8.0, 4.0, 9.0, 6.0, -1.0, 100.0],
                [9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0, 0.0, -1.0, 99.0],
                [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 9.0, 7.0, 5.0, 3.0, 1.0, 98.0],
            ]
        )
        local_logits = full_logits[:, rank * 6 : (rank + 1) * 6]
        actual = vocab_parallel_topk_ids(
            local_logits,
            4,
            dist.group.WORLD,
            chunk_size=2,
            global_vocab_size=11,
        )
        expected = full_logits[:, :11].topk(4, dim=-1).indices
        torch.testing.assert_close(actual, expected)
    finally:
        dist.destroy_process_group()


def test_vocab_parallel_topk_ids_matches_full_vocab_with_two_tp_ranks():
    import torch.multiprocessing as mp

    mp.spawn(_distributed_topk_worker, args=(2, _free_port()), nprocs=2, join=True)
