import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from slime.ray.placement_group import _create_placement_group, _get_placement_group_layout, sort_key

NUM_GPUS = 0


def _args(**overrides):
    values = {
        "actor_num_nodes": 2,
        "actor_num_gpus_per_node": 8,
        "rollout_num_gpus": 32,
        "debug_train_only": False,
        "debug_rollout_only": False,
        "colocate": False,
        "rollout_external": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        pytest.param({}, (48, 16), id="normal_non_colocate"),
        pytest.param({"debug_train_only": True}, (16, 0), id="debug_train_only"),
        pytest.param({"debug_rollout_only": True}, (32, 0), id="debug_rollout_only"),
        pytest.param({"colocate": True, "rollout_num_gpus": 8}, (16, 0), id="colocate_rollout_less_than_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 16}, (16, 0), id="colocate_rollout_equals_actor"),
        pytest.param({"colocate": True, "rollout_num_gpus": 32}, (32, 0), id="colocate_rollout_more_than_actor"),
        pytest.param({"rollout_num_gpus": 0}, (16, 16), id="zero_rollout_gpus"),
        pytest.param({"colocate": True, "rollout_num_gpus": 0}, (16, 0), id="colocate_zero_rollout_gpus"),
        pytest.param({"rollout_external": True}, (16, 16), id="external"),
        pytest.param({"rollout_external": True, "debug_rollout_only": True}, (16, 0), id="external_debug_rollout"),
    ],
)
def test_placement_group_layout(overrides, expected):
    assert _get_placement_group_layout(_args(**overrides)) == expected


def test_create_zero_gpu_placement_group_is_empty():
    assert _create_placement_group(0) == (None, [], [])


def test_sort_key_orders_by_ip_then_gpu_without_head_ip():
    # (index, node_ip, gpu_id) bundles across two nodes; default = pure IP/gpu ordering.
    bundles = [
        (0, "10.0.0.2", 1),
        (1, "10.0.0.1", 0),
        (2, "10.0.0.2", 0),
        (3, "10.0.0.1", 1),
    ]
    ordered = [b[0] for b in sorted(bundles, key=sort_key)]
    assert ordered == [1, 3, 2, 0]


def test_sort_key_pins_head_ip_first_even_with_larger_ip():
    # Head node has the numerically LARGER ip; without the pin it would sort last.
    head_ip = "10.0.0.9"
    bundles = [
        (0, "10.0.0.1", 0),
        (1, "10.0.0.9", 0),  # head
        (2, "10.0.0.1", 1),
        (3, "10.0.0.9", 1),  # head
    ]
    ordered = [b[0] for b in sorted(bundles, key=lambda x: sort_key(x, head_ip=head_ip))]
    # Head bundles come first (by gpu_id), then the rest in IP/gpu order.
    assert ordered == [1, 3, 0, 2]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
