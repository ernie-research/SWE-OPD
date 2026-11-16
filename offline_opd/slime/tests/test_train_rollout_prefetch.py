from types import SimpleNamespace

import pytest

import train


class _GenerateMethod:
    def __init__(self, events):
        self.events = events

    def remote(self, rollout_id):
        self.events.append(("generate", rollout_id))
        return ("generate_ref", rollout_id)


class _RolloutManager:
    def __init__(self, events):
        self.generate = _GenerateMethod(events)


def _args(**overrides):
    values = {
        "prefetch_rollout_data": True,
        "load_debug_rollout_data": "/data/rollout_{rollout_id}.pt",
        "offload_rollout": False,
        "rollout_global_dataset": False,
        "eval_interval": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_prefetch_keeps_exactly_one_next_load_in_flight(monkeypatch):
    events = []
    manager = _RolloutManager(events)

    def fake_get(ref):
        assert ref[0] == "generate_ref"
        events.append(("get", ref[1]))
        return ("rollout_data_ref", ref[1])

    monkeypatch.setattr(train.ray, "get", fake_get)

    pending = None
    for rollout_id in range(3):
        data_ref, pending = train._load_rollout_data_and_prefetch_next(
            manager, rollout_id, 3, pending, prefetch=True
        )
        assert data_ref == ("rollout_data_ref", rollout_id)
        events.append(("train", rollout_id))

    assert pending is None
    assert events == [
        ("generate", 0),
        ("get", 0),
        ("generate", 1),
        ("train", 0),
        ("get", 1),
        ("generate", 2),
        ("train", 1),
        ("get", 2),
        ("train", 2),
    ]


def test_disabled_prefetch_preserves_sequential_generation(monkeypatch):
    events = []
    manager = _RolloutManager(events)

    def fake_get(ref):
        events.append(("get", ref[1]))
        return ("rollout_data_ref", ref[1])

    monkeypatch.setattr(train.ray, "get", fake_get)

    pending = None
    for rollout_id in range(2):
        _, pending = train._load_rollout_data_and_prefetch_next(
            manager, rollout_id, 2, pending, prefetch=False
        )
        assert pending is None
        events.append(("train", rollout_id))

    assert events == [
        ("generate", 0),
        ("get", 0),
        ("train", 0),
        ("generate", 1),
        ("get", 1),
        ("train", 1),
    ]


def test_prefetch_requires_offline_replay():
    with pytest.raises(ValueError, match="only supported"):
        train._rollout_data_prefetch_enabled(_args(load_debug_rollout_data=None))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"offload_rollout": True}, "offload-rollout"),
        ({"rollout_global_dataset": True}, "global dataset"),
        ({"eval_interval": 10}, "eval-interval"),
    ],
)
def test_prefetch_rejects_incompatible_modes(override, message):
    with pytest.raises(ValueError, match=message):
        train._rollout_data_prefetch_enabled(_args(**override))


def test_prefetch_is_opt_in_for_standard_training():
    assert not train._rollout_data_prefetch_enabled(
        SimpleNamespace(prefetch_rollout_data=False)
    )
