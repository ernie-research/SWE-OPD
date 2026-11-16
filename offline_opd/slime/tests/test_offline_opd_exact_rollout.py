import uuid

import pytest

from examples.coding_agent_opd.scripts.offline_opd import exact_rollout
from slime.utils.types import Sample


def _sample() -> Sample:
    return Sample(
        index=7,
        group_index=0,
        prompt="problem",
        metadata={"ds": {"problem_statement": "problem"}},
    )


@pytest.mark.asyncio
async def test_retry_uses_clean_sample_and_new_routing_session(monkeypatch):
    original = _sample()
    seen_objects = []
    seen_input_sessions = []

    async def fake_generate(_args, group, *, sampling_params, evaluation):
        sample = group[0]
        seen_objects.append(sample)
        seen_input_sessions.append(sample.session_id)
        sample.session_id = str(uuid.uuid4())
        sample.response = "stale response"
        sample.status = (
            Sample.Status.ABORTED if len(seen_objects) == 1 else Sample.Status.COMPLETED
        )
        if sample.status == Sample.Status.ABORTED:
            sample.metadata["abort_reason"] = "pod_unavailable:test"
        return group

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(exact_rollout, "generate_and_rm_group", fake_generate)
    monkeypatch.setattr(exact_rollout.asyncio, "sleep", no_sleep)

    result = await exact_rollout._generate_group_with_retry(
        object(), [original], {}, max_attempts=3
    )

    assert len(seen_objects) == 2
    assert seen_objects[0] is not seen_objects[1]
    assert seen_input_sessions == [None, None]
    assert seen_objects[0].session_id != seen_objects[1].session_id
    assert result[0].status == Sample.Status.COMPLETED
    # Retrying is isolated from the caller-owned dataset object (and thus from
    # standard online/partial-rollout semantics).
    assert original.status == Sample.Status.PENDING
    assert original.session_id is None
    assert original.response == ""
    assert "abort_reason" not in original.metadata


@pytest.mark.asyncio
async def test_retry_remains_bounded(monkeypatch):
    calls = 0

    async def always_abort(_args, group, *, sampling_params, evaluation):
        nonlocal calls
        calls += 1
        group[0].status = Sample.Status.ABORTED
        group[0].metadata["abort_reason"] = "pod_unavailable:test"
        return group

    async def no_sleep(_seconds):
        return None

    monkeypatch.setattr(exact_rollout, "generate_and_rm_group", always_abort)
    monkeypatch.setattr(exact_rollout.asyncio, "sleep", no_sleep)

    with pytest.raises(RuntimeError, match="exhausted 4 attempts"):
        await exact_rollout._generate_group_with_retry(
            object(), [_sample()], {}, max_attempts=4
        )
    assert calls == 4
