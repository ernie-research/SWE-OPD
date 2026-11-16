import json

import pytest

from examples.coding_agent_opd.prepare_dataset import _build_row, _missing_field
from examples.coding_agent_opd.utils.rollout import _missing_ds_field


def _valid_yiyan_row() -> dict:
    return {
        "swe_type": "yiyan-sweimages-factory",
        "instance_id": "owner__repo-1",
        "docker_image": "registry.example/yiyan:owner-repo-1",
        "repo": "owner/repo",
        "base_commit": "abc123",
        "patch": "diff --git a/a.py b/a.py",
        "test_patch": "diff --git a/test_a.py b/test_a.py",
        "eval_script": "#!/bin/bash\necho OMNIGRIL_EXIT_CODE=0",
        "language": "Python",
        "problem_statement": "Fix the failing behavior.",
    }


def _assert_validators_agree(row: dict, expected: str | None) -> None:
    assert _missing_field(row) == expected
    assert _missing_ds_field(row) == expected


def test_yiyan_uses_problem_statement_without_swebench_test_lists():
    raw = _valid_yiyan_row()
    _assert_validators_agree(raw, None)

    row = _build_row(raw, check_run_scripts=True)

    assert row is not None
    assert row["prompt"] == raw["problem_statement"]
    assert row["label"] == raw["instance_id"]
    assert row["metadata"]["workdir"] == "/testbed"
    assert row["metadata"]["ds"]["problem_statement"] == raw["problem_statement"]
    assert "FAIL_TO_PASS" not in row["metadata"]["ds"]
    assert "PASS_TO_PASS" not in row["metadata"]["ds"]


def test_yiyan_unwraps_json_quoted_problem_statement():
    raw = _valid_yiyan_row()
    raw["problem_statement"] = json.dumps("Fix quoted prompt.")

    row = _build_row(raw, check_run_scripts=False)

    assert row is not None
    assert row["prompt"] == "Fix quoted prompt."
    assert row["metadata"]["ds"]["problem_statement"] == "Fix quoted prompt."


@pytest.mark.parametrize("missing", ["eval_script", "repo", "base_commit", "language"])
def test_yiyan_rejects_fields_required_by_runtime(missing: str):
    raw = _valid_yiyan_row()
    raw.pop(missing)
    _assert_validators_agree(raw, missing)
    assert _build_row(raw, check_run_scripts=False) is None


def test_existing_swebench_validation_is_unchanged():
    raw = {
        "instance_id": "owner__repo-2",
        "docker_image": "registry.example/swebench:repo-2",
        "test_patch": "diff --git a/test.py b/test.py",
        "FAIL_TO_PASS": ["test_failure"],
        "PASS_TO_PASS": ["test_regression"],
    }
    _assert_validators_agree(raw, None)
    raw.pop("FAIL_TO_PASS")
    _assert_validators_agree(raw, "fail_to_pass")


@pytest.mark.parametrize(
    ("empty_key", "empty_value"),
    [
        ("FAIL_TO_PASS", []),
        ("PASS_TO_PASS", []),
        ("FAIL_TO_PASS", "[]"),
        ("PASS_TO_PASS", "[]"),
    ],
)
def test_swebench_validation_keeps_empty_test_groups(empty_key: str, empty_value):
    raw = {
        "instance_id": "owner__repo-empty-tests",
        "docker_image": "registry.example/swebench:empty-tests",
        "test_patch": "diff --git a/test.py b/test.py",
        "FAIL_TO_PASS": ["test_failure"],
        "PASS_TO_PASS": ["test_regression"],
    }
    raw[empty_key] = empty_value

    _assert_validators_agree(raw, None)


def test_swebench_validation_still_rejects_absent_test_group():
    raw = {
        "instance_id": "owner__repo-missing-tests",
        "docker_image": "registry.example/swebench:missing-tests",
        "test_patch": "diff --git a/test.py b/test.py",
        "FAIL_TO_PASS": ["test_failure"],
    }

    _assert_validators_agree(raw, "pass_to_pass")


def test_existing_scale_swe_validation_is_unchanged():
    raw = {
        "swe_type": "scale-swe",
        "instance_id": "scale-1",
        "docker_image": "registry.example/scale:1",
        "workdir": "/workspace/repo",
        "f2p_script": "def test_failure(): ...",
        "fail_to_pass": ["test_failure"],
        "pass_to_pass": ["test_regression"],
    }
    _assert_validators_agree(raw, None)
    raw.pop("f2p_script")
    _assert_validators_agree(raw, "f2p_patch")
    raw["f2p_script"] = []
    _assert_validators_agree(raw, "f2p_patch")
