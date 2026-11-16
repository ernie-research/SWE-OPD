"""Pin the eval-alignment fixes for SWE-bench Multilingual / Pro.

Four measured divergences from the reference harness
(``other_infra/R2E-Gym-benchmark``) were closed; each is a one-line literal that a
future vendored-tree refresh would silently revert, so each gets a test:

1. the maven proxy ``100.66.74.54:8091`` was suspected to be unreachable from the
   RLE sandbox (26 of 43 java multilingual instances never completed a build, java
   resolve 9.3% vs 40-65% for the other six languages) and was made opt-in -- then
   an in-pod probe measured it and disproved that: there is no
   transparent egress at all and the proxy is the only working path, so the change
   was reverted and the tests below now pin the proxy ON. What the probe DID find is
   that the proxy drops ~6-9% of requests, so wagon transfer retries were added
   instead -- with only the property names found inside the pod's own wagon jar;
2. the per-command bash limit was 300s (fallback 120s) against the reference's
   900s, killing 95 commands across 64 of 300 multilingual instances;
5. a 0 reward whose grading log carried no parseable test results was
   indistinguishable from a measured failure -- the multilingual ``found=False``
   return now carries a marker and ``r2egym.env`` classifies it.

Everything is checked by reading source text, not by importing: the vendored
runtime needs ``docker`` / ``kubernetes`` / ``rle_open_sdk`` / ``swebench`` /
``unidiff``, none of which a bare checkout has, and the repo's own top-level
``docker/`` package shadows the pip SDK anyway (same reason
``test_prompt_workdir.py`` uses ``ast``). Every value under test is a literal.

Item 六 (``remove_git_info.sh``) has no test because it needed no change: all 300
multilingual rows use ``swebench-multilingal-rebuild`` images, and in BOTH trees
that branch of ``setup_env`` returns before any git cleanup. Verified against the
trajectories -- in all 53 instances where the model ran ``git log`` and got
hashes, HEAD already was ``base_commit``, and repos carry a single branch and no
tags, so there is no future git info to remove.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NUM_GPUS = 0

_OURS = REPO_ROOT / "examples" / "coding_agent_opd"
_VENDORED = _OURS / "3rdparty" / "R2E-Gym" / "src" / "r2egym"
_REFERENCE = REPO_ROOT / "other_infra" / "R2E-Gym-benchmark" / "src" / "r2egym"

_DOCKER_PY = _VENDORED / "agenthub" / "runtime" / "docker.py"
_EXEC_BASH = _VENDORED / "agenthub" / "tools" / "scale_swe" / "execute_bash.py"
_REF_EXEC_BASH = _REFERENCE / "agenthub" / "tools" / "scale_swe" / "execute_bash.py"
_ENV_PY = _OURS / "r2egym" / "env.py"
_EVAL_CHAT = _OURS / "eval" / "eval_chat.py"
_RUN_EVAL = _OURS / "eval" / "run_eval.sh"
_RUN_EVAL_FULL = _OURS / "eval" / "run_eval_full_fncall.sh"

_MAVEN_PROXY = "100.66.74.54"
_NO_RESULTS_MARKER = "[GRADING_NO_TEST_RESULTS]"


def _rle_session_body() -> str:
    """``_start_rle_session``'s source: from its ``def`` to the next ``def``."""
    text = _DOCKER_PY.read_text()
    start = text.index("    def _start_rle_session(")
    end = text.index("\n    def ", start + 1)
    return text[start:end]


def _code_only(body: str) -> str:
    """``body`` minus whole-line comments -- the diagnosis names the dead proxy IP."""
    return "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("#"))


# --------------------------------------------------------------------------
# item 一: the maven proxy stays ON (in-pod probe reverted the opt-in change)
# --------------------------------------------------------------------------
def test_rle_session_keeps_the_maven_proxy():
    """The proxy is required, not optional -- see the in-pod measurement:
    ``curl --noproxy '*'`` to maven central times out in 3/3 pods, the same URL
    through this proxy returns 200 in 3/3, and ``mvn dependency:get`` into an empty
    local repo succeeds only with the proxy env. Making it opt-in (an earlier
    version of this fix) removed the only working egress path."""
    body = _rle_session_body()
    assert _MAVEN_PROXY in _code_only(body)
    for var in ("JAVA_TOOL_OPTIONS", "MAVEN_OPTS"):
        assert f'"{var}": {var}' in body, f"{var} must be set unconditionally"


def test_maven_proxy_host_is_still_overridable():
    """``R2E_MAVEN_PROXY`` survived the revert as an override (not an opt-in), so a
    different proxy can be tried without editing the vendored tree."""
    body = _rle_session_body()
    assert 'os.environ.get("R2E_MAVEN_PROXY"' in body
    assert '"100.66.74.54:8091"' in body, "the default must remain the working proxy"


def test_both_jvm_flag_spellings_are_built():
    """The two vars differ on purpose: only JAVA_TOOL_OPTIONS quotes nonProxyHosts."""
    body = _rle_session_body()
    assert '-Dhttp.nonProxyHosts="{_no_proxy_hosts}"' in body
    assert "-Dhttp.nonProxyHosts={_no_proxy_hosts}" in body
    assert "-Dmaven.resolver.transport=wagon" in body


def test_wagon_transfer_retries_are_set():
    """The proxy is reachable but drops ~6-9% of requests, and a build aborts on the
    first transfer error, so retries are the actual java fix. Only property names
    that ``probe_java_proxy.py`` step F found inside the pod's own
    ``wagon-http*.jar`` may be used: a misspelled ``-D`` is accepted by the JVM and
    does nothing, which is the exact failure mode this audit exists to catch."""
    body = _rle_session_body()
    for flag in (
        "-Dmaven.wagon.http.retryHandler.class=standard",
        "-Dmaven.wagon.http.retryHandler.count=6",
        "-Dmaven.wagon.http.retryHandler.requestSentEnabled=true",
    ):
        assert flag in body, flag
    # both spellings must carry the retries, not just JAVA_TOOL_OPTIONS
    assert body.count("{_jvm_retry}") == 1
    assert "f\"{_jvm_proxy}-Dhttp.nonProxyHosts={_no_proxy_hosts}\"" in body


def test_unverified_wagon_property_is_not_shipped():
    """``httpconnectionManager.ttlSeconds`` was measured (step G variant
    ``retry_ttl``) but its name was NOT found in the pod's wagon jar, so it must stay
    out of the shipped flags."""
    assert "ttlSeconds" not in _code_only(_rle_session_body())


def test_k8s_path_is_untouched():
    """Only the RLE path is in scope; the k8s path is not the backend we run."""
    text = _DOCKER_PY.read_text()
    assert "100.64.80.249" in text


# --------------------------------------------------------------------------
# item 二: per-command bash timeout matches the reference
# --------------------------------------------------------------------------
def _default_timeout(path: Path) -> int:
    m = re.search(r"^DEFAULT_TIMEOUT = (\d+)", path.read_text(), re.M)
    assert m is not None, f"DEFAULT_TIMEOUT not found in {path}"
    return int(m.group(1))


def test_tool_default_timeout_matches_reference():
    assert _default_timeout(_EXEC_BASH) == _default_timeout(_REF_EXEC_BASH) == 900


def test_step_timeout_env_still_wins_over_the_default():
    """``DEFAULT_TIMEOUT`` is only a fallback: docker.py exports ``STEP_TIMEOUT``
    into the pod on every step, so the eval scripts' value is what actually caps
    a command. Both facts must hold or raising the default changes nothing."""
    assert 'os.environ.get("STEP_TIMEOUT")' in _EXEC_BASH.read_text()
    assert "export STEP_TIMEOUT={timeout};" in _DOCKER_PY.read_text()


def test_eval_scripts_ask_for_the_reference_timeout():
    assert 'SLIME_AGENT_TOOL_TIMEOUT_SEC="900"' in _RUN_EVAL_FULL.read_text()
    assert 'SLIME_AGENT_TOOL_TIMEOUT_SEC:-900' in _RUN_EVAL.read_text()


def test_training_default_is_not_dragged_along():
    """``utils/rollout.py``'s default also feeds ``_ROLLOUT_GUARD_SEC``
    (``max_steps_absolute * _TOOL_TIMEOUT_SEC + ...``), so raising it there would
    inflate the training wall-clock guard ~5x. Eval sets the env var instead."""
    assert 'SLIME_AGENT_TOOL_TIMEOUT_SEC", "180"' in (_OURS / "utils" / "rollout.py").read_text()


# --------------------------------------------------------------------------
# item 五: an unmeasured 0 is labelled (and persisted), not silently unresolved
# --------------------------------------------------------------------------
def test_multilingual_found_false_tags_its_log():
    """``SWEBenchMultilingualDocker._calculate_reward`` must mark the ``found=False``
    return; without the marker the log is byte-identical to a measured failure."""
    text = _DOCKER_PY.read_text()
    start = text.index("class SWEBenchMultilingualDocker(")
    end = text.index("\nclass ", start + 1)
    body = text[start:end]
    assert "if not found:" in body
    assert _NO_RESULTS_MARKER in body


def test_env_classifies_both_no_result_shapes():
    text = _ENV_PY.read_text()
    assert f'"{_NO_RESULTS_MARKER}": "no_parseable_test_results"' in text
    # swebench-pro's own pre-existing marker, emitted when /workspace/output.json
    # is missing or not valid JSON.
    assert '"[SWE_PRO_REWARD_PARSE_ERROR]": "pro_output_json_unparseable"' in text
    assert "[SWE_PRO_REWARD_PARSE_ERROR]" in _DOCKER_PY.read_text()


def test_no_results_is_reported_not_excluded():
    """The label must NOT feed the grading_error path: a build broken by the
    model's own patch produces the same unparseable log, so excluding these rows
    from the denominator would inflate the resolve rate."""
    env_text = _ENV_PY.read_text()
    infra = env_text.index("GRADING_INFRA_FAILURE_SIGNATURES = {")
    infra_end = env_text.index("}", infra)
    assert _NO_RESULTS_MARKER not in env_text[infra:infra_end]

    chat = _EVAL_CHAT.read_text()
    # status is still decided by grading_reason alone.
    assert 'status = "grading_error" if grading_reason is not None else' in chat
    assert "no_results_reason" not in chat[: chat.index('status = "grading_error"')]


def test_test_output_and_label_are_persisted():
    chat = _EVAL_CHAT.read_text()
    assert '"test_output": _trunc_middle(test_output, _TEST_OUTPUT_KEEP),' in chat
    assert '"no_test_results": no_results_reason,' in chat
    # dump + lean row + summary breakdown
    assert '"test_output": res.get("test_output") or "",' in chat
    assert '"no_test_results",' in chat[chat.index("_LEAN_KEYS = ("):]
    assert '"no_test_results_breakdown"' in chat


def test_trunc_middle_keeps_both_ends():
    """Head+tail: the invoked command is at the start, the error at the end."""
    src = _EVAL_CHAT.read_text()
    ns: dict = {}
    start = src.index("def _trunc_middle(")
    exec(src[start : src.index("\ndef ", start + 1)], ns)  # noqa: S102
    trunc = ns["_trunc_middle"]

    assert trunc(None, 4) == ""
    assert trunc("abcd", 4) == "abcd"  # <= 2*keep is kept whole
    assert trunc("a" * 8, 4) == "a" * 8
    out = trunc("H" * 5 + "M" * 3 + "T" * 5, 5)
    assert out.startswith("H" * 5) and out.endswith("T" * 5)
    assert "3 chars omitted" in out and "M" not in out
