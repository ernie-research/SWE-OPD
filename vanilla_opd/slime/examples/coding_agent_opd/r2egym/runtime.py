"""Runtime role: async wrapper over R2E-Gym's ``DockerRuntime`` (RLE backend).

R2E-Gym's ``DockerRuntime`` (and its ``DockerRuntimeMap`` subclasses:
``SWEBenchProDocker``, ``R2EDocker``, ...) is fully synchronous and blocks on
construction (``start_container`` + ``setup_env`` boot the pod). slime's rollout
is asyncio, and RLE calls are HTTP-I/O-bound, so we run every blocking call on a
**dedicated** ``ThreadPoolExecutor`` (not the default one, whose
``min(32, cpu+4)`` cap would throttle high-concurrency rollout).

``AsyncRuntime`` is the thin part slime needs for the **no-cheat grading pod**:
a fresh Runtime, ``apply_patch(model_diff)`` then ``calculate_reward()`` (which
resets tests + applies the *test* patch, never the model patch). The
interactive rollout pod is wrapped by :class:`~.env.AsyncRepoEnv`, which shares
this module's ``POOL``.

We import ``DockerRuntimeMap`` / ``SWEBenchDocker`` verbatim; the reward zoo and
all pod plumbing live inside the vendored classes.
"""

from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# Importing this package first runs the sys.path shim + RLE env defaults.
from . import VENDORED_R2EGYM_SRC  # noqa: F401  (import side effect: shim)

from r2egym.agenthub.runtime.docker import (  # noqa: E402
    DockerRuntimeMap,
    SWEBenchDocker,
    PodUnavailableError,
)

# --------------------------------------------------------------------------
# Dedicated thread pool. RLE calls are HTTP-bound, so we size the pool by the
# number of pods that may be mid-call at once, not by CPU count. Each live env
# has at most one outstanding blocking call (steps are sequential within a
# trajectory), plus grading spawns a second short-lived pod, so we give ample
# headroom over the boot-throttle (SWE_BOOT_CONCURRENCY). Overridable directly.
# --------------------------------------------------------------------------
_BOOT_CONCURRENCY = int(os.environ.get("SWE_BOOT_CONCURRENCY", "16"))
POOL_WORKERS = int(os.environ.get("SWE_RUNTIME_POOL_WORKERS", "0") or 0) or max(
    32, _BOOT_CONCURRENCY * 4
)

POOL = ThreadPoolExecutor(
    max_workers=POOL_WORKERS, thread_name_prefix="r2egym-runtime"
)

# Command passed to every Runtime constructor; matches RepoEnv._get_docker_runtime.
_RUNTIME_COMMAND = ["/bin/bash", "-l"]


async def run_in_pool(fn, /, *args, **kwargs):
    """Await a blocking ``fn(*args, **kwargs)`` on the dedicated pool."""
    loop = asyncio.get_running_loop()
    if kwargs:
        from functools import partial

        fn = partial(fn, **kwargs)
    return await loop.run_in_executor(POOL, fn, *args)


def _make_runtime(
    ds: dict,
    *,
    backend: str,
    rle_env_session_id: str | None,
    pod_cpu: int | str | None,
    pod_memory: int | str | None,
) -> Any:
    """Construct the right ``DockerRuntime`` subclass for ``ds['swe_type']``.

    Mirrors ``RepoEnv._get_docker_runtime``: unknown / missing ``swe_type``
    falls back to ``SWEBenchDocker``. Construction boots the pod, so callers
    must run this inside :data:`POOL`.
    """
    swe_type = ds.get("swe_type")
    cls = DockerRuntimeMap.get(swe_type, SWEBenchDocker)
    return cls(
        ds=ds,
        command=_RUNTIME_COMMAND,
        backend=backend,
        rle_env_session_id=rle_env_session_id,
        pod_cpu=pod_cpu,
        pod_memory=pod_memory,
    )


class AsyncRuntime:
    """Async facade over a single vendored ``DockerRuntime`` instance.

    Used for the fresh grading pod. The interactive pod goes through
    :class:`~.env.AsyncRepoEnv` (which owns a ``RepoEnv``); both share ``POOL``.
    """

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime

    @property
    def runtime(self) -> Any:
        return self._runtime

    @classmethod
    async def create(
        cls,
        ds: dict,
        *,
        backend: str = "rle",
        rle_session_id: str | None = None,
        pod_cpu: int | str | None = None,
        pod_memory: int | str | None = None,
    ) -> "AsyncRuntime":
        """Boot a fresh pod and return the wrapper (blocks in :data:`POOL`)."""
        runtime = await run_in_pool(
            _make_runtime,
            ds,
            backend=backend,
            rle_env_session_id=rle_session_id,
            pod_cpu=pod_cpu,
            pod_memory=pod_memory,
        )
        return cls(runtime)

    async def run(
        self,
        code: str,
        *,
        timeout: int | None = None,
        args: str = "",
        workdir: str | None = None,
    ) -> tuple[str, str]:
        """Run a shell command; returns ``(output, error_code_str)``."""
        kwargs: dict[str, Any] = {"args": args, "workdir": workdir}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await run_in_pool(self._runtime.run, code, **kwargs)

    async def get_patch(self) -> str:
        """``git add -A && git diff --cached`` inside the pod."""
        return await run_in_pool(self._runtime.get_patch)

    async def apply_patch(self, patch: str) -> tuple[str, str]:
        """Apply ``patch`` (e.g. the model diff) via ``git apply``."""
        return await run_in_pool(self._runtime.apply_patch, patch)

    async def reset_tests(self) -> Any:
        """Restore test files to their pristine state."""
        return await run_in_pool(self._runtime.reset_tests)

    async def calculate_reward(
        self, *, get_test_output: bool = False, timeout: int | None = None
    ):
        """Grade the current pod state via the swe_type's ``_calculate_reward``.

        With ``get_test_output=True`` returns ``(reward, test_output)``; the
        vendored method resets tests + applies the dataset's *test* patch, so
        the model diff must already be applied on this pod.
        """
        kwargs: dict[str, Any] = {"get_test_output": get_test_output}
        if timeout is not None:
            kwargs["timeout"] = timeout
        return await run_in_pool(self._runtime._calculate_reward, **kwargs)

    async def close(self) -> None:
        """Tear down the pod (best-effort; never raises)."""
        try:
            await run_in_pool(self._runtime.close)
        except Exception:  # noqa: BLE001  (teardown must not mask real errors)
            pass


__all__ = [
    "POOL",
    "POOL_WORKERS",
    "run_in_pool",
    "AsyncRuntime",
    "PodUnavailableError",
]
