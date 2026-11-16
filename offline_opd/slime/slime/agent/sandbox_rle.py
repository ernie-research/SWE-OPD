"""RLE (dockerRuntime) sandbox backend for agent rollouts.

Implements ``slime.agent.sandbox.Sandbox`` on top of
``rle_open_sdk.sdk.Environment``. One RLE session (pod) per instance: created
lazily in ``__aenter__``, destroyed in ``__aexit__``.

Session IDs are appended to ``$RLE_SESSION_IDS_DIR/session_ids.jsonl`` so a
batch cleanup step (e.g. ``examples/coding_agent_opd/3rdparty/R2E-Gym/scripts/kill_rle_sessions.py``)
can reap them if the training process crashes.

Behaviour is deliberately narrow -- this class is only a Sandbox backend;
baidu-mirror env vars, image-prefix policies, and grading logic belong at the
example / launcher layer.

Reference implementation being ported from:
``examples/coding_agent_opd/3rdparty/R2E-Gym/src/r2egym/agenthub/runtime/docker.py`` (RLE branch).
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import shlex
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Awaitable, Callable

from slime.agent.sandbox import ExecResult, FileContent, _getenv, Sandbox
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class PodUnavailableError(RuntimeError):
    """Raised when the RLE session/pod is gone (evicted, deleted, OCI failure).

    Callers (e.g. ``generate.py``'s boot loop, ``exec_and_wait``'s polling) can
    catch this to trigger trajectory-level retry instead of failing the batch.
    """


# Substrings that indicate an infra-level "pod is gone" error (not a normal
# non-zero exit from a user command). Sourced from rllm docker.py:80-93.
_POD_UNAVAILABLE_PATTERNS = (
    "oci runtime exec failed",
    "failed to exec in container",
    "failed to start exec",
    "container not found",
    "container is not running",
    "pod not found",
    "cannot exec in a stopped container",
    "handshake status 404",
    "code\":404",
)


def _looks_like_pod_gone(msg: str) -> bool:
    if not msg:
        return False
    low = msg.lower()
    return any(p in low for p in _POD_UNAVAILABLE_PATTERNS)

class RLESandbox:
    """Async sandbox backed by ``rle_open_sdk.sdk.Environment``.

    The SDK is synchronous, so blocking calls are dispatched via
    ``asyncio.to_thread`` to keep the event loop free.

    Contract matches ``slime.agent.sandbox.Sandbox``. The optional ``idempotent``
    kwarg on :meth:`exec` mirrors ``E2BSandbox``: it currently has no effect
    (no outer retry layer), but the signature is kept so ``exec_and_wait`` and
    other callers can pass it without special-casing the backend.
    """

    # Env-var aliases (canonical first, legacy fallbacks after) so callers can
    # override without editing the class. Names match rllm's docker.py where
    # applicable so a single env-set can drive both stacks side-by-side.
    api_key_env = ("RLE_API_KEY",)
    tool_type_env = ("RLE_TOOL_TYPE",)
    env_kind_env = ("RLE_ENV",)
    session_timeout_env = ("RLE_SESSION_TIMEOUT",)
    session_ids_dir_env = ("RLE_SESSION_IDS_DIR",)
    pod_cpu_env = ("SLIME_AGENT_RLE_POD_CPU", "R2E_POD_CPU")
    pod_memory_env = ("SLIME_AGENT_RLE_POD_MEMORY", "R2E_POD_MEMORY")
    image_prefix_env = ("SLIME_AGENT_RLE_IMAGE_PREFIX", "docker_image_prefix")

    default_tool_type = "swe-pdc"
    default_env_kind = "test"
    default_session_timeout = 1800
    default_pod_cpu = 2
    default_pod_memory = 8
    # RLE session_exec buffer: 100MB (first/last 50MB retained on overflow).
    exec_buffer_size = 100_000_000

    def __init__(
        self,
        image: str,
        *,
        container_env: dict[str, str] | None = None,
        pod_cpu: int | None = None,
        pod_memory: int | None = None,
        session_timeout: int | None = None,
        tool_type: str | None = None,
        env_kind: str | None = None,
        api_key: str | None = None,
        image_prefix: str | None = None,
        setup: Callable[["RLESandbox"], Awaitable[None]] | None = None,
    ) -> None:
        self.image = image
        self.container_env = dict(container_env or {})
        self.pod_cpu = int(
            pod_cpu if pod_cpu is not None
            else _getenv(*self.pod_cpu_env, default=str(self.default_pod_cpu))
        )
        self.pod_memory = int(
            pod_memory if pod_memory is not None
            else _getenv(*self.pod_memory_env, default=str(self.default_pod_memory))
        )
        self.session_timeout = int(
            session_timeout if session_timeout is not None
            else _getenv(*self.session_timeout_env, default=str(self.default_session_timeout))
        )
        self.tool_type = tool_type or _getenv(*self.tool_type_env, default=self.default_tool_type)
        self.env_kind = env_kind or _getenv(*self.env_kind_env, default=self.default_env_kind)
        self.api_key = api_key if api_key is not None else _getenv(*self.api_key_env, default="")
        self.image_prefix = (
            image_prefix if image_prefix is not None
            else _getenv(*self.image_prefix_env, default="")
        )
        self._env: Any = None  # rle_open_sdk.sdk.Environment instance
        self.sandbox_id = ""  # RLE session_id after __aenter__
        self.pod_view_url = ""
        self._setup = setup

    @staticmethod
    def _resolve_image_path(image: str, prefix: str) -> str:
        """Mirror rllm docker.py:525-550 image-path munging.

        - Optional prefix compose (skip if already prefixed or if it's an
          iregistry/ccr path).
        - RLE lives inside the VPC: swap ``-pub.cnc`` -> ``-vpc.cnc``.
        - iregistry images use their full path; other registries have the
          domain stripped (SDK expects "org/name:tag" for those).
        """
        if (
            prefix
            and prefix not in image
            and "ccr" not in image
            and "registry.internal.example.com" not in image
        ):
            image = f"{prefix}/{image}"
        image = image.replace("-pub.cnc", "-vpc.cnc")
        if "iregistry" in image:
            return image
        if "com/" in image:
            return image.split("com/")[-1]
        return image

    def _record_session_id(self, session_id: str) -> None:
        """Append session_id to session_ids.jsonl for post-hoc batch cleanup."""
        session_ids_dir = _getenv(*self.session_ids_dir_env, default="./")
        try:
            os.makedirs(session_ids_dir, exist_ok=True)
            path = os.path.join(session_ids_dir, "session_ids.jsonl")
            with open(path, "a+", encoding="utf-8") as f:
                f.write(session_id + "\n")
        except Exception as e:
            logger.warning("[rle.sandbox] session_id record skipped: %s", e)

    async def __aenter__(self) -> "RLESandbox":
        if not self.api_key:
            raise RuntimeError(
                "RLE_API_KEY is not set. Export it (or pass api_key=...) "
                "before entering RLESandbox."
            )
        # rle_open_sdk logs at INFO for every RPC; silence to match rllm.
        logging.getLogger("rle_open_sdk.sdk").setLevel(logging.ERROR)
        logging.getLogger("rle_sdk_logger").setLevel(logging.ERROR)

        from rle_open_sdk.sdk import Environment  # local import: SDK is optional

        self._env = Environment(
            api_key=self.api_key,
            tool_type=self.tool_type,
            env=self.env_kind,
            log_level=logging.ERROR,
        )
        image_path = self._resolve_image_path(self.image, self.image_prefix)
        env_spec = {k: str(v) for k, v in self.container_env.items()}

        def _create():
            return self._env.create_image_session(
                image_path=image_path,
                container_env=env_spec,
                container_command="tail",
                container_args=["-f"],
                session_timeout=self.session_timeout,
                cpu_cores=self.pod_cpu,
                memory=self.pod_memory,
                timeout=1800,
            )

        try:
            result = await asyncio.to_thread(_create)
        except Exception as e:
            # Any failure to spawn a pod is treated as PodUnavailable so the
            # caller's boot retry loop reacts uniformly.
            raise PodUnavailableError(
                f"RLE create_image_session failed for {image_path}: {e}"
            ) from e

        if not isinstance(result, dict) or "session_id" not in result:
            raise PodUnavailableError(
                f"RLE create_image_session returned unexpected payload: {result!r}"
            )

        self.sandbox_id = str(result["session_id"])
        self.pod_view_url = str(result.get("pod_view_url", ""))
        self._record_session_id(self.sandbox_id)
        logger.info(
            "[rle.sandbox] session created: id=%s image=%s pod_view=%s",
            self.sandbox_id[:12],
            image_path,
            self.pod_view_url,
        )
        if self._setup is not None:
            # Preprocess/adapter hook: agent-adapter setup (shebang, deps,
            # tool install via add_commands, ...). Runs after the RLE session
            # exists so it can drive real ``exec`` / ``write_file`` RPCs. If
            # setup fails we tear the session down before propagating so we
            # don't leak a half-initialized pod.
            try:
                await self._setup(self)
            except BaseException:
                try:
                    await asyncio.to_thread(self._env.session_destroy, self.sandbox_id)
                except Exception as e:
                    logger.warning(
                        "[rle.sandbox] session_destroy after setup failure "
                        "(%s) also failed: %s",
                        self.sandbox_id[:12], e,
                    )
                self.sandbox_id = ""
                raise
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if not self.sandbox_id or self._env is None:
            return
        try:
            await asyncio.to_thread(self._env.session_destroy, self.sandbox_id)
        except Exception as e:
            logger.warning(
                "[rle.sandbox] session_destroy %s failed: %s",
                self.sandbox_id[:12], e,
            )

    # ------------------------------------------------------------------
    # Sandbox Protocol methods
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap_shell_cmd(
        cmd: str, *, user: str, env: dict[str, str] | None, timeout: int
    ) -> list[str]:
        """Build the argv passed to ``session_exec``.

        Mirrors rllm docker.py:1114-1121: everything is executed via
        ``timeout <n> /bin/sh -c "..."`` so the SDK's outer wait, our timer,
        and the shell's SIGTERM agree.

        ``env`` is exported inside the shell so it survives any ``runuser``
        transition. ``user != "root"`` triggers ``runuser -u <user> --``.
        """
        exports = ""
        if env:
            exports = "".join(
                f"export {k}={shlex.quote(str(v))}; " for k, v in env.items()
            )
        body = f"{exports}{cmd}"
        if user and user != "root":
            body = f"runuser -u {shlex.quote(user)} -- /bin/sh -c {shlex.quote(body)}"
        return ["timeout", str(timeout + 10), "/bin/sh", "-c", body]

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,  # kept for parity with E2BSandbox; unused
    ) -> ExecResult:
        del idempotent  # no outer retry layer; RLE SDK handles transport
        full_command = self._wrap_shell_cmd(cmd, user=user, env=env, timeout=timeout)

        def _do():
            return self._env.session_exec(
                self.sandbox_id,
                commands=full_command,
                timeout=timeout + 20,
                fail_fast=False,
                buffer_size=self.exec_buffer_size,
            )

        try:
            res = await asyncio.to_thread(_do)
        except Exception as e:
            msg = str(e)
            if _looks_like_pod_gone(msg):
                raise PodUnavailableError(
                    f"RLE session '{self.sandbox_id}' unavailable: {msg}"
                ) from e
            # Unknown SDK exception: also surface as PodUnavailable so the
            # trajectory retries rather than crashing the whole rollout.
            raise PodUnavailableError(
                f"RLE session_exec raised {type(e).__name__}: {msg}"
            ) from e

        if not isinstance(res, dict):
            raise PodUnavailableError(
                f"RLE session_exec returned {type(res).__name__} "
                f"(session={self.sandbox_id[:12]}, cmd={cmd[:80]!r})"
            )

        stdout = res.get("stdout", "") or ""
        stderr = res.get("stderr", "") or ""
        rc = res.get("return_code")
        if rc is None:
            raise PodUnavailableError(
                f"RLE session_exec missing return_code "
                f"(session={self.sandbox_id[:12]}, cmd={cmd[:80]!r})"
            )
        rc = int(rc)

        if _looks_like_pod_gone(stderr):
            raise PodUnavailableError(
                f"RLE session '{self.sandbox_id}' unavailable: {stderr[:400]}"
            )

        if check and rc != 0:
            raise RuntimeError(
                f"rle exec failed (exit={rc}): {cmd[:120]}\n{stderr[:400]}"
            )
        return rc, stdout, stderr

    async def write_file(
        self, sandbox_path: str, content: FileContent, *, user: str = "root"
    ) -> None:
        """Upload content to the sandbox.

        - ``bytes``/``str`` are staged to a host tempfile and then ``copyin``ed
          (RLE SDK only accepts a local path).
        - ``Path`` is passed straight through.
        - On ``user != "root"``, chown the destination after upload since
          ``copyin`` writes as root.

        Retries the transfer up to 3 times (mirrors rllm docker.py:1548-1573)
        before raising ``PodUnavailableError``.
        """
        cleanup: Path | None = None
        if isinstance(content, Path):
            src_path = str(content)
        elif isinstance(content, (bytes, str)):
            data = content.encode("utf-8") if isinstance(content, str) else content
            fd, tmp = tempfile.mkstemp(prefix="rle_upload_")
            with os.fdopen(fd, "wb") as fp:
                fp.write(data)
            src_path = tmp
            cleanup = Path(tmp)
        else:
            raise TypeError(f"unsupported content type: {type(content).__name__}")

        def _copyin():
            return self._env.copyin(
                session_id=self.sandbox_id,
                local_file=src_path,
                dest_path=sandbox_path,
                timeout=1800,
            )

        try:
            last_err: BaseException | None = None
            succeeded = False
            for attempt in range(1, 4):
                try:
                    res = await asyncio.to_thread(_copyin)
                    if isinstance(res, dict) and res.get("status") == "success":
                        succeeded = True
                        break
                    last_err = RuntimeError(f"copyin returned {res!r}")
                except Exception as e:
                    last_err = e
                if attempt < 3:
                    await asyncio.sleep(2 * attempt)
            if not succeeded:
                raise PodUnavailableError(
                    f"RLE copyin failed after 3 attempts "
                    f"(session={self.sandbox_id[:12]}, dest={sandbox_path}): {last_err}"
                ) from last_err

            if user and user != "root":
                rc, _, err = await self.exec(
                    f"chown {shlex.quote(user)}:{shlex.quote(user)} "
                    f"{shlex.quote(sandbox_path)}",
                    user="root",
                    timeout=30,
                )
                if rc != 0:
                    logger.warning(
                        "[rle.sandbox] chown after write_file failed: %s",
                        err[:200],
                    )
        finally:
            if cleanup is not None:
                try:
                    cleanup.unlink()
                except OSError:
                    pass

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        """Read file via RLE's native ``download`` API (dedicated FS RPC).

        Mirrors :class:`E2BSandbox.read_file` in shape: a single filesystem
        RPC, binary-safe internally, no shell dependency, and returns ``""``
        on any failure so callers (adapters, ``exec_and_wait`` marker polling)
        can treat empty as "not present" without distinguishing missing-file
        from transport-failure.

        Mechanics: ``Environment.download`` asks the RLE controller to
        ``tar+gzip`` the requested paths server-side, upload the archive to
        BOS (Baidu Object Storage), and return a presigned URL. We fetch that
        URL, untar the archive in-memory, and pull out the single file entry
        matching ``sandbox_path``'s basename.
        """
        del user  # download is session-scoped; SDK does not accept a user arg
        target = os.path.basename(sandbox_path.rstrip("/")) or sandbox_path

        def _fetch_and_extract() -> str:
            res = self._env.download(
                session_id=self.sandbox_id,
                paths=[sandbox_path],
                auto_extract=False,
                timeout=600,
            )
            if not isinstance(res, dict) or res.get("status") != "success":
                return ""
            data = res.get("data")
            if not isinstance(data, dict):
                return ""
            bos_url = data.get("bos_url")
            if not bos_url:
                return ""
            with urllib.request.urlopen(bos_url, timeout=120) as resp:
                body = resp.read()
            with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
                for m in tf.getmembers():
                    if not m.isfile():
                        continue
                    if m.name == target or m.name.endswith("/" + target):
                        fp = tf.extractfile(m)
                        if fp is None:
                            return ""
                        return fp.read().decode("utf-8", errors="replace")
            return ""

        try:
            return await asyncio.to_thread(_fetch_and_extract)
        except Exception as e:
            logger.debug(
                "[rle.sandbox] read_file(%s) failed: %s",
                sandbox_path, e,
            )
            return ""
