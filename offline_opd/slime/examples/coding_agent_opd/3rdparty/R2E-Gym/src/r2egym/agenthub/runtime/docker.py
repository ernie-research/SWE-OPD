import json
from time import sleep
import time
import inspect
import uuid
import tempfile
import docker
from docker.models.containers import Container

from r2egym.repo_analysis.execution_log_parser import parse_log_fn, decolor_dict_keys
from r2egym.agenthub.runtime.base import (
    ExecutionEnvironment,
)
import base64
import subprocess
import datetime
import hashlib
import posixpath
import shlex
import shutil
import uuid

import docker
import kubernetes
import tarfile
import io
import os
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.action import Action
import re
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS, SKIP_FILES, SKIP_FILES_NEW, CMD_TIMEOUT
import concurrent.futures
from pathlib import Path

from r2egym.agenthub.trajectory.swebench_utils import (
    make_test_spec,
    swebench_parse,
    TestSpec,
)
from r2egym.agenthub.utils.utils import get_logger
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.swesmith.utils import get_test_command

from kubernetes import client, config, watch

# For Kubernetes exec.
from kubernetes.stream import stream
import os
import unidiff
import logging
from rle_open_sdk.sdk import Environment
import r2egym
_r2egym_project_root = os.path.dirname(os.path.dirname(os.path.dirname(r2egym.__file__)))
_third_party_dir = os.path.join(_r2egym_project_root, "third_party")
if os.path.isdir(_third_party_dir) and _third_party_dir not in os.sys.path:
    os.sys.path.append(_third_party_dir)

# 屏蔽 rle_open_sdk 的冗余日志（只显示 ERROR 及以上级别）
logging.getLogger("rle_open_sdk.sdk").setLevel(logging.ERROR)
# 新版本 SDK 使用了独立的 logger 名称，也需要屏蔽
logging.getLogger("rle_sdk_logger").setLevel(logging.ERROR)

DEFAULT_NAMESPACE = os.getenv("R2E_GYM_NAMESPACE", "default")
DOCKER_PATH = "/opt/miniconda3/envs/testbed/bin:/opt/conda/envs/testbed/bin:/root/.venv/bin/:/testbed/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


class PodUnavailableError(Exception):
    """
    Exception raised when a Kubernetes Pod is unavailable (evicted, deleted, or OCI runtime failure).
    
    This exception should be raised to trigger trajectory-level retry in the training framework.
    It indicates a transient infrastructure error rather than a model/action error.
    """
    pass


# Patterns that indicate the Pod is unavailable and should trigger a retry.
# These are OCI runtime errors that occur when a container is evicted, deleted, or in a bad state.
POD_UNAVAILABLE_ERROR_PATTERNS = [
    "OCI runtime exec failed",           # Container runtime failure
    "failed to exec in container",        # Exec failed due to container issues
    "failed to start exec",               # Cannot start exec in container
    "container not found",                # Container was deleted
    "container is not running",           # Container stopped/crashed
    "pod not found",                       # Pod was deleted
    "pods \"",                             # Kubernetes Status body: pods "<name>" not found
    "reason\":\"NotFound",                 # Kubernetes Status body for deleted/missing pods
    "kind\":\"pods",                       # Kubernetes Status details for pod 404
    "code\":404",                          # Kubernetes Status code for missing pod
    "Handshake status 404 Not Found",      # Kubernetes exec/copy websocket 404
    "cannot exec in a stopped container", # Container stopped
]


def is_pod_unavailable_error(error_message: str) -> bool:
    """
    Check if an error message indicates a Pod unavailable condition.
    
    Args:
        error_message: The error message string to check.
        
    Returns:
        True if the error indicates the Pod is unavailable and a retry should be attempted.
    """
    if not error_message:
        return False
    error_lower = error_message.lower()
    if any(pattern.lower() in error_lower for pattern in POD_UNAVAILABLE_ERROR_PATTERNS):
        return True
    return "pods" in error_lower and "not found" in error_lower

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec

from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
import unidiff


##############################################################################
# Docker runtime
##############################################################################
class DockerRuntime(ExecutionEnvironment):
    """
    docker runtime is responsible for the interacting with the docker environment.
    In particular, it should allow for accomodating the features of the particualr docker envs used for r2e-edits
    - collect files
    - list files excluding test files etc
    """

    # Container-level proxy env vars break test execution for these exact SWE-bench images.
    _PROXY_CLEANUP_IMAGES = [
        "django__django-16502",
        "psf__requests-6028",
        "psf__requests-5414",
        "psf__requests-2931",
        "psf__requests-2317",
        "psf__requests-1921",
        "psf__requests-1766",
        "psf__requests-1724",
    ]

    def _get_run_env_prefix(self) -> str:
        if any(tag in self.docker_image for tag in self._PROXY_CLEANUP_IMAGES):
            return "unset HTTP_PROXY HTTPS_PROXY NO_PROXY http_proxy https_proxy no_proxy; "
        return ""

    def _get_safe_new_file_paths_from_test_patch(self) -> list[str]:
        import posixpath

        test_patch = self.ds.get("test_patch", "") or ""
        safe_paths = []
        for chunk in re.split(r"(?=^diff --git )", test_patch, flags=re.MULTILINE):
            if not chunk.startswith("diff --git "):
                continue
            if not re.search(r"^new file mode\b", chunk, flags=re.MULTILINE):
                continue

            new_path = None
            diff_header_path = None
            for line in chunk.splitlines():
                if line.startswith("diff --git "):
                    diff_header_path = self._extract_new_path_from_diff_header(line)
                elif line.startswith("+++ "):
                    plus_path = line[4:].strip()
                    if plus_path != "/dev/null":
                        new_path = self._strip_git_diff_path_prefix(plus_path)
                        break

            if not new_path:
                new_path = diff_header_path
            if not new_path:
                self.logger.warning(f"Could not parse new file path from test_patch chunk: {chunk[:200]!r}")
                continue

            rel_path = new_path.lstrip("/")
            if rel_path == "" or rel_path == "." or rel_path.startswith("../") or "/../" in rel_path:
                self.logger.warning(f"Skip unsafe test_patch new file path: {new_path!r}")
                continue
            safe_paths.append(posixpath.join(self.repo_path, rel_path))
        return safe_paths

    @staticmethod
    def _strip_git_diff_path_prefix(path: str) -> str:
        if path.startswith("b/"):
            return path[2:]
        if path.startswith('"b/') and path.endswith('"'):
            return path[3:-1]
        return path

    @classmethod
    def _extract_new_path_from_diff_header(cls, line: str) -> str | None:
        if " b/" not in line:
            return None
        return cls._strip_git_diff_path_prefix(line.rsplit(" b/", 1)[-1].strip())

    def _remove_new_files_from_test_patch(self, file_paths: list[str] | None = None):
        import shlex

        if file_paths is None:
            file_paths = self._get_safe_new_file_paths_from_test_patch()

        for target_path in file_paths:
            out, rc = self.run(f"rm -rf -- {shlex.quote(target_path)}", timeout=30)
            self.logger.info(f"Removed test_patch new file: {target_path} rc={rc} out={out[:100]}")

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        docker_path: str = None,
        rle_env_session_id: str = None,
        pod_cpu=None,
        pod_memory=None,
        **docker_kwargs,
    ):
        # check if ds is provided (required for all dockers moving forward)
        assert ds, f"Dataset not provided for docker image: {docker_image}"
        assert backend in ["docker", "kubernetes", "rle", "e2b"], f"Invalid backend: {backend}"
        # swebench specific setup
        self.ds = ds
        self.rle_env_session_id = rle_env_session_id
        self.backend = backend
        ds_image = self.ds.get("docker_image") or self.ds.get("image_name")
        if not ds_image:
            raise ValueError(f"No docker image found in ds: {self.ds}")
        
        # refer to: https://github.com/SWE-agent/SWE-ReX/issues/263
        # 不同的镜像类型使用的默认 entrypoint shell type 不太一样，所以需要弄成可配置的类型
        self.default_entrypoint_shell_type = "/bin/sh"
        self.docker_path = docker_path or DOCKER_PATH
        self.max_output_chars = int(os.environ.get("MAX_OUTPUT_CHARS", "16000"))
        self.pod_cpu = pod_cpu
        self.pod_memory = pod_memory

        self.docker_image = ds_image if not docker_image else docker_image
        
        # set runtime params
        self.repo_path = repo_path
        self.alt_path = alt_path
        self.command = command
        if "repo" in self.ds:
            self.repo_name = self.ds["repo"]
        elif "repo_name" in self.ds:
            self.repo_name = self.ds["repo_name"]
        else:
            raise ValueError(f"No repository name found in ds: {self.ds}")

        self.docker_kwargs = docker_kwargs
        if logger is None:
            if self.backend == "docker":
                logger_name = "DockerRuntime"
            elif self.backend == "kubernetes":
                logger_name = "KubernetesRuntime"
            elif self.backend == "rle":
                logger_name = "RLERuntime"
            elif self.backend == "e2b":
                logger_name = "E2BRuntime"
            else:
                raise ValueError(f"Invalid backend: {self.backend}")
            self.logger = get_logger(logger_name)  # Pass the module name for clarity
        else:
            self.logger = logger

        if self.backend == "docker":
            self.client = docker.from_env(timeout=120)
        elif self.backend == "kubernetes":
            # Try in-cluster config first, fallback to kubeconfig
            config.load_kube_config()
            # try:
            #     config.load_incluster_config()
            # except Exception:
            #     config.load_kube_config()
            self.client = client.CoreV1Api()
        elif self.backend == "rle":
            # RLE相关初始化配置
            self.rle_api_key = os.getenv("RLE_API_KEY", "")
            # self.rle_tool_type = "sweagent"  # cce集群
            self.rle_tool_type = os.getenv("RLE_TOOL_TYPE", "swe-pdc")
            # 使用线下环境资源的话对应使用test
            # 使用线上环境资源的话对应使用prod
            self.rle_env = os.getenv("RLE_ENV", "test")
            # 初始化环境sdk类（设置 log_level=ERROR 屏蔽 INFO 日志）
            self.env = Environment(api_key=self.rle_api_key, tool_type=self.rle_tool_type, env=self.rle_env, log_level=logging.ERROR)
        elif self.backend == "e2b":
            from e2b import Sandbox as E2BSandbox
            self.E2BSandbox = E2BSandbox
            self.e2b_api_key = os.getenv("E2B_API_KEY", "")
            self.e2b_proxy = (
                os.getenv("E2B_PROXY")
                or os.getenv("https_proxy")
                or os.getenv("HTTPS_PROXY")
                or os.getenv("http_proxy")
                or os.getenv("HTTP_PROXY")
                or None
            )
            self.e2b_sandbox_timeout = int(os.getenv("E2B_SANDBOX_TIMEOUT", "10800"))
            self.e2b_metadata = os.getenv("E2B_METADATA", "").strip()
            self.e2b_template_id = self.ds.get("e2b_template_id") or self.docker_image
            self.e2b_sandbox = None
            self.e2b_sandbox_id = None

        # Start the container
        self.container = None
        self.container_name = self._get_container_name(self.docker_image)
        if self.backend == "kubernetes":
            # Generate a random UUID and truncate to 30 characters
            self.container_name = os.getenv("K8S_NAME_PREFIX", "") + str(uuid.uuid4())
            self.logger.info(f"🐙 Kubernetes mode: Generated pod name: {self.container_name}")
        elif self.backend == "e2b":
            self.container_name = self.e2b_template_id
        
        self.logger.info(f"🚀 Starting container/pod with image: {self.docker_image}")
        self.logger.info(f"🏷️  Container/Pod name: {self.container_name}")
        
        try:
            self.start_container(
                self.docker_image, command, self.container_name, **docker_kwargs
            )
            self.logger.info("✅ Container/Pod started successfully")
        except Exception as e:
            self.logger.error(f"❌ Failed to start container/pod: {e}")
            import traceback
            self.logger.error(f"❌ Exception details: {traceback.format_exc()}")
            raise

        # Initialize the environment. If setup fails after the pod/container has
        # been created, this constructor will not return and the outer env cannot
        # clean it up, so cleanup must happen here.
        try:
            self.setup_env()
        except Exception as e:
            self.logger.error(f"Environment setup failed, cleaning up container/pod: {repr(e)}")
            try:
                self.stop_container()
                self.logger.info("Cleanup completed after setup failure")
            except Exception as cleanup_error:
                self.logger.error(f"Cleanup after setup failure failed: {cleanup_error}")
            raise
        if self.backend == "kubernetes":
            self.logger.info("Kubernetes environment initialized")
        elif self.backend == "docker":
            self.logger.info("Docker environment initialized")
        elif self.backend == "rle":
            self.logger.info("RLE environment initialized")
        elif self.backend == "e2b":
            self.logger.info("E2B environment initialized")
        else:
            raise ValueError(f"Invalid backend: {self.backend}")

        self.logger.info("repo name: %s", self.repo_name)
        self.logger.info("Docker image: %s", self.docker_image)
        if self.backend == "docker":
            self.logger.info("Container ID: %s", self.container.id)
        elif self.backend == "kubernetes":
            # Assuming self.container is a V1Pod object after creation/retrieval
            pod_name = (
                self.container.metadata.name
                if self.container and self.container.metadata
                else "N/A"
            )
            self.logger.info("Pod Name: %s", pod_name)
        elif self.backend == "rle":
            pass
        elif self.backend == "e2b":
            self.logger.info("E2B Sandbox ID: %s", self.container)

    def get_task_instruction(self) -> str:
        # try getting the content inside of [ISSUE] [/ISSUE] using regex tags for ds['problem_statement'] else return ds['problem_statement']
        try:
            content = self.ds["problem_statement"]
            return re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception as e:
            return self.ds["problem_statement"]

    def _get_resource_value(self, configured_value, env_name: str, default: str) -> str:
        value = configured_value if configured_value is not None else os.getenv(env_name, default)
        return str(value)

    def _get_resource_int(self, configured_value, env_name: str, default: str) -> int:
        value = self._get_resource_value(configured_value, env_name, default).strip()
        lowered = value.lower()
        for suffix in ("gi", "g"):
            if lowered.endswith(suffix):
                value = value[: -len(suffix)]
                break
        return int(float(value))

    def _get_kubernetes_memory(self) -> str:
        value = self._get_resource_value(self.pod_memory, "R2E_POD_MEMORY", "8").strip()
        if re.search(r"[a-zA-Z]$", value):
            return value
        return f"{value}Gi"

    def get_added_files_from_patch(self, patch: str):
        """get added files from patch"""
        source_files = []
        for file in unidiff.PatchSet(patch):
            if file.source_file == "/dev/null" and file.target_file != "/dev/null":
                source_files.append(file.target_file[2:])
        return source_files

    def get_files_from_patch(self, patch: str):
        """get modified & removed files from patch"""
        source_files = []
        for file in unidiff.PatchSet(patch):
            if file.source_file != "/dev/null":
                source_files.append(file.source_file)
        source_files = [x[2:] for x in source_files if x.startswith("a/")]
        return source_files

    def _get_test_spec_with_cache(self, ds):
        """
        获取test spec，优先从缓存，缓存未命中时使用重试机制
        """
        try:
            # 尝试从缓存获取
            from r2egym.agenthub.utils.test_spec_cache import TestSpecCache
            cache = TestSpecCache()
        except ImportError:
            # 如果缓存模块不可用，直接使用重试机制
            self.logger.warning("Cache module not available, using retry mechanism only")
            return self._make_test_spec_with_retry(ds) 
        cached_spec = cache.get(ds)
        if cached_spec is not None:
            self.logger.info("✅ Test specification loaded from cache")
            return cached_spec
        
        # 缓存未命中，使用重试机制获取
        self.logger.info("Cache miss, fetching test specification with retry...")
        test_spec = self._make_test_spec_with_retry(ds)
        
        # 保存到缓存
        cache.set(ds, test_spec)
        self.logger.info("✅ Test specification cached for future use")
        
        return test_spec

    def _make_test_spec_with_retry(self, ds, max_retries=5):
        """
        Create test specification with retry mechanism.
        """
        import requests
        import urllib3
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Creating test specification (attempt {attempt+1}/{max_retries})...")
                test_spec = make_test_spec(ds)
                self.logger.info("✅ Test specification created successfully")
                return test_spec
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout,
                    urllib3.exceptions.ProtocolError,
                    ConnectionError) as e:
                if attempt < max_retries - 1:
                    delay = 5
                    self.logger.warning(f"⚠️  GitHub API call failed: {e}")
                    self.logger.warning(f"⏳ Retrying in {delay}s... (attempt {attempt+1}/{max_retries})")
                    time.sleep(delay)
                else:
                    self.logger.error(f"❌ Failed to create test specification after {max_retries} attempts")
                    self.logger.error(f"❌ Final error: {e}")
                    raise

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """Return name of container"""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-")
        image_name_sanitized = image_name_sanitized.replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def _start_rle_session(
        self, docker_image: str, command: str, **docker_kwargs
    ):
        """
        支持RLE的sdk调用
        """
        # rle 这里的 cpu 和 memory 都是数字
        pod_cpu = self._get_resource_int(self.pod_cpu, "R2E_POD_CPU", "2")
        pod_memory = self._get_resource_int(self.pod_memory, "R2E_POD_MEMORY", "8")
        JAVA_TOOL_OPTIONS = '-Dmaven.resolver.transport=wagon -Dhttp.proxyHost=100.66.74.54 -Dhttp.proxyPort=8091 -Dhttps.proxyHost=100.66.74.54 -Dhttps.proxyPort=8091 -Dhttp.nonProxyHosts="127.0.0.1|localhost|10.0.0.0/8|*.bcebos.com|*.baidu-int.com|inner.openapi.baidu.com|icode.baidu.com"'
        MAVEN_OPTS = '-Dmaven.resolver.transport=wagon -Dhttp.proxyHost=100.66.74.54 -Dhttp.proxyPort=8091 -Dhttps.proxyHost=100.66.74.54 -Dhttps.proxyPort=8091 -Dhttp.nonProxyHosts=127.0.0.1|localhost|10.0.0.0/8|*.bcebos.com|*.baidu-int.com|inner.openapi.baidu.com|icode.baidu.com'
        env_vars = {
            "UV_INDEX": "http://mirrors.baidubce.com/pypi/simple/",
            "PIP_INDEX_URL": "http://mirrors.baidubce.com/pypi/simple/",
            "PIP_TRUSTED_HOST": "mirrors.baidubce.com",  # pip 会自动识别这个环境变量
            "DJANGO_TEST_PROCESSES": str(pod_cpu),
            "JAVA_TOOL_OPTIONS": JAVA_TOOL_OPTIONS,
            "MAVEN_OPTS": MAVEN_OPTS,
            **docker_kwargs.get("environment", {})
        }
        env_vars.update({"no_proxy": "localdomain,localhost.localdomain,localhost,127.0.0.1,::1,10.0.0.0/8,100.64.0.0/10,192.168.0.0/16,172.16.0.0/12,*.lan,.baidu.com,.baidu-int.com,baidu.com,baidu-int.com,baidubce.com"})
        env_vars.update({"NO_PROXY": "localdomain,localhost.localdomain,localhost,127.0.0.1,::1,10.0.0.0/8,100.64.0.0/10,192.168.0.0/16,172.16.0.0/12,*.lan,.baidu.com,.baidu-int.com,baidu.com,baidu-int.com,baidubce.com"})

        # NEVER let http_proxy/https_proxy reach an RLE sandbox. The sandbox already
        # has transparent egress and cannot resolve the training host's proxy
        # (agent.baidu.com). Worse, RLE's own env_host agent inherits these vars and
        # tries to reach its control plane (100.64.80.148:8208) through the proxy, so
        # session_init fails outright:
        #   proxyconnect tcp: dial tcp: lookup agent.baidu.com ...: no such host
        # Verified: a session created with http_proxy set burns ~200s x 3 retries and
        # then fails to boot at all.
        for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy",
                   "ALL_PROXY", "proxy"):
            if env_vars.pop(_k, None) is not None:
                self.logger.warning(f"Dropped {_k} from RLE container env (breaks env_host boot)")

        env_spec = {k: str(v) for k, v in env_vars.items()}
        #container_command = ["/bin/sh", "-c"]
        docker_image_prefix = os.environ.get("docker_image_prefix", "").strip()
        if (
            docker_image_prefix
            and docker_image_prefix not in docker_image
            and "ccr" not in docker_image
            and "iregistry.baidu-int.com" not in docker_image
        ):
            docker_image = f"{docker_image_prefix}/{docker_image}"
        try:
            session_args = [command] if isinstance(command, str) else command

            # Backend-aware host swap: RLE 走内网域名 (-vpc.cnc), Kubernetes 走公网域名 (-pub.cnc).
            if self.backend == "rle":
                docker_image = docker_image.replace("-pub.cnc", "-vpc.cnc")
            elif self.backend == "kubernetes":
                docker_image = docker_image.replace("-vpc.cnc", "-pub.cnc")

            # Process docker image path for RLE.
            # Keep full path for iregistry.baidu-int.com (e.g. swe-rebench-v2 images) and
            # for any ccr-* registry: stripping the domain loses the CCR instance id, so
            # RLE resolves the relative path in its DEFAULT registry (ccr-2y3xupwh...).
            # That is why scale-swe images (ccr-29e67cht) failed with 33418
            # "template not ready" while swebench-verified (ccr-2y3xupwh) worked.
            # Verified: full path boots fine on both registries.
            # Strip registry domain otherwise (legacy behavior for yiyan / multilingual).
            if "iregistry" in docker_image or docker_image.split("/")[0].startswith("ccr-"):
                rle_image_path = docker_image
            elif "com/" in docker_image:
                rle_image_path = docker_image.split("com/")[-1]
            else:
                rle_image_path = docker_image

            # Inner retry around the (transient-prone) session-create RPC. Default
            # 1 == original single-shot behavior; set RLE_CREATE_RETRIES>1 to retry
            # transient create failures fast, below the coarse _boot_env reboot loop.
            create_retries = int(os.getenv("RLE_CREATE_RETRIES", "1"))
            create_image_session_result = None
            for create_attempt in range(1, create_retries + 1):
                try:
                    create_image_session_result = self.env.create_image_session(
                        image_path=rle_image_path,
                        container_env=env_spec,
                        container_command="tail",
                        container_args=["-f"],
                        session_timeout=os.getenv("RLE_SESSION_TIMEOUT", 180),
                        cpu_cores=pod_cpu,
                        memory=pod_memory,
                        timeout=1800,
                    )
                    break
                except Exception as ce:
                    self.logger.error(
                        f"create_image_session failed "
                        f"(attempt {create_attempt}/{create_retries}, image: {rle_image_path}): {ce}"
                    )
                    if create_attempt < create_retries:
                        time.sleep(2 * create_attempt)
                    else:
                        raise
            session_id, pod_view_url = create_image_session_result["session_id"], create_image_session_result["pod_view_url"]
            # to save session_id so we can destroy it
            session_ids_dir = os.getenv("RLE_SESSION_IDS_DIR", "./")
            os.makedirs(session_ids_dir, exist_ok=True)  # 确保目录存在
            session_ids_path = os.path.join(session_ids_dir, "session_ids.jsonl")
            with open(session_ids_path, "a+", encoding="utf-8") as f:
                f.write(session_id + "\n")

            self.container = session_id
        except Exception as e:
            self.logger.error(f"❌ Failed to start RLE session: {e}")
            raise e
        return session_id

    def _start_kubernetes_pod(
        self, docker_image: str, command: str, pod_name: str, **docker_kwargs
    ):
        """
        Starts or connects to a Kubernetes pod with the specified configuration.

        If a pod with the given name already exists, it attempts to connect to it.
        Otherwise, it creates a new pod based on the provided image, command,
        and environment variables, then waits for it to reach the 'Running' state.

        Args:
            docker_image: The Docker image to use for the pod's container.
            command: The command to run inside the container.
            pod_name: The desired name for the Kubernetes pod.
            **docker_kwargs: Additional keyword arguments. Currently used to extract
                             'environment' variables for the pod spec.

        Raises:
            kubernetes.client.ApiException: If there's an error interacting with the
                                           Kubernetes API (other than 404 Not Found
                                           when checking existence).
            RuntimeError: If the pod fails to reach the 'Running' state after creation.
        """
        self.logger.info(f"🔍 Checking if pod '{pod_name}' already exists...")
        not_found_error = None
        try:
            # Check if the pod already exists
            self.container = self.client.read_namespaced_pod(
                name=pod_name, namespace=DEFAULT_NAMESPACE, _request_timeout=60,
            )
            self.logger.info(f"Found existing Kubernetes pod: {pod_name}")
            self.logger.info(f"Pod status: {self.container.status.phase}")
            return
        except client.ApiException as e:
            not_found_error = e
            self.logger.info(f"Pod not found (status {e.status}), will create new one")

        if not_found_error.status != 404:
            self.logger.error(
                f"Error checking Kubernetes pod '{pod_name}' status: {not_found_error}. Check Kubernetes configuration and permissions."
            )
            raise not_found_error

        docker_image_prefix = os.environ.get("docker_image_prefix", "").strip()
        if docker_image_prefix and docker_image_prefix not in docker_image and "ccr" not in docker_image:
            docker_image = f"{docker_image_prefix}/{docker_image}"
        
        JAVA_TOOL_OPTIONS = '-Dmaven.resolver.transport=wagon -Dhttp.proxyHost=100.64.80.249 -Dhttp.proxyPort=8091 -Dhttps.proxyHost=100.64.80.249 -Dhttps.proxyPort=8091 -Dhttp.nonProxyHosts="127.0.0.1|localhost|10.0.0.0/8|*.bcebos.com|*.baidu-int.com|inner.openapi.baidu.com|icode.baidu.com"'
        MAVEN_OPTS = '-Dmaven.resolver.transport=wagon -Dhttp.proxyHost=100.64.80.249 -Dhttp.proxyPort=8091 -Dhttps.proxyHost=100.64.80.249 -Dhttps.proxyPort=8091 -Dhttp.nonProxyHosts=127.0.0.1|localhost|10.0.0.0/8|*.bcebos.com|*.baidu-int.com|inner.openapi.baidu.com|icode.baidu.com'
        env_vars = {
            "UV_INDEX": "http://mirrors.baidubce.com/pypi/simple/", 
            "PIP_INDEX_URL": "http://mirrors.baidubce.com/pypi/simple/",
            "PIP_TRUSTED_HOST": "mirrors.baidubce.com",  # pip 会自动识别这个环境变量
            "MAVEN_OPTS": MAVEN_OPTS,
            "JAVA_TOOL_OPTIONS": JAVA_TOOL_OPTIONS,
            **docker_kwargs.get("environment", {})
        }
        
        env_vars["CCE_HK_PROXY"] = "http://100.64.80.249:8091"
        env_vars["http_proxy"] = "http://100.64.80.249:8091"
        env_vars["https_proxy"] = "http://100.64.80.249:8091"
        env_vars["HTTP_PROXY"] = "http://100.64.80.249:8091"
        env_vars["HTTPS_PROXY"] = "http://100.64.80.249:8091"
        # disable proxy
        # env_vars.update({"http_proxy": "http://10.8.5.5:3128", "https_proxy": "http://10.8.5.5:3128", "no_proxy": "bcebos.com,baidu.com,baidu-int.com,127.0.0.1,localhost"})
        env_vars.update({"no_proxy": "localdomain,localhost.localdomain,localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12,*.lan,.baidu.com,.baidu-int.com,baidu.com,baidu-int.com,baidubce.com"})
        env_vars.update({"NO_PROXY": "localdomain,localhost.localdomain,localhost,127.0.0.1,::1,10.0.0.0/8,192.168.0.0/16,172.16.0.0/12,*.lan,.baidu.com,.baidu-int.com,baidu.com,baidu-int.com,baidubce.com"})        

        env_spec = [{"name": k, "value": str(v)} for k, v in env_vars.items()]

        pod_cpu = self._get_resource_value(self.pod_cpu, "R2E_POD_CPU", "2")
        pod_memory = self._get_kubernetes_memory()
        pod_body = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name},
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": pod_name,
                        "image": docker_image,
                        "command": [self.default_entrypoint_shell_type, "-c"],
                        "args": [command] if isinstance(command, str) else command,
                        "stdin": True,
                        "tty": True,
                        "env": env_spec,
                        "resources": {
                            "requests": {"cpu": pod_cpu, "memory": pod_memory},
                            "limits": {"cpu": pod_cpu, "memory": pod_memory},
                        },
                    }
                ],
                "imagePullSecrets": [{"name": os.getenv("imagePullSecret", "my-registry-secret-robot") }],
                # "nodeSelector": {"karpenter.sh/nodepool": "bigcpu-standby"},
                "tolerations": [
                    {
                        "key": "node.kubernetes.io/disk-pressure",
                        "operator": "Exists",
                        "effect": "NoExecute",
                        "tolerationSeconds": 10800
                    }
                ],
            },
        }

        # Create the Pod with retry logic & efficiently monitor with K8 Watch
        self.logger.info(f"Creating new Kubernetes pod '{pod_name}'...")
        self.logger.info(f"Pod image: {docker_image}")
        self.logger.info(f"Pod command: {command}")
        
        max_retries = 5
        backoff = 5  # seconds
        pod = None
        for attempt in range(1, max_retries + 1):
            self.logger.info(f"Pod creation attempt {attempt}/{max_retries}")
            try:
                pod = self.client.create_namespaced_pod(
                    namespace=DEFAULT_NAMESPACE, body=pod_body, _request_timeout=120,
                )
                self.logger.info(f"Pod '{pod_name}' created successfully on attempt {attempt}")
                break  # success
            except client.ApiException as e:
                # Retry on API-server throttling or transient errors
                if e.status in (409, 429, 500, 503):
                    self.logger.warning(
                        f"Transient Kubernetes error {e.status} while creating pod "
                        f"'{pod_name}' (attempt {attempt}/{max_retries}); "
                        f"retrying in {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                # Non-retryable error → propagate
                self.logger.error(f"Failed to create Kubernetes pod '{pod_name}': {e}")
                self.logger.error(f"Error details: {e.body if hasattr(e, 'body') else 'No additional details'}")
                raise
        else:
            raise RuntimeError(
                f"Exceeded retry limit ({max_retries}) while creating pod '{pod_name}'."
            )

        # Start monitoring pod status
        self.logger.info(f"Starting to monitor pod '{pod_name}' status...")
        try:
            rv = pod.metadata.resource_version
            self.logger.info(f"📍 Resource version: {rv}")
            w = watch.Watch()
            stream = w.stream(
                self.client.list_namespaced_pod,
                namespace=DEFAULT_NAMESPACE,
                field_selector=f"metadata.name={pod_name}",
                resource_version=rv,
                timeout_seconds=1200,  # 10 minutes timeout instead of 1 hour
            )
            
            start_time = time.time()
            self.logger.info(f"Started watching at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
            
            for event in stream:
                obj = event["object"]
                phase = obj.status.phase
                event_time = time.time() - start_time
                # self.logger.info(f"Event {event['type']} → pod.phase={phase} (after {event_time:.1f}s)")

                # Log pod conditions if available
                if obj.status.conditions:
                    for condition in obj.status.conditions:
                        if condition.status == "False":
                            self.logger.warning(f"Condition {condition.type}: {condition.message}")
                
                if time.time() - start_time > 1200:
                    w.stop()
                    raise RuntimeError(f"Kubernetes pod '{pod_name}' timed out after 1200 seconds.")
                # self.logger.info(f"Event {event['type']} → pod.phase={phase}")
                if phase == "Running":
                    self.logger.info(f"Kubernetes pod '{pod_name}' is Running after {event_time:.1f}s!")
                    w.stop()
                    break
                elif phase == "Pending":
                    # Log more details about why pod is pending
                    if obj.status.container_statuses:
                        for container_status in obj.status.container_statuses:
                            if container_status.state.waiting:
                                reason = container_status.state.waiting.reason
                                message = container_status.state.waiting.message
                                self.logger.info(f"Container waiting: {reason} - {message}")
                elif phase in ["Failed", "Succeeded", "Unknown"]:
                    w.stop()
                    # Log pod events for more context
                    self.logger.error(f"Pod entered terminal phase '{phase}'")
                    if obj.status.container_statuses:
                        for container_status in obj.status.container_statuses:
                            if container_status.state.terminated:
                                self.logger.error(f"Container terminated: {container_status.state.terminated.reason} - {container_status.state.terminated.message}")
                    raise RuntimeError(
                        f"Kubernetes pod '{pod_name}' entered terminal phase '{phase}'."
                    )
            self.container = pod
        except client.ApiException as create_error:
            self.logger.error(
                f"Failed to create Kubernetes pod '{pod_name}': {create_error}"
            )
            raise create_error
        except Exception as e:
            # Handle watch timeout or other errors
            self.logger.error(f"Error waiting for pod to start: {e}")
            # Check pod status directly as fallback
            try:
                pod_status = self.client.read_namespaced_pod(
                    name=pod_name, namespace=DEFAULT_NAMESPACE, _request_timeout=60,
                )
                if pod_status.status.phase == "Running":
                    self.logger.info(f"Pod '{pod_name}' is running (verified after watch error)")
                    self.container = pod_status
                else:
                    self.logger.warning(f"Pod '{pod_name}' is in state {pod_status.status.phase}")
                    raise RuntimeError(f"Pod '{pod_name}' failed to reach Running state: {pod_status.status.phase}")
            except Exception as status_error:
                self.logger.error(f"Failed to check pod status after watch error: {status_error}")
                raise RuntimeError(f"Failed to verify pod status: {status_error}")
    def reset_tests(self):
        """reset the f2p/p2p related test file and set the golden test patch"""
        all_files = self.get_files_from_patch(self.ds["test_patch"])
        commit_id = self.ds['base_commit']
        if len(all_files) == 0:
            self.logger.info("No test files to reset")
            return

        reset_command = (
            f'printf "%s\\n" {" ".join(all_files)} | '
            f'xargs -n1 -I{{}} git checkout {commit_id} -- "{{}}" 2>/dev/null'
        )
        self.run(reset_command)

    def start_container(
        self, docker_image: str, command: str, ctr_name: str, **docker_kwargs
    ):
        # Start or reuse a container
        self.logger.info(f"start_container called with backend: {self.backend}")
        self.logger.info(f"Docker image: {docker_image}")
        self.logger.info(f"Command: {command}")
        self.logger.info(f"Container name: {ctr_name}")
        
        try:
            if self.backend == "docker":
                containers = self.client.containers.list(
                    all=True, filters={"name": ctr_name}
                )
                if containers:
                    self.logger.info(f"Found existing container: {ctr_name}")
                    self.container = containers[0]
                    if self.container.status != "running":
                        self.logger.info("Starting existing container...")
                        self.container.start()
                        self.logger.info("Existing container started")
                    else:
                        self.logger.info("Container already running")
                else:
                    self.logger.info("Creating new Docker container...")
                    self.container = self.client.containers.run(
                        docker_image,
                        command,
                        name=ctr_name,
                        detach=True,
                        tty=True,
                        network_mode='host',
                        stdin_open=True,
                        environment={"http_proxy": "http://10.8.5.5:3128", "https_proxy": "http://10.8.5.5:3128", "no_proxy": "bcebos.com,baidu.com,baidu-int.com,127.0.0.1,localhost"},
                        **docker_kwargs,
                    )
                    self.logger.info("New Docker container created and started")
            elif self.backend == "kubernetes":
                self.logger.info("Using Kubernetes backend...")
                self._start_kubernetes_pod(
                    docker_image, command, ctr_name, **docker_kwargs
                )
                self.logger.info("Kubernetes pod creation process completed")
            elif self.backend == "rle":
                self.logger.info("Using RLE backend...")

                if self.rle_env_session_id is None:
                    self._start_rle_session(
                        docker_image, command, **docker_kwargs
                    )
                else:
                    self.container = self.rle_env_session_id
            elif self.backend == "e2b":
                self.logger.info("Using E2B backend...")
                self._start_e2b_sandbox(docker_image, **docker_kwargs)
            else:
                raise ValueError(f"Unknown backend: {self.backend}")
                
        except Exception as e:
            # 关键修改：使用logger而不是print，并抛出异常而不是静默返回
            self.logger.error(f"❌ Container start error: {repr(e)}")
            import traceback
            self.logger.error(f"❌ Full exception traceback: {traceback.format_exc()}")
            print("Container start error:", repr(e))
            # 尝试清理
            try:
                self.logger.info("🧹 Attempting cleanup after container start failure...")
                self.stop_container()
                self.logger.info("✅ Cleanup completed")
            except Exception as cleanup_error:
                self.logger.error(f"❌ Cleanup failed: {cleanup_error}")
            raise

    # Debian mirror reachable from inside the sandbox. Egress works there (a
    # transparent layer forwards to the real internet), but the images' default
    # source deb.debian.org is served from international Fastly nodes (AMS/NRT),
    # so pulling the ~9 MB Packages index routinely blows past a 120s timeout.
    # This intranet mirror serves the same content at ~7.5 MB/s.
    APT_MIRROR = "http://mirrors.baidubce.com"

    def ensure_cli_tools(self, tools=("curl", "wget", "jq"), timeout: int = 120) -> bool:
        """Make sure the given CLI tools exist in the container.

        Returns True if every tool is present afterwards. Never raises: a missing
        CLI tool degrades the agent's toolbox but must not fail container boot.
        """
        probe = "; ".join(f"command -v {t} >/dev/null || echo {t}" for t in tools)
        missing_out, _ = self.run(probe, timeout=20)
        missing = [t for t in tools if t in (missing_out or "").split()]
        if not missing:
            self.logger.info(f"CLI tools already present: {', '.join(tools)}")
            return True

        pkgs = " ".join(missing)
        self.logger.info(f"Installing missing CLI tools: {pkgs}")
        # Repoint apt at the fast mirror, keeping the suite from the image itself
        # (never assume bookworm) and only for Debian. The original sources are
        # saved to .orig and restored if the mirror does not work out.
        src = "/etc/apt/sources.list.d/debian.sources"
        key = "/usr/share/keyrings/debian-archive-keyring.gpg"
        fmt = ("Types: deb\\nURIs: %s/debian\\nSuites: %s %s-updates\\n"
               "Components: main\\nSigned-By: %s\\n\\n"
               "Types: deb\\nURIs: %s/debian-security\\nSuites: %s-security\\n"
               "Components: main\\nSigned-By: %s\\n")
        _out, code = self.run(
            ". /etc/os-release 2>/dev/null; C=$VERSION_CODENAME; "
            f'if [ "$ID" = debian ] && [ -n "$C" ] && [ -f {src} ]; then '
            f"cp -n {src} {src}.orig; "
            f"printf '{fmt}' \"{self.APT_MIRROR}\" \"$C\" \"$C\" \"{key}\" "
            f"\"{self.APT_MIRROR}\" \"$C\" \"{key}\" > {src}; fi; "
            f"if apt-get update -qq && apt-get install -y -qq {pkgs}; then exit 0; fi; "
            # mirror path failed -> restore the image's own sources and try once more
            f"[ -f {src}.orig ] && cp {src}.orig {src}; "
            f"(apt-get update -qq && apt-get install -y -qq {pkgs}) "
            f"|| yum install -y {pkgs} || apk add --no-cache {pkgs}",
            timeout=timeout,
        )
        if code != "0":
            self.logger.warning(f"Failed to install CLI tools {pkgs} (code={code})")
            return False
        self.logger.info(f"Installed CLI tools: {pkgs}")
        return True

    def install_python_package_with_retry(
        self,
        package_name: str,
        max_retries: int = 3,
        install_timeout: int = 300,
        check_timeout: int = 30,
        pip_install_commands: list = None,
        use_uv: bool = False,
        raise_on_failure: bool = False,
        skip_installed_check: bool = False,
    ) -> bool:
        """
        Install a Python package with retry logic using multiple pip install commands.
        
        Tries each source sequentially: first source with max_retries attempts, then second source with max_retries attempts, etc.

        Args:
            package_name: Name of the package to install
            max_retries: Number of retry attempts per source (not total attempts)
            install_timeout: Timeout for each installation attempt in seconds (default 300)
            check_timeout: Timeout for checking if package is installed in seconds
            pip_install_commands: List of pip install command templates. Use {package_name} as placeholder.
                                 If None, uses default commands with multiple sources.
            use_uv: If True, use 'uv pip install' instead of 'python -m pip install'
            raise_on_failure: If True, raises RuntimeError when installation fails after all retries (default False)
            skip_installed_check: If True, install even when the package is importable. Useful for version downgrades.

        Returns:
            bool: True if package was installed successfully, False otherwise

        Raises:
            RuntimeError: If installation fails after all retry attempts (when raise_on_failure=True)
        """
        PINNED_VERSIONS = {
            "chardet": "chardet<=6.0.0",
        }
        raw_package_name = package_name
        if package_name in PINNED_VERSIONS:
            package_name = PINNED_VERSIONS[package_name]
        import_name = re.split(r"[=<>!\[~]", package_name)[0]

        if self.backend in ("rle", "e2b", "kubernetes") and raw_package_name in PINNED_VERSIONS:
            self.run(f"python -m pip install '{package_name}' --break-system-packages")
        
        if pip_install_commands is None:
            if use_uv:
                pip_install_commands = [
                    "uv pip install '{package_name}'",
                    "uv pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com '{package_name}'",
                ]
            else:
                pip_install_commands = [
                    "python -m pip install '{package_name}'",
                    "python -m pip install -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com '{package_name}'",
                ]

        # Check if package is already installed unless the caller needs a version change.
        if not skip_installed_check:
            check_output, check_code = self.run(
                f"python -c \"import {import_name}; print('installed')\"",
                timeout=check_timeout
            )
            if "installed" in check_output and check_code == "0":
                self.logger.info(f"{package_name} package is already installed")
                return True

        # Try each source sequentially
        for cmd_idx, install_cmd_template in enumerate(pip_install_commands):
            install_cmd = install_cmd_template.format(package_name=package_name)
            
            # Extract source name for logging
            if "-i " in install_cmd:
                source_name = install_cmd.split("-i ")[1].split()[0]
            else:
                source_name = "default source"

            self.logger.info(f"Trying to install {package_name} from {source_name}")
            
            # Retry with current source
            for attempt in range(max_retries):
                self.logger.info(
                    f"Installing {package_name} (attempt {attempt + 1}/{max_retries}) from {source_name}"
                )
                output, code = self.run(install_cmd, timeout=install_timeout)
                
                if code == "0":
                    self.logger.info(f"{package_name} installed successfully from {source_name}")
                    return True
                else:
                    # Log the error output for debugging
                    self.logger.warning(f"Install attempt failed with exit code {code}. Error output: {output[-500:]}")

                if attempt < max_retries - 1:
                    time.sleep(2)
            
            self.logger.warning(f"Failed to install {package_name} from {source_name} after {max_retries} attempts")

        self.run(f"python -m pip install '{PINNED_VERSIONS.get('chardet', 'chardet')}' --break-system-packages")

        # All sources failed
        total_attempts = len(pip_install_commands) * max_retries
        error_msg = f"[INSTALL FAILED] package={package_name} attempts={total_attempts} pod={self.container_name} image={self.docker_image}"
        print(error_msg)
        
        if raise_on_failure:
            raise RuntimeError(error_msg)
        
        return False

    def _stop_kubernetes_pod(self):
        try:
            self.client.delete_namespaced_pod(
                name=self.container_name,
                namespace=DEFAULT_NAMESPACE,
                body=kubernetes.client.V1DeleteOptions(grace_period_seconds=0),
                _request_timeout=60,
            )

            w = watch.Watch()
            stream = w.stream(
                self.client.list_namespaced_pod,
                namespace=DEFAULT_NAMESPACE,
                field_selector=f"metadata.name={self.container_name}",
                timeout_seconds=60,  # 1 minute timeout instead of indefinite
            )

            deletion_confirmed = False
            for event in stream:
                if event["type"] == "DELETED":
                    self.logger.info(f"Kubernetes pod {self.container_name} deleted.")
                    deletion_confirmed = True
                    w.stop()
                    break
            
            # If watch times out without seeing deletion, verify pod is gone
            if not deletion_confirmed:
                try:
                    # Check if pod still exists
                    self.client.read_namespaced_pod(
                        name=self.container_name, namespace=DEFAULT_NAMESPACE
                    )
                    self.logger.warning(
                        f"Watch timed out but pod {self.container_name} still exists. Forcing deletion."
                    )
                    # Try deleting again with force
                    self.client.delete_namespaced_pod(
                        name=self.container_name,
                        namespace=DEFAULT_NAMESPACE,
                        body=kubernetes.client.V1DeleteOptions(
                            grace_period_seconds=0,
                            force=True
                        ),
                    )
                except kubernetes.client.rest.ApiException as e:
                    if e.status == 404:
                        # Pod is gone, which is what we want
                        self.logger.info(f"Confirmed pod {self.container_name} is deleted.")
                    else:
                        # Some other API error
                        self.logger.error(f"Error checking pod status after timeout: {e}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 404:
                # Pod already deleted, ignore
                self.logger.info(
                    f"Kubernetes pod '{self.container_name}' not found, likely already deleted."
                )
            else:
                # Log other K8s API errors during deletion
                self.logger.error(
                    f"Error deleting Kubernetes pod '{self.container_name}': {e}"
                )
                raise e  # Re-raise unexpected errors

    def stop_container(self):
        try:
            if self.backend == "e2b":
                sandbox = self.e2b_sandbox
                if sandbox is None and self.container and hasattr(self.E2BSandbox, "connect"):
                    try:
                        connect_kwargs = {"api_key": self.e2b_api_key} if self.e2b_api_key else {}
                        sandbox = self.E2BSandbox.connect(self.container, **connect_kwargs)
                    except TypeError:
                        sandbox = self.E2BSandbox.connect(self.container)
                    except Exception as connect_error:
                        self.logger.warning(f"[E2B] Failed to reconnect sandbox {self.container} for cleanup: {connect_error}")
                if sandbox is not None:
                    sandbox.kill()
                    self.logger.info(f"[E2B] Sandbox {self.container} killed.")
                elif self.container:
                    self.logger.warning(f"[E2B] stop_container called but no sandbox handle is available (sandbox_id: {self.container})")
                self.e2b_sandbox = None
                self.e2b_sandbox_id = None
                self.container = None
                return

            if self.container:
                if self.backend == "docker":
                    self.container.stop()
                    self.container.remove()
                elif self.backend == "kubernetes":
                    self._stop_kubernetes_pod()
                elif self.backend == "rle":
                    self.env.session_destroy(self.container)
        except Exception as e:
            if self.backend == "rle" and hasattr(self, 'container'):
                self.logger.error(f"Container stop/delete error (session_id: {self.container}): {repr(e)}")
            elif self.backend == "e2b":
                self.logger.error(f"[E2B] Sandbox kill error (sandbox_id: {self.container}): {repr(e)}")
            else:
                self.logger.error(f"Container stop/delete error: {repr(e)}")

    def setup_env(self):
        raise NotImplementedError

    def get_task_instruction(self) -> str:
        # try getting the content inside of [ISSUE] [/ISSUE] using regex tags for ds['problem_statement'] else return ds['problem_statement']
        try:
            content = self.ds["problem_statement"]
            return re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception as e:
            return self.ds["problem_statement"]

    def _run_rle(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir: str = "",
    ) -> tuple[str, str]:
        env_fix = self._get_run_env_prefix()
        command = f"export PATH={self.docker_path}:$PATH; export LANG=C.UTF-8; export LC_ALL=C.UTF-8; export PYTHONIOENCODING=utf-8; export STEP_TIMEOUT={timeout}; export MAX_OUTPUT_CHARS={self.max_output_chars}; {env_fix}"
        if workdir:
            # Use '&&' so that failure to change directory aborts the command
            command += f"cd {workdir}; "
        
        command += f"{code} {args}"
        # full_command = ["timeout", str(timeout), "/bin/sh", "-c", f"\"{command}\""]
        full_command = ["timeout", str(timeout + 10), "/bin/sh", "-c", f"{command}"]
        self.logger.info(f"run command: {full_command}")
        
        # 用于错误日志的命令简短表示（避免日志过长）
        cmd_short = f"{code[:200]}..." if len(code) > 200 else code
        
        exec_start_time = time.time()  # 记录开始时间（秒），用于区分外层超时和内层返回 124
        try:
            # 使用新版本 RLE SDK (0.1.4+) 的参数
            # buffer_size: 100MB (首尾各保留 50MB) - 防止输出过大被截断
            # fail_fast: False - 等待命令完成，不立即截断
            exec_res = self.env.session_exec(
                self.container, 
                commands=full_command, 
                timeout=timeout + 20,
                fail_fast=False,
                buffer_size=100000000
            )
            
            # 检查 RLE SDK 返回值有效性
            # SDK 失败时可能返回：None、空字符串 ""、或非字典类型
            if exec_res is None or exec_res == "":
                self.logger.error(f"RLE session_exec failed - returned None or empty string (session_id: {self.container}, command: {cmd_short})")
                return "RLE session_exec failed after retries", "-1"
            
            if not isinstance(exec_res, dict):
                self.logger.error(f"RLE session_exec returned invalid type: {type(exec_res)} (session_id: {self.container}, command: {cmd_short})")
                return f"RLE session_exec returned invalid type: {type(exec_res)}", "-1"
            
            # 安全提取返回值
            stdout = exec_res.get("stdout", "")
            stderr = exec_res.get("stderr", "")
            return_code = exec_res.get("return_code", None)
            
            # 检查 return_code 是否缺失（核心字段）
            # 注意：stdout 和 stderr 可能为空是正常的（某些命令无输出）
            if return_code is None:
                self.logger.error(f"RLE session_exec missing return_code field (session_id: {self.container}, command: {cmd_short})")
                return f"RLE session_exec missing return_code (stdout: {len(stdout)} bytes, stderr: {len(stderr)} bytes)", "-1"
            
            # 处理命令执行结果（对齐 kubernetes backend 的行为）
            output = stdout + "\n" + stderr if stderr else stdout
            
            # 检查命令超时 (对齐 k8s 后端逻辑)
            if return_code == 124:
                # 计算实际执行时间（秒），区分外层超时和内层返回 124
                actual_exec_time = time.time() - exec_start_time
                is_external_timeout = actual_exec_time >= timeout * 0.9  # 执行时间 >= 90% timeout 才认为是外层超时
                
                import uuid
                log_id = str(uuid.uuid4())[:8]
                
                # 把 code 中的换行符替换成空格，避免 Ray 日志多行前缀
                cmd_oneline = code[:150].replace('\n', ' ').replace('\r', '')
                
                if is_external_timeout:
                    # 真正的外层超时：返回超时消息
                    print(f"[{log_id}] TIMEOUT: {timeout}s actual={actual_exec_time:.0f}s | session_id={self.container} | image={self.docker_image} | cmd={cmd_oneline}")
                    return f"The command took too long to execute (>{timeout}s)", "-1"
                else:
                    # 内层返回了 124（如模型写的 timeout 10）：返回原始输出
                    print(f"[{log_id}] InnerTIMEOUT: actual={actual_exec_time:.0f}s<{timeout}s | session_id={self.container} | image={self.docker_image} | cmd={cmd_oneline[:100]}")
                    # 返回原始输出和 exit code，让模型看到真实的执行结果
                    return output, f"Error: Exit code {return_code}"
            
            # 检查命令执行失败
            # 注意：这里不输出 ERROR 日志，因为这是命令执行的正常结果
            # 只有 RLE SDK 层面的错误才输出 ERROR
            if return_code != 0:
                # self.logger.error(f"RLE exec Error: Exit code {return_code} (session_id: {self.container}, command: {cmd_short})\nError Message: {output}")
                return output, f"Error: Exit code {return_code}"
            
            # 成功：清理输出并返回
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(return_code)
            
        except Exception as e:
            import traceback
            error_str = str(e)
            self.logger.error(f"Exception during RLE exec (session_id: {self.container}, command: {cmd_short}): {repr(e)}\n{traceback.format_exc()}")
            # 直接抛出 PodUnavailableError，让框架层触发重试（对齐 k8s 后端逻辑）
            raise PodUnavailableError(f"RLE session '{self.container}' unavailable: {error_str}") from e

    def _run_kubernetes(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir: str = "",
    ) -> tuple[str, str]:
        """
        Kubernetes-specific method to execute code or commands in the pod, with a timeout.
        Mirrors the logic of the original Docker `run` method using Kubernetes API.
        """
        # Command includes 'timeout' and potentially 'cd <workdir> &&' from the main run method
        env_fix = self._get_run_env_prefix()
        command = f"export PATH={self.docker_path}:$PATH; export LANG=C.UTF-8; export LC_ALL=C.UTF-8; export PYTHONIOENCODING=utf-8; export STEP_TIMEOUT={timeout}; export MAX_OUTPUT_CHARS={self.max_output_chars}; {env_fix}"
        if workdir:
            # Use '&&' so that failure to change directory aborts the command
            command += f"cd {workdir}; "
        
        command += f"{code} {args}"
        full_command = ["timeout", str(timeout + 10), self.default_entrypoint_shell_type, "-c", f"{command}"]
        self.logger.info(f"run command: {full_command}")
        exec_start_time = time.time()  # 记录开始时间（秒），用于区分外层超时和内层返回 124
        try:
            # Define the exec function call within a lambda for the executor
            def execute_command():
                resp = stream(
                    self.client.connect_get_namespaced_pod_exec,
                    self.container_name,
                    DEFAULT_NAMESPACE,
                    command=full_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,  # Match docker exec_run settings
                    _preload_content=False,  # Important for streaming
                )
                # Read until the command exits, accumulating each channel
                combined_chunks = []
                stdout_chunks = []
                stderr_chunks = []
                while resp.is_open():
                    resp.update(timeout=1)  # wait for data
                    if resp.peek_stdout():
                        chunk = resp.read_stdout()
                        stdout_chunks.append(chunk)
                        combined_chunks.append(chunk)
                    if resp.peek_stderr():
                        chunk = resp.read_stderr()
                        stderr_chunks.append(chunk)
                        combined_chunks.append(chunk)
                resp.close()
                exit_code = resp.returncode
                combined_output = "".join(combined_chunks)
                return combined_output, exit_code

            # Execute with an overall timeout slightly larger than the command's timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(execute_command)
                combined_output, exit_code = future.result(timeout=timeout + 20)

            # Process results - combined_output already preserves inter-leaved stdout/stderr
            output = combined_output

            if exit_code is None:  # Should not happen if command finished
                self.logger.error("Kubernetes exec: Exit code not found.")
                return output, "-1"  # Unknown error state

            if exit_code == 124:
                # 计算实际执行时间（秒），区分外层超时和内层返回 124
                actual_exec_time = time.time() - exec_start_time
                is_external_timeout = actual_exec_time >= timeout * 0.9  # 执行时间 >= 90% timeout 才认为是外层超时
                
                import uuid
                log_id = str(uuid.uuid4())[:8]
                
                # 把 cmd 中的换行符替换成空格，避免 Ray 日志多行前缀
                cmd_oneline = code[:150].replace('\n', ' ').replace('\r', '')
                
                if is_external_timeout:
                    # 真正的外层超时：返回超时消息
                    print(f"[{log_id}] TIMEOUT: {timeout}s actual={actual_exec_time:.0f}s | pod={self.container_name} | image={self.docker_image} | cmd={cmd_oneline}")
                    return f"The command took too long to execute (>{timeout}s)", "-1"
                else:
                    # 内层返回了 124（如模型写的 timeout 10）：返回原始输出
                    print(f"[{log_id}] InnerTIMEOUT: actual={actual_exec_time:.0f}s<{timeout}s | pod={self.container_name} | image={self.docker_image} | cmd={cmd_oneline[:100]}")
                    # 返回原始输出和 exit code，让模型看到真实的执行结果
                    return output, f"Error: Exit code {exit_code}"

            if exit_code != 0:
                # 只对框架工具文件的错误打印 ERROR（如 SyntaxError），模型行为错误静默返回
                tool_files = ['/usr/local/bin/execute_bash', '/usr/local/bin/file_editor', '/usr/local/bin/search', '/usr/local/bin/finish']
                is_tool_error = any(tf in output for tf in tool_files)
                if is_tool_error:
                    print(f"[TOOL ERROR] exit={exit_code} pod={self.container_name} image={self.docker_image} err={output[:2000]}")
                # Return combined output and error code string
                return output, f"Error: Exit code {exit_code}"

            # Remove ANSI escape codes and \r characters from the combined output
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(exit_code)
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Kubernetes exec Overall Timeout: {timeout + 5}s")
            return f"The command took too long to execute (>{timeout}s)", "-1"
        except client.ApiException as e:
            error_str = str(e)
            # Check if this is a 404 Not Found error (Pod was deleted/evicted)
            if e.status == 404:
                self.logger.error(f"[POD_UNAVAILABLE] pod={self.container_name} image={self.docker_image} reason=not_found(404)")
                raise PodUnavailableError(f"Pod '{self.container_name}' not found: {repr(e)}") from e
            
            # Also check for Pod unavailable patterns in non-404 errors 
            # (e.g., 500 Internal Server Error with "container not found" from WebSocket handshake)
            if is_pod_unavailable_error(error_str):
                self.logger.error(f"[POD_UNAVAILABLE] pod={self.container_name} image={self.docker_image} reason=ApiException(status={e.status})")
                raise PodUnavailableError(f"Pod '{self.container_name}' unavailable: {error_str}") from e
            
            # Follow the function's established pattern: log the error and return an error tuple.
            self.logger.error(f"Kubernetes API Error during exec on pod '{self.container_name}': {e}")
            return f"Error executing command in pod: {repr(e)}", "-1"
        except Exception as e:
            # print trackback
            import traceback
            error_str = str(e)
            self.logger.error(f"Exception details: {traceback.format_exc()}")
            self.logger.error(f"Unexpected error during Kubernetes exec: {repr(e)}")
            
            # Check if this is a Pod unavailable error (OCI runtime failure, container eviction, etc.)
            # These errors indicate infrastructure issues that warrant a trajectory-level retry.
            if is_pod_unavailable_error(error_str):
                self.logger.error(f"[POD_UNAVAILABLE] pod={self.container_name} image={self.docker_image} reason=Exception")
                raise PodUnavailableError(f"Pod '{self.container_name}' unavailable: {error_str}") from e
            
            return f"Error: {repr(e)}", "-1"

    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir=None,
        type: str = None,
    ) -> tuple[str, str]:
        """
        General method to execute code or commands in the container, with a timeout.

        :param code: The code or command to execute.
        :param args: Arguments to pass to the code/script.
        :param workdir: The working directory inside the container (optional).
        :return: A tuple containing (output, error_message). If no error, error_message is the exit code (str).
        """
        exec_code = code
        exec_workdir = self.repo_path if workdir is None else workdir

        if self.backend == "kubernetes":
            return self._run_kubernetes(exec_code, timeout, args, workdir=exec_workdir)
        elif self.backend == "rle":
            return self._run_rle(exec_code, timeout, args, workdir=exec_workdir)
        elif self.backend == "e2b":
            return self._run_e2b(exec_code, timeout, args, workdir=exec_workdir)
        env_fix = self._get_run_env_prefix()
        command = f"export PATH={self.docker_path}:$PATH; export LANG=C.UTF-8; export LC_ALL=C.UTF-8; export PYTHONIOENCODING=utf-8; export STEP_TIMEOUT={timeout}; export MAX_OUTPUT_CHARS={self.max_output_chars}; {env_fix}{exec_code} {args}"
        full_command = ["timeout", str(timeout + 10), "/bin/sh", "-c", command]
        self.logger.info(f"run command: {full_command}")
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Notice we do NOT set tty=True here
                future = executor.submit(
                    self.container.exec_run,
                    cmd=full_command,
                    workdir=exec_workdir,
                    stdout=True,
                    stderr=True
                )
                exec_result = future.result(timeout=timeout + 20)

            # Retrieve output and exit code
            output = exec_result.output.decode("utf-8", errors="replace")
            # self.logger.info(f"output ->\n{output}")
            error_code = exec_result.exit_code

            if error_code == 124:
                self.logger.error(f"Internal Timeout: {timeout}s")
                return f"The command took too long to execute (>{timeout}s)", "-1"

            if error_code != 0:
                self.logger.error(
                    f"Error: Exit code {error_code} \nError Message: {output}"
                )
                return output, f"Error: Exit code {error_code}"

            # Remove ANSI escape codes and \r characters
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(error_code)

        ## timeout
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Timeout: {timeout}s")
            return f"The command took too long to execute (>{timeout}s)", "-1"

        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(
        self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir=None
    ) -> tuple[str, str]:
        command = f"timeout {timeout} {code} {args}"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Set demux=True to get separate stdout and stderr streams
                future = executor.submit(
                    self.container.exec_run,
                    cmd=command,
                    workdir=self.repo_path if workdir is None else workdir,
                    demux=True,  # This is the key change
                    environment={"PATH": self.docker_path},
                )
                exec_result = future.result(timeout=timeout + 5)

            # Unpack the result - when demux=True, output is a tuple of (stdout_data, stderr_data)
            output_data, error_data = exec_result.output
            error_code = exec_result.exit_code

            # Handle None cases and decode the outputs
            stdout = (
                output_data.decode("utf-8", errors="replace") if output_data else ""
            )
            stderr = error_data.decode("utf-8", errors="replace") if error_data else ""

            if error_code != 0:
                self.logger.error(
                    f"Error: Exit code {error_code} \nStdout Message: {stdout}, \nError Message: {stderr}"
                )
                return stdout, stderr, f"Error: Exit code {error_code}"

            return stdout, stderr, str(error_code)
        except Exception as e:
            return f"Error: {repr(e)}", f"Error: {repr(e)}", "-1"


    def _remote_join(self, root_path: str, relative_path: str) -> str:
        root_path = root_path.rstrip("/") or "/"
        if relative_path in ("", "."):
            return root_path
        parts = [part for part in relative_path.split(os.sep) if part and part != "."]
        if not parts:
            return root_path
        return posixpath.join(root_path, *parts)

    def _run_rle_mkdir(self, remote_dir: str):
        output, code = self.run(f"mkdir -p {shlex.quote(remote_dir)}", timeout=60, workdir="/")
        if code != "0":
            raise RuntimeError(f"mkdir failed for {remote_dir}: {output}")

    def _copy_url_to_container_rle_once(self, src_path: str, dest_path: str):
        copyin_res = self.env.copyin_url(
            session_id=self.container,
            file_url=src_path,
            dest_path=dest_path,
            timeout=1800,
        )
        if not isinstance(copyin_res, dict) or copyin_res.get("status") != "success":
            raise RuntimeError(f"copyin_url failed: {copyin_res}")

    def _copy_file_to_container_rle_once(self, src_path: str, dest_path: str):
        copyin_res = self.env.copyin(
            session_id=self.container,
            local_file=src_path,
            dest_path=dest_path,
            timeout=1800,
        )
        if not isinstance(copyin_res, dict) or copyin_res.get("status") != "success":
            raise RuntimeError(f"copyin failed: {copyin_res}")

    def _copy_directory_to_container_rle_once(self, src_path: str, dest_path: str):
        if not hasattr(self.env, "session_upload"):
            raise RuntimeError("RLE SDK does not provide session_upload")
        upload_res = self.env.session_upload(
            session_id=self.container,
            local_path=src_path,
            remote_path=dest_path,
        )
        if isinstance(upload_res, dict) and upload_res.get("status") not in (None, "success"):
            raise RuntimeError(f"session_upload failed: {upload_res}")

    def _copy_directory_to_container_rle_by_files(self, src_path: str, dest_path: str):
        dest_root = dest_path.rstrip("/") or "/"
        self._run_rle_mkdir(dest_root)

        for current_dir, _, filenames in os.walk(src_path):
            rel_dir = os.path.relpath(current_dir, src_path)
            remote_dir = self._remote_join(dest_root, rel_dir)
            self._run_rle_mkdir(remote_dir)
            for filename in filenames:
                local_file = os.path.join(current_dir, filename)
                remote_file = posixpath.join(remote_dir, filename)
                self._copy_file_to_container_rle_once(local_file, remote_file)

    def _copy_directory_to_container_rle(self, src_path: str, dest_path: str):
        dest_root = dest_path.rstrip("/") or "/"
        dest_parent = posixpath.dirname(dest_root)
        if dest_parent in ("", "/"):
            self.logger.info(
                f"Skipping RLE session_upload for top-level directory destination "
                f"(session_id: {self.container}, src: {src_path}, dest: {dest_path}); "
                "using per-file copy"
            )
            self._copy_directory_to_container_rle_by_files(src_path, dest_path)
            return

        try:
            self._copy_directory_to_container_rle_once(src_path, dest_path)
        except Exception as upload_error:
            self.logger.error(
                f"RLE session_upload failed for directory copy "
                f"(session_id: {self.container}, src: {src_path}, dest: {dest_path}): {upload_error}; "
                "falling back to per-file copy"
            )
            try:
                self._copy_directory_to_container_rle_by_files(src_path, dest_path)
            except Exception as fallback_error:
                raise RuntimeError(
                    f"session_upload failed: {upload_error}; per-file fallback failed: {fallback_error}"
                ) from fallback_error

    def _copy_to_container_rle(self, src_path: str, dest_path: str):
        """
        Copy a file, URL, or directory into the RLE session.
        """
        src_path = os.fspath(src_path)
        dest_path = os.fspath(dest_path)
        is_url = src_path.startswith(("http://", "https://"))

        if is_url:
            copy_kind = "URL"
        else:
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"RLE copy source does not exist: {src_path}")
            if os.path.isdir(src_path):
                copy_kind = "directory"
            elif os.path.isfile(src_path):
                copy_kind = "file"
            else:
                raise ValueError(f"RLE copy source is neither file nor directory: {src_path}")

        last_error = None
        copy_retries = int(os.getenv("RLE_COPY_RETRIES", "3"))  # default 3 == original behavior
        for attempt in range(1, copy_retries + 1):
            try:
                self.logger.info(
                    f"Copying {copy_kind} to RLE session "
                    f"(attempt {attempt}/{copy_retries}, session_id: {self.container}, src: {src_path}, dest: {dest_path})"
                )
                if is_url:
                    self._copy_url_to_container_rle_once(src_path, dest_path)
                elif copy_kind == "directory":
                    self._copy_directory_to_container_rle(src_path, dest_path)
                else:
                    self._copy_file_to_container_rle_once(src_path, dest_path)

                self.logger.info(f"Successfully copied {src_path} to {dest_path} in session {self.container}")
                return
            except Exception as e:
                last_error = e
                self.logger.error(
                    f"Failed to copy {copy_kind} to RLE session "
                    f"(attempt {attempt}/{copy_retries}, session_id: {self.container}, src: {src_path}, dest: {dest_path}): {e}"
                )
                if attempt < copy_retries:
                    time.sleep(2 * attempt)

        raise PodUnavailableError(
            f"RLE copy failed after {copy_retries} attempts for session '{self.container}' "
            f"(src: {src_path}, dest: {dest_path}, kind: {copy_kind}): {last_error}"
        ) from last_error

    def _start_e2b_sandbox(self, docker_image: str, **docker_kwargs):
        """
        Start an E2B sandbox.
        The dataset may provide e2b_template_id; otherwise docker_image is used
        as the E2B template id.
        """
        template_id = self.ds.get("e2b_template_id") or docker_image
        self.e2b_template_id = template_id
        self.logger.info(f"[E2B] Starting sandbox with template: {template_id}")
        try:
            kwargs = {}
            if self.e2b_proxy:
                kwargs["proxy"] = self.e2b_proxy
            if self.e2b_api_key:
                kwargs["api_key"] = self.e2b_api_key
            if self.e2b_metadata:
                if "=" in self.e2b_metadata:
                    metadata_key, metadata_value = self.e2b_metadata.split("=", 1)
                    metadata_key = metadata_key.strip()
                    metadata_value = metadata_value.strip()
                    if metadata_key and metadata_value:
                        kwargs["metadata"] = {metadata_key: metadata_value}
                    else:
                        self.logger.warning(f"[E2B] Ignoring invalid E2B_METADATA: {self.e2b_metadata!r}")
                else:
                    self.logger.warning(f"[E2B] Ignoring invalid E2B_METADATA: {self.e2b_metadata!r}")
            cpu_hint = self._get_resource_value(self.pod_cpu, "R2E_POD_CPU", "2")
            mem_hint = self._get_resource_value(self.pod_memory, "R2E_POD_MEMORY", "4")
            self.logger.info(
                f"[E2B] R2E_POD_CPU={cpu_hint}, R2E_POD_MEMORY={mem_hint}GB "
                f"(ignored - resource limits are baked into the E2B template)"
            )
            sandbox = self.E2BSandbox.create(
                template=template_id,
                timeout=self.e2b_sandbox_timeout,
                **kwargs,
            )
            self.e2b_sandbox = sandbox
            self.container = sandbox.sandbox_id
            self.e2b_sandbox_id = sandbox.sandbox_id
            self.logger.info(f"[E2B] Sandbox started: {self.container}")
            try:
                session_ids_dir = os.getenv("RLE_SESSION_IDS_DIR", "./")
                os.makedirs(session_ids_dir, exist_ok=True)
                e2b_ids_path = os.path.join(session_ids_dir, "e2b_sandbox_ids.jsonl")
                record = {
                    "sandbox_id": sandbox.sandbox_id,
                    "template_id": template_id,
                    "docker_image": self.docker_image,
                    "instance_id": self.ds.get("instance_id"),
                    "metadata": kwargs.get("metadata", {}),
                    "repo": self.repo_name,
                    "created_at": datetime.datetime.utcnow().isoformat() + "Z",
                }
                with open(e2b_ids_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as record_error:
                self.logger.warning(f"[E2B] Failed to save sandbox_id to file: {record_error}")
        except Exception as e:
            self.logger.error(f"[E2B] Failed to start sandbox: {e}")
            raise

    def _run_e2b(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir: str = "",
    ) -> tuple[str, str]:
        """
        Execute a command inside the E2B sandbox.
        Normal non-zero command exits are returned as error-code strings; only
        sandbox or infrastructure failures are raised as PodUnavailableError.
        """
        if self.e2b_sandbox is None:
            raise PodUnavailableError(f"E2B sandbox is not initialized (sandbox_id: {self.container})")

        import shlex

        env_fix = self._get_run_env_prefix()
        command = (
            f"export PATH={self.docker_path}:$PATH; "
            "export LANG=C.UTF-8; export LC_ALL=C.UTF-8; "
            "export PYTHONIOENCODING=utf-8; "
            f"export STEP_TIMEOUT={timeout}; export MAX_OUTPUT_CHARS={self.max_output_chars}; "
            f"{env_fix}"
        )
        if workdir:
            command += f"cd {shlex.quote(workdir)}; "
        command += f"{code} {args}"

        full_command = f"timeout {timeout + 10} /bin/sh -c {shlex.quote(command)}"
        self.logger.info(f"[E2B] run command: {full_command}")

        cmd_short = f"{code[:200]}..." if len(code) > 200 else code
        exec_start_time = time.time()

        try:
            result = self.e2b_sandbox.commands.run(
                full_command,
                timeout=timeout + 20,
                user="root",
            )
            stdout = getattr(result, "stdout", "") or ""
            stderr = getattr(result, "stderr", "") or ""
            exit_code = getattr(result, "exit_code", 0)
        except Exception as e:
            exit_code = getattr(e, "exit_code", None)
            stdout = getattr(e, "stdout", None)
            stderr = getattr(e, "stderr", None)
            if exit_code is None and hasattr(e, "result"):
                exit_code = getattr(e.result, "exit_code", None)
            if stdout is None and hasattr(e, "result"):
                stdout = getattr(e.result, "stdout", None)
            if stderr is None and hasattr(e, "result"):
                stderr = getattr(e.result, "stderr", None)
            stdout = stdout or ""
            stderr = stderr or ""

            if exit_code is None:
                import traceback
                self.logger.error(
                    f"[E2B] Unexpected exception during exec "
                    f"(sandbox_id: {self.container}, command: {cmd_short}): "
                    f"{repr(e)}\n{traceback.format_exc()}"
                )
                raise PodUnavailableError(
                    f"E2B sandbox '{self.container}' unavailable: {e}"
                ) from e

        output = stdout + ("\n" + stderr if stderr else "")

        if exit_code == 124:
            actual_exec_time = time.time() - exec_start_time
            is_external_timeout = actual_exec_time >= timeout * 0.9
            log_id = str(uuid.uuid4())[:8]
            cmd_oneline = code[:150].replace('\n', ' ').replace('\r', '')
            if is_external_timeout:
                print(f"[{log_id}] TIMEOUT: {timeout}s actual={actual_exec_time:.0f}s | sandbox_id={self.container} | image={self.docker_image} | cmd={cmd_oneline}")
                return f"The command took too long to execute (>{timeout}s)", "-1"
            print(f"[{log_id}] InnerTIMEOUT: actual={actual_exec_time:.0f}s<{timeout}s | sandbox_id={self.container} | image={self.docker_image} | cmd={cmd_oneline[:100]}")
            return output, f"Error: Exit code {exit_code}"

        if exit_code != 0:
            return output, f"Error: Exit code {exit_code}"

        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output, str(exit_code)

    def _copy_to_container_e2b(self, src_path: str, dest_path: str):
        """
        Copy a local file, local directory, or URL into the E2B sandbox.
        """
        if self.e2b_sandbox is None:
            raise PodUnavailableError(f"E2B sandbox is not initialized (sandbox_id: {self.container})")

        import shlex

        def ensure_remote_parent(path: str):
            parent = os.path.dirname(path.rstrip("/"))
            if parent:
                output, code = self._run_e2b(f"mkdir -p {shlex.quote(parent)}", timeout=30, workdir="/")
                if code != "0":
                    raise RuntimeError(f"Failed to create remote directory {parent}: {output}")

        def write_file(local_path: str, remote_path: str):
            ensure_remote_parent(remote_path)
            with open(local_path, "rb") as f:
                self.e2b_sandbox.files.write(remote_path, f)

        try:
            if src_path.startswith(("http://", "https://")):
                ensure_remote_parent(dest_path)
                download_cmd = (
                    "python -c "
                    + shlex.quote(
                        "import urllib.request; "
                        f"urllib.request.urlretrieve({src_path!r}, {dest_path!r})"
                    )
                )
                output, code = self._run_e2b(download_cmd, timeout=1800, workdir="/")
                if code != "0":
                    raise RuntimeError(f"Failed to download URL into E2B sandbox: {output}")
            elif os.path.isdir(src_path):
                for root, _, files in os.walk(src_path):
                    for filename in files:
                        local_file = os.path.join(root, filename)
                        rel_path = os.path.relpath(local_file, src_path)
                        remote_file = os.path.join(dest_path, rel_path)
                        write_file(local_file, remote_file)
            else:
                write_file(src_path, dest_path)
            self.logger.info(f"[E2B] Copied {src_path} -> sandbox:{dest_path}")
        except Exception as e:
            self.logger.error(f"[E2B] Failed to copy {src_path} -> {dest_path}: {e}")
            raise

    def _copy_to_container_kubernetes(self, src_path: str, dest_path: str):
        """
        Copy a file or directory from host into Kubernetes pod using tar over exec.
        """
        # Calculate destination directory and prepare in-memory tarball
        dest_dir = os.path.dirname(dest_path)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        tar_stream.seek(0)

        # Retry with a hard per-attempt timeout. Kubernetes exec streams can
        # occasionally stay open forever even after the remote tar process is gone.
        max_retries = 3
        copy_timeout_seconds = 60
        retry_delay = 2  # Initial delay in seconds
        for attempt in range(max_retries):
            resp = None
            try:
                deadline = time.time() + copy_timeout_seconds
                # Exec into pod to untar into the destination directory
                exec_command = ["tar", "xmf", "-", "-C", dest_dir]
                resp = stream(
                    self.client.connect_get_namespaced_pod_exec,
                    self.container_name,
                    DEFAULT_NAMESPACE,
                    command=exec_command,
                    stderr=True,
                    stdin=True,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                )
                # 分块写入 tar 文件内容，适用于大文件
                chunk_size = 1024 * 1024  # 1MB
                tar_stream.seek(0)
                while True:
                    if time.time() > deadline:
                        raise TimeoutError(f"kubernetes copy timed out while writing tar stream after {copy_timeout_seconds}s")
                    chunk = tar_stream.read(chunk_size)
                    if not chunk:
                        break
                    resp.write_stdin(chunk)
                
                # Close stdin after writing the tar stream. Waiting for the
                # Kubernetes exec stream's final status is unreliable here and
                # can keep rollout workers blocked even after remote tar exits.
                resp.close()

                break  # Success, exit the retry loop
            except Exception as e:
                error_str = str(e)
                if resp is not None:
                    try:
                        resp.close()
                    except Exception:
                        pass
                if isinstance(e, TimeoutError):
                    print(f"[COPY TIMEOUT] pod={self.container_name} | image={self.docker_image} | dest={dest_path} | attempt={attempt+1}/{max_retries} | timeout={copy_timeout_seconds}s")

                if is_pod_unavailable_error(error_str):
                    err_oneline = error_str[:300].replace('\n', ' ').replace('\r', '')
                    print(f"[COPY POD_UNAVAILABLE] pod={self.container_name} | image={self.docker_image} | dest={dest_path} | err={err_oneline}")
                    raise PodUnavailableError(f"Pod '{self.container_name}' unavailable while copying to {dest_path}: {error_str}") from e

                # For other exceptions, first adhere to the retry logic.
                if attempt < max_retries - 1:
                    self.logger.warning(f"Copy to container failed (attempt {attempt+1}/{max_retries}): {error_str}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    retry_delay = min(retry_delay, 60)
                    tar_stream.seek(0)  # Reset the stream for the next attempt
                else:
                    err_oneline = error_str[:300].replace('\n', ' ').replace('\r', '')
                    print(f"[COPY FAILED] pod={self.container_name} | image={self.docker_image} | dest={dest_path} | err={err_oneline}")
                    raise PodUnavailableError(f"Failed to copy to pod '{self.container_name}' after {max_retries} attempts: {error_str}") from e

    def copy_to_container(self, src_path: str, dest_path: str):
        """
        Copies a file or directory from the host into the container (Docker or Kubernetes).
        """
        if self.backend == "docker":
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(src_path, arcname=os.path.basename(dest_path))
            tar_stream.seek(0)
            self.container.put_archive(os.path.dirname(dest_path), tar_stream.read())
        elif self.backend == "rle":
            self._copy_to_container_rle(src_path, dest_path)
        elif self.backend == "e2b":
            self._copy_to_container_e2b(src_path, dest_path)
        else:
            # Kubernetes pod copy
            return self._copy_to_container_kubernetes(src_path, dest_path)

    @DeprecationWarning  # TODO: remove dependency on this method with new dockers
    def read_file(self, rel_file_path: str) -> str:
        output, _ = self.run(f"cat /{self.alt_path}/{rel_file_path}")
        return output

    def run_tests(self, timeout: int = 3000) -> tuple[str, str]:
        # if self.mswebench:
        #     output, error_code = self.run(f"cd .. && ./fix-run.sh", timeout=timeout)
        # else:
        #     output, error_code = self.run(f"bash {self.alt_path}/run_tests.sh", timeout=timeout)

        output, error_code = self.run(f"bash {self.alt_path}/run_tests.sh", timeout=timeout)
            
        # Remove ANSI escape codes and \r characters
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output, error_code

    def demux_run_tests(self) -> tuple[str, str, str]:
        stdout, stderr, error_code = self.demux_run(
            f"bash {self.alt_path}/run_tests.sh"
        )
        # Remove ANSI escape codes and \r characters
        stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
        stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)
        return stdout, stderr, error_code

    def checkout(self, commit_hash: str) -> tuple[str, str]:
        output, error_code = self.run(f"git checkout {commit_hash}")
        return output, error_code

    def get_patch(self) -> str:
        """
        Get the diff of the current state of the repository.
        """
        # git add -A && git diff --cached
        # self.run("git add -A")
        output, _ = self.run("git add -A && git diff --cached")
        # output, _ = self.run("git diff")
        return output

    def create_file(self, file_path: str, content: str) -> tuple[str, str]:
        # create a local file with the content
        uuid_ = uuid.uuid4()
        file_path_ = f"{file_path}_{uuid_}"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:    
            temp_file.write(content)
            temp_file.flush()  # Ensure content is written to disk
            temp_file_path = temp_file.name
            # copy the file to the container
            self.copy_to_container(temp_file_path, f"/{file_path}")
            os.unlink(temp_file_path)  # Clean up the temporary file

    def apply_patch(self, patch: str) -> tuple[str, str]:
        # store the patch locally in a file identifiable by docker container id and timestamp
        # must contain unique patch name with both timestamp and docker image name
        uuid_ = uuid.uuid4()
        patch_path = f"{uuid_}.patch"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_patch_path = os.path.join(temp_dir, patch_path)
            with open(temp_patch_path, "w") as f:
                f.write(patch)

            # copy the patch to / of the container
            # apply the patch
            self.copy_to_container(temp_patch_path, f"/{patch_path}")

        output, error_code = self.run(f"git apply --whitespace=nowarn /{patch_path}")
        return output, error_code

    def reverse_patch(self, patch: str) -> tuple[str, str]:
        # store the patch locally in a file identifiable by docker container id and timestamp
        uuid_ = uuid.uuid4()
        patch_path = f"{uuid_}.patch"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_patch_path = os.path.join(temp_dir, patch_path)
            with open(temp_patch_path, "w") as f:
                f.write(patch)

            # copy the patch to / of the container
            self.copy_to_container(temp_patch_path, f"/{patch_path}")
            # apply the patch
            output, error_code = self.run(f"git apply -R /{patch_path}")
            os.unlink(temp_patch_path)  # Clean up the temporary file
        return output, error_code

    def get_logs_eval(
        self, test_spec: TestSpec, content: str
    ) -> tuple[dict[str, str], bool]:
        """
        Retrieve evaluation results for a task instance from its corresponding log file

        Args:
            log_fp (str): path to log file
        Returns:
            bool: whether the patch applied successfully
            dict: status map

        modified from swebench/harness/grading.py
        """
        repo = test_spec.repo
        version = test_spec.version
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]

        # with open(log_fp) as f:
        # # TODO fix constant here
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            self.logger.error(f"Bad code found in log: {bad_codes}")
            return {}, False

        # elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
        #     # Test patch did not apply (should not happen at all)
        #     self.logger.error("Test patch did not apply")
        #     return {}, False

        # Get status map of evaluation results
        if self.backend != "rle":
            content = content.split(test_cmd)[-1]

        self.logger.info(f"using swebench log_parser for repo: {repo}")

        if inspect.signature(log_parser).parameters is None:
            raise ValueError(f"invalid log_parser<{log_parser}>")

        if len(inspect.signature(log_parser).parameters) == 1:
            return log_parser(content), True

        return log_parser(content, test_spec), True

    def parse_logs(self, log_output: str, use_test_spec: bool = False) -> dict:
        return parse_log_fn(f"{self.repo_name}")(log_output)

    def get_file_edit_ratio(self, steps, return_file_list: bool = False, repo_path: str = "/testbed"):
        """
        Calculates the ratio of files edited from trajectory steps (str_replace actions) compared to the golden fix patch.

        Args:
            steps: The list of trajectory steps.
            return_file_list (bool): Whether to also return the lists of files.
            repo_path (str): The repository path inside the Docker container. Defaults to "/testbed".

        Returns:
            float: The ratio of files in the golden fix patch that are also present in the edited files from trajectory.
            (optional) list: List of modified files if return_file_list is True.
        """
        # Ensure repo_path ends with a slash
        if not repo_path.endswith('/'):
            repo_path = f"{repo_path}/"

        if not steps:
            if return_file_list:
                return 0.0, []
            return 0.0

        # 1. get golden patch files
        golden_fix_patch_files = self.get_files_from_patch(self.ds.get("fix_patch", self.ds.get("patch", None)))
        if not golden_fix_patch_files:
            if return_file_list:
                return 0.0, []
            return 0.0

        # 2. extract modified files from trajectory actions (str_replace commands)
        modified_files = set()
        command_pattern = re.compile(r'<parameter=command>([^<]+)</parameter>')
        path_pattern = re.compile(r'<parameter=path>([^<]+)</parameter>')

        for step in steps:
            action = getattr(step, "action", None) if hasattr(step, "action") else step.get("action", "")
            if not action:
                continue

            action_str = str(action) if isinstance(action, dict) else action

            command_match = command_pattern.search(action_str)
            if command_match and command_match.group(1) in ['str_replace', 'insert']:
                path_match = path_pattern.search(action_str)
                if path_match:
                    file_path = path_match.group(1)
                    if file_path.startswith(repo_path):
                        file_path = file_path[len(repo_path):]
                    modified_files.add(file_path)

        golden_fix_patch_files_set = set(golden_fix_patch_files)
        common_files = golden_fix_patch_files_set & modified_files
        ratio = len(common_files) / len(golden_fix_patch_files_set) if golden_fix_patch_files_set else 0.0

        if return_file_list:
            return ratio, list(modified_files)
        else:
            return ratio

    def get_file_view_ratio(self, steps, return_file_list=False, repo_path: str = "/testbed"):
        """
        Extracts all viewed file paths from trajectory steps (including view, str_replace, etc.).

        Args:
            steps: The list of trajectory steps.
            return_file_list (bool): Whether to also return the viewed file list.
            repo_path (str): The repository path inside the Docker container. Defaults to "/testbed".

        Returns:
            float: The ratio of viewed files compared to golden fix patch files.
            (optional) list: List of viewed files if return_file_list is True.
        """
        # Ensure repo_path ends with a slash
        if not repo_path.endswith('/'):
            repo_path = f"{repo_path}/"

        if not steps:
            if return_file_list:
                return 0.0, []
            return 0.0

        viewed_files = set()
        path_pattern = re.compile(r'<parameter=path>([^<]+)</parameter>')

        for step in steps:
            # 支持两种形式：对象属性和字典
            action = getattr(step, "action", None) if hasattr(step, "action") else step.get("action", "")
            if not action:
                continue

            # 如果 action 是字典，转换为字符串进行匹配
            action_str = str(action) if isinstance(action, dict) else action

            # 查找所有path参数
            matches = path_pattern.findall(action_str)
            for match in matches:
                # 去掉 repo_path 前缀
                file_path = match[len(repo_path):] if match.startswith(repo_path) else match
                viewed_files.add(file_path)

        # 获取黄金修复补丁中的所有文件
        golden_fix_patch_files = self.get_files_from_patch(self.ds["fix_patch"]) if hasattr(self, "ds") and self.ds and "fix_patch" in self.ds else []
        golden_fix_patch_files_set = set(golden_fix_patch_files)
        viewed_files_set = viewed_files

        if not golden_fix_patch_files_set:
            ratio = 0.0
        else:
            common_files = viewed_files_set & golden_fix_patch_files_set
            ratio = len(common_files) / len(golden_fix_patch_files_set)

        if return_file_list:
            return ratio, list(viewed_files_set)
        else:
            return ratio

    def calculate_reward_with_stash(self, get_test_output=False, timeout: int = 1200):
        """
        在保存 git 状态的情况下计算 reward，执行完毕后完整恢复 git 状态。

        工作流程：
        1. 状态 S1：保存当前所有 git 信息（commit hash, branch, staged/unstaged changes）
        2. 状态 S2：执行 _calculate_reward（会执行 git checkout & git apply patch）
        3. 恢复到状态 S1：完整回滚所有 git 信息

        Args:
            get_test_output: 是否返回测试输出
            timeout: 超时时间（秒）

        Returns:
            reward 值，如果 get_test_output=True 则返回 (reward, output) 元组
        """
        # 初始化变量，确保在 finally 块中可用
        current_commit = None
        current_branch = None
        stashed = False
        staged_patch_file = None
        unstaged_patch_file = None

        try:
            # ============== 状态 S1: 保存当前 git 信息 ==============
            self.logger.info("=" * 60)
            self.logger.info("正在保存当前 git 状态 (S1)...")
            self.logger.info("=" * 60)

            # 1. 保存当前 commit hash
            current_commit, _ = self.run("git rev-parse HEAD")
            current_commit = current_commit.strip()
            self.logger.info(f"当前 commit: {current_commit}")

            # 2. 保存当前分支名（可能为空，表示 detached HEAD）
            current_branch, _ = self.run("git branch --show-current")
            current_branch = current_branch.strip()
            if current_branch:
                self.logger.info(f"当前分支: {current_branch}")
            else:
                self.logger.info("当前处于 detached HEAD 状态")

            # 3. 保存 staged 的修改到临时文件
            staged_diff, _ = self.run("git diff --cached")
            if staged_diff.strip():
                self.logger.info("检测到 staged 修改，保存到临时文件...")
                patch_uuid = uuid.uuid4()
                staged_patch_file = f"/tmp/staged_{patch_uuid}.patch"
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.patch') as f:
                    f.write(staged_diff)
                    temp_patch_path = f.name
                self.copy_to_container(temp_patch_path, staged_patch_file)
                os.unlink(temp_patch_path)
                self.logger.info(f"Staged 修改已保存到: {staged_patch_file}")

            # 4. 保存 unstaged 的修改到临时文件
            unstaged_diff, _ = self.run("git diff")
            if unstaged_diff.strip():
                self.logger.info("检测到 unstaged 修改，保存到临时文件...")
                patch_uuid = uuid.uuid4()
                unstaged_patch_file = f"/tmp/unstaged_{patch_uuid}.patch"
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.patch') as f:
                    f.write(unstaged_diff)
                    temp_patch_path = f.name
                self.copy_to_container(temp_patch_path, unstaged_patch_file)
                os.unlink(temp_patch_path)
                self.logger.info(f"Unstaged 修改已保存到: {unstaged_patch_file}")

            # 5. 保存 untracked 文件（使用 git stash）
            untracked_files, _ = self.run("git ls-files --others --exclude-standard")
            if untracked_files.strip():
                self.logger.info("检测到 untracked 文件，使用 git stash 保存...")
                # 使用 git stash 只保存 untracked 文件
                self.run("git stash push -u -m 'calculate_reward_untracked_stash'")
                stashed = True
                self.logger.info("Untracked 文件已 stash")

            self.logger.info("Git 状态保存完成！")

            # ============== 状态 S2: 执行 _calculate_reward ==============
            self.logger.info("=" * 60)
            self.logger.info("开始执行 _calculate_reward 方法...")
            self.logger.info("=" * 60)
            result = self._calculate_reward(get_test_output=get_test_output, timeout=timeout)
            self.logger.info(f"_calculate_reward 执行完成，结果: {result}")

        except Exception as e:
            self.logger.error(f"执行 _calculate_reward 时出错: {e}")
            import traceback
            self.logger.error(f"错误堆栈: {traceback.format_exc()}")
            # 即使出错也要尝试恢复
            result = (0, "") if get_test_output else 0

        finally:
            # ============== 恢复到状态 S1 ==============
            try:
                self.logger.info("=" * 60)
                self.logger.info("正在恢复 git 状态到 S1...")
                self.logger.info("=" * 60)

                if not current_commit:
                    self.logger.error("无法恢复：未能保存原始 commit hash")
                    return result

                # 1. 强制回到原始 commit，丢弃所有当前修改
                self.logger.info(f"执行 git reset --hard 回到 commit: {current_commit}")
                self.run(f"git reset --hard {current_commit}")

                # 2. 恢复 unstaged 修改
                if unstaged_patch_file:
                    self.logger.info("恢复 unstaged 修改...")
                    apply_output, apply_code = self.run(f"git apply {unstaged_patch_file}")
                    if apply_code == "0":
                        self.logger.info("Unstaged 修改恢复成功")
                    else:
                        self.logger.warning(f"Unstaged 修改恢复失败: {apply_output}")
                    self.run(f"rm -f {unstaged_patch_file}")

                # 3. 恢复 staged 修改
                if staged_patch_file:
                    self.logger.info("恢复 staged 修改...")
                    apply_output, apply_code = self.run(f"git apply --cached {staged_patch_file}")
                    if apply_code == "0":
                        self.logger.info("Staged 修改恢复成功")
                    else:
                        self.logger.warning(f"Staged 修改恢复失败: {apply_output}")
                    self.run(f"rm -f {staged_patch_file}")

                # 4. 恢复 untracked 文件
                if stashed:
                    self.logger.info("恢复 untracked 文件...")
                    stash_list, _ = self.run("git stash list")
                    if "calculate_reward_untracked_stash" in stash_list:
                        pop_output, pop_code = self.run("git stash pop")
                        if pop_code == "0":
                            self.logger.info("Untracked 文件恢复成功")
                        else:
                            self.logger.warning(f"Untracked 文件恢复失败: {pop_output}")
                    else:
                        self.logger.warning("未找到对应的 stash，可能已被删除")

                # 5. 如果原来在某个分支上，切换回去
                if current_branch:
                    self.logger.info(f"切换回原分支: {current_branch}")
                    checkout_output, checkout_code = self.run(f"git checkout {current_branch}")
                    if checkout_code == "0":
                        self.logger.info("分支切换成功")
                    else:
                        self.logger.warning(f"分支切换失败: {checkout_output}")

                self.logger.info("=" * 60)
                self.logger.info("Git 状态恢复完成！")
                self.logger.info("=" * 60)

            except Exception as restore_error:
                self.logger.error(f"恢复 git 状态时出错: {restore_error}")
                import traceback
                self.logger.error(f"恢复错误堆栈: {traceback.format_exc()}")

                # 最后的保底措施：至少回到原始 commit
                if current_commit:
                    try:
                        self.logger.warning("尝试使用 git reset --hard 作为保底恢复方案...")
                        self.run(f"git reset --hard {current_commit}")
                        self.logger.warning("保底恢复成功")
                    except Exception as fallback_error:
                        self.logger.error(f"保底恢复方案也失败了: {fallback_error}")

                # 清理临时文件
                if staged_patch_file:
                    self.run(f"rm -f {staged_patch_file}")
                if unstaged_patch_file:
                    self.run(f"rm -f {unstaged_patch_file}")

        return result

    def _calculate_reward(self, get_test_output=False, timeout: int = 1200) -> float:
        raise NotImplementedError

    def reset(self):
        self.stop_container()
        self.start_container(
            self.docker_image, self.command, self.container_name, **self.docker_kwargs
        )

    def close(self):
        self.stop_container()
        if self.backend == "docker":
            self.client.close()

    def run_swebv_regression(
        self, run_tests_regression: str | None = None, timeout: int = 300
    ) -> dict[str, str]:
        # run the regression tests for swebench verified dockers
        # copy the 'run_tests_regression' thing from ds into the container at /run_tests_regression.sh
        if run_tests_regression is None:
            run_tests_regression = self.ds["run_tests_regression"]

        with tempfile.NamedTemporaryFile("w") as f:
            f.write(run_tests_regression)
            f.flush()
            self.copy_to_container(f.name, "/run_tests_regression.sh")
        # make the script executable
        self.run("chmod +x /run_tests_regression.sh")

        # run the regression tests
        output, error_code = self.run("/run_tests_regression.sh", timeout=timeout)
        return output
        # return swebench_parse(self.ds, output)

    def start_new_branch(self, branch_name: str = "exp") -> tuple[str, str]:
        # ## save current branch-name
        # output, error_code = self.run("git branch --show-current")
        # self.current_branch = output.strip()
        # # new branch
        # output, error_code = self.run(f"git checkout -b {branch_name}")
        # # save commit hash

        output, error_code = self.run(
            "git config --global user.email 'you@example.com'"
        )
        output, error_code = self.run("git config --global user.name 'Your Name'")
        output, error_code = self.run("git rev-parse HEAD")
        self.current_commit = output.strip()
        return output, error_code

    def commit_after_step(self, step_idx: int) -> tuple[str, str]:
        # commit
        output, error_code = self.run("git add .")
        output, error_code = self.run(f"git commit -m '{step_idx}'")
        return output, error_code

    def undo_last_commit(self) -> tuple[str, str]:
        # undo last commit
        output, error_code = self.run("git reset --hard HEAD~1")
        return output, error_code

    def get_current_commit_hash(self) -> str:
        output, _ = self.run("git rev-parse HEAD")
        return output.strip()

    def soft_git_reset(self) -> tuple[str, str]:
        # soft reset to saved commit
        output, error_code = self.run(f"git reset --soft {self.current_commit}")

        # # checkout to saved branch
        # output, error_code = self.run(f"git checkout {self.current_branch}")

        return output, error_code

    def check_and_install_python3(self) -> str:
        """
        Check if Python 3 is installed in the container. If not, or if the version is less than 3,
        install Miniconda and return the path to the Python interpreter.

        This method follows the logic from MultiSWEBenchDocker.setup_env to ensure Python 3
        availability in containers that may have older Python versions or no Python at all.

        Returns:
            str: Path to the Python 3 interpreter

        Raises:
            FileNotFoundError: If Miniconda installer is not found at ./data/Miniconda3-latest-Linux-x86_64.sh
        """
        self.logger.info("Checking Python version in container...")

        # 检查容器中是否存在 python 命令
        python_check_output, python_check_exit_code = self.run("which python")
        need_install_conda = False

        if python_check_exit_code != "0":
            # 没有 python 命令，需要安装 Miniconda
            self.logger.info("Python not found in container, will install Miniconda")
            need_install_conda = True
        else:
            # 存在 python 命令，检查版本
            python_version_output, _ = self.run("python --version")
            match = re.search(r"Python (\d+)\.(\d+)\.(\d+)", python_version_output)
            if match:
                major_version = int(match.group(1))
                self.logger.info(f"Found Python version: {match.group(0)}")
                if major_version < 3:
                    # python 版本小于 3，需要安装 Miniconda
                    self.logger.info(f"Python version {major_version} is less than 3, will install Miniconda")
                    need_install_conda = True
            else:
                # 无法解析版本，保险起见也安装 Miniconda
                self.logger.warning(f"Unable to parse Python version from output: {python_version_output}")
                self.logger.info("Will install Miniconda to be safe")
                need_install_conda = True

        if need_install_conda:
            abs_path = os.path.join(os.path.dirname(r2egym.__file__), "agenthub/tools/Miniconda3-latest-Linux-x86_64.sh")
            if not os.path.exists(abs_path):
                self.logger.error(f"Miniconda installer not found at {abs_path}")
                raise FileNotFoundError(
                    f"Miniconda installer not found at {abs_path}. "
                    "Please ensure the file exists before running."
                )

            self.logger.info("Copying Miniconda installer to container...")
            self.copy_to_container(
                abs_path,
                "/home/Miniconda3-latest-Linux-x86_64.sh"
            )
            # 在容器内解压 Miniconda 到 /home/conda_env
            self.logger.info("Installing Miniconda in container at /home/conda_env...")
            self.run("bash /home/Miniconda3-latest-Linux-x86_64.sh -b -p /home/conda_env", timeout=300)

            python_interpreter_path = "/home/conda_env/bin/python"
            self.logger.info(f"✅ Miniconda installed successfully, Python path: {python_interpreter_path}")
        else:
            python_interpreter_path, _ = self.run("which python")
            python_interpreter_path = python_interpreter_path.strip()
            self.logger.info(f"✅ Using existing Python installation: {python_interpreter_path}")

        return python_interpreter_path

    def post_process_for_template(self, system_prompt: str, instance_prompt: str):
        return system_prompt, instance_prompt
    
    def apply_actions(self, actions: list[str, dict], timeout: float = CMD_TIMEOUT) -> list[str]:
        """
        Apply a sequence of actions by converting them to bash commands and executing them in the container.

        This method iterates through a list of actions, converts each action to its corresponding
        bash command representation, executes the command in the container/pod, and collects the
        execution outputs.

        Args:
            actions (list[tuple[str, dict]]): A list of action tuples, where each tuple contains:
                - action_name (str): The name of the action/function to execute
                  (e.g., 'file_editor', 'execute_bash', 'search', 'finish')
                - action_parameters (dict): A dictionary of parameters for the action
                  (e.g., {'command': 'view', 'path': '/testbed/file.py', 'concise': 'True'})
            timeout (float, optional): Maximum execution time in seconds for each action.
                Defaults to CMD_TIMEOUT (typically 120 seconds).

        Returns:
            list[str]: A list of observation strings, one for each executed action.
                Each observation contains the stdout/stderr output from the command execution.

        Example:
            >>> actions = [
            ...     ('file_editor', {'command': 'view', 'path': '/testbed/main.py'}),
            ...     ('execute_bash', {'command': 'search_dir', 'search_term': 'class MyClass'})
            ... ]
            >>> observations = runtime.apply_actions(actions, timeout=300)
            >>> print(observations[0])  # Output from viewing the file
            >>> print(observations[1])  # Output from directory search

        Note:
            - Actions are executed sequentially in the order provided
            - If an action fails or times out, its observation will contain error information,
              but subsequent actions will still be executed
            - The 'finish' or 'submit' action is handled specially and outputs '<<<Finished>>>'
        """
        observations = []
        for action_name, action_parameters in actions:
            action = Action(action_name, action_parameters)
            cmd = action.to_bashcmd()
            observation, status = self.run(cmd, timeout=timeout)
            observations.append(observation)
        return observations


class SWEBenchDevDocker(DockerRuntime):
    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, **docker_kwargs)
        self.test_spec = self._get_test_spec_with_cache(self.ds)

    def _get_test_spec_with_cache(self, ds):
        """
        获取test spec，优先从缓存，缓存未命中时使用重试机制
        """
        try:
            # 尝试从缓存获取
            from r2egym.agenthub.utils.test_spec_cache import TestSpecCache
            cache = TestSpecCache()
        except ImportError:
            # 如果缓存模块不可用，直接使用重试机制
            self.logger.warning("Cache module not available, using retry mechanism only")
            return self._make_test_spec_with_retry(ds) 
        cached_spec = cache.get(ds)
        if cached_spec is not None:
            self.logger.info("✅ Test specification loaded from cache")
            return cached_spec
        
        return None

    def _calculate_reward(self, get_test_output=False, timeout: int = 1200) -> float:
        if self.test_spec is None:
            self.logger.error(f"Test specification not found, skipping reward calculation for {self.ds['instance_id']}")
            if get_test_output:
                return 0, ""
            return 0

        self.reset_tests()
        self.apply_patch(self.ds["fix_patch"])

        out, _ = self.run(
            "bash /run_tests.sh", timeout=timeout
        )  # run the tests after applying the patch

        eval_status_map, found = self.get_logs_eval(self.test_spec, out)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }
        from swebench.harness.constants import FAIL_ONLY_REPOS, EvalType  
        eval_type = EvalType.FAIL_ONLY if self.test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL  

        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=eval_type
        )
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        if get_test_output:
            return int(success), out
        return int(success)
    
    def setup_env(self):
        try:
            commit_id = self.ds['base_commit']
            self.run("git fetch")
            self.run(f"git checkout {commit_id}")
            # Setup the run_test.sh script for subsequent testing.  
            eval_script_content = "\n".join(
                [
                    "#!/bin/bash",
                    "set -uxo pipefail",
                    "source /opt/miniconda3/bin/activate",
                    f"conda activate testbed",
                    f"cd /testbed",
                ]
            ) + "\n"

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                test_cmd = MAP_REPO_VERSION_TO_SPECS[self.ds["repo"]][self.ds["version"]]["test_cmd"]

                if isinstance(self.ds["PASS_TO_PASS"], str):
                    self.ds["PASS_TO_PASS"] = json.loads(self.ds["PASS_TO_PASS"])
                if isinstance(self.ds["FAIL_TO_PASS"], str):
                    self.ds["FAIL_TO_PASS"] = json.loads(self.ds["FAIL_TO_PASS"])

                all_test_names = self.ds["PASS_TO_PASS"] + self.ds["FAIL_TO_PASS"]
                fixed_test_names = []
                for name in all_test_names:
                    if '[' in name:
                        name = name.split('[')[0]
                    fixed_test_names.append(name)

                # add single quotes around each test name
                all_test_names = ["'"+t+"'" for t in fixed_test_names]
                all_test_module = " ".join(all_test_names)
                command =  eval_script_content + f"{test_cmd} {all_test_module}"
                temp_file.write(command)
                temp_file.flush()  # Ensure content is written to disk
                temp_file_path = temp_file.name
                # copy the file to container and clean up
                self.copy_to_container(temp_file_path, "/run_tests.sh")
                os.unlink(temp_file_path)  # Clean up the temporary file
            
            self.run("chmod +x /run_tests.sh")

            # Ensure can call and execute the tools in /usr/local/bin.
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            self.run('echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc')
            self.install_python_package_with_retry(package_name="chardet", max_retries=3)
        except Exception as e:
            self.logger.error(f"Error setting up environment: {repr(e)}")
        
        # apply test patch
        if "test_patch" in self.ds:
            self.apply_patch(self.ds["test_patch"])


class MultiSWEBenchDocker(DockerRuntime):

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        # Determine repo_path based on IS_TESTBED_DIR environment variable or lang field
        # If IS_TESTBED_DIR is set to "true", use /testbed
        # Otherwise, use /home/{repo} for multi-language projects
        import os as os_module
        is_testbed_dir = os_module.environ.get('IS_TESTBED_DIR', '').lower() == 'true'

        if is_testbed_dir:
            repo_path = "/testbed"
        else:
            repo_path = f"/home/{ds['repo']}"

        self.mswebench = True
        docker_path = "/go/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:" + DOCKER_PATH
        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, docker_path=docker_path, **docker_kwargs)

    def apply_patch(self, patch: str) -> tuple[str, str]:
        # store the patch locally in a file identifiable by docker container id and timestamp
        # must contain unique patch name with both timestamp and docker image name
        uuid_ = uuid.uuid4()
        patch_path = f"{self.container_name}_{uuid_}.patch"
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_patch_path = os.path.join(temp_dir, patch_path)
            with open(temp_patch_path, "w") as f:
                f.write(patch)

            # copy the patch to / of the container
            # apply the patch
            self.copy_to_container(temp_patch_path, "/home/fix.patch")
            output, error_code = self.run(f"git apply --whitespace=nowarn /home/fix.patch",timeout=600)
            os.unlink(temp_patch_path)  # Clean up the temporary file

            # clean the fix.patch file
            self.run("echo -n > /home/fix.patch")
        return "", "0"
    
    def reset_tests(self):
        all_files = self.get_files_from_patch(self.ds["test_patch"])
        reset_command = (
            f'printf "%s\\n" {" ".join(all_files)} | '
            f'xargs -n1 -I{{}} git checkout -- "{{}}" 2>/dev/null'
        )
        self.run(reset_command)

        # remove added files
        all_files = self.get_added_files_from_patch(self.ds["test_patch"])
        if all_files:
            self.run("rm -rf " + " ".join(all_files))

    def compute_reward(self, fix_patch_log: str):
        import tempfile
        from multi_swe_bench.harness.run_evaluation import CliArgs

        with tempfile.TemporaryDirectory() as temp_dir:
            # generate dataset file
            dataset_file = os.path.join(temp_dir, "dataset.jsonl")
            with open(dataset_file, "w") as f:
                f.write(json.dumps(self.ds, ensure_ascii=False))
            
            # create fix-path-run.log file
            # eg: data/workdir/iamkun/dayjs/evals/pr-938/fix-patch-run.log
            if "/" in self.ds["repo"]:
                org, repo = self.ds["repo"].split("/")
            else:
                org, repo = self.ds["org"], self.ds["repo"]

            fix_path_run_log_file = os.path.join(temp_dir, "workdir", org, repo , "evals", f"pr-{self.ds['number']}", "fix-patch-run.log")
            os.makedirs(os.path.dirname(fix_path_run_log_file), exist_ok=True)
            with open(fix_path_run_log_file, "w", encoding="utf-8") as f:
                f.write(fix_patch_log)

            args = {
                "mode": "compute_score",
                "workdir": os.path.join(temp_dir, "workdir"),
                "patch_files": [
                    "./data/traj_patches-filtered-fake.jsonl"
                ],
                "dataset_files": [
                    dataset_file
                ],
                "force_build": False,
                "output_dir": os.path.join(temp_dir, "dataset"),
                "specifics": [],
                "skips": [],
                "repo_dir": os.path.join(temp_dir, "repos"),
                "need_clone": False,
                "global_env": [],
                "clear_env": True,
                "stop_on_error": False,
                "max_workers": 1,
                "max_workers_build_image": 1,
                "max_workers_run_instance": 1,
                "log_dir": os.path.join(temp_dir, "logs"),
                "log_level": "DEBUG",
                "fix_patch_run_cmd": "",
                "log_to_console": True
            }
            cli = CliArgs.from_dict(args)
            cli.run()

            # read final report json file
            report_file = os.path.join(temp_dir, "dataset", "final_report.json")
            with open(report_file, "r") as f:
                report = json.load(f)
            
        return len(report["resolved_ids"]) > 0

    def _calculate_reward(self, get_test_output=False, timeout: int = 1200) -> float:

        self.reset_tests()
        # 2. apply test_patch
        self.apply_patch(self.ds["test_patch"])
        # 3. reset test
        out, status_code = self.run(
            "cd ../ && ./run.sh", timeout=timeout
        )  # run the tests after applying the patch

        if "The command took too long to execute" in (out or ""):
            if get_test_output:
                return 0.0, out
            return 0.0

        try:
            reward = self.compute_reward(out)
        except Exception as e:
            self.logger.error(f"Error computing reward from test output: {repr(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            if get_test_output:
                return 0.0, out
            return 0.0

        if get_test_output:
            return reward, out
        
        return reward

        from multi_swe_bench.harness.report import ReportTask
        from multi_swe_bench.harness.test_result import TestResult
        with tempfile.TemporaryDirectory() as temp_dir:
            fix_run_log_file = os.path.join(temp_dir, "fix-patch-run.log")
            with open(fix_run_log_file, "w") as f:
                f.write(out)
            
            report_task = ReportTask(org=self.ds["org"], repo=self.ds["repo"], number=self.ds["number"], instance_dir=Path(temp_dir))
            report = report_task.generate_report(
                run_log=TestResult.from_dict(self.ds["run_result"]),  
                test_patch_run_log=TestResult.from_dict(self.ds["test_patch_result"]), 
                fix_patch_run_log=out
            )

        if get_test_output:
            return int(report.valid), out
        
        return int(report.valid)
    
    def setup_env(self):
        try:
            fix_run_content, exit_code = self.run("cd /home")
            # 检查容器中是否存在 python 命令
            python_check_output, python_check_exit_code = self.run("which python")
            need_install_conda = False
            if python_check_exit_code != "0":
                # 没有 python 命令，需要安装 Miniconda
                need_install_conda = True
            else:
                # 存在 python 命令，检查版本
                python_version_output, _ = self.run("python --version")
                match = re.search(r"Python (\d+)\.(\d+)\.(\d+)", python_version_output)
                if match:
                    major_version = int(match.group(1))
                    if major_version < 3:
                        # python 版本小于 3，需要安装 Miniconda
                        need_install_conda = True
                else:
                    # 无法解析版本，保险起见也安装 Miniconda
                    need_install_conda = True

            if need_install_conda:
                abs_path = os.path.join(os.path.dirname(r2egym.__file__), "agenthub/tools/Miniconda3-latest-Linux-x86_64.sh")
                assert os.path.exists(abs_path), f"{abs_path} 文件不存在"
                self.copy_to_container(abs_path, "/home/Miniconda3-latest-Linux-x86_64.sh")
                # 在容器内解压 Miniconda 到 /home/conda_env
                self.run("bash /home/Miniconda3-latest-Linux-x86_64.sh -b -p /home/conda_env")
                # 将 conda 环境的 bin 路径添加到 .bashrc
                # WARNING(wj-Mcat): 如果将这个环境变量添加到PATH 当中，会污染镜像内部的 PATH 
                # self.docker_path = self.docker_path + ":/home/conda_env/bin"
                python_interpreter_path = "/home/conda_env/bin/python"
            else:
                python_interpreter_path, exit_code = self.run("which python")

            self.run("mkdir -p /root/.venv/bin/")
            self.run(f"ln -s {python_interpreter_path.strip()} /root/.venv/bin/python")
            self.docker_path = f"/root/.venv/bin/:{self.docker_path}"
            self.install_python_package_with_retry(package_name="chardet", max_retries=3)

            # clear
            self.run("echo -n > /home/fix.patch")
            self.run("echo -n > /home/test.patch")
            self.run("chmod +x /home/run.sh")

            # if "mutation_patch" in self.ds:
            #     self.apply_patch(self.ds["mutation_patch"])

        except Exception as e:
            if isinstance(e, PodUnavailableError):
                raise
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )


class SWESmithDocker(DockerRuntime):

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        if docker_image is None:
            if "image_name" in ds:
                docker_image = ds['image_name']
            elif "docker_image" in ds:
                docker_image = ds['docker_image']
            else:
                raise ValueError("No docker image found in dataset entry.")

        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, **docker_kwargs)

    def setup_env(self):
        try:
            # Setup the run_test.sh script for subsequent testing.  
            test_command, _ = get_test_command(self.ds)
            eval_script_content = "\n".join(
                [
                    "#!/bin/bash",
                    "set -uxo pipefail",
                    "source /opt/miniconda3/bin/activate",
                    f"conda activate testbed",
                    f"cd /testbed/",
                    f": '>>>>> Start Test Output'",
                    test_command,
                    f": '>>>>> End Test Output'",
                ]
            ) + "\n"
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                temp_file.write(eval_script_content)
                temp_file.flush()  # Ensure content is written to disk
                temp_file_path = temp_file.name
            
                # Copy the file to container and clean up
                self.copy_to_container(temp_file_path, "/run_tests.sh")
                os.unlink(temp_file_path)  # Clean up the temporary file
            
            self.run("chmod +x /run_tests.sh")

            # Ensure can call and execute the tools in /usr/local/bin.
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            self.run('echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc')
            self.install_python_package_with_retry(package_name="chardet", max_retries=3)
            # get current commit hash id
            self.ds['base_commit'] = self.get_current_commit_hash()

            # apply test patch into repo
            self.apply_patch(self.ds["patch"])

        except Exception as e:
            self.logger.error(f"Error setting up environment: {repr(e)}")

    def reset_tests(self):
        commit_id = self.ds['base_commit']
        all_files = self.get_files_from_patch(self.ds["test_patch"])
        reset_command = (
            f'printf "%s\\n" {" ".join(all_files)} | '
            f'xargs -n1 -I{{}} git checkout {commit_id} -- "{{}}" 2>/dev/null'
        )
        self.run(reset_command)

    def _calculate_reward(self, get_test_output=False, timeout: int = 1200) -> float:
        self.reset_tests()
        
        def _parse_reward(output: str):
            parse = self.parse_logs(output)
            fail2pass = [ ".".join(line.split("::")[1:]) for line in self.ds['FAIL_TO_PASS']]
            pass2pass = [ ".".join(line.split("::")[1:]) for line in self.ds['PASS_TO_PASS']]
            # @(Naman, Jas): Parse the output and return the reward. This implementation is a hack rn.
            if not parse:
                return 0.0
            
            # Check fail2pass
            for test_name in fail2pass:
                if test_name not in parse:
                    # Check if test_name is substring of any key
                    matching_key = next((k for k in parse.keys() if test_name in k), None)
                    if matching_key is None:
                        return 0.0
                    if parse[matching_key] != 'PASSED':
                        return 0.0
                    test_name = matching_key
                if parse[test_name] != 'PASSED':
                    return 0.0
            
            # Check pass2pass
            for test_name in pass2pass:
                if test_name not in parse:
                    # Check if test_name is substring of any key
                    matching_key = next((k for k in parse.keys() if test_name in k), None)
                    if matching_key is None:
                        return 0.0
                    test_name = matching_key
                if parse[test_name] != 'PASSED':
                    return 0.0
            return 1.0

        output, error_msg = self.run("/run_tests.sh", timeout=timeout)
        self.reset_tests()

        reward = _parse_reward(output)
        if get_test_output:
            return reward, output
        return reward

class SWEGymDocker(DockerRuntime):
    def setup_env(self):
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                if "test_cmds" in self.ds:
                    test_cmd = self.ds["test_cmds"][0]
                else:
                    from swegym_test_spec import MAP_REPO_VERSION_TO_SPECS as SWEGYM_SPECS
                    test_cmd = SWEGYM_SPECS[self.ds["repo"]][self.ds["version"]]["test_cmd"]

                if isinstance(self.ds["PASS_TO_PASS"], str):
                    self.ds["PASS_TO_PASS"] = json.loads(self.ds["PASS_TO_PASS"])
                if isinstance(self.ds["FAIL_TO_PASS"], str):
                    self.ds["FAIL_TO_PASS"] = json.loads(self.ds["FAIL_TO_PASS"])

                all_test_names = self.ds["PASS_TO_PASS"] + self.ds["FAIL_TO_PASS"]
                fixed_test_names = []
                for name in all_test_names:
                    if '[' in name:
                        name = name.split('[')[0]
                    fixed_test_names.append(name)

                # add single quotes around each test name
                all_test_names = ["'"+t+"'" for t in fixed_test_names]
                test_cmd_content = test_cmd + " " + " ".join(all_test_names)
                command = f"cd {self.repo_path} && {test_cmd_content}"
                temp_file.write(command)
                temp_file.flush()  # Ensure content is written to disk
                temp_file_path = temp_file.name
                # copy the file to container and clean up
                self.copy_to_container(temp_file_path, "/run_tests.sh")
                os.unlink(temp_file_path)  # Clean up the temporary file
                    
            # make the run_tests.sh executable
            self.run("chmod +x /run_tests.sh")

            # 先判断 /opt/miniconda3/envs/testbed 是否存在
            check_testbed_cmd = '[ -d /opt/miniconda3/envs/testbed ] && echo "EXIST" || echo "NOT_EXIST"'
            exist_flag, _ = self.run(check_testbed_cmd)
            if "NOT_EXIST" in exist_flag:
                # 获取 python 路径并软链到 /root/.venv/bin/python
                python_path, _ = self.run("which python")
                python_path = python_path.strip()
                self.run("mkdir -p /root/.venv/bin")
                self.run(f"ln -sf {python_path} /root/.venv/bin/python")
            else:
                self.run("ln -s /opt/miniconda3/envs/testbed /root/.venv")
                self.run('echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc')
            self.install_python_package_with_retry(package_name="chardet", max_retries=3)

            if os.getenv("R2E_REMOVE_GIT_INFO", "true").lower() == "true":
                remove_git_script = os.path.join(os.path.dirname(r2egym.__file__), "agenthub/tools/remove_git_info.sh")
                if os.path.exists(remove_git_script):
                    self.copy_to_container(remove_git_script, "/remove_git_info.sh")
                    remove_tag = "pytest-dev/pytest" not in self.ds["repo"]
                    self.run(f"bash /remove_git_info.sh {self.ds['base_commit']} --remove_tag {str(remove_tag).lower()}", timeout=1200)
                else:
                    self.logger.error(f"[SCRIPT NOT FOUND] pod={self.container_name} image={self.docker_image} path={remove_git_script}")
        except Exception as e:
            if isinstance(e, PodUnavailableError):
                raise
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )

    def _calculate_reward(self, get_test_output=False, timeout: int = 1200) -> float:
        new_file_paths = self._get_safe_new_file_paths_from_test_patch()
        self.reset_tests()
        self._remove_new_files_from_test_patch(new_file_paths)
        try:
            self.apply_patch(self.ds["test_patch"])
            out, error_msg = self.run("/run_tests.sh", timeout=timeout)
        finally:
            self._remove_new_files_from_test_patch(new_file_paths)

        def _get_reward(output: str):
            parse = self.parse_logs(output)
            fail2pass = [ ".".join(line.split("::")[1:]) for line in self.ds['FAIL_TO_PASS']]
            pass2pass = [ ".".join(line.split("::")[1:]) for line in self.ds['PASS_TO_PASS']]
            # @(Naman, Jas): Parse the output and return the reward. This implementation is a hack rn.
            if not parse:
                return 0.0
            
            # Check fail2pass
            for test_name in fail2pass:
                if test_name not in parse:
                    # Check if test_name is substring of any key
                    matching_key = next((k for k in parse.keys() if test_name in k), None)
                    if matching_key is None:
                        return 0.0
                    if parse[matching_key] != 'PASSED':
                        return 0.0
                    test_name = matching_key
                if parse[test_name] != 'PASSED':
                    return 0.0
            
            # Check pass2pass
            for test_name in pass2pass:
                if test_name not in parse:
                    # Check if test_name is substring of any key
                    matching_key = next((k for k in parse.keys() if test_name in k), None)
                    if matching_key is None:
                        return 0.0
                    test_name = matching_key
                if parse[test_name] != 'PASSED':
                    return 0.0
            return 1.0
        
        try:
            reward = _get_reward(out)
        except Exception as e:
            self.logger.error(f"Error computing reward from test output: {repr(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            if get_test_output:
                return 0.0, out
            return 0.0

        if get_test_output:
            return reward, out
        return reward


class SWEGymWithTestPatchDocker(SWEGymDocker):
    def setup_env(self):
        super().setup_env()
        self.apply_patch(self.ds["test_patch"])


class SWERebenchDocker(SWEGymDocker):
    def setup_env(self):
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                test_cmd = self.ds["install_config"]["test_cmd"]

                all_test_names = self.ds["PASS_TO_PASS"] + self.ds["FAIL_TO_PASS"]
                fixed_test_names = []
                for name in all_test_names:
                    if '[' in name:
                        name = name.split('[')[0]
                    fixed_test_names.append(name)

                # add single quotes around each test name
                all_test_names = ["'"+t+"'" for t in fixed_test_names]
                test_cmd_content = test_cmd + " " + " ".join(all_test_names)

                temp_file.write(test_cmd_content)
                temp_file.flush()  # Ensure content is written to disk
                temp_file_path = temp_file.name
                # copy the file to container and clean up
                self.copy_to_container(temp_file_path, "/run_tests.sh")
                os.unlink(temp_file_path)  # Clean up the temporary file
                    
            # make the run_tests.sh executable
            self.run("chmod +x /run_tests.sh")

            # 先判断 /opt/miniconda3/envs/testbed 是否存在
            check_testbed_cmd = '[ -d /opt/miniconda3/envs/testbed ] && echo "EXIST" || echo "NOT_EXIST"'
            exist_flag, _ = self.run(check_testbed_cmd)
            if "NOT_EXIST" in exist_flag:
                # 获取 python 路径并软链到 /root/.venv/bin/python
                python_path, _ = self.run("which python")
                python_path = python_path.strip()
                self.run("mkdir -p /root/.venv/bin")
                self.run(f"ln -sf {python_path} /root/.venv/bin/python")
            else:
                self.run("ln -s /opt/miniconda3/envs/testbed /root/.venv")
                self.run('echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc')
            self.install_python_package_with_retry(package_name="chardet", max_retries=3)
        except Exception as e:
            if isinstance(e, PodUnavailableError):
                raise
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )

class SWERebenchWithTestPatchDocker(SWERebenchDocker):
    def setup_env(self):
        super().setup_env()
        self.apply_patch(self.ds["test_patch"])

class SWEBenchDocker(DockerRuntime):
    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, **docker_kwargs)

    def parse_logs(self, log_output: str, use_test_spec: bool = False) -> dict:
        parsed_output, patch_apply_success = self.get_logs_eval(
            self.test_spec, log_output
        )
        return parsed_output

    def setup_env(self):
        try:
            # check whether /run_tests.sh file exists
            # log, _ = self.run("ls /run_tests.sh")
            # if "No such file or directory" in log:
            # make the run_tests.sh executable
            # 先判断 /run_tests.sh 是否存在，如果不存在则写入 self.test_spec.eval_script

            check_file_cmd = "ls /"
            out, _ = self.run(check_file_cmd)
            self.test_spec = self._get_test_spec_with_cache(self.ds)
            if "run_tests.sh" not in out.strip():
                # 文件不存在，写入 eval_script
                from tempfile import NamedTemporaryFile
                with NamedTemporaryFile("w", delete=False) as f:

                    f.write(self.test_spec.eval_script)
                    temp_script_path = f.name
                # 拷贝到容器
                self.copy_to_container(temp_script_path, "/run_tests.sh")

            self.run("chmod +x /run_tests.sh")

            # # move all skip files (if present) to /root
            # for skip_file in SKIP_FILES:
            #     self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")

            self.alt_path = (
                "/"  # the run_test is in the "/" directory for swebench dockers
            )

            # make symlink of conda env to /root/.venv
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")

            # install required packages TODO: check if working
            # self.run(
            #     "python -m pip install tree-sitter==0.20.4 tree_sitter_languages==1.10.2"
            # )
            
            # Install chardet with retry logic (3 retries per source, 2 sources = 6 total attempts)
            self.install_python_package_with_retry(package_name="chardet", max_retries=3, raise_on_failure=True)

            # Verified eval images no longer require future git info cleanup.
            # SWE-bench Pro still performs this cleanup in SWEBenchProDocker.

            self._fix_image_dependencies()
            self._patch_run_tests_script()

        except Exception as e:
            if isinstance(e, PodUnavailableError):
                raise
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )

    def _fix_image_dependencies(self):
        instance_id = self.ds.get("instance_id", "")

        if instance_id in ("astropy__astropy-8707", "astropy__astropy-8872"):
            self.install_python_package_with_retry("pytest<7.0.0", skip_installed_check=True)
            self.install_python_package_with_retry("setuptools<58", skip_installed_check=True)

        if instance_id == "psf__requests-1921":
            self.install_python_package_with_retry("pytest<7.0.0", skip_installed_check=True)

        if instance_id in ("sphinx-doc__sphinx-10323", "sphinx-doc__sphinx-10435"):
            self.install_python_package_with_retry("pygments<2.14", skip_installed_check=True)

        if instance_id == "psf__requests-6028":
            self.install_python_package_with_retry("pytest-mock")

        requests_need_httpbin = [
            "psf__requests-5414",
            "psf__requests-2931",
            "psf__requests-1921",
            "psf__requests-1766",
        ]
        if instance_id in requests_need_httpbin:
            self.install_python_package_with_retry("pytest-httpbin")
            self.install_python_package_with_retry("pytest-mock")
        if instance_id in ("psf__requests-5414", "psf__requests-2931", "psf__requests-1766"):
            self.install_python_package_with_retry("trustme")

    def _patch_run_tests_script(self):
        instance_id = self.ds.get("instance_id", "")
        patch_images = [
            "astropy__astropy-8707",
            "astropy__astropy-8872",
            "django__django-7530",
            "django__django-10097",
            "django__django-16502",
            "psf__requests-1921",
            "psf__requests-6028",
            "psf__requests-5414",
            "psf__requests-2931",
            "psf__requests-1766",
            "psf__requests-2317",
            "psf__requests-1724",
            "sphinx-doc__sphinx-10323",
            "sphinx-doc__sphinx-10435",
            "sphinx-doc__sphinx-8595",
            "sphinx-doc__sphinx-9711",
        ]

        # The reference eval harness also patches whenever `test_patch` introduces new
        # files. That gating is what gives the 9 django new-file instances (13516, 13837,
        # 14311, 15525, 16256, 16454, 16877, 11141, 11749) the
        # `./tests/runtests.py` -> `python ./tests/runtests.py` rewrite below.
        # The reference's other reason for this gating (appending `rm -f /testbed/<f>`)
        # is already covered out-of-script by `_remove_new_files_from_test_patch`.
        if instance_id not in patch_images and not self._get_safe_new_file_paths_from_test_patch():
            return

        run_tests_content = self.run("cat /run_tests.sh")[0].strip()
        new_lines = []

        for line in run_tests_content.split("\n"):
            stripped = line.lstrip()
            if "django__django" in instance_id and stripped.startswith("./tests/runtests.py"):
                indentation = line[: len(line) - len(stripped)]
                line = f"{indentation}python {stripped}"
            new_lines.append(line)

            stripped = line.strip()
            is_pip_install = "pip install" in stripped and "pip install -r" not in stripped
            if not is_pip_install:
                continue

            if instance_id in ("astropy__astropy-8707", "astropy__astropy-8872"):
                new_lines.append("pip install 'pytest<7.0.0' 'setuptools<58' -q")

            if instance_id == "psf__requests-1921":
                new_lines.append("pip install 'pytest<7.0.0' 'pytest-httpbin' -q")

            if instance_id in ("sphinx-doc__sphinx-10323", "sphinx-doc__sphinx-10435"):
                new_lines.append("pip install 'pygments<2.14' -q")

            if instance_id == "psf__requests-6028":
                new_lines.append("pip install pytest-mock -q")

            if instance_id in ("psf__requests-5414", "psf__requests-2931", "psf__requests-1766"):
                new_lines.append("pip install 'pytest-httpbin>=0.3.0' pytest-mock trustme -q")

        if instance_id == "psf__requests-6028":
            patched_lines = []
            for line in new_lines:
                if line.strip().startswith("pytest"):
                    patched_lines.append("rm -f /tmp/test_utils.py")
                patched_lines.append(line)
            new_lines = patched_lines

        if instance_id in ("astropy__astropy-8707", "astropy__astropy-8872"):
            patched_lines = []
            for line in new_lines:
                stripped = line.lstrip()
                if stripped.startswith("pytest"):
                    indentation = line[: len(line) - len(stripped)]
                    line = indentation + stripped.replace(
                        "pytest",
                        "python -W ignore::DeprecationWarning -m pytest -o filterwarnings=",
                        1,
                    )
                patched_lines.append(line)
            new_lines = patched_lines

        if instance_id in ("sphinx-doc__sphinx-8595", "sphinx-doc__sphinx-9711"):
            patched_lines = []
            for line in new_lines:
                if line.strip().startswith("tox "):
                    patched_lines.append(
                        "sed -i 's/python -X dev -m pytest --durations/python -X dev -m pytest -rA --durations/g' /testbed/tox.ini"
                    )
                patched_lines.append(line)
            new_lines = patched_lines

        unset_proxy_instances = [
            "django__django-16502",
            "psf__requests-2931",
            "psf__requests-2317",
            "psf__requests-1724",
            "psf__requests-5414",
            "psf__requests-1766",
            "psf__requests-1921",
            "psf__requests-6028",
        ]
        if instance_id in unset_proxy_instances:
            unset_line = "unset HTTP_PROXY http_proxy HTTPS_PROXY https_proxy no_proxy NO_PROXY"
            patched_lines = []
            for line in new_lines:
                stripped = line.strip()
                if (
                    stripped.startswith("pytest")
                    or stripped.startswith("python -m pytest")
                    or stripped.startswith("python -W")
                    or stripped.startswith("./tests/runtests.py")
                    or stripped.startswith("python ./tests/runtests.py")
                ):
                    patched_lines.append(unset_line)
                patched_lines.append(line)
            new_lines = patched_lines

        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write("\n".join(new_lines))
            temp_script_path = f.name
        self.copy_to_container(temp_script_path, "/run_tests.sh")
        os.unlink(temp_script_path)
        self.run("chmod +x /run_tests.sh")

    def _calculate_reward(self, get_test_output=False, timeout: int = 1200) -> float:
        # gt_test_patch = self.commit.get_patch(test_file=True,non_test_file=False)
        # self.apply_patch(gt_test_patch)
        new_file_paths = self._get_safe_new_file_paths_from_test_patch()
        self._remove_new_files_from_test_patch(new_file_paths)
        try:
            out, _ = self.run(
                "bash /run_tests.sh 2>&1", timeout=timeout
            )  # run the tests after applying the patch
        finally:
            self._remove_new_files_from_test_patch(new_file_paths)
        
        eval_status_map, found = self.get_logs_eval(self.test_spec, out)
        p2p = self._fix_p2p_phantom_entries(self.test_spec.PASS_TO_PASS, eval_status_map)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: p2p,
        }
        from swebench.harness.constants import FAIL_ONLY_REPOS, EvalType  
        eval_type = EvalType.FAIL_ONLY if self.test_spec.repo in FAIL_ONLY_REPOS else EvalType.PASS_AND_FAIL  

        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=eval_type
        )
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        if get_test_output:
            return int(success), out
        return int(success)

    _P2P_PHANTOM_IMAGES = [
        "astropy__astropy-7606",
    ]

    def _fix_p2p_phantom_entries(self, p2p_list: list, eval_status_map: dict) -> list:
        if not any(tag in self.docker_image for tag in self._P2P_PHANTOM_IMAGES):
            return p2p_list
        if not p2p_list or not eval_status_map:
            return p2p_list

        fixed = []
        for entry in p2p_list:
            if entry.endswith("[]") and entry not in eval_status_map:
                base = entry[:-2]
                variants = {k: v for k, v in eval_status_map.items() if k.startswith(base + "[")}
                if variants:
                    all_ok = all(
                        v in (TestStatus.PASSED.value, TestStatus.XFAIL.value)
                        for v in variants.values()
                    )
                    if all_ok:
                        self.logger.info(f"P2P phantom entry dropped (all {len(variants)} variants passed): {entry}")
                        continue
                    failed = {
                        k: v
                        for k, v in variants.items()
                        if v not in (TestStatus.PASSED.value, TestStatus.XFAIL.value)
                    }
                    self.logger.warning(f"P2P phantom entry kept (variants failed): {entry} -> {failed}")
            fixed.append(entry)
        return fixed

class YiYanSWEImagesDocker(DockerRuntime):
    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, **docker_kwargs)

    def setup_env(self):
        try:
            if "eval_script" in self.ds:
                run_test_content = self.ds["eval_script"]
            elif "test_cmds" in self.ds:
                run_test_content = "\n".join(self.ds["test_cmds"]) 
            else:
                raise ValueError(f"invalid dataset format with keys<{self.ds.keys()}> ...")

            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
                temp_file.write(run_test_content)
                temp_file.flush()
                temp_file_path = temp_file.name
            self.copy_to_container(temp_file_path, "/run_tests.sh")
            os.unlink(temp_file_path)

            self.run("chmod +x /run_tests.sh")

            # 检查 /root/.vent/bin/python
            self.run("mkdir -p /root/.venv/bin")
            out, _ = self.run("ls -A /root/.venv/bin | wc -l")
            files_count = int(out.strip())
            if files_count < 1:
                self.run("ln -s $(which python) /root/.venv/bin/python")
            
            # 判断是否有 uv 命令，如果有的话，就用 uv 来执行上面的 pip install
            check_uv_cmd = '[ -x "$(command -v uv)" ] && echo "EXIST" || echo "NOT_EXIST"'
            out, _ = self.run(check_uv_cmd)
            if "NOT_EXIST" in out:
                self.install_python_package_with_retry(package_name="chardet", max_retries=3, use_uv=False)
            else:
                self.install_python_package_with_retry(package_name="chardet", max_retries=3, use_uv=True)
            
            # # nodejs binary file
            # node_binary_file = "./data/node-v24.11.0-linux-x64.tar.xz"
            # self.copy_to_container(node_binary_file, "/root/node-v24.11.0-linux-x64.tar.xz")
            # self.run("tar -xvf /root/node-v24.11.0-linux-x64.tar.xz -C /root/")
            # self.run("ln -s /root/node-v24.11.0-linux-x64/bin/node /usr/local/bin/node")
            # self.run("ln -s /root/node-v24.11.0-linux-x64/bin/npm /usr/local/bin/npm")
            # self.run("ln -s /root/node-v24.11.0-linux-x64/bin/npx /usr/local/bin/npx")

            # self.run("npm install -g @anthropic-ai/claude-code")
            # self.run("python -m pip install claude-agent-sdk")

        except Exception as e:

            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )

    def _calculate_reward(self, get_test_output=False, timeout: int = 1200) -> float:

        self.reset_tests()
        self.apply_patch(self.ds["test_patch"])
        out, status = self.run(
            "bash /run_tests.sh", timeout=timeout
        )  # run the tests after applying the patch

        if "OMNIGRIL_EXIT_CODE=" in out:
            if out.strip().split("\n")[-1].endswith("OMNIGRIL_EXIT_CODE=0"):
                success = True
            else:
                success = False
        else:
            success = status == "0"

        if get_test_output:
            return int(success), out
        return int(success)


class YiYanPythonDocker(SWEBenchDocker):
    def get_logs_eval(
        self, test_spec: TestSpec, content: str, repo: str = None
    ) -> tuple[dict[str, str], bool]:
        """
        Retrieve evaluation results for a task instance from its corresponding log file

        Args:
            log_fp (str): path to log file
        Returns:
            bool: whether the patch applied successfully
            dict: status map

        modified from swebench/harness/grading.py
        """
        repo = test_spec.repo
        version = test_spec.version
        if repo =='secdev/scapy':
            log_parser = MAP_REPO_TO_PARSER['secdev/scapy']
        else:
            log_parser = MAP_REPO_TO_PARSER["custom"]

        if repo not in MAP_REPO_VERSION_TO_SPECS and test_spec is not None and test_spec.eval_script:
            test_cmd = test_spec.eval_script
        elif repo in MAP_REPO_VERSION_TO_SPECS:
            test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        else:
            raise ValueError(f"No test command found for repo: {repo}")

        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]

        # with open(log_fp) as f:
        # # TODO fix constant here
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            self.logger.error(f"Bad code found in log: {bad_codes}")
            return {}, False

        # elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
        #     # Test patch did not apply (should not happen at all)
        #     self.logger.error("Test patch did not apply")
        #     return {}, False

        # Get status map of evaluation results
        if self.backend != "rle":
            content = content.split(test_cmd)[-1]
        self.logger.info(f"using swebench log_parser for repo: {repo}")

        # Use inspect to check function argument count to determine whether to pass test_spec
        import inspect
        sig = inspect.signature(log_parser)
        params = list(sig.parameters.values())
        # Exclude 'self' if present (for methods)
        args_to_check = [p for p in params if p.name != 'self']

        if len(args_to_check) == 1:
            # Only one argument besides self, just pass content
            return log_parser(content), True

        # More than one argument, pass both content and test_spec
        return log_parser(content, test_spec), True

class R2EDocker(DockerRuntime):

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, **docker_kwargs)
        if "parsed_commit" in self.ds:
            self.commit_json = self.ds["parsed_commit"]
        elif "parsed_commit_content" in self.ds:
            self.commit_json = self.ds["parsed_commit_content"]
        else:
            raise ValueError("No parsed commit found in dataset entry")

        self.commit = ParsedCommit(**json.loads(self.commit_json))

    
    def setup_env(self):
        try:
            # setup venv
            # modify the repo path to a common path
            # self.run(f"cp -r {self.repo_path} /workspace")

            # create a symlink from repo_path/.venv to /root/.venv
            self.run(f"ln -s {self.repo_path}/.venv {self.alt_path}/.venv")

            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python"
            )

            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3"
            )
            self.run(
                f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;"
            )

            # self.run(f"mv {self.repo_path} /workspace")
            # self.repo_path = "/workspace"

            # install required packages
            # self.run("uv pip install tree_sitter_languages") # remove since already installed in new dockers
            self.install_python_package_with_retry(package_name="chardet", max_retries=3, use_uv=True)
            self.run("find . -name '*.pyc' -delete")
            self.run("find . -name '__pycache__' -exec rm -rf {} +")

            # also delete pycache and pyc from /r2e_tests
            self.run("find /r2e_tests -name '*.pyc' -delete")
            self.run("find /r2e_tests -name '__pycache__' -exec rm -rf {} +")
            # move all skip files (if present) to /root
            for skip_file in SKIP_FILES_NEW:
                self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")

            # r2e_tests are in the / directory, move them to /root
            self.run(f"mv /r2e_tests {self.alt_path}/r2e_tests")

            # make a softlink for /root/r2e_tests (if present)
            self.run(f"ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests")
            # self.run(f"ln -s /r2e_tests {self.repo_path}/r2e_tests")
        except Exception as e:
            self.logger.error(f"Error setting up environment: {repr(e)}")


    def _calculate_reward(self, get_test_output=False, timeout: int = 1200) -> float:
        # calculate reward based for r2e-edit dockers
        output, error_code = self.run_tests(timeout=timeout)
        # print(output)x
        parse = self.parse_logs(output)
        parse = decolor_dict_keys(parse)
        try:
            expected_json = self.ds["expected_output_json"]
        except Exception as e:
            expected_json = self.read_file("expected_test_output.json")

        expected: dict = json.loads(expected_json)
        expected = decolor_dict_keys(expected)
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

        # Compare
        if len(parse) != len(expected):
            reward = 0.0
        else:
            # If ANY mismatch, reward = 0.0, else = 1.0
            match = True
            for k in parse.keys():
                if not k:
                    continue
                if k not in expected:
                    match = False
                    break
                if parse[k] != expected[k]:
                    match = False
                    break
            reward = 1.0 if match else 0.0
        # If the caller wants the test output as well, return (reward, output)
        if get_test_output:
            return reward, output
        return reward


class YiYanSWEFactoryImagesDocker(DockerRuntime):
    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, **docker_kwargs)
    
    
    def remove_added_files_by_patch(self, patch):
        # remove added file
        added_files = self.get_added_files_from_patch(patch)
        for file in added_files:
            self.run(f"rm -rf {file}")

    def setup_env(self):
        try:
            run_test_content = self.ds["eval_script"]

            self.run(f"git reset --hard {self.ds['base_commit']}")
            self.remove_added_files_by_patch(self.ds["test_patch"])
            self.remove_added_files_by_patch(self.ds["patch"])

            with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8") as temp_file:
                temp_file.write(run_test_content)
                temp_file.flush()
                temp_file_path = temp_file.name
            self.copy_to_container(temp_file_path, "/run_tests.sh")
            os.unlink(temp_file_path)

            self.run("chmod +x /run_tests.sh")
            # self.run("python -m pip install -i http://mirrors.baidubce.com/pypi/simple/ --trusted-host=mirrors.baidubce.com chardet")
            self.run("mkdir -p /root/.venv/bin")
            if self.ds["language"].lower() in ["c", "cpp"]:
                python_path = self.check_and_install_python3()
                self.run(f"ln -s {python_path} /root/.venv/bin/python")
            else:
                self.run(f"ln -s $(which python) /root/.venv/bin/python")
            self.install_python_package_with_retry("chardet")
            self.run("chmod +x /root/.venv/bin/python")
            # print("python version -> ", self.run("python --version")[0], self.run("cat /run_tests.sh")[0])

            if os.getenv("R2E_REMOVE_GIT_INFO", "true").lower() == "true":
                remove_git_script = os.path.join(os.path.dirname(r2egym.__file__), "agenthub/tools/remove_git_info.sh")
                if os.path.exists(remove_git_script):
                    self.copy_to_container(remove_git_script, "/remove_git_info.sh")
                    remove_tag = "pytest-dev/pytest" not in self.ds["repo"]
                    self.run(f"bash /remove_git_info.sh {self.ds['base_commit']} --remove_tag {str(remove_tag).lower()}", timeout=1200)
                else:
                    self.logger.error(f"[SCRIPT NOT FOUND] pod={self.container_name} image={self.docker_image} path={remove_git_script}")

        except Exception as e:
            if isinstance(e, PodUnavailableError):
                raise
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )

    def _calculate_reward(self, get_test_output=False, timeout: int = 1200) -> float:
        new_file_paths = self._get_safe_new_file_paths_from_test_patch()
        self._remove_new_files_from_test_patch(new_file_paths)
        self.reset_tests()
        try:
            out, status = self.run(
                "bash /run_tests.sh 2>&1", timeout=timeout
            )  # run the tests after applying the patch
        finally:
            self._remove_new_files_from_test_patch(new_file_paths)

        if "OMNIGRIL_EXIT_CODE=" in out:
            lines = out.strip().split("\n")
            target_lines = [line for line in lines if line.startswith("OMNIGRIL_EXIT_CODE=")]
            if len(target_lines) > 0:
                success = target_lines[-1] == "OMNIGRIL_EXIT_CODE=0"
            else:
                success = False
        else:
            success = False

        if get_test_output:
            return int(success), out
        return int(success)
class SWEBenchMultilingualDocker(DockerRuntime):

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        self.swebenchmultilingual = True
        docker_path = "/go/bin:/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:" + DOCKER_PATH
        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, docker_path=docker_path, **docker_kwargs)
        self.swe_test_spec = self.get_test_spec()
    
    def get_test_spec(self):
        # self.clean_and_reset_env()
        import swebench
        self.logger.info(swebench.__version__)
        from swebench.harness import test_spec
        from swebench.harness.constants import SWEbenchInstance
        instance = SWEbenchInstance(
            repo=self.ds["repo"],
            instance_id=self.ds["instance_id"],
            base_commit=self.ds["base_commit"],
            patch=self.ds["patch"],
            test_patch=self.ds["test_patch"],
            problem_statement=self.ds["problem_statement"],
            hints_text=self.ds["hints_text"],
            created_at=self.ds["created_at"],
            version=self.ds["version"],
            FAIL_TO_PASS=self.ds["FAIL_TO_PASS"],
            PASS_TO_PASS=self.ds["PASS_TO_PASS"]
        )
        return test_spec.test_spec.make_test_spec(instance=instance)

    # def clean_and_reset_env(self):
    #     need_clean_key = []
    #     for key, value in os.sys.modules.items():
    #         if "swebench" in key:
    #             need_clean_key.append(key)
    #     for item in need_clean_key:
    #         del os.sys.modules[item]
    #     third_party_path = os.path.join("./", 'third_party')
    #     os.sys.path.insert(0, third_party_path)

    def setup_env(self):
        if "swebench-multilingal-rebuild" in self.ds["docker_image"]:
            python_interpreter_path = "/home/conda_env/bin/python"
            self.run("mkdir -p /root/.venv/bin/")
            self.run(f"ln -s {python_interpreter_path.strip()} /root/.venv/bin/python")
            self.docker_path = f"/root/.venv/bin/:{self.docker_path}"
            self.install_python_package_with_retry(package_name="chardet", max_retries=3)

            # These images ship Cargo 1.81.0, which cannot parse edition2024 deps.
            _tokio_instances = {"tokio-rs__tokio-4384", "tokio-rs__tokio-4898"}
            instance_id = self.ds.get("instance_id", "")
            if any(iid in instance_id for iid in _tokio_instances):
                try:
                    self.logger.info(f"[sweml] {instance_id}: upgrading Rust toolchain for edition2024 support")
                    upgrade_cmd = (
                        "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs "
                        "| sh -s -- -y --default-toolchain stable --no-modify-path 2>&1"
                    )
                    out, rc = self.run(upgrade_cmd, timeout=600)
                    self.logger.info(f"[sweml] rustup install rc={rc} out={out[-300:]}")
                    cargo_home = "/root/.cargo/bin"
                    if cargo_home not in self.docker_path:
                        self.docker_path = f"{cargo_home}:{self.docker_path}"
                    ver_out, _ = self.run("cargo --version 2>&1")
                    self.logger.info(f"[sweml] {instance_id}: cargo version after upgrade: {ver_out.strip()}")
                except Exception as e:
                    self.logger.error(f"[sweml] {instance_id}: failed to upgrade Rust: {repr(e)}")
            return

        try:
            # 将本地的 Miniconda 安装包复制到容器内部
            # 在试验前必须保证 ./data 目录下有 Miniconda3-latest-Linux-x86_64.sh 文件
            abs_path = os.path.join(os.path.dirname(r2egym.__file__), "agenthub/tools/Miniconda3-latest-Linux-x86_64.sh")
            assert os.path.exists(abs_path), f"{abs_path} 文件不存在"
            self.copy_to_container(abs_path, "/home/Miniconda3-latest-Linux-x86_64.sh")
            # 在容器内解压 Miniconda 到 /home/conda_env
            self.run("bash /home/Miniconda3-latest-Linux-x86_64.sh -b -p /home/conda_env")
            # 将 conda 环境的 bin 路径添加到 .bashrc
            # WARNING(wj-Mcat): 如果将这个环境变量添加到PATH 当中，会污染镜像内部的 PATH 
            # self.docker_path = self.docker_path + ":/home/conda_env/bin"
            python_interpreter_path = "/home/conda_env/bin/python"

            self.run("mkdir -p /root/.venv/bin/")
            self.run(f"ln -s {python_interpreter_path.strip()} /root/.venv/bin/python")
            self.docker_path = f"/root/.venv/bin/:{self.docker_path}"
            self.install_python_package_with_retry(package_name="chardet", max_retries=3)
            # Multilingual eval images no longer require future git info cleanup.
            # SWE-bench Pro still performs this cleanup in SWEBenchProDocker.
            try:
                if "babel" in self.docker_image:
                    output, error = self.run("grep -n 'const numWorkers' Gulpfile.mjs")
                    print("Before Regex", output, error)
                    self.run(r"sed -i 's/const numWorkers = require(\"os\")\.cpus()\.length \/ 2 - 1;/const numWorkers = 2;/g' Gulpfile.mjs")
                    self.run(r"sed -i 's/const numWorkers = Math\.ceil(Math\.max(cpus()\.length, 1) \/ 2) - 1;/const numWorkers = 2;/g' Gulpfile.mjs")
                    output, error = self.run("grep -n 'const numWorkers' Gulpfile.mjs")
                    print("After Regex", output, error)
            except Exception as e:
                self.logger.error(
                    f"Error fixing babel repo error for RLE: {repr(e)} @ {self.docker_image}"
                )
        except Exception as e:
            if isinstance(e, PodUnavailableError):
                raise
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )
    def _calculate_reward(self, get_test_output=False, timeout: int = 300) -> int:
        # self.clean_and_reset_env()
        from swebench.harness import grading
        from swebench.harness.constants import (
            FAIL_TO_FAIL,
            FAIL_TO_PASS,
            KEY_INSTANCE_ID,
            FAIL_ONLY_REPOS,
            EvalType,
            ResolvedStatus
        )
        _RESET_TEST_PATCH_INSTANCES = {
            "tokio-rs__tokio-4384",
            "babel__babel-16130",
            "tokio-rs__tokio-6551",
            "tokio-rs__tokio-4898",
            "apache__lucene-11760",
        }
        instance_id = self.ds.get("instance_id", "")
        new_file_paths = self._get_safe_new_file_paths_from_test_patch()
        self._remove_new_files_from_test_patch(new_file_paths)
        if any(iid in instance_id for iid in _RESET_TEST_PATCH_INSTANCES):
            test_patch = self.ds.get("test_patch", "")
            if test_patch:
                import re as _re
                for m in _re.finditer(r'^diff --git a/(\S+) b/(\S+)', test_patch, _re.MULTILINE):
                    a_path, b_path = m.group(1), m.group(2)
                    chunk_start = m.start()
                    chunk_end = test_patch.find('\ndiff --git', chunk_start + 1)
                    chunk = test_patch[chunk_start: chunk_end if chunk_end != -1 else len(test_patch)]
                    is_new_file = bool(_re.search(r'^--- /dev/null', chunk, _re.MULTILINE))
                    is_deleted_file = bool(_re.search(r'^\+\+\+ /dev/null', chunk, _re.MULTILINE))
                    is_rename = (a_path != b_path) and not is_new_file and not is_deleted_file
                    full_b_path = f"{self.repo_path}/{b_path}"
                    if is_new_file:
                        rm_out, rm_rc = self.run(f"rm -f {full_b_path}", timeout=30)
                        self.logger.info(f"[sweml] reset test_patch: removed new file {full_b_path} rc={rm_rc}")
                    elif is_rename:
                        rm_out, rm_rc = self.run(f"rm -f {full_b_path}", timeout=30)
                        self.logger.info(f"[sweml] reset test_patch: removed rename dst {full_b_path} rc={rm_rc}")
                        co_out, co_rc = self.run(
                            f"git -C {self.repo_path} checkout HEAD -- {a_path}", timeout=30
                        )
                        self.logger.info(f"[sweml] reset test_patch: restored rename src {a_path} rc={co_rc} out={co_out[:100]}")
                    else:
                        co_out, co_rc = self.run(
                            f"git -C {self.repo_path} checkout HEAD -- {b_path}", timeout=30
                        )
                        self.logger.info(f"[sweml] reset test_patch: restored {b_path} rc={co_rc} out={co_out[:100]}")

        command = self.swe_test_spec.eval_script
        self.create_file("eval.sh", command)
        self.run("chmod +x /eval.sh")
        # self.apply_patch(self.ds["test_patch"])
        try:
            output, error = self.run("/eval.sh 2>&1", timeout=timeout)
        finally:
            self._remove_new_files_from_test_patch(new_file_paths)

        with tempfile.TemporaryDirectory() as temp_dir:
            eval_log = os.path.join(temp_dir, "eval.log")
            with open(eval_log, "w") as f:
                f.write(output)
            report_map = {
                "patch_is_None": False,
                "patch_exists": True,
                "patch_successfully_applied": False,
                "resolved": False,
            }
            eval_status_map, found = grading.get_logs_eval(self.swe_test_spec, eval_log)
            if not found:
                if get_test_output:
                    return 0.0, output
                return 0.0
            report_map["patch_successfully_applied"] = True
            eval_ref = {
                KEY_INSTANCE_ID: self.swe_test_spec.instance_id,
                FAIL_TO_PASS: self.swe_test_spec.FAIL_TO_PASS,
                PASS_TO_PASS: self.swe_test_spec.PASS_TO_PASS,
            }
            if "apache__lucene-11760" in instance_id and not eval_status_map:
                import re as _re
                from swebench.harness.constants import TestStatus
                all_tests = list(self.swe_test_spec.FAIL_TO_PASS) + list(self.swe_test_spec.PASS_TO_PASS)
                if _re.search(r':\S+:test \(SUCCESS\)', output):
                    self.logger.info(f"[sweml] lucene-11760: Gradle SUCCESS summary detected, synthesizing PASSED for all {len(all_tests)} tests")
                    eval_status_map = {t: TestStatus.PASSED.value for t in all_tests}
                else:
                    failed = set()
                    for m in _re.finditer(r'^(\S.*?)\s+FAILED\s*$', output, _re.MULTILINE):
                        failed.add(m.group(1).strip())
                    self.logger.info(f"[sweml] lucene-11760: Gradle non-SUCCESS, failed tests={failed}")
                    eval_status_map = {}
                    for t in all_tests:
                        eval_status_map[t] = TestStatus.FAILED.value if t in failed else TestStatus.PASSED.value
            if "tokio-rs__tokio-6551" in instance_id:
                _skip = {"io_driver_fd_count", "io_driver_ready_count"}
                eval_ref[PASS_TO_PASS] = [t for t in eval_ref[PASS_TO_PASS] if t not in _skip]
                self.logger.info(f"[sweml] tokio-6551: removed network tests from PASS_TO_PASS, remaining={eval_ref[PASS_TO_PASS]}")
            eval_type = (
                EvalType.FAIL_ONLY
                if self.swe_test_spec.repo in FAIL_ONLY_REPOS
                else EvalType.PASS_AND_FAIL
            )
            report = grading.get_eval_tests_report(eval_status_map, eval_ref, eval_type=eval_type)
            if grading.get_resolution_status(report) == ResolvedStatus.FULL.value:
                report_map["resolved"] = True
            report_map["tests_status"] = report
            if report_map["resolved"]:
                return 1.0, output
            else:
                return 0.0, output


class SWEBenchProDocker(DockerRuntime):
    """
    SWE-bench Pro Docker runtime adapted from:
    https://github.com/scaleapi/SWE-bench_Pro-os/blob/2d71624f1908af6122871a9eb679c148b9ba97f0/swe_bench_pro_eval.py
    
    This class supports:
    - Loading local run scripts (run_script.sh and parser.py)
    - Creating entryscript to orchestrate test execution
    - Extracting ENV variables from Dockerfiles
    - Parsing output.json format test results
    """
    
    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/app",  # main repo path  
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        scripts_dir: str = None,  # Directory containing run scripts
        **docker_kwargs,
    ):
        self.scripts_dir = scripts_dir or os.path.join(_third_party_dir, "swebench_pro_run_scripts")
        self.dockerfiles_dir = os.path.join(_third_party_dir, "swebench_pro_dockerfiles")
        repo_path = "/app"
        docker_path = "/go/bin:/usr/local/go/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:" + DOCKER_PATH
        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, docker_path=docker_path, **docker_kwargs)
        self.default_entrypoint_shell_type = "/bin/bash"
        
    
    def load_dockerfile(self, dockerfile_type: str = "base"):
        """
        Load Dockerfile content for the instance.
        
        Args:
            dockerfile_type: Either "base" or "instance"
            
        Returns:
            Dockerfile content as string, or None if not found
        """
        instance_id = self.ds.get("instance_id", "")
        if not instance_id:
            self.logger.warning("No instance_id found in dataset")
            return None
            
        dockerfile_path = os.path.join(
            self.dockerfiles_dir,
            f"{dockerfile_type}_dockerfile",
            instance_id,
            "Dockerfile"
        )
        
        if os.path.exists(dockerfile_path):
            with open(dockerfile_path, 'r') as f:
                return f.read()
        else:
            self.logger.warning(f"Dockerfile not found: {dockerfile_path}")
            return None
    
    def extract_env_from_dockerfiles(self):
        """
        Extract ENV commands from base and instance Dockerfiles.
        
        Returns:
            String containing export statements for all ENV variables
        """
        env_cmds = []
        
        for dockerfile_type in ["base", "instance"]:
            dockerfile_content = self.load_dockerfile(dockerfile_type)
            if dockerfile_content:
                for line in dockerfile_content.split("\n"):
                    line = line.strip()
                    if line.startswith("ENV"):
                        # Convert ENV commands to export statements
                        env_cmd = line.replace("ENV", "export", 1)
                        env_cmds.append(env_cmd)
        
        return "\n".join(env_cmds) if env_cmds else ""
    
    def load_local_script(self, script_name: str):
        """
        Load a script file from local scripts directory.
        
        Args:
            script_name: Name of the script file (e.g., "run_script.sh", "parser.py")
            
        Returns:
            Script content as string
        """
        instance_id = self.ds.get("instance_id", "")
        if not instance_id:
            raise ValueError("No instance_id found in dataset")
        
        script_path = os.path.join(self.scripts_dir, instance_id, script_name)
        
        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")
        
        with open(script_path, 'r') as f:
            return f.read()
    
    def create_entryscript(self):
        """
        Create entry script that orchestrates the test execution.
        This script will:
        1. Set environment variables from Dockerfiles
        2. Reset git to base commit
        3. Apply patch
        4. Run before_repo_set_cmd
        5. Execute run_script.sh with selected tests
        6. Parse results with parser.py
        
        Returns:
            Entry script content as string
        """
        # Get environment variables from Dockerfiles
        env_cmds = self.extract_env_from_dockerfiles()
        
        # Get configuration from dataset
        base_commit = self.ds.get("base_commit", "")
        
        # Get before_repo_set_cmd (take last line if multi-line)
        before_repo_set_cmd = self.ds.get("before_repo_set_cmd", "")
        if before_repo_set_cmd:
            before_repo_set_cmd = before_repo_set_cmd.strip().split("\n")[-1]
        
        # Get selected test files
        selected_test_files = self.ds.get("selected_test_files_to_run", [])
        if isinstance(selected_test_files, str):
            selected_test_files = eval(selected_test_files)
        selected_test_files_str = ",".join(selected_test_files) if selected_test_files else ""
        
        # Build the entry script
        entry_script = f"""
{env_cmds}

{before_repo_set_cmd}
# Run tests and save stdout and stderr to separate files
bash /workspace/run_script.sh {selected_test_files_str} > /workspace/stdout.log 2> /workspace/stderr.log || true

# Run parsing script to generate output.json
python /workspace/parser.py /workspace/stdout.log /workspace/stderr.log /workspace/output.json
"""
        return entry_script
    
    def remove_added_files_by_patch(self, patch):
        """Remove files that were added by a patch"""
        if not patch:
            return
        added_files = self.get_added_files_from_patch(patch)
        for file in added_files:
            self.run(f"rm -rf {self.repo_path}/{file}")

    def setup_env(self):
        """
        Setup the environment for SWE-bench Pro evaluation.
        This includes:
        1. Resetting to base commit
        2. Copying run scripts and parser to /workspace
        3. Setting up Python environment
        4. Preparing for test execution
        """
        try:
            instance_id = self.ds.get("instance_id", "")
            base_commit = self.ds.get("base_commit", "")
            
            self.logger.info(f"Setting up environment for {instance_id}")
            
            # Reset to base commit
            self.run(f"git reset --hard {base_commit}")
            self.run(f"git checkout {base_commit}")
            
            # Remove added files from patches if they exist
            if "test_patch" in self.ds and self.ds["test_patch"]:
                self.remove_added_files_by_patch(self.ds["test_patch"])
            if "patch" in self.ds and self.ds["patch"]:
                self.remove_added_files_by_patch(self.ds["patch"])
            
            # Create workspace directory
            self.run("mkdir -p /workspace")
            workspace_files = {
                "patch.diff": self.ds["test_patch"],
            }
            
            # Try to load and copy local scripts if they exist
            try:
                run_script_content = self.load_local_script("run_script.sh")
                workspace_files["run_script.sh"] = run_script_content
                self.logger.info("Loaded run_script.sh from local scripts directory")
            except FileNotFoundError as e:
                self.logger.warning(f"run_script.sh not found in scripts directory: {e}")
                # Fallback: use eval_script from dataset if available
                if "eval_script" in self.ds:
                    workspace_files["run_script.sh"] = self.ds["eval_script"]
                    self.logger.info("Using eval_script from dataset")
            
            try:
                parser_content = self.load_local_script("parser.py")
                workspace_files["parser.py"] = parser_content
                self.logger.info("Loaded parser.py from local scripts directory")
            except FileNotFoundError as e:
                self.logger.warning(f"parser.py not found in scripts directory: {e}")
                # Create a default parser that outputs a simple JSON format
                raise ValueError("parser.py not found in scripts directory")
            
            # load entryscript
            entryscript_content = self.create_entryscript()
            workspace_files["entryscript.sh"] = entryscript_content

            with tempfile.TemporaryDirectory() as temp_dir:
                for filename, content in workspace_files.items():
                    with open(os.path.join(temp_dir, filename), "w", encoding="utf-8") as f:
                        f.write(content)
                self.copy_to_container(temp_dir, "/workspace")

            chmod_targets = ["/workspace/entryscript.sh"]
            if "run_script.sh" in workspace_files:
                chmod_targets.insert(0, "/workspace/run_script.sh")
            self.run(f"chmod +x {' '.join(chmod_targets)}")
            
            # Setup Python environment
            self.run("mkdir -p /root/.venv/bin")
            
            # Check language and setup appropriate Python
            language = self.ds.get("language", "python").lower()

            if language in ["c", "cpp", "c++"]:
                python_path = self.check_and_install_python3()
                self.run(f"ln -s {python_path} /root/.venv/bin/python")
            else:
                self.run(f"ln -s $(which python) /root/.venv/bin/python")
            
            # Install required Python packages
            self.install_python_package_with_retry("chardet")
            self.run("chmod +x /root/.venv/bin/python")
            
            if os.getenv("R2E_REMOVE_GIT_INFO", "true").lower() == "true":
                remove_git_script = os.path.join(os.path.dirname(r2egym.__file__), "agenthub/tools/remove_git_info.sh")
                if os.path.exists(remove_git_script):
                    self.copy_to_container(remove_git_script, "/remove_git_info.sh")
                    remove_tag = "pytest-dev/pytest" not in self.ds.get("repo", "")
                    self.run(
                        f"bash /remove_git_info.sh {base_commit} --remove_tag {str(remove_tag).lower()}",
                        timeout=1200
                    )
                else:
                    self.logger.error(f"[SCRIPT NOT FOUND] pod={self.container_name} image={self.docker_image} path={remove_git_script}")
            
            self.logger.info(f"Environment setup completed for {instance_id}")
            
        except Exception as e:
            if isinstance(e, PodUnavailableError):
                raise
            self.logger.error(f"Error setting up environment: {repr(e)} @ {self.docker_image}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def parse_output_json(self, output_json_path: str = "/workspace/output.json"):
        """
        Parse the output.json file generated by parser.py.
        
        Args:
            output_json_path: Path to output.json in container
            
        Returns:
            Dictionary containing test results, or None if parsing fails
        """
        try:
            # Read output.json from container
            output_content, _ = self.run(f"cat {output_json_path}")
            output = json.loads(output_content)
            return output
        except Exception as e:
            self.logger.error(f"Failed to parse output.json: {e}")
            return None

    def _calculate_reward(self, get_test_output=False, timeout: int = 300) -> float:
        """
        Calculate reward based on test execution results.
        
        This method:
        1. Removes test patch files
        2. Resets tests to base state
        3. Applies the model's patch
        4. Runs tests via run_script.sh
        5. Parses results from output.json
        6. Compares with fail_to_pass and pass_to_pass requirements
        
        Args:
            get_test_output: If True, returns (reward, output) tuple
            timeout: Timeout for test execution in seconds
            
        Returns:
            Reward value (0 or 1), and optionally test output
        """
        instance_id = self.ds.get("instance_id", "")
        test_output = ""
        try:
            self.logger.info(f"[swe-pro] _calculate_reward start: instance_id={instance_id} timeout={timeout}")

            new_file_paths = self._get_safe_new_file_paths_from_test_patch()
            self._remove_new_files_from_test_patch(new_file_paths)
            self.reset_tests()
            self.apply_patch(self.ds["test_patch"])

            out, status = self.run("bash /workspace/entryscript.sh", timeout=timeout)
            self.logger.info(f"[swe-pro] entryscript.sh finished: status={status} out_len={len(out)}")
            self.logger.info(f"[swe-pro] entryscript.sh output (last 500 chars):\n{out[-500:]}")
            test_output = f"=== entryscript ===\n{out}"

            stdout_log, stdout_rc = self.run("cat /workspace/stdout.log")
            stderr_log, stderr_rc = self.run("cat /workspace/stderr.log")
            self.logger.info(f"[swe-pro] stdout.log ({stdout_rc}, {len(stdout_log)} chars), last 1000:\n{stdout_log[-1000:]}")
            self.logger.info(f"[swe-pro] stderr.log ({stderr_rc}, {len(stderr_log)} chars), last 500:\n{stderr_log[-500:]}")
            test_output = f"=== stdout ===\n{stdout_log}\n=== stderr ===\n{stderr_log}\n=== entryscript ===\n{out}"

            raw_json, json_rc = self.run("cat /workspace/output.json")
            self.logger.info(f"[swe-pro] output.json ({json_rc}, {len(raw_json)} chars): {raw_json[:500]}")

            if str(json_rc).strip() != "0" or not raw_json.strip():
                reason = "output_json_missing_or_empty"
                self.logger.warning(
                    f"[swe-pro] reward parse failed: reason={reason} "
                    f"json_rc={json_rc} raw_len={len(raw_json)} instance_id={instance_id}"
                )
                test_output = (
                    f"{test_output}\n"
                    f"=== output.json ({json_rc}) ===\n{raw_json}\n"
                    f"[SWE_PRO_REWARD_PARSE_ERROR] reason={reason} json_rc={json_rc}"
                )
                if get_test_output:
                    return 0, test_output
                return 0

            try:
                output = json.loads(raw_json)
            except json.JSONDecodeError as e:
                reason = "output_json_invalid_json"
                self.logger.warning(
                    f"[swe-pro] reward parse failed: reason={reason} "
                    f"json_rc={json_rc} raw_len={len(raw_json)} error={e!r} instance_id={instance_id}"
                )
                test_output = (
                    f"{test_output}\n"
                    f"=== output.json ({json_rc}) ===\n{raw_json}\n"
                    f"[SWE_PRO_REWARD_PARSE_ERROR] reason={reason} json_rc={json_rc} error={repr(e)}"
                )
                if get_test_output:
                    return 0, test_output
                return 0

            all_tests = output.get("tests", [])
            passed_tests = {x["name"] for x in all_tests if x["status"] == "PASSED"}
            failed_tests = {x["name"] for x in all_tests if x["status"] != "PASSED"}
            self.logger.info(f"[swe-pro] output.json: total={len(all_tests)} passed={len(passed_tests)} not_passed={len(failed_tests)}")
            self.logger.info(f"[swe-pro] passed_tests={sorted(passed_tests)}")
            self.logger.info(f"[swe-pro] not_passed_tests={sorted(failed_tests)}")

            f2p = set(eval(self.ds["fail_to_pass"]))
            p2p = set(eval(self.ds["pass_to_pass"]))
            self.logger.info(f"[swe-pro] f2p={sorted(f2p)}")
            self.logger.info(f"[swe-pro] p2p={sorted(p2p)}")

            import re as _re

            def _remap_year_tests(required: set, actual: set) -> set:
                remapped = set()
                for test_name in required:
                    if test_name in actual:
                        remapped.add(test_name)
                        continue
                    match = _re.match(r"^(.*\[)\d{4}-(.*)$", test_name)
                    if not match:
                        remapped.add(test_name)
                        continue
                    prefix, suffix = match.group(1), match.group(2)
                    candidates = [
                        actual_name for actual_name in actual
                        if _re.match(r"^" + _re.escape(prefix) + r"\d{4}-" + _re.escape(suffix) + r"$", actual_name)
                    ]
                    if len(candidates) == 1:
                        self.logger.info(f"[swe-pro] year-remap: {test_name!r} -> {candidates[0]!r}")
                        remapped.add(candidates[0])
                    elif len(candidates) > 1:
                        self.logger.warning(f"[swe-pro] year-remap: multiple candidates for {test_name!r}: {candidates}")
                        remapped.add(test_name)
                    else:
                        remapped.add(test_name)
                return remapped

            if "internetarchive__openlibrary" in instance_id:
                all_actual = {x["name"] for x in all_tests}
                f2p = _remap_year_tests(f2p, all_actual)
                p2p = _remap_year_tests(p2p, all_actual)
                self.logger.info(f"[swe-pro] f2p (after remap)={sorted(f2p)}")
                self.logger.info(f"[swe-pro] p2p (after remap)={sorted(p2p)}")

            if (
                "ansible__ansible-de5858f48dc9e1ce9117034e0d7e76806f420ca8" in instance_id
                or "future-architect__vuls-bff6b7552370b55ff76d474860eead4ab5de785a" in instance_id
            ):
                def _prefix_match(required: set, actual: set) -> set:
                    remapped = set()
                    for test_name in required:
                        if test_name in actual:
                            remapped.add(test_name)
                            continue
                        candidates = [actual_name for actual_name in actual if actual_name.startswith(test_name)]
                        if len(candidates) == 1:
                            self.logger.info(f"[swe-pro] prefix-match: {test_name!r} -> {candidates[0]!r}")
                            remapped.add(candidates[0])
                        elif len(candidates) > 1:
                            self.logger.warning(f"[swe-pro] prefix-match: multiple candidates for {test_name!r}: {candidates}")
                            remapped.add(test_name)
                        else:
                            remapped.add(test_name)
                    return remapped

                all_actual = {x["name"] for x in all_tests}
                f2p = _prefix_match(f2p, all_actual)
                p2p = _prefix_match(p2p, all_actual)
                self.logger.info(f"[swe-pro] f2p (after prefix-match)={sorted(f2p)}")
                self.logger.info(f"[swe-pro] p2p (after prefix-match)={sorted(p2p)}")

            missing = (f2p | p2p) - passed_tests
            success = len(missing) == 0
            self.logger.info(f"[swe-pro] missing tests={sorted(missing)}")
            self.logger.info(f"[swe-pro] reward={int(success)} (success={success})")

            if get_test_output:
                return int(success), test_output
            return int(success)

        except Exception as e:
            self.logger.error(f"Error calculating reward: {repr(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            if get_test_output:
                return 0, test_output
            return 0

    def post_process_for_template(self, system_prompt: str, instance_prompt: str):
        return system_prompt.replace("/testbed", "/app"), instance_prompt.replace("/testbed", "/app")

    def apply_patch(self, patch: str) -> tuple[str, str]:
        reset_out, reset_rc = self.run(f"git -C {self.repo_path} reset HEAD", timeout=60)
        self.logger.info(f"[swe-pro] apply_patch: git reset HEAD rc={reset_rc} out={reset_out[:200]}")
        success, output = self._apply_patch_with_fallback(patch)
        self.logger.info(f"[swe-pro] apply_patch: success={success} out={output}")
        return output, "0" if success else "1"

    def _apply_patch_with_fallback(self, patch: str) -> tuple[bool, str]:
        import uuid as _uuid

        patch_path = f"/tmp/_swepro_patch_{_uuid.uuid4().hex[:8]}.patch"
        self.create_file(patch_path.lstrip("/"), patch)
        strategies = [
            (f"git apply --3way --whitespace=nowarn {patch_path}", False),
            (f"git apply --whitespace=nowarn {patch_path}", False),
            (f"git apply --whitespace=nowarn --ignore-space-change --ignore-whitespace {patch_path}", False),
            (f"patch --batch --fuzz=5 -p1 -i {patch_path}", False),
            (f"git apply --reject {patch_path}", True),
            (f"git apply --reject --ignore-space-change --ignore-whitespace {patch_path}", True),
            (f"git apply --reject --ignore-space-change --ignore-whitespace --allow-empty {patch_path}", True),
        ]

        last_output = ""
        for cmd, is_reject in strategies:
            output, rc = self.run(cmd, workdir=self.repo_path)
            last_output = output
            rc_str = str(rc).strip()
            self.logger.info(f"[swe-pro] _apply_patch_with_fallback: cmd={cmd!r} rc={rc_str} out={output}")
            if rc_str == "0":
                self.logger.info(f"[swe-pro] _apply_patch_with_fallback: success with cmd={cmd!r}")
                return True, output
            if is_reject and rc_str == "1":
                self.logger.info(f"[swe-pro] _apply_patch_with_fallback: partial success with cmd={cmd!r}")
                return True, output

        self.logger.warning(f"[swe-pro] _apply_patch_with_fallback: all strategies failed, last_output={last_output}")
        return False, last_output


class SWERebenchV2Docker(DockerRuntime):
    """
    SWE-rebench-V2 Docker runtime.

    Integrates the evaluation logic from SWE-rebench-V2/scripts/eval.py into R2E-Gym.
    This runtime supports:
    - Dynamic log parser selection based on install_config
    - Direct integration with eval.py's parsing logic
    - Flexible test command execution
    - Standardized test result parsing across multiple frameworks
    """

    def __init__(
        self,
        ds,  # dataset entry
        repo_path: str = None,
        alt_path: str = "/root",
        docker_image: str = None,
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        """
        Initialize SWERebenchV2Docker runtime.

        Args:
            ds: Dataset entry containing task specification
            repo_path: Path to repository in container (auto-detected from ds.repo)
            alt_path: Alternative path for scripts
            docker_image: Docker image to use (auto-detected from ds.image_name)
            command: Shell command
            logger: Logger instance
            backend: Backend type ("docker" or "kubernetes")
            **docker_kwargs: Additional docker arguments
        """
        # Auto-detect repo_path from ds.repo if not provided
        if repo_path is None:
            repo = ds.get("repo", "")
            if "/" in repo:
                repo_name = repo.split("/")[1]
                repo_path = f"/{repo_name}"
            else:
                repo_path = "/app"

        # Initialize cache for log parser BEFORE calling super().__init__
        # because super().__init__() calls setup_env() which needs this attribute
        self._log_parser = None

        # Setup path to SWE-rebench-V2 lib BEFORE calling super().__init__
        # because super().__init__() calls setup_env() which needs to import log_parsers
        self._setup_rebench_path()

        super().__init__(ds, repo_path, alt_path, docker_image, command, logger, backend, **docker_kwargs)
        self.default_entrypoint_shell_type = "/bin/bash"

    def _setup_rebench_path(self):
        """Setup Python path to include SWE-rebench-V2 libraries."""
        import sys
        from pathlib import Path

        # Find SWE-rebench-V2 directory.
        # __file__ = R2E-Gym/src/r2egym/agenthub/runtime/docker.py
        # parents[4] = R2E-Gym/, where third_party/SWE-rebench-V2 lives.
        rebench_dir = Path(__file__).parent.parent.parent.parent.parent / "third_party" / "SWE-rebench-V2"
        if rebench_dir.exists():
            sys.path.insert(0, str(rebench_dir))
            sys.path.insert(0, str(rebench_dir / "lib"))
            # Only log if logger is available (after super().__init__)
            if hasattr(self, 'logger'):
                self.logger.info(f"Added SWE-rebench-V2 to path: {rebench_dir}")
        else:
            # Only log if logger is available (after super().__init__)
            if hasattr(self, 'logger'):
                self.logger.warning(f"SWE-rebench-V2 directory not found: {rebench_dir}")

    def _normalize_test_name(self, name: str) -> str:
        """
        Normalize test name by removing timing suffixes.
        Copied from eval.py to maintain consistency.
        """
        # Remove [1.34 ms] style timing
        name = re.sub(r"\s*\[\s*\d+(?:\.\d+)?\s*ms\s*\]", "", name)
        # Remove "in 29.08 msec" style timing
        name = re.sub(r"\s+in\s+\d+(?:\.\d+)?\s+(?:m?s|sec)", "", name)
        # Remove (1.234s) style timing
        name = re.sub(r"\s*\(\s*\d+(?:\.\d+)?\s*s\s*\)", "", name)
        return name.strip()

    def get_log_parser(self):
        """
        Get the log parser function for this task.

        Returns:
            Callable log parser function, or None if not found
        """
        if self._log_parser is not None:
            return self._log_parser

        try:
            from agent import log_parsers
        except ImportError:
            self.logger.error("Could not import log_parsers from SWE-rebench-V2/lib/agent")
            return None

        # Get parser name from install_config
        install_config = self.ds.get("install_config", {})
        parser_name = install_config.get("log_parser")

        if not parser_name:
            self.logger.error("No log_parser specified in install_config")
            return None

        # Look up parser in NAME_TO_PARSER map, with fallback to module attribute (mirrors eval.py)
        parser_fn = log_parsers.NAME_TO_PARSER.get(parser_name)
        if parser_fn is None:
            parser_fn = getattr(log_parsers, parser_name, None)

        if parser_fn is None:
            self.logger.error(f"Unknown log_parser: {parser_name}")
            return None

        self._log_parser = parser_fn
        self.logger.info(f"Using log parser: {parser_name}")
        return self._log_parser

    def remove_added_files_by_patch(self, patch):
        """Remove files that were added by a patch"""
        if not patch:
            return
        added_files = self.get_added_files_from_patch(patch)
        for file in added_files:
            self.run(f"rm -rf {self.repo_path}/{file}")

    def setup_env(self):
        """
        Setup the environment for SWE-rebench-V2 evaluation.

        This includes:
        1. Resetting to base commit
        2. Removing test patch files
        3. Installing Python packages (chardet)
        4. Fixing Django run_tests.sh if needed
        5. Removing future git info
        6. Preparing for test execution
        """
        try:
            instance_id = self.ds.get("instance_id", "")
            base_commit = self.ds.get("base_commit", "")

            self.logger.info(f"Setting up environment for {instance_id}")

            # Step 1: Reset to base commit
            if base_commit:
                self.run(f"git reset --hard {base_commit}", timeout=60)


            # Step 2: Remove added files from test_patch if they exist
            if "test_patch" in self.ds and self.ds["test_patch"]:
                try:
                    self.remove_added_files_by_patch(self.ds["test_patch"])
                except Exception as e:
                    self.logger.warning(f"Failed to remove test_patch files: {e}")

            # Step 3: Create /root/.venv/bin/python symlink (required by execute_bash.py shebang)
            self.run("mkdir -p /root/.venv/bin")
            self.run("ln -s $(which python) /root/.venv/bin/python")
            self.run("chmod +x /root/.venv/bin/python")

            # Step 3.5: Install curl / wget / jq (skipped when already present)
            self.ensure_cli_tools()

            # Step 4: Install chardet
            try:
                self.install_python_package_with_retry(
                    package_name="chardet",
                    max_retries=3,
                    raise_on_failure=True
                )
                self.logger.info("chardet installed successfully")
            except Exception as e:
                self.logger.warning(f"Failed to install chardet: {e}")

            # Step 5: Proxy variables (http_proxy/https_proxy) injected by the backend are unset
            # directly inside _execute_test_commands before running the test command, so that
            # they don't leak into test processes. No action needed here.

            # Step 6: 清理未来 git 信息（通过环境变量 R2E_REMOVE_GIT_INFO 控制，默认开启）
            # 错误处理与其它 *Docker 类保持一致：不包 try/except，让 self.run 抛出的
            # PodUnavailableError 等致命错穿透到上层（容器被 kill），由 rllm rollout
            # mask 逻辑处理；脚本意外缺失时记录 error 日志，与 YiYanSWEFactoryImagesDocker 对齐。
            if os.getenv("R2E_REMOVE_GIT_INFO", "true").lower() == "true":
                remove_git_script = os.path.join(os.path.dirname(r2egym.__file__), "agenthub/tools/remove_git_info.sh")
                if os.path.exists(remove_git_script):
                    self.copy_to_container(remove_git_script, "/remove_git_info.sh")
                    remove_tag = "pytest-dev/pytest" not in self.ds["repo"]
                    self.run(
                        f"bash /remove_git_info.sh {self.ds['base_commit']} --remove_tag {str(remove_tag).lower()}",
                        timeout=1200,
                    )
                else:
                    self.logger.error(
                        f"[SCRIPT NOT FOUND] pod={self.container_name} "
                        f"image={self.docker_image} path={remove_git_script}"
                    )

            # Step 7: Verify log parser is available
            parser = self.get_log_parser()
            if parser is None:
                raise ValueError(f"Could not load log parser for {instance_id}")

            self.logger.info(f"Environment setup completed for {instance_id}")

        except Exception as e:
            if isinstance(e, PodUnavailableError):
                raise
            # 外层 try 行为对齐 YiYanSWEFactoryImagesDocker 等老类：吞异常仅记 ERROR 日志。
            # PodUnavailableError 等致命错由 step 阶段 self.runtime.run 二次触发，经
            # RepoEnv.run_action 上抛到 rllm distributed_task_executor 的 PodUnavailableError
            # 兜底（覆盖 envprep / step / reward 三阶段），最终 mask 行为完全一致。
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )

    def _normalize_command_list(self, cmds):
        """
        Normalize test command to list format.

        Args:
            cmds: Command as string or list

        Returns:
            List of command strings
        """
        if isinstance(cmds, str):
            return [cmds]
        elif isinstance(cmds, list):
            return [item for item in cmds if isinstance(item, str) and item.strip()]
        else:
            raise ValueError(f"Invalid test_cmd type: {type(cmds)}")

    def _execute_test_commands(self, timeout: int = 300):
        """
        Execute test commands and capture output.

        Args:
            timeout: Timeout for test execution in seconds

        Returns:
            Tuple of (output_string, exit_code)
        """
        install_config = self.ds.get("install_config", {})
        test_cmds = install_config.get("test_cmd", [])

        if not test_cmds:
            raise ValueError("No test_cmd specified in install_config")

        test_cmds = self._normalize_command_list(test_cmds)

        # Combine with newline (mirrors eval.py's script = "\n".join(cmd_lines))
        # Using && would cause later test_cmds to be skipped if an earlier one exits non-zero
        combined_cmd = "\n".join(test_cmds)

        # Prepend _JAVA_OPTIONS to avoid IPv6 preference issues in Java projects (mirrors eval.py)
        combined_cmd = f"export _JAVA_OPTIONS=-Djava.net.preferIPv6Addresses=false\n{combined_cmd}"

        # Unset proxy environment variables so they don't leak into test processes.
        # Backend (Kubernetes/RLE) injects http_proxy/https_proxy into the container, which can
        # trigger proxy-agent code paths in projects that conditionally import proxy packages
        # (e.g. intuit/auto imports https-proxy-agent only when a proxy URL env var is set).
        # If the package wasn't installed at base_commit time this causes ReferenceError at runtime.
        # combined_cmd = (
        #     "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY "
        #     "no_proxy NO_PROXY REQUESTS_CA_BUNDLE\n"
        #     + combined_cmd
        # )

        self.logger.info(f"Executing test commands: {combined_cmd}")

        try:
            output, exit_code = self.run(combined_cmd, timeout=timeout)
            return output, exit_code
        except Exception as e:
            self.logger.error(f"Test execution failed: {e}")
            # Return empty string so log parser gets clean input rather than exception text
            return "", -1

    def _parse_test_output(self, output: str):
        """
        Parse test output using the configured log parser.

        Args:
            output: Raw test output

        Returns:
            Dictionary mapping test_name -> status (PASSED/FAILED/SKIPPED/ERROR)
        """
        parser = self.get_log_parser()
        if parser is None:
            self.logger.error("No log parser available")
            return {}

        try:
            # Parse with the selected parser
            parsed = parser(output)

            # Normalize test names (remove timing)
            normalized = {self._normalize_test_name(k): v for k, v in parsed.items()}

            self.logger.info(f"Parsed {len(normalized)} test results")
            return normalized

        except Exception as e:
            self.logger.error(f"Failed to parse test output: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {}

    def _get_expected_tests(self):
        """
        Get expected test sets from dataset.

        Returns:
            Tuple of (fail_to_pass, pass_to_pass, expected_passed)
        """
        # Get F2P and P2P from dataset
        f2p_raw = self.ds.get("FAIL_TO_PASS", [])
        p2p_raw = self.ds.get("PASS_TO_PASS", [])

        # Handle string format (e.g., "['test1', 'test2']")
        if isinstance(f2p_raw, str):
            try:
                f2p_raw = eval(f2p_raw)
            except Exception:
                f2p_raw = []

        if isinstance(p2p_raw, str):
            try:
                p2p_raw = eval(p2p_raw)
            except Exception:
                p2p_raw = []

        # Normalize test names
        fail_to_pass = set(self._normalize_test_name(t) for t in f2p_raw)
        pass_to_pass = set(self._normalize_test_name(t) for t in p2p_raw)
        expected_passed = fail_to_pass | pass_to_pass

        return fail_to_pass, pass_to_pass, expected_passed

    def _calculate_reward(self, get_test_output: bool = False, timeout: int = 1200):
        """
        Calculate reward based on test execution results.

        This method:
        1. Applies test_patch
        2. Applies model's patch (already applied by agent)
        3. Runs tests via test_cmd
        4. Parses results using log parser
        5. Compares with fail_to_pass and pass_to_pass requirements

        Args:
            get_test_output: If True, returns (reward, output) tuple
            timeout: Timeout for test execution in seconds

        Returns:
            Reward value (0.0 or 1.0), and optionally test output
        """
        try:
            instance_id = self.ds.get("instance_id", "")
            self.logger.info(f"Calculating reward for {instance_id}")
            new_file_paths = self._get_safe_new_file_paths_from_test_patch()
            self._remove_new_files_from_test_patch(new_file_paths)

            # Step 1: Apply test_patch if it exists
            test_patch = self.ds.get("test_patch", "")
            try:
                if test_patch:
                    self.logger.info("Applying test_patch...")
                    try:
                        self.apply_patch(test_patch)
                    except Exception as e:
                        self.logger.warning(f"Failed to apply test_patch: {e}")

                # Step 2: Execute tests
                self.logger.info("Executing tests...")
                output, exit_code = self._execute_test_commands(timeout=timeout)
            finally:
                self._remove_new_files_from_test_patch(new_file_paths)

            # Step 3: Parse test results
            self.logger.info("Parsing test results...")
            parsed_results = self._parse_test_output(output)

            if not parsed_results:
                self.logger.error("No test results parsed from output")
                if get_test_output:
                    return 0.0, output
                else:
                    return 0.0

            # Step 4: Extract passed and failed tests
            passed_actual = set(k for k, v in parsed_results.items() if v == "PASSED")
            failed_actual = set(k for k, v in parsed_results.items() if v == "FAILED")

            self.logger.info(f"Tests passed: {len(passed_actual)}, failed: {len(failed_actual)}")

            # Step 5: Get expected tests
            fail_to_pass, pass_to_pass, expected_passed = self._get_expected_tests()

            self.logger.info(f"Expected F2P: {len(fail_to_pass)}, P2P: {len(pass_to_pass)}")

            # Step 6: Check if all expected tests passed
            passed_match = (passed_actual >= expected_passed)

            # Log detailed results
            if passed_match:
                self.logger.info("All expected tests passed")
            else:
                missing = expected_passed - passed_actual
                self.logger.warning(f"Missing passed tests: {missing}")

            # Calculate reward
            reward = 1.0 if passed_match else 0.0

            if get_test_output:
                # Build detailed output
                test_output = {
                    "instance_id": instance_id,
                    "reward": reward,
                    "exit_code": exit_code,
                    "passed_match": passed_match,
                    "passed_actual": sorted(list(passed_actual)),
                    "failed_actual": sorted(list(failed_actual)),
                    "fail_to_pass_expected": sorted(list(fail_to_pass)),
                    "pass_to_pass_expected": sorted(list(pass_to_pass)),
                    "raw_output": output,
                }
                return reward, json.dumps(test_output, indent=2)
            else:
                return reward

        except Exception as e:
            self.logger.error(f"Error calculating reward: {repr(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

            if get_test_output:
                return 0.0, f"Error: {repr(e)}"
            else:
                return 0.0

    def reset_tests(self):
        """
        Reset tests to base state by removing test patch.
        """
        test_patch = self.ds.get("test_patch", "")
        if test_patch:
            try:
                self.remove_added_files_by_patch(test_patch)
                self.logger.info("Reset tests to base state")
            except Exception as e:
                self.logger.warning(f"Failed to reset tests: {e}")

    def post_process_for_template(self, system_prompt: str, instance_prompt: str):
        """
        Post-process prompts to replace placeholders.

        Args:
            system_prompt: System prompt template
            instance_prompt: Instance prompt template

        Returns:
            Tuple of (processed_system_prompt, processed_instance_prompt)
        """
        # Replace common placeholders
        repo = self.ds.get("repo", "")
        if "/" in repo:
            repo_name = repo.split("/")[1]
            repo_path = f"/{repo_name}"
        else:
            repo_path = self.repo_path

        system_prompt = system_prompt.replace("/testbed", repo_path)
        instance_prompt = instance_prompt.replace("/testbed", repo_path)

        return system_prompt, instance_prompt


class ScaleSWEDocker(DockerRuntime):
    """
    ScaleSWE Docker runtime.

    Ported as-written from the reference implementation (~/docker.py:5004). Two
    deviations, both forced by this vendored tree:
      * ``remove_git_info`` does not exist on the vendored ``DockerRuntime``
        (every runtime here inlines the script copy+run), so it is defined below
        with the reference base-class body.
      * this tree's ``DockerRuntimeMap`` gains the ``"scale-swe"`` key.

    评测流程：
      1. setup_env: 执行 pre_commands（含 git checkout parent_commit）、上传 f2p_script
      2. _calculate_reward:
         a. 若有 f2p_patch，git apply 引入失败测试
         b. 合并 FAIL_TO_PASS + PASS_TO_PASS，用注入的 pytest runner 一起跑
         c. 全部通过 → 1.0，否则 → 0.0（二元评分）

    数据字段说明：
      - ds["workdir"]         : 容器内 repo 路径，如 /workspace/auth0-python
      - ds["parent_commit"]   : base commit（替代 base_commit）
      - ds["pre_commands"]    : git 初始化命令串（含 checkout、建分支、清 history）
      - ds["f2p_patch"]       : 引入失败测试的 diff（可为空）
      - ds["f2p_script"]      : 上传为 test_fail_to_pass.py 的测试脚本
      - ds["FAIL_TO_PASS"]    : JSON 数组字符串，需通过的测试 ID
      - ds["PASS_TO_PASS"]    : JSON 数组字符串，不能回退的测试 ID
    """

    _PYTEST_RUNNER_SCRIPT = '''\
import json, sys, os
import pytest

if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        config = json.load(f)
    test_ids = config["test_ids"]
    xml_path = config.get("xml_path", "/tmp/_scaleswe_test_results.xml")
    sys.path.insert(0, os.getcwd())
    sys.argv = ["pytest"]
    args = ["-vv", f"--junitxml={xml_path}", "-o", "addopts=", "--rootdir=."] + test_ids
    ret = pytest.main(args)
    print("<pytest>true</pytest>" if ret == 0 else "<pytest>false</pytest>")
'''

    def __init__(
        self,
        ds,
        repo_path: str = None,
        alt_path: str = "/root",
        docker_image: str = None,
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        # 避免修改调用方传入的原始 dict
        ds = dict(ds)

        # workdir 字段直接作为 repo_path
        if repo_path is None:
            repo_path = ds.get("workdir", "/testbed")

        # parent_commit → base_commit 兼容（基类 __init__ 需要 base_commit）
        if "base_commit" not in ds and "parent_commit" in ds:
            ds["base_commit"] = ds["parent_commit"]

        # repo 字段：ScaleSWE 用 user/repo 两个字段，合并写回 ds["repo"]
        if "repo" in ds and "/" not in ds["repo"] and "user" in ds:
            ds["repo"] = f"{ds['user']}/{ds['repo']}"

        super().__init__(
            ds, repo_path, alt_path, docker_image, command, logger, backend, **docker_kwargs
        )

    @staticmethod
    def _parse_test_ids(raw) -> list:
        """解析测试 ID，支持 JSON 数组字符串 / Python list / 单个 ID 字符串。"""
        if not raw:
            return []
        if isinstance(raw, list):
            return [str(t).strip() for t in raw if t]
        raw = raw.strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(t).strip() for t in parsed if t]
            if isinstance(parsed, str) and parsed:
                return [parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return [raw]

    @staticmethod
    def _parse_pytest_summary(output: str):
        """
        解析 pytest 最后一行 summary，返回含 passed/failed/errors 及 all_passed 属性的对象。
        """
        _SUMMARY_LINE_RE = re.compile(
            r"\d+\s+(?:passed|failed|errors?|skipped|xfailed|xpassed)\b"
        )
        _COUNT_RE = re.compile(r"(\d+)\s+(\w+)")
        _LABEL_MAP = {
            "passed": "passed", "pass": "passed",
            "failed": "failed", "fail": "failed", "failure": "failed", "failures": "failed",
            "error": "errors", "errors": "errors",
            "skipped": "skipped",
        }

        counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0}
        summary_line = ""
        for line in output.splitlines():
            if _SUMMARY_LINE_RE.search(line):
                summary_line = line  # 取最后一行匹配

        for m in _COUNT_RE.finditer(summary_line):
            label = m.group(2).lower()
            field = _LABEL_MAP.get(label)
            if field:
                counts[field] = int(m.group(1))

        class _Summary:
            def __init__(self, c):
                self.passed = c["passed"]
                self.failed = c["failed"]
                self.errors = c["errors"]

            @property
            def all_passed(self):
                return self.failed == 0 and self.errors == 0 and self.passed > 0

        return _Summary(counts)

    def remove_git_info(self):
        # 清理未来 git 信息（通过环境变量 R2E_REMOVE_GIT_INFO 控制，默认开启）
        # 参考实现挂在基类上；此 vendored 树的基类没有该方法，故原样搬到本类。
        if os.getenv("R2E_REMOVE_GIT_INFO", "true").lower() == "true":
            if "base_commit" not in self.ds:
                self.logger.warning("`base_commit` not found in dataset, skipping git cleanup")
                return

            remove_git_script = os.path.join(os.path.dirname(r2egym.__file__), "agenthub/tools/remove_git_info.sh")
            if os.path.exists(remove_git_script):
                self.copy_to_container(remove_git_script, "/remove_git_info.sh")
                remove_tag = "pytest-dev/pytest" not in self.ds["repo"]
                self.run(f"bash /remove_git_info.sh {self.ds['base_commit']} --remove_tag {str(remove_tag).lower()}", timeout=1200)
            else:
                self.logger.error(f"[SCRIPT NOT FOUND] pod={self.container_name} image={self.docker_image} path={remove_git_script}")

    def apply_patch(self, patch: str) -> tuple[str, str]:
        """
        覆盖父类 apply_patch，使用带 fallback 的策略。

        父类只用 git apply --whitespace=nowarn，当 gold patch 里包含与 f2p_patch
        相同的 hunk 时（f2p_patch 已 applied），git apply 报 'patch already applied'
        直接失败，导致源码文件的改动也不会 apply。

        使用 --3way 或 --ignore-whitespace 等 fallback，让 already-applied 的 hunk
        被跳过，其余 hunk（源码文件）照常 apply。
        """
        success, output = self._apply_patch_with_fallback(patch)
        # 返回格式对齐父类 (output, error_code)
        return output, "0" if success else "1"

    def _apply_patch_with_fallback(self, patch: str) -> tuple[bool, str]:
        """
        带 fallback 的 patch 应用策略。

        策略顺序：
          1. git apply --3way --whitespace=nowarn
          2. git apply --whitespace=nowarn
          3. git apply --whitespace=nowarn --ignore-space-change --ignore-whitespace
          4. patch --batch --fuzz=5 -p1
          5. git apply --reject（部分 apply，exit=1 视为成功）
          6. git apply --reject --ignore-space-change --ignore-whitespace
          7. git apply --reject --ignore-space-change --ignore-whitespace --allow-empty

        Returns:
            (success: bool, output: str)
        """
        import uuid as _uuid
        patch_path = f"/tmp/_scaleswe_f2p_{_uuid.uuid4().hex[:8]}.patch"
        self.create_file(patch_path.lstrip("/"), patch)

        strategies = [
            # 优先用 --3way：当 hunk 已经 applied（与 f2p_patch 重叠时）会直接 skip 而非报错
            (f"git apply --3way --whitespace=nowarn {patch_path}", False),
            (f"git apply --whitespace=nowarn {patch_path}", False),
            (f"git apply --whitespace=nowarn --ignore-space-change --ignore-whitespace {patch_path}", False),
            (f"patch --batch --fuzz=5 -p1 -i {patch_path}", False),
            (f"git apply --reject {patch_path}", True),
            (f"git apply --reject --ignore-space-change --ignore-whitespace {patch_path}", True),
            (f"git apply --reject --ignore-space-change --ignore-whitespace --allow-empty {patch_path}", True),
        ]

        last_output = ""
        for cmd, is_reject in strategies:
            output, rc = self.run(cmd, workdir=self.repo_path)
            last_output = output
            rc_str = str(rc).strip()
            if rc_str == "0":
                return True, output
            if is_reject and rc_str == "1":
                # --reject 模式：部分 hunk 写入 .rej，视为可接受的局部成功
                return True, output

        return False, last_output

    def _run_pytest_with_runner(self, test_ids: list, timeout: int = 1800) -> tuple:
        """
        注入 pytest runner 脚本到容器并执行。

        流程：
          1. 上传 runner 脚本 → /tmp/_scaleswe_pytest_runner.py
          2. 上传测试配置 JSON → /tmp/_scaleswe_test_config.json
          3. 执行 runner
          4. 快速路径：检测 <pytest>true</pytest> 标记
          5. 兜底：解析 pytest summary 文本

        Returns:
            (all_passed: bool, output: str)
        """
        runner_path = "/tmp/_scaleswe_pytest_runner.py"
        config_path = "/tmp/_scaleswe_test_config.json"
        xml_path    = "/tmp/_scaleswe_test_results.xml"

        self.create_file(runner_path.lstrip("/"), self._PYTEST_RUNNER_SCRIPT)
        self.create_file(config_path.lstrip("/"),
                         json.dumps({"test_ids": test_ids, "xml_path": xml_path}))

        output, _rc = self.run(
            f"python {runner_path} {config_path}",
            timeout=timeout,
            workdir=self.repo_path,
        )

        # 快速路径
        if "<pytest>true</pytest>" in output:
            return True, output

        # 兜底：解析 summary
        summary = self._parse_pytest_summary(output)
        return summary.all_passed, output

    def setup_env(self):
        """
        容器启动后的环境初始化：
          1. 执行 pre_commands（git checkout parent_commit、建分支、清 history）
          2. 上传 f2p_script → workdir/test_fail_to_pass.py
          3. 安装 chardet
          4. 清除未来 git 信息（防止 agent 作弊）
        """
        try:
            # 1. pre_commands（数据末尾有 \n 字面量，先清理）
            pre_commands = self.ds.get("pre_commands", "")
            if isinstance(pre_commands, str):
                pre_commands = pre_commands.strip().rstrip("\\n").strip()
            if pre_commands:
                self.logger.info("Running pre_commands...")
                _out, rc = self.run(pre_commands, timeout=300, workdir=self.repo_path)
                self.logger.info(f"pre_commands exit: {rc}")

            # 2. 上传 f2p_script
            f2p_script = self.ds.get("f2p_script", "")
            if f2p_script:
                self.create_file(
                    f"{self.repo_path.lstrip('/')}/test_fail_to_pass.py",
                    f2p_script,
                )
                self.logger.info("Uploaded f2p_script → test_fail_to_pass.py")

            # 3. 创建 /root/.venv/bin/python 软链（execute_bash.py shebang 依赖此路径）
            self.run("mkdir -p /root/.venv/bin")
            self.run("ln -s $(which python) /root/.venv/bin/python")
            self.run("chmod +x /root/.venv/bin/python")

            # 4. 安装 curl / wget / jq（已存在则跳过；缺失时走可达的 baidu 镜像）
            self.ensure_cli_tools()

            # 6. 工具依赖
            self.install_python_package_with_retry("chardet", max_retries=3)

            # 7. 清除未来 git 信息
            self.remove_git_info()

        except Exception as e:
            self.logger.error(f"setup_env failed: {repr(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _calculate_reward(self, get_test_output: bool = False, timeout: int = 1800):
        """
        评测流程：
          1. apply f2p_patch（若有，引入失败测试）
          2. 合并 F2P + P2P test IDs，用 pytest runner 一起跑
          3. 全部通过 → 1.0，否则 → 0.0（二元评分）
        """
        try:
            f2p_patch = self.ds.get("f2p_patch", "")
            # Step 1: apply f2p_patch
            if f2p_patch and f2p_patch.strip():
                self.logger.info("Applying f2p_patch (with fallback strategies)...")
                success, _out = self._apply_patch_with_fallback(f2p_patch)
                if not success:
                    self.logger.error(f"f2p_patch apply failed: {_out[-300:]}")
                    if get_test_output:
                        return 0.0, f"f2p_patch apply failed: {_out}"
                    return 0.0

            # Step 2: 合并测试 ID
            f2p_ids = self._parse_test_ids(self.ds.get("FAIL_TO_PASS", ""))
            p2p_ids = self._parse_test_ids(self.ds.get("PASS_TO_PASS", ""))
            all_tests = f2p_ids + p2p_ids

            if not all_tests:
                self.logger.warning("No F2P/P2P test IDs found")
                if get_test_output:
                    return 0.0, "No test IDs"
                return 0.0

            self.logger.info(f"Running {len(f2p_ids)} F2P + {len(p2p_ids)} P2P tests...")

            # Step 3: 执行
            all_passed, output = self._run_pytest_with_runner(all_tests, timeout=timeout)
            reward = 1.0 if all_passed else 0.0
            self.logger.info(f"Reward={reward} (all_passed={all_passed})")

            if get_test_output:
                return reward, output
            return reward

        except Exception as e:
            self.logger.error(f"_calculate_reward failed: {repr(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            if get_test_output:
                return 0.0, f"Error: {repr(e)}"
            return 0.0

    def reset_tests(self):
        """重置测试态：逆向 f2p_patch，删除 test_fail_to_pass.py。"""
        f2p_patch = self.ds.get("f2p_patch", "")
        if f2p_patch and f2p_patch.strip():
            try:
                self.reverse_patch(f2p_patch)
            except Exception as e:
                self.logger.warning(f"Failed to reverse f2p_patch: {e}")
        self.run(f"rm -f {self.repo_path}/test_fail_to_pass.py")

    def post_process_for_template(self, system_prompt: str, instance_prompt: str):
        """替换 prompt 中 /testbed 占位符为实际 workdir。"""
        system_prompt = system_prompt.replace("/testbed", self.repo_path)
        instance_prompt = instance_prompt.replace("/testbed", self.repo_path)
        return system_prompt, instance_prompt



DockerRuntimeMap = {
    "swebench": SWEBenchDocker,
    "scale-swe": ScaleSWEDocker,
    "swebench-dev": SWEBenchDevDocker,
    "r2e": R2EDocker,
    "mswebench": MultiSWEBenchDocker,
    "swesmith": SWESmithDocker,
    "swegym": SWEGymDocker,
    "swe_bench_multilingual": SWEBenchMultilingualDocker,
    "yiyan-sweimages-factory": YiYanSWEFactoryImagesDocker,
    "swebench-pro": SWEBenchProDocker,
    "swe-rebenchv2": SWERebenchV2Docker,
}