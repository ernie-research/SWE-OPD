"""
ClaudeCodeAgent - Use Claude Code SDK as a black-box agent inside R2E-Gym.

We simply send it a prompt, let it run autonomously inside a Docker container,
and capture the output messages to convert into R2E-Gym's trajectory format.
"""

import os
import time
import json
import traceback
import requests, docker
from pathlib import Path
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from tempfile import NamedTemporaryFile

from r2egym.agenthub.agent.agent import AgentArgs
from r2egym.agenthub.action import Action
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from r2egym.agenthub.utils.log import get_logger

logger = get_logger(__name__)


class ClaudeCodeAgent:
    """
    A black-box agent that delegates everything to Claude Code SDK.

    Pipeline:
      1. Build prompt (system + instance)
      2. 在容器里运行 claude_sdk_runner.py（调 claude-agent-sdk）
      3. 读取 JSON 输出，转成 R2E-Gym 的 Trajectory
    """

    def __init__(self,
        name: str,
        args: AgentArgs,
        logger=None,
        claude_setting_path: str = "./third_party/claude_code/claude_settings.json",
        node_binary_path: str = "./third_party/claude_code/node-v24.11.0-linux-x64.tar.xz",
        claude_binary_path: str = "./third_party/claude_code/claude-code-2.0.36.tgz",
        claude_sdk_wheels_path: str = "./third_party/claude_code/claude_sdk_wheels.tar.gz",
        anthropic_api_key: str = None,
        anthropic_base_url: str = None,
    ):
        """Initialize ClaudeCodeAgent with the same interface as Agent."""
        self.name = name
        self.args = args
        self.logger = logger or get_logger(name)

        self.llm_name = args.llm_name
        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt

        self.claude_setting_path = claude_setting_path
        self.node_binary_path = node_binary_path  # Path to local Node.js binary tarball
        self.claude_binary_path = claude_binary_path  # Path to local Claude Code binary tarball
        self.claude_sdk_wheels_path = claude_sdk_wheels_path  # Path to claude-agent-sdk wheels tar.gz

        if anthropic_api_key is None:
            raise ValueError("anthropic_api_key is required for ClaudeCodeAgent")
        if anthropic_base_url is None:
            raise ValueError("anthropic_base_url is required for ClaudeCodeAgent")

        self.anthropic_api_key = anthropic_api_key
        self.anthropic_base_url = anthropic_base_url

        # Python 绝对路径，在 setup 时检测
        self.python_cmd = "/usr/bin/python3.10"  # 默认值，会在 setup 时更新为实际路径

        self.logger.info(f"Initialized ClaudeCodeAgent: {name} (black box mode)")
        self.logger.info(f"Claude Setting path: {claude_setting_path}")
        self.logger.info(f"Node.js binary path: {node_binary_path}")
        self.logger.info(f"Claude Code binary path: {claude_binary_path}")
        self.logger.info(f"Anthropic Base URL: {anthropic_base_url}")
    
    def _run_simple(self, runtime: "DockerRuntime", cmd: str, timeout: int = 300, desc: str = "") -> Tuple[str, str]:
        """统一封装 runtime.run + 错误日志。"""
        if desc:
            self.logger.info(f"[Run] {desc}: {cmd}")
        out, code = runtime.run(cmd, timeout=timeout)
        if code != "0":
            self.logger.error(f"[Run-Fail] {desc} (exit={code}): {out[-400:]}")
        return out, code

    def _is_python_version_valid(self, version_str: str) -> bool:
        """验证 Python 版本是否 >= 3.10"""
        import re
        # 解析版本号，例如 "Python 3.11.2"
        match = re.search(r'Python (\d+)\.(\d+)', version_str)
        if not match:
            return False

        major, minor = int(match.group(1)), int(match.group(2))
        return major == 3 and minor >= 10

    def _detect_python_cmd(self, runtime: "DockerRuntime") -> Optional[str]:
        """检测 Python 3.10+（支持特定版本和通用命令）"""
        # 检测顺序：优先特定版本，然后通用命令
        candidates = ["python3.10", "python3.11", "python3.12", "python3", "python"]

        for py_cmd in candidates:
            # 1. 检查命令是否存在
            out, code = runtime.run(f"which {py_cmd}", timeout=5)
            if code != "0":
                continue

            python_path = out.strip()

            # 2. 获取版本号
            version_out, version_code = runtime.run(f"{python_path} --version 2>&1", timeout=5)
            if version_code != "0":
                continue

            # 3. 验证版本 >= 3.10
            if self._is_python_version_valid(version_out):
                self.logger.info(f"✓ Found {py_cmd} at {python_path}, version: {version_out.strip()}")
                return python_path  # 返回绝对路径

        return None

    def _install_python310(self, runtime: "DockerRuntime") -> Optional[str]:
        """在线安装 Python 3.10"""
        self.logger.info("Installing python3.10...")
        runtime.run("apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv 2>&1 | tail -10", timeout=300)
        out, code = runtime.run("which python3.10", timeout=5)
        if code == "0":
            python_path = out.strip()
            self.logger.info(f"✓ Installed python3.10 at {python_path}")
            return python_path

        self.logger.error("Failed to install python3.10")
        return None

    def _ensure_pip_available(self, runtime: "DockerRuntime", python_cmd: str) -> bool:
        """确保 pip 可用，如果不可用则尝试安装"""
        # 1. 检查 pip 是否已可用
        out, code = runtime.run(f"{python_cmd} -m pip --version 2>&1", timeout=5)
        if code == "0":
            self.logger.info(f"✓ pip is available: {out.strip()}")
            return True

        self.logger.info("pip not found, attempting to install...")

        # 2. 尝试使用 ensurepip（Python 自带模块）
        self.logger.info("Trying ensurepip...")
        runtime.run(f"{python_cmd} -m ensurepip 2>&1 | tail -5", timeout=60)
        out, code = runtime.run(f"{python_cmd} -m pip --version 2>&1", timeout=5)
        if code == "0":
            self.logger.info(f"✓ pip installed via ensurepip: {out.strip()}")
            return True

        # 3. 尝试通过 apt-get 安装 python3-pip
        self.logger.info("Trying apt-get install python3-pip...")
        runtime.run("apt-get update -qq && apt-get install -y -qq python3-pip 2>&1 | tail -10", timeout=300)
        out, code = runtime.run(f"{python_cmd} -m pip --version 2>&1", timeout=5)
        if code == "0":
            self.logger.info(f"✓ pip installed via apt-get: {out.strip()}")
            return True

        self.logger.error("Failed to install pip")
        return False

    def setup_claude_code_env(
        self,
        env: "RepoEnv",
        force_reinstall: bool = False,
    ) -> bool:
        """
        在容器里安装：Node → CLI → DNS → settings.json → SDK
          1. Node.js（本地 tar.xz 拷进去解压）
          2. Claude Code CLI（npm -g 安装 .tgz）
          3. claude-agent-sdk（python 包）

        在容器里用一个 marker 文件避免重复安装 (force_reinstall=True)
        """
        runtime = env.runtime
        marker = "/root/.claude_setup_done"
        
        # 已经装过且不强制重装，直接复用
        if not force_reinstall:
            out, code = self._run_simple(
                runtime,
                f"test -f {marker} && echo OK || echo NO",
                timeout=10,
                desc="Check Claude env marker",
            )
            if code == "0" and "OK" in out:
                self.logger.info("Claude Code environment already set up, skip install.")
                return True

        self.logger.info("Setting up Claude Code environment in Docker container...")

        try:
            # ---------- Step 1: Node.js ----------
            if not os.path.exists(self.node_binary_path):
                self.logger.error(f"Node.js binary not found at: {self.node_binary_path}")
                return False
            
            node_binary_filename = os.path.basename(self.node_binary_path)
            node_tar_in_ctr = f"/root/{node_binary_filename}"
            node_dir_name = node_binary_filename.replace(".tar.xz", "")

            self.logger.info(f"Copying Node.js binary to container: {self.node_binary_path} -> {node_tar_in_ctr}")
            runtime.copy_to_container(self.node_binary_path, node_tar_in_ctr)

            cmds = [
                (f"tar -xf {node_tar_in_ctr} -C /root/", "Extract Node.js"),
                (f"ln -sf /root/{node_dir_name}/bin/node /usr/local/bin/node", "Link node"),
                (f"ln -sf /root/{node_dir_name}/bin/npm /usr/local/bin/npm", "Link npm"),
                (f"ln -sf /root/{node_dir_name}/bin/npx /usr/local/bin/npx", "Link npx"),
                ("node --version && npm --version", "Check node/npm"),
            ]
            for cmd, desc in cmds:
                _, code = self._run_simple(runtime, cmd, timeout=300, desc=desc)
                if code != "0":
                    return False
            
            self.logger.info("✓ Node.js installed successfully from local binary")

            # ---------- Step 2: Claude Code CLI ----------
            if not os.path.exists(self.claude_binary_path):
                self.logger.error(f"Claude Code tgz not found: {self.claude_binary_path}")
                return False
            
            claude_tgz_in_ctr = f"/root/{os.path.basename(self.claude_binary_path)}"
            runtime.copy_to_container(self.claude_binary_path, claude_tgz_in_ctr)

            _, code = self._run_simple(runtime, f"npm install -g {claude_tgz_in_ctr}", timeout=900, desc="Install Claude Code CLI")
            if code != "0":
                return False
            
            _, code = self._run_simple(runtime, f"ln -sf /root/{node_dir_name}/bin/claude /usr/local/bin/claude", timeout=30, desc="Link claude into PATH")
            if code != "0":
                return False

            out, code = self._run_simple(runtime, "claude --version", timeout=30, desc="Check claude --version")
            if code != "0":
                # 简单打印搜索结果，不再额外判断
                self._run_simple(runtime, "find /root -name claude -type f 2>/dev/null | head -5", timeout=30, desc="Search claude binary")
                return False
            
            self.logger.info(f"✓ Claude Code CLI installed and verified: {out.strip()}")

            # ---------- Step 3: Configure DNS to match host for internal domain resolution ----------
            try:
                with open('/etc/resolv.conf', 'r') as f:
                    host_resolv = f.read()
                self.logger.info(f"Host DNS config:\n{host_resolv}")

                # Copy to container
                with NamedTemporaryFile(mode='w', delete=False, suffix='.conf') as f:
                    f.write(host_resolv)
                    temp_resolv = f.name
                size = os.path.getsize(temp_resolv)
                self.logger.info(f"[temp_resolv] path={temp_resolv}, size={size} bytes")
                
                runtime.copy_to_container(temp_resolv, "/tmp/resolv.conf.host")
                os.unlink(temp_resolv)
                
                for cmd, desc in [
                    ("cp /etc/resolv.conf /etc/resolv.conf.backup", "Backup container resolv.conf"),
                    ("cp /tmp/resolv.conf.host /etc/resolv.conf", "Replace container resolv.conf"),
                    ("cat /etc/resolv.conf", "Show container resolv.conf"),
                ]:
                    _, code = self._run_simple(runtime, cmd, timeout=30, desc=desc)
                    if code != "0":
                        return False
            except Exception as e:
                self.logger.error(f"Failed to configure DNS: {e}")
                self.logger.error("Traceback: " + traceback.format_exc())
                return False

            # ---------- Step 4: 写 ~/.claude/settings.json + API key helper ----------
            api_key = self.anthropic_api_key
            base_url = self.anthropic_base_url

            # 加载本地设置文件
            if not os.path.exists(self.claude_setting_path):
                self.logger.error(f"Local claude_settings.json not found: {self.claude_setting_path}")
                return False
            
            # 先把本地 settings 文件拷到容器临时路径，再在容器里移动到 ~/.claude
            container_settings_tmp = "/tmp/claude_settings.json"
            runtime.copy_to_container(self.claude_setting_path, container_settings_tmp)
            
            cmds = [
                ("mkdir -p ~/.claude", "Create ~/.claude"),
                (f"cp {container_settings_tmp} ~/.claude/settings.json", "Write settings.json"),
                (f'echo "echo \\"{api_key}\\"" > ~/.claude/anthropic_key.sh', "Write API key helper"),
                ("chmod +x ~/.claude/anthropic_key.sh", "Make key helper executable"),
                ("cat ~/.claude/settings.json", "Show settings.json"),
            ]
            for cmd, desc in cmds:
                _, code = self._run_simple(runtime, cmd, timeout=30, desc=desc)
                if code != "0":
                    return False
            self.logger.info(f"✓ Claude settings.json and API key configured (from {self.claude_setting_path})")

            # ---------- Step 5: 检测/安装 Python 3.10+ ----------
            python_cmd = self._detect_python_cmd(runtime)
            if python_cmd is None:
                python_cmd = self._install_python310(runtime)
                if python_cmd is None:
                    self.logger.error("No Python 3.10+ available, cannot install Claude SDK")
                    return False

            # 保存为实例变量，供后续使用
            self.python_cmd = python_cmd

            # ---------- Step 5.5: 确保 pip 可用 ----------
            if not self._ensure_pip_available(runtime, python_cmd):
                self.logger.error("pip is not available, cannot install Claude SDK")
                return False

            # ---------- Step 6: 安装 claude-agent-sdk (离线安装 - 使用 tar.gz) ----------
            # 检查本地 wheels tar 包
            if not os.path.exists(self.claude_sdk_wheels_path):
                self.logger.error(f"Claude SDK wheels tar.gz not found: {self.claude_sdk_wheels_path}")
                return False

            # 拷贝 tar 包到容器
            wheels_tar_name = os.path.basename(self.claude_sdk_wheels_path)
            wheels_tar_in_ctr = f"/root/{wheels_tar_name}"
            self.logger.info(f"Copying wheels tar.gz to container: {self.claude_sdk_wheels_path} -> {wheels_tar_in_ctr}")
            runtime.copy_to_container(self.claude_sdk_wheels_path, wheels_tar_in_ctr)

            # 解压 tar 包
            _, code = self._run_simple(
                runtime, f"tar -xzf {wheels_tar_in_ctr} -C /root/", timeout=60, desc="Extract wheels tar.gz"
            )
            if code != "0":
                return False

            # 离线安装（优先）
            wheels_in_ctr = "/root/claude_sdk_wheels"
            pip_cmd_offline = f"{python_cmd} -m pip install --no-index --find-links={wheels_in_ctr} claude-agent-sdk==0.1.6"
            out, code = self._run_simple(
                runtime, pip_cmd_offline, timeout=900, desc="Install claude-agent-sdk (offline)"
            )

            if code != "0":
                # 离线安装失败，尝试在线安装
                self.logger.warning("Offline install failed, trying online install...")
                pip_cmd_online = f"{python_cmd} -m pip install claude-agent-sdk==0.1.6"
                _, code = self._run_simple(
                    runtime, pip_cmd_online, timeout=900, desc="Install claude-agent-sdk (online)"
                )
                if code != "0":
                    self.logger.error("Both offline and online install failed")
                    return False
                self.logger.info("✓ Installed claude-agent-sdk online")
            else:
                self.logger.info("✓ Installed claude-agent-sdk offline")

            # 验证安装
            out, code = self._run_simple(
                runtime,
                f"{python_cmd} -c 'import claude_agent_sdk; print(claude_agent_sdk.__version__)'",
                timeout=30,
                desc="Check claude_agent_sdk import",
            )
            if code == "0":
                self.logger.info(f"✓ Claude Agent SDK installed: {out.strip()}")
            else:
                self.logger.warning("claude_agent_sdk import failed, but continuing (see logs above)")
            
            self.logger.info(f"{'=' * 80}\n✓ Claude Code environment setup completed successfully!\n{'=' * 80}")
            self._run_simple(runtime, f"touch {marker}", timeout=10, desc="Mark setup done")

            # ---------- Step 6: 容器 request 测试 ----------
            test_script_path = "./my_scripts/test_rle_api.py"
            if os.path.exists(test_script_path):
                self.logger.info("Testing service connectivity...")
                try:
                    # 复制测试脚本到容器
                    runtime.copy_to_container(test_script_path, "/tmp/test_request.py")
                    self.logger.info(f"Copied {test_script_path} to /tmp/test_request.py in container")

                    # 运行测试
                    out, code = self._run_simple(
                        runtime,
                        f"{python_cmd} /tmp/test_request.py --in-container",
                        timeout=60,
                        desc="Test service connectivity"
                    )

                    if code == "0" and "✅" in out:
                        self.logger.info("✓ Service connectivity test passed")
                    else:
                        self.logger.warning(f"Service connectivity test returned code {code}")
                        self.logger.warning(f"Test output: {out[-500:]}")
                except Exception as e:
                    self.logger.warning(f"Service connectivity test failed (non-critical): {e}")
            else:
                self.logger.warning(f"Test script not found at {test_script_path}, skipping service test")

            return True

        except Exception as e:
            self.logger.error(f"Error during Claude Code environment setup: {e}")
            self.logger.error("Traceback: " + traceback.format_exc())
            return False

    def _run_with_streaming(
        self,
        runtime: "DockerRuntime",
        command: str,
        timeout: int = 3600,
    ) -> Tuple[str, Optional[int]]:
        """
        在 Docker 容器里执行命令，并流式读 stdout/stderr。
        返回 (完整输出, exit_code)
        """
            # K8s/RLE 直接复用 runtime.run
        if runtime.backend != "docker":
            self.logger.warning(f"{runtime.backend} backend doesn't support streaming yet, falling back to regular run")
            output, exit_code_str = runtime.run(command, timeout=timeout)
            # 转换字符串 exit_code 为整数
            try:
                exit_code = int(exit_code_str) if exit_code_str.isdigit() or (exit_code_str.startswith('-') and exit_code_str[1:].isdigit()) else None
            except:
                exit_code = None
            return output, exit_code

        container = runtime.container  # docker.models.containers.Container
        client = container.client
        api = client.api

        # 取消代理，加入 PATH
        full_shell_cmd = (
            "unset http_proxy; unset https_proxy; "
            f"export PATH={runtime.docker_path}:$PATH; "
            f"{command}"
        )
        wrapped_cmd = ["timeout", str(timeout), "/bin/sh", "-lc", full_shell_cmd]

        self.logger.info("Note: Unsetting http_proxy/https_proxy for Claude SDK")
        self.logger.info(f"Streaming command in container: {wrapped_cmd}")

        output_chunks: List[str] = []
        exit_code: Optional[int] = None

        try:
            # 创建 exec 实例
            exec_create_res = api.exec_create(
                container=container.id,
                cmd=wrapped_cmd,
                workdir=getattr(runtime, "repo_path", None),
                stdout=True,
                stderr=True,
            )
            exec_id = exec_create_res.get("Id")
            if not exec_id:
                self.logger.error(f"Failed to create docker exec for command: {wrapped_cmd}")
                return "", -1

            stream = api.exec_start(exec_id, stream=True, demux=True)

            for stdout_chunk, stderr_chunk in stream:

                if stdout_chunk:
                    text = stdout_chunk.decode("utf-8", errors="replace")
                    output_chunks.append(text)
                    
                    if any(tag in text for tag in ["[CLAUDE_TEXT]", "[CLAUDE_TOOL_RESULT]", "[CLAUDE_COMPLETE]"]):
                        self.logger.info(text)
                        continue
                    
                    for line in text.splitlines():
                        line = line.rstrip()
                        if not line:
                            continue
                        if line.startswith("[CLAUDE_"):
                            self.logger.info(line)
                        elif line.startswith("SUCCESS:"):
                            self.logger.info(f"✅ {line}")
                        elif line.startswith("ERROR:"):
                            self.logger.error(f"❌ {line}")
                        elif line.startswith("[WARNING]"):
                            self.logger.warning(line.replace("[WARNING]", "").strip())
                        elif "Traceback" in line or "Exception" in line or "Error" in line:
                            self.logger.error(line)
                        else:
                            self.logger.info(line)

                if stderr_chunk:
                    text = stderr_chunk.decode("utf-8", errors="replace")
                    output_chunks.append(text)
                    for line in text.splitlines():
                        line = line.rstrip()
                        if not line:
                            continue
                        if "Traceback" in line or "Exception" in line or "Error" in line:
                            self.logger.error(line)
                        else:
                            self.logger.warning(line)

            try:
                inspect_res = api.exec_inspect(exec_id)
                exit_code = inspect_res.get("ExitCode")
            except Exception as e_inspect:
                self.logger.error(f"Failed to inspect exec {exec_id}: {e_inspect}")
                exit_code = -1

        except requests.exceptions.ReadTimeout as e:
            self.logger.error(f"Docker exec_start ReadTimeout: {e}")
            exit_code = -1
        except docker.errors.APIError as e:
            self.logger.error(f"Docker APIError during streaming exec: {e}")
            exit_code = -1
        except Exception as e:
            self.logger.error(f"Unexpected error during streaming exec: {e}")
            self.logger.error("Traceback: " + traceback.format_exc())
            exit_code = -1

        full_output = "".join(output_chunks)
        self.logger.info(f"Streaming command finished: exit_code={exit_code}, output_len={len(full_output)}")
        return full_output, exit_code

    def _run_claude_code_in_container(
        self,
        runtime: "DockerRuntime",
        system: str,
        prompt: str,
        working_dir: str,
        max_steps_absolute: int = 50,
    ) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
        """
        在容器中运行 claude_sdk_runner.py，读取 JSON 输出

        Args:
            runtime: The Docker runtime to execute in
            prompt: The complete prompt (system + user)
            working_dir: The working directory for Claude Code to operate in
            max_steps_absolute: Hard limit on steps (for safety)

        Returns:
            Tuple of (message_list, exit_reason, stats)
        """
        self.logger.info("Running Claude Code SDK inside Docker container...")

        # 1. 拷 runner 到容器
        runner_script_path = os.path.join(os.path.dirname(__file__), "claude_sdk_runner.py")
        container_runner_path = "/tmp/.claude_sdk_runner.py"

        if not os.path.exists(runner_script_path):
            self.logger.error(f"Runner script not found: {runner_script_path}")
            return [], "setup_error", {}

        runtime.copy_to_container(runner_script_path, container_runner_path)

        # 2. 写 config.json，拷到容器
        config = {
            "system": system,
            "prompt": prompt,
            "model": self.llm_name,
            "api_key": self.anthropic_api_key,
            "base_url": self.anthropic_base_url,
            "max_steps": max_steps_absolute,
            "output_file": "/tmp/claude_output.json",
            "working_dir": working_dir,
        }

        with NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            tmp_cfg = f.name
        size = os.path.getsize(tmp_cfg)
        self.logger.info(f"[config.json] path={tmp_cfg}, size={size} bytes")
        container_cfg = "/tmp/.claude_config.json"
        try:
            runtime.copy_to_container(tmp_cfg, container_cfg)
        finally:
            os.unlink(tmp_cfg)
            pass

        # 3. 执行 claude_sdk_runner.py（流式）
        cmd = f"cd {working_dir} && {self.python_cmd} {container_runner_path} {container_cfg}"
        output, exit_code = self._run_with_streaming(runtime, cmd, timeout=3600 * 2)
        self.logger.info(f"Runner exit code: {exit_code!r}. Final streaming output length: {len(output)} chars")

        # 4. 根据 exit_code 处理
        if exit_code in (0, None):
            # 正常结束 -> 读 JSON 文件
            cat_output, cat_code = runtime.run("cat /tmp/claude_output.json", timeout=30)
            if cat_code != "0":
                self.logger.error(
                    f"Runner exit_code={exit_code!r} but cannot read output file "
                    f"(exit={cat_code}): {cat_output[-500:]}"
                )
                return [], "output_read_error", {}
            
            try:
                result = json.loads(cat_output)
            except Exception as e:
                self.logger.error(f"Failed to parse claude_output.json: {e}")
                self.logger.error(f"Raw tail (500 chars): {cat_output[-500:]}")
                return [], "output_parse_error", {}
            
            messages = result.get("messages", [])
            exit_reason = result.get("exit_reason", "agent")
            stats = result.get("stats", {})

            # 把 docker json 存到宿主机
            host_dir = Path(os.getenv("CLAUDE_OUTPUT_DIR", "./claude_outputs"))
            host_dir.mkdir(parents=True, exist_ok=True)
            instance_id = runtime.ds.get("instance_id")
            host_path = host_dir / f"claude_output_{instance_id}_{int(time.time())}.json"
            try:
                host_path.write_text(cat_output)
                self.logger.info(f"Saved Claude output JSON to {host_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save Claude output JSON: {e}")

            self.logger.info(
                f"Retrieved {len(messages)} messages from container "
                f"(exit_reason={exit_reason}, stats={stats})"
            )
            return messages, exit_reason, stats

        if exit_code == 124:
            self.logger.error("Runner hit timeout inside container (linux `timeout`)")
            self.logger.error(f"Partial output (last 500 chars):\n{output[-500:]}")
            return [], "runner_timeout", {}
        
        if exit_code == -1:
            self.logger.error("Runner failed with unknown docker/streaming error")
            self.logger.error(f"Command: {cmd}")
            self.logger.error(f"Partial output (last 500 chars):\n{output[-500:]}")
            return [], "runner_unknown", {}
        
        self.logger.error(f"Runner failed with with non-zero exit code {exit_code}")
        self.logger.error(f"Partial output (last 500 chars):\n{output[-500:]}")
        return [], "runner_error", {}

    def _convert_messages_to_trajectory(
        self,
        messages: List[Dict[str, Any]],
        stats: Dict[str, Any]
    ) -> List[TrajectoryStep]:
        """
        把 SDK 输出的消息转成 R2E-Gym trajectory steps
        约定：
        - AssistantMessage:
            - text => 累积为 thought
            - tool_use => 记录为待匹配的 action（pending_tool_uses）
        - UserMessage:
            - tool_result => 与对应的 tool_use_id 匹配，形成一个 TrajectoryStep
        """
        trajectory_steps: List[TrajectoryStep] = []
        step_idx = 0
        total_time_traj = 0.0

        # 当前累积的 “思考内容”，只来自 Assistant 的 text
        current_thought = ""

        # 挂起的 tool_use：tool_use_id -> { thought, action, timestamp }
        pending_tool_uses: Dict[str, Dict[str, Any]] = {}
        
        for msg_data in messages:
            msg_type = msg_data.get("type", "")
            content_blocks = msg_data.get("content") or []

            # 1) AssistantMessage：累积 thought + 记录 tool_use
            if msg_type == "AssistantMessage":
                for block in content_blocks:
                    block_type = block.get("type", "")

                    if block_type == "text":
                        # 只在 Assistant 中累积 thought
                        text = block.get("text", "")
                        if current_thought:
                            current_thought += "\n" + text
                        else:
                            current_thought = text

                    elif block_type == "tool_use":
                        tool_id = block.get("id", "")
                        if not tool_id:
                            continue
                        
                        action = Action(
                            function_name=block.get("name", ""),
                            parameters=(
                                block.get("input", {})
                                if isinstance(block.get("input"), dict)
                                else {}
                            ),
                        )

                        pending_tool_uses[tool_id] = {
                            "thought": current_thought.strip(),
                            "action": action,
                            "timestamp": msg_data.get("timestamp", time.time()),
                        }

                        # 风险：同一个 Assistant 里多个 tool_use 共享一段 thought
                        current_thought = ""
            
            # 2) UserMessage：只关心 tool_result，用 tool_use_id 匹配
            elif msg_type == "UserMessage":
                for block in content_blocks:
                    block_type = block.get("type", "")
                   
                    if block_type != "tool_result":
                        # UserMessage 中的 text 或其他内容不参与 thought 累积
                        continue
                    
                    tool_use_id = block.get("tool_use_id", "")
                    if not tool_use_id or tool_use_id not in pending_tool_uses:
                        # 找不到对应的 tool_use，说明这个 result 是 “孤立的”，跳过
                        continue
                    step_data = pending_tool_uses.pop(tool_use_id)

                    observation = block.get("content", "")
                    action: Action = step_data["action"]

                    # 简单估一个 step_time（也可以用 timestamp 做 delta，更精细）
                    step_time = 5.0
                    total_time_traj += step_time

                    is_done = action.function_name.lower() in ("finish", "submit")

                    trajectory_step = TrajectoryStep(
                        step_idx=step_idx,
                        thought=step_data["thought"],
                        action=action.to_xml_string(),
                        observation=observation,
                        done=is_done,
                        info={
                            "tool_use_id": tool_use_id,
                            "is_error": block.get("is_error", False),
                        },
                        token_usage_prompt=-1,
                        token_usage_completion=-1,
                        token_usage_total=-1,
                        llm_exec_time=step_time * 0.5,
                        env_exec_time=step_time * 0.5,
                        total_step_time=step_time,
                        total_time_traj=total_time_traj,
                        step_count=step_idx + 1,
                    )

                    trajectory_steps.append(trajectory_step)
                    step_idx += 1
            # 3) 其他类型（SystemMessage / ResultMessage / Error 等）这里不转为 step
        
        if pending_tool_uses:
            self.logger.error(
                f"{len(pending_tool_uses)} tool_use blocks had no matching tool_result; "
                "most likely run was interrupted or timed out."
            )
        self.logger.info(f"✅ SUCCESS: transform trajectory steps.")

        # Distribute total tokens across steps
        if trajectory_steps and stats.get("total_tokens", 0) > 0:
            tokens_per_step = stats["total_tokens"] // len(trajectory_steps)
            for step in trajectory_steps:
                step.token_usage_total = tokens_per_step
                step.token_usage_prompt = int(tokens_per_step * 0.7)
                step.token_usage_completion = int(tokens_per_step * 0.3)

        return trajectory_steps

    def _reconstruct_history(
        self,
        trajectory_steps: List[TrajectoryStep],
        system_prompt: str,
        user_prompt: str,
    ) -> List[Dict]:
        """Reconstruct conversation history from trajectory for compatibility."""
        history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        for step in trajectory_steps:
            action = Action.from_string(step.action)

            # tool call
            history.append({
                "role": "assistant",
                "content": step.thought,
                "tool_calls": [{
                    "id": f"call_{step.step_idx}",
                    "type": "function",
                    "function": {
                        "name": action.function_name,
                        "arguments": json.dumps(action.parameters),
                    }
                }]
            })

            # Tool result
            history.append({
                "role": "tool",
                "name": action.function_name,
                "content": step.observation,
                "tool_call_id": f"call_{step.step_idx}",
            })

        return history

    def run(
        self,
        env: "RepoEnv",
        use_fn_calling: bool = True,  # Ignored - Claude Code handles this internally
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        max_token_limit: int = 65536,
        max_exec_time: int = 90,
        max_total_time: int = 50000,
        max_llm_time: int = 7200,
        temperature: float = 0.0,
        top_p: float = 1.0,
        metadata: Optional[Dict[str, Any]] = {},
        scaffold: str = "claude_code",
    ) -> Tuple[Trajectory, List[Dict]]:
        """
        与标准 Agent.run() API 对齐，但内部走的是: send prompt → capture output → convert to trajectory

        Args:
            env: The RepoEnv environment
            max_steps: Soft limit (for display purposes)
            max_steps_absolute: Hard limit (safety check)
            Other parameters: Maintained for API compatibility but some are ignored

        Returns:
            Tuple of (Trajectory, history)
        """
        self.logger.info(f"=== Running ClaudeCodeAgent (BLACK BOX MODE) ===")
        start_time = time.time()

        # env.reset()

        # 第一次：正常走（如果 marker 存在会复用已有环境）
        if not self.setup_claude_code_env(env):
            self.logger.warning("First Claude Code setup failed, retrying with force_reinstall=True...")
            
            # 第二次：强制重装，忽略 marker，重新装 Node / Claude / SDK
            if not self.setup_claude_code_env(env, force_reinstall=True):
                self.logger.error("Failed to set up Claude Code environment. Abort.")
                empty_trajectory = Trajectory(
                    trajectory_steps=[],
                    problem_statement="Setup failed",
                    docker_image=env.runtime.docker_image,
                    agent_args=asdict(self.args),
                    env_args=asdict(env.args),
                    max_steps=max_steps,
                    max_steps_absolute=max_steps_absolute,
                    max_token_limit=max_token_limit,
                    max_llm_time=max_llm_time,
                    max_exec_time=max_exec_time,
                    max_total_time=max_total_time,
                    exit_reason="setup_error",
                    output_patch="",
                )
                history = [
                    {"role": "system", "content": self.system_prompt_template},
                    {"role": "user", "content": "Setup failed"},
                ]
                return empty_trajectory, history
        
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem: {problem_statement}...")

        # Get ground truth patch if available (for prompt context)
        gt_patch = ""
        if "fix_patch" in env.runtime.ds:
            gt_patch = env.runtime.ds["fix_patch"]
        elif "patch" in env.runtime.ds:
            gt_patch = env.runtime.ds["patch"]
        elif hasattr(env.runtime, "commit"):
            gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)
        
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            gt_patch=gt_patch,
            working_dir=env.runtime.repo_path,
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct" if metadata.get("candidate_patch_correctness", False) else "incorrect"
            ),
        )
        # full_prompt = f"{self.system_prompt_template}\n\n{user_prompt}"
        full_prompt = user_prompt

        if self.args.use_demo and self.args.demo_file:
            with open(self.args.demo_file, "r") as f:
                demo = f.read()
            full_prompt = f"{demo}\n\n{full_prompt}"

        self.logger.info(f"Prompt length: {len(full_prompt)} chars")
        self.logger.info(f"Working directory: {env.runtime.repo_path}")

        # === 调 Claude Code SDK ===
        try:
            messages, exit_reason, stats = self._run_claude_code_in_container(
                runtime=env.runtime,
                system=self.system_prompt_template,
                prompt=full_prompt,
                working_dir=env.runtime.repo_path,
                max_steps_absolute=max_steps_absolute,
            )
        except Exception as e:
            self.logger.error(f"Failed to run Claude Code: {e}")
            self.logger.error("Traceback: " + traceback.format_exc())
            messages, exit_reason, stats = [], "llm_query_error", {}
        
        trajectory_steps = self._convert_messages_to_trajectory(messages, stats)
        history = self._reconstruct_history(trajectory_steps, self.system_prompt_template, user_prompt)
        output_patch = env.runtime.get_patch()

        trajectory = Trajectory(
            trajectory_steps=[step.model_dump() for step in trajectory_steps],
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            agent_args=asdict(self.args),
            env_args=asdict(env.args),
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            exit_reason=exit_reason,
            output_patch=output_patch,
        )

        total_time = time.time() - start_time
        self.logger.info(f"=== ClaudeCodeAgent completed in {total_time:.2f}s ===")
        self.logger.info(f"Steps: {len(trajectory_steps)}, Exit: {exit_reason}")
        self.logger.info(f"Tokens: {stats.get('total_tokens', 'N/A')}, Cost: ${stats.get('cost_usd', 0):.4f}")

        return trajectory, history
