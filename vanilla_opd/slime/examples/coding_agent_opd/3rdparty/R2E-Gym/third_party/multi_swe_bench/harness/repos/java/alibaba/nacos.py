#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2025 Baidu.com, Inc. All Rights Reserved
#
########################################################################
"""
alibaba nacos
"""

import re
from typing import Optional

from multi_swe_bench.harness.image import Config
from multi_swe_bench.harness.image import File
from multi_swe_bench.harness.image import Image
from multi_swe_bench.harness.instance import Instance
from multi_swe_bench.harness.instance import TestResult
from multi_swe_bench.harness.pull_request import PullRequest

MAVEN_VERSION = "3.8.6"
MAVEN_SETTINGS_XML = """<?xml version="1.0" encoding="UTF-8"?><settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 http://maven.apache.org/xsd/settings-1.0.0.xsd"><mirrors><mirror><id>alimaven</id><mirrorOf>central</mirrorOf><name>阿里云公共仓库</name><url>https://maven.aliyun.com/repository/public</url></mirror></mirrors><profiles><profile><id>baidu</id><repositories><repository><id>baidu-public</id><url>https://maven.baidu-int.com/nexus/content/groups/public</url><releases><enabled>true</enabled></releases></repository><repository><id>Baidu_Local</id><url>https://maven.baidu-int.com/nexus/content/repositories/Baidu_Local</url><releases><enabled>true</enabled></releases></repository><repository><id>Baidu_Local_Snapshots</id><url>https://maven.baidu-int.com/nexus/content/repositories/Baidu_Local_Snapshots</url><snapshots><enabled>true</enabled><updatePolicy>always</updatePolicy></snapshots></repository></repositories><pluginRepositories><pluginRepository><id>baidu-public</id><url>https://maven.baidu-int.com/nexus/content/groups/public</url><releases><enabled>true</enabled></releases></pluginRepository><pluginRepository><id>Baidu_Local_Snapshots</id><url>https://maven.baidu-int.com/nexus/content/repositories/Baidu_Local_Snapshots</url><snapshots><enabled>true</enabled></snapshots></pluginRepository></pluginRepositories></profile></profiles><activeProfiles><activeProfile>baidu</activeProfile></activeProfiles></settings>"""

WORKDIR = "/home"


class NacosConfig:
    """Nacos 配置管理类"""

    # Java 和 Maven 配置
    JAVA_TOOL_OPTIONS = "-Dfile.encoding=UTF-8 -Duser.timezone=Asia/Shanghai"
    MAVEN_FAST_TEST = "mvn test -Dfast  -Dmaven.test.failure.ignore=true"
    MAVEN_INSTALL = "mvn install -DskipTests  -Dmaven.test.failure.ignore=true"

    # CentOS 环境配置
    BASE_IMAGE = "iregistry.baidu-int.com/baidu-base/openjdk:17-noble"
    JAVA_HOME = "/usr/lib/jvm/java-17-openjdk-amd64"


class NacosImageBase(Image):
    """Nacos CentOS 基础镜像"""

    def __init__(self, pr: PullRequest, config: Config):
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    @property
    def config(self) -> Config:
        return self._config

    def dependency(self) -> str:
        return NacosConfig.BASE_IMAGE

    def image_tag(self) -> str:
        return "base"

    def workdir(self) -> str:
        return "base"

    def files(self) -> list[File]:
        return []

    def dockerfile(self) -> str:
        """生成 CentOS 基础镜像 Dockerfile"""
        code_cmd = self._get_code_command()

        return f"""FROM {NacosConfig.BASE_IMAGE}

{self.global_env}

# 环境变量设置
ENV JAVA_TOOL_OPTIONS="{NacosConfig.JAVA_TOOL_OPTIONS}"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV JAVA_HOME={NacosConfig.JAVA_HOME}
ENV MAVEN_HOME={WORKDIR}/apache-maven-{MAVEN_VERSION}
ENV PATH=${{PATH}}:${{MAVEN_HOME}}/bin

WORKDIR {WORKDIR}

# 清理并更新系统
RUN apt-get clean && \
    apt-get update && \
    apt-get upgrade -y && \
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y build-essential git wget curl python3 python3-pip vim nodejs sudo

RUN curl -fsSL https://archive.apache.org/dist/maven/maven-3/{MAVEN_VERSION}/binaries/apache-maven-{MAVEN_VERSION}-bin.tar.gz | tar -xzv -C {WORKDIR}
RUN mkdir -p /root/.m2 && \
    printf '%s' '{MAVEN_SETTINGS_XML}' >  /root/.m2/settings.xml

{code_cmd}

# 提前拉取依赖

RUN cd /home/nacos && mvn clean install -DskipTests -Dmaven.test.failure.ignore=true -U -T 8C

{self.clear_env}
"""

    def _get_code_command(self) -> str:
        """获取源代码命令"""
        if self.config.need_clone:
            return f"RUN git clone https://github.com/{self.pr.org}/{self.pr.repo}.git /{WORKDIR}/{self.pr.repo}"
        return f"COPY    {self.pr.repo} {WORKDIR}/{self.pr.repo}"


# ========================================================================================
# 工作镜像类
# ========================================================================================

class NacosImageDefault(Image):
    """Nacos 工作镜像"""

    def __init__(self, pr: PullRequest, config: Config):
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    @property
    def config(self) -> Config:
        return self._config

    def dependency(self) -> Image:
        return NacosImageBase(self.pr, self._config)

    def image_tag(self) -> str:
        return f"pr-{self.pr.number}"

    def workdir(self) -> str:
        return f"pr-{self.pr.number}"

    def files(self) -> list[File]:
        """生成测试所需文件"""
        return [
            File(".", "fix.patch", f"{self.pr.fix_patch}"),
            File(".", "test.patch", f"{self.pr.test_patch}"),
            File(".", "check_git_changes.sh", self._git_check_script()),
            File(".", "prepare.sh", self._prepare_script()),
            File(".", "run.sh", self._run_script()),
            File(".", "test-run.sh", self._test_run_script()),
            File(".", "fix-run.sh", self._fix_run_script()),
        ]

    def dockerfile(self) -> str:
        """生成工作镜像 Dockerfile"""
        image = self.dependency()
        copy_commands = "\n".join(f"COPY    {f.name} {WORKDIR}" for f in self.files())

        return f"""FROM {image.image_name()}:{image.image_tag()}

ENV JAVA_TOOL_OPTIONS="{NacosConfig.JAVA_TOOL_OPTIONS}"
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV JAVA_HOME={NacosConfig.JAVA_HOME}
ENV MAVEN_HOME={WORKDIR}/apache-maven-{MAVEN_VERSION}
ENV PATH=${{PATH}}:${{MAVEN_HOME}}/bin


{self.global_env}

{copy_commands}
RUN chmod 775 {WORKDIR}/*.sh

RUN bash {WORKDIR}/prepare.sh

{self.clear_env}
"""

    def _git_check_script(self) -> str:
        """Git 状态检查脚本"""
        return """#!/bin/bash
set -e
if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "check_git_changes: Not inside a git repository"; exit 1
fi
if [[ -n $(git status --porcelain) ]]; then
  echo "check_git_changes: Uncommitted changes"; exit 1
fi
echo "check_git_changes: No uncommitted changes"
"""

    def _prepare_script(self) -> str:
        """环境准备脚本"""
        return f"""#!/bin/bash
set -e
cd {WORKDIR}/{self.pr.repo}
git reset --hard
bash {WORKDIR}/check_git_changes.sh
git checkout {self.pr.base.sha}
bash {WORKDIR}/check_git_changes.sh
{NacosConfig.MAVEN_INSTALL} || true
"""

    def _run_script(self) -> str:
        """基础测试脚本"""
        return f"""#!/bin/bash
set -e
cd {WORKDIR}/{self.pr.repo}
{NacosConfig.MAVEN_FAST_TEST}
"""

    def _test_run_script(self) -> str:
        """测试补丁脚本"""
        return f"""#!/bin/bash
set -e
cd {WORKDIR}/{self.pr.repo}
git apply {WORKDIR}/test.patch
{NacosConfig.MAVEN_FAST_TEST}
"""

    def _fix_run_script(self) -> str:
        """修复补丁脚本"""
        return f"""#!/bin/bash
set -e
cd {WORKDIR}/{self.pr.repo}
git apply {WORKDIR}/test.patch {WORKDIR}/fix.patch
{NacosConfig.MAVEN_FAST_TEST}
"""


# ========================================================================================
# 测试实例类
# ========================================================================================

@Instance.register("alibaba", "nacos")
class Nacos(Instance):
    """nacos 测试实例"""

    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Optional[Image]:
        return NacosImageDefault(self.pr, self._config)

    def run(self, run_cmd: str = "") -> str:
        return run_cmd or f"bash {WORKDIR}/run.sh"

    def test_patch_run(self, test_patch_run_cmd: str = "") -> str:
        return test_patch_run_cmd or f" bash {WORKDIR}/test-run.sh"

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        return fix_patch_run_cmd or f"bash {WORKDIR}/fix-run.sh"

    def parse_log(self, test_log: str) -> TestResult:
        """解析 Maven 测试日志"""
        passed_tests, failed_tests, skipped_tests = set(), set(), set()

        patterns = {
            'passed': [
                r"\[INFO\] Tests run: \d+, Failures: 0, Errors: 0, Skipped: \d+.+- in (.+)$",
                r"^(.+) \.\.\. PASSED$",
            ],
            'failed': [
                r"\[ERROR\] Tests run: \d+, Failures: [1-9]\d*.+- in (.+)$",
                r"\[ERROR\] Tests run: \d+, Failures: \d+, Errors: [1-9]\d*.+- in (.+)$",
                r"^(.+) \.\.\. FAILED$",
            ],
            'skipped': [
                r"^(.+) \.\.\. SKIPPED$",
            ]
        }

        for line in test_log.splitlines():
            for status, regex_list in patterns.items():
                for pattern in regex_list:
                    match = re.search(pattern, line)
                    if match:
                        test_name = match.group(1)
                        if status == 'passed' and test_name not in failed_tests:
                            passed_tests.add(test_name)
                        elif status == 'failed':
                            failed_tests.add(test_name)
                            passed_tests.discard(test_name)
                        elif status == 'skipped':
                            skipped_tests.add(test_name)
                        break

        return TestResult(
            passed_count=len(passed_tests),
            failed_count=len(failed_tests),
            skipped_count=len(skipped_tests),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
        )