import re
import json
from typing import Optional, Union

from multi_swe_bench.harness.image import Config, File, Image
from multi_swe_bench.harness.instance import Instance, TestResult
from multi_swe_bench.harness.pull_request import PullRequest


class ParcelImageBase(Image):
    """
    基础镜像，适配 parcel-bundler/parcel
    安装 Node.js、yarn 等构建工具
    """
    def __init__(self, pr: PullRequest, config: Config):
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    @property
    def config(self) -> Config:
        return self._config

    def dependency(self) -> Union[str, "Image"]:
        return "debian:latest"

    def image_name(self) -> str:
        return (
            f"{self.image_prefix()}/{self.pr.org}_m_{self.pr.repo}".lower()
            if self.image_prefix()
            else f"{self.pr.org}_m_{self.pr.repo}".lower()
        )

    def image_tag(self) -> str:
        return "base"

    def workdir(self) -> str:
        return "base"

    def files(self) -> list[File]:
        return []

    def dockerfile(self) -> str:
        image_name = self.dependency()
        if isinstance(image_name, Image):
            image_name = image_name.image_full_name()

        # 克隆或复制仓库代码
        if self.config.need_clone:
            code = (
                f"RUN git clone https://github.com/{self.pr.org}/{self.pr.repo}.git "
                f"/home/{self.pr.repo}"
            )
        else:
            code = f"COPY {self.pr.repo} /home/{self.pr.repo}"

        return f"""FROM {image_name}

{self.global_env}

WORKDIR /home/

# 国内源
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

# 安装基础依赖
RUN apt update && \
    apt install -y git curl build-essential

# 安装 nvm
ENV NVM_DIR=/root/.nvm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash

# 设置 nvm 环境变量并安装 Node.js 14
RUN . "$NVM_DIR/nvm.sh" && \
    nvm install 14 && \
    nvm alias default 14 && \
    nvm use default

# 确保 node 和 npm 可以全局使用
ENV NODE_PATH=$NVM_DIR/versions/node/v14.21.3/lib/node_modules
ENV PATH=$NVM_DIR/versions/node/v14.21.3/bin:$PATH

# 验证
RUN node -v && npm -v

# npm 国内源安装 (阿里)
RUN npm config set registry https://registry.npmmirror.com/

RUN npm install yarn -g --force

# yarn 国内源安装 (阿里)
RUN yarn config set npmRegistryServer https://registry.npmmirror.com/

{code}

{self.clear_env}

"""


class ParcelImageDefault(Image):
    """
    默认构建镜像，包含测试需要的脚本与补丁
    """
    def __init__(self, pr: PullRequest, config: Config):
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    @property
    def config(self) -> Config:
        return self._config

    def dependency(self) -> Optional[Image]:
        return ParcelImageBase(self.pr, self._config)

    def image_name(self) -> str:
        return (
            f"{self.image_prefix()}/{self.pr.org}_m_{self.pr.repo}".lower()
            if self.image_prefix()
            else f"{self.pr.org}_m_{self.pr.repo}".lower()
        )

    def image_tag(self) -> str:
        return f"pr-{self.pr.number}"

    def workdir(self) -> str:
        return f"pr-{self.pr.number}"

    def files(self) -> list[File]:
        print('lingsan is heear')
        return [
            File(".", "fix.patch", f"{self.pr.fix_patch}"),
            File(".", "test.patch", f"{self.pr.test_patch}"),
            File(
                ".",
                "check_git_changes.sh",
                """#!/bin/bash
set -e

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  echo "check_git_changes: Not inside a git repository"
  exit 1
fi

if [[ -n $(git status --porcelain) ]]; then
  echo "check_git_changes: Uncommitted changes"
  exit 1
fi

echo "check_git_changes: No uncommitted changes"
exit 0
""",
            ),
            File(
                ".",
                "prepare.sh",
                """#!/bin/bash
cd /home/{pr.repo}
git reset --hard
bash /home/check_git_changes.sh
git checkout {pr.base.sha}
bash /home/check_git_changes.sh

yarn install --frozen-lockfile
# yarn prettier:check
# yarn test -- --json
""".format(pr=self.pr),
            ),
            File(
                ".",
                "run.sh",
                """#!/bin/bash
set -e
cd /home/{pr.repo}

yarn test -- --json
""".format(pr=self.pr),
            ),
            File(
                ".",
                "test-run.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
git apply --whitespace=nowarn /home/test.patch

# yarn prettier:check
yarn test -- --json

""".format(pr=self.pr),
            ),
            File(
                ".",
                "fix-run.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
git apply --whitespace=nowarn /home/test.patch /home/fix.patch

# yarn prettier:check
yarn test -- --json
""".format(pr=self.pr),
            ),
        ]

    def dockerfile(self) -> str:
        image = self.dependency()
        name = image.image_name()
        tag = image.image_tag()

        copy_commands = "".join(f"COPY {file.name} /home/\n" for file in self.files())
        prepare_commands = "RUN bash /home/prepare.sh"

        return f"""FROM {name}:{tag}

{self.global_env}

{copy_commands}

{prepare_commands}

{self.clear_env}

"""


@Instance.register("pugjs", "pug")
class Pug(Instance):
    """
    Parcel 仓库的测试适配器
    """
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config

    @property
    def pr(self) -> PullRequest:
        return self._pr

    def dependency(self) -> Optional[Image]:
        return ParcelImageDefault(self.pr, self._config)

    def run(self, run_cmd: str = "") -> str:
        return run_cmd if run_cmd else "bash /home/run.sh"

    def test_patch_run(self, test_patch_run_cmd: str = "") -> str:
        return test_patch_run_cmd if test_patch_run_cmd else "bash /home/test-run.sh"

    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        return fix_patch_run_cmd if fix_patch_run_cmd else "bash /home/fix-run.sh"

    def parse_log(self, test_log: str) -> TestResult:
        # 找到 Ran all test suites. 之后的内容
        match = re.search(r"Ran all test suites\.\n(.+)", test_log, re.DOTALL)
        if not match:
            raise ValueError("日志中未找到 Ran all test suites. 后的 JSON")

        content = match.group(1).strip()

        # 从第一个 { 开始
        json_start = content.find("{")
        if json_start == -1:
            raise ValueError("日志中未找到 JSON 开始标志 {")
        
        json_sub = content[json_start:]

        # 找到最后一个 } 的位置，去掉末尾冗余字符串
        json_end = json_sub.rfind("}")
        if json_end == -1:
            raise ValueError("日志中未找到 JSON 结束标志 }")
        
        json_str = json_sub[:json_end+1]

        # 修复跨行导致的换行符
        json_str = json_str.replace("\n", "").replace("\r", "")

        # 解析 JSON
        data = json.loads(json_str)

        passed_tests = set()
        failed_tests = set()
        skipped_tests = set()
        
        # 遍历 testResults 提取 fullName 和 status
        for suite in data.get("testResults", []):
            for test in suite.get("assertionResults", []):
                status = test.get("status")
                full_name = test.get("fullName")
                if status == "passed" and full_name not in failed_tests and full_name not in skipped_tests:
                    passed_tests.add(full_name)
                elif status == "failed" and full_name not in passed_tests and full_name not in skipped_tests:
                    failed_tests.add(full_name)
                elif status == "pending" and full_name not in passed_tests and full_name not in failed_tests:
                    skipped_tests.add(full_name)

        return TestResult(
            passed_count=len(passed_tests),
            failed_count=len(failed_tests),
            skipped_count=len(skipped_tests),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests
        )