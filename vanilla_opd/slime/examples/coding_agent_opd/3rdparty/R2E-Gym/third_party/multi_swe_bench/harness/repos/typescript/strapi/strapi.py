# -- 依赖引入 start --
import re # 如果log只能输出文本，可以用re库来正则匹配日志数据
from typing import Optional, Union
from multi_swe_bench.harness.image import Config, File, Image
from multi_swe_bench.harness.instance import Instance, TestResult
from multi_swe_bench.harness.pull_request import PullRequest
# -- 依赖引入 end --

# -- 基础镜像类 start --
class strapiImageBase(Image):
    # 初始化，可以不用修改
    def __init__(self, pr: PullRequest, config: Config):
        self._pr = pr
        self._config = config
        # ...

    # 属性值PR，可以不用修改
    @property
    def pr(self) -> PullRequest:
        return self._pr
        # ...

    # 属性值Config，可以不用修改
    @property
    def config(self) -> Config:
        return self._config
        # ...

    # 依赖镜像，按照实际情况申明
    def dependency(self) -> Union[str, "Image"]:
        return "debian:latest"
    
    def image_name(self) -> str:
        return (
            f"{self.image_prefix()}/{self.pr.org}_m_{self.pr.repo}".lower()
            if self.image_prefix()
            else f"{self.pr.org}_m_{self.pr.repo}".lower()
        )
    # 镜像标签，可以不用修改
    def image_tag(self) -> str:
        return "base"
        # ...

    # 镜像目录，可以不用修改
    def workdir(self) -> str:
        return "base"
        # ...

    # 镜像关联文件，可以不用修改
    def files(self) -> list[File]:
        return []

    # 镜像Dockerfile，按照实际构建逻辑编写
    def dockerfile(self) -> str:
        image_name = self.dependency()
        if isinstance(image_name, Image):
            image_name = image_name.image_full_name()

        if self.config.need_clone:
            code = f"RUN git clone https://github.com/{self.pr.org}/{self.pr.repo}.git /home/{self.pr.repo}"
        else:
            code = f"COPY {self.pr.repo} /home/{self.pr.repo}"

        return f"""FROM {image_name}
# 全局参数
{self.global_env}

# 指定工作目录、环境参数
WORKDIR /home/

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 国内源
RUN sed -i 's/deb.debian.org/mirrors.tuna.tsinghua.edu.cn/g' /etc/apt/sources.list.d/debian.sources

# 安装必要环境依赖
RUN apt update && apt install -y git curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt install -y nodejs

# npm 国内源安装 (阿里)
RUN npm config set registry https://registry.npmmirror.com/

RUN npm install -g yarn

# yarn 国内源安装 (阿里)
RUN yarn config set npmRegistryServer https://registry.npmmirror.com/

# 更新PATH，避免指令无法执行
ENV PATH="/root/.bun/bin:$PATH"

# 拉取代码库（或拷贝代码库）
{code}

# 清理不必要的参数
{self.clear_env}
"""
# -- 基础镜像类 end --

# -- PR镜像类 start --
class strapiImageDefault(Image):
    # 初始化，可以不用修改
    def __init__(self, pr: PullRequest, config: Config):
        self._pr = pr
        self._config = config
        # ...

    # 属性值PR，可以不用修改
    @property
    def pr(self) -> PullRequest:
        return self._pr
        # ...

    # 属性值Config，可以不用修改
    @property
    def config(self) -> Config:
        return self._config
        # ...

    # 镜像依赖，可以不用修改
    def dependency(self) -> Image | None:
        return strapiImageBase(self.pr, self.config)
    
    def image_name(self) -> str:
        return (
            f"{self.image_prefix()}/{self.pr.org}_m_{self.pr.repo}".lower()
            if self.image_prefix()
            else f"{self.pr.org}_m_{self.pr.repo}".lower()
        )
    # 镜像标签，可以不用修改
    def image_tag(self) -> str:
        return f"pr-{self.pr.number}"
        # ...

    # 镜像目录，可以不用修改
    def workdir(self) -> str:
        return f"pr-{self.pr.number}"
        # ...

    # 镜像关联文件，按照实际情况编写
    def files(self) -> list[File]:
        return [
            File(
                ".",
                "fix.patch",
                f"{self.pr.fix_patch}",
            ),
            File(
                ".",
                "test.patch",
                f"{self.pr.test_patch}",
            ),
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
""".format(
                    pr=self.pr
                ),
            ),
            # prepare.sh，在每个脚本执行前，会先跑一下的脚本，主要用于切换目录、切换代码分支、安装依赖等
            File(
                ".",
                "prepare.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
git reset --hard
bash /home/check_git_changes.sh
git checkout {pr.base.sha}
bash /home/check_git_changes.sh
yarn install --ignore-workspace-root-check || true
""".format(
                    pr=self.pr
                ),
            ),
            # run.sh，不做任何事情的情况下运行实例，确保正常输出测试的结果日志
            File(
                ".",
                "run.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
yarn test:unit --json || true
""".format(
                    pr=self.pr
                ),
            ),
            # test.sh，打上test.patch后运行实例，确保正常输出测试的结果日志（产出F2P的结果）
            File(
                ".",
                "test-run.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
git apply --whitespace=nowarn /home/test.patch

yarn test:unit --json || true
""".format(
                    pr=self.pr
                ),
            ),
            # fix-run.sh，打上test.patch和fix.patch后运行实例，确保正常输出测试的结果日志（产出P2P的结果）
            File(
                ".",
                "fix-run.sh",
                """#!/bin/bash
set -e

cd /home/{pr.repo}
git apply --whitespace=nowarn /home/test.patch /home/fix.patch

yarn test:unit --json || true
""".format(
                    pr=self.pr
                ),
            ),
        ]

    # 镜像Dockerfile，按照实际构建逻辑编写，前端方向基本可以参考下面内容写
    def dockerfile(self) -> str:
        image = self.dependency()
        name = image.image_name()
        tag = image.image_tag()

        copy_commands = ""
        for file in self.files():
            copy_commands += f"COPY {file.name} /home/\n"

        prepare_commands = "RUN bash /home/prepare.sh"

        return f"""FROM {name}:{tag}

{self.global_env}

{copy_commands}

{prepare_commands}

{self.clear_env}

"""
# -- PR镜像类 end --

# -- Main：REPO库解析逻辑 start --
@Instance.register("strapi", "strapi")
class Strapi(Instance):
    # 初始化，可以不用修改
    def __init__(self, pr: PullRequest, config: Config, *args, **kwargs):
        super().__init__()
        self._pr = pr
        self._config = config
        # ...

    # PR属性，可以不用修改
    @property
    def pr(self) -> PullRequest:
        return self._pr
        # ...

    # 依赖，可以不用修改
    def dependency(self) -> Optional[Image]:
        return strapiImageDefault(self.pr, self._config)
        # ...

    # 跑run.sh，可以不用修改
    def run(self, run_cmd: str = "") -> str:
        if run_cmd:
            return run_cmd

        return "bash /home/run.sh"
        # ...

    # 跑test-run.sh，可以不用修改
    def test_patch_run(self, test_patch_run_cmd: str = "") -> str:
        if test_patch_run_cmd:
            return test_patch_run_cmd

        return "bash /home/test-run.sh"
        # ...

    # 跑fix-run.sh，可以不用修改
    def fix_patch_run(self, fix_patch_run_cmd: str = "") -> str:
        if fix_patch_run_cmd:
            return fix_patch_run_cmd

        return "bash /home/fix-run.sh"
        # ...

    # 解析日志，从日志里提取test集合里：成功的名称、失败的名称、跳过的名称
    def parse_log(self, test_log: str) -> TestResult:
        passed_tests = set()
        failed_tests = set()
        skipped_tests = set()
        
        re_pass_test = re.compile(r"^[\s✓✔]+\s*(.+?)(?:\s+\d+ms)?$")
        re_fail_test = re.compile(r"^[\s✕✗]+\s*(.+?)(?:\s+\d+ms)?$")
        re_skip_test = re.compile(r"^[\s↓]+\s*(.+?)(?:\s+\d+ms)?$")

        for index, line in enumerate(test_log.splitlines()):
            line = line.strip()
            if not line:
                continue

            pass_test_match = re_pass_test.match(line)
            if pass_test_match:
                test = pass_test_match.group(1)
                if test in passed_tests:
                    print(f"{test} 已存在 passed_tests中")
                    continue
                else:
                    passed_tests.add(f"{test}_{index}")
                    continue

            fail_test_match = re_fail_test.match(line)
            if fail_test_match:
                test = fail_test_match.group(1)
                if test in failed_tests:
                    print(f"{test} 已存在 failed_tests中")
                else:
                    failed_tests.add(f"{test}_{index}")
        # 返回测试结果
        return TestResult(
            passed_count=len(passed_tests),
            failed_count=len(failed_tests),
            skipped_count=len(skipped_tests),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            skipped_tests=skipped_tests,
        )
 # -- Main：REPO库解析逻辑 end --