#!/bin/bash
# Stage 40: Megatron-LM (editable) + slime (editable) + the r2egym runtime deps.
#

set -ex
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
activate_venv

cd "$MEGATRON_DIR"
# Reused tree (GOLDEN_OPD_DIR/Megatron-LM) is a .git-less snapshot (see
# ../repos.lock) -- rev-parse is best-effort.
git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no git metadata at $MEGATRON_DIR (snapshot; see repos.lock)"
git rev-parse --short HEAD 2>/dev/null || true
uv pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
# --no-build-isolation so the helpers_cpp extension really builds (otherwise
# it is silently skipped and GPT dataset loading breaks).
uv pip install -e . --no-build-isolation

cd "$SLIME_DIR"
git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "no git metadata at $SLIME_DIR (snapshot; see repos.lock)"
git rev-parse --short HEAD 2>/dev/null || true
uv pip install -r requirements.txt
uv pip install -e . --no-deps

# r2egym runtime deps (single resolve so uv keeps them mutually consistent).
uv pip install \
   gym==0.26.2 fire diff-parser==1.1 simple-parsing together markdown \
   pexpect libtmux bashlex litellm apscheduler kubernetes \
   asttokens flask ipython pandas tqdm pydantic rich pygments \
   "docker==7.2.0" "dockerfile-parse==2.0.1" \
   "rle-open-sdk==0.2.1"

# Second wave of r2egym imports: examples/coding_agent_opd/r2egym/__init__.py
# eagerly imports .runtime -> r2egym.agenthub.runtime.docker, which pulls the
# docker SDK, bs4, ghapi, unidiff, modal and tenacity at module level even when
# SLIME_AGENT_SANDBOX_BACKEND=rle. Versions match the reference venv; ghapi
# 2.0.3 in particular needs fastspec==0.1.1 (0.2.x dropped
# fastspec.oapi._build_groups -> ImportError at `import ghapi`).
uv pip install \
   "beautifulsoup4==4.15.0" bs4 "soupsieve==2.9.1" \
   "ghapi==2.0.3" "unidiff==0.7.5" "modal==1.5.2" "tenacity==9.1.4"
uv pip install --no-deps "fastspec==0.1.1"

# slime's requirements.txt pulls nvidia-resiliency-ext from a git ref that
# resolves to 0.4.1, which has no `__version__` attribute. Megatron's
# dist_checkpointing/strategies/nvrx.py asserts NVRX_MIN_VERSION=="0.6.0" and
# reads nvrx.__version__ at `import megatron.core` time -> AttributeError
# (recorded against the build checkout of main 4bf7fca05; the 0c5d1c445 branch
# does not carry nvrx.py, the pin is kept to match the accepted env).
# Megatron's own pyproject.toml pins ==0.6.0, so its pin wins over the
# reference venv's 0.4.1 (that venv carries an older Megatron).
uv pip install --no-deps "nvidia-resiliency-ext==0.6.0"

# numpy<2 last: megatron/apex/flash-attn were all built against the 1.x ABI.
uv pip install "numpy<2"

# ...which means scipy has to come back down too: slime's resolve lands on scipy
# 1.18.x, and that build hard-requires numpy>=2.0 -- `import scipy.sparse` dies
# with AttributeError: module 'numpy' has no attribute 'long'. 1.15.3 is the
# reference venv's pin and the last line that supports numpy 1.26.
uv pip install --no-deps "scipy==1.15.3"

# Same story for pandas/pyarrow: today's resolves land on pandas 3.1.x and
# pyarrow >= 26, both of which now hard-require numpy>=2.0 at import time --
# incompatible with the numpy 1.26.4 ABI pin above. 3.0.5 / 25.0.1 are the last
# 3.x / 2x releases that still support numpy 1.26, and are exactly what the
# original verified build froze (freeze_final.txt).
uv pip install --no-deps "pandas==3.0.5" "pyarrow==25.0.1"

# Present in the reference venv, missing from every resolve above, and actually
# imported by the code: chardet by r2egym's str_replace_editor/file_editor,
# GitPython by the vendored swebench/multi_swe_bench helpers, httpx-sse by the
# SSE clients.
uv pip install "chardet==5.2.0" gitpython httpx-sse

python - <<'PY'
import numpy, torch
print("numpy:", numpy.__version__, "| torch:", torch.__version__)
PY
