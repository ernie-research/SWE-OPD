#!/usr/bin/env bash
###############################################################################
# Multi-node helper for slime coding_agent_opd jobs.
#
# Functions implemented here:
#   1. exec   - run a command on worker nodes from the head node
#   2. sync   - rsync a local path from the head node to worker nodes
#   3. clean  - clean slime training leftovers on all nodes
#   4. status - rotating live GPU monitor across all nodes (delegates to
#               gpu_status.sh; non-interleaved, clears the screen per node)
#
# Node discovery order:
#   PADDLE_TRAINERS (comma-separated) -> HOSTFILE (one host per line, first
#   whitespace field per line; "IP slots=1" style is fine) -> localhost.
#   HOSTFILE defaults to /path/to/workspace/hostfile if present.
#
# Usage:
#   bash dispatch.sh exec "nvidia-smi -L"
#   bash dispatch.sh sync                         # defaults to /path/to/workspace/env_run
#   bash dispatch.sh sync /path/to/repo --exclude='.git'
#   bash dispatch.sh clean
#   bash dispatch.sh status                 # rotating GPU monitor (nvitop --once)
#   bash dispatch.sh status --once          # one-shot snapshot of every node
###############################################################################
set -euo pipefail

DEFAULT_SYNC_DIR="${DEFAULT_SYNC_DIR:-/path/to/workspace/env_run}"
DEFAULT_HOSTFILE="${DEFAULT_HOSTFILE:-/path/to/workspace/hostfile}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
RSYNC_OPTS=(-aHz --inplace --delete --mkpath --info=progress2,stats1)
# Repo-oriented excludes. These match names that are common REPO artifacts (VCS
# metadata, run outputs, huge tarballs). DO NOT apply these when syncing a venv:
# names like `wandb`/`outputs`/`logs` and the archive extensions also match real
# Python PACKAGES / package data inside site-packages (e.g. the `wandb` package,
# `litellm/llms/wandb`, `cleo/io/outputs`, `modelscope/outputs`, `.gz`/`.zip`
# data files), so applying them to a venv silently strips importable modules and
# produces ModuleNotFoundError on the workers. See is_venv_dir / cmd_sync below.
DEFAULT_EXCLUDES=(
  "--exclude=.git"
  "--exclude=__pycache__"
  "--exclude=*.pyc"
  "--exclude=.ipynb_checkpoints"
  "--exclude=.pytest_cache"
  "--exclude=.mypy_cache"
  "--exclude=.ruff_cache"
  "--exclude=wandb"
  "--exclude=outputs"
  "--exclude=logs"
  "--exclude=*.log"
  # rollout trajectory dumps under the default sync root
  # (/path/to/workspace/env_run/traj). Anchored with a leading
  # slash so it only matches the top-level dir of the transfer root, and an
  # explicit `sync .../traj` still works.
  "--exclude=/traj"
  # compressed archives / tarballs (can be huge, e.g. slime_bundle.tar ~18GB)
  "--exclude=*.tar"
  "--exclude=*.tar.gz"
  "--exclude=*.tgz"
  "--exclude=*.tar.bz2"
  "--exclude=*.tar.xz"
  "--exclude=*.tar.zst"
  "--exclude=*.gz"
  "--exclude=*.bz2"
  "--exclude=*.xz"
  "--exclude=*.zst"
  "--exclude=*.zip"
  "--exclude=*.7z"
  "--exclude=*.rar"
)

# Venv-safe excludes: ONLY byte-compiled caches, which are never import-critical
# (they are regenerated on demand) and never legitimately part of a package's
# distributed name space. Everything else in a venv is kept verbatim so no
# importable module or package data file is ever dropped.
VENV_EXCLUDES=(
  "--exclude=__pycache__"
  "--exclude=*.pyc"
)

usage() {
  cat <<'EOF'
Usage:
  bash dispatch.sh exec <command>
  bash dispatch.sh sync [path] [rsync args...]
  bash dispatch.sh launch <script> [script args...]
  bash dispatch.sh clean
  bash dispatch.sh status [gpu_status.sh args...]

status:
  Rotating live GPU monitor across all nodes (delegates to gpu_status.sh in
  this dir, which uses `nvitop --once` per node). With NO extra args it defaults
  to the equivalent of `gpu_status.sh --stay 0.01` (fast rotation, one node's
  nvitop report at a time). Any args after `status` are forwarded and override
  the default, e.g. `status --once` (one-shot), `status --stay 2`.

launch:
  Start a multi-node run with ONE command: the SAME <script> is launched on
  every worker (backgrounded, fully detached) and then on the head (rank 0).
  Each node's rank/branch (head vs worker) is decided inside <script> from
  PADDLE_TRAINER_ID, exactly like a manual run. Typical loop:
    bash dispatch.sh clean
    bash dispatch.sh sync /path/to/workspace/env_run/slime
    bash dispatch.sh launch examples/coding_agent_opd/scripts/<run>.sh
    bash dispatch.sh clean

Environment:
  PADDLE_TRAINERS   Comma-separated host list from PDC (preferred).
  PADDLE_TRAINER_ID Current rank. dispatch commands must run on rank 0.
  HOSTFILE          Fallback file with one host per line (default:
                    /path/to/workspace/hostfile). First whitespace
                    field per line is used, so "IP slots=1" lines work.
  DEFAULT_SYNC_DIR  Default path for `sync` when none is passed
                    (default: /path/to/workspace/env_run).
  SLIME_VENV        Forwarded to every node by `launch` (pins the interpreter).
  NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME / TP_SOCKET_IFNAME
                    Cross-node NIC for `launch`. If unset, auto-detected from
                    the route to the first worker (this cluster: eth0).
  LAUNCH_FORWARD_ENV  Space-separated extra env var NAMES `launch` forwards
                    (values taken from the current env), e.g.
                    "SAVE_PATH LOAD_PATH NUM_ROLLOUT".
  LAUNCH_HEAD_BG    1 => run the head detached in the background too (default
                    0 => head runs in the foreground and streams to the log).
  LAUNCH_LOG_DIR    Where launch writes per-node logs (default: this dir).
  LAUNCH_SSH_TIMEOUT  Seconds to wait per worker ssh dispatch (default 25).
EOF
}

load_hosts() {
  HOSTS=()

  local hostfile="${HOSTFILE:-${DEFAULT_HOSTFILE}}"

  if [[ -n "${PADDLE_TRAINERS:-}" ]]; then
    IFS=',' read -ra HOSTS <<<"${PADDLE_TRAINERS}"
  elif [[ -f "${hostfile}" ]]; then
    while IFS= read -r line; do
      line="${line%%#*}"
      # keep only the first whitespace-separated field (handles "IP slots=1")
      local host="${line%%[[:space:]]*}"
      host="${host//[[:space:]]/}"
      [[ -n "${host}" ]] && HOSTS+=("${host}")
    done <"${hostfile}"
  else
    HOSTS=("localhost")
  fi

  if [[ "${#HOSTS[@]}" -eq 0 ]]; then
    echo "No hosts found. Set PADDLE_TRAINERS or provide a HOSTFILE." >&2
    exit 1
  fi

  HEAD_HOST="${HOSTS[0]}"
  WORKERS=("${HOSTS[@]:1}")
}

require_head() {
  local rank="${PADDLE_TRAINER_ID:-0}"
  if [[ "${rank}" != "0" ]]; then
    echo "dispatch.sh should run on rank 0/head. Current PADDLE_TRAINER_ID=${rank}" >&2
    exit 1
  fi
}

run_on_host() {
  local host="$1"
  shift
  local command="$*"

  if [[ "${host}" == "localhost" || "${host}" == "127.0.0.1" || "${host}" == "$(hostname -s 2>/dev/null || true)" || "${host}" == "$(hostname -i 2>/dev/null || true)" ]]; then
    bash -lc "${command}"
  else
    ssh "${SSH_OPTS[@]}" "${host}" "bash -lc $(printf '%q' "${command}")"
  fi
}

cmd_exec() {
  local command="$*"
  if [[ -z "${command}" ]]; then
    echo "exec requires a command." >&2
    usage
    exit 1
  fi

  load_hosts
  require_head

  local targets=("${WORKERS[@]}")
  if [[ "${#targets[@]}" -eq 0 ]]; then
    targets=("${HEAD_HOST}")
  fi

  echo "Running on ${#targets[@]} target(s): ${command}"
  local pids=()
  for host in "${targets[@]}"; do
    (
      echo "===== ${host} ====="
      run_on_host "${host}" "${command}"
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  return "${failed}"
}

# A directory is treated as a Python venv if it has the venv marker `pyvenv.cfg`
# at its root, or a `bin/python*` interpreter. Venv syncs use VENV_EXCLUDES so
# the repo-oriented DEFAULT_EXCLUDES never strip importable packages.
is_venv_dir() {
  local d="$1"
  [[ -f "${d}/pyvenv.cfg" ]] && return 0
  compgen -G "${d}/bin/python*" >/dev/null 2>&1 && return 0
  return 1
}

cmd_sync() {
  load_hosts
  require_head

  local src="${1:-${DEFAULT_SYNC_DIR}}"
  if [[ "${#}" -gt 0 ]]; then
    shift
  fi

  if [[ ! -e "${src}" ]]; then
    echo "Sync source does not exist: ${src}" >&2
    exit 1
  fi

  local src_abs
  src_abs="$(cd -- "$(dirname -- "${src}")" && pwd)/$(basename -- "${src}")"

  if [[ "${#WORKERS[@]}" -eq 0 ]]; then
    echo "No worker nodes found. Nothing to sync."
    return 0
  fi

  # Pick the exclude set based on whether the source is a venv. Venvs must NOT
  # use the repo-oriented excludes (they match real package names like `wandb`).
  local excludes=("${DEFAULT_EXCLUDES[@]}")
  if is_venv_dir "${src_abs}"; then
    excludes=("${VENV_EXCLUDES[@]}")
    echo "Detected venv at ${src_abs}: using venv-safe excludes (${VENV_EXCLUDES[*]})."
  fi

  echo "Syncing ${src_abs} to ${#WORKERS[@]} worker(s)."
  local pids=()
  for host in "${WORKERS[@]}"; do
    (
      echo "===== ${host} ====="
      rsync "${RSYNC_OPTS[@]}" "${excludes[@]}" "$@" -e "ssh ${SSH_OPTS[*]}" "${src_abs}/" "${host}:${src_abs}/"
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  return "${failed}"
}

cmd_clean() {
  load_hosts
  require_head

  local clean_script="${SCRIPT_DIR}/cleanup.sh"
  local targets=("${HOSTS[@]}")
  echo "Cleaning ${#targets[@]} node(s)."

  local pids=()
  for host in "${targets[@]}"; do
    (
      echo "===== ${host} ====="
      if [[ "${host}" == "${HEAD_HOST}" ]]; then
        bash "${clean_script}"
      else
        run_on_host "${host}" "bash ${clean_script}"
      fi
    ) &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || failed=1
  done
  return "${failed}"
}

cmd_status() {
  require_head

  # Delegate to the standalone GPU monitor (same dir). It does its own host
  # discovery (PADDLE_TRAINERS -> HOSTFILE -> localhost, identical order) and
  # guarantees non-interleaved output using `nvitop --once` per node. With NO
  # extra args, `status` defaults to the equivalent of `gpu_status.sh --stay
  # 0.01` (a fast rotating live monitor, one node's nvitop report at a time).
  # Any args passed after `status` are forwarded verbatim and OVERRIDE the
  # default, e.g.:
  #   bash dispatch.sh status                 # == gpu_status.sh --stay 0.01
  #   bash dispatch.sh status --once          # one-shot snapshot of every node
  #   bash dispatch.sh status --stay 2        # rotating monitor, 2s per node
  local status_script="${SCRIPT_DIR}/gpu_status.sh"
  if [[ ! -f "${status_script}" ]]; then
    echo "gpu_status.sh not found at ${status_script}" >&2
    exit 1
  fi
  if [[ "$#" -eq 0 ]]; then
    set -- --stay 0.01
  fi
  exec bash "${status_script}" "$@"
}

# launch: start the SAME <script> on every worker (detached) then on the head.
# Each node decides its own rank branch from PADDLE_TRAINER_ID, exactly like a
# manual multi-node run. This replicates the robust one-line launch (setsid +
# </dev/null full detach + login shell so each worker reads its own rank env).
cmd_launch() {
  local script="${1:-}"
  if [[ -z "${script}" ]]; then
    echo "launch requires a <script>." >&2
    usage
    exit 1
  fi
  shift

  load_hosts
  require_head

  # Resolve the script to an absolute path so remote shells (different CWD) find it.
  local script_abs
  script_abs="$(cd -- "$(dirname -- "${script}")" && pwd)/$(basename -- "${script}")"
  if [[ ! -f "${script_abs}" ]]; then
    echo "launch script not found: ${script_abs}" >&2
    exit 1
  fi

  # Preserve the launch CWD so workers run from the same directory as the head.
  local run_dir
  run_dir="$(pwd)"

  local log_dir="${LAUNCH_LOG_DIR:-${SCRIPT_DIR}}"
  mkdir -p "${log_dir}"

  # Cross-node NIC. If the caller did not pin it, auto-detect from the route to
  # the first worker (this cluster resolves to eth0, NOT bond0).
  local ifname="${NCCL_SOCKET_IFNAME:-}"
  if [[ -z "${ifname}" && "${#WORKERS[@]}" -gt 0 ]]; then
    ifname="$(ip route get "${WORKERS[0]}" 2>/dev/null | grep -oP 'dev \K[^ ]+' | head -1 || true)"
  fi

  # Build the env prefix forwarded to every node (values from the current env).
  # SLIME_VENV pins the interpreter; the three *_SOCKET_IFNAME pin the NIC.
  local env_prefix=""
  if [[ -n "${SLIME_VENV:-}" ]]; then
    env_prefix+="export SLIME_VENV=$(printf '%q' "${SLIME_VENV}"); "
  fi
  if [[ -n "${ifname}" ]]; then
    env_prefix+="export NCCL_SOCKET_IFNAME=$(printf '%q' "${ifname}") GLOO_SOCKET_IFNAME=$(printf '%q' "${ifname}") TP_SOCKET_IFNAME=$(printf '%q' "${ifname}"); "
  fi
  # Extra named env vars the caller asked to forward (values from current env).
  local extra
  for extra in ${LAUNCH_FORWARD_ENV:-}; do
    if [[ -n "${!extra:-}" ]]; then
      env_prefix+="export ${extra}=$(printf '%q' "${!extra}"); "
    fi
  done

  local args_q=""
  local a
  for a in "$@"; do
    args_q+=" $(printf '%q' "${a}")"
  done

  echo "launch: script=${script_abs}"
  echo "        run_dir=${run_dir}  workers=${#WORKERS[@]}  head=${HEAD_HOST}"
  [[ -n "${ifname}" ]] && echo "        NIC=${ifname}"
  [[ -n "${SLIME_VENV:-}" ]] && echo "        SLIME_VENV=${SLIME_VENV}"

  # --- workers: detached background, each logs to its own file ---
  local ssh_timeout="${LAUNCH_SSH_TIMEOUT:-25}"
  local host
  for host in "${WORKERS[@]}"; do
    local wlog="${log_dir}/launch.worker.${host}.log"
    # Keep \$!, \$(hostname -i), \${PADDLE_TRAINER_ID} literal for WORKER-side
    # expansion; setsid + </dev/null fully detaches so the ssh can return.
    local remote="cd $(printf '%q' "${run_dir}") && ${env_prefix}setsid bash $(printf '%q' "${script_abs}")${args_q} >> $(printf '%q' "${wlog}") 2>&1 </dev/null & echo \"launched pid \$! rank \${PADDLE_TRAINER_ID:-?} host \$(hostname -i 2>/dev/null)\""
    echo "===== ${host} (worker) -> ${wlog} ====="
    timeout "${ssh_timeout}" ssh "${SSH_OPTS[@]}" "${host}" "bash -lc $(printf '%q' "${remote}")" </dev/null || \
      echo "WARN: ssh dispatch to ${host} returned non-zero (it may still have launched)."
  done

  # --- head (rank 0): foreground by default so its log streams to the console ---
  local hlog="${log_dir}/launch.head.log"
  echo "===== ${HEAD_HOST} (head, rank 0) -> ${hlog} ====="
  if [[ "${LAUNCH_HEAD_BG:-0}" == "1" ]]; then
    ( cd "${run_dir}" && setsid bash "${script_abs}" "$@" >"${hlog}" 2>&1 </dev/null & echo "head launched pid $!" )
  else
    ( cd "${run_dir}" && bash "${script_abs}" "$@" 2>&1 | tee "${hlog}" )
  fi
}

main() {
  local cmd="${1:-}"
  if [[ -z "${cmd}" ]]; then
    usage
    exit 1
  fi
  shift || true

  case "${cmd}" in
    exec) cmd_exec "$@" ;;
    sync) cmd_sync "$@" ;;
    launch) cmd_launch "$@" ;;
    clean) cmd_clean ;;
    status) cmd_status "$@" ;;
    -h|--help|help) usage ;;
    *)
      echo "Unknown command: ${cmd}" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
