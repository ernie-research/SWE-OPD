#!/usr/bin/env bash
###############################################################################
# Cluster-wide GPU monitor from the head node, WITHOUT interleaved output.
#
# Uses `nvitop --once` (richer than nvidia-smi: colored per-GPU table +
# per-process GPU-MEM/%SM/%GMBW/CPU/TIME/COMMAND in one view). Run on each node,
# aggregated back to the head.
#
# WHY a separate script (and why NOT dispatch.sh `status`):
#   dispatch.sh's old status fanned every host out with `&` writing straight to
#   the shared stdout, so multi-line output interleaved line-by-line and was
#   unreadable. This script serializes per-node output so it never interleaves.
#
# TWO display modes:
#   loop (DEFAULT) - a rotating live monitor. The screen is CLEARED, ONE node's
#                    nvitop report is shown, it STAYS for --stay seconds, then
#                    the next node replaces it. Cycles forever until you Ctrl-C.
#                    Only one node is ever on screen, so it can never interleave.
#                    `bash gpu_status.sh` runs this.
#   --once         - probe EVERY node in parallel (fast), then print all node
#                    blocks back-to-back in host order. Non-interleaved because
#                    each probe is captured to its own temp file first.
#
# Node discovery (same order as dispatch.sh):
#   PADDLE_TRAINERS (comma-separated) -> HOSTFILE (first field per line) ->
#   localhost. HOSTFILE defaults to /path/to/workspace/hostfile.
#
# Usage:
#   bash gpu_status.sh                 # rotating live monitor (default)
#   bash gpu_status.sh --stay 2        # stay 2s on each node
#   bash gpu_status.sh --once          # one-shot: all nodes at once, no loop
#   STAY_SEC=3 bash gpu_status.sh      # env alternative to --stay
#   NVITOP_ARGS='--colorful' bash gpu_status.sh   # extra flags forwarded to nvitop
#
# nvitop is always run as `nvitop --once --force-color ${NVITOP_ARGS}`.
# --force-color keeps color alive when nvitop's stdout is a pipe/ssh (not a TTY);
# NVITOP_ARGS (default empty) forwards extra nvitop flags, e.g. --ascii/--colorful.
###############################################################################
set -uo pipefail

DEFAULT_HOSTFILE="${DEFAULT_HOSTFILE:-/path/to/workspace/hostfile}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=8)
FANOUT_TIMEOUT="${FANOUT_TIMEOUT:-20}"
STAY_SEC="${STAY_SEC:-1}"
NVITOP_ARGS="${NVITOP_ARGS:-}"

DISPLAY_MODE="loop"   # loop | once

while [[ $# -gt 0 ]]; do
  case "$1" in
    --once)     DISPLAY_MODE="once" ;;
    --stay)     STAY_SEC="${2:?--stay needs a value}"; shift ;;
    --stay=*)   STAY_SEC="${1#*=}" ;;
    -h|--help)
      sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
  shift
done

load_hosts() {
  HOSTS=()
  local hostfile="${HOSTFILE:-${DEFAULT_HOSTFILE}}"
  if [[ -n "${PADDLE_TRAINERS:-}" ]]; then
    IFS=',' read -ra HOSTS <<<"${PADDLE_TRAINERS}"
  elif [[ -f "${hostfile}" ]]; then
    while IFS= read -r line; do
      line="${line%%#*}"
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
}

# The command run ON each node. Kept as a single string so it ships over ssh.
# `nvitop --once` gives the colored per-GPU table + per-process view in one shot;
# --force-color keeps ANSI alive when stdout is a pipe/ssh (not a TTY).
build_remote_cmd() {
  local body='set +e; '
  body+='if command -v nvitop >/dev/null 2>&1; then '
  body+="nvitop --once --force-color ${NVITOP_ARGS}; "
  body+='else echo "  nvitop not found on this node (pip install nvitop)"; fi'
  printf '%s' "${body}"
}

is_local() {
  local host="$1"
  [[ "${host}" == "localhost" || "${host}" == "127.0.0.1" \
     || "${host}" == "$(hostname -s 2>/dev/null || true)" \
     || "${host}" == "$(hostname -i 2>/dev/null || true)" ]]
}

# Probe ONE host and echo its captured output (blocking). No interleave risk:
# callers invoke this for a single host at a time (loop) or into a temp file (once).
probe_host() {
  local host="$1" cmd="$2"
  if is_local "${host}"; then
    bash -c "${cmd}"
  else
    timeout "${FANOUT_TIMEOUT}" ssh "${SSH_OPTS[@]}" "${host}" \
      "bash -c $(printf '%q' "${cmd}")" </dev/null \
      || echo "  (unreachable or timed out after ${FANOUT_TIMEOUT}s)"
  fi
}

role_of() { [[ "$1" -eq 0 ]] && echo head || echo worker; }

# --once: probe every node in parallel into per-host temp files, then print the
# whole blocks in order. Sequential printing => never interleaved.
run_once() {
  local remote_cmd; remote_cmd="$(build_remote_cmd)"
  tmpdir="$(mktemp -d)"; trap 'rm -rf "${tmpdir:-}"' EXIT
  local i pids=()
  for i in "${!HOSTS[@]}"; do
    probe_host "${HOSTS[$i]}" "${remote_cmd}" >"${tmpdir}/${i}.out" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "${pid}"; done
  for i in "${!HOSTS[@]}"; do
    echo "======================= ${HOSTS[$i]} ($(role_of "$i"), rank ${i}) ======================="
    cat "${tmpdir}/${i}.out"; echo
  done
}

# loop (default): rotating live monitor. Clear screen, show ONE node, stay
# STAY_SEC, advance. Cycle forever until Ctrl-C. Only one node on screen at a
# time, so the terminal stays short and output can never interleave.
run_loop() {
  local remote_cmd; remote_cmd="$(build_remote_cmd)"
  # Robust screen clear. Prefer terminfo (tput clear also emits E3 = wipe
  # scrollback on modern ncurses); fall back to raw ANSI (3J scrollback + 2J
  # screen + H home) so stale frames never stack even on bare terminals.
  local CLR
  CLR="$(tput clear 2>/dev/null)" || CLR=""
  [[ -z "${CLR}" ]] && CLR=$'\033[3J\033[H\033[2J'
  trap 'printf "\033[?25h"; echo; echo "gpu_status: stopped."; exit 0' INT TERM
  printf '\033[?25l'  # hide cursor to reduce flicker
  local cycle=0 i n="${#HOSTS[@]}"
  while true; do
    cycle=$((cycle + 1))
    for i in "${!HOSTS[@]}"; do
      local out; out="$(probe_host "${HOSTS[$i]}" "${remote_cmd}")"
      printf '%s' "${CLR}"  # clear screen + scrollback before drawing the frame
      echo "===== gpu_status live | cycle ${cycle} | node $((i + 1))/${n} | stay ${STAY_SEC}s | Ctrl-C to quit ====="
      echo "$(date '+%F %T')  ${HOSTS[$i]} ($(role_of "$i"), rank ${i})"
      echo
      printf '%s\n' "${out}"
      sleep "${STAY_SEC}"
    done
  done
}

main() {
  load_hosts
  case "${DISPLAY_MODE}" in
    once) run_once ;;
    loop) run_loop ;;
  esac
}

main
