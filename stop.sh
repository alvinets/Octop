#!/usr/bin/env bash
# Octop stopper — graceful SIGTERM, escalating to SIGKILL only if needed.
#
#   ./stop.sh
#   ./stop.sh --timeout 60
#
# Env overrides: OCTOP_HOME, OCTOP_STOP_TIMEOUT

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${OCTOP_HOME:-$HOME/.octop}"
RUN_DIR="$DATA_DIR/run"
PID_FILE="$RUN_DIR/octop.pid"
PORT="${OCTOP_PORT:-8088}"
TIMEOUT="${OCTOP_STOP_TIMEOUT:-30}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --timeout) TIMEOUT="${2:?--timeout needs a value}"; shift 2 ;;
        -h|--help) sed -n '2,8p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ -t 1 ]]; then
    R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[34m'; N=$'\033[0m'
else
    R=""; G=""; Y=""; B=""; N=""
fi

info() { echo "${B}[octop]${N} $*"; }
ok()   { echo "${G}[  ok  ]${N} $*"; }
warn() { echo "${Y}[ warn ]${N} $*"; }

find_running_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null)"
        if [[ -n "$pid" ]] && pid_alive "$pid"; then
            echo "$pid"
            return 0
        fi
    fi
    # Match this port too: an instance on another port (or another user's data
    # dir) must not be mistaken for ours.
    pgrep -f "[.]venv/bin/octop run .*--port[ =]${PORT}([^0-9]|$)" 2>/dev/null | head -1
}

# `kill -0` reports EPERM (non-zero) for a live process owned by another user,
# which is indistinguishable from ESRCH by exit code alone. /proc is
# authoritative for existence, so liveness must not rely on kill -0.
pid_alive() {
    [[ -d "/proc/$1" ]] && return 0
    kill -0 "$1" 2>/dev/null
}

# True when we are permitted to signal the pid (i.e. it is ours).
can_signal() {
    kill -0 "$1" 2>/dev/null
}

pid="$(find_running_pid)"
if [[ -z "$pid" ]]; then
    ok "not running"
    rm -f "$PID_FILE"
    # Report a lingering listener rather than claiming success if the port is
    # still held — that is the state an operator actually needs to know about.
    if command -v ss >/dev/null 2>&1 && ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
        warn "port $PORT is still bound by another process"
    fi
    exit 0
fi

if ! can_signal "$pid"; then
    warn "pid $pid is alive but owned by another user — cannot signal it."
    warn "run: sudo kill $pid   (nothing was stopped)"
    exit 1
fi

info "stopping pid $pid (SIGTERM, waiting up to ${TIMEOUT}s)…"
kill -TERM "$pid" 2>/dev/null || true

deadline=$((SECONDS + TIMEOUT))
while ((SECONDS < deadline)); do
    pid_alive "$pid" || break
    sleep 1
done

if pid_alive "$pid"; then
    warn "still alive after ${TIMEOUT}s — sending SIGKILL"
    kill -KILL "$pid" 2>/dev/null || true
    sleep 2
fi

if pid_alive "$pid"; then
    warn "pid $pid survives SIGKILL; inspect it manually"
    exit 1
fi

ok "stopped (pid $pid)"
rm -f "$PID_FILE"

if command -v ss >/dev/null 2>&1; then
    if ss -tln 2>/dev/null | grep -qE "[:.]${PORT}[[:space:]]"; then
        warn "port $PORT is still bound — something else is holding it"
    else
        ok "port $PORT released"
    fi
fi