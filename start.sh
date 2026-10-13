#!/usr/bin/env bash
# Octop launcher — starts the server detached and waits until it is healthy.
#
#   ./start.sh              # bind 0.0.0.0:8088 (reachable over Tailscale + LAN)
#   ./start.sh --host 127.0.0.1
#   OCTOP_PORT=9000 ./start.sh
#
# Env overrides: OCTOP_BIND_HOST, OCTOP_PORT, OCTOP_HOME, OCTOP_START_TIMEOUT

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCTOP_BIN="$PROJECT_DIR/.venv/bin/octop"
# Resolve data directory: OCTOP_HOME > original user home (via SUDO_USER) > $HOME
    DATA_DIR="${OCTOP_HOME:-$(if [[ -n "${SUDO_USER:-}" ]]; then echo "/home/${SUDO_USER}/.octop"; else echo "$HOME/.octop"; fi)}"
RUN_DIR="$DATA_DIR/run"
LOG_FILE="$RUN_DIR/octop.log"
PID_FILE="$RUN_DIR/octop.pid"

HOST="${OCTOP_BIND_HOST:-0.0.0.0}"
PORT="${OCTOP_PORT:-8088}"
HEALTH_URL="http://127.0.0.1:$PORT/api/health"
MAX_WAIT="${OCTOP_START_TIMEOUT:-90}"

if [[ -t 1 ]]; then
    R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[34m'; N=$'\033[0m'
else
    R=""; G=""; Y=""; B=""; N=""
fi

info() { echo "${B}[octop]${N} $*"; }
ok()   { echo "${G}[  ok  ]${N} $*"; }
warn() { echo "${Y}[ warn ]${N} $*"; }
die()  { echo "${R}[ fail ]${N} $*" >&2; exit 1; }

# Resolve the live PID: prefer the pidfile, fall back to a cmdline scan so a
# server started outside these scripts can still be detected.
find_running_pid() {
    if [[ -f "$PID_FILE" ]]; then
        local pid
        pid="$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "$pid"
            return 0
        fi
    fi
    # The bracket keeps this pattern from matching the pgrep command itself.
    pgrep -f "[.]venv/bin/octop run" 2>/dev/null | head -1
}

# --- preflight -------------------------------------------------------------

[[ -x "$OCTOP_BIN" ]] || die "virtualenv missing: $OCTOP_BIN not found. Run: uv sync --python 3.12"

if ! "$OCTOP_BIN" --version >/dev/null 2>&1; then
    die "$OCTOP_BIN is not runnable (broken venv). Run: uv sync --python 3.12"
fi

[[ -f "$DATA_DIR/octop.db" ]] || die "not initialized: $DATA_DIR/octop.db absent. Run: $OCTOP_BIN init"

mkdir -p "$RUN_DIR" || die "cannot create $RUN_DIR"

existing="$(find_running_pid)"
if [[ -n "$existing" ]]; then
    info "already running (pid $existing)"
    ok "http://127.0.0.1:$PORT"
    exit 0
fi

# `uv sync` and friends replace site-packages and silently drop the local
# octop-harness patch, so the workspace-root fix would quietly disappear.
if [[ -x "$PROJECT_DIR/harness-patches/reapply.sh" ]]; then
    "$PROJECT_DIR/harness-patches/reapply.sh" --check >/dev/null 2>&1 || warn \
        "octop-harness patch is NOT applied (uv sync resets it) — run: make harness-patch"
fi

# A foreign process holding the port would make uvicorn die with an opaque
# address-in-use traceback; name the culprit instead.
if command -v ss >/dev/null 2>&1; then
    holder="$(ss -tlnp 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {print $NF}' | head -1)"
    if [[ -n "$holder" ]]; then
        die "port $PORT already in use by: $holder (stop it, or set OCTOP_PORT)"
    fi
fi

avail_gb="$(df -BG --output=avail "$DATA_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [[ -n "$avail_gb" && "$avail_gb" -lt 2 ]]; then
    warn "only ${avail_gb}G free on the volume holding $DATA_DIR — chat history and model caches will fail"
fi

# --- start -----------------------------------------------------------------

info "host=$HOST port=$PORT log=$LOG_FILE"

# setsid detaches into its own session so the server survives the caller's
# shell going away; plain nohup/& gets killed with the process group.
setsid nohup "$OCTOP_BIN" run --host "$HOST" --port "$PORT" \
    >"$LOG_FILE" 2>&1 </dev/null &
disown 2>/dev/null || true

# --- wait for health -------------------------------------------------------

info "waiting up to ${MAX_WAIT}s for the API to answer..."
deadline=$((SECONDS + MAX_WAIT))
while ((SECONDS < deadline)); do
    if [[ -n "$(find_running_pid)" ]] && curl -fsS -m 3 "$HEALTH_URL" >/dev/null 2>&1; then
        pid="$(find_running_pid)"
        echo "$pid" > "$PID_FILE"
        ok "healthy (pid $pid)"
        curl -fsS -m 3 "$HEALTH_URL" 2>/dev/null | sed 's/^/         /'
        echo

        if [[ "$HOST" == "0.0.0.0" || "$HOST" == "::" ]]; then
            ok "http://127.0.0.1:$PORT"
            if command -v tailscale >/dev/null 2>&1; then
                ts_ip="$(tailscale ip -4 2>/dev/null | head -1)"
                [[ -n "$ts_ip" ]] && ok "http://$ts_ip:$PORT   (Tailscale)"
            fi
            lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
            [[ -n "$lan_ip" ]] && ok "http://$lan_ip:$PORT   (LAN)"
        else
            ok "http://$HOST:$PORT  (loopback only — not reachable from other devices)"
        fi
        exit 0
    fi
    sleep 2
done

# --- failed ----------------------------------------------------------------

warn "did not become healthy within ${MAX_WAIT}s; last 25 log lines:"
tail -25 "$LOG_FILE" 2>/dev/null | sed 's/^/         /'
exit 1