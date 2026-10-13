#!/usr/bin/env bash
# Octop status — process, listener, API health, and reachable URLs.
# Exits 0 when healthy, 1 otherwise, so it works as a probe.
#
#   ./status.sh
#
# Env overrides: OCTOP_BIND_HOST, OCTOP_PORT, OCTOP_HOME

set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCTOP_BIN="$PROJECT_DIR/.venv/bin/octop"
DATA_DIR="${OCTOP_HOME:-$HOME/.octop}"
RUN_DIR="$DATA_DIR/run"
LOG_FILE="$RUN_DIR/octop.log"
PID_FILE="$RUN_DIR/octop.pid"

HOST="${OCTOP_BIND_HOST:-0.0.0.0}"
PORT="${OCTOP_PORT:-8088}"
HEALTH_URL="http://127.0.0.1:$PORT/api/health"

if [[ -t 1 ]]; then
    R=$'\033[31m'; G=$'\033[32m'; Y=$'\033[33m'; B=$'\033[34m'; N=$'\033[0m'
else
    R=""; G=""; Y=""; B=""; N=""
fi

ok()   { echo "${G}  ✔${N} $*"; }
bad()  { echo "${R}  x${N} $*"; }
warn() { echo "${Y}  !${N} $*"; }
row()  { printf "     %-22s %s\n" "$1" "$2"; }

echo "${B}Octop${N}  $PROJECT_DIR"
echo

# --- process ---------------------------------------------------------------

pid=""
if [[ -f "$PID_FILE" ]]; then
    cand="$(tr -dc '0-9' < "$PID_FILE" 2>/dev/null)"
    [[ -n "$cand" ]] && kill -0 "$cand" 2>/dev/null && pid="$cand"
fi
if [[ -z "$pid" ]]; then
    pid="$(pgrep -f "[.]venv/bin/octop run" 2>/dev/null | head -1)"
fi

if [[ -n "$pid" ]]; then
    started="$(ps -o etimes= -p "$pid" 2>/dev/null | tr -dc '0-9')"
    rss_kb="$(ps -o rss= -p "$pid" 2>/dev/null | tr -dc '0-9')"
    ok "process running (pid $pid, up $((started / 60))m$((started % 60))s, ${rss_kb}K RSS)"
    [[ -f "$PID_FILE" ]] || warn "no pidfile at $PID_FILE — started outside start.sh?"
else
    bad "process not running"
    [[ -f "$PID_FILE" ]] && warn "stale pidfile: $PID_FILE"
fi

# --- listener --------------------------------------------------------------

listen=""
if command -v ss >/dev/null 2>&1; then
    listen="$(ss -tln 2>/dev/null | awk -v p=":$PORT" '$4 ~ p"$" {print $4}' | paste -sd, -)"
fi
if [[ -n "$listen" ]]; then
    ok "listening on $listen"
else
    bad "nothing listening on port $PORT"
fi

# --- API -------------------------------------------------------------------

if body="$(curl -fsS -m 5 "$HEALTH_URL" 2>/dev/null)"; then
    ok "API healthy — $HEALTH_URL"
    row "response" "$body"
    if echo "$body" | grep -qE '"ok":false|"db":false'; then
        warn "health payload reports a degraded subsystem"
    fi
else
    bad "API not answering at $HEALTH_URL"
fi

# --- reachability ----------------------------------------------------------

echo
if [[ -z "$listen" ]]; then
    row "server down" "./start.sh   to bring it up"
elif [[ "$listen" == 0.0.0.0* || "$listen" == '['* ]]; then
    ok "bound to all interfaces — reachable from other devices"
    if command -v tailscale >/dev/null 2>&1; then
        ts_ip="$(tailscale ip -4 2>/dev/null | head -1)"
        if [[ -n "$ts_ip" ]]; then
            if curl -fsS -m 5 "http://$ts_ip:$PORT/api/health" >/dev/null 2>&1; then
                ok "Tailscale reachable — http://$ts_ip:$PORT/"
            else
                bad "Tailscale URL not answering — http://$ts_ip:$PORT/"
                row "likely cause" "host firewall, or tailscaled not running"
            fi
        fi
    fi
    lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    [[ -n "$lan_ip" ]] && row "LAN" "http://$lan_ip:$PORT/"
else
    bad "bound to $listen (loopback only) — other devices CANNOT reach it"
    row "to expose it" "./stop.sh && ./start.sh   (binds 0.0.0.0)"
fi

# --- storage ---------------------------------------------------------------

avail_gb="$(df -BG --output=avail "$DATA_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')"
if [[ -n "$avail_gb" ]]; then
    if [[ "$avail_gb" -lt 2 ]]; then
        warn "disk: only ${avail_gb}G free — chat history and model caches will fail"
    else
        ok "disk: ${avail_gb}G free on $DATA_DIR"
    fi
fi

[[ -n "$pid" && -n "$listen" ]] && exit 0
exit 1