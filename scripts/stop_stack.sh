#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${REPO_ROOT}/run"

pid_alive () {
  local pid="${1:-}"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

# Best-effort: find the PID listening on a TCP port (Linux `ss`).
pid_listening_on_port () {
  local port="$1"
  ss -ltnp 2>/dev/null \
    | awk -v p=":${port}" '
        $0 ~ p && $0 ~ /pid=[0-9]+/ {
          if (match($0, /pid=[0-9]+/)) {
            s = substr($0, RSTART, RLENGTH);
            sub(/^pid=/, "", s);
            print s;
            exit;
          }
        }'
}

kill_tree () {
  local pid="$1"
  if command -v pgrep >/dev/null 2>&1; then
    local kids
    kids="$(pgrep -P "${pid}" 2>/dev/null || true)"
    if [[ -n "${kids}" ]]; then
      # shellcheck disable=SC2086
      kill ${kids} 2>/dev/null || true
    fi
  fi
  kill "${pid}" 2>/dev/null || true
}

stop_pid () {
  local name="$1"
  local pidfile="${RUN_DIR}/${name}.pid"
  if [[ ! -f "${pidfile}" ]]; then
    echo "${name}: no pidfile"
    return 0
  fi
  local pid
  pid="$(cat "${pidfile}" 2>/dev/null || true)"
  if pid_alive "${pid}"; then
    echo "Stopping ${name} (pid ${pid})..."
    kill_tree "${pid}"
  else
    echo "${name}: pid ${pid} not running"
  fi
  rm -f "${pidfile}"
}

echo "Stopping stack..."

# 1) Stop simulation/arena loop
stop_pid "sim"
if command -v pkill >/dev/null 2>&1; then
  pkill -f "arena:simulate" 2>/dev/null || true
  pkill -f "python .*scripts/run_reverie_headless.py" 2>/dev/null || true
  pkill -f "python .*reverie/backend_server/scripts/run_sim_loop.py" 2>/dev/null || true
fi

# 2) Stop Social Pet API (:3001)
stop_pid "api"
API_PID="$(pid_listening_on_port 3001 || true)"
if pid_alive "${API_PID:-}"; then
  echo "Stopping API listener on :3001 (pid ${API_PID})..."
  kill_tree "${API_PID}"
fi
if command -v pkill >/dev/null 2>&1; then
  pkill -f "yarn dev:api" 2>/dev/null || true
  pkill -f "tsx watch src/app.ts" 2>/dev/null || true
fi

# 3) Stop Django (:8000)
stop_pid "django"
DJ_PID="$(pid_listening_on_port 8000 || true)"
if pid_alive "${DJ_PID:-}"; then
  echo "Stopping django listener on :8000 (pid ${DJ_PID})..."
  kill_tree "${DJ_PID}"
fi
if command -v pkill >/dev/null 2>&1; then
  pkill -f "python .*manage.py runserver .*:8000" 2>/dev/null || true
fi

# 4) Stop vLLM (:8001)
stop_pid "vllm"
VLLM_PID="$(pid_listening_on_port 8001 || true)"
if pid_alive "${VLLM_PID:-}"; then
  echo "Stopping vLLM listener on :8001 (pid ${VLLM_PID})..."
  kill_tree "${VLLM_PID}"
fi
if command -v pkill >/dev/null 2>&1; then
  pkill -f "run_vllm_qwen_student.sh" 2>/dev/null || true
  pkill -f "vllm" 2>/dev/null || true
fi

echo "Done."
