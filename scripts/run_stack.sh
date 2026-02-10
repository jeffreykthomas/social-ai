#!/usr/bin/env bash
set -euo pipefail

# Run the full Social-AI stack:
# - vLLM student server (8001)
# - Django environment/monitor (8000)
# - simulation loop
#
# Logs go to ./logs/*.log, pids to ./run/*.pid

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_ROOT}/venv/bin"

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/run"

if [[ -f "${VENV_BIN}/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_BIN}/activate"
else
  echo "Missing venv at ${VENV_BIN}. Create/activate your venv first." >&2
  exit 1
fi

echo "Repo: ${REPO_ROOT}"
SIM_MODE="${SIM_MODE:-classic_hybrid}"

pid_alive () {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

# Best-effort: find the PID listening on a TCP port.
# Works on Linux with `ss`. Returns empty string if not found.
pid_listening_on_port () {
  local port="$1"
  # Example ss output fragment:
  # LISTEN ... 0.0.0.0:8000 ... users:(("python",pid=708275,fd=6))
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

refresh_pidfile_from_port () {
  local name="$1"
  local port="$2"
  local pidfile="${REPO_ROOT}/run/${name}.pid"
  local pid
  pid="$(pid_listening_on_port "${port}" || true)"
  if [[ -n "${pid}" ]]; then
    echo "${pid}" > "${pidfile}"
  fi
}

start_bg () {
  local name="$1"
  local pidfile="${REPO_ROOT}/run/${name}.pid"
  local logfile="${REPO_ROOT}/logs/${name}.log"
  shift

  if [[ -f "${pidfile}" ]]; then
    local existing_pid
    existing_pid="$(cat "${pidfile}" 2>/dev/null || true)"
    if pid_alive "${existing_pid}"; then
      echo "${name}: already running (pid ${existing_pid})"
      return 0
    fi
    # Stale pidfile: clear it so we can restart or re-discover.
    rm -f "${pidfile}"
  fi

  echo "Starting ${name}..."
  nohup "$@" >"${logfile}" 2>&1 &
  echo $! > "${pidfile}"
  echo "${name}: pid $(cat "${pidfile}") (log: ${logfile})"
  # If the process dies immediately, surface it early.
  sleep 0.5
  if ! kill -0 "$(cat "${pidfile}")" 2>/dev/null; then
    echo "${name}: exited immediately. Check log: ${logfile}" >&2
    return 1
  fi
}

wait_http () {
  local url="$1"
  local name="$2"
  echo "Waiting for ${name} at ${url} ..."
  for _ in $(seq 1 180); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${name}: ready"
      return 0
    fi
    sleep 2
  done
  echo "${name}: NOT ready (timed out). Check logs." >&2
  return 1
}

# 1) vLLM student (optional but recommended if action is routed to student)
# If something is already bound to 8001, treat vLLM as running and refresh pidfile.
if [[ -n "$(pid_listening_on_port 8001 || true)" ]]; then
  echo "vllm: already listening on :8001"
  refresh_pidfile_from_port "vllm" 8001
else
  start_bg "vllm" bash "${REPO_ROOT}/reverie/backend_server/scripts/run_vllm_qwen_student.sh"
fi
wait_http "http://127.0.0.1:8001/health" "vLLM"
refresh_pidfile_from_port "vllm" 8001

# 2) Django environment server
# If something is already bound to 8000, treat Django as running and refresh pidfile.
if [[ -n "$(pid_listening_on_port 8000 || true)" ]]; then
  echo "django: already listening on :8000"
  refresh_pidfile_from_port "django" 8000
else
  start_bg "django" bash -lc "cd '${REPO_ROOT}/environment/frontend_server' && python manage.py runserver 0.0.0.0:8000"
fi
wait_http "http://127.0.0.1:8000/" "Django"
refresh_pidfile_from_port "django" 8000

# 3) Simulation loop
if [[ "${SIM_MODE}" == "predictive" ]]; then
  # If already running, refresh pidfile via pgrep (pidfile may be stale).
  if [[ -z "$(pid_listening_on_port 8000 || true)" ]]; then
    # no-op: sim doesn't bind a port, so just fall through
    true
  fi
  if [[ -z "${SIM_PID:-}" ]]; then
    SIM_PID="$( (pgrep -af "python .*run_sim_loop.py" 2>/dev/null || true) | awk 'NR==1{print $1}')"
  fi
  if [[ -n "${SIM_PID:-}" ]] && pid_alive "${SIM_PID}"; then
    echo "${SIM_PID}" > "${REPO_ROOT}/run/sim.pid"
    echo "sim: already running (pid ${SIM_PID})"
  else
    start_bg "sim" python "${REPO_ROOT}/reverie/backend_server/scripts/run_sim_loop.py"
  fi
else
  # Classic Reverie backend (tile-world). Enable hybrid agent mode by default.
  SIM_PID="$( (pgrep -af "python scripts/run_reverie_headless.py" 2>/dev/null || true) | awk 'NR==1{print $1}')"
  if [[ -n "${SIM_PID:-}" ]] && pid_alive "${SIM_PID}"; then
    echo "${SIM_PID}" > "${REPO_ROOT}/run/sim.pid"
    echo "sim: already running (pid ${SIM_PID})"
  else
    start_bg "sim" bash -lc "cd '${REPO_ROOT}/reverie/backend_server' && REVERIE_AGENT_MODE='${REVERIE_AGENT_MODE:-hybrid}' python scripts/run_reverie_headless.py"
  fi
fi

echo ""
echo "Stack started."
echo "- Sim UI:     http://localhost:8000/simulator_home"
echo "- Monitor UI: http://localhost:8000/agent_monitor/ (needs/monologue/predictions per agent)"
echo "- vLLM:       http://localhost:8001/v1/models"
echo "- Logs:       ${REPO_ROOT}/logs/"
echo ""


