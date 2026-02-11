#!/usr/bin/env bash
set -euo pipefail

# Run the Social-AI stack.
# Default mode (from config/social-pet-game.json):
# - Social Pet API (3001)
# - Social Pet arena simulation loop
# Optional:
# - vLLM student server (8001)
# - (optional) training watcher for online distillation
#
# Legacy mode (STACK_MODE=reverie) keeps the old Django + Reverie simulation path.
#
# Logs go to ./logs/*.log, pids to ./run/*.pid
#
# Environment:
#   ENABLE_TRAINING=1     Launch the training watcher (default: off)
#   TRAIN_THRESHOLD=300   New teacher entries before training triggers
#   TRAIN_POLL=300        Seconds between watcher polls

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_ROOT}/venv/bin"

mkdir -p "${REPO_ROOT}/logs" "${REPO_ROOT}/run"

echo "Repo: ${REPO_ROOT}"

CONFIG_DEFAULT_MODE="$(node -e 'const fs=require("fs");const p=process.argv[1];try{const c=JSON.parse(fs.readFileSync(p,"utf8"));process.stdout.write((c.arena&&c.arena.defaultMode)||"social_pet");}catch{process.stdout.write("social_pet");}' "${REPO_ROOT}/config/social-pet-game.json" 2>/dev/null || true)"
if [[ -z "${CONFIG_DEFAULT_MODE}" ]]; then
  CONFIG_DEFAULT_MODE="social_pet"
fi

if [[ -z "${STACK_MODE:-}" ]] && [[ -n "${SIM_MODE:-}" ]]; then
  STACK_MODE="reverie"
fi
STACK_MODE="${STACK_MODE:-${CONFIG_DEFAULT_MODE}}"
LEGACY_SIM_MODE="${SIM_MODE:-classic_hybrid}"
START_VLLM="${START_VLLM:-true}"

pid_alive () {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

# Best-effort: find the PID listening on a TCP port.
# Works on Linux with `ss`. Returns empty string if not found.
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

ensure_venv () {
  if [[ -f "${VENV_BIN}/activate" ]]; then
    # shellcheck disable=SC1091
    source "${VENV_BIN}/activate"
    return 0
  fi
  return 1
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
    rm -f "${pidfile}"
  fi

  echo "Starting ${name}..."
  nohup "$@" >"${logfile}" 2>&1 &
  echo $! > "${pidfile}"
  echo "${name}: pid $(cat "${pidfile}") (log: ${logfile})"
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

# Optional: vLLM student service
if [[ "${START_VLLM}" == "true" ]]; then
  if [[ -n "$(pid_listening_on_port 8001 || true)" ]]; then
    echo "vllm: already listening on :8001"
    refresh_pidfile_from_port "vllm" 8001
  else
    if ensure_venv; then
      start_bg "vllm" bash "${REPO_ROOT}/reverie/backend_server/scripts/run_vllm_qwen_student.sh"
    else
      echo "vllm: skipped (missing venv at ${VENV_BIN}; set START_VLLM=false to silence)"
    fi
  fi

  if [[ -n "$(pid_listening_on_port 8001 || true)" ]]; then
    wait_http "http://127.0.0.1:8001/health" "vLLM"
    refresh_pidfile_from_port "vllm" 8001
  fi
fi

if [[ "${STACK_MODE}" == "reverie" ]]; then
  if ! ensure_venv; then
    echo "Missing venv at ${VENV_BIN}. Create/activate your venv first for STACK_MODE=reverie." >&2
    exit 1
  fi

  # Django environment server
  if [[ -n "$(pid_listening_on_port 8000 || true)" ]]; then
    echo "django: already listening on :8000"
    refresh_pidfile_from_port "django" 8000
  else
    start_bg "django" bash -lc "cd '${REPO_ROOT}/environment/frontend_server' && python manage.py runserver 0.0.0.0:8000"
  fi
  wait_http "http://127.0.0.1:8000/" "Django"
  refresh_pidfile_from_port "django" 8000

  # Reverie simulation loop
  if [[ "${LEGACY_SIM_MODE}" == "predictive" ]]; then
    SIM_PID="$( (pgrep -af "python .*run_sim_loop.py" 2>/dev/null || true) | awk 'NR==1{print $1}')"
    if [[ -n "${SIM_PID:-}" ]] && pid_alive "${SIM_PID}"; then
      echo "${SIM_PID}" > "${REPO_ROOT}/run/sim.pid"
      echo "sim: already running (pid ${SIM_PID})"
    else
      start_bg "sim" python "${REPO_ROOT}/reverie/backend_server/scripts/run_sim_loop.py"
    fi
  else
    SIM_PID="$( (pgrep -af "python scripts/run_reverie_headless.py" 2>/dev/null || true) | awk 'NR==1{print $1}')"
    if [[ -n "${SIM_PID:-}" ]] && pid_alive "${SIM_PID}"; then
      echo "${SIM_PID}" > "${REPO_ROOT}/run/sim.pid"
      echo "sim: already running (pid ${SIM_PID})"
    else
      start_bg "sim" bash -lc "cd '${REPO_ROOT}/reverie/backend_server' && REVERIE_AGENT_MODE='${REVERIE_AGENT_MODE:-hybrid}' python scripts/run_reverie_headless.py"
    fi
  fi

  echo ""
  echo "Stack started (legacy reverie mode)."
  echo "- Sim UI:     http://localhost:8000/simulator_home"
  echo "- Monitor UI: http://localhost:8000/agent_monitor/"
  echo "- vLLM:       http://localhost:8001/v1/models"
  echo "- Logs:       ${REPO_ROOT}/logs/"
  echo ""
  exit 0
fi

# Social Pet mode (default)
if [[ -n "$(pid_listening_on_port 3001 || true)" ]]; then
  echo "api: already listening on :3001"
  refresh_pidfile_from_port "api" 3001
else
  start_bg "api" bash -lc "cd '${REPO_ROOT}' && yarn dev:api"
fi
wait_http "http://127.0.0.1:3001/healthz" "Social Pet API"
refresh_pidfile_from_port "api" 3001

SIM_PID="$( (pgrep -af "arena:simulate" 2>/dev/null || true) | awk 'NR==1{print $1}')"
if [[ -n "${SIM_PID:-}" ]] && pid_alive "${SIM_PID}"; then
  echo "${SIM_PID}" > "${REPO_ROOT}/run/sim.pid"
  echo "sim: already running (pid ${SIM_PID})"
else
  start_bg "sim" bash -lc "cd '${REPO_ROOT}' && while true; do yarn workspace @social-pet/api arena:simulate; sleep \${ARENA_LOOP_SLEEP_SECONDS:-20}; done"
fi

# 4) Optional: Training watcher (online distillation)
ENABLE_TRAINING="${ENABLE_TRAINING:-0}"
if [[ "${ENABLE_TRAINING}" == "1" ]]; then
  WATCHER_PID="$( (pgrep -af "training_watcher.sh" 2>/dev/null || true) | awk 'NR==1{print $1}')"
  if [[ -n "${WATCHER_PID:-}" ]] && pid_alive "${WATCHER_PID}"; then
    echo "${WATCHER_PID}" > "${REPO_ROOT}/run/trainer.pid"
    echo "trainer: already running (pid ${WATCHER_PID})"
  else
    start_bg "trainer" bash "${REPO_ROOT}/scripts/training_watcher.sh" \
      --threshold "${TRAIN_THRESHOLD:-300}" \
      --poll-interval "${TRAIN_POLL:-300}"
  fi
  echo "trainer: watcher active (log: ${REPO_ROOT}/logs/training.log)"
fi

echo ""
echo "Stack started (social_pet mode)."
echo "- API:        http://localhost:3001/healthz"
echo "- Arena loop: running in background (logs/sim.log)"
if [[ -n "$(pid_listening_on_port 8001 || true)" ]]; then
  echo "- vLLM:       http://localhost:8001/v1/models"
fi
echo "- Logs:       ${REPO_ROOT}/logs/"
if [[ "${ENABLE_TRAINING}" == "1" ]]; then
  echo "- Training:   watcher active (threshold=${TRAIN_THRESHOLD:-300}, poll=${TRAIN_POLL:-300}s)"
  echo "              log: ${REPO_ROOT}/logs/training.log"
fi
echo ""
