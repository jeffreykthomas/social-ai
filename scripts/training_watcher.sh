#!/usr/bin/env bash
set -euo pipefail

# Training Watcher — online distillation loop
#
# Monitors teacher.jsonl for new entries. When enough accumulate,
# pauses the sim, stops vLLM, runs LoRA fine-tuning, restarts
# vLLM with the new adapter, and resumes the sim.
#
# Usage:
#   bash scripts/training_watcher.sh [OPTIONS]
#
#   --threshold N      New teacher entries needed to trigger training (default: 300)
#   --poll-interval S  Seconds between checks (default: 300 = 5 min)
#   --max-rounds N     Stop after N training rounds (default: unlimited)
#   --dry-run          Log actions without executing them
#   --once             Run one training round immediately (ignore threshold) and exit
#
# Environment:
#   REPO_ROOT          Override repo root (default: auto-detected)
#   BASE_MODEL         Override base model (default: Qwen/Qwen2.5-32B-Instruct)
#   TRAIN_EPOCHS       Override epochs (default: 2)
#   LORA_RANK          Override LoRA rank (default: 32)

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_BIN="${REPO_ROOT}/venv/bin"

DISTILL_DIR="${REPO_ROOT}/reverie/backend_server/distill_logs"
TEACHER_LOG="${DISTILL_DIR}/teacher.jsonl"
ADAPTER_DIR="${REPO_ROOT}/models/student_lora/adapter"
TRAIN_SCRIPT="${REPO_ROOT}/scripts/finetune_student.py"
ACCELERATE_CONFIG="${REPO_ROOT}/configs/accelerate_fsdp.yaml"
VLLM_SCRIPT="${REPO_ROOT}/reverie/backend_server/scripts/run_vllm_qwen_student.sh"
LOG_DIR="${REPO_ROOT}/logs"
TRAIN_LOG="${LOG_DIR}/training.log"

# Defaults
THRESHOLD=300
POLL_INTERVAL=300
MAX_ROUNDS=0  # 0 = unlimited
DRY_RUN=false
ONCE=false
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
LORA_RANK="${LORA_RANK:-32}"

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threshold)    THRESHOLD="$2"; shift 2 ;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
    --max-rounds)   MAX_ROUNDS="$2"; shift 2 ;;
    --dry-run)      DRY_RUN=true; shift ;;
    --once)         ONCE=true; shift ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Activate venv
if [[ -f "${VENV_BIN}/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_BIN}/activate"
fi

mkdir -p "${LOG_DIR}" "${ADAPTER_DIR}"

# State file tracking last trained line count
STATE_FILE="${DISTILL_DIR}/.training_state"

get_line_count() {
  if [[ -f "${TEACHER_LOG}" ]]; then
    wc -l < "${TEACHER_LOG}" | tr -d ' '
  else
    echo 0
  fi
}

get_last_trained_count() {
  if [[ -f "${STATE_FILE}" ]]; then
    cat "${STATE_FILE}" | tr -d ' '
  else
    echo 0
  fi
}

save_trained_count() {
  echo "$1" > "${STATE_FILE}"
}

log() {
  local ts
  ts="$(date '+%Y-%m-%d %H:%M:%S')"
  echo "[${ts}] $*" | tee -a "${TRAIN_LOG}"
}

# Get PID from pidfile if process is alive
get_pid() {
  local pidfile="$1"
  if [[ -f "${pidfile}" ]]; then
    local pid
    pid="$(cat "${pidfile}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      echo "${pid}"
      return
    fi
  fi
  echo ""
}

find_vllm_pid() {
  # 1. Try pidfile
  local pid
  pid="$(get_pid "${REPO_ROOT}/run/vllm.pid")"
  if [[ -n "${pid}" ]]; then echo "${pid}"; return; fi
  # 2. Try port 8001
  pid="$(ss -ltnp 2>/dev/null \
    | awk '/:8001\b/ && /pid=[0-9]+/ { match($0,/pid=[0-9]+/); s=substr($0,RSTART,RLENGTH); sub(/^pid=/,"",s); print s; exit }')"
  if [[ -n "${pid}" ]]; then echo "${pid}"; return; fi
  echo ""
}

find_sim_pid() {
  # 1. Try pidfile
  local pid
  pid="$(get_pid "${REPO_ROOT}/run/sim.pid")"
  if [[ -n "${pid}" ]]; then echo "${pid}"; return; fi
  # 2. Try pgrep
  pid="$( (pgrep -af "run_reverie_headless\|run_sim_loop" 2>/dev/null || true) | awk 'NR==1{print $1}')"
  if [[ -n "${pid}" ]]; then echo "${pid}"; return; fi
  echo ""
}

stop_vllm() {
  log "Stopping vLLM..."

  # Kill by pidfile (the bash wrapper)
  local pid
  pid="$(get_pid "${REPO_ROOT}/run/vllm.pid")"
  if [[ -n "${pid}" ]]; then
    log "  Killing process group for wrapper pid ${pid}"
    kill -- -"${pid}" 2>/dev/null || kill "${pid}" 2>/dev/null || true
  fi

  # Also kill any process actually listening on port 8001
  local port_pid
  port_pid="$(find_vllm_pid)"
  if [[ -n "${port_pid}" ]] && [[ "${port_pid}" != "${pid}" ]]; then
    log "  Killing vLLM server pid ${port_pid} (from port 8001)"
    kill "${port_pid}" 2>/dev/null || true
  fi

  # Kill any remaining vLLM processes (server, engine, workers)
  local vllm_pids
  vllm_pids="$(pgrep -if 'vllm' 2>/dev/null || true)"
  if [[ -n "${vllm_pids}" ]]; then
    log "  Killing remaining vLLM processes: $(echo ${vllm_pids} | tr '\n' ' ')"
    echo "${vllm_pids}" | xargs kill 2>/dev/null || true
  fi

  # Wait for everything to exit
  for _ in $(seq 1 15); do
    if [[ -z "$(pgrep -if 'vllm' 2>/dev/null || true)" ]]; then
      break
    fi
    sleep 1
  done

  # Force kill anything left
  local remaining
  remaining="$(pgrep -if 'vllm' 2>/dev/null || true)"
  if [[ -n "${remaining}" ]]; then
    log "  Force killing remaining vLLM: $(echo ${remaining} | tr '\n' ' ')"
    echo "${remaining}" | xargs kill -9 2>/dev/null || true
    sleep 3
  fi

  # Final check: verify GPUs are free
  local gpu_used
  gpu_used="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | head -1 || true)"
  if [[ -n "${gpu_used}" ]]; then
    log "  WARNING: GPU still has processes after cleanup!"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | while read -r line; do
      log "    ${line}"
    done
  fi

  rm -f "${REPO_ROOT}/run/vllm.pid"
  log "vLLM stopped."
}

pause_sim() {
  log "Pausing sim (SIGSTOP)..."
  local pid
  pid="$(find_sim_pid)"
  if [[ -n "${pid}" ]]; then
    kill -STOP "${pid}" 2>/dev/null || true
    log "Sim paused (pid ${pid})."
  else
    log "Sim not running — nothing to pause."
  fi
}

resume_sim() {
  log "Resuming sim (SIGCONT)..."
  local pid
  pid="$(find_sim_pid)"
  if [[ -n "${pid}" ]]; then
    kill -CONT "${pid}" 2>/dev/null || true
    log "Sim resumed (pid ${pid})."
  else
    log "Sim not running — nothing to resume."
  fi
}

start_vllm() {
  log "Starting vLLM with LoRA adapter..."
  local vllm_log="${LOG_DIR}/vllm.log"
  local pidfile="${REPO_ROOT}/run/vllm.pid"

  local env_args=()
  if [[ -d "${ADAPTER_DIR}" ]] && [[ -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
    env_args+=(LORA_ADAPTER_PATH="${ADAPTER_DIR}")
    # Lower GPU memory utilization to leave room for LoRA adapter weights
    env_args+=(GPU_MEM_UTIL=0.85)
    log "  LoRA adapter: ${ADAPTER_DIR}"
    log "  GPU memory utilization: 0.85 (reduced for LoRA headroom)"
  else
    log "  No adapter found — starting base model only."
  fi

  env "${env_args[@]+"${env_args[@]}"}" \
    nohup bash "${VLLM_SCRIPT}" >"${vllm_log}" 2>&1 &
  echo $! > "${pidfile}"
  log "vLLM started (pid $(cat "${pidfile}")), waiting for health..."

  # Wait for vLLM to be ready (up to 6 min — model loading can be slow)
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:8001/health" >/dev/null 2>&1; then
      log "vLLM healthy."
      return 0
    fi
    sleep 2
  done
  log "WARNING: vLLM did not become healthy within timeout!"
  return 1
}

run_training() {
  local current_count="$1"
  log "=== Training round starting (${current_count} teacher entries) ==="

  if [[ "${DRY_RUN}" == "true" ]]; then
    log "[DRY RUN] Would train on ${current_count} entries."
    save_trained_count "${current_count}"
    return 0
  fi

  # 1. Pause sim
  pause_sim

  # 2. Stop vLLM to free GPU memory
  stop_vllm

  # 3. Run training (multi-GPU via accelerate if config exists, else single-GPU)
  log "Running fine-tuning..."
  local train_exit=0
  if [[ -f "${ACCELERATE_CONFIG}" ]]; then
    log "  Using accelerate FSDP (4 GPUs)..."
    accelerate launch --config_file "${ACCELERATE_CONFIG}" \
      "${TRAIN_SCRIPT}" \
      --base-model "${BASE_MODEL}" \
      --teacher-log "${TEACHER_LOG}" \
      --output-dir "${ADAPTER_DIR}" \
      --epochs "${TRAIN_EPOCHS}" \
      --lora-rank "${LORA_RANK}" \
      2>&1 | tee -a "${TRAIN_LOG}" \
      || train_exit=$?
  else
    log "  Using single-GPU QLoRA fallback..."
    python "${TRAIN_SCRIPT}" \
      --base-model "${BASE_MODEL}" \
      --teacher-log "${TEACHER_LOG}" \
      --output-dir "${ADAPTER_DIR}" \
      --epochs "${TRAIN_EPOCHS}" \
      --lora-rank "${LORA_RANK}" \
      2>&1 | tee -a "${TRAIN_LOG}" \
      || train_exit=$?
  fi

  if [[ "${train_exit}" -eq 0 ]]; then
    log "Training completed successfully."
    save_trained_count "${current_count}"
  elif [[ "${train_exit}" -eq 2 ]]; then
    log "Training skipped (not enough examples)."
  else
    log "ERROR: Training failed with exit code ${train_exit}."
  fi

  # 4. Restart vLLM (with adapter if training succeeded)
  start_vllm

  # 5. Resume sim
  resume_sim

  log "=== Training round complete ==="
  return "${train_exit}"
}

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

log "Training watcher started."
log "  Repo:       ${REPO_ROOT}"
log "  Teacher log: ${TEACHER_LOG}"
log "  Threshold:   ${THRESHOLD} new entries"
log "  Poll:        every ${POLL_INTERVAL}s"
log "  Max rounds:  ${MAX_ROUNDS:-unlimited}"
log "  Base model:  ${BASE_MODEL}"
log "  Epochs:      ${TRAIN_EPOCHS}"
log "  LoRA rank:   ${LORA_RANK}"
log ""

round=0

if [[ "${ONCE}" == "true" ]]; then
  current_count="$(get_line_count)"
  run_training "${current_count}"
  exit $?
fi

while true; do
  current_count="$(get_line_count)"
  last_trained="$(get_last_trained_count)"
  new_entries=$((current_count - last_trained))

  if [[ "${new_entries}" -ge "${THRESHOLD}" ]]; then
    round=$((round + 1))
    log "Trigger: ${new_entries} new entries (threshold: ${THRESHOLD}). Round ${round}."
    run_training "${current_count}" || true

    if [[ "${MAX_ROUNDS}" -gt 0 ]] && [[ "${round}" -ge "${MAX_ROUNDS}" ]]; then
      log "Reached max rounds (${MAX_ROUNDS}). Exiting."
      break
    fi
  fi

  sleep "${POLL_INTERVAL}"
done

log "Training watcher exiting."
