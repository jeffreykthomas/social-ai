#!/usr/bin/env bash
set -euo pipefail
#
# End-to-end student training pipeline:
#   1. Convert teacher distillation logs -> training data
#   2. Fine-tune student model (LoRA on Qwen2.5)
#   3. Optionally merge adapter into full model
#   4. Optionally restart vLLM with updated weights
#
# Usage:
#   bash scripts/train_student.sh              # convert + train (LoRA adapter)
#   bash scripts/train_student.sh --merge      # also merge adapter into full model
#   bash scripts/train_student.sh --restart    # also restart vLLM with new weights
#   bash scripts/train_student.sh --dry-run    # just convert data; show stats
#
# Environment variables:
#   STUDENT_BASE_MODEL  Base model to fine-tune (default: openai/gpt-oss-20b)
#   TRAIN_EPOCHS        Number of training epochs (default: 3)
#   LORA_R              LoRA rank (default: 32)
#

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${REPO_ROOT}/venv/bin"

if [[ -f "${VENV_BIN}/activate" ]]; then
  # shellcheck disable=SC1091
  source "${VENV_BIN}/activate"
fi

MERGE=false
RESTART=false
DRY_RUN=false
EXTRA_ARGS=()

for arg in "$@"; do
  case "${arg}" in
    --merge)    MERGE=true ;;
    --restart)  RESTART=true; MERGE=true ;;
    --dry-run)  DRY_RUN=true ;;
    *)          EXTRA_ARGS+=("${arg}") ;;
  esac
done

DISTILL_LOGS="${REPO_ROOT}/reverie/backend_server/distill_logs"
TEACHER_LOG="${DISTILL_LOGS}/teacher.jsonl"
TRAINING_DIR="${DISTILL_LOGS}/training"
OUTPUT_DIR="${REPO_ROOT}/models/student_lora"

echo "============================================"
echo " Student Training Pipeline"
echo "============================================"
echo "  Teacher log:  ${TEACHER_LOG}"
echo "  Training dir: ${TRAINING_DIR}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Base model:   ${STUDENT_BASE_MODEL:-openai/gpt-oss-20b}"
echo ""

# -----------------------------------------------------------------------
# Step 1: Convert distillation logs to training data
# -----------------------------------------------------------------------
echo "[1/3] Converting teacher logs to training data ..."

if [[ ! -f "${TEACHER_LOG}" ]]; then
  echo "ERROR: No teacher log found at ${TEACHER_LOG}" >&2
  echo "  Run the simulation first to generate teacher distillation logs." >&2
  exit 1
fi

ENTRY_COUNT=$(wc -l < "${TEACHER_LOG}")
echo "  Found ${ENTRY_COUNT} entries in teacher.jsonl"

python "${REPO_ROOT}/scripts/distill_to_training_data.py" \
  --input "${TEACHER_LOG}" \
  --outdir "${TRAINING_DIR}"

echo ""

if ${DRY_RUN}; then
  echo "[DRY RUN] Showing formatted example preview ..."
  python "${REPO_ROOT}/scripts/finetune_student.py" \
    --train-data "${TRAINING_DIR}/train.jsonl" \
    --val-data "${TRAINING_DIR}/val.jsonl" \
    --output-dir "${OUTPUT_DIR}" \
    --dry-run
  echo ""
  echo "Dry run complete. No training performed."
  exit 0
fi

# -----------------------------------------------------------------------
# Step 2: Fine-tune with LoRA
# -----------------------------------------------------------------------
echo "[2/3] Fine-tuning student model ..."

FINETUNE_ARGS=(
  --train-data "${TRAINING_DIR}/train.jsonl"
  --val-data "${TRAINING_DIR}/val.jsonl"
  --output-dir "${OUTPUT_DIR}"
  --epochs "${TRAIN_EPOCHS:-3}"
  --lora-r "${LORA_R:-32}"
)

if ${MERGE}; then
  FINETUNE_ARGS+=(--merge)
fi

# Pass through any extra args (e.g., --base-model, --lr, etc.)
FINETUNE_ARGS+=("${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}")

python "${REPO_ROOT}/scripts/finetune_student.py" "${FINETUNE_ARGS[@]}"

echo ""

# -----------------------------------------------------------------------
# Step 3: Optionally restart vLLM
# -----------------------------------------------------------------------
if ${RESTART}; then
  echo "[3/3] Restarting vLLM with updated model ..."

  # Determine model path: prefer merged model, fallback to LoRA adapter
  if [[ -d "${OUTPUT_DIR}/merged" ]]; then
    NEW_MODEL="${OUTPUT_DIR}/merged"
    SERVE_MODE="merged"
  elif [[ -d "${OUTPUT_DIR}/adapter" ]]; then
    NEW_MODEL="${OUTPUT_DIR}/adapter"
    SERVE_MODE="lora"
  else
    echo "WARNING: No trained model found. Skipping vLLM restart." >&2
    exit 0
  fi

  echo "  Model:  ${NEW_MODEL}"
  echo "  Mode:   ${SERVE_MODE}"

  # Stop existing vLLM
  VLLM_PID=""
  if [[ -f "${REPO_ROOT}/run/vllm.pid" ]]; then
    VLLM_PID="$(cat "${REPO_ROOT}/run/vllm.pid" 2>/dev/null || true)"
  fi
  if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
    echo "  Stopping existing vLLM (pid ${VLLM_PID}) ..."
    kill "${VLLM_PID}" 2>/dev/null || true
    sleep 3
  fi

  if [[ "${SERVE_MODE}" == "merged" ]]; then
    # Serve the merged model directly
    echo "  Starting vLLM with merged model ..."
    MODEL="${NEW_MODEL}" nohup bash "${REPO_ROOT}/reverie/backend_server/scripts/run_vllm_qwen_student.sh" \
      > "${REPO_ROOT}/logs/vllm.log" 2>&1 &
    echo $! > "${REPO_ROOT}/run/vllm.pid"
  else
    # Serve base model with LoRA adapter
    echo "  Starting vLLM with LoRA adapter ..."
    LORA_ADAPTER_PATH="${NEW_MODEL}" \
      nohup bash "${REPO_ROOT}/reverie/backend_server/scripts/run_vllm_qwen_student.sh" \
      > "${REPO_ROOT}/logs/vllm.log" 2>&1 &
    echo $! > "${REPO_ROOT}/run/vllm.pid"
  fi

  echo "  vLLM restarted (pid $(cat "${REPO_ROOT}/run/vllm.pid"))"
else
  echo "[3/3] Skipping vLLM restart (pass --restart to auto-restart)"
fi

echo ""
echo "============================================"
echo " Training pipeline complete"
echo "============================================"
echo "  LoRA adapter:  ${OUTPUT_DIR}/adapter/"
if ${MERGE}; then
  echo "  Merged model:  ${OUTPUT_DIR}/merged/"
fi
echo ""
echo "To serve manually:"
echo "  # Option A: merged model"
echo "  MODEL=${OUTPUT_DIR}/merged bash reverie/backend_server/scripts/run_vllm_qwen_student.sh"
echo ""
echo "  # Option B: LoRA adapter on base model"
echo "  LORA_ADAPTER_PATH=${OUTPUT_DIR}/adapter bash reverie/backend_server/scripts/run_vllm_qwen_student.sh"
echo ""
