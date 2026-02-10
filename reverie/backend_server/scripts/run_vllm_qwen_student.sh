#!/usr/bin/env bash
set -euo pipefail

# Serve the student model via an OpenAI-compatible API.
#
# Default model: GPT-OSS-120B (OpenAI open-weight MoE, Apache 2.0)
#   - 117B total params, 5.1B active -> fits on a single 80GB GPU
#   - Native tool calling, vLLM-optimised MoE kernels
#   - TP=1 is usually sufficient; scale to TP=2 for higher throughput
#
# Fallback: set MODEL=Qwen/Qwen2.5-32B-Instruct TP=4 for the previous config.
#
# Endpoint: http://localhost:8001/v1/chat/completions
#
# Prereqs:
#   pip install "vllm>=0.10.2"  (GPT-OSS requires 0.10.2+)
#   and a working CUDA stack.
#
# LoRA adapter support:
#   Set LORA_ADAPTER_PATH to serve a LoRA adapter on top of the base model.
#   The adapter will be registered as model name "student-distilled" in the
#   vLLM server, so you can target it via model="student-distilled" in API
#   calls (the base model name also remains available).
#
#   Example:
#     LORA_ADAPTER_PATH=./models/student_lora/adapter bash run_vllm_qwen_student.sh
#
# Merged model support:
#   Set MODEL to a local path to serve a fully merged fine-tuned model:
#     MODEL=./models/student_lora/merged bash run_vllm_qwen_student.sh

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# Default: Qwen2.5-32B-Instruct (works on all GPU architectures with TP=4).
#
# To use GPT-OSS (requires H100/H200 for MXFP4 kernels):
#   MODEL=openai/gpt-oss-120b TP=1 bash run_vllm_qwen_student.sh
#   MODEL=openai/gpt-oss-20b  TP=1 bash run_vllm_qwen_student.sh
MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
# Qwen2.5-32B needs TP=4 on 48GB GPUs. GPT-OSS-120B needs TP=1 on H100.
TP="${TP:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_LEN="${MAX_LEN:-8192}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"

EXTRA_ARGS=()

# GPT-OSS benefits from async scheduling for MoE workloads
EXTRA_ARGS+=(--async-scheduling)

if [[ -n "${LORA_ADAPTER_PATH}" && -d "${LORA_ADAPTER_PATH}" ]]; then
  echo "LoRA adapter detected: ${LORA_ADAPTER_PATH}"
  EXTRA_ARGS+=(--enable-lora)
  EXTRA_ARGS+=(--lora-modules "student-distilled=${LORA_ADAPTER_PATH}")
  EXTRA_ARGS+=(--max-lora-rank 64)
fi

echo "Serving model: ${MODEL} (TP=${TP})"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_LEN}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"


