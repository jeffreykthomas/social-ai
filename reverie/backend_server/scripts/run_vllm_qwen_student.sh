#!/usr/bin/env bash
set -euo pipefail

# Serve the student model (Qwen2.5-32B-Instruct) via an OpenAI-compatible API
# across all 4 GPUs using tensor parallelism.
#
# Endpoint: http://localhost:8001/v1/chat/completions
#
# Prereqs:
#   pip install "vllm>=0.6.0"  (version depends on your environment)
#   and a working CUDA stack.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL="${MODEL:-Qwen/Qwen2.5-32B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8001}"
TP="${TP:-4}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.92}"
MAX_LEN="${MAX_LEN:-8192}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TP}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --max-model-len "${MAX_LEN}" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes


