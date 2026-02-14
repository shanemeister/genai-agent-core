#!/usr/bin/env bash
# ============================================================================
# launch_vllm.sh — Start vLLM serving DeepSeek-R1-Distill-Llama-70B
#
# Runs on Axiom Core (2x NVIDIA RTX A6000, 98GB total VRAM).
# Tensor-parallel across both GPUs for full FP16 or auto-quantized inference.
#
# Usage:
#   chmod +x launch_vllm.sh
#   ./launch_vllm.sh
#
# Then start the proxy:
#   uvicorn core.api.main_vllm:app --host 0.0.0.0 --port 8080 --reload
# ============================================================================

set -euo pipefail

MODEL_DIR="./models/deepseek-r1-70b-w4a16"
PORT=8081
TP_SIZE=2                  # tensor-parallel across 2x A6000
MAX_MODEL_LEN=4096         # conservative context; raise if needed
GPU_UTIL=0.90              # 90% of 98GB = ~88GB usable

echo "=== Noesis vLLM Server ==="
echo "Model:  $MODEL_DIR"
echo "Port:   $PORT"
echo "GPUs:   $TP_SIZE (tensor-parallel)"
echo "Quant:  GPTQ W4A16 (pre-quantized)"
echo "=========================="

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --tensor-parallel-size "$TP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --dtype auto
