#!/usr/bin/env bash
# ============================================================================
# setup_ollama.sh — Install Ollama and pull models on Axiom Core
#
# Run on Axiom Core (192.168.4.25):
#   chmod +x scripts/setup_ollama.sh
#   ./scripts/setup_ollama.sh
#
# Prerequisites: NVIDIA drivers + CUDA already installed (verified by nvidia-smi)
# ============================================================================

set -euo pipefail

echo "=== Noesis Axiom Core — Ollama Setup ==="
echo ""

# ── Step 1: Check GPU availability ──────────────────────────────────────────
echo "[1/6] Checking GPU availability..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "  Found $GPU_COUNT GPU(s):"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/    /'
echo ""

# ── Step 2: Install Ollama ──────────────────────────────────────────────────
echo "[2/6] Installing Ollama..."
if command -v ollama &>/dev/null; then
    echo "  Ollama already installed: $(ollama --version)"
    echo "  Updating..."
fi
curl -fsSL https://ollama.com/install.sh | sh
echo "  Installed: $(ollama --version)"
echo ""

# ── Step 3: Configure Ollama systemd service ────────────────────────────────
echo "[3/6] Configuring Ollama for dual GPU..."

# Create systemd override directory
sudo mkdir -p /etc/systemd/system/ollama.service.d

sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null <<'EOF'
[Service]
Environment="OLLAMA_NUM_GPU=2"
Environment="CUDA_VISIBLE_DEVICES=0,1"
Environment="OLLAMA_HOST=127.0.0.1:11434"
Environment="OLLAMA_MAX_LOADED_MODELS=3"
Environment="OLLAMA_KEEP_ALIVE=24h"
Environment="DO_NOT_TRACK=1"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama
echo "  Ollama configured and restarted"

# Wait for Ollama to be ready
echo "  Waiting for Ollama to start..."
for i in $(seq 1 15); do
    if curl -s http://127.0.0.1:11434/api/tags &>/dev/null; then
        echo "  Ollama is ready"
        break
    fi
    sleep 1
done
echo ""

# ── Step 4: Pull primary reasoning model ────────────────────────────────────
echo "[4/6] Pulling primary reasoning model: qwen3.5:35b-a3b"
echo "  This may take 15-30 minutes on first download..."
ollama pull qwen3.5:35b-a3b
echo "  Done"
echo ""

# ── Step 5: Pull embedding model (for Continue.dev codebase indexing) ───────
echo "[5/6] Pulling embedding model: nomic-embed-text"
ollama pull nomic-embed-text
echo "  Done"
echo ""

# ── Step 6: Create axiom-primary custom model ───────────────────────────────
echo "[6/6] Creating axiom-primary custom model..."
MODELFILE_DIR="$(cd "$(dirname "$0")/.." && pwd)/modelfiles"
if [ -f "$MODELFILE_DIR/Modelfile.axiom-primary" ]; then
    ollama create axiom-primary -f "$MODELFILE_DIR/Modelfile.axiom-primary"
    echo "  Created axiom-primary with custom system prompt"
else
    echo "  WARNING: $MODELFILE_DIR/Modelfile.axiom-primary not found"
    echo "  Skipping custom model creation — will use base qwen3.5:35b-a3b"
fi
echo ""

# ── Verification ────────────────────────────────────────────────────────────
echo "=== Verification ==="
echo ""
echo "Installed models:"
ollama list
echo ""

echo "Ollama binding:"
ss -tlnp 2>/dev/null | grep 11434 || echo "  (check with: ss -tlnp | grep 11434)"
echo ""

echo "Quick smoke test..."
RESPONSE=$(curl -s http://127.0.0.1:11434/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "axiom-primary",
        "messages": [{"role": "user", "content": "Respond with exactly: OLLAMA_OK"}],
        "max_tokens": 10,
        "temperature": 0.0
    }' 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null || echo "FAILED")

if echo "$RESPONSE" | grep -qi "OK"; then
    echo "  Smoke test: PASSED"
else
    echo "  Smoke test response: $RESPONSE"
    echo "  (Model may still be loading — try again in a few seconds)"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Stop vLLM if still running:  pkill -f 'vllm'"
echo "  2. Restart Noesis API server:   (uses Ollama on port 11434 now)"
echo "  3. Test clinical validation:    curl -X POST http://localhost:8008/validate/concepts ..."
echo ""
echo "To add coding model later:"
echo "  ollama pull qwen2.5-coder:32b"
echo "  ollama create axiom-coder -f modelfiles/Modelfile.axiom-coder"
