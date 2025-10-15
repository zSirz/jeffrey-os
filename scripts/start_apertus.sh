#!/bin/bash
# Script pour démarrer le serveur Apertus avec vLLM

set -euo pipefail

echo "🚀 Starting Apertus Language Model Server for Jeffrey OS"

# Configuration
MODEL="swiss-ai/Apertus-8B-Instruct-2509"
PORT=9010
MAX_MODEL_LEN=8192

# Vérifier si GPU disponible
if command -v nvidia-smi &> /dev/null; then
    echo "✅ GPU detected, using CUDA acceleration"
else
    echo "⚠️  No GPU detected, using CPU (slower performance)"
fi

# Lancer vLLM
echo "📦 Loading model: $MODEL"
vllm serve "$MODEL" \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code

echo "✨ Apertus server ready on port $PORT"
