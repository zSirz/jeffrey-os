#!/bin/bash
# Script adaptatif qui détecte l'environnement et lance le bon serveur

set -euo pipefail

echo "🔍 Detecting environment..."

# Détection de l'OS
OS=$(uname -s)
ARCH=$(uname -m)

# Configuration par défaut
PORT=${LLM_PORT:-9010}
MODEL=${LLM_MODEL:-"mistral:7b-instruct"}

# Détection GPU NVIDIA
HAS_NVIDIA_GPU=false
if command -v nvidia-smi &>/dev/null; then
    if nvidia-smi &>/dev/null; then
        HAS_NVIDIA_GPU=true
    fi
fi

# Fonction pour créer le proxy Ollama
create_ollama_proxy() {
    cat > /tmp/ollama_proxy.py << 'EOF'
import os, time, asyncio, json, aiohttp
from aiohttp import web

PORT = int(os.getenv("PORT", "9010"))
HOST = "127.0.0.1"  # Sécurité: loopback only

async def proxy_handler(request: web.Request):
    path = request.path
    if path == "/v1/models":
        async with aiohttp.ClientSession() as session:
            async with session.get("http://127.0.0.1:11434/api/tags") as resp:
                data = await resp.json()
                models = [{"id": m.get("name",""), "object": "model"} for m in data.get("models", [])]
                return web.json_response({"data": models})

    elif path == "/v1/chat/completions":
        body = await request.json()
        ollama_req = {
            "model": body.get("model", "mistral:7b-instruct"),
            "messages": body.get("messages", []),
            "stream": False,
            "options": {
                "temperature": body.get("temperature", 0.7),
                "top_p": body.get("top_p", 0.9),
                "num_predict": body.get("max_tokens", 512),
            },
        }
        async with aiohttp.ClientSession() as session:
            async with session.post("http://127.0.0.1:11434/api/chat", json=ollama_req) as resp:
                result = await resp.json()
                response = {
                    "id": "chatcmpl-jeffrey",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.get("model", "mistral:7b-instruct"),
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.get("message", {}).get("content", "")
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0),
                        "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                    }
                }
                return web.json_response(response)

    elif path == "/metrics":
        metrics = (
            "# HELP llm_ready LLM server ready\n"
            "# TYPE llm_ready gauge\n"
            "llm_ready 1\n"
        )
        return web.Response(text=metrics, content_type="text/plain")

    return web.json_response({"error": "Not found"}, status=404)

app = web.Application()
app.router.add_route("*", "/{tail:.*}", proxy_handler)

if __name__ == "__main__":
    web.run_app(app, host=HOST, port=PORT)
EOF
}

# Fonction pour lancer Ollama
setup_ollama() {
    echo "🐙 Setting up Ollama..."

    # Vérifier qu'Ollama est installé
    if ! command -v ollama &>/dev/null; then
        echo "❌ Ollama not installed. Installing..."
        if [[ "$OS" == "Darwin" ]]; then
            # Sur Mac, préférer Homebrew si disponible
            if command -v brew &>/dev/null; then
                brew install ollama
            else
                curl -fsSL https://ollama.com/install.sh | sh
            fi
        else
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    fi

    # Démarrer Ollama si pas déjà lancé
    if ! pgrep -x "ollama" >/dev/null; then
        echo "🚀 Starting Ollama server..."
        ollama serve &
        OLLAMA_PID=$!
        sleep 3
    fi

    # Télécharger le modèle si nécessaire
    echo "📦 Ensuring model $MODEL is available..."
    if ! ollama list | grep -q "$MODEL"; then
        ollama pull "$MODEL" || {
            echo "⚠️ Model $MODEL not found, trying mistral:7b-instruct"
            MODEL="mistral:7b-instruct"
            ollama pull "$MODEL"
        }
    fi

    # Créer et lancer le proxy OpenAI-compatible
    echo "🌉 Creating OpenAI-compatible proxy on port $PORT..."
    create_ollama_proxy
    PORT=$PORT python /tmp/ollama_proxy.py &
    PROXY_PID=$!

    echo "✅ Ollama ready on port $PORT (via proxy)"
    echo "   Model: $MODEL"
    echo "   Original Ollama: http://localhost:11434"
    echo "   OpenAI-compatible: http://localhost:$PORT"
}

# Choix du serveur selon l'environnement
if [[ "$OS" == "Darwin" ]]; then
    echo "🍎 macOS detected - Using Ollama"
    setup_ollama

elif [[ "$HAS_NVIDIA_GPU" == "true" ]]; then
    echo "🎮 NVIDIA GPU detected - Using vLLM"

    if ! command -v vllm &>/dev/null; then
        echo "📦 Installing vLLM..."
        pip install vllm
    fi

    # Pour vLLM, on utilise Apertus ou un modèle similaire
    if [[ "$MODEL" == *"mistral"* ]] || [[ "$MODEL" == *"llama"* ]]; then
        # Remapper vers modèles HuggingFace pour vLLM
        MODEL="mistralai/Mistral-7B-Instruct-v0.2"
    else
        MODEL="swiss-ai/Apertus-8B-Instruct-2509"
    fi

    echo "🚀 Starting vLLM with model: $MODEL"
    vllm serve "$MODEL" \
        --port "$PORT" \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        --trust-remote-code &
    VLLM_PID=$!

else
    echo "⚠️ No GPU detected on Linux - Using Ollama CPU mode"
    setup_ollama
fi

# Attendre que le serveur soit prêt
echo "⏳ Waiting for LLM server to be ready..."
for i in {1..30}; do
    if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "✅ LLM server is ready!"
        break
    fi
    sleep 2
done

# Trap pour nettoyer à la sortie
trap "kill $PROXY_PID $OLLAMA_PID $VLLM_PID 2>/dev/null || true" EXIT

# Garder le script actif
echo "💚 LLM server running. Press Ctrl+C to stop."
wait
