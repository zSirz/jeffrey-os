#!/bin/bash
# Script d'installation et configuration d'Ollama pour Jeffrey OS

set -euo pipefail

echo "🚀 Installing and configuring Ollama for Jeffrey OS"

OS=$(uname -s)

# Installation d'Ollama selon l'OS
if [[ "$OS" == "Darwin" ]]; then
    echo "🍎 macOS detected"

    # Vérifier si Ollama est déjà installé
    if command -v ollama &>/dev/null; then
        echo "✅ Ollama is already installed"
    else
        # Préférer Homebrew si disponible
        if command -v brew &>/dev/null; then
            echo "📦 Installing Ollama via Homebrew..."
            brew install ollama
        else
            echo "📦 Installing Ollama via official script..."
            curl -fsSL https://ollama.com/install.sh | sh
        fi
    fi
else
    echo "🐧 Linux detected"

    if command -v ollama &>/dev/null; then
        echo "✅ Ollama is already installed"
    else
        echo "📦 Installing Ollama via official script..."
        curl -fsSL https://ollama.com/install.sh | sh
    fi
fi

# Démarrer le serveur Ollama
echo "🚀 Starting Ollama server..."
if ! pgrep -x "ollama" >/dev/null; then
    ollama serve &
    OLLAMA_PID=$!
    echo "⏳ Waiting for Ollama to be ready..."
    sleep 5
else
    echo "✅ Ollama server is already running"
fi

# Fonction pour télécharger un modèle
download_model() {
    local model=$1
    echo "📦 Downloading model: $model"
    if ollama list | grep -q "$model"; then
        echo "✅ Model $model already downloaded"
    else
        ollama pull "$model"
    fi
}

# Télécharger les modèles recommandés
echo ""
echo "📦 Downloading recommended models..."
echo "This may take a few minutes on first run..."

# Mistral 7B - Excellent pour le français
download_model "mistral:7b-instruct"

# Optionnel : modèles supplémentaires (commentés par défaut)
echo ""
echo "Optional models (uncomment to download):"
echo "  # download_model 'llama3.2:3b'  # Llama 3.2 - Plus récent, multilingue"
echo "  # download_model 'phi3:medium'   # Phi-3 - Léger et rapide"

# Tester que tout fonctionne
echo ""
echo "🧪 Testing Ollama installation..."
echo "Test query" | ollama run mistral:7b-instruct "Réponds simplement 'OK' si tu fonctionnes." || {
    echo "❌ Test failed. Please check Ollama installation."
    exit 1
}

echo ""
echo "✨ Ollama is ready!"
echo ""
echo "📊 Installed models:"
ollama list

echo ""
echo "🎯 Next steps:"
echo "  1. Start the LLM server: ./scripts/start_llm_server.sh"
echo "  2. Run tests: python tests/test_llm_adaptive.py"
echo "  3. Start Jeffrey OS: python main.py"
echo ""
echo "💡 Tip: Ollama will use the OpenAI-compatible API on port 9010 via our proxy"
