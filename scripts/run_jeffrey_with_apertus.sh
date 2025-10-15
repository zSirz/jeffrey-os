#!/bin/bash
# Lance Jeffrey OS avec Apertus intégré

echo "🚀 Starting Jeffrey OS with Apertus Language Model"

# 1. Vérifier les dépendances
echo "📦 Checking dependencies..."
python -c "import vllm; import openai" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install vllm openai
}

# 2. Démarrer Apertus en arrière-plan
echo "🧠 Starting Apertus server..."
./scripts/start_apertus.sh &
APERTUS_PID=$!

# Attendre que le serveur soit prêt
echo "⏳ Waiting for Apertus to be ready..."
for i in {1..60}; do
    curl -sf http://localhost:9010/v1/models >/dev/null && break
    sleep 2
done

# 3. Lancer les tests smoke
echo "🧪 Running smoke tests..."
python -m pytest tests/test_apertus_smoke.py -v

# 4. Démarrer Jeffrey OS
echo "✨ Starting Jeffrey OS Core..."
python main.py

# Cleanup à la fin
trap "kill $APERTUS_PID 2>/dev/null" EXIT
