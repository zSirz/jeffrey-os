#!/bin/bash

echo "🧠 Jeffrey OS - NeuralBus Launch Sequence"
echo "========================================="

# 1. Installer dépendances performance
echo "Installing performance dependencies..."
pip install -q uvloop lz4 orjson 2>/dev/null || {
    echo "⚠️ Some optional dependencies could not be installed"
}

# 2. Vérifier NATS
if ! pgrep -x "nats-server" > /dev/null; then
    echo "Starting NATS JetStream..."
    # Try to start NATS if installed
    if command -v nats-server &> /dev/null; then
        nats-server -js &
        sleep 2
    else
        echo "⚠️ NATS server not found. Please install with: brew install nats-server"
        echo "   Continuing with mock bus..."
    fi
else
    echo "✅ NATS server already running"
fi

# 3. Environnement optimisé
export PYTHONUNBUFFERED=1
export NEURALBUS_BATCH_SIZE=100  # Augmenter batching
export NEURALBUS_USE_MSGPACK=false  # JSON plus compatible pour debug
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 4. Créer répertoire de logs si nécessaire
mkdir -p logs

# 5. Lancer avec monitoring
echo "Starting Jeffrey with NeuralBus..."
echo "Press Ctrl+C to stop"
echo ""

# Lancer Python avec le script de monitoring
python3 -u scripts/run_with_monitoring.py 2>&1 | tee logs/jeffrey_neuralbus_$(date +%Y%m%d_%H%M%S).log
