#!/bin/bash
set -e

echo "🔧 Jeffrey OS - Project Setup"
echo "=============================="

# Détection du répertoire racine
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd "${PROJECT_ROOT}"

# Installation en mode éditable (résout TOUS les problèmes d'import)
echo "📦 Installing project in editable mode..."
pip install -e . --quiet

# Vérification avec le bon import path (jeffrey.* pas src.jeffrey.*)
echo "✅ Verifying installation..."
python -c "from jeffrey.core.loops.loop_manager import LoopManager; print('✅ Imports working!')"

# Création des dossiers nécessaires
echo "📁 Creating required directories..."
mkdir -p logs .nats data/metrics

echo "✅ Setup complete! No more import errors!"
