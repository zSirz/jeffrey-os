#!/bin/bash
# Script de démarrage pour la Phase 1

echo "🚀 Starting Jeffrey OS - Phase 1"
echo "================================"

# Vérifier la config
if [ ! -f "config/modules.yaml" ]; then
    echo "⚠️  Creating default config..."
    mkdir -p config
    cat > config/modules.yaml << 'EOF'
modules:
  - name: guardians_hub
    import: "src.jeffrey.core.guardians.guardians_hub:GuardiansHub"
    enabled: true
    critical: false

  - name: cognitive_core
    import: "src.jeffrey.core.cognition.cognitive_core_lite:CognitiveCoreLite"
    enabled: true
    critical: true
EOF
fi

# Vérifier Redis (optionnel)
redis-cli ping > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✅ Redis is running"
else
    echo "⚠️  Redis not available (using fallback)"
fi

# Démarrer l'API
echo ""
echo "📡 Starting API on http://localhost:8000"
echo "📚 Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Lancer avec uvicorn
uvicorn src.jeffrey.core.control.control_plane:app \
    --host 127.0.0.1 \
    --port 8000 \
    --reload
