#!/bin/bash
# Monitoring en temps réel pendant le soak test
set -euo pipefail  # GPT improvement: strict mode

echo "📊 MONITORING JEFFREY OS - Actualisation toutes les 5 secondes"
echo "Appuyez sur Ctrl+C pour arrêter le monitoring (n'arrête pas le test)"
echo "================================================"

while true; do
    clear
    echo "📊 JEFFREY OS - MONITORING TEMPS RÉEL"
    echo "================================================"
    echo "$(date)"
    echo ""

    # Métriques depuis l'endpoint
    curl -s http://localhost:8000/metrics 2>/dev/null | grep -E "(p99|drops|symbiosis|memory_mb|dlq_count|events_processed)" || echo "⚠️ Métriques non disponibles"

    echo ""
    echo "================================================"
    echo "Actualisation dans 5 secondes..."
    sleep 5
done
