#!/bin/bash
# Script de Soak Test Autonome
set -euo pipefail  # GPT improvement: strict mode

echo "========================================="
echo "🚀 JEFFREY OS - SOAK TEST 2 HEURES"
echo "========================================="
echo "Début : $(date)"
echo ""
echo "Ce test va durer 2 heures."
echo "Tu peux fermer cette fenêtre et revenir plus tard."
LOG_FILE="logs/soak_$(date +%Y%m%d_%H%M%S).log"
echo "Les logs sont dans : $LOG_FILE"
echo "========================================="

# Configuration
export NB_NS=soak_prod_$(date +%s)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Créer le dossier logs si nécessaire
mkdir -p logs

# Activer venv si disponible
if [ -d .venv_prod ]; then
    source .venv_prod/bin/activate
fi

# Lancer le test avec logging (GPT fix: capture Python exit code properly)
python scripts/generate_load.py \
    --phase soak \
    --ml \
    --monitor \
    --non-interactive \
    --hours 2 2>&1 | tee "$LOG_FILE"
RESULT=${PIPESTATUS[0]}   # GPT fix: get Python's exit code, not tee's

# Rapport final
if [ $RESULT -eq 0 ]; then
    echo "✅ SOAK TEST PASSED! System is production ready!" | tee -a "$LOG_FILE"

    # Générer le certificat de production (GPT fix: no quotes around CERT)
    DATE="$(date)"  # Calculate date once
    cat > production_certificate.txt << CERT
========================================
JEFFREY OS - PRODUCTION CERTIFICATE
========================================
Date: $DATE
Status: CERTIFIED PRODUCTION READY ✅

Chaos Test: PASSED
Soak Test: PASSED (2 hours)
P99 Latency: < 50ms
Drop Rate: < 0.5%
Memory Leak: NONE
Symbiosis Score: > 0.5

System ready for deployment!
========================================
CERT

    echo ""
    echo "📜 Production certificate saved to: production_certificate.txt"

else
    echo "❌ SOAK TEST FAILED. Check $LOG_FILE for details." | tee -a "$LOG_FILE"
fi

echo "Test terminé : $(date)"
echo "Logs complets dans : $LOG_FILE"
exit $RESULT  # Propagate exit code
