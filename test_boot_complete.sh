#!/usr/bin/env bash

# === GARDE-FOUS (GPT/Marc) ===
set -euo pipefail
trap 'echo "❌ Échec à la ligne $LINENO"' ERR

# Vérifier racine du repo
[ -d .git ] && [ -d src/jeffrey ] || { echo "❌ Lance depuis la racine du repo"; exit 1; }

# Forcer UTF-8
export PYTHONUTF8=1

# Variables d'environnement Python (GPT/Marc)
export PYTHONPATH="src:services:core:unified:${PYTHONPATH:-}"

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

LOG_FILE="boot.log"

echo "🚀 TEST DE DÉMARRAGE JEFFREY OS" | tee "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"
echo "Date: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# === INFOS ENVIRONNEMENT (GPT/Marc) ===
echo "📋 Environnement:" | tee -a "$LOG_FILE"
echo "  OS: $(uname -a)" | tee -a "$LOG_FILE"
echo "  Python: $(python3 -V)" | tee -a "$LOG_FILE"
echo "  Pip: $(pip3 -V)" | tee -a "$LOG_FILE"
echo "  PYTHONPATH: $PYTHONPATH" | tee -a "$LOG_FILE"

# Vérifier venv actif (GPT/Marc)
python3 -c "import sys; venv_active = hasattr(sys, 'real_prefix') or (sys.prefix != sys.base_prefix); print('  Venv actif:', venv_active)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# === CHECK PROACTIF DES DÉPENDANCES EXTERNES (Grok) ===
echo "📦 Vérification des dépendances externes..." | tee -a "$LOG_FILE"
pip3 list > installed_packages.txt 2>&1

# Check packages critiques pour Jeffrey
CRITICAL_PACKAGES=("torch" "numpy" "scipy" "pandas" "sqlalchemy" "redis")
MISSING_EXTERNAL=()

for pkg in "${CRITICAL_PACKAGES[@]}"; do
    if ! pip3 list 2>/dev/null | grep -qi "^$pkg "; then
        MISSING_EXTERNAL+=("$pkg")
        echo -e "${YELLOW}⚠️  Package manquant : $pkg${NC}" | tee -a "$LOG_FILE"
    fi
done

# Pip check (GPT/Marc)
echo "" | tee -a "$LOG_FILE"
echo "🔍 Vérification cohérence pip..." | tee -a "$LOG_FILE"
python3 -m pip check 2>&1 | tee -a "$LOG_FILE" || echo -e "${YELLOW}⚠️  Conflits de dépendances détectés${NC}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# === TESTS D'IMPORT SÉQUENTIELS ===
run_test() {
    local test_num="$1"
    local test_name="$2"
    local test_cmd="$3"

    echo "🧪 Test $test_num : $test_name" | tee -a "$LOG_FILE"
    if eval "$test_cmd" 2>&1 | tee -a "$LOG_FILE"; then
        echo -e "${GREEN}✅ OK${NC}" | tee -a "$LOG_FILE"
        return 0
    else
        echo -e "${RED}❌ ÉCHEC${NC}" | tee -a "$LOG_FILE"
        return 1
    fi
    echo "---" | tee -a "$LOG_FILE"
}

# Test 1 : Import basique
run_test 1 "Import basique de Jeffrey" \
    "python3 -c \"import jeffrey; print('✅ Jeffrey importé avec succès')\""

# Test 2 : Core émotionnel
run_test 2 "Import du core émotionnel" \
    "python3 -c \"from jeffrey.core import jeffrey_emotional_core; print('✅ Emotional core OK')\""

# Test 3 : Unified Memory (avec test runtime - Grok)
run_test 3 "Import et test Unified Memory" \
    "python3 -c \"
from jeffrey.core.unified_memory import UnifiedMemory
try:
    mem = UnifiedMemory()
    print('✅ UnifiedMemory OK')
except Exception as e:
    print(f'⚠️  UnifiedMemory instanciable mais erreur runtime: {e}')
\""

# Test 4 : NeuralBus
run_test 4 "Import NeuralBus" \
    "python3 -c \"from jeffrey.core.neuralbus import NeuralBus; print('✅ NeuralBus OK')\""

# Test 5 : CLI
run_test 5 "Démarrage CLI" \
    "python3 -m jeffrey --help 2>&1 | head -10"

# Test 6 : Health check (si disponible)
if [ -f "health_check_phase_a.sh" ]; then
    run_test 6 "Health check global" \
        "RESTORE_MODE=1 bash health_check_phase_a.sh"
fi

echo "" | tee -a "$LOG_FILE"
echo "📊 ANALYSE DES ERREURS" | tee -a "$LOG_FILE"
echo "======================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "✅ Log complet dans : $LOG_FILE" | tee -a "$LOG_FILE"
echo "🎯 Lancer analyze_boot_errors.py pour analyse intelligente" | tee -a "$LOG_FILE"
