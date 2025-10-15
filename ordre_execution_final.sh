#!/bin/bash
set -euo pipefail

echo "🔧 RÉPARATION JEFFREY OS - VERSION FINALE"
echo "=========================================="

# ÉTAPE 0 : Nettoyage total
echo ""
echo "ÉTAPE 0 : Nettoyage complet..."
rm -rf src/jeffrey/simple_modules/ 2>/dev/null || true
git checkout -- src/jeffrey/core/consciousness_loop.py 2>/dev/null || true
rm -f inventory_8regions.json best_modules.json discovered_regions.json 2>/dev/null || true
rm -f create_simple_modules.py fix_all_modules.py final_fix_modules.py 2>/dev/null || true

# ÉTAPE 1 : Détection
echo ""
echo "ÉTAPE 2 : Détection des modules..."
python3 detect_best_modules.py || exit 1
python3 discover_all_regions.py || exit 1

# ÉTAPE 2 : Patch inventaire
echo ""
echo "ÉTAPE 3 : Patch de l'inventaire..."
python3 patch_inventory_strict.py || {
    echo ""
    echo "❌ ÉCHEC : Impossible de trouver 8/8 régions"
    echo "TU DOIS DÉVELOPPER LES MANQUANTES PROPREMENT"
    exit 1
}

# ÉTAPE 3 : Validations
echo ""
echo "ÉTAPE 4 : Validation stricte..."
python3 scripts/ban_simple_modules.py || exit 1
python3 scripts/git_trust_check.py || exit 1
python3 scripts/hard_verify_realness.py || exit 1
PYTHONPATH="$(pwd)/src" python3 validate_8_regions_strict.py || exit 1

# ÉTAPE 4 : Clean room
echo ""
echo "ÉTAPE 5 : Clean room validation..."
bash scripts/ci_cleanroom.sh || exit 1

echo ""
echo "=========================================="
echo "✅ RÉPARATION TERMINÉE"
echo "Jeffrey OS v10.0.0 - 8/8 RÉGIONS RÉELLES"
echo "=========================================="
echo ""
echo "📊 Vérifier le rapport: artifacts/validation_report.json"
echo ""
