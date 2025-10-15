#!/usr/bin/env bash

# === GARDE-FOUS (GPT/Marc) ===
set -euo pipefail
trap 'echo "❌ Échec à la ligne $LINENO - Voir boot.log pour détails"' ERR

[ -d .git ] && [ -d src/jeffrey ] || { echo "❌ Lance depuis la racine du repo"; exit 1; }

export PYTHONUTF8=1

echo "🚀 SÉQUENCE COMPLÈTE DE RESTAURATION JEFFREY OS"
echo "==============================================="
echo ""

# PHASE 1 : CHECKPOINT
echo "📌 PHASE 1 : Checkpoint Git"
echo "----------------------------"
git checkout -b chore/pre-boot-codemod 2>/dev/null || git checkout chore/pre-boot-codemod
git add -A
git commit -m "chore: checkpoint avant corrections pré-boot" 2>/dev/null || echo "⏭️  Aucun changement à committer"
echo "✅ Checkpoint créé"
echo ""

# PHASE 2 : CORRECTIONS IMPORTS
echo "🔧 PHASE 2 : Correction des imports"
echo "------------------------------------"
python3 fix_core_imports_complete.py
git add -A
git commit -m "fix(imports): Correction from core. → from jeffrey.core." 2>/dev/null || echo "⏭️  Aucun changement"
echo "✅ Imports corrigés"
echo ""

# PHASE 3 : PRE-COMMIT
echo "📦 PHASE 3 : Migration pre-commit"
echo "----------------------------------"
pre-commit migrate-config 2>/dev/null || echo "⏭️  Déjà migré"
git add .pre-commit-config.yaml
git commit -m "chore(pre-commit): migration config" 2>/dev/null || echo "⏭️  Aucun changement"
echo "✅ Pre-commit configuré"
echo ""

# PHASE 4 : VALIDATION
echo "🔒 PHASE 4 : Validation stricte"
echo "--------------------------------"
RESTORE_MODE=1 bash validate_strict.sh
echo "✅ Validation passée"
echo ""

# PHASE 5 : TEST BOOT
echo "🧪 PHASE 5 : Test de démarrage"
echo "-------------------------------"
bash test_boot_complete.sh
echo "✅ Tests exécutés"
echo ""

# PHASE 6 : ANALYSE
echo "🔍 PHASE 6 : Analyse intelligente"
echo "----------------------------------"
python3 analyze_boot_errors.py
echo "✅ Analyse terminée"
echo ""

# PHASE 7 : RAPPORT
echo "📊 PHASE 7 : Consultation du rapport"
echo "-------------------------------------"
echo ""
if [ -f "BOOT_ANALYSIS.md" ]; then
    echo "📖 Rapport complet dans : BOOT_ANALYSIS.md"
    echo ""
    echo "🎯 TOP 3 ACTIONS RECOMMANDÉES :"
    echo "--------------------------------"
    grep -A 3 "^### 1\." BOOT_ANALYSIS.md || echo "Voir BOOT_ANALYSIS.md"
    echo ""
    grep -A 3 "^### 2\." BOOT_ANALYSIS.md || echo ""
    echo ""
    grep -A 3 "^### 3\." BOOT_ANALYSIS.md || echo ""
else
    echo "❌ Rapport non généré"
fi

echo ""
echo "✅ SÉQUENCE COMPLÈTE TERMINÉE"
echo "=============================="
echo ""
echo "🎯 PROCHAINE ÉTAPE :"
echo "1. Consulter BOOT_ANALYSIS.md"
echo "2. Appliquer l'action Priorité N°1"
echo "3. Relancer : bash test_boot_complete.sh"
echo "4. Ré-analyser : python3 analyze_boot_errors.py"
echo "5. Répéter jusqu'à boot OK ✅"
echo ""
echo "📖 Rappel Doctrine Gemini (La Règle des Trois) :"
echo "   1. Prioriser : Focus sur le N°1"
echo "   2. Isoler : Une branche par fix"
echo "   3. Valider : Re-tester après chaque fix"
