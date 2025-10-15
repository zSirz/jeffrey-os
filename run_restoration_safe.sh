#!/bin/bash
# Script d'exécution sécurisée avec dry-run

set -e

echo "🚀 RESTAURATION SÉCURISÉE JEFFREY OS V2"
echo "========================================"
echo ""

# Check prerequisites
echo "🔍 Vérification des prérequis..."
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 requis"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Git requis"
    exit 1
fi

echo "   ✅ Prérequis OK"
echo ""

# Menu
echo "📋 OPTIONS :"
echo "  1. Diagnostic complet (dry-run, safe)"
echo "  2. Diagnostic + Extraction contrats (dry-run)"
echo "  3. Diagnostic + Contrats + Shims (dry-run)"
echo "  4. TOUT EXÉCUTER avec --apply (ÉCRITURE RÉELLE)"
echo "  5. Validation seulement"
echo ""
read -p "Choix (1-5) : " CHOICE
echo ""

case $CHOICE in
    1)
        echo "=== DIAGNOSTIC COMPLET ==="
        python3 comprehensive_diagnostic_v2.py
        ;;

    2)
        echo "=== DIAGNOSTIC + CONTRATS ==="
        python3 comprehensive_diagnostic_v2.py
        echo ""
        python3 extract_interface_contracts.py
        ;;

    3)
        echo "=== DIAGNOSTIC + CONTRATS + SHIMS (DRY-RUN) ==="
        python3 comprehensive_diagnostic_v2.py
        echo ""
        python3 extract_interface_contracts.py
        echo ""
        python3 create_shims_safe.py  # Pas de --apply
        echo ""
        python3 generate_priority_report_v2.py
        ;;

    4)
        echo "⚠️  ATTENTION : ÉCRITURE RÉELLE"
        read -p "Continuer ? (yes/no) : " CONFIRM
        if [ "$CONFIRM" != "yes" ]; then
            echo "❌ Annulé"
            exit 0
        fi

        echo ""
        echo "=== PHASE 1 : DIAGNOSTIC ==="
        python3 comprehensive_diagnostic_v2.py
        echo ""

        echo "=== PHASE 2 : CONTRATS ==="
        python3 extract_interface_contracts.py
        echo ""

        echo "=== PHASE 3 : SHIMS (AVEC --apply) ==="
        python3 create_shims_safe.py --apply --shims-dir
        echo ""

        echo "=== PHASE 4 : RAPPORT ==="
        python3 generate_priority_report_v2.py
        echo ""

        echo "=== PHASE 5 : VALIDATION ==="
        bash validate_strict.sh
        ;;

    5)
        echo "=== VALIDATION ==="
        bash validate_strict.sh
        ;;

    *)
        echo "❌ Choix invalide"
        exit 1
        ;;
esac

echo ""
echo "=================================="
echo "✅ TERMINÉ"
echo "=================================="
echo ""
echo "📊 Consultez les rapports :"
echo "   • COMPREHENSIVE_DIAGNOSTIC_V2.json"
echo "   • PRIORITIZATION_REPORT_V2.md (workflow détaillé)"
echo "   • interface_contracts/ (contrats par module)"
echo ""
echo "💡 Prochaines étapes :"
echo "   1. Lire PRIORITIZATION_REPORT_V2.md"
echo "   2. Suivre la boucle de restauration unitaire"
echo "   3. Traiter le TOP 20 un par un"
