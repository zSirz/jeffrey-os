#!/bin/bash

# Script de migration prioritaire des modules Jeffrey manquants

ICLOUD_BASE="$HOME/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps"
JEFFREY_OS="$HOME/Desktop/Jeffrey_OS"

echo "🚀 MIGRATION PRIORITAIRE DES MODULES JEFFREY"
echo "============================================="

# 1. JEFFREY_UNIFIED Services
echo ""
echo "📦 Phase 1: Migration JEFFREY_UNIFIED/services..."
if [ -d "$ICLOUD_BASE/JEFFREY_UNIFIED/services" ]; then
    echo "  Copie des services orchestrateur..."
    find "$ICLOUD_BASE/JEFFREY_UNIFIED/services" -name "*.py" -type f \
        ! -path "*/venv/*" ! -path "*/__pycache__/*" | while read -r file; do
        relative_path=$(echo "$file" | sed "s|$ICLOUD_BASE/JEFFREY_UNIFIED/services/||")
        target_dir="$JEFFREY_OS/src/jeffrey/services/unified/$(dirname "$relative_path")"
        mkdir -p "$target_dir"
        cp "$file" "$target_dir/"
        echo "  ✅ $(basename "$file")"
    done
fi

# 2. Jeffrey_Phoenix PRODUCTION
echo ""
echo "📦 Phase 2: Migration Jeffrey_Phoenix/PRODUCTION..."
if [ -d "$ICLOUD_BASE/Jeffrey_Phoenix/PRODUCTION/Jeffrey_Consolidated" ]; then
    for module in consciousness emotions memory voice; do
        if [ -d "$ICLOUD_BASE/Jeffrey_Phoenix/PRODUCTION/Jeffrey_Consolidated/$module" ]; then
            echo "  Copie module $module..."
            find "$ICLOUD_BASE/Jeffrey_Phoenix/PRODUCTION/Jeffrey_Consolidated/$module" -name "*.py" -type f \
                ! -path "*/__pycache__/*" | while read -r file; do
                relative_path=$(echo "$file" | sed "s|$ICLOUD_BASE/Jeffrey_Phoenix/PRODUCTION/Jeffrey_Consolidated/$module/||")
                target_dir="$JEFFREY_OS/src/jeffrey/core/$module/phoenix/$(dirname "$relative_path")"
                mkdir -p "$target_dir"
                cp "$file" "$target_dir/"
                echo "    ✅ $(basename "$file")"
            done
        fi
    done
fi

# 3. CashZen Application
echo ""
echo "📦 Phase 3: Migration CashZen..."
if [ -d "$ICLOUD_BASE/CashZen" ]; then
    echo "  Copie application CashZen..."
    find "$ICLOUD_BASE/CashZen" -name "*.py" -type f \
        ! -path "*/venv/*" ! -path "*/__pycache__/*" \
        ! -path "*/test_*" -size +5k | while read -r file; do
        relative_path=$(echo "$file" | sed "s|$ICLOUD_BASE/CashZen/||")
        target_dir="$JEFFREY_OS/apps/cashzen/original/$(dirname "$relative_path")"
        mkdir -p "$target_dir"
        cp "$file" "$target_dir/"
        echo "  ✅ $(basename "$file")"
    done
fi

# 4. Modules spéciaux manquants
echo ""
echo "📦 Phase 4: Recherche modules critiques manquants..."
critical_modules=(
    "consciousness_engine"
    "emotion_matrix"
    "living_memory"
    "voice_engine"
    "orchestrator"
    "jeffrey_vivant"
)

for module_pattern in "${critical_modules[@]}"; do
    echo "  Recherche: $module_pattern..."
    find "$ICLOUD_BASE" -name "*${module_pattern}*.py" -type f \
        ! -path "*/venv/*" ! -path "*/__pycache__/*" \
        ! -path "*/Archives/*" ! -path "*/_backup*/*" \
        -size +10k | head -5 | while read -r file; do
        target_dir="$JEFFREY_OS/src/jeffrey/core/critical"
        mkdir -p "$target_dir"
        cp "$file" "$target_dir/$(basename "$file")"
        echo "    ✅ $(basename "$file") ($(du -h "$file" | cut -f1))"
    done
done

# Statistiques finales
echo ""
echo "📊 RÉSUMÉ DE LA MIGRATION:"
echo "=========================="
total_before=$(find "$JEFFREY_OS" -name "*.py" -type f | wc -l)
echo "✅ Fichiers Python totaux : $total_before"
echo ""
echo "🎯 PROCHAINES ÉTAPES:"
echo "1. Vérifier les imports dans les nouveaux fichiers"
echo "2. Résoudre les conflits de versions"
echo "3. Mettre à jour requirements.txt"
echo "4. Tester les modules critiques"
