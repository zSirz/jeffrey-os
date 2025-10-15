#!/usr/bin/env bash

# === GARDE-FOUS (GPT/Marc) ===
set -euo pipefail
trap 'echo "❌ Échec à la ligne $LINENO"' ERR

[ -d .git ] && [ -d src/jeffrey ] || { echo "❌ Lance depuis la racine du repo"; exit 1; }

# Mode restauration (1 = actif, 0 = désactivé)
RESTORE_MODE=${RESTORE_MODE:-0}

# Couleurs
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Grep robuste (ignore venv & .git) - GPT/Marc
GREP_COMMON="grep -R --exclude-dir=.git --exclude-dir=.venv --exclude-dir=venv --exclude-dir=__pycache__"

echo "🔒 VALIDATION STRICTE JEFFREY OS"
if [ "$RESTORE_MODE" = "1" ]; then
    echo "   Mode: RESTAURATION (warnings autorisés)"
else
    echo "   Mode: PRODUCTION (strict)"
fi
echo "=================================="
echo ""

ERRORS=0

# 1. Mode strict
echo "1️⃣  Vérification mode strict..."
if $GREP_COMMON -nE "JEFFREY_ALLOW_FALLBACK.*=['\"]1['\"]" src/ services/ core/ 2>/dev/null | grep -v "!=" | grep -v "!=="; then
    echo -e "${RED}❌ Mode fallback activé !${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}   ✅ Mode strict OK${NC}"
fi

# 2. NotImplementedError allow-list
echo "2️⃣  Vérification NotImplementedError..."
ALLOW_DIRS=("experimental/" "_archive/" "archive/" "backup/")
ALLOW_FILES=(
    "src/jeffrey/core/learning/kg/__init__.py"
    "src/jeffrey/core/loops/base.py"
    "src/jeffrey/services/providers/provider_manager.py"
    "src/jeffrey/services/voice/adapters/voice_emotion_renderer.py"
)

NOT_IMPL_FILES=()
while IFS= read -r file; do
    ALLOWED=false

    for allow_dir in "${ALLOW_DIRS[@]}"; do
        if [[ "$file" == *"$allow_dir"* ]]; then
            ALLOWED=true
            break
        fi
    done

    if [ "$ALLOWED" = false ]; then
        for allow_file in "${ALLOW_FILES[@]}"; do
            if [[ "$file" == "$allow_file" ]]; then
                ALLOWED=true
                break
            fi
        done
    fi

    if [ "$ALLOWED" = false ]; then
        NOT_IMPL_FILES+=("$file")
    fi
done < <(find services/ src/ core/ -name "*.py" 2>/dev/null | xargs grep -l "NotImplementedError" 2>/dev/null || true)

if [ ${#NOT_IMPL_FILES[@]} -gt 0 ]; then
    echo -e "${RED}❌ ${#NOT_IMPL_FILES[@]} fichiers avec NotImplementedError hors allow-list${NC}"
    for file in "${NOT_IMPL_FILES[@]}"; do
        echo "   $file"
    done
    ERRORS=$((ERRORS + 1))
else
    echo -e "${GREEN}   ✅ Aucun NotImplementedError hors allow-list${NC}"
fi

# 3. Stubs
echo "3️⃣  Vérification stubs..."
if [ -f "COMPREHENSIVE_DIAGNOSTIC_V2.json" ]; then
    STUB_COUNT=$(python3 -c "import json; print(len(json.load(open('COMPREHENSIVE_DIAGNOSTIC_V2.json'))['stubs_with_scores']))" 2>/dev/null || echo "0")
    if [ "$STUB_COUNT" -gt "0" ]; then
        if [ "$RESTORE_MODE" = "1" ]; then
            echo -e "${YELLOW}⚠️  $STUB_COUNT stubs détectés (mode restauration)${NC}"
        else
            echo -e "${RED}❌ $STUB_COUNT stubs détectés${NC}"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo -e "${GREEN}   ✅ Aucun stub${NC}"
    fi
else
    echo -e "${YELLOW}   ⏭️  Diagnostic non trouvé${NC}"
fi

# 4. Imports cassés
echo "4️⃣  Vérification imports..."
if [ -f "COMPREHENSIVE_DIAGNOSTIC_V2.json" ]; then
    BROKEN=$(python3 -c "import json; print(json.load(open('COMPREHENSIVE_DIAGNOSTIC_V2.json'))['stats']['runtime_broken'])" 2>/dev/null || echo "0")
    if [ "$BROKEN" -gt "0" ]; then
        if [ "$RESTORE_MODE" = "1" ]; then
            echo -e "${YELLOW}⚠️  $BROKEN imports cassés (mode restauration)${NC}"
        else
            echo -e "${RED}❌ $BROKEN imports cassés${NC}"
            ERRORS=$((ERRORS + 1))
        fi
    else
        echo -e "${GREEN}   ✅ Tous imports OK${NC}"
    fi
else
    echo -e "${YELLOW}   ⏭️  Diagnostic non trouvé${NC}"
fi

# 5. Vérifier que RESTORE_MODE pas laissé en prod (GPT/Marc)
if [ "$RESTORE_MODE" = "0" ]; then
    if grep -q "RESTORE_MODE.*:-1" "$0" 2>/dev/null; then
        echo -e "${RED}❌ RESTORE_MODE par défaut non strict dans le script${NC}"
        ERRORS=$((ERRORS + 1))
    fi
fi

# Résultat final
echo ""
echo "=================================="
if [ $ERRORS -gt 0 ]; then
    echo -e "${RED}❌ $ERRORS problème(s) détecté(s)${NC}"
    exit 1
else
    echo -e "${GREEN}✅ Validation OK${NC}"
    exit 0
fi
