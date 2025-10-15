#!/bin/bash
# Script de réveil iCloud pour forcer le téléchargement des fichiers Python
# Évite les fichiers "placeholder" non téléchargés

echo "🌤️  Réveil iCloud Drive en cours..."

# Vérifier que iCloud Drive existe
ICLOUD_PATH="$HOME/Library/Mobile Documents/com~apple~CloudDocs"
if [ ! -d "$ICLOUD_PATH" ]; then
    echo "❌ iCloud Drive non trouvé à $ICLOUD_PATH"
    exit 1
fi

echo "📁 Scan de $ICLOUD_PATH..."

# Forcer le téléchargement de tous les fichiers .py
# Utilise find avec xargs pour traiter par lots et éviter les timeouts
find "$ICLOUD_PATH" -type f -name "*.py" -print0 | xargs -0 -n 50 cat >/dev/null 2>&1 || true

# Compter les fichiers traités
PY_COUNT=$(find "$ICLOUD_PATH" -type f -name "*.py" | wc -l)
echo "✅ $PY_COUNT fichiers Python traités"

# Vérification rapide de quelques fichiers critiques
echo "🔍 Vérification de quelques fichiers critiques..."
CRITICAL_FILES=(
    "jeffrey_os_v1.0"
    "JEFFREY_ARCHIVES"
    "jeffrey"
)

for pattern in "${CRITICAL_FILES[@]}"; do
    FOUND=$(find "$ICLOUD_PATH" -name "*$pattern*" -type d | head -1)
    if [ -n "$FOUND" ]; then
        echo "   ✅ Trouvé: $FOUND"
    else
        echo "   ⚠️  Non trouvé: *$pattern*"
    fi
done

echo "🌤️  iCloud Drive réveillé et prêt pour l'inventaire !"
echo ""
echo "🚀 Tu peux maintenant lancer l'inventaire :"
echo "   export INVENTORY_TAG=\"claude\""
echo "   python3 inventory_exhaustif_jeffrey.py"
