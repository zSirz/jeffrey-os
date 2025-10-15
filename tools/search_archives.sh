#!/usr/bin/env bash

SEARCH_PATTERN="$1"
shift
ARCHIVE_DIRS=("$@")

echo "Recherche de : $SEARCH_PATTERN"
echo "Dans les archives :"
for dir in "${ARCHIVE_DIRS[@]}"; do
    echo "  - $dir"
done
echo ""

for archive in "${ARCHIVE_DIRS[@]}"; do
    if [ ! -d "$archive" ]; then
        echo "⚠️  Archive non trouvée : $archive"
        continue
    fi

    echo "🔍 Scan de $archive..."
    find "$archive" -type f -name "*${SEARCH_PATTERN}*" -print 2>/dev/null | while read -r file; do
        echo "  ✓ Trouvé : $file"
        echo "    Taille : $(du -h "$file" | cut -f1)"
        echo "    Date : $(stat -f "%Sm" "$file")"
    done
done
