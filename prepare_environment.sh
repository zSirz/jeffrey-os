#!/usr/bin/env bash
# Fichier: prepare_environment.sh
set -euo pipefail  # Mode strict pour bash

echo "🔧 PRÉPARATION ENVIRONNEMENT JEFFREY OS"
echo "======================================="

# Backup classique
cd /Users/davidproz/Desktop/Jeffrey_OS
tar -czf backup_jeffrey_$(date +%Y%m%d_%H%M%S).tar.gz src/ data/ 2>/dev/null || true
echo "✅ Backup tar créé"

# Git snapshot si disponible
if [ -d .git ]; then
    git add -A
    git stash push -m "Pre-reconstruction snapshot $(date +%Y%m%d_%H%M%S)"
    echo "✅ Git snapshot sauvegardé"
else
    git init
    git add -A
    git commit -m "Initial state before reconstruction"
    echo "✅ Git initialisé et état sauvegardé"
fi

# Créer structure de dossiers
mkdir -p diagnostics logs data/{audit,sandbox,context,memory}
mkdir -p src/jeffrey/{api,core/{memory,orchestration,diagnostics},services,utilities}

# Environnement virtuel et dépendances (AVANT les diagnostics)
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --quiet --upgrade pip setuptools wheel

# Créer un pyproject.toml minimal pour les diagnostics
if [ ! -f "pyproject.toml" ]; then
    cat > pyproject.toml <<'EOF'
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "jeffrey"
version = "0.1.0"
requires-python = ">=3.10"
dependencies = [
    "aiofiles>=23.0",
    "httpx>=0.24",
    "pydantic>=2.0",
    "rich>=13.0",
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
]

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.packages.find]
where = ["src"]
EOF
fi

pip install --quiet -e .
echo "✅ Environnement prêt"
