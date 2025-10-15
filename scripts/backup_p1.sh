#!/bin/bash
# Backup complet et immuable de P1

set -e  # Exit on error

echo "💾 Creating P1 backup..."

# Variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/p1_backup_${TIMESTAMP}"

# 1. Git tag et branch
git add -A
git commit -m "P1 Final State - Pre-migration to P2" --no-verify || true
git tag -a "p1-final-${TIMESTAMP}" -m "P1 Final state before P2 migration"
git checkout -b "p1-backup-${TIMESTAMP}"

# 2. Archive physique
mkdir -p ${BACKUP_DIR}
tar -czf "${BACKUP_DIR}/jeffrey_p1_complete.tar.gz" \
    --exclude='venv*' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    src/ tests/ docs/ requirements.txt README.md

# 3. Export des dépendances
pip freeze > "${BACKUP_DIR}/requirements_p1_frozen.txt"

# 4. Métadonnées
cat > "${BACKUP_DIR}/backup_metadata.json" << EOF
{
    "timestamp": "${TIMESTAMP}",
    "git_commit": "$(git rev-parse HEAD)",
    "git_branch": "$(git branch --show-current)",
    "python_version": "$(python --version)",
    "backup_size": "$(du -sh ${BACKUP_DIR}/jeffrey_p1_complete.tar.gz | cut -f1)",
    "modules_count": $(find src -name "*.py" | wc -l)
}
EOF

echo "✅ Backup created in ${BACKUP_DIR}"
echo "✅ Git tag: p1-final-${TIMESTAMP}"
