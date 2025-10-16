#!/bin/bash
set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKUP_DIR="${PROJECT_ROOT}/backups/postgres"
RETENTION_DAYS=7
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
DATE=$(date +"%Y-%m-%d")

# Créer le répertoire si nécessaire
mkdir -p "$BACKUP_DIR"

echo "🔒 Starting backup for Jeffrey OS..."
echo "   Date: $DATE"
echo "   Target: $BACKUP_DIR"

# Faire le backup
cd "$PROJECT_ROOT"
if docker-compose exec -T postgres pg_dump -U jeffrey jeffrey_brain | gzip > "$BACKUP_DIR/jeffrey_${TIMESTAMP}.sql.gz"; then
    SIZE=$(ls -lh "$BACKUP_DIR/jeffrey_${TIMESTAMP}.sql.gz" | awk '{print $5}')
    echo "✅ Backup successful: jeffrey_${TIMESTAMP}.sql.gz ($SIZE)"

    # Compter les mémoires pour le log
    COUNT=$(docker-compose exec -T postgres psql -U jeffrey -d jeffrey_brain -t -c 'SELECT COUNT(*) FROM memories;' | tr -d ' ')
    echo "   Memories backed up: $COUNT"
else
    echo "❌ Backup failed!"
    exit 1
fi

# Nettoyer les vieux backups
echo "🧹 Cleaning backups older than $RETENTION_DAYS days..."
find "$BACKUP_DIR" -name "jeffrey_*.sql.gz" -mtime +$RETENTION_DAYS -delete

# Lister les backups restants
echo "📦 Current backups:"
ls -lht "$BACKUP_DIR" | head -n 10

echo "✅ Backup process complete!"