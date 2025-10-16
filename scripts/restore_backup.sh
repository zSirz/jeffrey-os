#!/bin/bash
set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    echo "Available backups:"
    ls -lh backups/postgres/*.sql.gz
    exit 1
fi

BACKUP_FILE="$1"
if [ ! -f "$BACKUP_FILE" ]; then
    echo "❌ Backup file not found: $BACKUP_FILE"
    exit 1
fi

echo "⚠️  WARNING: This will restore the database from: $BACKUP_FILE"
echo "All current data will be LOST!"
read -p "Are you sure? (type 'yes' to continue): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

echo "🔄 Restoring database..."
gunzip -c "$BACKUP_FILE" | docker-compose exec -T postgres psql -v ON_ERROR_STOP=1 -U jeffrey -d jeffrey_brain

echo "✅ Database restored successfully!"