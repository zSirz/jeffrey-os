#!/bin/bash
# Backup avec rotation automatique

set -euo pipefail

# Charger les variables depuis .env.p2
if [ -f .env.p2 ]; then
    export $(grep -E '^(POSTGRES_USER|POSTGRES_DB)=' .env.p2 | xargs)
fi

# Configuration
BACKUP_DIR="backups"
PG_BACKUP_DIR="$BACKUP_DIR/postgres"
REDIS_BACKUP_DIR="$BACKUP_DIR/redis"
RETENTION_DAYS=7
POSTGRES_USER=${POSTGRES_USER:-jeffrey}
POSTGRES_DB=${POSTGRES_DB:-jeffrey_p2}

# Créer les dossiers
mkdir -p "$PG_BACKUP_DIR" "$REDIS_BACKUP_DIR"

# Timestamp
TS=$(date +%Y%m%d_%H%M%S)

# Backup PostgreSQL
echo "📦 Backup PostgreSQL..."
docker exec jeffrey-postgres-p2 sh -c \
  "PGPASSWORD=\$(cat /run/secrets/postgres_password) pg_dump -U $POSTGRES_USER $POSTGRES_DB" \
  | gzip > "$PG_BACKUP_DIR/pg_${TS}.sql.gz"

# Backup Redis
echo "📦 Backup Redis..."
docker exec jeffrey-redis-p2 sh -c \
  "redis-cli -a \$(cat /run/secrets/redis_password) --rdb /tmp/redis_${TS}.rdb"
docker cp jeffrey-redis-p2:/tmp/redis_${TS}.rdb "$REDIS_BACKUP_DIR/" 2>/dev/null || true

# Rotation
echo "🔄 Rotation des backups..."
find "$PG_BACKUP_DIR" -name "pg_*.sql.gz" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
find "$REDIS_BACKUP_DIR" -name "redis_*.rdb" -mtime +$RETENTION_DAYS -delete 2>/dev/null || true

echo "✅ Backup terminé"
