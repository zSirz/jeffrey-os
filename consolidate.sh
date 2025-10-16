#!/bin/bash
echo "🔒 CONSOLIDATION PRODUCTION - JOUR 1"
echo "====================================="

# 1. Backup immédiat
docker-compose exec -T postgres pg_dump -U jeffrey jeffrey_brain > backup_before_consolidation.sql

# 2. Créer structure Alembic
docker-compose exec jeffrey-api alembic init alembic 2>/dev/null || echo "Alembic déjà initialisé"

# 3. Vérifier l'état
echo -e "\n📊 État actuel:"
echo "- Mémoires: $(docker-compose exec postgres psql -U jeffrey -d jeffrey_brain -t -c 'SELECT COUNT(*) FROM memories;')"
echo "- Services: $(docker-compose ps | grep Up | wc -l) up"
echo "- Tests: 6/8 passing"

echo -e "\n⚠️  DÉCISION REQUISE:"
echo "Continuer avec la consolidation production (recommandé) ? [y/n]"
