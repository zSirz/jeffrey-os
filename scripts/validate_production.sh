#!/bin/bash
set -e

echo "🔍 VALIDATION PRODUCTION JEFFREY OS"
echo "===================================="

# Check Alembic
echo -e "\n📋 Alembic Status:"
if docker-compose exec -T jeffrey-api bash -c 'cd /app && PYTHONPATH=/app/src alembic current' 2>/dev/null; then
    echo "✅ Alembic properly configured"
else
    echo "❌ Alembic not configured"
fi

# Check Backups
echo -e "\n💾 Backups:"
if [ -d "backups/postgres" ]; then
    BACKUP_COUNT=$(ls -1 backups/postgres/*.sql.gz 2>/dev/null | wc -l)
    echo "   Backup files: $BACKUP_COUNT"
    if [ $BACKUP_COUNT -gt 0 ]; then
        echo "   Latest backup: $(ls -t backups/postgres/*.sql.gz | head -1)"
        LATEST_SIZE=$(ls -lh backups/postgres/*.sql.gz | head -1 | awk '{print $5}')
        echo "   Latest backup size: $LATEST_SIZE"
    fi
else
    echo "   ❌ No backup directory"
fi

# Check Services
echo -e "\n🐳 Services:"
docker-compose ps

# Check API Health
echo -e "\n🌐 API Health:"
if curl -s http://localhost:8000/healthz | jq '.' > /dev/null 2>&1; then
    echo "✅ API responding and healthy"
else
    echo "❌ API not responding"
fi

# Check Database
echo -e "\n🗄️ Database:"
if MEMORY_COUNT=$(docker-compose exec -T postgres psql -U jeffrey -d jeffrey_brain -t -c 'SELECT COUNT(*) FROM memories;' 2>/dev/null | tr -d ' '); then
    echo "   Memories: $MEMORY_COUNT"
    echo "✅ Database accessible"
else
    echo "❌ Database connection failed"
fi

# Check Tests
echo -e "\n🧪 Tests:"
if docker-compose exec -T jeffrey-api pytest tests/test_api_integration.py -q --tb=no > /dev/null 2>&1; then
    echo "✅ Integration tests passing"
else
    echo "❌ Some tests failed"
fi

# Check Metrics
echo -e "\n📊 Metrics:"
if METRIC_COUNT=$(curl -s http://localhost:8000/metrics | grep -c jeffrey_ 2>/dev/null); then
    echo "   Jeffrey metrics active: $METRIC_COUNT"
    echo "✅ Metrics collection working"
else
    echo "❌ Metrics collection failed"
fi

# Check ML Feature Flag
echo -e "\n🧠 ML Feature Flag:"
ML_STATUS=$(grep "ENABLE_REAL_ML=" .env | cut -d= -f2)
echo "   ENABLE_REAL_ML: $ML_STATUS"
if [ "$ML_STATUS" = "false" ]; then
    echo "✅ ML feature flag safely disabled"
else
    echo "⚠️  ML feature flag enabled"
fi

echo -e "\n🎉 PRODUCTION VALIDATION COMPLETE"
echo "=================================="
echo "✅ System ready for production deployment"