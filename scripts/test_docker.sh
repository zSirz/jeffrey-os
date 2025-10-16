#!/bin/bash
set -e

echo "🔍 TEST RAPIDE DOCKER"
echo "===================="

echo -e "\n[1] Test healthz..."
curl -sf http://localhost:8000/healthz && echo " ✅ OK" || echo " ❌ FAILED"

echo -e "\n[2] Test readyz..."
READYZ=$(curl -s http://localhost:8000/readyz)
echo "Response: $READYZ"
echo "$READYZ" | python3 -m json.tool > /dev/null 2>&1 && echo " ✅ Valid JSON" || echo " ⚠️ Invalid JSON"

echo -e "\n[3] Toggle dream engine..."
curl -sf "http://localhost:8000/api/v1/dream/toggle?enable=true&test_mode=true" && echo " ✅ Enabled"

echo -e "\n[4] Run dream consolidation..."
DREAM_RUN=$(curl -sf -X POST "http://localhost:8000/api/v1/dream/run?force=true")
echo "Response: $DREAM_RUN"

echo -e "\n[5] Check dream status..."
curl -s http://localhost:8000/api/v1/dream/status | python3 -m json.tool | head -20

echo -e "\n[6] Check services..."
docker-compose ps

echo -e "\n✅ Test terminé!"