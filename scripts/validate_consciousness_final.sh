#!/bin/bash
echo "🔧 FINAL CONSCIOUSNESS VALIDATION"
echo "=================================="

API_KEY=$(grep JEFFREY_API_KEY .env | cut -d= -f2)
BASE_URL="http://localhost:8000"

# Test 1: Trigger returns 202
echo -e "\n1️⃣ Testing trigger endpoint (should return 202)..."
RESPONSE=$(curl -s -w "\n%{http_code}" -X POST \
  "${BASE_URL}/api/v1/consciousness/trigger" \
  -H "X-API-Key: $API_KEY")
  
STATUS=$(echo "$RESPONSE" | tail -1)
if [ "$STATUS" = "202" ]; then
  echo "✅ Trigger returns 202 Accepted"
  TASK_ID=$(echo "$RESPONSE" | head -n -1 | jq -r '.task_id')
  echo "   Task ID: $TASK_ID"
else
  echo "❌ Wrong status: $STATUS (expected 202)"
fi

# Test 2: Rate limiting on /bonds
echo -e "\n2️⃣ Testing rate limiting on /bonds..."
for i in {1..3}; do
  curl -s -o /dev/null -w "%{http_code}\n" \
    "${BASE_URL}/api/v1/bonds?limit=1" \
    -H "X-API-Key: $API_KEY"
done | grep -q "429" && echo "✅ Rate limiting works" || echo "⚠️ Rate limiting may not be active"

# Test 3: Métriques après trigger
echo -e "\n3️⃣ Checking metrics after trigger..."
sleep 5
curl -s "${BASE_URL}/metrics" | grep -E "jeffrey_bonds_active|jeffrey_consciousness_cycles_total" | head -5

# Test 4: Run unit tests
echo -e "\n4️⃣ Running improved unit tests..."
docker-compose exec -T jeffrey-api pytest tests/test_bonds_service.py -v

echo -e "\n✅ Final validation complete!"