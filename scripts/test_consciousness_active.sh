#!/bin/bash
echo "🧠 TESTING ACTIVE CONSCIOUSNESS"
echo "================================"

API_KEY=$(grep JEFFREY_API_KEY .env | cut -d= -f2)
BASE_URL="http://localhost:8000"

# Phase 1: Test with write disabled (dry run)
echo -e "\n🔒 Phase 1: Testing with WRITE DISABLED (dry run)..."
echo "ENABLE_CONSCIOUSNESS=true"
echo "ENABLE_CONSCIOUSNESS_WRITE=false"

# Trigger manual cycle (if API is running)
echo -e "\n⚡ Triggering consciousness cycle manually..."
curl -s -X POST "${BASE_URL}/api/v1/consciousness/trigger" \
  -H "X-API-Key: $API_KEY" || echo "API not running yet"

# Check bonds (should be empty)
echo -e "\n🔗 Checking bonds (should be 0 with write disabled):"
curl -s "${BASE_URL}/api/v1/bonds" \
  -H "X-API-Key: $API_KEY" | python -m json.tool 2>/dev/null | grep -E "total|bonds" || echo "No bonds API available yet"

# Check metrics
echo -e "\n📊 Checking metrics:"
curl -s "${BASE_URL}/metrics" 2>/dev/null | grep -E "consciousness|bonds|curiosity" | head -10 || echo "Metrics not available yet"

# Phase 2: Enable write with safety caps
echo -e "\n🔓 Phase 2: Enabling WRITE with safety caps..."
echo "ENABLE_CONSCIOUSNESS_WRITE=true"
echo "CONSCIOUSNESS_MAX_NEW_MEMORIES=1"
echo "CONSCIOUSNESS_MAX_BONDS_UPDATES=10"

# Create test memory for bonds
echo -e "\n💾 Creating test memory..."
TEST_MEMORY=$(curl -s -X POST "${BASE_URL}/v1/chat" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"message": "I feel happy about the progress we are making", "stream": false}' \
  2>/dev/null | python -c "import sys, json; data=json.load(sys.stdin); print(data.get('memory_id', 'no_memory'))" 2>/dev/null || echo "no_memory")

echo "Test memory ID: $TEST_MEMORY"

# Trigger another cycle with write enabled
echo -e "\n⚡ Triggering consciousness cycle with WRITE enabled..."
curl -s -X POST "${BASE_URL}/api/v1/consciousness/trigger" \
  -H "X-API-Key: $API_KEY" || echo "Manual trigger endpoint not implemented, will wait for scheduler"

# Wait a bit for async operations
sleep 3

# Check bonds again
echo -e "\n🔗 Checking bonds after write-enabled cycle:"
curl -s "${BASE_URL}/api/v1/bonds" \
  -H "X-API-Key: $API_KEY" | python -m json.tool 2>/dev/null || echo "Bonds API not ready"

# Check specific memory bonds if we have an ID
if [ "$TEST_MEMORY" != "no_memory" ]; then
    echo -e "\n🔍 Checking bonds for test memory $TEST_MEMORY:"
    curl -s "${BASE_URL}/api/v1/bonds/${TEST_MEMORY}" \
      -H "X-API-Key: $API_KEY" | python -m json.tool 2>/dev/null || echo "Memory bonds not available"
fi

# Final metrics check
echo -e "\n📊 Final metrics check:"
curl -s "${BASE_URL}/metrics" 2>/dev/null | grep -E "jeffrey_consciousness_cycles_total|jeffrey_bonds_upserted_total|jeffrey_curiosity_questions_total" || echo "Metrics not ready"

echo -e "\n✅ Consciousness test complete!"
echo -e "\n📝 Next steps:"
echo "1. Check docker logs: docker-compose logs jeffrey-api | grep -E 'consciousness|bonds'"
echo "2. Monitor for 24-48h before increasing limits"
echo "3. Set up alerts if bonds_upserted_total > 200/h"