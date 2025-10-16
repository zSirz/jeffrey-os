#!/bin/bash
set -e

echo "🔍 JEFFREY OS - VALIDATION COMPLÈTE"
echo "===================================="

# Fonction de check
check() {
    if [ $? -eq 0 ]; then
        echo "✅ $1"
    else
        echo "❌ $1 FAILED"
        exit 1
    fi
}

# 1. Services health
echo -e "\n📊 Checking services..."
curl -s http://localhost:8000/healthz | grep -q "alive"
check "API Health"

curl -s http://localhost:8000/readyz | grep -q "true"
check "API Ready"

# 2. Database
echo -e "\n🗄️ Checking database..."
docker-compose exec postgres pg_isready -U jeffrey -d jeffrey_brain > /dev/null 2>&1
check "PostgreSQL connection"

MEMORY_COUNT=$(curl -s "http://localhost:8000/api/v1/memories/recent?hours=24" | jq '. | length')
echo "   📚 Memories in DB: $MEMORY_COUNT"

# 3. Metrics
echo -e "\n📈 Checking metrics..."
METRIC_COUNT=$(curl -s http://localhost:8000/metrics | grep -c jeffrey_ || true)
echo "   🎯 Jeffrey metrics active: $METRIC_COUNT"

# 4. ML Model (test with feature flag disabled)
echo -e "\n🧠 Testing emotion detection..."
EMOTION=$(echo '{"text":"I am happy about these test results!"}' | curl -s -X POST http://localhost:8000/api/v1/emotion/detect -H "Content-Type: application/json" -d @- | jq -r '.emotion')
[ "$EMOTION" != "null" ] && [ "$EMOTION" != "" ]
check "Emotion detection (detected: $EMOTION)"

# 5. Run tests
echo -e "\n🧪 Running automated tests..."
docker-compose exec jeffrey-api pytest tests/test_api_integration.py -v --tb=short > /dev/null 2>&1
check "API integration tests passed"

# 6. Check test database
echo -e "\n🧪 Testing database operations..."
docker-compose exec postgres psql -U jeffrey -d jeffrey_brain_test -c "SELECT COUNT(*) FROM memories;" > /dev/null 2>&1
check "Test database accessible"

echo -e "\n🎉 SYSTEM VALIDATION COMPLETE!"
echo "================================"
echo "✅ All systems operational"
echo "📊 Metrics: $METRIC_COUNT active"
echo "💾 Memories: $MEMORY_COUNT stored"
echo "🧠 Emotion Detection: Working with keyword fallback"
echo "🧪 Tests: Passing"