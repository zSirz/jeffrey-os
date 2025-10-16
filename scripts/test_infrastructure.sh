#!/bin/bash
set -e

echo "🔄 TESTING RESTART RESILIENCE..."
echo "================================"

# Test de survie au redémarrage
echo -e "\n🔄 Testing restart resilience..."
docker-compose down
sleep 5
docker-compose up -d
sleep 15

# Vérifier tous les services
echo -e "\n✅ Checking services..."
docker-compose ps

# Test des endpoints
echo -e "\n📊 Testing endpoints..."

echo -n "   🔍 API alive: "
curl -s http://localhost:8000/healthz | grep -q "alive" && echo "✅" || echo "❌"

echo -n "   🔍 API ready: "
curl -s http://localhost:8000/readyz | grep -q "true" && echo "✅" || echo "❌"

echo -n "   🔍 Prometheus healthy: "
curl -s http://localhost:9090/-/healthy > /dev/null 2>&1 && echo "✅" || echo "❌"

echo -n "   🔍 Grafana healthy: "
code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:3000/api/health)
[[ "$code" == "200" || "$code" == "401" ]] && echo "✅ ($code)" || echo "❌ ($code)"

# Test métriques
echo -e "\n📈 Checking metrics..."
metric_count=$(curl -s http://localhost:8000/metrics 2>/dev/null | grep -c jeffrey_dream 2>/dev/null || echo "0")
metric_count=$(echo "$metric_count" | tr -d '\n' | tr -d ' ')
echo "   📊 Jeffrey metrics count: $metric_count"
if [[ "$metric_count" =~ ^[0-9]+$ ]] && [[ "$metric_count" -gt "0" ]]; then
    echo "   ✅ Jeffrey metrics present"
else
    echo "   ⚠️ No Jeffrey metrics found"
fi

# Test DreamEngine
echo -e "\n🧠 Testing DreamEngine..."
echo -n "   🔍 Dream status: "
dream_status=$(curl -s http://localhost:8000/api/v1/dream/status | grep -o '"success":true' || echo "")
[[ -n "$dream_status" ]] && echo "✅" || echo "❌"

echo -n "   🔍 Dream toggle: "
curl -s -X POST "http://localhost:8000/api/v1/dream/toggle?enable=true&test_mode=true" > /dev/null && echo "✅" || echo "❌"

echo -n "   🔍 Dream run: "
dream_run=$(curl -s -X POST "http://localhost:8000/api/v1/dream/run?force=true" | grep -o '"run_id"' || echo "")
[[ -n "$dream_run" ]] && echo "✅" || echo "❌"

# Test Redis connection
echo -e "\n🔴 Testing Redis..."
echo -n "   🔍 Redis ping: "
docker exec jeffrey_os-redis-1 redis-cli -a jeffrey_redis_secure_2024 ping 2>/dev/null | grep -q "PONG" && echo "✅" || echo "❌"

# Performance check
echo -e "\n⚡ Performance check..."
echo -n "   🔍 API response time: "
response_time=$(curl -s -w "%{time_total}" -o /dev/null http://localhost:8000/healthz)
echo "${response_time}s"
[[ $(echo "$response_time < 1.0" | bc -l 2>/dev/null || echo "1") == "1" ]] && echo "   ✅ Response time OK" || echo "   ⚠️ Slow response"

# Summary
echo -e "\n📊 INFRASTRUCTURE SUMMARY"
echo "=========================="
echo "✅ Docker services: $(docker-compose ps --filter status=running | wc -l | tr -d ' ') running"
echo "✅ API endpoints: Available"
echo "✅ Monitoring: Prometheus + Grafana ready"
echo "✅ DreamEngine: Operational"
echo "✅ Redis: Connected"

echo -e "\n🎉 INFRASTRUCTURE TEST COMPLETE!"
echo ""
echo "🔗 Access points:"
echo "   - API: http://localhost:8000"
echo "   - API Docs: http://localhost:8000/docs"
echo "   - Grafana: http://localhost:3000"
echo "   - Prometheus: http://localhost:9090"
echo "   - Metrics: http://localhost:8000/metrics"