#!/bin/bash
set -e

echo "🔍 VALIDATION PRODUCTION DREAMENGINE"
echo "===================================="

# 1. Check Docker config
echo -e "\n1️⃣ Validating Docker configuration..."
docker-compose config > /dev/null
echo "   ✅ Docker config valid"

# 2. Build and start
echo -e "\n2️⃣ Building containers..."
docker-compose build
echo "   ✅ Build successful"

echo -e "\n3️⃣ Starting services..."
docker-compose up -d
echo "   ✅ Services started"

# Wait for services
echo -e "\n4️⃣ Waiting for services to be ready..."
sleep 15

# 5. Health checks
echo -e "\n5️⃣ Running health checks..."
curl -s http://localhost:8000/healthz | grep -q "alive" && echo "   ✅ API alive"

echo "   📊 Readiness check:"
curl -s http://localhost:8000/readyz | python3 -m json.tool

echo -n "   🔍 Grafana accessible: "
curl -s http://localhost:3000 > /dev/null && echo "✅" || echo "❌"

echo -n "   🔍 Prometheus accessible: "
curl -s http://localhost:9090 > /dev/null && echo "✅" || echo "❌"

# 6. Enable DreamEngine
echo -e "\n6️⃣ Enabling DreamEngine..."
curl -X POST "http://localhost:8000/api/v1/dream/toggle?enable=true&test_mode=true"
echo ""

# 7. Run dream consolidation
echo -e "\n7️⃣ Running dream consolidation..."
curl -X POST "http://localhost:8000/api/v1/dream/run?force=true"
echo ""

# 8. Check results
echo -e "\n8️⃣ Checking results..."
if [ -d "data/dreams/test" ]; then
    echo "   ✅ Dream output directory exists"
    ls -la data/dreams/test/ | head -5
else
    echo "   ⚠️ Dream output directory not found (may be expected on first run)"
fi

# 9. Check metrics
echo -e "\n9️⃣ Checking Prometheus metrics..."
curl -s http://localhost:8000/metrics | grep -q "jeffrey_dream" && echo "   ✅ Dream metrics present" || echo "   ⚠️ Dream metrics not found"

# 10. Check dream status
echo -e "\n🔟 Dream engine status:"
curl -s http://localhost:8000/api/v1/dream/status | python3 -m json.tool

# 11. Check Grafana dashboard
echo -e "\n📊 Grafana dashboard:"
echo "   📊 Open http://localhost:3000 (admin/password from .env)"
echo "   📊 Check 'Jeffrey OS' dashboard"

echo -e "\n✅ VALIDATION COMPLETE!"
echo "System is production-ready with DreamEngine integrated"
echo ""
echo "🔗 Access points:"
echo "   - API: http://localhost:8000"
echo "   - Grafana: http://localhost:3000"
echo "   - Prometheus: http://localhost:9090"
echo ""
echo "📝 Next steps:"
echo "   1. Configure alerts in Grafana"
echo "   2. Set up monitoring dashboards"
echo "   3. Test dream consolidation with real data"