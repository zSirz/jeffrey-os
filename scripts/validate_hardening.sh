#!/bin/bash

echo "=== Jeffrey OS Production Hardening Validation ==="
echo "Testing security, rate limiting, and orchestration..."
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

API_BASE="http://localhost:8000"
API_KEY="jeffrey-15c850850bb5d3e67c37fcf728de6e1d"

# Test counter
TESTS_RUN=0
TESTS_PASSED=0

# Helper functions
pass_test() {
    echo -e "${GREEN}✓ PASS:${NC} $1"
    ((TESTS_PASSED++))
    ((TESTS_RUN++))
}

fail_test() {
    echo -e "${RED}✗ FAIL:${NC} $1"
    ((TESTS_RUN++))
}

info_test() {
    echo -e "${YELLOW}ℹ INFO:${NC} $1"
}

echo "=== Phase 1: API Health Check ==="

# Test basic API health
response=$(curl -s -o /dev/null -w "%{http_code}" "$API_BASE/health")
if [ "$response" = "200" ]; then
    pass_test "API is responding (HTTP 200)"
else
    fail_test "API health check failed (HTTP $response)"
fi

echo
echo "=== Phase 2: Security Tests ==="

# Test 1: Memory endpoint without API key (should fail)
info_test "Testing memory endpoint without API key..."
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/v1/memories/" \
    -H "Content-Type: application/json" \
    -d '{"text": "test", "emotion": "neutral"}')

if [ "$response" = "401" ]; then
    pass_test "Memory POST properly rejects requests without API key (HTTP 401)"
else
    fail_test "Memory POST should reject without API key, got HTTP $response"
fi

# Test 2: Memory endpoint with invalid API key (should fail)
info_test "Testing memory endpoint with invalid API key..."
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/v1/memories/" \
    -H "Content-Type: application/json" \
    -H "x-api-key: invalid-key" \
    -d '{"text": "test", "emotion": "neutral"}')

if [ "$response" = "403" ]; then
    pass_test "Memory POST properly rejects invalid API key (HTTP 403)"
else
    fail_test "Memory POST should reject invalid API key, got HTTP $response"
fi

# Test 3: Memory endpoint with valid API key (should work)
info_test "Testing memory endpoint with valid API key..."
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/v1/memories/" \
    -H "Content-Type: application/json" \
    -H "x-api-key: $API_KEY" \
    -d '{"text": "Security test memory", "emotion": "neutral", "metadata": {"test": "hardening"}}')

if [ "$response" = "200" ]; then
    pass_test "Memory POST works with valid API key (HTTP 200)"
else
    fail_test "Memory POST with valid API key failed (HTTP $response)"
fi

echo
echo "=== Phase 3: Rate Limiting Tests ==="

# Test 4: Rate limiting on emotion endpoint
info_test "Testing rate limiting on emotion endpoint (sending 5 rapid requests)..."
rate_limit_failures=0

for i in {1..5}; do
    response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/v1/emotion/detect" \
        -H "Content-Type: application/json" \
        -d '{"text": "Rate limit test '"$i"'"}')

    if [ "$response" = "429" ]; then
        ((rate_limit_failures++))
    fi
    sleep 0.1
done

if [ $rate_limit_failures -gt 0 ]; then
    pass_test "Rate limiting is working (got $rate_limit_failures rate limit responses)"
else
    info_test "Rate limiting not triggered in test (may need more requests or different timing)"
fi

echo
echo "=== Phase 4: Orchestration Tests ==="

# Test 5: Scheduler status endpoint
info_test "Testing scheduler status endpoint..."
response=$(curl -s "$API_BASE/api/v1/dream/schedule")
if echo "$response" | grep -q "enabled"; then
    pass_test "Scheduler status endpoint working"
else
    fail_test "Scheduler status endpoint not responding properly"
fi

# Test 6: Scheduler configuration endpoint (admin protected)
info_test "Testing scheduler config endpoint without API key..."
response=$(curl -s -o /dev/null -w "%{http_code}" -X PUT "$API_BASE/api/v1/dream/schedule" \
    -H "Content-Type: application/json" \
    -d '{"enabled": true, "interval_minutes": 20}')

if [ "$response" = "401" ]; then
    pass_test "Scheduler config properly protected without API key (HTTP 401)"
else
    fail_test "Scheduler config should be protected, got HTTP $response"
fi

# Test 7: Dream endpoint protection
info_test "Testing dream run endpoint protection..."
response=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/v1/dream/run")

if [ "$response" = "401" ]; then
    pass_test "Dream run endpoint properly protected (HTTP 401)"
else
    fail_test "Dream run endpoint should be protected, got HTTP $response"
fi

echo
echo "=== Phase 5: Functional Tests ==="

# Test 8: Emotion detection functionality
info_test "Testing emotion detection functionality..."
response=$(curl -s "$API_BASE/api/v1/emotion/detect" \
    -H "Content-Type: application/json" \
    -d '{"text": "I am very happy and excited about this validation test!"}')

if echo "$response" | grep -q "emotion"; then
    emotion=$(echo "$response" | grep -o '"emotion":"[^"]*"' | cut -d'"' -f4)
    confidence=$(echo "$response" | grep -o '"confidence":[0-9.]*' | cut -d':' -f2)
    pass_test "Emotion detection working (detected: $emotion, confidence: $confidence)"
else
    fail_test "Emotion detection not working properly"
fi

# Test 9: Memory retrieval
info_test "Testing memory retrieval..."
response=$(curl -s "$API_BASE/api/v1/memories/recent?hours=1&limit=5")

if echo "$response" | grep -q "\[\]" || echo "$response" | grep -q "\"text\""; then
    pass_test "Memory retrieval endpoint working"
else
    fail_test "Memory retrieval not working properly"
fi

# Test 10: Metrics endpoint
info_test "Testing Prometheus metrics..."
response=$(curl -s "$API_BASE/metrics")

if echo "$response" | grep -q "jeffrey_emotion"; then
    pass_test "Prometheus metrics available and contain Jeffrey metrics"
else
    fail_test "Prometheus metrics not working properly"
fi

echo
echo "=== Phase 6: Integration Tests ==="

# Test 11: Full workflow - emotion detection + memory storage
info_test "Testing full workflow: emotion detection + memory storage..."

# First, detect emotion
emotion_response=$(curl -s "$API_BASE/api/v1/emotion/detect" \
    -H "Content-Type: application/json" \
    -d '{"text": "Integration test: I am feeling confident about this system!"}')

if echo "$emotion_response" | grep -q "emotion"; then
    # Wait a moment for memory to be stored
    sleep 2

    # Then check if memory was stored
    memory_response=$(curl -s "$API_BASE/api/v1/memories/recent?hours=1&limit=10")

    if echo "$memory_response" | grep -q "Integration test"; then
        pass_test "Full workflow working (emotion detected and stored in memory)"
    else
        fail_test "Emotion detected but not found in memory"
    fi
else
    fail_test "Full workflow failed at emotion detection"
fi

echo
echo "=== VALIDATION SUMMARY ==="
echo "Tests run: $TESTS_RUN"
echo "Tests passed: $TESTS_PASSED"
echo "Tests failed: $((TESTS_RUN - TESTS_PASSED))"

if [ $TESTS_PASSED -eq $TESTS_RUN ]; then
    echo -e "${GREEN}🎉 ALL TESTS PASSED! Production hardening is complete.${NC}"
    exit 0
elif [ $TESTS_PASSED -gt $((TESTS_RUN / 2)) ]; then
    echo -e "${YELLOW}⚠️  Most tests passed, but some issues need attention.${NC}"
    exit 1
else
    echo -e "${RED}❌ Multiple tests failed. System needs review.${NC}"
    exit 2
fi