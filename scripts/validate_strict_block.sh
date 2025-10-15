#!/bin/bash
# Step 3: Check module count (deterministic, strict)
echo -e "\n📈 Checking module counts (deterministic)..."

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Ensure relaxed mode is OFF
unset JEFFREY_MEASURE_RELAXED
if [ "${JEFFREY_MEASURE_RELAXED:-}" = "1" ]; then
    echo -e "  ${RED}❌ JEFFREY_MEASURE_RELAXED must be off during validation${NC}"
    exit 1
fi

# Get counts
if command -v jq >/dev/null 2>&1; then
    JSON_COUNT=$(jq '.summary.total_modules' artifacts/inventory_ultimate.json)
else
    JSON_COUNT=$(python3 -c "import json; print(json.load(open('artifacts/inventory_ultimate.json'))['summary']['total_modules'])")
fi

ELIGIBLE_COUNT=$(python3 scripts/eligible_count.py)

echo "  Modules in JSON:     $JSON_COUNT"
echo "  Eligible on disk:    $ELIGIBLE_COUNT"

# Calculate difference
DIFF=$((JSON_COUNT - ELIGIBLE_COUNT))
[ $DIFF -lt 0 ] && DIFF=$((-DIFF))

if [ $ELIGIBLE_COUNT -eq 0 ]; then
    VARIANCE=0
else
    VARIANCE=$(awk -v a="$DIFF" -v b="$ELIGIBLE_COUNT" 'BEGIN{printf "%.2f", (a*100.0)/b}')
fi

# Strict threshold: max 10% variance
if (( $(awk -v v="$VARIANCE" 'BEGIN{print (v > 10)}') )); then
    echo -e "  ${RED}❌ Count mismatch: ${VARIANCE}% (max 10%)${NC}"
    echo "  Check exclusion patterns and whitelist consistency"
    exit 1
else
    echo -e "  ${GREEN}✅ Count aligned: ${VARIANCE}% (≤10%)${NC}"
fi
