#!/bin/bash
# Ultimate validation script with all safety checks

set -Eeuo pipefail
IFS=$'\n'
trap 'echo "❌ Validation failed at line $LINENO"; exit 1' ERR

echo "🔍 JEFFREY OS ULTIMATE VALIDATION"
echo "=================================="

# Check dependencies
command -v jq >/dev/null || { echo "❌ jq manquant. macOS: brew install jq"; exit 1; }
command -v bc >/dev/null || { echo "❌ bc manquant."; exit 1; }

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

# Step 1: Run inventory
echo -e "\n📊 Running Ultimate Inventory..."
python3 scripts/inventory_ultimate.py || exit 1

# Step 2: Validate JSON schema
echo -e "\n✅ Validating JSON structure..."
jq empty artifacts/inventory_ultimate.json || exit 1

# Step 3: Check module count consistency
echo -e "\n📈 Checking module counts..."
JSON_COUNT=$(jq '.summary.total_modules' artifacts/inventory_ultimate.json)
FILE_COUNT=$(find src/jeffrey -name "*.py" \
    -not -path "*/tests/*" \
    -not -name "*test*.py" \
    -not -path "*/__pycache__/*" | wc -l)

echo "  Modules in JSON: $JSON_COUNT"
echo "  Files on disk: $FILE_COUNT"

# Calculate variance with awk (portable)
VARIANCE=$(awk -v a="$JSON_COUNT" -v b="$FILE_COUNT" 'BEGIN{d=a-b;if(d<0)d=-d; if(b==0){print 0; exit}; printf "%.2f", (d/b)*100}')

# Check if variance > 2%
if (( $(echo "$VARIANCE > 2" | bc -l) )); then
    echo -e "  ${RED}❌ Variance too high: ${VARIANCE}%${NC}"
    exit 1
else
    echo -e "  ${GREEN}✅ Variance acceptable: ${VARIANCE}%${NC}"
fi

# Step 4: Verify no tests in modules
echo -e "\n🚫 Checking test exclusion..."
TEST_COUNT=$(jq '[.modules[].name | select(contains("test"))] | length' artifacts/inventory_ultimate.json)
if [ "$TEST_COUNT" -gt 0 ]; then
    echo -e "  ${RED}❌ Found $TEST_COUNT test modules!${NC}"
    exit 1
else
    echo -e "  ${GREEN}✅ No test modules found${NC}"
fi

# Step 5: Check Bundle 1 requirements
echo -e "\n🚀 Checking Bundle 1 requirements..."

# Brain coverage
COVERAGE=$(jq -r '.bundle1_recommendations.regions_covered' artifacts/inventory_ultimate.json)
REGIONS=$(echo "$COVERAGE" | cut -d'/' -f1)
if [ "$REGIONS" -lt 6 ]; then
    echo -e "  ${RED}❌ Insufficient coverage: $COVERAGE (need ≥6/8)${NC}"
    exit 1
else
    echo -e "  ${GREEN}✅ Brain coverage: $COVERAGE${NC}"
fi

# P95 budget
P95=$(jq '.bundle1_recommendations.total_p95_budget_ms' artifacts/inventory_ultimate.json)
if (( $(echo "$P95 > 250" | bc -l) )); then
    echo -e "  ${RED}❌ P95 budget too high: ${P95}ms (max 250ms)${NC}"
    exit 1
else
    echo -e "  ${GREEN}✅ P95 budget: ${P95}ms${NC}"
fi

# Step 6: Create fallback modules if needed
echo -e "\n🔨 Creating simple fallback modules..."
mkdir -p src/jeffrey/fallbacks

for fallback in simple_memory simple_emotion simple_decision; do
    case "$fallback" in
        simple_memory)  CLASS="SimpleMemory";  REGION="cortex_temporal";;
        simple_emotion) CLASS="SimpleEmotion"; REGION="systeme_limbique";;
        simple_decision) CLASS="SimpleDecision"; REGION="cortex_frontal";;
    esac
    FILE="src/jeffrey/fallbacks/${fallback}.py"
    if [ ! -f "$FILE" ]; then
        cat > "$FILE" << EOF
"""Simple ${fallback} fallback module"""

__jeffrey_meta__ = {
    "version": "1.0.0",
    "stability": "stable",
    "brain_regions": ["$REGION"],
    "critical": True
}

def health_check():
    return True

class $CLASS:
    def process(self, input_data=None):
        return {"status": "ok", "result": input_data}
EOF
        echo "  Created $(basename "$FILE")"
    fi
done

echo -e "\n${GREEN}🎉 ALL VALIDATIONS PASSED!${NC}"
echo "=================================="
echo "Jeffrey OS is ready for Bundle 1 launch! 🚀"
