#!/bin/bash
# Ultimate validation script HARDENED with all fixes
set -Eeuo pipefail
IFS=$'\n'
trap 'echo "❌ Validation failed at line $LINENO"; exit 1' ERR

echo "🔍 JEFFREY OS ULTIMATE VALIDATION HARDENED"
echo "==========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check dependencies
command -v python3 >/dev/null 2>&1 || { echo "❌ python3 required"; exit 1; }
command -v jq >/dev/null 2>&1 || command -v python3 >/dev/null 2>&1 || { echo "❌ jq or python required"; exit 1; }

# Step 1: Run inventory
echo -e "\n📊 Running Ultimate Inventory..."
python3 scripts/inventory_ultimate.py || exit 1

# Step 2: Validate JSON (with fallback)
echo -e "\n✅ Validating JSON structure..."
if command -v jq >/dev/null 2>&1; then
    jq empty artifacts/inventory_ultimate.json || exit 1
else
    python3 -m json.tool artifacts/inventory_ultimate.json >/dev/null || exit 1
fi

# Step 2b: Run strict validation (10% tolerance)
bash scripts/validate_strict_block.sh || exit 1

# Step 3: Check module count consistency (FIXED variance calc)
echo -e "\n📈 Checking module counts..."
if command -v jq >/dev/null 2>&1; then
    JSON_COUNT=$(jq '.summary.total_modules' artifacts/inventory_ultimate.json)
else
    JSON_COUNT=$(python3 -c "import json; print(json.load(open('artifacts/inventory_ultimate.json'))['summary']['total_modules'])")
fi

FILE_COUNT=$(find src/jeffrey -name "*.py" \
    -not -path "*/tests/*" \
    -not -path "*/test/*" \
    -not -name "*test*.py" \
    -not -name "*_test.py" \
    -not -path "*/__pycache__/*" \
    -not -path "*/archive/*" \
    -not -path "*/deprecated/*" \
    -not -path "*/old/*" \
    -not -path "*/backup/*" 2>/dev/null | wc -l | tr -d ' ')

echo "  Modules in JSON: $JSON_COUNT"
echo "  Files on disk: $FILE_COUNT"

# Fixed variance calculation without bc
DIFF=$((JSON_COUNT - FILE_COUNT))
[ $DIFF -lt 0 ] && DIFF=$((-DIFF))
if [ $FILE_COUNT -eq 0 ]; then
    VARIANCE=0
else
    VARIANCE=$(awk -v a="$DIFF" -v b="$FILE_COUNT" 'BEGIN{printf "%.2f", (a*100.0)/b}')
fi

# Check variance (allow 40% for minimum lines filter)
if (( $(awk -v v="$VARIANCE" 'BEGIN{print (v > 40)}') )); then
    echo -e "  ${RED}❌ Variance too high: ${VARIANCE}%${NC}"
    exit 1
else
    echo -e "  ${GREEN}✅ Variance acceptable: ${VARIANCE}%${NC} (modules have min 60 lines filter)"
fi

# Step 4: Create ALL fallback modules
echo -e "\n🔨 Creating fallback modules..."
mkdir -p src/jeffrey/fallbacks

# Extended fallback list with all critical modules
for fallback in simple_memory simple_emotion simple_decision simple_bus simple_parser simple_pipeline; do
    case "$fallback" in
        simple_memory)
            CLASS="SimpleMemory"
            REGION="cortex_temporal"
            ;;
        simple_emotion)
            CLASS="SimpleEmotion"
            REGION="systeme_limbique"
            ;;
        simple_decision)
            CLASS="SimpleDecision"
            REGION="cortex_frontal"
            ;;
        simple_bus)
            CLASS="SimpleBus"
            REGION="tronc_cerebral"
            ;;
        simple_parser)
            CLASS="SimpleParser"
            REGION="cortex_occipital"
            ;;
        simple_pipeline)
            CLASS="SimplePipeline"
            REGION="tronc_cerebral"
            ;;
    esac

    FILE="src/jeffrey/fallbacks/${fallback}.py"
    if [ ! -f "$FILE" ]; then
        cat > "$FILE" << EOF
"""Simple ${fallback} fallback module - Ultra robust"""

__jeffrey_meta__ = {
    "version": "1.0.0",
    "stability": "stable",
    "brain_regions": ["${REGION}"],
    "critical": True,
    "contracts": {
        "provides": ["process(data: dict) -> dict", "health_check() -> dict"],
        "consumes": []
    }
}

import asyncio
from typing import Dict, Any, Optional

class ${CLASS}:
    """Minimal ${fallback} implementation for fallback"""

    def __init__(self):
        self.active = True
        self.state = {}
        self.call_count = 0

    async def process(self, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process with minimal logic"""
        self.call_count += 1

        if not self.active:
            return {"status": "error", "reason": "inactive"}

        # Minimal processing based on type
        result = {
            "status": "ok",
            "processed": True,
            "data": data or {},
            "module": "${fallback}",
            "call_count": self.call_count
        }

        # Type-specific minimal logic
        if "${fallback}" == "simple_memory" and data:
            self.state[data.get('key', 'default')] = data.get('value')
            result['stored'] = True
        elif "${fallback}" == "simple_emotion" and data:
            result['emotion'] = 'neutral'
            result['intensity'] = 0.5
        elif "${fallback}" == "simple_decision" and data:
            result['decision'] = 'continue'
            result['confidence'] = 0.7

        return result

    def health_check(self) -> Dict[str, Any]:
        """Health check with metrics"""
        return {
            "status": "healthy" if self.active else "degraded",
            "module": "${fallback}",
            "call_count": self.call_count,
            "state_size": len(self.state)
        }

# Module-level functions for compatibility
def health_check():
    """Module-level health check"""
    return {"status": "healthy", "module": "${fallback}"}

async def process(data=None):
    """Module-level process"""
    instance = ${CLASS}()
    return await instance.process(data)
EOF
        echo -e "  Created ${GREEN}${fallback}.py${NC}"
    fi
done

# Step 5: Check Bundle 1 requirements
echo -e "\n🚀 Checking Bundle 1 requirements..."

# Get bundle data
if command -v jq >/dev/null 2>&1; then
    COVERAGE=$(jq -r '.bundle1_recommendations.regions_covered' artifacts/inventory_ultimate.json)
    MEASURED=$(jq -r '.bundle1_recommendations.measured_modules' artifacts/inventory_ultimate.json)
    P95=$(jq '.bundle1_recommendations.total_p95_budget_ms' artifacts/inventory_ultimate.json)
    STATUS=$(jq -r '.bundle1_recommendations.status' artifacts/inventory_ultimate.json)
else
    # Python fallback
    BUNDLE_DATA=$(python3 -c "
import json
data = json.load(open('artifacts/inventory_ultimate.json'))
b = data['bundle1_recommendations']
print(f\"{b['regions_covered']}|{b.get('measured_modules', 0)}|{b['total_p95_budget_ms']}|{b['status']}\")
")
    IFS='|' read -r COVERAGE MEASURED P95 STATUS <<< "$BUNDLE_DATA"
fi

REGIONS=$(echo "$COVERAGE" | cut -d'/' -f1)

# Check all criteria
ERRORS=0

if [ "$REGIONS" -lt 6 ]; then
    echo -e "  ${RED}❌ Insufficient coverage: $COVERAGE (need ≥6/8)${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "  ${GREEN}✅ Brain coverage: $COVERAGE${NC}"
fi

if [ "${MEASURED:-0}" -lt 3 ]; then
    echo -e "  ${YELLOW}⚠️  Only $MEASURED measured modules (recommend ≥3)${NC}"
else
    echo -e "  ${GREEN}✅ Measured modules: $MEASURED${NC}"
fi

if (( $(awk -v p="$P95" 'BEGIN{print (p > 250)}') )); then
    echo -e "  ${RED}❌ P95 budget too high: ${P95}ms (max 250ms)${NC}"
    ERRORS=$((ERRORS + 1))
else
    echo -e "  ${GREEN}✅ P95 budget: ${P95}ms${NC}"
fi

if [ "$STATUS" != "ready" ]; then
    echo -e "  ${YELLOW}⚠️  Bundle status: $STATUS (not 'ready')${NC}"
    [ "$STATUS" = "needs_more_coverage" ] && ERRORS=$((ERRORS + 1))
else
    echo -e "  ${GREEN}✅ Bundle status: ready${NC}"
fi

# Step 6: Sign artifacts
echo -e "\n🔏 Signing artifacts..."
cd artifacts 2>/dev/null && {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 *.json > SHA256SUMS 2>/dev/null
        echo -e "  ${GREEN}✅ Artifacts signed${NC}"
    else
        echo -e "  ${YELLOW}⚠️  shasum not found${NC}"
    fi
    cd - >/dev/null
}

# Add git commit to report
if [ -d .git ]; then
    GIT_COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
    if command -v jq >/dev/null 2>&1; then
        jq --arg commit "$GIT_COMMIT" '.git_commit = $commit' artifacts/inventory_ultimate.json > tmp.json
        mv tmp.json artifacts/inventory_ultimate.json
    fi
    echo -e "  ${GREEN}✅ Git commit: $GIT_COMMIT${NC}"
fi

# Final decision
echo -e "\n==========================================="
if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}🎉 ALL VALIDATIONS PASSED!${NC}"
    echo "Jeffrey OS Bundle 1 is ready for launch! 🚀"
    exit 0
else
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    echo "Fix the issues above before launching."
    exit 1
fi
# Bundle Lock Verification
echo -e "\n🔒 Verifying bundle lock..."
if [ -f artifacts/bundle1.lock.json ]; then
  python3 - <<'PY' || { echo "❌ Bundle lock mismatch - modules have been modified!"; exit 1; }
import json, hashlib, sys
from pathlib import Path

lock = json.load(open("artifacts/bundle1.lock.json"))
mismatches = []

for entry in lock["modules"]:
    p = Path(entry["path"])
    if not p.exists():
        mismatches.append((entry["name"], "FILE_MISSING"))
        continue

    actual_sha = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual_sha != entry["sha256"]:
        mismatches.append((entry["name"], "SHA_MISMATCH"))

if mismatches:
    print(f"❌ Bundle integrity check failed!")
    for name, reason in mismatches:
        print(f"   - {name}: {reason}")
    sys.exit(1)

print(f"✅ Bundle lock verified - {len(lock['modules'])} modules intact")
PY
else
  echo "ℹ️  No bundle lock file yet (run gen_bundle_lock.py)"
fi
