#!/bin/bash
set -e

echo "🔧 BUNDLE 1 STRICT PRODUCTION SETUP"
echo "===================================="

# 0. Ensure clean state
echo "📦 Backing up current state..."
git add -A && git commit -m "chore: pre-strict-setup snapshot" || true

# 1. Clear cache and rebuild
echo "🧹 Clearing cache..."
rm -f artifacts/module_cache.json

# 2. Run full validation sequence
echo "📊 Running inventory with strict criteria..."
make -f Makefile_hardened inventory

echo "🔍 Checking status..."
make -f Makefile_hardened status

# Get status
STATUS=$(python3 -c "import json; print(json.load(open('artifacts/inventory_ultimate.json'))['bundle1_recommendations']['status'])" 2>/dev/null || echo "error")
REGIONS=$(python3 -c "import json; print(json.load(open('artifacts/inventory_ultimate.json'))['bundle1_recommendations']['regions_covered'])" 2>/dev/null || echo "0/8")
MEASURED=$(python3 -c "import json; print(json.load(open('artifacts/inventory_ultimate.json'))['bundle1_recommendations']['measured_modules'])" 2>/dev/null || echo "0")
P95=$(python3 -c "import json; print(json.load(open('artifacts/inventory_ultimate.json'))['bundle1_recommendations']['total_p95_budget_ms'])" 2>/dev/null || echo "999")

echo ""
echo "Current status: $STATUS"
echo "Regions covered: $REGIONS"
echo "Measured modules: $MEASURED"
echo "P95 budget: ${P95}ms"
echo ""

# 3. Validate
echo "✅ Running strict validation..."
make -f Makefile_hardened validate

if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Check errors above."
    exit 1
fi

# 4. Dry run
echo "🧪 Running dry-run test..."
make -f Makefile_hardened dry-run

if [ $? -ne 0 ]; then
    echo "❌ Dry-run failed. Check errors above."
    exit 1
fi

# 5. Final check
if [ "$STATUS" = "ready" ]; then
    REGIONS_NUM=$(echo "$REGIONS" | cut -d'/' -f1)
    if [ "$REGIONS_NUM" -ge 6 ]; then
        echo ""
        echo "✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅"
        echo "🎉 BUNDLE 1 IS PRODUCTION-READY!"
        echo "✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅"
        echo ""
        echo "Status: $STATUS"
        echo "Regions: $REGIONS (requirement: ≥6/8)"
        echo "Measured: $MEASURED modules"
        echo "P95: ${P95}ms"
        echo ""
        echo "Ready to commit and launch:"
        echo "  git add -A"
        echo "  git commit -m \"feat: Bundle1 strict 6/8 production-ready\""
        echo "  git tag -a v6.3.0-bundle1-prod -m \"Bundle 1 PRODUCTION (strict criteria)\""
        echo "  make -f Makefile_hardened launch"
    else
        echo "⚠️  Status is ready but only $REGIONS_NUM/8 regions (need ≥6)"
        echo "Check that fallback modules were created properly"
    fi
else
    echo "❌ Status is '$STATUS' (need 'ready')"
    echo "Debug: Check which criteria are failing"
fi
