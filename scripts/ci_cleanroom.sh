#!/bin/bash
set -euo pipefail

echo "🧪 CLEAN ROOM VALIDATION"
echo "========================"

echo "1. Ban simple_modules..."
python3 scripts/ban_simple_modules.py

echo "2. Git trust check..."
python3 scripts/git_trust_check.py

echo "3. Hard verify realness..."
python3 scripts/hard_verify_realness.py

echo "4. Final validation..."
PYTHONPATH="$(pwd)/src" python3 validate_8_regions_strict.py

echo ""
echo "✅ CLEAN ROOM VALIDATION RÉUSSIE"
