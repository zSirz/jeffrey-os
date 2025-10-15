#!/bin/bash
# launch_phase23.sh

echo "🚀 JEFFREY OS - Phase 2.3 Launch Sequence"
echo "=========================================="

# 1. Appliquer toutes les corrections
echo "✓ Applying critical fixes..."
python scripts/integrate_phase23.py

if [ $? -ne 0 ]; then
    echo "❌ Integration failed"
    exit 1
fi

# 2. Tests de validation
echo "✓ Running validation tests..."
pytest tests/test_phase23_complete.py -v

if [ $? -ne 0 ]; then
    echo "❌ Tests failed"
    exit 1
fi

# 3. Test de scalabilité
echo "✓ Running scalability test..."
pytest tests/test_scalability.py -v

if [ $? -ne 0 ]; then
    echo "❌ Scalability test failed"
    exit 1
fi

# 4. Smoke test (30s) avec Python au lieu de timeout
echo "✓ Smoke test (30s)..."
python -c "
import asyncio
import sys
import os
import time

# Add project root
sys.path.insert(0, os.path.abspath('.'))

from jeffrey.core.loops.loop_manager import LoopManager
from jeffrey.utils.test_helpers import NullBus, SimpleState, DummyMemoryFederation

async def test():
    print('Starting smoke test...')
    m = LoopManager(
        cognitive_core=SimpleState(),
        emotion_orchestrator=None,
        memory_federation=DummyMemoryFederation(),
        bus=NullBus(),
        mode_getter=lambda: 'normal',
        latency_budget_ok=lambda: True
    )

    await m.start()

    # Run for 20 seconds
    start = time.time()
    while time.time() - start < 20:
        await asyncio.sleep(5)
        metrics = m.get_metrics()
        print(f'  {int(time.time() - start)}s - Symbiosis: {metrics[\"system\"][\"symbiosis_score\"]:.3f}')

    metrics = m.get_metrics()
    print(f'Final Symbiosis: {metrics[\"system\"][\"symbiosis_score\"]:.3f}')

    assert metrics['system']['symbiosis_score'] > 0.3, 'Symbiosis too low'

    await m.stop()
    print('✅ Smoke test passed!')

asyncio.run(test())
"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ ALL TESTS PASSED!"
    echo ""
    echo "Ready to tag:"
    echo "  git add -A"
    echo "  git commit -m 'feat: Phase 2.3 - Production ready with ML intelligence'"
    echo "  git tag -a v2.3.0 -m 'Release 2.3.0 - Stable, intelligent, production-ready'"
    echo "  git push origin main --tags"
else
    echo "❌ Smoke test failed"
    exit 1
fi
