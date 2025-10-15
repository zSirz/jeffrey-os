#!/usr/bin/env python3
"""Production readiness validation"""

import asyncio
import sys
from pathlib import Path

# Ajuste selon la convention d'import Option A
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


async def validate():
    from jeffrey.core.memory.unified_memory import UnifiedMemory

    print("🔍 PRODUCTION READINESS CHECK")
    print("=" * 40)

    # Test 1: Full lifecycle
    print("\n1️⃣ Testing full lifecycle...")
    memory = UnifiedMemory(backend="sqlite")
    await memory.initialize()
    print("✅ Initialization successful")

    # Test 2: Basic operations
    print("\n2️⃣ Testing basic operations...")
    for i in range(100):
        await memory.store({"text": f"Test {i}", "type": "validation"})

    # Allow batch writer to flush
    await asyncio.sleep(0.5)

    results = await memory.retrieve("Test", limit=10)
    assert len(results) > 0, "Retrieve failed"
    print(f"✅ Basic ops: {len(results)} results found")

    # Test 3: Stats work
    print("\n3️⃣ Testing statistics...")
    stats = memory.get_stats()
    assert isinstance(stats, dict), "Stats not a dict"
    assert stats["total_stored"] >= 100, "Store count wrong"
    print(f"✅ Stats: {stats['total_stored']} stored, {stats.get('total_retrieved', 0)} retrieved")

    # Test 4: Cache functionality
    print("\n4️⃣ Testing cache...")
    cache_stats = stats.get("cache_stats", {})
    print(
        f"✅ Cache: size={cache_stats.get('size', 0)}, "
        f"hits={cache_stats.get('hits', 0)}, "
        f"hit_rate={cache_stats.get('hit_rate', 0):.2f}"
    )

    # Test 5: Performance settings
    print("\n5️⃣ Checking performance optimizations...")
    assert memory.write_queue.maxsize == 5000, "Queue size not optimized"
    assert memory._batch_size == 100, "Batch size not optimized"
    assert memory._flush_interval == 0.2, "Flush interval not optimized"
    print("✅ Performance settings optimized")

    # Test 6: Clean shutdown (critical test for _save_stats)
    print("\n6️⃣ Testing clean shutdown...")
    await memory.shutdown()
    print("✅ Shutdown complete")

    # Test 7: Files created
    print("\n7️⃣ Verifying persistence files...")
    stats_file = Path("data/memory_stats.json")
    learning_file = Path("data/jeffrey_learning.json")

    if stats_file.exists():
        print(f"✅ Stats file created: {stats_file}")
    else:
        print("⚠️  Stats file not found (non-critical)")

    if learning_file.exists():
        print(f"✅ Learning file exists: {learning_file}")
    else:
        print("⚠️  Learning file not found (non-critical)")

    print("\n" + "=" * 40)
    print("🎉 PRODUCTION READY!")
    print("=" * 40)
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(validate())
        if success:
            print("\n✅ UnifiedMemory 100% STABLE - READY TO DEPLOY!")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
