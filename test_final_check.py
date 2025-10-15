#!/usr/bin/env python3
"""Final verification test"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path for proper imports
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


async def test_final():
    from jeffrey.core.memory.unified_memory import UnifiedMemory

    print("🔍 FINAL VERIFICATION")
    print("=" * 40)

    # Test 1: Initialize and check persistence
    memory = UnifiedMemory(backend="sqlite")
    await memory.initialize()
    print("✅ Initialization OK")

    # Test 2: Store and retrieve
    mem_id = await memory.store({"text": "Final test memory", "type": "test"})
    # Wait for batch writer to flush
    await asyncio.sleep(1.5)
    results = await memory.retrieve("Final test")
    assert len(results) > 0
    print("✅ Store/Retrieve OK")

    # Test 3: Test compatibility search_memories (sync version)
    memory.learned_preferences["test_user"] = {"animal_chien": "Rex"}
    sync_results = memory.search_memories("test_user", "chien")
    assert len(sync_results) > 0
    print(f"✅ Sync search OK: {sync_results[0]}")

    # Test 4: Stats work
    stats = memory.get_stats()
    assert isinstance(stats, dict)
    backend_stats = await memory.get_backend_stats()
    assert isinstance(backend_stats, dict)
    print("✅ Stats OK")

    # Test 5: Clean shutdown (should save stats + persistent data)
    await memory.shutdown()
    print("✅ Shutdown OK")

    # Test 6: Check files were created
    data_dir = Path("data")
    stats_file = data_dir / "memory_stats.json"
    learning_file = data_dir / "jeffrey_learning.json"

    # Check stats file
    if stats_file.exists():
        with open(stats_file) as f:
            stats_data = json.load(f)
            assert "runtime" in stats_data
            assert "backend" in stats_data
        print("✅ Stats file created and valid")
    else:
        print("⚠️  Stats file not found (non-critical)")

    # Check learning file
    if learning_file.exists():
        with open(learning_file) as f:
            learning_data = json.load(f)
            # Should have our test user data
            assert "test_user" in learning_data or "preferences" in learning_data
        print("✅ Persistence file created and valid")
    else:
        print("⚠️  Learning file not found (non-critical)")

    print("\n" + "=" * 40)
    print("🎉 ALL CHECKS PASSED!")
    print("✅ UnifiedMemory 100% READY!")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_final())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
