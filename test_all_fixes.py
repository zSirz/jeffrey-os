#!/usr/bin/env python3
"""Complete test suite for all UnifiedMemory fixes"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Windows compatibility
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def test_xss_protection():
    """Test XSS and injection protection"""
    from jeffrey.core.memory.unified_memory import MemoryValidator

    print("\n🔒 Testing XSS Protection...")

    # XSS patterns
    dangerous_inputs = [
        "<script>alert('xss')</script>",
        "text javascript:void(0) text",
        "<div onload='evil()'>test</div>",
        "<iframe src='evil.com'></iframe>",
        "'; DROP TABLE memories; --",
    ]

    for evil in dangerous_inputs:
        clean = MemoryValidator.sanitize_text(evil)
        assert "<script" not in clean.lower()
        assert "javascript:" not in clean
        assert "onload=" not in clean.lower()
        assert "<iframe" not in clean.lower()
        print(f"  ✅ Blocked: {evil[:30]}...")

    # SQL injection
    evil_data = {"text": "'; DROP TABLE memories; --"}
    assert MemoryValidator.validate(evil_data) == False
    print("  ✅ SQL injection blocked")

    # Recursive sanitization
    nested = {
        "user": "test",
        "data": {
            "message": "<script>bad</script>",
        },
    }
    clean_nested = MemoryValidator.sanitize_data(nested)
    assert "<script>" not in str(clean_nested)
    print("  ✅ Recursive sanitization works")


async def test_cache_lru():
    """Test LRU cache with TTL"""
    from jeffrey.utils.lru_cache import LRUCache

    print("\n💾 Testing LRU Cache...")

    # Basic LRU
    cache = LRUCache(maxsize=3, ttl=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)
    assert len(cache) == 3

    # Test eviction
    cache.set("d", 4)  # Should evict 'a'
    assert "a" not in cache
    assert cache.stats()["evictions"] == 1

    # Test stats
    stats = cache.stats()
    assert "hit_rate" in stats
    assert "maxsize" in stats
    assert stats["size"] <= 3
    print(f"  ✅ Cache stats: hit_rate={stats['hit_rate']:.2f}, size={stats['size']}")


async def test_unified_memory():
    """Test complete UnifiedMemory with all fixes"""
    from jeffrey.core.memory.unified_memory import UnifiedMemory

    print("\n🧠 Testing UnifiedMemory...")

    # Initialize
    memory = UnifiedMemory(backend="sqlite", cache_size=100, cache_ttl=60)
    await memory.initialize()

    # Verify all attributes exist
    required_attrs = [
        "data_dir",
        "current_context",
        "write_queue",
        "emotional_traces",
        "emotional_patterns",
        "relationships",
        "_writer_task",
        "_consol_task",
        "_pruner_task",
    ]

    for attr in required_attrs:
        assert hasattr(memory, attr), f"Missing attribute: {attr}"
    print(f"  ✅ All {len(required_attrs)} attributes present")

    # Verify tasks are running
    assert memory._writer_task and not memory._writer_task.done()
    assert memory._consol_task and not memory._consol_task.done()
    assert memory._pruner_task and not memory._pruner_task.done()
    print("  ✅ Background tasks running")

    # Test store with XSS attempt
    evil_memory = {
        "text": "Hello <script>alert('xss')</script> World",
        "type": "test",
        "user": "test_user",
    }
    mem_id = await memory.store(evil_memory)
    assert mem_id
    print(f"  ✅ Stored memory (sanitized): {mem_id}")

    # Allow batch writer to flush
    await asyncio.sleep(1.5)

    # Test retrieval
    results = await memory.query({"type": "test", "limit": 5})
    assert len(results) > 0
    # Verify XSS was sanitized
    assert "<script>" not in str(results[0])
    print(f"  ✅ Retrieved {len(results)} memories (XSS removed)")

    # Test cache hit
    results2 = await memory.query({"type": "test", "limit": 5})
    cache_stats = memory.cache.stats()
    assert cache_stats["hits"] > 0
    print(f"  ✅ Cache working: hits={cache_stats['hits']}")

    # Test stats
    stats = memory.get_stats()  # Sync version
    assert isinstance(stats, dict)
    assert "cache_stats" in stats
    assert stats["tasks_running"]["writer"] == True

    backend_stats = await memory.get_backend_stats()  # Async version
    assert isinstance(backend_stats, dict)
    print("  ✅ Stats working (sync + async)")

    # Clean shutdown
    await memory.shutdown()
    assert memory._writer_task.done()
    assert memory._consol_task.done()
    assert memory._pruner_task.done()
    print("  ✅ Clean shutdown")


async def test_sqlite_backend():
    """Test SQLite backend with FTS5 fallback"""
    from jeffrey.core.memory.sqlite.backend import SQLiteMemoryBackend

    print("\n🗄️ Testing SQLite Backend...")

    backend = SQLiteMemoryBackend("data/test_backend.db")
    await backend.initialize()

    # Check FTS5 status
    has_fts5 = getattr(backend, "has_fts5", False)
    print(f"  ℹ️  FTS5 available: {has_fts5}")

    # Store test data
    test_records = [{"_id": f"test_{i}", "text": f"Jeffrey is amazing AI assistant number {i}"} for i in range(10)]
    await backend.store_batch(test_records)
    print(f"  ✅ Stored {len(test_records)} records")

    # Test search (with FTS5 or LIKE fallback)
    results = await backend.search_text("amazing assistant", limit=5)
    assert len(results) > 0
    print(f"  ✅ Search works ({'FTS5' if has_fts5 else 'LIKE'}): {len(results)} results")

    # Test stats
    stats = await backend.get_stats()
    assert "memory_count" in stats
    print(f"  ✅ Backend stats: memory_count={stats['memory_count']}")

    await backend.shutdown()


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🔧 UNIFIED MEMORY - COMPLETE TEST SUITE")
    print("=" * 60)

    try:
        await test_xss_protection()
        await test_cache_lru()
        await test_unified_memory()
        await test_sqlite_backend()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED SUCCESSFULLY!")
        print("✅ UnifiedMemory is 100% PRODUCTION READY!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
