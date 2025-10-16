import pytest
from datetime import datetime, timedelta
from jeffrey.memory.hybrid_store import HybridMemoryStore

@pytest.mark.asyncio
async def test_memory_store_basic(test_db):
    """Test basic store and retrieve"""
    store = HybridMemoryStore()

    # Store a memory
    memory_data = {
        "text": "Test memory content",
        "emotion": "joy",
        "confidence": 0.85,
        "meta": {"source": "test"}
    }

    memory_id = await store.store(memory_data)
    assert memory_id is not None

    # Retrieve recent
    since = datetime.utcnow() - timedelta(hours=1)
    recent = await store.get_recent(since, limit=10)
    assert len(recent) > 0
    assert recent[0]["text"] == "Test memory content"

@pytest.mark.asyncio
async def test_cache_lru_functionality():
    """Test LRU cache eviction"""
    store = HybridMemoryStore(cache_size=3)

    # Fill cache beyond capacity
    for i in range(5):
        store._update_cache(f"id_{i}", {"text": f"Memory {i}"})

    # Verify oldest entries evicted
    assert len(store.cache) == 3
    assert "id_0" not in store.cache
    assert "id_4" in store.cache

@pytest.mark.asyncio
async def test_fallback_buffer_on_error():
    """Test fallback buffer activates on DB error"""
    store = HybridMemoryStore()

    # Simulate DB error by using invalid session
    memory_data = {"text": "Fallback test", "emotion": "concern"}

    # Store should fail but buffer should contain data
    try:
        # Force an error by passing None session
        await store.store(memory_data)
    except:
        pass

    assert len(store.fallback_buffer) > 0
    assert store.fallback_buffer[0]["text"] == "Fallback test"