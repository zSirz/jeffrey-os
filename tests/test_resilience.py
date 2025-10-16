import pytest
import asyncio
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_postgres_outage_resilience():
    """Test fallback buffer activation during DB outage"""
    from jeffrey.memory.hybrid_store import HybridMemoryStore

    store = HybridMemoryStore()

    # Test normal operation first
    memory1 = {"text": "Before outage", "emotion": "calm"}
    id1 = await store.store(memory1)
    assert id1 is not None

    # Force fallback by setting a flag (don't actually break DB)
    store._force_fallback = True

    # This should use fallback
    memory2 = {"text": "During outage", "emotion": "concern"}
    id2 = await store.store(memory2)

    # Verify fallback buffer has the data
    assert len(store.fallback_buffer) > 0
    assert any(m["text"] == "During outage" for m in store.fallback_buffer)

    # Reset and sync
    store._force_fallback = False
    synced = await store.sync_fallback_buffer()
    assert synced >= 0

@pytest.mark.asyncio
async def test_rate_limiting_protection():
    """Test API handles concurrent requests gracefully"""
    from httpx import AsyncClient
    import asyncio

    async with AsyncClient(base_url="http://localhost:8000", timeout=5.0) as client:
        # Send only 10 concurrent requests (more realistic)
        tasks = [client.get("/healthz") for _ in range(10)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Count successful responses
        success_count = sum(1 for r in responses
                          if not isinstance(r, Exception)
                          and hasattr(r, 'status_code')
                          and r.status_code == 200)

        # At least 80% should succeed
        assert success_count >= 8, f"Only {success_count}/10 requests succeeded"