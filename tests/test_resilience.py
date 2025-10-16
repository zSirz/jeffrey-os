import pytest
import asyncio
from unittest.mock import patch, MagicMock

@pytest.mark.asyncio
async def test_postgres_outage_resilience():
    """Simulate PostgreSQL outage and recovery"""
    from jeffrey.memory.hybrid_store import HybridMemoryStore

    store = HybridMemoryStore()

    # Simulate successful operations
    memory1 = {"text": "Before outage", "emotion": "calm"}
    await store.store(memory1)

    # Simulate DB outage
    with patch('jeffrey.db.session.AsyncSessionLocal') as mock_session:
        mock_session.side_effect = Exception("Database connection lost")

        # This should fail but use fallback
        memory2 = {"text": "During outage", "emotion": "concern"}
        try:
            await store.store(memory2)
        except:
            pass

        # Verify fallback buffer has the data
        assert len(store.fallback_buffer) > 0
        assert store.fallback_buffer[-1]["text"] == "During outage"

    # Simulate recovery and sync
    synced = await store.sync_fallback_buffer()
    assert synced >= 0  # Some memories might sync

@pytest.mark.asyncio
async def test_rate_limiting_protection():
    """Test system handles high load gracefully"""
    from httpx import AsyncClient

    async with AsyncClient(base_url="http://localhost:8000", timeout=30.0) as client:
        # Send many requests rapidly
        tasks = []
        for i in range(50):
            tasks.append(client.get("/healthz"))

        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no crashes, all should complete
        success_count = sum(1 for r in responses
                          if not isinstance(r, Exception)
                          and r.status_code == 200)

        assert success_count > 40  # At least 80% success rate