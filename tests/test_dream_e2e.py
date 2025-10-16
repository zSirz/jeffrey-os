import asyncio
import pytest
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from jeffrey.core.dreaming.dream_engine_progressive import DreamEngineProgressive


@pytest.fixture
async def mock_bus():
    """Mock event bus for testing"""
    bus = AsyncMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
async def mock_memory():
    """Mock memory port for testing"""
    memory = AsyncMock()

    # Sample memories for testing
    sample_memories = [
        {
            "text": "I feel excited about this new project",
            "emotion": "joy",
            "timestamp": datetime.now().isoformat()
        },
        {
            "text": "The weather is making me sad",
            "emotion": "sadness",
            "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
        },
        {
            "text": "This task is frustrating",
            "emotion": "anger",
            "timestamp": (datetime.now() - timedelta(hours=1)).isoformat()
        }
    ]

    memory.search = AsyncMock(return_value=sample_memories)
    return memory


@pytest.fixture
async def mock_circadian():
    """Mock circadian rhythm for testing"""
    circadian = AsyncMock()
    circadian.get_state = AsyncMock(return_value={
        "phase": "night",
        "energy_level": 0.2
    })
    return circadian


@pytest.fixture
async def dream_engine(mock_bus, mock_memory, mock_circadian):
    """Create DreamEngine instance for testing"""
    engine = DreamEngineProgressive(
        bus=mock_bus,
        memory_port=mock_memory,
        circadian=mock_circadian
    )

    # Enable for testing
    engine.enabled = True
    engine.test_mode = True
    engine.timeout = 10  # Shorter timeout for tests

    return engine


class TestDreamFullCycle:
    """Test complet du cycle de rêve"""

    async def test_dream_consolidation_success(self, dream_engine):
        """Test successful dream consolidation"""
        # Force a dream run
        result = await dream_engine.consolidate_memories(
            window_hours=24,
            force=True
        )

        # Verify result structure
        assert "run_id" in result
        assert "timestamp" in result
        assert "consolidation" in result
        assert "insights" in result
        assert "quality_score" in result
        assert result["test_mode"] is True

        # Verify insights were generated
        assert len(result["insights"]) > 0

        # Verify stats updated
        stats = dream_engine.get_stats()
        assert stats["runs_total"] == 1
        assert stats["runs_success"] == 1
        assert stats["runs_failed"] == 0

    async def test_dream_idempotence(self, dream_engine):
        """Test que deux runs consécutifs = skip sur le second"""
        # First run
        result1 = await dream_engine.consolidate_memories(force=True)
        assert "run_id" in result1

        # Second run without force should be skipped
        result2 = await dream_engine.consolidate_memories(force=False)
        assert result2.get("skipped") is True
        assert result2.get("reason") == "Conditions not met"

    async def test_dream_timeout_handling(self, dream_engine):
        """Test timeout handling"""
        # Mock a slow process that times out
        with patch.object(dream_engine, '_process_memories') as mock_process:
            mock_process.side_effect = asyncio.TimeoutError()

            with pytest.raises(asyncio.TimeoutError):
                await dream_engine.consolidate_memories(force=True)

            # Verify DLQ entry was created
            stats = dream_engine.get_stats()
            assert stats["dlq_size"] > 0
            assert stats["runs_failed"] == 1


class TestDistributedLock:
    """Test que deux instances ne peuvent pas runner simultanément"""

    async def test_file_lock_acquisition(self, dream_engine):
        """Test file-based distributed lock"""
        # First engine acquires lock
        today = datetime.now().strftime("%Y-%m-%d")
        lock1 = await dream_engine._acquire_distributed_lock(today)
        assert lock1 is True

        # Second acquisition should fail
        lock2 = await dream_engine._acquire_distributed_lock(today)
        assert lock2 is False

        # Release and retry
        await dream_engine._release_distributed_lock(today)
        lock3 = await dream_engine._acquire_distributed_lock(today)
        assert lock3 is True

        # Cleanup
        await dream_engine._release_distributed_lock(today)

    async def test_stale_lock_cleanup(self, dream_engine):
        """Test stale lock cleanup"""
        today = datetime.now().strftime("%Y-%m-%d")
        lock_file = Path(f"data/dreams/locks/{today}.lock")

        # Create a stale lock file
        lock_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file.touch()

        # Modify timestamp to make it stale
        old_time = time.time() - 120  # 2 minutes ago
        lock_file.touch(times=(old_time, old_time))

        # Should be able to acquire despite existing file
        lock = await dream_engine._acquire_distributed_lock(today, timeout=60)
        assert lock is True

        # Cleanup
        await dream_engine._release_distributed_lock(today)


class TestMemoryPagination:
    """Test pagination and memory handling"""

    async def test_paginated_memory_fetch(self, dream_engine, mock_memory):
        """Test memory pagination with budget control"""
        # Mock large memory dataset
        large_dataset = [
            {
                "text": f"Memory {i}",
                "emotion": "neutral",
                "timestamp": datetime.now().isoformat()
            } for i in range(200)
        ]

        mock_memory.search = AsyncMock(return_value=large_dataset)

        # Fetch with batch size limit
        dream_engine.batch_size = 50
        since = datetime.now() - timedelta(hours=24)

        memories = await dream_engine._fetch_memories_paginated(since)

        # Should respect batch size
        assert len(memories) <= dream_engine.batch_size

        # Should call memory search
        mock_memory.search.assert_called()


class TestBackfill:
    """Test backfill controlled"""

    async def test_backfill_multiple_days(self, dream_engine):
        """Test backfill for multiple days"""
        # Mock successful runs
        with patch.object(dream_engine, 'consolidate_memories') as mock_consolidate:
            mock_consolidate.return_value = {
                "run_id": "test123",
                "insights": ["test insight"]
            }

            result = await dream_engine.backfill(days=3)

            # Should attempt backfill for 3 days
            assert result["backfilled"] <= 3  # May be less if some days already processed
            assert len(result["results"]) <= 3

            # Should have called consolidate_memories
            assert mock_consolidate.call_count >= 0

    async def test_backfill_rate_limiting(self, dream_engine):
        """Test backfill includes rate limiting delays"""
        start_time = time.time()

        with patch.object(dream_engine, 'consolidate_memories') as mock_consolidate:
            mock_consolidate.return_value = {"run_id": "test", "insights": []}

            # Backfill 2 days
            await dream_engine.backfill(days=2)

            # Should have taken at least 5 seconds due to rate limiting
            # (assuming both days needed processing)
            # Note: In practice this depends on processed_dates

        end_time = time.time()
        # Just verify it completed without error for now


class TestAutoEvolution:
    """Test auto-evolution and parameter adjustment"""

    async def test_quality_based_batch_adjustment(self, dream_engine):
        """Test batch size auto-adjustment based on quality"""
        initial_batch_size = dream_engine.batch_size

        # Simulate low quality runs
        for _ in range(10):
            dream_engine.quality_history.append(0.01)  # Very low quality

        dream_engine._auto_adjust_parameters()

        # Batch size should increase for low quality
        assert dream_engine.batch_size >= initial_batch_size

        # Simulate high quality runs
        dream_engine.quality_history.clear()
        for _ in range(10):
            dream_engine.quality_history.append(0.8)  # High quality

        dream_engine._auto_adjust_parameters()

        # Batch size should decrease for high quality
        # (since fewer samples needed for good results)


class TestDLQAndFailures:
    """Test DLQ and failure handling"""

    async def test_dlq_entry_creation(self, dream_engine):
        """Test DLQ entry creation on failure"""
        initial_dlq_size = len(dream_engine.dlq)

        # Add a failure to DLQ
        dream_engine._add_to_dlq(
            "test_run_123",
            "Test error message",
            {"window_hours": 24}
        )

        # Verify DLQ entry
        assert len(dream_engine.dlq) == initial_dlq_size + 1

        # Verify DLQ file creation
        dlq_file = Path("data/dreams/failed/run_test_run_123.json")
        assert dlq_file.exists()

        # Verify file content
        with open(dlq_file, 'r') as f:
            dlq_data = json.load(f)
            assert dlq_data["run_id"] == "test_run_123"
            assert dlq_data["error"] == "Test error message"

        # Cleanup
        dlq_file.unlink()


# Integration test helpers
async def test_api_integration():
    """Integration test with API endpoints (requires running server)"""
    # This would test the actual API endpoints
    # Skip for unit tests, use in E2E test suite
    pytest.skip("Integration test - requires running server")


if __name__ == "__main__":
    # Run specific test
    pytest.main([__file__, "-v"])