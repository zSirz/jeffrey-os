"""
Tests complets Phase 2.3
"""

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jeffrey.core.loops.base import BaseLoop
from jeffrey.core.ml.memory_clusterer import AdaptiveMemoryClusterer
from jeffrey.core.monitoring.entropy_guardian import EntropyGuardian
from jeffrey.utils.test_helpers import NullBus


class TestPhase23Complete:
    """Tests complets Phase 2.3"""

    @pytest.mark.asyncio
    async def test_replay_buffer_persistence(self):
        """Vérifie sauvegarde du replay buffer"""

        # Create a test loop
        class TestLoop(BaseLoop):
            async def _tick(self):
                return {"result": "ok"}

        loop = TestLoop(name="test", interval_s=1.0, bus=NullBus())

        # Mock replay buffer
        loop.replay_buffer = MagicMock()
        loop.replay_buffer.save = MagicMock()
        loop.replay_buffer.__len__ = MagicMock(return_value=100)

        await loop.stop()

        # Doit avoir appelé save()
        loop.replay_buffer.save.assert_called_once()

    @pytest.mark.asyncio
    async def test_safe_publish_drops(self):
        """Vérifie comptage des drops"""

        class TestLoop(BaseLoop):
            async def _tick(self):
                return {"result": "ok"}

        loop = TestLoop(name="test", interval_s=1.0, bus=None)

        # Mock bus that times out
        loop.bus = AsyncMock()
        loop.bus.publish = AsyncMock(side_effect=asyncio.TimeoutError)

        # Publier avec timeout
        await loop.safe_publish("test_event", {}, timeout=0.001)

        # Doit avoir incrémenté le compteur
        assert loop.bus_dropped_count == 1

    def test_entropy_detection(self):
        """Vérifie détection de biais"""
        guardian = EntropyGuardian()

        # Données biaisées (faible entropie)
        biased_data = {
            "state1": {"action1": 1.0, "action2": 1.0},
            "state2": {"action1": 1.0, "action2": 1.0},
        }

        entropy = guardian.check_entropy("test_component", biased_data)

        # Entropie doit être faible
        assert entropy < 1.5

        # Après plusieurs checks, doit générer alerte
        for _ in range(15):
            guardian.check_entropy("test_component", biased_data)

        assert len(guardian.bias_alerts) > 0
        assert guardian.bias_alerts[0]["risk"] in ["medium", "high"]

    def test_adaptive_memory_clusterer(self):
        """Vérifie auto-tune du clustering"""
        clusterer = AdaptiveMemoryClusterer()
        initial_eps = clusterer.eps

        # Memories de test
        memories = [{"content": f"Memory about topic {i % 3}"} for i in range(20)]

        # Cluster plusieurs fois
        for _ in range(5):
            clustered = clusterer.cluster_memories(memories)

        # Doit avoir de l'historique de qualité
        assert len(clusterer.quality_history) > 0

        # Eps peut avoir changé (auto-tune)
        # Note: eps might or might not change depending on quality
        # Just check it's still in valid range
        assert 0.05 <= clusterer.eps <= 1.0

    def test_percentile_without_numpy(self):
        """Test percentile calculation without numpy"""

        class TestLoop(BaseLoop):
            async def _tick(self):
                return {"result": "ok"}

        loop = TestLoop(name="test", interval_s=1.0, bus=NullBus())

        # Add test latencies
        loop._latencies_ms = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        # Calculate percentiles
        p50 = loop._percentile(sorted(loop._latencies_ms), 50)
        p95 = loop._percentile(sorted(loop._latencies_ms), 95)
        p99 = loop._percentile(sorted(loop._latencies_ms), 99)

        # For a 10-element list, index calculations:
        # P50: round(0.5 * 9) = 4 -> 50
        # P95: round(0.95 * 9) = 9 -> 100 (not 90)
        # P99: round(0.99 * 9) = 9 -> 100

        assert p50 == 50  # Median
        assert p95 == 100  # 95th percentile (last element for small list)
        assert p99 == 100  # 99th percentile

    def test_metrics_with_bus_dropped(self):
        """Test metrics include bus_dropped counter"""

        class TestLoop(BaseLoop):
            async def _tick(self):
                return {"result": "ok"}

        loop = TestLoop(name="test", interval_s=1.0, bus=NullBus())

        # Simulate some dropped messages
        loop.bus_dropped_count = 5

        metrics = loop.get_metrics()

        assert "bus_dropped" in metrics
        assert metrics["bus_dropped"] == 5

    def test_memory_compression(self):
        """Test memory compression in consolidation loop"""
        from jeffrey.core.loops.memory_consolidation import MemoryConsolidationLoop

        loop = MemoryConsolidationLoop(memory_federation=None, budget_gate=None, bus=NullBus())

        # Add test memories
        for i in range(60):
            loop.short_term.append({"id": i, "content": f"Memory {i}"})

        # Trigger compression
        loop._compress_old_memories()

        # Should have compressed some memories
        assert len(loop.compressed_history) > 0
        assert len(loop.short_term) < 60

        # Check compression worked
        first_compressed = loop.compressed_history[0]
        assert "data" in first_compressed
        assert "size_ratio" in first_compressed
        assert first_compressed["size_ratio"] < 1.0  # Should be compressed

    def test_entropy_guardian_recommendations(self):
        """Test entropy guardian recommendations"""
        guardian = EntropyGuardian()

        # Add low entropy component
        for _ in range(20):
            guardian.check_entropy("low_entropy_component", {"a": 1, "b": 1})

        recommendations = guardian.get_recommendations()

        assert len(recommendations) > 0
        assert "low_entropy_component" in recommendations[0]
        assert "entropy" in recommendations[0].lower()

    @pytest.mark.asyncio
    async def test_loop_manager_learning(self):
        """Test LoopManager pattern detection"""
        from jeffrey.core.loops.loop_manager import LoopManager

        manager = LoopManager(bus=NullBus())

        # Add fake metrics history
        for i in range(20):
            manager.metrics_history.append(
                {
                    "loops": {
                        "awareness": {"cycles": i * 2},
                        "curiosity": {"cycles": i * 2},  # Perfect correlation
                        "emotional_decay": {"cycles": 20 - i},  # Negative correlation
                    }
                }
            )

        # Detect patterns
        patterns = manager._detect_patterns()

        # Should find correlations
        assert len(patterns) > 0

        # Check for positive correlation
        positive_found = False
        negative_found = False

        for pattern in patterns:
            if pattern["type"] == "positive_correlation":
                if "awareness" in pattern["loops"] and "curiosity" in pattern["loops"]:
                    positive_found = True
                    assert pattern["correlation"] > 0.9  # Should be high

            if pattern["type"] == "negative_correlation":
                if "emotional_decay" in pattern["loops"]:
                    negative_found = True
                    assert pattern["correlation"] < -0.5  # Should be negative

        assert positive_found, "Should detect positive correlation"
        assert negative_found, "Should detect negative correlation"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
