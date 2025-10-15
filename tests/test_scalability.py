"""
Test de scalabilité pour Jeffrey OS
Phase 2.3 - Test avec 120 loops simultanées
"""

import asyncio
import os
import sys

import psutil
import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jeffrey.core.loops.base import BaseLoop
from jeffrey.core.loops.loop_manager import LoopManager
from jeffrey.utils.test_helpers import NullBus


class MockLoop(BaseLoop):
    """Mock loop pour tests de charge"""

    def __init__(self, name: str, interval: float = 1.0):
        super().__init__(name=name, interval_s=interval, jitter_s=0.1, hard_timeout_s=0.5, bus=NullBus())
        self.tick_count = 0

    async def _tick(self):
        """Simple tick qui compte"""
        self.tick_count += 1
        await asyncio.sleep(0.01)  # Simulate light work

    def _get_state(self) -> str:
        """Return simple state for RL"""
        return f"tick_{self.tick_count}"

    def _get_action(self, state: str, epsilon: float = 0.1) -> str:
        """Return simple action"""
        return "continue"


class TestScalability:
    """Tests de scalabilité et performance"""

    @pytest.mark.asyncio
    async def test_120_loops_stress(self):
        """Test avec 120 boucles simultanées"""
        print("\n🚀 Starting 120 loops stress test...")

        # Créer manager avec bus mock
        manager = LoopManager(cognitive_core=None, emotion_orchestrator=None, memory_federation=None, bus=NullBus())

        # Créer 120 loops mock supplémentaires
        extra_loops = {}
        for i in range(120):
            loop_name = f"test_loop_{i}"
            extra_loops[loop_name] = MockLoop(loop_name, interval=0.5 + (i % 10) * 0.1)

        # Ajouter les loops au manager
        manager.loops.update(extra_loops)

        # Mesures avant
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = process.cpu_percent()

        # Démarrer manager
        print(f"Memory before: {mem_before:.1f}MB")
        await manager.start()

        # Attendre 10 secondes
        print("Running for 10 seconds...")
        await asyncio.sleep(10)

        # Mesures après
        mem_after = process.memory_info().rss / 1024 / 1024
        cpu_avg = process.cpu_percent(interval=1)

        # Obtenir métriques
        metrics = manager.get_all_metrics()

        # Arrêter manager
        await manager.stop()

        # Vérifications
        mem_increase = mem_after - mem_before
        print("\n📊 Results:")
        print(f"  Memory: {mem_before:.1f}MB → {mem_after:.1f}MB (Δ{mem_increase:.1f}MB)")
        print(f"  CPU average: {cpu_avg:.1f}%")
        print(f"  Symbiosis: {metrics['system']['symbiosis_score']:.2f}")
        print(f"  Total cycles: {metrics['system']['total_cycles']}")
        print(f"  Total errors: {metrics['system']['total_errors']}")
        print(f"  Bus dropped: {metrics['system']['bus_dropped']}")

        assert mem_increase < 500, f"Fuite mémoire détectée (>{mem_increase}MB)"
        assert cpu_avg < 80, f"CPU trop élevé: {cpu_avg}%"
        assert metrics["system"]["symbiosis_score"] > 0.3, "Symbiosis trop bas"

        print("✅ 120 loops test PASSED")

    @pytest.mark.asyncio
    async def test_memory_under_pressure(self):
        """Test comportement quand RAM limitée"""
        print("\n🧪 Testing memory pressure behavior...")

        manager = LoopManager(cognitive_core=None, emotion_orchestrator=None, memory_federation=None, bus=NullBus())

        # Simuler haute utilisation RAM
        for loop in manager.loops.values():
            if hasattr(loop, "budget_gate"):
                # Simulate high memory by making gate return False
                loop.budget_gate = lambda: False

        await manager.start()
        await asyncio.sleep(2)

        metrics = manager.get_all_metrics()
        await manager.stop()

        # Vérifier que les loops sont throttled
        print(f"  Cycles under pressure: {metrics['system']['total_cycles']}")
        print(f"  Errors under pressure: {metrics['system']['total_errors']}")

        # Should have very few cycles due to gate
        assert metrics["system"]["total_cycles"] < 10, "Loops should be throttled"

        print("✅ Memory pressure test PASSED")

    @pytest.mark.asyncio
    async def test_rapid_start_stop(self):
        """Test démarrages/arrêts rapides"""
        print("\n⚡ Testing rapid start/stop...")

        manager = LoopManager(bus=NullBus())

        for i in range(5):
            print(f"  Cycle {i + 1}/5")
            await manager.start()
            await asyncio.sleep(0.5)
            await manager.stop()

        # Should not crash
        print("✅ Rapid start/stop test PASSED")


if __name__ == "__main__":
    # Run tests directly
    import asyncio

    async def run_all_tests():
        test = TestScalability()

        print("=" * 50)
        print("🧪 SCALABILITY TESTS")
        print("=" * 50)

        try:
            await test.test_120_loops_stress()
            await test.test_memory_under_pressure()
            await test.test_rapid_start_stop()
            print("\n🎉 ALL TESTS PASSED!")
        except Exception as e:
            print(f"\n❌ TEST FAILED: {e}")
            import traceback

            traceback.print_exc()

    asyncio.run(run_all_tests())
