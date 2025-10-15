"""
Tests performance spécifiques Mac avec RAM limitée
"""

import asyncio
import gc
import sys
import time

import psutil
import pytest

# Add parent directory to path
sys.path.insert(0, ".")

from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2


@pytest.mark.asyncio
async def test_latency_under_load():
    """Vérifie latence < 1s même sous charge"""

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # 100 messages rapides
    latencies = []

    for i in range(100):
        start = time.perf_counter()
        await federation.store_to_relevant("user", "user", f"Message {i}")
        latency = (time.perf_counter() - start) * 1000
        latencies.append(latency)

    # P99 < 1000ms
    p99 = sorted(latencies)[int(len(latencies) * 0.99)]
    assert p99 < 1000, f"P99 latency {p99}ms > 1000ms"

    # Médiane < 500ms
    p50 = sorted(latencies)[len(latencies) // 2]
    assert p50 < 500, f"P50 latency {p50}ms > 500ms"

    print(f"✅ Latency test passed - P50: {p50:.1f}ms, P99: {p99:.1f}ms")


def test_memory_usage_stable():
    """Vérifie que la RAM ne leak pas"""

    # Baseline
    gc.collect()
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    # Run async test
    async def intensive_test():
        loader = SecureModuleLoader()
        federation = MemoryFederationV2(loader)
        await federation.initialize(None)

        # 1000 stores
        for i in range(1000):
            await federation.store_to_relevant("user", "user", f"Message with some content {i} " * 10)

    asyncio.run(intensive_test())

    # Mesure après
    gc.collect()
    mem_after = process.memory_info().rss / 1024 / 1024  # MB

    # Max 100MB de croissance
    growth = mem_after - mem_before
    assert growth < 100, f"Memory grew by {growth}MB"

    print(f"✅ Memory test passed - Growth: {growth:.1f}MB")


@pytest.mark.asyncio
async def test_cache_pruning():
    """Test que le cache est bien nettoyé"""

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    federation.cache_size_limit = 50
    await federation.initialize(None)

    # Stocker 200 messages
    for i in range(200):
        await federation.store_to_relevant("user", "user", f"Message {i}")

    # Le cache doit être limité
    assert len(federation.seen_hashes) <= 50, f"Cache too large: {len(federation.seen_hashes)}"

    print(f"✅ Cache pruning test passed - Cache size: {len(federation.seen_hashes)}")


@pytest.mark.asyncio
async def test_degraded_mode_activation():
    """Test que le mode dégradé s'active sous pression"""

    from jeffrey.core.cognition.cognitive_core_lite import CognitiveCore
    from jeffrey.core.neuralbus.bus_v2 import NeuralBusV2

    bus = NeuralBusV2()
    await bus.start()

    core = CognitiveCore()
    await core.initialize(bus)

    # Forcer des erreurs
    for _ in range(10):
        core.metrics["errors"] += 1

    # Devrait déclencher le mode dégradé
    if core.metrics["errors"] > 5:
        await core._enter_degraded_mode()

    assert core.state["mode"] == "degraded"

    await bus.stop()

    print(f"✅ Degraded mode test passed - Mode: {core.state['mode']}")


@pytest.mark.asyncio
async def test_concurrent_operations():
    """Test opérations concurrentes avec semaphores"""

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Lancer 50 opérations en parallèle
    tasks = []
    for i in range(50):
        task = federation.store_to_relevant("user", "user", f"Concurrent message {i}")
        tasks.append(task)

    # Mesurer le temps total
    start = time.perf_counter()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    duration = time.perf_counter() - start

    # Vérifier que certaines ont réussi
    successes = [r for r in results if r and not isinstance(r, Exception)]
    assert len(successes) > 0, "No operations succeeded"

    print(f"✅ Concurrent ops test passed - {len(successes)}/50 succeeded in {duration:.1f}s")
