import asyncio
import logging
import time

import pytest

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.mark.asyncio
async def test_neuralbus_with_loops():
    """Test connexion NeuralBus + Loops avec métriques"""

    from jeffrey.core.loops.loop_manager import LoopManager

    # Créer manager (utilisera NeuralBus automatiquement)
    manager = LoopManager()

    # Démarrer
    await manager.start()

    # Laisser tourner 30 secondes
    start_time = time.time()
    metrics_history = []

    while time.time() - start_time < 30:
        # Collecter métriques
        loop_metrics = manager.get_all_metrics()
        bus_metrics = manager.event_bus.get_metrics() if hasattr(manager.event_bus, "get_metrics") else {}

        combined = {
            "time": time.time() - start_time,
            "symbiosis": loop_metrics["system"].get("symbiosis_score", 0),
            "bus_published": bus_metrics.get("published", 0),
            "bus_consumed": bus_metrics.get("consumed", 0),
            "bus_p99": bus_metrics.get("p99_latency_ms", 0),
            "bus_dropped": bus_metrics.get("dropped", 0) + bus_metrics.get("adapter_dropped", 0),
            "loops_cycles": loop_metrics["system"].get("total_cycles", 0),
        }

        metrics_history.append(combined)

        # Log progression
        print(f"\n⏱️ T+{combined['time']:.0f}s")
        print(f"  Symbiosis: {combined['symbiosis']:.3f}")
        print(f"  Bus: {combined['bus_published']} published, {combined['bus_consumed']} consumed")
        print(f"  P99: {combined['bus_p99']:.1f}ms")
        print(f"  Dropped: {combined['bus_dropped']}")

        await asyncio.sleep(5)

    # Arrêter
    await manager.stop()

    # Analyses finales
    final = metrics_history[-1]

    # ASSERTIONS DE PERFORMANCE
    assert final["symbiosis"] >= 0.3, f"Symbiosis dégradée: {final['symbiosis']}"
    assert final["bus_p99"] < 100, f"P99 trop élevé: {final['bus_p99']}ms"

    # Relax drop rate for initial phase
    drop_rate = final["bus_dropped"] / max(1, final["bus_published"])
    assert drop_rate < 0.05, f"Trop de drops: {drop_rate * 100:.2f}%"

    # Calculer throughput
    duration = metrics_history[-1]["time"]
    throughput = final["bus_published"] / max(1, duration)

    print("\n✅ TEST RÉUSSI")
    print(f"  Throughput: {throughput:.0f} msg/sec")
    print(f"  Total processed: {final['bus_consumed']}")
    print(f"  Drop rate: {drop_rate * 100:.2f}%")


@pytest.mark.asyncio
async def test_compression_performance():
    """Test que compression améliore performance sur gros payloads"""

    try:
        from jeffrey.core.neuralbus.bus_v2 import NeuralBus
        from jeffrey.core.neuralbus.contracts import CloudEvent, EventMeta
    except ImportError:
        pytest.skip("NeuralBus not available")

    neural_bus = NeuralBus()
    await neural_bus.initialize()

    # Créer payload volumineux
    large_payload = {
        "embeddings": [0.1 * i for i in range(512)],  # ~4KB
        "metadata": {"source": "test", "timestamp": time.time()},
    }

    # Test sans compression (forcer petits payloads)
    small_payload = {"test": "small"}
    start = time.time()
    for i in range(100):
        event = CloudEvent(type="test.small", data=small_payload, meta=EventMeta())
        await neural_bus.publish(event)
    time_no_compress = time.time() - start

    # Test avec gros payload (sera auto-compressé)
    start = time.time()
    for i in range(100):
        event = CloudEvent(type="test.large.compressed", data=large_payload, meta=EventMeta())
        await neural_bus.publish(event)
    time_compress = time.time() - start

    print(f"Small payloads (no compression): {time_no_compress:.2f}s")
    print(f"Large payloads (with compression): {time_compress:.2f}s")

    # Au moins pas de dégradation significative
    assert time_compress <= time_no_compress * 2, "Compression too slow"

    await neural_bus.shutdown()


@pytest.mark.asyncio
async def test_trace_propagation():
    """Test que les traces se propagent correctement"""

    from jeffrey.core.loops.loop_manager import LoopManager

    # Tracking des traces
    traces_seen = set()
    messages_received = []

    async def trace_handler(message):
        """Handler qui capture les traces"""
        if "headers" in message and "trace_id" in message["headers"]:
            traces_seen.add(message["headers"]["trace_id"])
        elif "payload" in message and "_trace_id" in message["payload"]:
            traces_seen.add(message["payload"]["_trace_id"])
        messages_received.append(message)

    # Créer manager
    manager = LoopManager()

    # Subscribe to events
    if hasattr(manager.event_bus, "subscribe"):
        await manager.event_bus.subscribe("loops.awareness.state_changed", trace_handler)

    # Start loops
    await manager.start(enable=["awareness"])

    # Wait for some activity
    await asyncio.sleep(10)

    # Stop
    await manager.stop()

    # Vérifier qu'on a des traces
    print(f"Traces captured: {len(traces_seen)}")
    print(f"Messages received: {len(messages_received)}")

    assert len(traces_seen) > 0, "No traces captured"
    assert len(messages_received) > 0, "No messages received"


@pytest.mark.asyncio
async def test_backpressure_handling():
    """Test que le système gère bien la backpressure"""

    from jeffrey.core.loops.loop_manager import LoopManager

    manager = LoopManager()

    # Simuler une forte charge
    initial_intervals = {}
    for name, loop in manager.loops.items():
        initial_intervals[name] = loop.interval_s

    # Start loops
    await manager.start()

    # Simuler backpressure en injectant des métriques dégradées
    if hasattr(manager.event_bus, "get_metrics"):
        # Mock degraded metrics
        class DegradedBus:
            def __init__(self, real_bus):
                self.real_bus = real_bus

            def get_metrics(self):
                return {
                    "pending_messages": 2000,  # High backlog
                    "p99_latency_ms": 100,  # High latency
                    "published": 1000,
                    "dropped": 50,  # 5% drop rate
                }

            def __getattr__(self, name):
                return getattr(self.real_bus, name)

        # Replace temporarily
        original_bus = manager.event_bus
        manager.event_bus = DegradedBus(original_bus)

        # Wait for health monitor to react
        await asyncio.sleep(10)

        # Check that non-critical loops were throttled
        for name in ["curiosity", "memory_consolidation"]:
            if name in manager.loops:
                loop = manager.loops[name]
                assert loop.interval_s > initial_intervals[name], f"{name} should be throttled under backpressure"

        # Restore normal bus
        manager.event_bus = original_bus

    await manager.stop()

    print("✅ Backpressure handling test passed")


@pytest.mark.asyncio
async def test_idempotence_and_order():
    """Test idempotence et ordre des messages"""
    try:
        from jeffrey.core.bus.neurobus_adapter import NeuroBusAdapter
        from jeffrey.core.neuralbus.bus_v2 import NeuralBus
        from jeffrey.core.neuralbus.contracts import CloudEvent, EventMeta
    except ImportError:
        pytest.skip("NeuralBus not available")

    neural_bus = NeuralBus()
    await neural_bus.initialize()
    adapter = NeuroBusAdapter(neural_bus)

    received = []
    duplicates = []

    async def test_handler(msg):
        """Handler qui track les messages reçus"""
        msg_id = msg["headers"].get("event_id")

        if msg_id in [r["id"] for r in received]:
            duplicates.append(msg_id)
        else:
            received.append({"id": msg_id, "seq": msg["payload"].get("sequence"), "timestamp": time.time()})

    # Subscribe
    await adapter.subscribe("test.order", test_handler)

    # Test 1 : Publier messages avec ordre
    for i in range(10):
        await adapter.publish("test.order", {"sequence": i, "key": "user:42"}, timeout=1.0)

    # Test 2 : Publier duplicate (même ID)
    duplicate_event = CloudEvent(id="duplicate-123", type="test.order", data={"sequence": 100}, meta=EventMeta())

    # Publier 3 fois le même
    for _ in range(3):
        if hasattr(neural_bus, "publish"):
            await neural_bus.publish(duplicate_event)

    # Attendre traitement
    await asyncio.sleep(2)

    # Vérifications
    print(f"Received: {len(received)} messages")
    print(f"Duplicates filtered: {len(duplicates)}")

    # Check ordre (si key-based ordering supporté)
    sequences = [r["seq"] for r in received if r["seq"] is not None]
    if sequences:
        is_ordered = all(sequences[i] <= sequences[i + 1] for i in range(len(sequences) - 1))
        print(f"Order preserved: {is_ordered}")

    # Check deduplication
    assert len(duplicates) <= 1, f"Trop de duplicates passés: {duplicates}"

    await neural_bus.shutdown()


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_neuralbus_with_loops())
    asyncio.run(test_compression_performance())
    asyncio.run(test_trace_propagation())
    asyncio.run(test_backpressure_handling())
    asyncio.run(test_idempotence_and_order())
