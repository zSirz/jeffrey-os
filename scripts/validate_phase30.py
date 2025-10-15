#!/usr/bin/env python3
"""
Validation complète Phase 3.0 avec tous les fixes
"""

import asyncio
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def validate():
    """Valide tous les fixes"""
    print("🔍 VALIDATION PHASE 3.0")
    print("=" * 50)

    # 1. Test import
    print("✓ Importing modules...")
    try:
        from jeffrey.core.loops.loop_manager import LoopManager

        # On utilise l'import unifié pour le bus
        from jeffrey.core.neuralbus.bus_v2 import NeuralBusV2 as NeuralBus
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return 1

    # 2. Test initialization
    print("✓ Initializing NeuralBus...")
    manager = LoopManager()
    await manager.start()

    # 3. Test metrics
    print("✓ Checking metrics...")
    await asyncio.sleep(5)

    try:
        loop_metrics = manager.get_all_metrics()
        bus_metrics = manager.event_bus.get_metrics() if hasattr(manager.event_bus, "get_metrics") else {}

        print(f"  Symbiosis: {loop_metrics['system'].get('symbiosis_score', 0):.3f}")
        print(f"  Bus Published: {bus_metrics.get('published', 0)}")
        print(f"  Bus P99: {bus_metrics.get('p99_latency_ms', 0)}ms")
    except Exception as e:
        print(f"⚠️ Metrics error: {e}")

    # 4. Test publish/subscribe
    print("✓ Testing pub/sub...")

    received = []

    async def test_handler(msg):
        received.append(msg)
        print(f"  Received message: {msg.get('topic', 'unknown')}")

    try:
        sub_id = await manager.event_bus.subscribe("test.validation", test_handler)

        for i in range(10):
            await manager.event_bus.safe_publish("test.validation", {"test": i})

        await asyncio.sleep(2)
        print(f"  Messages sent: 10, received: {len(received)}")
    except Exception as e:
        print(f"⚠️ Pub/sub error: {e}")

    # 5. Test specific fixes
    print("✓ Testing specific fixes...")

    # Test subscribe handler robustness
    try:
        if hasattr(manager.event_bus, "_subscriptions"):
            print(f"  Active subscriptions: {len(manager.event_bus._subscriptions)}")
    except:
        pass

    # Test metrics mapping
    metrics = manager.event_bus.get_metrics()
    required_metrics = ["published", "consumed", "dropped", "p99_latency_ms", "pending_messages"]
    for metric in required_metrics:
        if metric in metrics:
            print(f"  ✓ {metric}: {metrics[metric]}")
        else:
            print(f"  ⚠️ Missing metric: {metric}")

    # 6. Test shutdown
    print("✓ Testing shutdown...")
    await manager.stop()

    # Résultats
    print("\n" + "=" * 50)

    # Critères de validation
    success = True
    if len(received) == 0:
        print("❌ No messages received - handler issue")
        success = False
    elif loop_metrics["system"].get("symbiosis_score", 0) < 0.3:
        print("❌ Symbiosis score too low")
        success = False

    if success:
        print("✅ VALIDATION RÉUSSIE - Phase 3.0 prête pour production!")
        print("\nNext steps:")
        print("  1. Run: python scripts/run_with_monitoring.py")
        print("  2. Run: pytest tests/integration/test_neuralbus_connection.py -v")
        return 0
    else:
        print("❌ VALIDATION ÉCHOUÉE - Vérifier les logs")
        print("\nDebug tips:")
        print("  1. Check jeffrey_neuralbus.log")
        print("  2. Ensure NATS is running: nats-server -js")
        print("  3. Check dependencies: pip install nats-py uvloop lz4 orjson")
        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(validate())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nValidation interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
