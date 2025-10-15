"""
Integration test for autonomous loops with proper helpers
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jeffrey.core.loops.loop_manager import LoopManager
from jeffrey.utils.test_helpers import DummyMemoryFederation, NullBus, SimpleState


async def run_integration_test(duration: int = 60):
    """Run loops for specified duration and collect metrics"""
    print("🚀 Starting integration test...")
    print(f"Duration: {duration}s")

    # Create test infrastructure
    event_bus = NullBus()
    state_manager = SimpleState()

    # Load config
    config_path = Path("config/modules.yaml")
    config = {}
    if config_path.exists():
        import yaml

        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Ensure loops are enabled for test
    if "autonomous_loops" not in config or not config["autonomous_loops"].get("enabled"):
        print("⚠️  Loops not enabled in config, using test config")
        config["autonomous_loops"] = {
            "enabled": True,
            "awareness": {"enabled": True, "interval": 2.0},
            "emotional_decay": {"enabled": True, "interval": 5.0},
            "memory_consolidation": {
                "enabled": True,
                "interval": 10.0,
                "use_ml_clustering": False,  # Start without ML
                "metadata_only": False,
            },
            "curiosity": {"enabled": True, "interval": 15.0},
        }

    # Create loop manager
    manager = LoopManager(event_bus, state_manager, config)

    # Inject dummy memory for consolidation loop
    if "memory_consolidation" in manager.loops:
        manager.loops["memory_consolidation"].memory_federation = DummyMemoryFederation()
        print("✅ Memory federation injected")

    # Start loops
    await manager.start()

    # Run for duration
    start_time = time.time()
    metrics_history = []

    while time.time() - start_time < duration:
        await asyncio.sleep(10)

        # Collect metrics
        metrics = manager.get_all_metrics()
        metrics["timestamp"] = time.time()
        metrics_history.append(metrics)

        # Print summary
        elapsed = int(time.time() - start_time)
        print(f"\n📊 Metrics at {elapsed}s:")
        print(f"  Symbiosis: {manager.symbiosis_score:.2f}")

        for name in ["awareness", "emotional_decay", "memory_consolidation", "curiosity"]:
            if name in metrics:
                loop_metrics = metrics[name]
                print(
                    f"  {name}: cycles={loop_metrics.get('cycles', 0)}, "
                    f"errors={loop_metrics.get('errors', 0)}, "
                    f"p95={loop_metrics.get('p95_latency_ms', 0):.1f}ms"
                )

    # Stop loops
    await manager.stop()

    # Final report
    print("\n" + "=" * 50)
    print("📈 FINAL REPORT")
    print("=" * 50)

    # Calculate totals
    total_cycles = sum(manager.loops[name].cycles for name in manager.loops if hasattr(manager.loops[name], "cycles"))
    total_errors = sum(
        manager.loops[name].total_errors for name in manager.loops if hasattr(manager.loops[name], "total_errors")
    )

    print(f"Total cycles: {total_cycles}")
    print(f"Total errors: {total_errors}")
    print(f"Final symbiosis: {manager.symbiosis_score:.2f}")
    print(f"Duration: {duration}s")

    # Save metrics
    with open("test_metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2, default=str)
    print("\n💾 Metrics saved to test_metrics.json")

    # Check memory consolidation
    if "memory_consolidation" in manager.loops:
        memory_fed = manager.loops["memory_consolidation"].memory_federation
        if hasattr(memory_fed, "get_stats"):
            stats = await memory_fed.get_stats()
            print(f"📚 Memory stats: {stats}")

    success = total_errors == 0
    print(f"\n{'✅ TEST PASSED' if success else '❌ TEST FAILED'}")

    return success


if __name__ == "__main__":
    # Parse duration from command line
    duration = 60
    if len(sys.argv) > 1:
        try:
            duration = int(sys.argv[1])
        except:
            pass

    success = asyncio.run(run_integration_test(duration))
    exit(0 if success else 1)
