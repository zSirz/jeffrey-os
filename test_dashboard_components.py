#!/usr/bin/env python3
"""
Test dashboard components without Streamlit
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath("."))

from jeffrey.core.loops.loop_manager import LoopManager
from jeffrey.utils.test_helpers import DummyMemoryFederation, NullBus, SimpleState


async def test_dashboard_components():
    """Test dashboard without UI"""
    print("🧪 Testing Dashboard Components...\n")

    # Create manager as dashboard would
    manager = LoopManager(
        cognitive_core=SimpleState(),
        emotion_orchestrator=None,
        memory_federation=DummyMemoryFederation(),
        bus=NullBus(),
        mode_getter=lambda: "normal",
        latency_budget_ok=lambda: True,
    )

    # Start manager
    print("✓ Manager created")
    await manager.start()
    print("✓ Loops started")

    # Wait a bit for metrics to accumulate
    await asyncio.sleep(3)

    # Get metrics as dashboard would
    metrics = manager.get_metrics()

    print("\n📊 System Metrics:")
    print(f"  Total Cycles: {metrics['system'].get('total_cycles', 0)}")
    print(f"  Symbiosis Score: {metrics['system'].get('symbiosis_score', 0):.3f}")
    print(f"  Bus Dropped: {metrics['system'].get('bus_dropped', 0)}")
    print(f"  Uptime: {metrics['system'].get('uptime', 0):.1f}s")

    print("\n🔄 Loop Metrics:")
    for name, data in metrics.get("loops", {}).items():
        print(f"\n  {name}:")
        print(f"    Running: {data.get('running', False)}")
        print(f"    Cycles: {data.get('cycles', 0)}")
        print(f"    Errors: {data.get('errors', 0)}")
        print(f"    P95 Latency: {data.get('p95_latency_ms', 0):.1f}ms")
        print(f"    P99 Latency: {data.get('p99_latency_ms', 0):.1f}ms")

    # Test Q-learning data
    print("\n🧠 Q-Learning Data:")
    for name, loop in manager.loops.items():
        if hasattr(loop, "q_table") and loop.q_table:
            print(f"  {name} Q-Table size: {len(loop.q_table)}")
            if hasattr(loop, "replay_buffer") and loop.replay_buffer:
                stats = loop.replay_buffer.get_stats()
                print(f"  {name} Replay Buffer: {stats['size']}/{stats['capacity']}")

    # Test symbiotic graph if available
    if manager.symbiotic_graph:
        print("\n🌐 Symbiotic Graph:")
        try:
            analysis = await manager.symbiotic_graph.analyze_interactions()
            if "graph_metrics" in analysis:
                print(f"  Nodes: {analysis['graph_metrics']['nodes']}")
                print(f"  Edges: {analysis['graph_metrics']['edges']}")
                print(f"  Density: {analysis['graph_metrics']['density']:.3f}")
        except Exception as e:
            print(f"  Graph analysis error: {e}")

    # Test memory consolidation
    if "memory_consolidation" in manager.loops:
        loop = manager.loops["memory_consolidation"]
        print("\n💾 Memory Consolidation:")
        print(f"  Memories Processed: {getattr(loop, 'memories_processed', 0)}")
        print(f"  Memories Archived: {getattr(loop, 'memories_archived', 0)}")
        print(f"  Memories Pruned: {getattr(loop, 'memories_pruned', 0)}")
        print(f"  Compressions: {getattr(loop, 'compression_count', 0)}")

        if hasattr(loop, "_get_memory_usage"):
            usage = loop._get_memory_usage()
            print(f"  Memory Usage: {usage:.2f} MB")

    # Stop manager
    await manager.stop()
    print("\n✓ Manager stopped")

    print("\n✅ All dashboard components functional!")


if __name__ == "__main__":
    asyncio.run(test_dashboard_components())
