"""
Test Phase 2.2 features (Graph ML + Clustering)
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jeffrey.core.loops.loop_manager import LoopManager
from jeffrey.utils.test_helpers import DummyMemoryFederation, NullBus, SimpleState


async def test_phase22():
    """Test Graph ML and Clustering features"""
    print("=" * 50)
    print("🚀 TESTING PHASE 2.2 FEATURES")
    print("=" * 50)

    # Setup
    event_bus = NullBus()
    state_manager = SimpleState()
    config = {
        "autonomous_loops": {
            "enabled": True,
            "awareness": {"enabled": True, "interval": 1.0},
            "emotional_decay": {"enabled": True, "interval": 2.0},
            "memory_consolidation": {
                "enabled": True,
                "interval": 5.0,
                "use_ml_clustering": True,  # Enable ML
                "metadata_only": False,
            },
            "curiosity": {"enabled": True, "interval": 3.0},
        },
        "resource_management": {"cpu_threshold": 80, "memory_threshold": 85},
    }

    # Create simple mocks for dependencies
    class MockCognitiveCore:
        def get_state(self):
            return {"mode": "idle", "latency_ms": 50}

    class MockEmotionOrchestrator:
        global_state = {"pleasure": 0.6, "arousal": 0.4, "dominance": 0.5}

    manager = LoopManager(
        cognitive_core=MockCognitiveCore(),
        emotion_orchestrator=MockEmotionOrchestrator(),
        memory_federation=None,  # Will be added below
        bus=event_bus,
        mode_getter=lambda: "normal",
        latency_budget_ok=lambda: True,
    )

    # Add memory federation
    if "memory_consolidation" in manager.loops:
        manager.loops["memory_consolidation"].memory_federation = DummyMemoryFederation()
        manager.loops["memory_consolidation"].loop_manager = manager  # Link for clusterer

    await manager.start()

    # Test 1: Graph Analysis
    print("\n📊 Testing Symbiotic Graph...")
    if manager.symbiotic_graph:
        print(f"  Graph initialized: {manager.symbiotic_graph.graph}")
        await asyncio.sleep(5)  # Let loops run

        analysis = await manager.symbiotic_graph.analyze_interactions()
        print(f"  Analysis result: {analysis}")

        if "error" not in analysis:
            metrics = analysis.get("graph_metrics", {})
            print(f"  Graph nodes: {metrics.get('nodes', 0)}")
            print(f"  Graph edges: {metrics.get('edges', 0)}")
            print(f"  Graph density: {metrics.get('density', 0):.3f}")
            print(f"  Synergies found: {len(analysis.get('synergies', []))}")

            for rec in analysis.get("recommendations", [])[:3]:
                print(f"  - {rec}")
        else:
            print(f"  ⚠️ {analysis['error']}")
    else:
        print("  ⚠️ Graph not available (install networkx)")

    # Test 2: ML Clustering
    print("\n🧠 Testing ML Clustering...")
    if manager.memory_clusterer:
        # Create test memories
        test_memories = [
            {"text": "Learning about neural networks and deep learning", "timestamp": 1000},
            {"text": "Studying machine learning algorithms and models", "timestamp": 1001},
            {"text": "Making breakfast with eggs and toast", "timestamp": 1002},
            {"text": "Cooking dinner with pasta and vegetables", "timestamp": 1003},
            {"text": "Understanding artificial intelligence concepts", "timestamp": 1004},
            {"text": "Preparing lunch sandwich with cheese", "timestamp": 1005},
        ]

        clusters = await manager.memory_clusterer.cluster_memories(test_memories)
        print(f"  Clustered {len(test_memories)} memories into {len(clusters)} groups")

        for i, cluster in enumerate(clusters):
            summary = manager.memory_clusterer.get_cluster_summary(cluster)
            themes = summary.get("themes", [])[:3]
            print(f"  Cluster {i + 1}: {themes} ({len(cluster)} items)")
    else:
        print("  ⚠️ ML Clustering not available (install scikit-learn sentence-transformers)")

    # Test 3: Similarity Search
    if manager.memory_clusterer and test_memories:
        print("\n🔍 Testing Similarity Search...")
        query = "artificial intelligence and machine learning"
        similar = await manager.memory_clusterer.find_similar(query, test_memories, top_k=3)
        print(f"  Query: '{query}'")
        print("  Top 3 similar memories:")
        for i, mem in enumerate(similar):
            text_preview = mem.get("text", "")[:60]
            print(f"    {i + 1}. {text_preview}...")

    # Let it run for a bit
    print("\n⏳ Running loops for 15 seconds...")
    await asyncio.sleep(15)

    # Check final metrics
    metrics = manager.get_all_metrics()
    system = metrics.get("system", {})

    print("\n📈 Final Metrics:")
    print(f"  Symbiosis score: {system.get('symbiosis_score', 0):.2f}")

    # Show loop metrics
    for name in ["awareness", "emotional_decay", "memory_consolidation", "curiosity"]:
        if name in metrics:
            loop = metrics[name]
            cycles = loop.get("cycles", 0)
            errors = loop.get("errors", 0)
            p95 = loop.get("p95_latency_ms", 0)
            print(f"  {name}: {cycles} cycles, {errors} errors, {p95:.1f}ms p95")

    await manager.stop()

    print("\n✅ Phase 2.2 tests complete!")
    print("\n📦 Optional dependencies to install:")
    print("  pip install networkx           # For graph analysis")
    print("  pip install scikit-learn       # For clustering")
    print("  pip install sentence-transformers  # For embeddings")


if __name__ == "__main__":
    asyncio.run(test_phase22())
