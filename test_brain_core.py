#!/usr/bin/env python3
"""
Core smoke test pour l'architecture CERVEAU Jeffrey
Test uniquement les nouveaux modules sans dépendances cassées
"""

import asyncio
from pathlib import Path

# Tester uniquement les nouveaux modules
from jeffrey.core.brain.executive_cortex import ExecutiveCortex
from jeffrey.core.brain.quality_critic import QualityCritic
from jeffrey.core.memory.triple_memory import TripleMemorySystem


async def test_core_modules():
    """Test des modules core du cerveau"""
    print("🧠 Testing Jeffrey Brain Core Modules\n")

    try:
        # 1. Test Executive Cortex
        print("1️⃣ Testing Executive Cortex (Bandit Contextuel)...")
        cortex = ExecutiveCortex()

        # Contexte de test
        context = {
            "intent_type": "question",
            "complexity": 0.5,
            "familiarity": 0.3,
            "domain": "general",
        }

        # Décision
        arm, metadata = await cortex.decide(context)
        print(f"  ✅ Decision: {arm}")
        print(f"  📊 Strategy: {metadata['strategy']}")

        # Reward
        await cortex.reward(arm, context, quality_score=0.8, latency_ms=100, success=True)
        stats = cortex.get_stats()
        print(f"  📈 Mean reward: {stats['mean_reward']:.3f}\n")

        # 2. Test Triple Memory
        print("2️⃣ Testing Triple Memory System...")
        memory = TripleMemorySystem()

        # Stocker un épisode
        episode_id = await memory.remember(
            text_in="What is consciousness?",
            text_out="Consciousness is the state of being aware.",
            intent="question",
            quality_score=0.75,
            metadata={"test": True},
        )
        print(f"  ✅ Episode stored: {episode_id[:8]}...")

        # Rechercher
        similar = await memory.recall_similar("awareness", k=2)
        print(f"  📚 Found {len(similar)} similar episodes")

        mem_stats = memory.get_stats()
        print(f"  💾 Working memory: {mem_stats['working_memory_size']} items")
        print(f"  📝 Episodic memory: {mem_stats['episodic_memory_size']} episodes\n")

        # 3. Test Quality Critic
        print("3️⃣ Testing Quality Critic...")
        critic = QualityCritic()

        # Évaluer une réponse
        response = "I find myself wondering about the nature of consciousness and its emergent properties."
        context = {
            "input": "What is consciousness?",
            "intent_type": "question",
            "emotional_state": {"curiosity": 0.8},
        }

        report = await critic.evaluate(response, context)
        print(f"  ⭐ Overall quality: {report.overall_quality:.2f}")
        print(f"  ✅ Acceptable: {report.is_acceptable}")
        print("  📊 Scores:")
        print(f"     - Coherence: {report.coherence_score:.2f}")
        print(f"     - Style: {report.style_score:.2f}")
        print(f"     - Safety: {report.safety_score:.2f}")
        print(f"     - Consistency: {report.consistency_score:.2f}")
        print(f"     - Helpfulness: {report.helpfulness_score:.2f}")

        if report.issues:
            print(f"  ⚠️  Issues: {', '.join(report.issues[:2])}")

        # 4. Vérifier la persistance
        print("\n4️⃣ Checking Persistence...")
        files = {
            "data/episodic.db": Path("data/episodic.db").exists(),
            "data/faiss.index": Path("data/faiss.index").exists(),
            "data/lsh.pkl": Path("data/lsh.pkl").exists(),
            "data/patterns/": Path("data/patterns/").exists(),
        }

        for path, exists in files.items():
            status = "✅" if exists else "❌"
            print(f"  {status} {path}")

        # Cleanup
        await memory.shutdown()

        print("\n✅ Core modules test completed successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_integration():
    """Test d'intégration basique"""
    print("\n🔗 Testing Basic Integration...\n")

    try:
        from jeffrey.core.memory.triple_memory import LocalEmbedder

        # Test embedder local
        embedder = LocalEmbedder(dim=384)
        embedding = embedder.encode("Test query about consciousness")
        print(f"✅ Local embedder working: vector dim={len(embedding)}")

        # Test bandit avec plusieurs décisions
        cortex = ExecutiveCortex(exploration_rate=0.2)

        print("\n🎰 Testing Bandit Algorithm (10 rounds):")
        arms_selected = {"cache": 0, "autonomous": 0, "llm": 0}

        for i in range(10):
            context = {
                "intent_type": "question" if i % 2 else "exploration",
                "complexity": i / 10,
                "familiarity": 1 - (i / 10),
                "domain": "test",
            }

            arm, _ = await cortex.decide(context)
            arms_selected[arm] += 1

            # Simuler reward variable
            quality = 0.9 if arm == "cache" else 0.7 if arm == "autonomous" else 0.5
            await cortex.reward(arm, context, quality, latency_ms=50 + i * 10, success=True)

        print(
            f"  Distribution: Cache={arms_selected['cache']}, "
            f"Autonomous={arms_selected['autonomous']}, LLM={arms_selected['llm']}"
        )

        stats = cortex.get_stats()
        print(f"  Mean reward: {stats['mean_reward']:.3f}")
        print(f"  Contexts tracked: {stats['contexts_tracked']}")

        print("\n✅ Integration test passed!")
        return True

    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print(" JEFFREY BRAIN ARCHITECTURE - CORE TEST ".center(60))
    print("=" * 60)

    # Run tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    success = loop.run_until_complete(test_core_modules())
    if success:
        success = loop.run_until_complete(test_integration())

    print("\n" + "=" * 60)
    if success:
        print(" ✅ ALL TESTS PASSED ".center(60))
    else:
        print(" ❌ SOME TESTS FAILED ".center(60))
    print("=" * 60)

    exit(0 if success else 1)
