#!/usr/bin/env python3
"""
Smoke test pour l'architecture CERVEAU Jeffrey
"""

import asyncio

from jeffrey.core.llm.autonomous_language_system import AutonomousLanguageSystem


async def smoke_test():
    """Test de fumée pour valider l'architecture"""
    print("🧪 Starting Jeffrey Brain Smoke Test\n")

    try:
        # 1. Initialiser le système
        print("1️⃣ Initializing Autonomous Language System...")
        als = AutonomousLanguageSystem(fallback_llm=None)
        await als.initialize()
        print("✅ System initialized\n")

        # 2. Test simple sans LLM externe
        print("2️⃣ Testing autonomous generation...")
        query = {
            "content": "What is emergence?",
            "type": "question",
            "emotional_state": {"curiosity": 0.9},
        }

        result = await als.process(query, force_external=False)

        print(f"✅ Success: {result['success']}")
        print(f"📊 Routing: {result['routing']}")
        print(f"🎯 Model: {result.get('model', 'unknown')}")
        print(f"⭐ Quality score: {result.get('quality_score', 0):.2f}")
        print(f"💬 Response: {result.get('response', 'No response')[:200]}...")
        print()

        # 3. Test avec différents contextes
        print("3️⃣ Testing different contexts...")
        test_queries = [
            {"content": "Tell me about consciousness", "type": "exploration"},
            {"content": "How do patterns emerge from chaos?", "type": "question"},
            {"content": "I wonder about the nature of awareness", "type": "reflection"},
        ]

        for i, test_query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {test_query['content'][:50]}...")
            test_query["emotional_state"] = {"curiosity": 0.7, "wonder": 0.5}

            result = await als.process(test_query)
            print(f"  → Routing: {result['routing']}")
            print(f"  → Quality: {result.get('quality_score', 0):.2f}")

        # 4. Vérifier les statistiques
        print("\n4️⃣ System Statistics:")
        stats = await als.get_stats()

        print(f"  → Autonomy level: {stats['autonomy_level']:.3f}")
        print(f"  → Cache size: {stats['cache_size']}")

        if "executive_stats" in stats:
            exec_stats = stats["executive_stats"]
            print(f"  → Total decisions: {exec_stats['total_decisions']}")
            print(f"  → Mean reward: {exec_stats['mean_reward']:.3f}")

        if "memory_stats" in stats:
            mem_stats = stats["memory_stats"]
            print(f"  → Episodes stored: {mem_stats['episodic_memory_size']}")
            print(f"  → Patterns: {mem_stats['semantic_patterns']}")

        # 5. Vérifier la création des fichiers
        print("\n5️⃣ Checking persistence...")
        from pathlib import Path

        files_to_check = ["data/episodic.db", "data/faiss.index", "data/lsh.pkl"]

        for file_path in files_to_check:
            exists = Path(file_path).exists()
            status = "✅" if exists else "❌"
            print(f"  {status} {file_path}")

        # Cleanup
        await als.shutdown()
        print("\n✅ Smoke test completed successfully!")

    except Exception as e:
        print(f"\n❌ Smoke test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = asyncio.run(smoke_test())
    exit(0 if success else 1)
