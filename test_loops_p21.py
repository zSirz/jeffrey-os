#!/usr/bin/env python3
"""
Test des boucles autonomes Phase 2.1 Ultimate
Démonstration du système vivant et intelligent
"""

import asyncio
import logging
import os
import sys
import time

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_loops")


async def test_loops_system():
    """Test principal du système de boucles"""

    # Import des composants
    from jeffrey.core.loops import LoopManager
    from jeffrey.core.neuralbus import NeuralBus

    print("\n" + "=" * 60)
    print("🚀 Jeffrey OS - Phase 2.1 Ultimate Loops Test")
    print("=" * 60 + "\n")

    # Créer un bus minimal pour les événements
    bus = NeuralBus()

    # Mock des dépendances
    class MockCognitiveCore:
        def get_state(self):
            return {
                "recent_memories": ["test"] * 10,
                "current_emotion": "curious",
                "active_modules": ["memory", "emotion"],
                "messages_processed": 3,
            }

        def get_available_modules(self):
            return ["memory", "emotion", "learning", "consciousness"]

    class MockEmotionOrchestrator:
        global_state = {"pleasure": 0.6, "arousal": 0.4, "dominance": 0.5}

    class MockMemoryFederation:
        seen_hashes = set()

        async def recall_from_all(self, query, max_results):
            # Simuler quelques souvenirs
            memories = []
            for i in range(10):
                memories.append(
                    {
                        "text": f"Memory {i}",
                        "timestamp": time.time() - (i * 60),
                        "metadata": {"importance": 0.5},
                        "user_id": "test_user",
                        "role": "assistant",
                    }
                )
            return memories

    # Créer le gestionnaire de boucles
    logger.info("Creating Loop Manager...")
    loop_manager = LoopManager(
        cognitive_core=MockCognitiveCore(),
        emotion_orchestrator=MockEmotionOrchestrator(),
        memory_federation=MockMemoryFederation(),
        bus=bus,
        mode_getter=lambda: "normal",
        latency_budget_ok=lambda: True,
    )

    # Démarrer les boucles
    logger.info("Starting all loops...")
    await loop_manager.start()

    # Attendre un peu pour voir les boucles fonctionner
    print("\n⏳ Letting loops run for 30 seconds...\n")

    for i in range(6):
        await asyncio.sleep(5)

        # Afficher le statut
        status = loop_manager.get_status()
        print(f"\n--- Status at T+{(i + 1) * 5}s ---")
        print(f"Symbiosis Score: {status['symbiosis_score']:.2%}")

        for loop_name, loop_status in status["loops"].items():
            print(f"\n{loop_name.upper()}:")
            print(f"  Cycles: {loop_status['cycles']}")
            print(f"  P95 Latency: {loop_status['p95_latency_ms']:.1f}ms")

            if loop_name == "awareness":
                print(f"  Awareness Level: {loop_status['awareness_level']:.2%}")
                print(f"  Thinking Mode: {loop_status['thinking_mode']}")
            elif loop_name == "emotional_decay":
                if "dominant_emotion" in loop_status:
                    print(f"  Emotion: {loop_status['dominant_emotion']}")
                pad = loop_status.get("pad_state", {})
                if pad:
                    print(
                        f"  PAD: P={pad.get('pleasure', 0):.2f}, A={pad.get('arousal', 0):.2f}, D={pad.get('dominance', 0):.2f}"
                    )
            elif loop_name == "curiosity":
                print(f"  Curiosity: {loop_status['curiosity_level']:.2%}")
                print(f"  Questions: {loop_status['questions_pending']}")
                print(f"  Insights: {loop_status['insights_gathered']}")
            elif loop_name == "memory_consolidation":
                print(f"  Processed: {loop_status['memories_processed']}")
                print(f"  Pruned: {loop_status['memories_pruned']}")

    # Tester l'injection d'émotion
    print("\n\n💉 Injecting emotional stimulus...")
    await loop_manager.inject_emotion(pleasure=0.3, arousal=0.7, dominance=-0.2)

    # Attendre pour voir la réaction
    await asyncio.sleep(10)

    # Afficher l'état émotionnel
    emotional_state = loop_manager.get_emotional_state()
    print("\n🎭 Emotional state after injection:")
    pad = emotional_state.get("pad_state", {})
    print(f"  Pleasure: {pad.get('pleasure', 0):.2f}")
    print(f"  Arousal: {pad.get('arousal', 0):.2f}")
    print(f"  Dominance: {pad.get('dominance', 0):.2f}")

    # Ajouter une question curieuse
    print("\n\n🤔 Adding curious question...")
    await loop_manager.add_question("What is the nature of consciousness in autonomous systems?")

    # Attendre pour voir l'exploration
    await asyncio.sleep(35)

    # Récupérer les insights
    insights = await loop_manager.get_insights(3)
    if insights:
        print("\n💡 Recent insights:")
        for insight in insights:
            print(f"  Q: {insight['question']}")
            if insight.get("conclusion"):
                print(f"  A: {insight['conclusion']}")
            print(f"  Confidence: {insight.get('confidence', 0.5):.2%}")
            print()

    # Arrêter les boucles
    print("\n🛑 Stopping loops...")
    await loop_manager.stop()

    # Statut final
    final_status = loop_manager.get_status()
    print("\n📊 Final Statistics:")
    print(f"Final Symbiosis Score: {final_status['symbiosis_score']:.2%}")

    total_cycles = sum(s["cycles"] for s in final_status["loops"].values())
    print(f"Total Cycles: {total_cycles}")

    awareness_level = loop_manager.get_awareness_level()
    print(f"Final Awareness: {awareness_level:.2%}")

    print("\n✨ Test completed successfully!")
    print("=" * 60)


async def main():
    """Point d'entrée principal"""
    try:
        await test_loops_system()
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    # Lancer le test
    asyncio.run(main())
