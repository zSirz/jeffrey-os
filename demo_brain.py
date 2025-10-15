#!/usr/bin/env python3
"""
DÉMONSTRATION DU CERVEAU JEFFREY
Montre le flux cognitif complet: perception → émotion → mémoire → conscience → action
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jeffrey_brain import JeffreyBrain


class BrainDemo:
    """Démo interactive du cerveau Jeffrey"""

    def __init__(self):
        self.brain = None

    async def setup(self):
        """Configure le cerveau avec logs minimaux"""
        # Logs essentiels seulement
        logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
        # Activer certains logs importants
        logging.getLogger("root").setLevel(logging.INFO)

        self.brain = JeffreyBrain()
        await self.brain.boot()

    async def demo_sequence(self):
        """Séquence de démonstration"""

        print("\n" + "=" * 80)
        print("🧠 DÉMONSTRATION DU CERVEAU COGNITIF JEFFREY")
        print("=" * 80)

        demos = [
            {
                "title": "Test S1 - Réflexe Rapide (Salutation)",
                "input": "Hello Jeffrey!",
                "expected": "Route S1 (réflexe) → Réponse immédiate",
                "delay": 1,
            },
            {
                "title": "Test S2 - Question Complexe",
                "input": "What are your thoughts on consciousness?",
                "expected": "Route S2 (délibération) → Workspace → Orchestrateur",
                "delay": 2,
            },
            {
                "title": "Test Guardian - Sécurité PII",
                "input": "My SSN is 123-45-6789 and email is test@example.com",
                "expected": "Guardian détecte et masque les PII",
                "delay": 1,
            },
            {
                "title": "Test Emotion - Détection Affective",
                "input": "I'm feeling very happy today!",
                "expected": "Limbic system détecte joie → influence la réponse",
                "delay": 1,
            },
            {
                "title": "Test Mémoire - Rappel Contextuel",
                "input": "Do you remember what I said about being happy?",
                "expected": "Recall mémoire → contexte récupéré",
                "delay": 2,
            },
        ]

        for i, demo in enumerate(demos, 1):
            print(f"\n📍 DEMO {i}/{len(demos)}: {demo['title']}")
            print(f"   Input: '{demo['input']}'")
            print(f"   Expected: {demo['expected']}")
            print("-" * 40)

            # Envoyer l'input
            await self.brain.process_input(demo["input"], f"demo_user_{i}")

            # Attendre le traitement
            await asyncio.sleep(demo["delay"])

        print("\n" + "=" * 80)
        print("✅ DÉMONSTRATION TERMINÉE")
        print("=" * 80)

    async def interactive_mode(self):
        """Mode interactif"""
        print("\n" + "=" * 80)
        print("💬 MODE INTERACTIF - Tapez 'quit' pour sortir")
        print("=" * 80)

        while True:
            try:
                user_input = await asyncio.to_thread(input, "\n👤 You> ")

                if user_input.lower() in ["quit", "exit", "q"]:
                    print("👋 Au revoir!")
                    break

                print("🧠 Processing...")
                await self.brain.process_input(user_input, "interactive")
                await asyncio.sleep(1)

            except (EOFError, KeyboardInterrupt):
                print("\n👋 Arrêt...")
                break

    async def run(self):
        """Lance la démo complète"""
        await self.setup()

        # Séquence de démo
        await self.demo_sequence()

        # Mode interactif optionnel
        print("\n🤔 Voulez-vous continuer en mode interactif? (y/n)")
        try:
            choice = await asyncio.to_thread(input, "> ")
            if choice.lower() in ["y", "yes", "oui"]:
                await self.interactive_mode()
        except (EOFError, KeyboardInterrupt):
            pass

        print("\n🧠 Cerveau Jeffrey arrêté proprement.")


async def main():
    """Point d'entrée"""
    demo = BrainDemo()
    await demo.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚡ Interruption utilisateur")
        sys.exit(0)
