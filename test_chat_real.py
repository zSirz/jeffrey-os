#!/usr/bin/env python3
"""
Test de chat réel avec Jeffrey
Vérifie que le cerveau peut vraiment converser
"""

import os

os.environ["KIVY_NO_ARGS"] = "1"
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "error"

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from jeffrey.bridge.adapters.http_adapter import HttpAdapter

from jeffrey.bridge.registry import BridgeRegistry
from jeffrey.core.kernel import BrainKernel
from jeffrey.core.neural_bus import NeuralEnvelope


async def test_real_chat():
    """Test une vraie conversation avec Jeffrey"""

    print("\n" + "=" * 60)
    print("🧠 TEST DE CHAT RÉEL AVEC JEFFREY")
    print("=" * 60 + "\n")

    try:
        # Configuration optimisée pour test rapide
        config = {
            "load_ui_modules": False,
            "redis_url": None,
            "enable_proactive": False,
            "enable_symbiosis": False,
            "load_nlp_models": False,
            "fast_test_mode": True,
            "use_stubs_for_missing": True,  # Utiliser les stubs
        }

        # Setup
        print("🔧 Initialisation du cerveau...")
        bridge = BridgeRegistry()
        bridge.register(HttpAdapter())

        kernel = BrainKernel(bridge, config)
        await kernel.initialize()
        print("✅ Cerveau initialisé\n")

        # Conversation de test
        test_conversation = [
            "Bonjour Jeffrey, comment vas-tu ?",
            "Quel est ton rôle principal ?",
            "Peux-tu m'aider avec Python ?",
            "Raconte-moi une blague",
            "Merci Jeffrey, au revoir !",
        ]

        session_id = f"chat_test_{datetime.now().timestamp()}"

        print("💬 DÉBUT DE LA CONVERSATION")
        print("-" * 40)

        for message in test_conversation:
            print(f"\n👤 Vous: {message}")

            # Envoyer le message
            result = await kernel.bus.publish(
                NeuralEnvelope(topic="chat.in", payload={"text": message}, meta={"session_id": session_id}),
                wait_for_response=True,
                timeout=10,
            )

            # Afficher la réponse
            if result and "response" in result:
                response = result["response"]
                emotion = result.get("emotion", {})
                if isinstance(emotion, dict):
                    emotion_str = emotion.get("dominant", "neutral")
                else:
                    emotion_str = "neutral"
                print(f"🤖 Jeffrey ({emotion_str}): {response}")
            else:
                print("❌ Pas de réponse reçue")

        print("\n" + "-" * 40)
        print("💬 FIN DE LA CONVERSATION\n")

        # Métriques
        metrics = kernel.bus.get_metrics()
        print("📊 MÉTRIQUES:")
        print(f"  • Messages envoyés: {metrics.get('events_published', 0)}")
        print(f"  • Messages traités: {metrics.get('events_processed', 0)}")

        # Shutdown
        await kernel.shutdown()
        print("\n✅ Test terminé avec succès!")
        return True

    except Exception as e:
        logger.error(f"Erreur pendant le test: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_real_chat())
    sys.exit(0 if success else 1)
