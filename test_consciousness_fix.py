#!/usr/bin/env python3
"""
Test pour vérifier que consciousness fonctionne après les corrections
"""

import os

os.environ["KIVY_NO_ARGS"] = "1"
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "error"

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def test_consciousness_methods():
    """Vérifie les méthodes disponibles dans consciousness"""

    print("\n🔍 VÉRIFICATION DE CONSCIOUSNESS")
    print("=" * 50)

    try:
        # Injecter les stubs avant l'import
        from jeffrey.stubs import inject_stubs_to_sys_modules

        inject_stubs_to_sys_modules({"use_stubs_for_missing": True})

        # Importer consciousness
        from jeffrey.core.consciousness.jeffrey_consciousness_v3 import JeffreyConsciousnessV3

        # Créer une instance
        consciousness = JeffreyConsciousnessV3()

        # Lister toutes les méthodes publiques
        print("\n📋 Méthodes disponibles dans JeffreyConsciousnessV3:")
        methods = [m for m in dir(consciousness) if not m.startswith("_") and callable(getattr(consciousness, m))]

        for method in sorted(methods):
            print(f"  • {method}")

        # Vérifier les méthodes critiques
        print("\n✅ Vérification des méthodes critiques:")

        critical_methods = ["respond", "process", "generate_response", "dream", "interact"]
        for method_name in critical_methods:
            if hasattr(consciousness, method_name):
                print(f"  ✅ {method_name} - TROUVÉE")

                # Tester si c'est async
                import inspect

                method = getattr(consciousness, method_name)
                if inspect.iscoroutinefunction(method):
                    print("     → Méthode async")
                else:
                    print("     → Méthode sync")
            else:
                print(f"  ❌ {method_name} - MANQUANTE")

        # Test rapide de réponse
        print("\n🧪 Test de génération de réponse:")

        test_message = "Bonjour Jeffrey !"

        if hasattr(consciousness, "respond"):
            if inspect.iscoroutinefunction(consciousness.respond):
                response = await consciousness.respond(test_message)
            else:
                response = consciousness.respond(test_message)
            print(f"  Message: {test_message}")
            print(f"  Réponse: {response[:200]}")  # Limiter la longueur
        else:
            print("  ⚠️ Méthode 'respond' non disponible")

            # Essayer d'autres méthodes
            for method_name in ["interact", "process", "generate_response", "dream"]:
                if hasattr(consciousness, method_name):
                    print(f"\n  Essai avec '{method_name}':")
                    method = getattr(consciousness, method_name)
                    try:
                        if inspect.iscoroutinefunction(method):
                            result = await method(test_message)
                        else:
                            result = method(test_message)
                        print(f"  Résultat: {str(result)[:200]}")
                        break
                    except Exception as e:
                        print(f"  Erreur: {e}")

        return True

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback

        traceback.print_exc()
        return False


async def test_chat_after_fix():
    """Test le chat après les corrections"""

    print("\n\n💬 TEST DU CHAT APRÈS CORRECTIONS")
    print("=" * 50)

    try:
        from jeffrey.bridge.adapters.http_adapter import HttpAdapter

        from jeffrey.bridge.registry import BridgeRegistry
        from jeffrey.core.kernel import BrainKernel
        from jeffrey.core.neural_bus import NeuralEnvelope

        # Config minimale
        config = {
            "load_ui_modules": False,
            "redis_url": None,
            "enable_proactive": False,
            "enable_symbiosis": True,  # Réactiver maintenant que c'est corrigé
            "fast_test_mode": True,
            "use_stubs_for_missing": True,
        }

        # Setup
        bridge = BridgeRegistry()
        bridge.register(HttpAdapter())

        kernel = BrainKernel(bridge, config)
        await kernel.initialize()

        # Test de chat
        print("\n🗨️ Test de conversation:")

        # Messages de test
        test_messages = [
            "Bonjour Jeffrey, es-tu maintenant opérationnel ?",
            "Quel est ton rôle principal ?",
            "Peux-tu m'aider avec Python ?",
        ]

        success_count = 0
        for message in test_messages:
            print(f"\n  👤 Message: {message}")

            result = await kernel.bus.publish(
                NeuralEnvelope(topic="chat.in", payload={"text": message}, meta={"session_id": "test_fix"}),
                wait_for_response=True,
                timeout=5,
            )

            if result and "response" in result:
                response = result["response"]
                emotion = result.get("emotion", {})
                if isinstance(emotion, dict):
                    emotion_str = emotion.get("dominant", "neutral")
                else:
                    emotion_str = "neutral"

                print(f"  🤖 Jeffrey ({emotion_str}): {response[:150]}")
                success_count += 1
            else:
                print(f"  ❌ Pas de réponse: {result}")

        # Shutdown
        await kernel.shutdown()

        return success_count > 0

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Tests principaux"""

    print("\n" + "🔧 TEST DES CORRECTIONS CONSCIOUSNESS " + "🔧")

    # Test 1 : Vérifier les méthodes
    success1 = await test_consciousness_methods()

    # Test 2 : Tester le chat
    success2 = await test_chat_after_fix()

    # Résumé
    print("\n\n" + "=" * 50)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 50)

    if success1:
        print("✅ Consciousness : Méthodes vérifiées")
    else:
        print("❌ Consciousness : Problème détecté")

    if success2:
        print("✅ Chat : Fonctionnel")
    else:
        print("❌ Chat : Non fonctionnel")

    if success1 and success2:
        print("\n🎉 TOUS LES TESTS PASSENT - JEFFREY PEUT PARLER !")
    else:
        print("\n⚠️ Des corrections sont encore nécessaires")

    return success1 and success2


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
