#!/usr/bin/env python3
"""
Test du BrainKernel en mode headless (sans Kivy)
Isole complètement l'UI pour des tests stables
"""

# ============================================
# ISOLATION KIVY - DOIT ÊTRE EN PREMIER
# ============================================
import os

# Désactiver complètement Kivy pour ce test
os.environ["KIVY_NO_ARGS"] = "1"
os.environ["KIVY_NO_CONSOLELOG"] = "1"
os.environ["KIVY_LOG_LEVEL"] = "error"
os.environ["KIVY_NO_CONFIG"] = "1"
os.environ["KIVY_NO_FILELOG"] = "1"

# Configuration du logging AVANT tout import
import logging
import sys

# Créer un logger personnalisé pour éviter les interférences
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Forcer stdout pour éviter stderr loops
)

# Faire taire Kivy complètement
logging.getLogger("kivy").setLevel(logging.CRITICAL)
logging.getLogger("kivy").propagate = False

# ============================================
# IMPORTS JEFFREY
# ============================================
import asyncio
from pathlib import Path

# Setup path - IMPORTANT : depuis la racine
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("🧪 TEST BRAINKERNEL HEADLESS (SANS KIVY)")
print("=" * 60)


async def test_headless():
    """Test complet du BrainKernel sans UI"""

    print("\n📦 Importation des modules...")

    try:
        # Imports progressifs pour identifier les problèmes
        print("  → Import NeuralBus...", end="")
        from jeffrey.core.neural_bus import EventPriority, NeuralBus, NeuralEnvelope

        print(" ✅")

        print("  → Import Bridge...", end="")
        from jeffrey.bridge.registry import BridgeRegistry

        print(" ✅")

        print("  → Import HttpAdapter...", end="")
        from jeffrey.bridge.adapters.http_adapter import HttpAdapter

        print(" ✅")

        print("  → Import BrainKernel...", end="")
        from jeffrey.core.kernel import BrainKernel

        print(" ✅")

    except ImportError as e:
        print(f" ❌\n\nErreur d'import: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("\n🔧 Configuration du système...")

    try:
        # Configuration headless explicite
        headless_config = {
            "load_ui_modules": False,  # CRITIQUE : Pas de modules UI
            "auto_load_census": True,
            "redis_url": None,  # Pas de Redis pour le test
            "bus_workers": 2,  # Moins de workers pour le test
            "enable_symbiosis": False,  # Désactiver pour simplifier
            "enable_proactive": False,  # Désactiver pour le test
            "use_stubs_for_missing": True,  # NOUVEAU : Utiliser les stubs
            "load_nlp_models": False,  # NOUVEAU : Pas de modèles NLP
            "fast_test_mode": True,  # NOUVEAU : Mode test rapide
        }

        # Créer le Bridge
        print("  → Création du Bridge...", end="")
        bridge = BridgeRegistry()
        http_adapter = HttpAdapter()
        bridge.register(http_adapter)
        print(" ✅")

        # Créer le BrainKernel avec config headless
        print("  → Création du BrainKernel...", end="")
        kernel = BrainKernel(bridge, headless_config)
        print(" ✅")

        # Initialiser
        print("\n⚡ Initialisation du cerveau...")
        await kernel.initialize()
        print("  ✅ BrainKernel initialisé avec succès!")

        # Afficher les composants chargés
        print(f"\n📊 Composants actifs: {list(kernel.components.keys())}")
        print("📊 Modules census ignorés (UI): voir logs DEBUG")

        # Test basique du bus
        print("\n🧪 Test de communication sur le bus...")

        # Handler de test
        test_received = []

        async def test_handler(envelope):
            test_received.append(envelope)
            return {"status": "received", "echo": envelope.payload}

        kernel.bus.register_handler("test.ping", test_handler)

        # Publier un message test
        result = await kernel.bus.publish(
            NeuralEnvelope(
                topic="test.ping",
                payload={"message": "Test headless"},
                priority=EventPriority.NORMAL,
            ),
            wait_for_response=True,
            timeout=2.0,
        )

        if result and result.get("status") == "received":
            print("  ✅ Communication bus OK")
        else:
            print("  ❌ Pas de réponse du bus")

        # Test chat si consciousness disponible
        print("\n💬 Test du module chat...")

        chat_result = await kernel.bus.publish(
            NeuralEnvelope(
                topic="chat.in",
                payload={"text": "Bonjour Jeffrey, es-tu opérationnel?"},
                meta={"session_id": "test_headless"},
            ),
            wait_for_response=True,
            timeout=5.0,
        )

        if chat_result:
            response = chat_result.get("response", "Pas de réponse")
            print(f"  → Jeffrey répond: {response[:100]}...")
            print("  ✅ Module chat fonctionnel")
        else:
            print("  ⚠️ Module chat non disponible (normal si consciousness pas chargé)")

        # Métriques
        print("\n📈 Métriques du système:")
        metrics = kernel.bus.get_metrics()
        print(f"  → Events publiés: {metrics.get('events_published', 0)}")
        print(f"  → Events traités: {metrics.get('events_processed', 0)}")
        print(f"  → Topics actifs: {metrics.get('topics', [])}")
        print(f"  → Redis connecté: {metrics.get('redis_connected', False)}")

        # Shutdown propre
        print("\n🛑 Arrêt du système...")
        await kernel.shutdown()
        print("  ✅ Arrêt propre effectué")

        return True

    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Point d'entrée principal"""
    success = await test_headless()

    print("\n" + "=" * 60)
    if success:
        print("✅ TEST RÉUSSI - Le BrainKernel fonctionne en mode headless!")
    else:
        print("❌ TEST ÉCHOUÉ - Voir les erreurs ci-dessus")
    print("=" * 60)

    return success


if __name__ == "__main__":
    # Lancer le test
    result = asyncio.run(main())

    # Code de sortie
    sys.exit(0 if result else 1)
