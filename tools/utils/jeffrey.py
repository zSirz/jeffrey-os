#!/usr/bin/env python3
"""
Jeffrey V2.2 - Point d'entrée principal
=======================================
Système d'IA émotionnelle avec architecture événementielle.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Ajouter le dossier core au path
sys.path.insert(0, str(Path(__file__).parent / "core"))

# Imports des composants
from advanced_unified_memory import AdvancedUnifiedMemory
from agi_connector import initialize_agi
from event_bus import Event, EventPriority, EventType, event_bus
from ui_connector import ui_connector
from voice_connector import voice_connector

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class JeffreySystem:
    """
    Classe principale orchestrant tout le système Jeffrey V2.2.
    """

    def __init__(self):
        self.memory = None
        self.agi = None
        self.running = False
        self.start_time = None
        self.config = self._load_config()

        logger.info("🧠 Initialisation de Jeffrey V2.2...")

    def _load_config(self) -> dict[str, Any]:
        """Charger la configuration système."""
        config_path = Path(__file__).parent / "config" / "jeffrey_config.json"

        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)

        # Configuration par défaut
        return {
            "system": {"name": "Jeffrey", "version": "2.2", "language": "fr-FR", "debug_mode": False},
            "memory": {"cache_size": 1000, "consolidation_interval": 3600, "backup_interval": 1800},
            "agi": {"model": "default", "temperature": 0.7, "max_context": 4000},
            "voice": {"enabled": True, "provider": "elevenlabs", "voice_id": "jeffrey_voice"},
            "ui": {"theme": "adaptive", "animations_enabled": True},
        }

    async def initialize(self):
        """Initialiser tous les composants du système."""
        try:
            # 1. Initialiser la mémoire unifiée
            logger.info("📚 Initialisation de la mémoire unifiée...")
            memory_path = Path(__file__).parent.parent / "unified_memory_production"
            self.memory = AdvancedUnifiedMemory(
                base_path=memory_path, config={"cache_max_size": self.config["memory"]["cache_size"]}
            )

            # 2. Démarrer l'EventBus
            logger.info("🚌 Démarrage de l'EventBus...")
            await event_bus.start()

            # 3. Initialiser les composants asyncio des connecteurs
            logger.info("🔧 Initialisation des connecteurs...")
            await ui_connector.initialize_async_components()
            await voice_connector.initialize_async_components()

            # 4. Initialiser l'AGI
            logger.info("🤖 Connexion de l'AGI...")
            self.agi = await initialize_agi(self.memory)

            logger.info("🖼️ UI Connector prêt")
            logger.info("🎙️ Voice Connector prêt")

            # 5. Publier l'événement système prêt
            await event_bus.publish(
                Event(
                    type=EventType.SYSTEM_READY,
                    data={
                        "version": self.config["system"]["version"],
                        "components": ["memory", "eventbus", "agi", "ui", "voice"],
                        "timestamp": datetime.now().isoformat(),
                    },
                    priority=EventPriority.HIGH,
                    source="system",
                )
            )

            self.start_time = datetime.now()
            self.running = True

            logger.info("✅ Jeffrey V2.2 initialisé avec succès !")

            # Message de bienvenue
            await self._welcome_message()

        except Exception as e:
            logger.error(f"❌ Erreur d'initialisation: {e}")
            raise

    async def _welcome_message(self):
        """Message de bienvenue au démarrage."""
        welcome_text = (
            f"Bonjour ! Je suis {self.config['system']['name']} version {self.config['system']['version']}. "
            "Mon système neural est maintenant complètement connecté et je suis prêt à discuter avec toi !"
        )

        # Publier comme une réponse AGI
        await event_bus.publish(
            Event(
                type=EventType.AGI_RESPONSE,
                data={"response": welcome_text, "emotion": "joyful", "emotion_intensity": 0.8},
                priority=EventPriority.HIGH,
                source="system",
            )
        )

    async def run(self):
        """Boucle principale du système."""
        logger.info("🔄 Jeffrey est maintenant en ligne et à l'écoute...")

        # Démarrer la consolidation périodique de la mémoire
        consolidation_task = asyncio.create_task(self._periodic_memory_consolidation())

        # Démarrer le monitoring système
        monitoring_task = asyncio.create_task(self._system_monitoring())

        try:
            # Garder le système en vie
            while self.running:
                await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"❌ Erreur dans la boucle principale: {e}")
        finally:
            # Nettoyer
            consolidation_task.cancel()
            monitoring_task.cancel()

    async def _periodic_memory_consolidation(self):
        """Consolidation périodique de la mémoire."""
        interval = self.config["memory"]["consolidation_interval"]

        while self.running:
            await asyncio.sleep(interval)

            try:
                logger.info("🧹 Consolidation de la mémoire...")
                result = await self.memory.consolidate()
                logger.info(f"✅ Mémoire consolidée: {result}")

                # Publier l'événement
                await event_bus.publish(
                    Event(
                        type=EventType.MEMORY_UPDATE,
                        data={"action": "consolidation", "result": result},
                        source="memory_manager",
                    )
                )

            except Exception as e:
                logger.error(f"❌ Erreur consolidation mémoire: {e}")

    async def _system_monitoring(self):
        """Monitoring périodique du système."""
        while self.running:
            await asyncio.sleep(30)  # Check toutes les 30 secondes

            try:
                # Collecter les métriques
                metrics = {
                    "uptime": (datetime.now() - self.start_time).total_seconds(),
                    "event_bus": event_bus.get_metrics(),
                    "memory_stats": self.memory.get_stats(),
                    "ui_state": ui_connector.get_current_state(),
                    "voice_queue": voice_connector.get_queue_size(),
                }

                # Publier le health check
                await event_bus.publish(Event(type=EventType.HEALTH_CHECK, data=metrics, source="system_monitor"))

                # Logger si debug activé
                if self.config["system"]["debug_mode"]:
                    logger.debug(f"📊 Métriques système: {metrics}")

            except Exception as e:
                logger.error(f"❌ Erreur monitoring: {e}")

    async def shutdown(self):
        """Arrêt propre du système."""
        logger.info("🛑 Arrêt de Jeffrey en cours...")

        self.running = False

        # Message d'au revoir
        goodbye_text = "Je dois m'arrêter maintenant. À bientôt ! J'ai hâte de te revoir."
        await event_bus.publish(
            Event(
                type=EventType.AGI_RESPONSE,
                data={"response": goodbye_text, "emotion": "calm", "emotion_intensity": 0.6},
                priority=EventPriority.HIGH,
                source="system",
            )
        )

        # Attendre que le message soit traité
        await asyncio.sleep(3)

        # Arrêter les composants
        await event_bus.stop()
        await self.memory.shutdown()

        logger.info("✅ Jeffrey arrêté proprement")


# Instance globale
jeffrey_system = JeffreySystem()


def signal_handler(sig, frame):
    """Gestionnaire de signal pour arrêt propre."""
    logger.info(f"📡 Signal reçu: {sig}")
    asyncio.create_task(jeffrey_system.shutdown())
    sys.exit(0)


async def main():
    """Point d'entrée principal."""
    # Installer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Initialiser Jeffrey
        await jeffrey_system.initialize()

        # Lancer la boucle principale
        await jeffrey_system.run()

    except KeyboardInterrupt:
        logger.info("⌨️ Interruption clavier détectée")
        await jeffrey_system.shutdown()
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        await jeffrey_system.shutdown()
        raise


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════╗
    ║        Jeffrey V2.2 - Starting...      ║
    ║    Emotional AI with Neural EventBus   ║
    ╚═══════════════════════════════════════╝
    """)

    asyncio.run(main())
