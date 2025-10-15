"""
Runtime singleton pour Jeffrey OS
Assure qu'il n'y a qu'une seule instance de chaque composant
"""

import logging
import threading
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Thread lock pour le singleton
_singleton_lock = threading.Lock()
_singleton = None


@dataclass
class JeffreyRuntime:
    """Container pour tous les composants runtime de Jeffrey"""

    bus: object | None = None  # NeuralBus si disponible
    blackboard: object | None = None
    scheduler: object | None = None
    orchestrator: object | None = None
    apertus_client: object | None = None


def get_runtime() -> JeffreyRuntime:
    """
    Récupère ou crée le runtime singleton de Jeffrey
    Thread-safe pour éviter les doubles initialisations
    """
    global _singleton
    if _singleton:
        return _singleton

    with _singleton_lock:
        if not _singleton:
            logger.info("Initializing Jeffrey Runtime...")

            # 1) Créer d'abord l'ApertusClient (si dispo)
            apertus = None
            try:
                from src.jeffrey.core.llm.apertus_client import ApertusClient

                apertus = ApertusClient()
                logger.info("ApertusClient initialized")
            except Exception as e:
                logger.warning(f"ApertusClient not available: {e}")

            # 2) Créer l'orchestrateur (qui instancie son propre état)
            from src.jeffrey.core.response.neural_response_orchestrator import NeuralResponseOrchestrator

            orchestrator = NeuralResponseOrchestrator(None, None, apertus)

            # 3) RÉ-EXPOSER l'état interne de l'orchestrateur comme "runtime.*"
            # Cela garantit un seul état partagé dans tout le système
            blackboard = getattr(orchestrator, "blackboard", None)
            scheduler = getattr(orchestrator, "scheduler", None)

            # 4) Charger l'état du scheduler une fois qu'on a la bonne instance
            try:
                import os

                state_file = "data/scheduler_state.json"
                if scheduler and os.path.exists(state_file):
                    scheduler.load_state(state_file)
                    logger.info(f"Loaded scheduler state from {state_file}")
            except Exception as e:
                logger.warning(f"Could not load scheduler state: {e}")

            _singleton = JeffreyRuntime(
                bus=None,  # À connecter plus tard si nécessaire
                blackboard=blackboard,
                scheduler=scheduler,
                orchestrator=orchestrator,
                apertus_client=apertus,
            )

            logger.info("Jeffrey Runtime initialized successfully")

    return _singleton


def shutdown_runtime():
    """Arrête proprement le runtime"""
    global _singleton
    if _singleton:
        # Sauvegarder l'état du scheduler
        if _singleton.scheduler:
            try:
                _singleton.scheduler.save_state("data/scheduler_state.json")
                logger.info("Scheduler state saved")
            except Exception as e:
                logger.error(f"Could not save scheduler state: {e}")

        _singleton = None
        logger.info("Jeffrey Runtime shutdown")
