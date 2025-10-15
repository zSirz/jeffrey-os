"""
Interface unifiée pour les Guardians existants
VERSION CORRIGÉE : Non-bloquante et tolérante aux erreurs
"""

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)


class GuardiansHub:
    """Point d'entrée unique pour tous les gardiens"""

    def __init__(self):
        self.symphony: object | None = None
        self.ethical: object | None = None
        self.initialized = False

    async def initialize(self):
        """Initialise les gardiens existants de façon non-bloquante"""
        try:
            # Guardian Symphony
            try:
                from .guardian_symphony import GuardianSymphony

                self.symphony = GuardianSymphony()
                # Lance en arrière-plan pour ne pas bloquer
                asyncio.create_task(self._start_symphony())
                logger.info("✅ Guardian Symphony initialization started")
            except ImportError as e:
                logger.warning(f"Guardian Symphony not available: {e}")
            except Exception as e:
                logger.error(f"Guardian Symphony error: {e}")

            # Ethical Guardian
            try:
                from .ethical_guardian import EthicalGuardian

                self.ethical = EthicalGuardian()

                # Vérifie si la méthode existe
                if hasattr(self.ethical, "initialize_ethical_systems"):
                    await self.ethical.initialize_ethical_systems()
                    logger.info("✅ Ethical Guardian initialized")
                else:
                    logger.warning("Ethical Guardian missing initialize method")
            except ImportError as e:
                logger.warning(f"Ethical Guardian not available: {e}")
            except Exception as e:
                logger.error(f"Ethical Guardian error: {e}")

            self.initialized = True
            logger.info("✅ Guardians Hub initialization complete")

        except Exception as e:
            logger.error(f"Critical error in Guardians Hub: {e}")
            self.initialized = False

    async def _start_symphony(self):
        """Démarre Symphony en arrière-plan"""
        try:
            if self.symphony and hasattr(self.symphony, "start"):
                await self.symphony.start()
        except Exception as e:
            logger.error(f"Symphony start error: {e}")

    def get_status(self) -> dict[str, Any]:
        """Retourne le statut de tous les gardiens"""
        status = {"initialized": self.initialized, "components": {}}

        # Symphony status
        if self.symphony:
            try:
                if hasattr(self.symphony, "get_dashboard_data"):
                    status["components"]["symphony"] = self.symphony.get_dashboard_data()
                else:
                    status["components"]["symphony"] = {"status": "running"}
            except Exception as e:
                status["components"]["symphony"] = {"error": str(e)}
        else:
            status["components"]["symphony"] = {"status": "not loaded"}

        # Ethical status
        if self.ethical:
            try:
                if hasattr(self.ethical, "get_ethical_status"):
                    status["components"]["ethical"] = self.ethical.get_ethical_status()
                else:
                    status["components"]["ethical"] = {"status": "running"}
            except Exception as e:
                status["components"]["ethical"] = {"error": str(e)}
        else:
            status["components"]["ethical"] = {"status": "not loaded"}

        return status

    async def process_event(self, event: dict) -> dict:
        """Traite un événement via les gardiens"""
        results = {"status": "processed", "checks": {}}

        # Ethical check si disponible
        if self.ethical and hasattr(self.ethical, "evaluate_ethical_decision"):
            try:
                # Utilise string au lieu de l'enum si non importé
                ethical_result = await self.ethical.evaluate_ethical_decision(
                    event,
                    "ACTION_APPROVAL",  # String au lieu de EthicalDecisionType.ACTION_APPROVAL
                )

                if not ethical_result.get("recommendation", {}).get("approved", True):
                    results["status"] = "rejected"
                    results["reason"] = "ethical_violation"
                    results["checks"]["ethical"] = ethical_result
                else:
                    results["checks"]["ethical"] = {"approved": True}
            except Exception as e:
                logger.error(f"Ethical check error: {e}")
                results["checks"]["ethical"] = {"error": str(e), "defaulted": "approved"}

        # Symphony processing si disponible
        if self.symphony:
            results["checks"]["symphony"] = {"status": "available"}

        return results
