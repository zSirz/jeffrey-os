#!/usr/bin/env python3
"""
BridgeRegistry - Gestionnaire central des adapters du Bridge
"""

import logging
from typing import Any

# Import propre depuis base
from .base import BaseBridgeAdapter

logger = logging.getLogger(__name__)


class BridgeRegistry:
    """
    Registre central pour tous les adapters du Bridge
    Gère les connexions externes de manière sécurisée
    """

    def __init__(self):
        self.adapters: dict[str, BaseBridgeAdapter] = {}
        self._permissions: dict[str, dict[str, bool]] = {
            "http": {"get": True, "post": True},
            "mail": {"send": True, "receive": True},
            "storage": {"read": True, "write": True},
        }

    def register(self, adapter: BaseBridgeAdapter) -> None:
        """Enregistre un adapter"""
        self.adapters[adapter.adapter_type] = adapter
        logger.info(f"Registered adapter: {adapter.adapter_type}")

    def get(self, adapter_type: str) -> BaseBridgeAdapter | None:
        """Récupère un adapter"""
        return self.adapters.get(adapter_type)

    def list_adapters(self) -> list[str]:
        """Liste tous les adapters disponibles"""
        return list(self.adapters.keys())

    async def initialize_all(self) -> None:
        """Initialise tous les adapters"""
        logger.info("Initializing all Bridge adapters...")
        for adapter_type, adapter in self.adapters.items():
            try:
                await adapter.initialize()
                logger.info(f" Initialized: {adapter_type}")
            except Exception as e:
                logger.error(f"Failed to initialize {adapter_type}: {e}")

    async def shutdown_all(self) -> None:
        """Arrête tous les adapters"""
        logger.info("Shutting down all Bridge adapters...")
        for adapter_type, adapter in self.adapters.items():
            try:
                await adapter.shutdown()
                logger.info(f" Shutdown: {adapter_type}")
            except Exception as e:
                logger.error(f"Failed to shutdown {adapter_type}: {e}")

    async def health_check(self) -> dict[str, bool]:
        """Vérifie la santé de tous les adapters"""
        health_status = {}
        for adapter_type, adapter in self.adapters.items():
            try:
                health_status[adapter_type] = await adapter.health()
            except Exception as e:
                logger.error(f"Health check failed for {adapter_type}: {e}")
                health_status[adapter_type] = False
        return health_status

    def check_permission(self, adapter_type: str, action: str, context: dict[str, Any]) -> bool:
        """Vérifie les permissions pour une action"""
        # Pour le moment, permissions basiques
        permissions = self._permissions.get(adapter_type, {})
        return permissions.get(action, False)

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les métriques du Bridge"""
        metrics = {"total_adapters": len(self.adapters), "adapters": {}}

        for adapter_type, adapter in self.adapters.items():
            try:
                metrics["adapters"][adapter_type] = adapter.get_metrics()
            except Exception as e:
                metrics["adapters"][adapter_type] = {"error": str(e)}

        return metrics


# ===== REGISTRY DES 8 RÉGIONS CÉRÉBRALES =====
# Mapping canonique vers les adaptateurs EXPLICITES

REGION_ADAPTERS = {
    "perception": "jeffrey.bridge.adapters.perception_adapter.PerceptionAdapter",
    "memory": "jeffrey.bridge.adapters.memory_adapter.MemoryAdapter",
    "emotion": "jeffrey.bridge.adapters.emotion_adapter.EmotionAdapter",
    "conscience": "jeffrey.bridge.adapters.conscience_adapter.ConscienceAdapter",
    "language": "jeffrey.bridge.adapters.language_adapter.LanguageAdapter",
    "executive": "jeffrey.bridge.adapters.executive_adapter.ExecutiveAdapter",
    "motor": "jeffrey.bridge.adapters.motor_adapter.MotorAdapter",
    "integration": "jeffrey.bridge.adapters.integration_adapter.IntegrationAdapter",
}
