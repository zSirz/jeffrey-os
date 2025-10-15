#!/usr/bin/env python3
"""
BaseBridgeAdapter - Classe de base pour tous les adapters du Bridge
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class BaseBridgeAdapter(ABC):
    """
    Classe de base abstraite pour tous les adapters du Bridge
    """

    def __init__(self, adapter_type: str):
        self.adapter_type = adapter_type
        self.initialized = False

    async def initialize(self) -> None:
        """Initialise l'adapter"""
        self.initialized = True
        logger.info(f"{self.adapter_type} adapter: initialized")

    @abstractmethod
    async def health(self) -> bool:
        """Vérifie la santé de l'adapter"""
        pass

    async def shutdown(self) -> None:
        """Ferme l'adapter proprement"""
        self.initialized = False
        logger.info(f"{self.adapter_type} adapter: shutdown")

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les métriques de l'adapter"""
        return {"type": self.adapter_type, "initialized": self.initialized}
