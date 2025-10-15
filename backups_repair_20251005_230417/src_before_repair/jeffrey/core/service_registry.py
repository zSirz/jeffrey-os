#!/usr/bin/env python3
"""
ServiceRegistry - Gestionnaire des services internes
Stub pour compatibilité (correction GPT)
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """
    Registre central des services du Core
    Garde trace de tous les composants actifs
    """

    def __init__(self):
        self.services: dict[str, dict[str, Any]] = {}

    def register(self, name: str, service: Any, metadata: dict[str, Any] = None) -> None:
        """Enregistre un service"""
        self.services[name] = {
            "instance": service,
            "type": type(service).__name__,
            "status": "registered",
            "metadata": metadata or {},
        }
        logger.info(f"Service registered: {name}")

    def get(self, name: str) -> Any:
        """Récupère un service"""
        if name in self.services:
            return self.services[name]["instance"]
        return None

    def list_services(self) -> list[str]:
        """Liste tous les services"""
        return list(self.services.keys())

    def get_status(self, name: str) -> str:
        """Retourne le statut d'un service"""
        if name in self.services:
            return self.services[name]["status"]
        return "unknown"

    def set_status(self, name: str, status: str) -> None:
        """Met à jour le statut d'un service"""
        if name in self.services:
            self.services[name]["status"] = status

    def get_all_statuses(self) -> dict[str, str]:
        """Retourne tous les statuts"""
        return {name: service["status"] for name, service in self.services.items()}
