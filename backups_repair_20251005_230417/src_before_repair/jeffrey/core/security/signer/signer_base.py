"""
Base Signer - Interface de base pour tous les signers
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class SignerBase(ABC):
    """Interface de base pour tous les algorithmes de signature"""

    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.stats = {"signatures_created": 0, "signatures_verified": 0, "verification_failures": 0}

    @abstractmethod
    async def sign(self, data: dict[str, Any]) -> str:
        """
        Signe les données et retourne la signature

        Args:
            data: Données à signer

        Returns:
            Signature encodée
        """
        pass

    @abstractmethod
    async def verify(self, data: dict[str, Any], signature: str) -> tuple[bool, str | None]:
        """
        Vérifie une signature

        Args:
            data: Données signées
            signature: Signature à vérifier

        Returns:
            (is_valid, error_message)
        """
        pass

    @abstractmethod
    async def generate_keypair(self) -> dict[str, str]:
        """
        Génère une nouvelle paire de clés

        Returns:
            Dict avec 'public_key' et 'private_key'
        """
        pass

    def _serialize_data(self, data: dict[str, Any]) -> bytes:
        """
        Sérialise les données de manière déterministe pour la signature

        Args:
            data: Données à sérialiser

        Returns:
            Données sérialisées en bytes
        """
        # Tri des clés pour assurer un ordre déterministe
        return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def _compute_hash(self, data: bytes, algorithm: str = "sha256") -> bytes:
        """
        Calcule le hash des données

        Args:
            data: Données à hasher
            algorithm: Algorithme de hashage

        Returns:
            Hash des données
        """
        h = hashlib.new(algorithm)
        h.update(data)
        return h.digest()

    def get_status(self) -> dict[str, Any]:
        """Retourne le statut du signer"""
        return {"algorithm": self.algorithm, "stats": self.stats}
