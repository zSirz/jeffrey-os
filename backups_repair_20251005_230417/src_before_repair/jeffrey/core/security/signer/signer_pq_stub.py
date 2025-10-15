"""
Post-Quantum Signer STUB - Placeholder pour signature post-quantique
"""

import base64
import logging
import secrets
from typing import Any

from .signer_base import SignerBase

logger = logging.getLogger(__name__)


class SignerPQStub(SignerBase):
    """
    Stub pour signature post-quantique (Dilithium/Falcon)
    Version simplifiée en attendant liboqs-python
    """

    def __init__(self):
        super().__init__("PQ-STUB")
        self.mock_private_key: bytes | None = None
        self.mock_public_key: bytes | None = None
        logger.warning("⚠️ Using PQ Signer STUB - Not quantum-resistant!")

    async def initialize(self):
        """Initialise le stub avec des clés mock"""
        self.mock_private_key = secrets.token_bytes(64)
        self.mock_public_key = secrets.token_bytes(32)
        logger.info("📝 PQ Stub initialized with mock keys")

    async def sign(self, data: dict[str, Any]) -> str:
        """
        Signature mock (HMAC-like pour le stub)

        Args:
            data: Données à signer

        Returns:
            Signature mock en base64
        """
        if not self.mock_private_key:
            await self.initialize()

        try:
            # Sérialiser les données
            serialized = self._serialize_data(data)

            # Créer une signature mock (hash simple)
            import hmac

            signature = hmac.new(self.mock_private_key, serialized, "sha256").digest()

            # Encoder en base64
            encoded_signature = base64.b64encode(signature).decode("utf-8")

            self.stats["signatures_created"] += 1
            logger.debug("PQ Stub signature created (NOT quantum-resistant)")

            return encoded_signature

        except Exception as e:
            logger.error(f"PQ Stub signing error: {e}")
            raise

    async def verify(self, data: dict[str, Any], signature: str) -> tuple[bool, str | None]:
        """
        Vérification mock

        Args:
            data: Données signées
            signature: Signature en base64

        Returns:
            (is_valid, error_message)
        """
        if not self.mock_private_key:
            return False, "NO_MOCK_KEY"

        try:
            # Recréer la signature
            expected_signature = await self.sign(data)

            # Comparer
            is_valid = expected_signature == signature

            if is_valid:
                self.stats["signatures_verified"] += 1
                return True, None
            else:
                self.stats["verification_failures"] += 1
                return False, "SIGNATURE_MISMATCH"

        except Exception as e:
            self.stats["verification_failures"] += 1
            return False, f"VERIFICATION_ERROR_{str(e)}"

    async def generate_keypair(self) -> dict[str, str]:
        """
        Génère une paire de clés mock

        Returns:
            Dict avec des clés mock en base64
        """
        mock_private = secrets.token_bytes(64)
        mock_public = secrets.token_bytes(32)

        return {
            "private_key": base64.b64encode(mock_private).decode("utf-8"),
            "public_key": base64.b64encode(mock_public).decode("utf-8"),
            "algorithm": self.algorithm,
            "warning": "STUB_KEYS_NOT_QUANTUM_RESISTANT",
        }

    def upgrade_to_real_pq(self) -> str:
        """
        Instructions pour upgrade vers vrai post-quantique

        Returns:
            Instructions d'installation
        """
        return """
        Pour activer la vraie signature post-quantique:

        1. Installer liboqs:
           brew install liboqs  # macOS
           apt install liboqs-dev  # Ubuntu

        2. Installer le binding Python:
           pip install liboqs-python

        3. Remplacer SignerPQStub par SignerPQ dans le code

        Algorithmes supportés:
        - Dilithium2, Dilithium3, Dilithium5
        - Falcon-512, Falcon-1024
        - SPHINCS+-SHA256-128f
        """

    def get_status(self) -> dict[str, Any]:
        """Retourne le statut du stub"""
        status = super().get_status()
        status.update(
            {
                "mode": "STUB",
                "quantum_resistant": False,
                "warning": "Using mock signatures - NOT secure against quantum attacks",
                "upgrade_instructions": "Run upgrade_to_real_pq() for instructions",
            }
        )
        return status
