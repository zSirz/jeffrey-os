"""
EdDSA Signer - Signature EdDSA (Ed25519)
"""

import base64
import logging
from typing import Any

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

from .signer_base import SignerBase

logger = logging.getLogger(__name__)


class SignerEdDSA(SignerBase):
    """
    Signer utilisant EdDSA (Ed25519) pour les signatures
    Rapide et sécurisé, recommandé pour la production
    """

    def __init__(self):
        super().__init__("EdDSA-Ed25519")
        self.private_key: ed25519.Ed25519PrivateKey | None = None
        self.public_key: ed25519.Ed25519PublicKey | None = None

    async def initialize(self, private_key_path: str | None = None):
        """
        Initialise le signer avec une clé existante ou en génère une nouvelle

        Args:
            private_key_path: Chemin vers la clé privée (optionnel)
        """
        if private_key_path:
            try:
                with open(private_key_path, "rb") as f:
                    self.private_key = serialization.load_pem_private_key(
                        f.read(), password=None, backend=default_backend()
                    )
                    self.public_key = self.private_key.public_key()
                    logger.info("✅ EdDSA keys loaded from file")
            except Exception as e:
                logger.error(f"Failed to load EdDSA key: {e}")
                await self._generate_new_keypair()
        else:
            await self._generate_new_keypair()

    async def _generate_new_keypair(self):
        """Génère une nouvelle paire de clés Ed25519"""
        self.private_key = ed25519.Ed25519PrivateKey.generate()
        self.public_key = self.private_key.public_key()
        logger.info("✅ New EdDSA keypair generated")

    async def sign(self, data: dict[str, Any]) -> str:
        """
        Signe les données avec EdDSA

        Args:
            data: Données à signer

        Returns:
            Signature en base64
        """
        if not self.private_key:
            await self._generate_new_keypair()

        try:
            # Sérialiser les données
            serialized = self._serialize_data(data)

            # Signer avec Ed25519
            signature = self.private_key.sign(serialized)

            # Encoder en base64 pour le transport
            encoded_signature = base64.b64encode(signature).decode("utf-8")

            self.stats["signatures_created"] += 1
            return encoded_signature

        except Exception as e:
            logger.error(f"EdDSA signing error: {e}")
            raise

    async def verify(self, data: dict[str, Any], signature: str) -> tuple[bool, str | None]:
        """
        Vérifie une signature EdDSA

        Args:
            data: Données signées
            signature: Signature en base64

        Returns:
            (is_valid, error_message)
        """
        if not self.public_key:
            return False, "NO_PUBLIC_KEY"

        try:
            # Décoder la signature
            signature_bytes = base64.b64decode(signature)

            # Sérialiser les données
            serialized = self._serialize_data(data)

            # Vérifier avec Ed25519
            self.public_key.verify(signature_bytes, serialized)

            self.stats["signatures_verified"] += 1
            return True, None

        except Exception as e:
            self.stats["verification_failures"] += 1
            logger.debug(f"EdDSA verification failed: {e}")
            return False, f"INVALID_SIGNATURE_{str(e)}"

    async def generate_keypair(self) -> dict[str, str]:
        """
        Génère une nouvelle paire de clés Ed25519

        Returns:
            Dict avec les clés en PEM
        """
        private_key = ed25519.Ed25519PrivateKey.generate()
        public_key = private_key.public_key()

        # Encoder en PEM
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode("utf-8")

        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

        return {"private_key": private_pem, "public_key": public_pem, "algorithm": self.algorithm}

    def export_public_key(self) -> str | None:
        """
        Exporte la clé publique en PEM

        Returns:
            Clé publique en PEM ou None
        """
        if not self.public_key:
            return None

        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("utf-8")

    async def save_keys(self, private_path: str, public_path: str):
        """
        Sauvegarde les clés sur disque

        Args:
            private_path: Chemin pour la clé privée
            public_path: Chemin pour la clé publique
        """
        if not self.private_key or not self.public_key:
            await self._generate_new_keypair()

        # Sauvegarder la clé privée
        private_pem = self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        with open(private_path, "wb") as f:
            f.write(private_pem)

        # Sauvegarder la clé publique
        public_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        with open(public_path, "wb") as f:
            f.write(public_pem)

        logger.info(f"✅ EdDSA keys saved to {private_path} and {public_path}")
