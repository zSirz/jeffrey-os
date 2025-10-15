"""
Utilitaires pour privacy et GDPR
"""

import hashlib
import logging
import os
import re
from typing import Any

from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


class PrivacyGuard:
    """
    Garde la privacy avec Fernet (AES-128-CBC + HMAC-SHA256).

    Rotation recommandée : tous les 90 jours
    """

    def __init__(self):
        # Support multi-keys for rotation
        self.keyring = {}

        # CLÉ OBLIGATOIRE EN PRODUCTION
        key = os.environ.get("JEFFREY_ENCRYPTION_KEY")
        current_kid = os.environ.get("JEFFREY_KID", "v1")

        if not key:
            mode = os.environ.get("JEFFREY_MODE", "dev")
            if mode == "production":
                raise ValueError(
                    "JEFFREY_ENCRYPTION_KEY required in production! "
                    "Generate with: python -c 'from cryptography.fernet import Fernet; "
                    "print(Fernet.generate_key().decode())'"
                )
            else:
                # Dev only : clé éphémère
                logger.warning("Using ephemeral key (DEV ONLY)")
                key = Fernet.generate_key()
        else:
            key = key.encode() if isinstance(key, str) else key

        # Store current key
        self.keyring[current_kid] = Fernet(key)
        self.current_kid = current_kid
        self.cipher = self.keyring[current_kid]

        # Load old keys for decryption (format: JEFFREY_KEY_v0=..., JEFFREY_KEY_v1=...)
        for env_key, value in os.environ.items():
            if env_key.startswith("JEFFREY_KEY_"):
                kid = env_key.replace("JEFFREY_KEY_", "")
                try:
                    old_key = value.encode() if isinstance(value, str) else value
                    self.keyring[kid] = Fernet(old_key)
                    logger.info(f"Loaded rotation key: {kid}")
                except Exception as e:
                    logger.error(f"Failed to load key {kid}: {e}")

        # NEW: Keyring for search keys
        self.search_keyring = {}

        # Current search key
        current_search_key = os.environ.get("JEFFREY_SEARCH_KEY")
        self.current_skid = os.environ.get("JEFFREY_SEARCH_KID", "s1")

        if current_search_key:
            # Handle hex format
            import binascii

            try:
                self.search_keyring[self.current_skid] = binascii.unhexlify(current_search_key)
            except (binascii.Error, ValueError):
                self.search_keyring[self.current_skid] = (
                    current_search_key.encode() if isinstance(current_search_key, str) else current_search_key
                )

        # Load old search keys (format: JEFFREY_SEARCH_KEY_s0=..., JEFFREY_SEARCH_KEY_s1=...)
        for env_key, value in os.environ.items():
            if env_key.startswith("JEFFREY_SEARCH_KEY_"):
                skid = env_key.replace("JEFFREY_SEARCH_KEY_", "")
                try:
                    # Handle hex format
                    import binascii

                    try:
                        self.search_keyring[skid] = binascii.unhexlify(value)
                    except (binascii.Error, ValueError):
                        self.search_keyring[skid] = value.encode() if isinstance(value, str) else value
                    logger.info(f"Loaded search key: {skid}")
                except Exception as e:
                    logger.error(f"Failed to load search key {skid}: {e}")

        # Patterns PII (regex seulement par défaut)
        self.pii_patterns = {
            "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "phone": r"\b\d{3}[-.\s]?\d{4}\b|\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "credit_card": r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
            "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
            "iban": r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b",
        }

    def detect_pii(self, text: str) -> bool:
        """Détection PII par regex (ML désactivé par défaut)"""
        for pattern in self.pii_patterns.values():
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def mask_for_logging(self, text: str) -> str:
        """Masque PII pour logs sécurisés"""
        masked = text
        for pii_type, pattern in self.pii_patterns.items():
            masked = re.sub(pattern, f"[{pii_type.upper()}]", masked, flags=re.IGNORECASE)
        return masked

    def encrypt_if_pii(self, data: dict[str, Any]) -> dict[str, Any]:
        """Chiffre les champs contenant des PII"""
        encrypted = data.copy()

        for key, value in data.items():
            if isinstance(value, str) and self.detect_pii(value):
                try:
                    encrypted[key] = self.cipher.encrypt(value.encode()).decode()
                    encrypted[f"{key}_encrypted"] = True
                except Exception as e:
                    logger.error(f"Failed to encrypt field {key}: {e}")

        return encrypted

    def anonymize_user_id(self, user_id: str) -> str:
        """Anonymise un user_id de manière réversible"""
        return hashlib.sha256(user_id.encode()).hexdigest()[:16]

    def decrypt_with_kid(self, ciphertext: bytes, kid: str) -> bytes:
        """Decrypt with a specific key from the keyring"""
        if kid in self.keyring:
            return self.keyring[kid].decrypt(ciphertext)
        else:
            # Try with current key as fallback
            logger.warning(f"Unknown key ID: {kid}, trying current key")
            return self.cipher.decrypt(ciphertext)

    def get_search_key_for_skid(self, skid: str) -> bytes:
        """Get search key for a given SKID"""
        if skid in self.search_keyring:
            return self.search_keyring[skid]
        elif self.current_skid in self.search_keyring:
            return self.search_keyring[self.current_skid]
        else:
            # Fallback to generating ephemeral key
            logger.warning(f"Unknown SKID: {skid}, generating ephemeral key")
            return os.urandom(32)
