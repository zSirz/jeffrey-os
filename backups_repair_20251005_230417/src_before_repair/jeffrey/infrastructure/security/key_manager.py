"""
Module de infrastructure système de base pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de infrastructure système de base pour jeffrey os.
Il fournit une architecture robuste et évolutive intégrant les composants
nécessaires au fonctionnement optimal du système. L'implémentation suit
les principes de modularité et d'extensibilité pour faciliter l'évolution
future du système.

Le module gère l'initialisation, la configuration, le traitement des données,
la communication inter-composants, et la persistance des états. Il s'intègre
harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
une séparation claire des responsabilités.

L'architecture interne permet une évolution adaptative basée sur les interactions
et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
cohérente et authentique.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

try:
    import keyring
    from keyring.backends import SecretService, Windows, macOS

    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    keyring = None

logger = logging.getLogger(__name__)


class KeyType(Enum):
    """Types de clés"""

    API_KEY = "api_key"
    DATABASE_KEY = "database_key"
    ENCRYPTION_KEY = "encryption_key"
    JWT_SECRET = "jwt_secret"
    OAUTH_SECRET = "oauth_secret"
    WEBHOOK_SECRET = "webhook_secret"


class KeySecurity(Enum):
    """Niveaux de sécurité des clés"""

    STANDARD = "standard"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class KeyMetadata:
    """Métadonnées d'une clé"""

    key_id: str
    key_type: KeyType
    security_level: KeySecurity
    created_at: datetime
    last_used: datetime | None = None
    expires_at: datetime | None = None
    usage_count: int = 0
    max_usage: int | None = None
    description: str = ""
    tags: list[str] = None


class KeyManager:
    """Gestionnaire de clés sécurisé"""

    def __init__(self, service_name: str = "jeffrey_os") -> None:
        self.service_name = service_name
        self.keyring_available = KEYRING_AVAILABLE
        self.metadata_file = f".{service_name}_key_metadata.json"
        self.metadata: dict[str, KeyMetadata] = {}

        # Charger les métadonnées
        self._load_metadata()

        # Initialiser le keyring si disponible
        if self.keyring_available:
            self._init_keyring()
        else:
            logger.warning("Keyring non disponible, utilisation du stockage de fichiers")

    def _init_keyring(self):
        """Initialise le keyring avec le backend approprié"""
        try:
            # Définir le backend préféré selon l'OS
            import platform

            system = platform.system()

            if system == "Darwin":  # macOS
                keyring.set_keyring(macOS.Keyring())
            elif system == "Windows":
                keyring.set_keyring(Windows.WinVaultKeyring())
            elif system == "Linux":
                keyring.set_keyring(SecretService.Keyring())

            logger.info(f"Keyring initialisé avec {keyring.get_keyring()}")
        except Exception as e:
            logger.warning(f"Impossible d'initialiser le keyring: {e}")
            self.keyring_available = False

    def _load_metadata(self):
        """Charge les métadonnées des clés"""
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file) as f:
                    data = json.load(f)

                # Reconstruire les objets KeyMetadata
                for key_id, meta_dict in data.items():
                    self.metadata[key_id] = KeyMetadata(
                        key_id=meta_dict["key_id"],
                        key_type=KeyType(meta_dict["key_type"]),
                        security_level=KeySecurity(meta_dict["security_level"]),
                        created_at=datetime.fromisoformat(meta_dict["created_at"]),
                        last_used=(
                            datetime.fromisoformat(meta_dict["last_used"]) if meta_dict.get("last_used") else None
                        ),
                        expires_at=(
                            datetime.fromisoformat(meta_dict["expires_at"]) if meta_dict.get("expires_at") else None
                        ),
                        usage_count=meta_dict.get("usage_count", 0),
                        max_usage=meta_dict.get("max_usage"),
                        description=meta_dict.get("description", ""),
                        tags=meta_dict.get("tags", []),
                    )

                logger.info(f"Métadonnées chargées pour {len(self.metadata)} clés")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des métadonnées: {e}")

    def _save_metadata(self):
        """Sauvegarde les métadonnées des clés"""
        try:
            # Convertir en dictionnaire sérialisable
            data = {}
            for key_id, metadata in self.metadata.items():
                data[key_id] = {
                    "key_id": metadata.key_id,
                    "key_type": metadata.key_type.value,
                    "security_level": metadata.security_level.value,
                    "created_at": metadata.created_at.isoformat(),
                    "last_used": metadata.last_used.isoformat() if metadata.last_used else None,
                    "expires_at": metadata.expires_at.isoformat() if metadata.expires_at else None,
                    "usage_count": metadata.usage_count,
                    "max_usage": metadata.max_usage,
                    "description": metadata.description,
                    "tags": metadata.tags or [],
                }

            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=2)

            # Sécuriser le fichier
            os.chmod(self.metadata_file, 0o600)

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des métadonnées: {e}")

    def generate_key(
        self,
        key_id: str,
        key_type: KeyType,
        security_level: KeySecurity = KeySecurity.STANDARD,
        length: int = 32,
        description: str = "",
        expires_in_days: int | None = None,
        max_usage: int | None = None,
        tags: list[str] = None,
    ) -> str:
        """
        Génère une nouvelle clé

        Args:
            key_id: Identifiant unique de la clé
            key_type: Type de clé
            security_level: Niveau de sécurité
            length: Longueur de la clé en bytes
            description: Description de la clé
            expires_in_days: Expiration en jours
            max_usage: Nombre maximum d'utilisations
            tags: Tags pour la clé

        Returns:
            str: Clé générée
        """
        if key_id in self.metadata:
            raise ValueError(f"Clé '{key_id}' existe déjà")

        try:
            # Générer la clé selon le niveau de sécurité
            if security_level == KeySecurity.CRITICAL:
                # Clé cryptographiquement forte
                key = secrets.token_urlsafe(length)
            elif security_level == KeySecurity.HIGH:
                # Clé avec entropie élevée
                key = secrets.token_hex(length)
            else:
                # Clé standard
                key = secrets.token_urlsafe(length // 2)

            # Stocker la clé
            self._store_key(key_id, key)

            # Créer les métadonnées
            metadata = KeyMetadata(
                key_id=key_id,
                key_type=key_type,
                security_level=security_level,
                created_at=datetime.now(),
                expires_at=(datetime.now() + timedelta(days=expires_in_days) if expires_in_days else None),
                max_usage=max_usage,
                description=description,
                tags=tags or [],
            )

            self.metadata[key_id] = metadata
            self._save_metadata()

            logger.info(f"Clé générée: {key_id} ({key_type.value})")
            return key

        except Exception as e:
            logger.error(f"Erreur lors de la génération de la clé: {e}")
            raise

    def get_key(self, key_id: str) -> str | None:
        """
        Récupère une clé

        Args:
            key_id: Identifiant de la clé

        Returns:
            str: Clé ou None si non trouvée
        """
        if key_id not in self.metadata:
            return None

        metadata = self.metadata[key_id]

        # Vérifier l'expiration
        if metadata.expires_at and datetime.now() > metadata.expires_at:
            logger.warning(f"Clé expirée: {key_id}")
            return None

        # Vérifier le nombre d'utilisations
        if metadata.max_usage and metadata.usage_count >= metadata.max_usage:
            logger.warning(f"Clé épuisée: {key_id}")
            return None

        try:
            # Récupérer la clé
            key = self._retrieve_key(key_id)

            if key:
                # Mettre à jour les métadonnées d'utilisation
                metadata.last_used = datetime.now()
                metadata.usage_count += 1
                self._save_metadata()

            return key

        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la clé: {e}")
            return None

    def _store_key(self, key_id: str, key: str):
        """
        Stocke une clé de manière sécurisée

        Args:
            key_id: Identifiant de la clé
            key: Clé à stocker
        """
        if self.keyring_available:
            try:
                keyring.set_password(self.service_name, key_id, key)
                logger.debug(f"Clé stockée dans le keyring: {key_id}")
                return
            except Exception as e:
                logger.warning(f"Impossible de stocker dans le keyring: {e}")

        # Fallback: stockage chiffré dans un fichier
        self._store_key_file(key_id, key)

    def _retrieve_key(self, key_id: str) -> str | None:
        """
        Récupère une clé du stockage sécurisé

        Args:
            key_id: Identifiant de la clé

        Returns:
            str: Clé ou None si non trouvée
        """
        if self.keyring_available:
            try:
                key = keyring.get_password(self.service_name, key_id)
                if key:
                    logger.debug(f"Clé récupérée du keyring: {key_id}")
                    return key
            except Exception as e:
                logger.warning(f"Impossible de récupérer du keyring: {e}")

        # Fallback: récupération depuis le fichier
        return self._retrieve_key_file(key_id)

    def _store_key_file(self, key_id: str, key: str):
        """
        Stocke une clé dans un fichier chiffré

        Args:
            key_id: Identifiant de la clé
            key: Clé à stocker
        """
        try:
            # Créer le répertoire des clés
            keys_dir = f".{self.service_name}_keys"
            os.makedirs(keys_dir, exist_ok=True)

            # Chiffrer la clé avec un mot de passe dérivé
            password = self._get_master_password()
            encrypted_key = self._encrypt_key(key, password)

            # Stocker dans un fichier
            key_file = os.path.join(keys_dir, f"{key_id}.key")
            with open(key_file, "w") as f:
                f.write(encrypted_key)

            # Sécuriser le fichier
            os.chmod(key_file, 0o600)

            logger.debug(f"Clé stockée dans le fichier: {key_file}")

        except Exception as e:
            logger.error(f"Erreur lors du stockage de la clé: {e}")
            raise

    def _retrieve_key_file(self, key_id: str) -> str | None:
        """
        Récupère une clé depuis un fichier chiffré

        Args:
            key_id: Identifiant de la clé

        Returns:
            str: Clé ou None si non trouvée
        """
        try:
            keys_dir = f".{self.service_name}_keys"
            key_file = os.path.join(keys_dir, f"{key_id}.key")

            if not os.path.exists(key_file):
                return None

            # Lire le fichier chiffré
            with open(key_file) as f:
                encrypted_key = f.read()

            # Déchiffrer la clé
            password = self._get_master_password()
            key = self._decrypt_key(encrypted_key, password)

            logger.debug(f"Clé récupérée du fichier: {key_file}")
            return key

        except Exception as e:
            logger.error(f"Erreur lors de la récupération de la clé: {e}")
            return None

    def _get_master_password(self) -> str:
        """
        Génère le mot de passe maître pour le chiffrement des clés

        Returns:
            str: Mot de passe maître
        """
        # Utiliser des informations système pour créer un mot de passe
        import platform

        system_info = f"{platform.node()}-{platform.system()}-{self.service_name}"
        return hashlib.sha256(system_info.encode()).hexdigest()

    def _encrypt_key(self, key: str, password: str) -> str:
        """
        Chiffre une clé avec un mot de passe

        Args:
            key: Clé à chiffrer
            password: Mot de passe

        Returns:
            str: Clé chiffrée
        """
        # Utilisation simple de XOR pour le chiffrement
        # En production, utiliser AES
        key_bytes = key.encode()
        password_bytes = password.encode()

        encrypted = bytearray()
        for i, byte in enumerate(key_bytes):
            encrypted.append(byte ^ password_bytes[i % len(password_bytes)])

        return encrypted.hex()

    def _decrypt_key(self, encrypted_key: str, password: str) -> str:
        """
        Déchiffre une clé avec un mot de passe

        Args:
            encrypted_key: Clé chiffrée
            password: Mot de passe

        Returns:
            str: Clé déchiffrée
        """
        # Décoder et déchiffrer avec XOR
        encrypted_bytes = bytes.fromhex(encrypted_key)
        password_bytes = password.encode()

        decrypted = bytearray()
        for i, byte in enumerate(encrypted_bytes):
            decrypted.append(byte ^ password_bytes[i % len(password_bytes)])

        return decrypted.decode()

    def delete_key(self, key_id: str) -> bool:
        """
        Supprime une clé

        Args:
            key_id: Identifiant de la clé

        Returns:
            bool: True si la suppression a réussi
        """
        if key_id not in self.metadata:
            return False

        try:
            # Supprimer du keyring
            if self.keyring_available:
                try:
                    keyring.delete_password(self.service_name, key_id)
                except:
                    pass

            # Supprimer du fichier
            keys_dir = f".{self.service_name}_keys"
            key_file = os.path.join(keys_dir, f"{key_id}.key")
            if os.path.exists(key_file):
                os.remove(key_file)

            # Supprimer les métadonnées
            del self.metadata[key_id]
            self._save_metadata()

            logger.info(f"Clé supprimée: {key_id}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la suppression de la clé: {e}")
            return False

    def list_keys(self, key_type: KeyType = None, tags: list[str] = None) -> list[KeyMetadata]:
        """
        Liste les clés disponibles

        Args:
            key_type: Filtrer par type de clé
            tags: Filtrer par tags

        Returns:
            List[KeyMetadata]: Liste des métadonnées
        """
        keys = list(self.metadata.values())

        # Filtrer par type
        if key_type:
            keys = [k for k in keys if k.key_type == key_type]

        # Filtrer par tags
        if tags:
            keys = [k for k in keys if any(tag in (k.tags or []) for tag in tags)]

        return keys

    def rotate_key(self, key_id: str, new_length: int = None) -> str | None:
        """
        Effectue une rotation de clé

        Args:
            key_id: Identifiant de la clé
            new_length: Nouvelle longueur (optionnel)

        Returns:
            str: Nouvelle clé ou None si erreur
        """
        if key_id not in self.metadata:
            return None

        try:
            metadata = self.metadata[key_id]

            # Générer une nouvelle clé
            length = new_length or 32
            if metadata.security_level == KeySecurity.CRITICAL:
                new_key = secrets.token_urlsafe(length)
            elif metadata.security_level == KeySecurity.HIGH:
                new_key = secrets.token_hex(length)
            else:
                new_key = secrets.token_urlsafe(length // 2)

            # Stocker la nouvelle clé
            self._store_key(key_id, new_key)

            # Mettre à jour les métadonnées
            metadata.created_at = datetime.now()
            metadata.usage_count = 0
            metadata.last_used = None
            self._save_metadata()

            logger.info(f"Clé rotée: {key_id}")
            return new_key

        except Exception as e:
            logger.error(f"Erreur lors de la rotation de la clé: {e}")
            return None

    def cleanup_expired_keys(self) -> int:
        """
        Nettoie les clés expirées

        Returns:
            int: Nombre de clés supprimées
        """
        now = datetime.now()
        expired_keys = []

        for key_id, metadata in self.metadata.items():
            if metadata.expires_at and now > metadata.expires_at:
                expired_keys.append(key_id)

        deleted_count = 0
        for key_id in expired_keys:
            if self.delete_key(key_id):
                deleted_count += 1

        if deleted_count > 0:
            logger.info(f"Clés expirées supprimées: {deleted_count}")

        return deleted_count

    def get_key_info(self, key_id: str) -> KeyMetadata | None:
        """
        Récupère les informations d'une clé

        Args:
            key_id: Identifiant de la clé

        Returns:
            KeyMetadata: Métadonnées de la clé
        """
        return self.metadata.get(key_id)

    def validate_key_strength(self, key: str) -> dict[str, Any]:
        """
        Valide la force d'une clé

        Args:
            key: Clé à valider

        Returns:
            Dict: Résultat de validation
        """
        result = {"valid": True, "strength": "weak", "issues": []}

        # Vérifier la longueur
        if len(key) < 16:
            result["issues"].append("Clé trop courte (< 16 caractères)")
            result["valid"] = False
        elif len(key) < 32:
            result["strength"] = "medium"
        elif len(key) >= 32:
            result["strength"] = "strong"

        # Vérifier la complexité
        has_upper = any(c.isupper() for c in key)
        has_lower = any(c.islower() for c in key)
        has_digit = any(c.isdigit() for c in key)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in key)

        complexity_score = sum([has_upper, has_lower, has_digit, has_special])

        if complexity_score < 2:
            result["issues"].append("Complexité insuffisante")
            result["strength"] = "weak"
        elif complexity_score < 3:
            result["strength"] = "medium"

        return result


# Instance globale du gestionnaire de clés
key_manager = KeyManager()


# Fonctions utilitaires
def get_api_key(service: str) -> str | None:
    """Récupère une clé API"""
    return key_manager.get_key(f"api_key_{service}")


def get_database_key() -> str | None:
    """Récupère la clé de base de données"""
    return key_manager.get_key("database_encryption_key")


def get_jwt_secret() -> str | None:
    """Récupère le secret JWT"""
    return key_manager.get_key("jwt_secret")


def generate_api_key(service: str, expires_in_days: int = 365) -> str:
    """Génère une clé API"""
    return key_manager.generate_key(
        key_id=f"api_key_{service}",
        key_type=KeyType.API_KEY,
        security_level=KeySecurity.HIGH,
        description=f"Clé API pour {service}",
        expires_in_days=expires_in_days,
    )


def rotate_all_keys() -> dict[str, bool]:
    """Effectue une rotation de toutes les clés"""
    results = {}
    for key_id in key_manager.metadata.keys():
        results[key_id] = key_manager.rotate_key(key_id) is not None
    return results
