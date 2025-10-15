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

import base64
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# 🔧 PATCH DEV/PROD SWITCH - Ajouté automatiquement
BYPASS_SECURECONFIG = os.getenv("JEFFREY_BYPASS_SECURECONFIG", "false").lower() == "true"

if BYPASS_SECURECONFIG:
    print("🔓 [SECURECONFIG] Bypass activé : toutes les clés sont chargées en clair depuis .env")
    print("   ⚠️  MODE DÉVELOPPEMENT - Ne pas utiliser en production !")
else:
    print("🔒 [SECURECONFIG] Chiffrement SecureConfig ACTIVÉ !")


def get_api_key_with_bypass(key: str) -> str:
    """Récupère une clé API avec support bypass DEV/PROD"""
    if BYPASS_SECURECONFIG:
        # MODE DEV : charge directement depuis l'environnement
        variations = [
            key,
            key.upper(),
            f"{key.upper()}_API_KEY",
            key.replace("_API_KEY", "").upper() + "_API_KEY",
        ]

        for var in variations:
            val = os.getenv(var)
            if val and len(val) > 10:  # Clé valide
                return val

        raise SecureConfigError(f"❌ Clé manquante pour {key} (mode bypass DEV)")

    # MODE PROD : utilise le système chiffré normal
    # (Le code normal suit après ce patch)
    return None  # Placeholder, le code normal prendra le relais


@dataclass
class APIKeyConfig:
    """Configuration d'une clé API"""

    key: str
    service: str
    created_at: str
    last_rotated: str | None = None
    expires_at: str | None = None


class SecureConfigError(Exception):
    """Erreur de configuration sécurisée"""

    pass


class SecureConfig:
    """
    Gestionnaire sécurisé des clés API et secrets

    Features:
    - Chiffrement Fernet des clés sensibles
    - Validation obligatoire au démarrage
    - Rotation automatique des clés
    - Cache mémoire sécurisé
    """

    def __init__(self, config_path: str | None = None, skip_validation: bool = False) -> None:
        # 🔧 PATCH DEV/PROD : Bypass complet si demandé
        if BYPASS_SECURECONFIG:
            self.config_path = None
            self.salt_path = None
            self._cache: dict[str, Any] = {}
            self._cipher: Fernet | None = None
            self._master_key: str | None = None

            # Charger .env pour bypass
            load_dotenv()

            logger.info("🔓 SecureConfig bypassé - Mode développement")
            return

        # MODE PROD : Logique normale
        self.config_path = Path(config_path or os.path.join(os.path.dirname(__file__), "..", ".env.encrypted"))
        self.salt_path = Path(str(self.config_path).replace(".encrypted", ".salt"))
        self._cache: dict[str, Any] = {}
        self._cipher: Fernet | None = None
        self._master_key: str | None = None

        # Charger les variables d'environnement non-chiffrées pour la transition
        load_dotenv()

        self._initialize_encryption()
        self._load_config()

        # Ne valider les clés que si explicitement demandé (sauf pour le mode setup)
        if not skip_validation:
            self._validate_required_keys()

    def _initialize_encryption(self):
        """Initialise le système de chiffrement"""
        try:
            # Récupérer ou générer la clé maître
            self._master_key = os.getenv("JEFFREY_MASTER_KEY")
            if not self._master_key:
                # Générer une nouvelle clé maître si elle n'existe pas
                self._master_key = secrets.token_urlsafe(32)
                logger.warning(
                    "🔑 Nouvelle clé maître générée. Sauvegardez: JEFFREY_MASTER_KEY=%s",
                    self._master_key,
                )

            # Générer ou charger le salt
            if self.salt_path.exists():
                with open(self.salt_path, "rb") as f:
                    salt = f.read()
            else:
                salt = os.urandom(16)
                with open(self.salt_path, "wb") as f:
                    f.write(salt)
                # Sécuriser le fichier salt
                os.chmod(self.salt_path, 0o600)

            # Dériver la clé de chiffrement
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=480000,  # Haute sécurité
            )
            key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
            self._cipher = Fernet(key)

            logger.info("🔐 Système de chiffrement initialisé")

        except Exception as e:
            raise SecureConfigError(f"Erreur d'initialisation du chiffrement: {e}")

    def _load_config(self):
        """Charge la configuration chiffrée"""
        if not self.config_path.exists():
            logger.info("📁 Fichier de configuration chiffré non trouvé, migration nécessaire")
            self._migrate_from_env()
            return

        try:
            with open(self.config_path, "rb") as f:
                encrypted_data = f.read()

            decrypted_data = self._cipher.decrypt(encrypted_data)
            config_data = json.loads(decrypted_data.decode())

            for key, value in config_data.items():
                self._cache[key] = value

            logger.info("✅ Configuration sécurisée chargée avec %d clés", len(self._cache))

        except Exception as e:
            logger.error("❌ Erreur de chargement de la configuration: %s", e)
            raise SecureConfigError(f"Impossible de charger la configuration chiffrée: {e}")

    def _migrate_from_env(self):
        """Migre les clés depuis .env vers le format chiffré"""
        logger.info("🔄 Migration des clés depuis .env vers format chiffré")

        # Clés sensibles à migrer
        sensitive_keys = {
            "OPENAI_API_KEY": "OpenAI API Key",
            "ELEVENLABS_API_KEY": "ElevenLabs API Key",
            "ANTHROPIC_API_KEY": "Anthropic API Key",
            "GEMINI_API_KEY": "Gemini API Key",
            "MISTRAL_API_KEY": "Mistral API Key",
            "COHERE_API_KEY": "Cohere API Key",
            "JEFFREY_SECRET_KEY": "Jeffrey Secret Key",
            "JEFFREY_DEBUG_KEY": "Jeffrey Debug Key",
        }

        migrated_count = 0
        for env_key, description in sensitive_keys.items():
            value = os.getenv(env_key)
            if value and value != "your_key_here":
                self.set_secret(env_key, value, description)
                migrated_count += 1

        if migrated_count > 0:
            self._save_config()
            logger.info("✅ Migration terminée: %d clés sécurisées", migrated_count)
        else:
            logger.warning("⚠️ Aucune clé trouvée à migrer")

    def _validate_required_keys(self):
        """Valide que toutes les clés requises sont présentes"""
        required_keys = ["OPENAI_API_KEY", "ELEVENLABS_API_KEY"]

        missing_keys = []
        for key in required_keys:
            if not self.get_secret(key):
                missing_keys.append(key)

        if missing_keys:
            raise SecureConfigError(
                f"❌ Clés API manquantes: {', '.join(missing_keys)}. "
                f"Configurez-les avec set_secret() ou dans les variables d'environnement."
            )

        logger.info("✅ Validation des clés requises: OK")

    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """Récupère une clé secrète du cache sécurisé"""
        # Vérifier d'abord le cache chiffré
        cached_value = self._cache.get(key)
        if cached_value:
            if isinstance(cached_value, dict) and "key" in cached_value:
                return cached_value["key"]
            return cached_value

        # Fallback sur les variables d'environnement (pour la transition)
        env_value = os.getenv(key)
        if env_value and env_value != "your_key_here":
            return env_value

        return default

    def set_secret(self, key: str, value: str, description: str = "") -> None:
        """Stocke une clé secrète de manière chiffrée"""
        config_entry = APIKeyConfig(key=value, service=description, created_at=datetime.now().isoformat())

        self._cache[key] = config_entry.__dict__
        logger.info("🔑 Clé '%s' sécurisée: %s", key, description)

    def rotate_key(self, key: str, new_value: str):
        """Fait la rotation d'une clé API"""
        if key not in self._cache:
            raise SecureConfigError(f"Clé '{key}' non trouvée pour rotation")

        old_config = self._cache[key]
        if isinstance(old_config, dict):
            old_config["key"] = new_value
            old_config["last_rotated"] = datetime.now().isoformat()
        else:
            # Format legacy
            self._cache[key] = new_value

        logger.info("🔄 Rotation de la clé '%s' effectuée", key)

    def _save_config(self):
        """Sauvegarde la configuration chiffrée"""
        try:
            config_json = json.dumps(self._cache, indent=2, ensure_ascii=False)
            encrypted_data = self._cipher.encrypt(config_json.encode())

            # Sauvegarde atomique
            temp_path = Path(str(self.config_path) + ".tmp")
            with open(temp_path, "wb") as f:
                f.write(encrypted_data)

            # Sécuriser le fichier
            os.chmod(temp_path, 0o600)

            # Renommage atomique
            temp_path.rename(self.config_path)

            logger.info("💾 Configuration sécurisée sauvegardée")

        except Exception as e:
            logger.error("❌ Erreur de sauvegarde: %s", e)
            raise SecureConfigError(f"Impossible de sauvegarder la configuration: {e}")

    def save(self):
        """Sauvegarde publique"""
        self._save_config()

    def list_keys(self) -> dict[str, str]:
        """Liste les clés configurées (sans les valeurs)"""
        result = {}
        for key, value in self._cache.items():
            if isinstance(value, dict) and "service" in value:
                result[key] = value["service"]
            else:
                result[key] = "Legacy key"
        return result

    def validate_key(self, key: str) -> bool:
        """Valide une clé avec support bypass"""
        if BYPASS_SECURECONFIG:
            try:
                val = get_api_key_with_bypass(key)
                return bool(val and len(val) > 10)
            except SecureConfigError:
                return False

        # MODE PROD : Validation normale
        """Valide qu'une clé est présente et non vide"""
        value = self.get_secret(key)
        return value is not None and len(value.strip()) > 0

    def get_stats(self) -> dict[str, Any]:
        """Statistiques de configuration"""
        return {
            "total_keys": len(self._cache),
            "config_file_exists": self.config_path.exists(),
            "encryption_active": self._cipher is not None,
            "required_keys_valid": all(self.validate_key(k) for k in ["OPENAI_API_KEY", "ELEVENLABS_API_KEY"]),
        }


# Instance globale pour l'application
_secure_config: SecureConfig | None = None


def get_secure_config(skip_validation: bool = False) -> SecureConfig:
    """Récupère l'instance globale de SecureConfig (Singleton)"""
    global _secure_config
    if _secure_config is None:
        _secure_config = SecureConfig(skip_validation=skip_validation)
    return _secure_config


def get_api_key(service: str) -> str:
    """
    Récupère une clé API de manière sécurisée

    Args:
        service: Nom du service (openai, elevenlabs, anthropic, etc.)

    Returns:
        Clé API ou lève une exception si non trouvée
    """
    config = get_secure_config()

    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
    }

    env_key = key_mapping.get(service.lower())
    if not env_key:
        raise SecureConfigError(f"Service inconnu: {service}")

    api_key = config.get_secret(env_key)
    if not api_key:
        raise SecureConfigError(f"Clé API manquante pour {service} ({env_key})")

    return api_key


# Fonctions de convenance
def validate_all_keys() -> bool:
    """Valide toutes les clés requises"""
    try:
        config = get_secure_config()
        return config.get_stats()["required_keys_valid"]
    except Exception:
        return False


if __name__ == "__main__":
    # Test du système
    try:
        config = SecureConfig()
        print("✅ SecureConfig initialisé avec succès")
        print(f"📊 Stats: {config.get_stats()}")
        print(f"🔑 Clés configurées: {list(config.list_keys().keys())}")
    except Exception as e:
        print(f"❌ Erreur: {e}")
