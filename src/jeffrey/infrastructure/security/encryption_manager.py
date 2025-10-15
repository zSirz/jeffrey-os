"""
Gestionnaire de chiffrement et protection.

Ce module implémente les fonctionnalités essentielles pour gestionnaire de chiffrement et protection.
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
import hashlib
import json
import logging
import os
import secrets
from dataclasses import dataclass
from enum import Enum
from typing import Any

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)


class EncryptionLevel(Enum):
    """Niveaux de chiffrement"""

    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    MILITARY = "military"


@dataclass
class EncryptionConfig:
    """Configuration de chiffrement"""

    key_size: int = 256
    salt_size: int = 16
    iv_size: int = 16
    iterations: int = 100000
    algorithm: str = "AES-256-GCM"
    level: EncryptionLevel = EncryptionLevel.STANDARD


class EncryptionManager:
    """Gestionnaire de chiffrement sécurisé"""

    def __init__(self, config: EncryptionConfig = None) -> None:
        self.config = config or EncryptionConfig()
        self.backend = default_backend()
        self._master_key = None
        self._rsa_key = None

        # Initialiser le gestionnaire
        self._initialize()

    def _initialize(self):
        """Initialise le gestionnaire de chiffrement"""
        try:
            # Charger ou générer la clé maître
            self._load_or_generate_master_key()

            # Charger ou générer les clés RSA
            self._load_or_generate_rsa_keys()

            logger.info("Gestionnaire de chiffrement initialisé")
        except Exception as e:
            logger.error(f"Erreur lors de l'initialisation du chiffrement: {e}")
            raise

    def _load_or_generate_master_key(self):
        """Charge ou génère la clé maître"""
        key_file = ".encryption_key"

        if os.path.exists(key_file):
            try:
                with open(key_file, "rb") as f:
                    encrypted_key = f.read()

                # Déchiffrer la clé avec un mot de passe système
                password = self._get_system_password()
                self._master_key = self._decrypt_master_key(encrypted_key, password)

                logger.info("Clé maître chargée depuis le fichier")
            except Exception as e:
                logger.warning(f"Impossible de charger la clé maître: {e}")
                self._generate_new_master_key()
        else:
            self._generate_new_master_key()

    def _generate_new_master_key(self):
        """Génère une nouvelle clé maître"""
        try:
            # Générer une clé aléatoire sécurisée
            self._master_key = secrets.token_bytes(32)  # 256 bits

            # Chiffrer et sauvegarder la clé
            password = self._get_system_password()
            encrypted_key = self._encrypt_master_key(self._master_key, password)

            with open(".encryption_key", "wb") as f:
                f.write(encrypted_key)

            # Sécuriser le fichier (permission lecture seule propriétaire)
            os.chmod(".encryption_key", 0o600)

            logger.info("Nouvelle clé maître générée et sauvegardée")
        except Exception as e:
            logger.error(f"Erreur lors de la génération de la clé maître: {e}")
            raise

    def _get_system_password(self) -> bytes:
        """Récupère le mot de passe système pour chiffrer la clé maître"""
        # Utiliser des informations système pour créer un mot de passe unique
        import platform

        system_info = f"{platform.node()}-{platform.system()}-{platform.release()}"
        return hashlib.sha256(system_info.encode()).digest()

    def _encrypt_master_key(self, key: bytes, password: bytes) -> bytes:
        """Chiffre la clé maître avec un mot de passe"""
        salt = secrets.token_bytes(16)

        # Dérivation de clé PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.iterations,
            backend=self.backend,
        )
        derived_key = kdf.derive(password)

        # Chiffrement AES-GCM
        iv = secrets.token_bytes(16)
        cipher = Cipher(algorithms.AES(derived_key), modes.GCM(iv), backend=self.backend)
        encryptor = cipher.encryptor()

        ciphertext = encryptor.update(key) + encryptor.finalize()

        # Retourner salt + iv + tag + ciphertext
        return salt + iv + encryptor.tag + ciphertext

    def _decrypt_master_key(self, encrypted_data: bytes, password: bytes) -> bytes:
        """Déchiffre la clé maître"""
        # Extraire les composants
        salt = encrypted_data[:16]
        iv = encrypted_data[16:32]
        tag = encrypted_data[32:48]
        ciphertext = encrypted_data[48:]

        # Dérivation de clé
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self.config.iterations,
            backend=self.backend,
        )
        derived_key = kdf.derive(password)

        # Déchiffrement AES-GCM
        cipher = Cipher(algorithms.AES(derived_key), modes.GCM(iv, tag), backend=self.backend)
        decryptor = cipher.decryptor()

        return decryptor.update(ciphertext) + decryptor.finalize()

    def _load_or_generate_rsa_keys(self):
        """Charge ou génère les clés RSA"""
        private_key_file = ".rsa_private_key"
        public_key_file = ".rsa_public_key"

        if os.path.exists(private_key_file) and os.path.exists(public_key_file):
            try:
                with open(private_key_file, "rb") as f:
                    private_pem = f.read()

                # Déchiffrer la clé privée
                password = self._get_system_password()
                self._rsa_key = serialization.load_pem_private_key(private_pem, password=password, backend=self.backend)

                logger.info("Clés RSA chargées depuis les fichiers")
            except Exception as e:
                logger.warning(f"Impossible de charger les clés RSA: {e}")
                self._generate_new_rsa_keys()
        else:
            self._generate_new_rsa_keys()

    def _generate_new_rsa_keys(self):
        """Génère de nouvelles clés RSA"""
        try:
            # Générer une paire de clés RSA 2048 bits
            self._rsa_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=self.backend)

            # Sauvegarder la clé privée chiffrée
            password = self._get_system_password()
            private_pem = self._rsa_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.BestAvailableEncryption(password),
            )

            with open(".rsa_private_key", "wb") as f:
                f.write(private_pem)

            # Sauvegarder la clé publique
            public_key = self._rsa_key.public_key()
            public_pem = public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo,
            )

            with open(".rsa_public_key", "wb") as f:
                f.write(public_pem)

            # Sécuriser les fichiers
            os.chmod(".rsa_private_key", 0o600)
            os.chmod(".rsa_public_key", 0o644)

            logger.info("Nouvelles clés RSA générées et sauvegardées")
        except Exception as e:
            logger.error(f"Erreur lors de la génération des clés RSA: {e}")
            raise

    def encrypt_data(self, data: str | bytes | dict[str, Any], level: EncryptionLevel = None) -> str:
        """
        Chiffre des données

        Args:
            data: Données à chiffrer
            level: Niveau de chiffrement

        Returns:
            str: Données chiffrées en base64
        """
        if not self._master_key:
            raise ValueError("Clé maître non initialisée")

        level = level or self.config.level

        try:
            # Convertir en bytes si nécessaire
            if isinstance(data, dict):
                data_bytes = json.dumps(data, ensure_ascii=False).encode("utf-8")
            elif isinstance(data, str):
                data_bytes = data.encode("utf-8")
            else:
                data_bytes = data

            # Générer IV aléatoire
            iv = secrets.token_bytes(16)

            # Chiffrement AES-GCM
            cipher = Cipher(algorithms.AES(self._master_key), modes.GCM(iv), backend=self.backend)
            encryptor = cipher.encryptor()

            # Ajouter des données d'authentification si niveau élevé
            if level in [EncryptionLevel.HIGH, EncryptionLevel.MILITARY]:
                timestamp = str(int(os.urandom(8).hex(), 16))
                encryptor.authenticate_additional_data(timestamp.encode())

            ciphertext = encryptor.update(data_bytes) + encryptor.finalize()

            # Construire le résultat: iv + tag + ciphertext
            encrypted_data = iv + encryptor.tag + ciphertext

            # Encoder en base64
            return base64.b64encode(encrypted_data).decode("utf-8")

        except Exception as e:
            logger.error(f"Erreur lors du chiffrement: {e}")
            raise

    def decrypt_data(self, encrypted_data: str, expected_type: type = str) -> Any:
        """
        Déchiffre des données

        Args:
            encrypted_data: Données chiffrées en base64
            expected_type: Type attendu des données

        Returns:
            Any: Données déchiffrées
        """
        if not self._master_key:
            raise ValueError("Clé maître non initialisée")

        try:
            # Décoder base64
            data_bytes = base64.b64decode(encrypted_data.encode("utf-8"))

            # Extraire les composants
            iv = data_bytes[:16]
            tag = data_bytes[16:32]
            ciphertext = data_bytes[32:]

            # Déchiffrement AES-GCM
            cipher = Cipher(
                algorithms.AES(self._master_key),
                modes.GCM(iv, tag),
                backend=self.backend,
            )
            decryptor = cipher.decryptor()

            plaintext = decryptor.update(ciphertext) + decryptor.finalize()

            # Convertir selon le type attendu
            if expected_type == dict:
                return json.loads(plaintext.decode("utf-8"))
            elif expected_type == str:
                return plaintext.decode("utf-8")
            else:
                return plaintext

        except Exception as e:
            logger.error(f"Erreur lors du déchiffrement: {e}")
            raise

    def encrypt_file(self, file_path: str, output_path: str = None) -> str:
        """
        Chiffre un fichier

        Args:
            file_path: Chemin du fichier à chiffrer
            output_path: Chemin de sortie (optionnel)

        Returns:
            str: Chemin du fichier chiffré
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {file_path}")

        output_path = output_path or f"{file_path}.encrypted"

        try:
            # Lire le fichier
            with open(file_path, "rb") as f:
                data = f.read()

            # Chiffrer les données
            encrypted_data = self.encrypt_data(data)

            # Écrire le fichier chiffré
            with open(output_path, "w") as f:
                f.write(encrypted_data)

            # Sécuriser le fichier
            os.chmod(output_path, 0o600)

            logger.info(f"Fichier chiffré: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Erreur lors du chiffrement du fichier: {e}")
            raise

    def decrypt_file(self, encrypted_file_path: str, output_path: str = None) -> str:
        """
        Déchiffre un fichier

        Args:
            encrypted_file_path: Chemin du fichier chiffré
            output_path: Chemin de sortie (optionnel)

        Returns:
            str: Chemin du fichier déchiffré
        """
        if not os.path.exists(encrypted_file_path):
            raise FileNotFoundError(f"Fichier non trouvé: {encrypted_file_path}")

        output_path = output_path or encrypted_file_path.replace(".encrypted", "")

        try:
            # Lire le fichier chiffré
            with open(encrypted_file_path) as f:
                encrypted_data = f.read()

            # Déchiffrer les données
            data = self.decrypt_data(encrypted_data, bytes)

            # Écrire le fichier déchiffré
            with open(output_path, "wb") as f:
                f.write(data)

            logger.info(f"Fichier déchiffré: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Erreur lors du déchiffrement du fichier: {e}")
            raise

    def encrypt_database_field(self, value: Any, field_name: str = None) -> str:
        """
        Chiffre un champ de base de données

        Args:
            value: Valeur à chiffrer
            field_name: Nom du champ (pour les métadonnées)

        Returns:
            str: Valeur chiffrée
        """
        if value is None:
            return None

        # Ajouter des métadonnées si nécessaire
        if field_name:
            metadata = {
                "field": field_name,
                "timestamp": int(os.urandom(4).hex(), 16),
                "value": value,
            }
            return self.encrypt_data(metadata, EncryptionLevel.HIGH)
        else:
            return self.encrypt_data(value, EncryptionLevel.STANDARD)

    def decrypt_database_field(self, encrypted_value: str, field_name: str = None) -> Any:
        """
        Déchiffre un champ de base de données

        Args:
            encrypted_value: Valeur chiffrée
            field_name: Nom du champ attendu

        Returns:
            Any: Valeur déchiffrée
        """
        if encrypted_value is None:
            return None

        try:
            # Tenter de déchiffrer comme métadonnées
            if field_name:
                metadata = self.decrypt_data(encrypted_value, dict)
                if metadata.get("field") == field_name:
                    return metadata["value"]
                else:
                    raise ValueError(f"Champ inattendu: {metadata.get('field')}")
            else:
                return self.decrypt_data(encrypted_value, str)

        except:
            # Fallback: déchiffrer comme valeur simple
            return self.decrypt_data(encrypted_value, str)

    def generate_hash(self, data: str | bytes) -> str:
        """
        Génère un hash SHA-256 sécurisé

        Args:
            data: Données à hasher

        Returns:
            str: Hash hexadécimal
        """
        if isinstance(data, str):
            data = data.encode("utf-8")

        return hashlib.sha256(data).hexdigest()

    def verify_integrity(self, data: str | bytes, expected_hash: str) -> bool:
        """
        Vérifie l'intégrité des données

        Args:
            data: Données à vérifier
            expected_hash: Hash attendu

        Returns:
            bool: True si l'intégrité est vérifiée
        """
        actual_hash = self.generate_hash(data)
        return actual_hash == expected_hash

    def secure_delete(self, file_path: str, passes: int = 3) -> bool:
        """
        Suppression sécurisée d'un fichier

        Args:
            file_path: Chemin du fichier à supprimer
            passes: Nombre de passes d'écrasement

        Returns:
            bool: True si la suppression a réussi
        """
        if not os.path.exists(file_path):
            return True

        try:
            # Écraser le fichier plusieurs fois
            file_size = os.path.getsize(file_path)

            with open(file_path, "rb+") as f:
                for _ in range(passes):
                    f.seek(0)
                    f.write(secrets.token_bytes(file_size))
                    f.flush()
                    os.fsync(f.fileno())

            # Supprimer le fichier
            os.remove(file_path)

            logger.info(f"Fichier supprimé de manière sécurisée: {file_path}")
            return True

        except Exception as e:
            logger.error(f"Erreur lors de la suppression sécurisée: {e}")
            return False


# Instance globale du gestionnaire
encryption_manager = EncryptionManager()


# Fonctions utilitaires
def encrypt(data: Any, level: EncryptionLevel = EncryptionLevel.STANDARD) -> str:
    """Chiffre des données"""
    return encryption_manager.encrypt_data(data, level)


def decrypt(encrypted_data: str, expected_type: type = str) -> Any:
    """Déchiffre des données"""
    return encryption_manager.decrypt_data(encrypted_data, expected_type)


def encrypt_sensitive(data: Any) -> str:
    """Chiffre des données sensibles (niveau élevé)"""
    return encryption_manager.encrypt_data(data, EncryptionLevel.HIGH)


def hash_data(data: str | bytes) -> str:
    """Génère un hash sécurisé"""
    return encryption_manager.generate_hash(data)
