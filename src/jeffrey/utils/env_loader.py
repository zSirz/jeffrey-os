#!/usr/bin/env python

"""
Module utilitaire pour le chargement des variables d'environnement.
Version unifiée pour Mac et iOS (Pythonista), compatible mode avion.
"""

import logging
import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv

# Configuration du logger
logger = logging.getLogger(__name__)


def detecter_plateforme() -> tuple[str, str]:
    """
    Détecte la plateforme et le chemin iCloud approprié.

    Returns:
        Tuple[str, str]: (plateforme, chemin_base)
    """
    is_pythonista = "Pythonista" in sys.executable
    system = platform.system()

    if is_pythonista:
        # iPhone/iPad via Pythonista
        base_path = os.path.expanduser("~/Documents/Inbox/Jeffrey_Memoire")
        platform_name = "iOS"
    elif system == "Darwin":
        # Mac
        base_path = os.path.expanduser("~/Documents/Jeffrey_Memoire")
        platform_name = "macOS"
    else:
        # Fallback
        base_path = os.path.expanduser("~/Documents/Jeffrey_Local")
        platform_name = "unknown"

    return platform_name, base_path


def get_env_path() -> str | None:
    """
    Retourne le chemin vers le fichier .env selon la plateforme.

    Returns:
        Optional[str]: Chemin vers le fichier .env ou None si non trouvé
    """
    platform_name, base_path = detecter_plateforme()
    env_path = os.path.join(base_path, ".env")

    # Créer le répertoire s'il n'existe pas
    Path(base_path).mkdir(parents=True, exist_ok=True)

    return env_path if os.path.exists(env_path) else None


def charger_env_fallback_manuel(env_path: str) -> None:
    """
    Charge manuellement les variables d'environnement depuis le fichier .env
    Utilisé comme fallback sur iOS/Pythonista où python-dotenv peut ne pas fonctionner correctement.

    Args:
        env_path (str): Chemin vers le fichier .env
    """
    try:
        if not os.path.exists(env_path):
            logger.warning(f"Fichier .env non trouvé à {env_path}")
            return

        with open(env_path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip()
                    value = value.strip().strip('"\'')
                    os.environ[key] = value
                    logger.debug(f"Variable chargée manuellement : {key}")
    except Exception as e:
        logger.error(f"Erreur lors du chargement manuel des variables : {e}")


def charger_env() -> bool:
    """
    Charge les variables d'environnement depuis le fichier .env
    avec fallback automatique selon la plateforme.

    Returns:
        bool: True si le chargement a réussi, False sinon
    """
    env_path = get_env_path()
    if not env_path:
        logger.warning("Aucun fichier .env trouvé")
        return False

    # Détection de la plateforme
    platform_name, _ = detecter_plateforme()

    try:
        # Méthode standard avec python-dotenv
        load_dotenv(env_path)

        # Fallback manuel pour iOS/Pythonista
        if platform_name == "iOS":
            charger_env_fallback_manuel(env_path)

        # Vérification des clés API essentielles
        openai_key = os.getenv('OPENAI_API_KEY')
        eleven_key = os.getenv('ELEVEN_API_KEY')

        if not openai_key or not eleven_key:
            logger.warning("Certaines clés API essentielles sont manquantes")
            return False

        logger.info(f"Variables d'environnement chargées avec succès depuis {env_path}")
        return True

    except Exception as e:
        logger.error(f"Erreur lors du chargement des variables d'environnement : {e}")
        # Tentative de fallback manuel en cas d'échec
        charger_env_fallback_manuel(env_path)
        return bool(os.getenv('OPENAI_API_KEY') and os.getenv('ELEVEN_API_KEY'))


if __name__ == "__main__":
    # Configuration du logging pour les tests
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    # Test du chargement
    success = charger_env()
    print(f"Chargement des variables d'environnement : {'Succès' if success else 'Échec'}")

    # Affichage des variables (masquées)
    openai_key = os.getenv('OPENAI_API_KEY')
    eleven_key = os.getenv('ELEVEN_API_KEY')

    if openai_key:
        print(f"OPENAI_API_KEY : {'*' * (len(openai_key) - 4) + openai_key[-4:]}")
    if eleven_key:
        print(f"ELEVEN_API_KEY : {'*' * (len(eleven_key) - 4) + eleven_key[-4:]}")
