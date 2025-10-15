"""
Service de suivi d'utilisation de l'API GPT pour CashZen
Gère les quotas d'utilisation par utilisateur, l'historique et les statistiques
"""

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

# Configuration du logging
logger = logging.getLogger(__name__)

# Quota quotidien par défaut
DAILY_GPT_QUOTA = 10

# Dossier de stockage des données
DATA_DIR = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))) / "data"
USAGE_FILE = DATA_DIR / "usage_data.json"
HISTORY_FILE = DATA_DIR / "gpt_history.json"


class UsageTracker:
    """
    Gestionnaire de quotas d'utilisation de l'API GPT
    Permet de suivre et limiter l'utilisation par utilisateur
    """

    def __init__(self) -> None:
        """
        Initialise le tracker d'utilisation
        """
        # Créer le répertoire de données si nécessaire
        if not DATA_DIR.exists():
            DATA_DIR.mkdir(parents=True, exist_ok=True)

        # Initialiser les fichiers de données s'ils n'existent pas
        self._init_storage()

    def _init_storage(self) -> None:
        """Initialise les fichiers de stockage s'ils n'existent pas"""
        # Fichier de suivi d'utilisation
        if not USAGE_FILE.exists():
            with open(USAGE_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)

        # Fichier d'historique
        if not HISTORY_FILE.exists():
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump({}, f)

    def _load_usage_data(self) -> dict[str, Any]:
        """
        Charge les données d'utilisation depuis le fichier

        Returns:
            Dict: Données d'utilisation
        """
        try:
            with open(USAGE_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données d'utilisation: {e}")
            return {}

    def _save_usage_data(self, data: dict[str, Any]) -> None:
        """
        Sauvegarde les données d'utilisation dans le fichier

        Args:
            data: Données à sauvegarder
        """
        try:
            with open(USAGE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données d'utilisation: {e}")

    def _check_and_reset(self, user_id: str) -> None:
        """
        Vérifie si le quota doit être réinitialisé pour l'utilisateur
        (si un jour est passé depuis la dernière utilisation)

        Args:
            user_id: Identifiant de l'utilisateur
        """
        try:
            data = self._load_usage_data()

            # Si l'utilisateur n'existe pas encore, pas besoin de vérifier
            if user_id not in data:
                return

            today = datetime.date.today().isoformat()
            last_reset_date = data[user_id].get("last_reset", "")

            # Si la date du jour est différente de la dernière date de réinitialisation,
            # réinitialiser le compteur
            if last_reset_date != today:
                data[user_id]["count"] = 0
                data[user_id]["last_reset"] = today
                self._save_usage_data(data)
                logger.info(f"Quota réinitialisé pour l'utilisateur {user_id}")
        except Exception as e:
            logger.error(f"Erreur lors de la vérification de réinitialisation: {e}")

    def increment_usage(self, user_id: str, amount: int = 1) -> None:
        """
        Incrémente le compteur d'utilisation pour un utilisateur

        Args:
            user_id: Identifiant de l'utilisateur
            amount: Montant à incrémenter (défaut: 1)
        """
        if not user_id:
            logger.warning("Tentative d'incrémenter l'usage avec un ID utilisateur vide")
            return

        # Vérifier et réinitialiser le compteur si nécessaire
        self._check_and_reset(user_id)

        try:
            data = self._load_usage_data()
            today = datetime.date.today().isoformat()

            # Créer l'entrée utilisateur si elle n'existe pas
            if user_id not in data:
                data[user_id] = {"count": 0, "last_reset": today}

            # Incrémenter le compteur
            data[user_id]["count"] += amount

            # Sauvegarder les modifications
            self._save_usage_data(data)
            logger.debug(f"Utilisation incrémentée pour {user_id}: {data[user_id]['count']}")
        except Exception as e:
            logger.error(f"Erreur lors de l'incrémentation de l'usage: {e}")

    def get_usage(self, user_id: str) -> int:
        """
        Récupère le nombre actuel d'utilisations pour un utilisateur

        Args:
            user_id: Identifiant de l'utilisateur

        Returns:
            int: Nombre d'utilisations
        """
        if not user_id:
            logger.warning("Tentative de récupérer l'usage avec un ID utilisateur vide")
            return 0

        # Vérifier et réinitialiser le compteur si nécessaire
        self._check_and_reset(user_id)

        try:
            data = self._load_usage_data()
            user_data = data.get(user_id, {})
            return user_data.get("count", 0)
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'usage: {e}")
            return 0

    def reset_daily_usage(self, user_id: str) -> None:
        """
        Réinitialise le compteur d'utilisation pour un utilisateur

        Args:
            user_id: Identifiant de l'utilisateur
        """
        if not user_id:
            logger.warning("Tentative de réinitialiser l'usage avec un ID utilisateur vide")
            return

        try:
            data = self._load_usage_data()
            today = datetime.date.today().isoformat()

            # Réinitialiser le compteur
            if user_id in data:
                data[user_id]["count"] = 0
                data[user_id]["last_reset"] = today
                self._save_usage_data(data)
                logger.info(f"Compteur réinitialisé pour l'utilisateur {user_id}")
        except Exception as e:
            logger.error(f"Erreur lors de la réinitialisation de l'usage: {e}")

    def is_quota_exceeded(self, user_id: str, is_premium: bool = False) -> bool:
        """
        Vérifie si l'utilisateur a dépassé son quota

        Args:
            user_id: Identifiant de l'utilisateur
            is_premium: Indique si l'utilisateur est premium (ignore les quotas)

        Returns:
            bool: True si le quota est dépassé, False sinon
        """
        # Les utilisateurs premium ne sont pas limités
        if is_premium:
            return False

        usage = self.get_usage(user_id)
        return usage >= DAILY_GPT_QUOTA

    def get_remaining_requests(self, user_id: str, is_premium: bool = False) -> int:
        """
        Calcule le nombre de requêtes restantes pour un utilisateur

        Args:
            user_id: Identifiant de l'utilisateur
            is_premium: Indique si l'utilisateur est premium

        Returns:
            int: Nombre de requêtes restantes (infini pour premium)
        """
        # Les utilisateurs premium ont des requêtes illimitées
        if is_premium:
            return float('inf')

        usage = self.get_usage(user_id)
        return max(0, DAILY_GPT_QUOTA - usage)

    def get_remaining_quota(self, user_id: str, limit: int = DAILY_GPT_QUOTA) -> int:
        """
        Calcule le nombre de requêtes restantes pour un utilisateur (fonction de compatibilité)

        Args:
            user_id: Identifiant de l'utilisateur
            limit: Limite de quota à utiliser

        Returns:
            int: Nombre de requêtes restantes
        """
        usage = self.get_usage(user_id)
        return max(0, limit - usage)

    def add_free_request(self, user_id: str) -> bool:
        """
        Ajoute une requête gratuite (après avoir regardé une publicité)
        Cette fonction ne peut être utilisée qu'une fois par jour.

        Args:
            user_id: Identifiant de l'utilisateur

        Returns:
            bool: True si la requête a été ajoutée, False sinon
        """
        if not user_id:
            logger.warning("Tentative d'ajouter une requête gratuite avec un ID utilisateur vide")
            return False

        # Vérifier et réinitialiser le compteur si nécessaire
        self._check_and_reset(user_id)

        try:
            data = self._load_usage_data()
            today = datetime.date.today().isoformat()

            # Créer l'entrée utilisateur si elle n'existe pas
            if user_id not in data:
                data[user_id] = {"count": 0, "last_reset": today, "free_request_used": False}

            # Vérifier si l'utilisateur a déjà utilisé sa requête gratuite aujourd'hui
            if data[user_id].get("free_request_used", False):
                logger.info(f"L'utilisateur {user_id} a déjà utilisé sa requête gratuite aujourd'hui")
                return False

            # Ajouter une requête gratuite en décrémentant le compteur
            self.increment_usage(user_id, amount=-1)

            # Marquer la requête gratuite comme utilisée
            data[user_id]["free_request_used"] = True
            self._save_usage_data(data)

            logger.info(f"Requête gratuite ajoutée pour l'utilisateur {user_id}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout d'une requête gratuite: {e}")
            return False

    def get_all_usage_stats(self) -> dict[str, Any]:
        """
        Récupère les statistiques d'utilisation pour tous les utilisateurs

        Returns:
            Dict: Statistiques d'utilisation
        """
        try:
            data = self._load_usage_data()

            stats = {
                "total_users": len(data),
                "total_requests": sum(user_data.get("count", 0) for user_data in data.values()),
                "users_at_limit": sum(1 for user_data in data.values() if user_data.get("count", 0) >= DAILY_GPT_QUOTA),
                "user_details": data,
            }

            return stats
        except Exception as e:
            logger.error(f"Erreur lors de la récupération des statistiques: {e}")
            return {"error": str(e)}

    def log_prompt(self, user_id: str, prompt: str, model: str) -> None:
        """
        Enregistre un prompt dans l'historique

        Args:
            user_id: Identifiant de l'utilisateur
            prompt: Texte du prompt
            model: Modèle utilisé (ex: gpt-3.5, gpt-4)
        """
        if not user_id:
            logger.warning("Tentative de journaliser un prompt avec un ID utilisateur vide")
            return

        try:
            # Charger l'historique existant
            history_data = self._load_history_data()

            # Créer l'entrée utilisateur si elle n'existe pas
            if user_id not in history_data:
                history_data[user_id] = []

            # Ajouter l'entrée à l'historique
            history_data[user_id].append(
                {"prompt": prompt, "model": model, "timestamp": datetime.datetime.now().isoformat()}
            )

            # Limiter la taille de l'historique (garder les 100 dernières entrées)
            if len(history_data[user_id]) > 100:
                history_data[user_id] = history_data[user_id][-100:]

            # Sauvegarder l'historique
            self._save_history_data(history_data)
        except Exception as e:
            logger.error(f"Erreur lors de l'enregistrement du prompt: {e}")

    def get_history(self, user_id: str) -> list[dict[str, Any]]:
        """
        Récupère l'historique des prompts pour un utilisateur

        Args:
            user_id: Identifiant de l'utilisateur

        Returns:
            List: Liste des entrées d'historique
        """
        if not user_id:
            logger.warning("Tentative de récupérer l'historique avec un ID utilisateur vide")
            return []

        try:
            history_data = self._load_history_data()
            return history_data.get(user_id, [])
        except Exception as e:
            logger.error(f"Erreur lors de la récupération de l'historique: {e}")
            return []

    def _load_history_data(self) -> dict[str, list[dict[str, Any]]]:
        """
        Charge les données d'historique depuis le fichier

        Returns:
            Dict: Données d'historique
        """
        try:
            with open(HISTORY_FILE, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données d'historique: {e}")
            return {}

    def _save_history_data(self, data: dict[str, list[dict[str, Any]]]) -> None:
        """
        Sauvegarde les données d'historique dans le fichier

        Args:
            data: Données à sauvegarder
        """
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données d'historique: {e}")


# Singleton pour accès global au tracker
_tracker_instance: UsageTracker | None = None


def get_usage_tracker() -> UsageTracker:
    """
    Accède à l'instance singleton du tracker d'utilisation

    Returns:
        UsageTracker: Instance du tracker
    """
    # pylint: disable=global-statement
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = UsageTracker()
    return _tracker_instance
