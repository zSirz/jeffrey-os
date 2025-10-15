"""
Module de synchronisation émotionnelle iCloud pour Jeffrey.
Gère la persistance et la synchronisation des émotions entre Mac et iPhone.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Literal

# Configuration du logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class EmotionalSync:
    """
    Gère la synchronisation des émotions et du profil émotionnel via iCloud.
    """

    def __init__(self):
        # Chemins par défaut
        self.icloud_path = None
        self.local_path = None

        # Données en mémoire
        self.data: dict[str, Any] = {
            "emotional_history": [],  # type: List[Dict[str, Any]]
            "profile": {
                "mode": "neutre",  # "neutre", "innocent", "coquin"
                "last_updated": None,
                "user": None,
            },
            "last_sync": None,
            "sync_status": {"last_icloud_sync": None, "last_local_sync": None, "is_online": True},
        }

        # Vérifications de sécurité après chargement
        self._ensure_data_structure()

    def set_storage_paths(self, icloud_path: str, local_path: str) -> None:
        """
        Configure les chemins de stockage pour iCloud et le fallback local.

        Args:
            icloud_path: Chemin vers le fichier iCloud
            local_path: Chemin vers le fichier local de fallback
        """
        self.icloud_path = icloud_path
        self.local_path = local_path

        # Créer les dossiers si nécessaire
        for path in [os.path.dirname(icloud_path), os.path.dirname(local_path)]:
            os.makedirs(path, exist_ok=True)

        # Charger les données
        self._load_data()

    def _ensure_data_structure(self) -> None:
        """Garantit la structure correcte des données."""
        corrections_appliquees: list[str] = []

        # Vérification de emotional_history
        if not isinstance(self.data.get("emotional_history"), list):
            logger.warning("Correction automatique : emotional_history invalide")
            self.data["emotional_history"] = []
            corrections_appliquees.append("emotional_history")

        # Vérification de profile
        if not isinstance(self.data.get("profile"), dict):
            logger.warning("Correction automatique : profile invalide")
            self.data["profile"] = {"mode": "neutre", "last_updated": None, "user": None}
            corrections_appliquees.append("profile")

        # Vérification de sync_status
        if not isinstance(self.data.get("sync_status"), dict):
            logger.warning("Correction automatique : sync_status invalide")
            self.data["sync_status"] = {"last_icloud_sync": None, "last_local_sync": None, "is_online": True}
            corrections_appliquees.append("sync_status")

        if corrections_appliquees:
            logger.info(f"Corrections appliquées : {', '.join(corrections_appliquees)}")

    def _check_icloud_availability(self) -> bool:
        """Vérifie si iCloud est disponible."""
        if not self.icloud_path:
            return False

        try:
            # Tenter d'accéder au dossier iCloud
            test_file = os.path.join(os.path.dirname(self.icloud_path), ".icloud_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception:
            return False

    def _load_data(self) -> None:
        """Charge les données depuis iCloud ou le stockage local."""
        if not self.icloud_path or not self.local_path:
            logger.error("Chemins de stockage non configurés")
            return

        # Vérifier la disponibilité d'iCloud
        self.data["sync_status"]["is_online"] = self._check_icloud_availability()

        # Essayer d'abord iCloud si disponible
        if self.data["sync_status"]["is_online"]:
            try:
                if os.path.exists(self.icloud_path):
                    with open(self.icloud_path, encoding='utf-8') as f:
                        loaded = json.load(f)
                        if self._validate_data(loaded):
                            self.data = loaded
                            self.data["sync_status"]["last_icloud_sync"] = datetime.now().isoformat()
                            logger.info("Données chargées depuis iCloud")
                            return
            except Exception as e:
                logger.warning(f"Erreur lors du chargement depuis iCloud : {e}")

        # Fallback sur le stockage local
        try:
            if os.path.exists(self.local_path):
                with open(self.local_path, encoding='utf-8') as f:
                    loaded = json.load(f)
                    if self._validate_data(loaded):
                        self.data = loaded
                        self.data["sync_status"]["last_local_sync"] = datetime.now().isoformat()
                        logger.info("Données chargées depuis le stockage local")
                        return
        except Exception as e:
            logger.warning(f"Erreur lors du chargement depuis le stockage local : {e}")

        # Si aucun chargement n'a réussi, initialiser avec les données par défaut
        self._ensure_data_structure()
        self._save_data()

    def _validate_data(self, data: dict[str, Any]) -> bool:
        """Valide la structure des données chargées."""
        required_keys = {"emotional_history", "profile", "sync_status"}
        if not all(key in data for key in required_keys):
            return False

        if not isinstance(data.get("emotional_history"), list):
            return False

        if not isinstance(data.get("profile"), dict):
            return False

        return True

    def _save_data(self) -> None:
        """Sauvegarde les données dans iCloud et/ou le stockage local."""
        if not self.icloud_path or not self.local_path:
            logger.error("Chemins de stockage non configurés")
            return

        # Mettre à jour les timestamps
        self.data["last_sync"] = datetime.now().isoformat()

        # Sauvegarder localement en priorité
        try:
            with open(self.local_path, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2)
            self.data["sync_status"]["last_local_sync"] = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde locale : {e}")

        # Sauvegarder sur iCloud si disponible
        if self._check_icloud_availability():
            try:
                with open(self.icloud_path, 'w', encoding='utf-8') as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)
                self.data["sync_status"]["last_icloud_sync"] = datetime.now().isoformat()
                self.data["sync_status"]["is_online"] = True
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde iCloud : {e}")
                self.data["sync_status"]["is_online"] = False
        else:
            self.data["sync_status"]["is_online"] = False

    def _ensure_emotional_history_is_list(self) -> None:
        """Garantit que emotional_history est une liste."""
        if not isinstance(self.data.get("emotional_history"), list):
            logger.warning("[⚠️] Correction automatique : emotional_history n'était pas une liste.")
            self.data["emotional_history"] = []
            try:
                self._save_data()
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde après correction de emotional_history : {e}")

    def log_emotion(self, emotion: str, intensity: float, context: str | None = None) -> None:
        """
        Enregistre une nouvelle émotion.

        Args:
            emotion: L'émotion à enregistrer
            intensity: L'intensité de l'émotion (entre 0 et 1)
            context: Contexte optionnel de l'émotion

        Raises:
            ValueError: Si l'intensité est invalide
            RuntimeError: Si la sauvegarde échoue
        """
        try:
            if not isinstance(emotion, str):
                raise ValueError("L'émotion doit être une chaîne de caractères")

            intensity = float(intensity)
            if not 0 <= intensity <= 1:
                raise ValueError("L'intensité doit être comprise entre 0 et 1")

            self._ensure_emotional_history_is_list()

            entry = {
                "timestamp": datetime.now().isoformat(),
                "emotion": emotion,
                "intensity": round(intensity, 2),
                "context": str(context) if context else "",
                "mode": self.data["profile"]["mode"],
            }

            if not isinstance(self.data["emotional_history"], list):
                raise RuntimeError("emotional_history n'est pas une liste")

            self.data["emotional_history"].append(entry)
            logger.info(f"Émotion enregistrée: {emotion} (intensité: {intensity})")

            try:
                self._save_data()
            except Exception as e:
                logger.error(f"Échec de la sauvegarde: {e}")
                raise RuntimeError("Impossible de sauvegarder l'émotion") from e

        except ValueError as e:
            logger.error(f"Erreur de validation: {e}")
            raise
        except Exception as e:
            logger.error(f"Erreur inattendue: {e}")
            raise

    def set_emotional_mode(self, mode: Literal["neutre", "innocent", "coquin"], user: str) -> bool:
        """
        Change le mode émotionnel de Jeffrey.

        Args:
            mode: Le nouveau mode ("neutre", "innocent", "coquin")
            user: L'utilisateur qui effectue le changement

        Returns:
            bool: True si le changement a réussi
        """
        if mode not in ["neutre", "innocent", "coquin"]:
            return False

        self.data["profile"].update({"mode": mode, "last_updated": datetime.now().isoformat(), "user": user})
        self._save_data()
        return True

    def get_emotional_mode(self) -> str:
        """Retourne le mode émotionnel actuel."""
        return self.data["profile"]["mode"]

    def get_recent_emotions(self, limit: int = 10) -> list:
        """Retourne les émotions récentes."""
        return self.data["emotional_history"][-limit:]

    def get_emotional_summary(self) -> dict[str, Any]:
        """Retourne un résumé de l'état émotionnel."""
        return {
            "mode": self.data["profile"]["mode"],
            "last_updated": self.data["profile"]["last_updated"],
            "last_sync": self.data["last_sync"],
            "recent_emotions": self.get_recent_emotions(5),
        }
