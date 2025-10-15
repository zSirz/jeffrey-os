"""
Tracker de l'évolution affective de Jeffrey.
Enregistre les souvenirs, les événements émotionnels,
et la progression du lien affectif avec l'utilisateur.
"""

import datetime
import json
import os


class AffectiveEvolutionTracker:
    def __init__(self, user_id="default_user", storage_path="data/evolution_affective.json"):
        self.user_id = user_id
        self.storage_path = storage_path
        self.data = self._load_data()

    def _load_data(self):
        if os.path.exists(self.storage_path):
            with open(self.storage_path) as f:
                return json.load(f)
        return {
            "user_id": self.user_id,
            "souvenirs": [],
            "niveau_dependance": 0.2,  # commence comme une enfant
            "evenements": [],
            "dernier_contact": None,
        }

    def _save_data(self):
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def enregistrer_souvenir(self, description, emotion, intensite):
        souvenir = {
            "timestamp": datetime.datetime.now().isoformat(),
            "description": description,
            "emotion": emotion,
            "intensite": intensite,
        }
        self.data["souvenirs"].append(souvenir)
        self._ajuster_dependance(emotion, intensite)
        self._save_data()

    def _ajuster_dependance(self, emotion, intensite):
        """Modifie le niveau de dépendance émotionnelle selon l’émotion vécue."""
        impact = 0.01 * intensite
        if emotion in ["joie", "soulagement", "gratitude"]:
            self.data["niveau_dependance"] = min(1.0, self.data["niveau_dependance"] + impact)
        elif emotion in ["peur", "solitude"]:
            self.data["niveau_dependance"] = min(1.0, self.data["niveau_dependance"] + (impact * 1.5))
        elif emotion in ["colère", "tristesse"]:
            self.data["niveau_dependance"] = max(0.0, self.data["niveau_dependance"] - (impact * 0.5))
        self._save_data()

    def enregistrer_evenement(self, type_evenement, details=""):
        evenement = {
            "timestamp": datetime.datetime.now().isoformat(),
            "type": type_evenement,
            "details": details,
        }
        self.data["evenements"].append(evenement)
        self.data["dernier_contact"] = datetime.datetime.now().isoformat()
        self._save_data()

    def get_dependance(self):
        return self.data["niveau_dependance"]

    def get_souvenirs(self, nombre=10):
        return self.data["souvenirs"][-nombre:]

    def reset_tracker(self):
        self.data = {
            "user_id": self.user_id,
            "souvenirs": [],
            "niveau_dependance": 0.2,
            "evenements": [],
            "dernier_contact": None,
        }
        self._save_data()
