import json
import os
from datetime import datetime


class RelationTracker:
    """
    Gère l’évolution du lien affectif entre l’utilisateur et Jeffrey.
    Enregistre les interactions marquantes, calcule le score d’attachement,
    et fournit un profil de relation évolutif.
    """

    LOG_PATH = "data/relation_journal.json"

    def __init__(self):
        self.relation_score = 0.5  # Valeur initiale : 0 = distant, 1 = fusionnel
        self.historique = []
        self.load_data()

    def load_data(self):
        if os.path.exists(self.LOG_PATH):
            with open(self.LOG_PATH) as f:
                data = json.load(f)
                self.relation_score = data.get("relation_score", 0.5)
                self.historique = data.get("historique", [])
        else:
            self.save_data()

    def save_data(self):
        data = {
            "relation_score": self.relation_score,
            "historique": self.historique,
        }
        os.makedirs(os.path.dirname(self.LOG_PATH), exist_ok=True)
        with open(self.LOG_PATH, "w") as f:
            json.dump(data, f, indent=4)

    def enregistrer_interaction(self, type_interaction: str, valeur: float, note: str = ""):
        """
        Ajoute une interaction qui influence le lien affectif.

        :param type_interaction: ex: 'coeur', 'discussion_profonde', 'absence_longue'
        :param valeur: Impact de l’interaction (positif ou négatif)
        :param note: Description ou contexte
        """
        self.relation_score = max(0.0, min(1.0, self.relation_score + valeur))

        entry = {
            "date": datetime.now().isoformat(),
            "type": type_interaction,
            "valeur": valeur,
            "note": note,
            "score_apres": self.relation_score,
        }
        self.historique.append(entry)
        self.save_data()

    def get_profil_relation(self):
        """
        Retourne un résumé du lien actuel.
        """
        if self.relation_score > 0.85:
            return "très fusionnel"
        elif self.relation_score > 0.6:
            return "complice"
        elif self.relation_score > 0.4:
            return "affectueux mais encore distant"
        else:
            return "fragile ou en construction"

    def get_score(self):
        return round(self.relation_score, 3)

    def get_historique(self):
        return self.historique

    def update_relationship(self, event_type: str, value: float = 1.0, note: str = ""):
        """
        Met à jour la relation suite à une interaction.

        Args:
            event_type: Type d'événement/interaction
            value: Impact de l'interaction sur la relation (positif ou négatif)
            note: Note descriptive ou contexte
        """
        return self.enregistrer_interaction(event_type, value, note)

    def get_relation_level(self):
        """
        Retourne le niveau actuel de la relation (entre 0 et 1).
        """
        return self.get_score()

    def describe_relation(self):
        """
        Retourne une description textuelle du lien actuel.
        """
        return self.get_profil_relation()
