import json
import os
from collections import Counter
from datetime import datetime, timedelta


class DetectionCyclesHumeur:
    """
    Détecte des tendances émotionnelles sur plusieurs jours pour anticiper les besoins émotionnels de l'utilisateur.
    """

    def __init__(self, filepath='data/humeur_history.json'):
        self.filepath = filepath
        self.histoire = []
        self.load_history()

    def load_history(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, encoding='utf-8') as f:
                    self.histoire = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement de l'historique d'humeur : {e}")
                self.histoire = []
        else:
            self.histoire = []

    def save_history(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.histoire, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde de l'historique d'humeur : {e}")

    def enregistrer_humeur(self, humeur):
        """
        Enregistre l'humeur du jour dans l'historique.
        """
        entree = {"date": datetime.utcnow().isoformat(), "humeur": humeur}
        self.histoire.append(entree)
        self.save_history()

    def humeur_predominante_sur_7_jours(self):
        """
        Analyse les 7 derniers jours pour déterminer l'humeur prédominante.
        """
        derniere_semaine = datetime.utcnow() - timedelta(days=7)
        recent_entries = [entry for entry in self.histoire if datetime.fromisoformat(entry["date"]) >= derniere_semaine]

        if not recent_entries:
            return "inconnu"

        humeur_list = [entry["humeur"] for entry in recent_entries]
        compteur = Counter(humeur_list)
        humeur_majoritaire, _ = compteur.most_common(1)[0]

        return humeur_majoritaire

    def humeur_du_jour_precedent(self):
        """
        Retourne l'humeur enregistrée la veille, si disponible.
        """
        if not self.histoire:
            return "inconnu"

        hier = datetime.utcnow() - timedelta(days=1)
        for entry in reversed(self.histoire):
            if datetime.fromisoformat(entry["date"]).date() == hier.date():
                return entry["humeur"]

        return "inconnu"
