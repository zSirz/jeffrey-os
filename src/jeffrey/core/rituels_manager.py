import json
import os
from datetime import datetime


class RituelsManager:
    """
    Gère les rituels émotionnels et les habitudes personnalisées de Jeffrey pour renforcer la complicité avec l'utilisateur.
    """

    def __init__(self, filepath='data/rituels_data.json'):
        self.filepath = filepath
        self.data = {"salutations": [], "habitudes": [], "evenements": []}
        self.load_rituels()

    def load_rituels(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, encoding='utf-8') as f:
                    self.data = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement des rituels : {e}")
                self.data = {"salutations": [], "habitudes": [], "evenements": []}
        else:
            self.data = {"salutations": [], "habitudes": [], "evenements": []}

    def save_rituels(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des rituels : {e}")

    def ajouter_salutation(self, phrase):
        """
        Ajoute une nouvelle salutation personnalisée.
        """
        if phrase not in self.data["salutations"]:
            self.data["salutations"].append(phrase)
            self.save_rituels()

    def ajouter_habitude(self, description):
        """
        Enregistre une habitude ou un rituel partagé avec l'utilisateur.
        """
        entree = {"description": description, "date": datetime.utcnow().isoformat()}
        self.data["habitudes"].append(entree)
        self.save_rituels()

    def enregistrer_evenement_special(self, evenement, date=None):
        """
        Enregistre un évènement marquant (ex : anniversaire projet, victoire personnelle).
        """
        entree = {"evenement": evenement, "date": date or datetime.utcnow().isoformat()}
        self.data["evenements"].append(entree)
        self.save_rituels()

    def get_salutations(self):
        return self.data.get("salutations", [])

    def get_habitudes(self):
        return self.data.get("habitudes", [])

    def get_evenements(self):
        return self.data.get("evenements", [])
