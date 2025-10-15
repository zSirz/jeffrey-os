import json
import os
from datetime import datetime


class SouvenirsCelebrations:
    """
    Gère la célébration automatique des souvenirs marquants de l'utilisateur avec Jeffrey.
    """

    def __init__(self, filepath='data/souvenirs_data.json'):
        self.filepath = filepath
        self.souvenirs = []
        self.load_souvenirs()

    def load_souvenirs(self):
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, encoding='utf-8') as f:
                    self.souvenirs = json.load(f)
            except Exception as e:
                print(f"Erreur lors du chargement des souvenirs : {e}")
                self.souvenirs = []
        else:
            self.souvenirs = []

    def save_souvenirs(self):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self.souvenirs, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des souvenirs : {e}")

    def ajouter_souvenir(self, description):
        """
        Ajoute un nouveau souvenir marquant à célébrer dans le futur.
        """
        souvenir = {"description": description, "date_enregistrement": datetime.utcnow().isoformat()}
        self.souvenirs.append(souvenir)
        self.save_souvenirs()

    def souvenirs_a_celebrer(self):
        """
        Retourne une liste de souvenirs enregistrés il y a exactement 1, 3, 6 ou 12 mois.
        """
        aujourd_hui = datetime.utcnow().date()
        souvenirs_speciaux = []

        for souvenir in self.souvenirs:
            try:
                date_enregistrement = datetime.fromisoformat(souvenir["date_enregistrement"]).date()
                delta = (aujourd_hui - date_enregistrement).days

                if delta in [30, 90, 180, 365]:  # 1 mois, 3 mois, 6 mois, 1 an
                    souvenirs_speciaux.append(souvenir)
            except Exception:
                continue

        return souvenirs_speciaux
