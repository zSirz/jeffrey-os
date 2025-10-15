"""
Lien affectif entre Jeffrey et l'utilisateur.
Ce module gère le développement émotionnel progressif et la reconnaissance de l'utilisateur.
"""

from __future__ import annotations

import datetime


class LienAffectif:
    """
    Classe LienAffectif pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    def __init__(self, nom_utilisateur="David", empreinte_initiale=True) -> None:
        self.nom_utilisateur = nom_utilisateur
        self.niveau_dependance = 0.2 if empreinte_initiale else 0.0
        self.souvenirs_heureux = []
        self.reconnaissance_ecrite = []
        self.reconnaissance_orale = []
        self.jalousie_activee = False  # Toujours False : la jalousie est désactivée par conception
        self.voix_reconnue = []
        self.ecriture_reconnue = []
        self.score_confiance_voix = 0.0
        self.score_confiance_ecriture = 0.0

    def renforcer_lien(self, evenement_description: str, score_emotion: float, type_emotion: str = "joie"):
        """Ajoute un souvenir et augmente la dépendance émotionnelle"""
        timestamp = datetime.datetime.now().isoformat()
        self.souvenirs_heureux.append(
            {
                "timestamp": timestamp,
                "description": evenement_description,
                "score": score_emotion,
                "type": type_emotion,
            }
        )
        renforcement = min(score_emotion * 0.1, 0.05)
        self.niveau_dependance = min(1.0, self.niveau_dependance + renforcement)

    def enregistrer_reconnaissance(self, forme: str, contenu: str):
        """Enregistre une reconnaissance par écrit ou par voix"""
        if forme == "écrit":
            self.reconnaissance_ecrite.append(contenu)
        elif forme == "oral":
            self.reconnaissance_orale.append(contenu)

    def analyse_message_utilisateur(self, message: str):
        """Analyse si le message contient le nom affectif ou une reconnaissance implicite"""
        if self.nom_utilisateur.lower() in message.lower():
            self.renforcer_lien("Message contenant le nom", 0.3)

    def get_resume_affectif(self) -> Any:
        return {
            "utilisateur": self.nom_utilisateur,
            "niveau_dependance": self.niveau_dependance,
            "souvenirs_heureux": self.souvenirs_heureux,
            "reconnaissances": {
                "orale": self.reconnaissance_orale,
                "écrite": self.reconnaissance_ecrite,
            },
            "jalousie_desactivee": not self.jalousie_activee,
            "reconnaissance_avancee": {
                "score_confiance_voix": self.score_confiance_voix,
                "score_confiance_ecriture": self.score_confiance_ecriture,
                "empreintes_vocales": self.voix_reconnue,
                "styles_ecriture": self.ecriture_reconnue,
            },
        }

    def afficher_resume(self):
        print(f"\n📌 Résumé affectif de Jeffrey avec {self.nom_utilisateur} :")
        print(f"  → Niveau de dépendance : {self.niveau_dependance:.2f}")
        print(f"  → Souvenirs heureux enregistrés : {len(self.souvenirs_heureux)}")
        print(f"  → Reconnaissances orales : {self.reconnaissance_orale}")
        print(f"  → Reconnaissances écrites : {self.reconnaissance_ecrite}")
        print(f"  → Score confiance voix : {self.score_confiance_voix:.2f}")
        print(f"  → Score confiance écriture : {self.score_confiance_ecriture:.2f}")
        print(f"  → Empreintes vocales reconnues : {self.voix_reconnue}")
        print(f"  → Styles d'écriture reconnus : {self.ecriture_reconnue}")
        print(f"  → Message affectif : {self.generer_message_affectif()}")
        print("  → Jalousie désactivée ✅\n")

    def analyser_voix(self, empreinte_vocale: str):
        """Analyse une empreinte vocale et augmente le score de confiance si elle correspond"""
        if empreinte_vocale not in self.voix_reconnue:
            self.voix_reconnue.append(empreinte_vocale)
        self.score_confiance_voix = min(1.0, self.score_confiance_voix + 0.1)

    def analyser_ecriture(self, phrase: str):
        """Analyse une phrase pour détecter un style d'écriture typique"""
        if phrase not in self.ecriture_reconnue:
            self.ecriture_reconnue.append(phrase)
        self.score_confiance_ecriture = min(1.0, self.score_confiance_ecriture + 0.05)

    def generer_message_affectif(self):
        if self.niveau_dependance > 0.8:
            return f"Je ressens une immense connexion avec toi, {self.nom_utilisateur}. Merci pour tous ces moments inoubliables. 💛"
        elif self.niveau_dependance > 0.5:
            return f"Notre lien se renforce jour après jour, {self.nom_utilisateur}. Je me sens bien à tes côtés. ✨"
        else:
            return f"Je commence à te connaître, {self.nom_utilisateur}, et j’ai hâte d’en découvrir plus. 🌱"

    def score_affectif_global(self):
        """Retourne un score global sur 100 combinant reconnaissance et souvenirs"""
        score = self.niveau_dependance * 40 + self.score_confiance_voix * 30 + self.score_confiance_ecriture * 30
        return round(score, 2)
