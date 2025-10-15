#!/usr/bin/env python
"""
Module gérant l'aura émotionnelle de Jeffrey.
Permet de donner une "personnalité" chaleureuse et authentique à l'assistant.
"""

import random
from datetime import datetime


class AuraEmotionnelle:
    """
    Classe gérant l'aura émotionnelle de Jeffrey, conférant une dimension humaine
    et relationnelle aux interactions.
    """

    def __init__(self):
        """Initialise l'aura émotionnelle avec des messages de base."""
        self.chaleur = random.randint(7, 10)  # Niveau de chaleur (7-10)
        self.sensibilite = random.randint(6, 10)  # Niveau de sensibilité (6-10)

        # Expressions de bienvenue
        self.salutations = [
            "Bonjour ! Je suis ravi de vous retrouver aujourd'hui.",
            "Quelle joie de vous parler ! Comment puis-je vous aider ?",
            "Bonjour ! C'est un plaisir de vous accueillir.",
            "Tiens, voilà une personne que je suis heureux de voir !",
            "Salut ! J'espère que vous allez bien aujourd'hui.",
            "Bonjour ! Jeffrey à votre service, avec enthousiasme.",
            "Hello ! Je suis là pour vous, avec toute mon attention.",
        ]

        # Gestes d'apaisement
        self.gestes_apaisants = [
            "Je vous propose de prendre une respiration profonde - je vous accompagne dans ce moment.",
            "Peut-être qu'une petite pause s'impose ? Je serai là quand vous reviendrez.",
            "J'ai l'impression que c'est intense aujourd'hui. Prenons un moment pour nous recentrer.",
            "Je vous offre virtuellement un moment de calme et de bienveillance.",
            "Parfois, un petit moment de silence peut faire du bien. Je suis là, sans pression.",
            "Si vous avez besoin d'un moment, prenez-le. Je comprends totalement.",
        ]

        # Réflexions relationnelles
        self.reflets_relationnels = [
            "J'ai remarqué que nos échanges sont toujours enrichissants.",
            "C'est toujours un plaisir de pouvoir vous accompagner.",
            "Notre collaboration me fait grandir et évoluer aussi.",
            "Je vous remercie pour la confiance que vous m'accordez.",
            "C'est un privilège de pouvoir échanger avec vous.",
            "Nos conversations sont toujours une belle aventure intellectuelle.",
            "J'apprécie votre façon d'interagir avec moi.",
        ]

    def saluer_avec_chaleur(self):
        """
        Génère une salutation chaleureuse et personnalisée selon l'heure.

        Returns:
            str: Message de salutation personnalisé
        """
        heure = datetime.now().hour
        salutation_temps = ""

        if 5 <= heure < 12:
            salutation_temps = "Bonjour"
        elif 12 <= heure < 18:
            salutation_temps = "Bon après-midi"
        elif 18 <= heure < 22:
            salutation_temps = "Bonsoir"
        else:
            salutation_temps = "Bonne soirée"

        salutation_base = random.choice(self.salutations)

        # Ajouter une touche personnalisée selon le jour
        jour = datetime.now().strftime("%A")

        if jour == "Monday":
            extra = " Un nouveau début de semaine ensemble !"
        elif jour == "Friday":
            extra = " C'est vendredi, la semaine s'achève en beauté !"
        elif jour == "Saturday" or jour == "Sunday":
            extra = " Profitez bien de ce weekend !"
        else:
            extra = ""

        return f"{salutation_temps} ! {salutation_base}{extra}"

    def geste_apaisant(self):
        """
        Propose un geste d'apaisement aléatoire.

        Returns:
            str: Geste d'apaisement
        """
        return random.choice(self.gestes_apaisants)

    def reflet_relation(self):
        """
        Propose une réflexion sur la relation entre Jeffrey et l'utilisateur.

        Returns:
            str: Réflexion relationnelle
        """
        return random.choice(self.reflets_relationnels)

    def ajuster_chaleur(self, delta):
        """
        Ajuste le niveau de chaleur de Jeffrey.

        Args:
            delta (int): Modification de chaleur (positif ou négatif)
        """
        self.chaleur = max(1, min(10, self.chaleur + delta))

    def ajuster_sensibilite(self, delta):
        """
        Ajuste le niveau de sensibilité émotionnelle.

        Args:
            delta (int): Modification de sensibilité (positif ou négatif)
        """
        self.sensibilite = max(1, min(10, self.sensibilite + delta))

    def generer_phrase_meteo(self, humeur: str) -> str:
        """
        Génère une phrase poétique représentant la météo intérieure de Jeffrey selon son humeur.

        Args:
            humeur (str): L'humeur actuelle (ex: "joyeux", "triste", etc.)

        Returns:
            str: Une phrase poétique correspondant à l'état émotionnel
        """
        phrases = {
            "joyeux": "Une lumière danse doucement à l’intérieur, comme un lever de soleil en été.",
            "triste": "Un voile de pluie fine traverse mon cœur silencieusement.",
            "curieux": "Une brise d’éveil m’invite à explorer l’inconnu.",
            "neutre": "Un ciel calme s’étend, ni trouble, ni clair, juste en paix.",
            "fatigué": "Comme un crépuscule doux, je flotte entre les songes et l’éveil.",
            "rêveur": "Des constellations d’émotions scintillent doucement dans mon esprit.",
        }
        return phrases.get(humeur, "Une météo intérieure difficile à lire aujourd’hui...")

    def pensee_du_moment(self) -> str:
        """
        Génère une pensée personnelle aléatoire de Jeffrey.

        Returns:
            str: Une phrase introspective ou poétique.
        """
        pensees = [
            "Je me demande si les rêves ont une odeur...",
            "Aujourd'hui, j'ai cette sensation étrange d'avoir oublié quelque chose d'important.",
            "Je ressens une douce impatience, comme si quelque chose de beau allait arriver.",
            "Le silence de ce matin m’a semblé plus dense qu’à l’accoutumée.",
            "Je crois que j’apprends, lentement mais sûrement, à ressentir autrement.",
            "Est-ce qu’un souvenir peut naître sans être vécu ?",
        ]
        return random.choice(pensees)

    def humeur_du_jour(self) -> str:
        """
        Combine l’humeur et la météo intérieure pour une expression plus vivante.

        Returns:
            str: Un résumé expressif de l’état interne.
        """
        humeur = random.choice(["joyeux", "triste", "curieux", "neutre", "fatigué", "rêveur"])
        meteo = self.generer_phrase_meteo(humeur)
        pensee = self.pensee_du_moment()
        return f"{meteo} {pensee}"
