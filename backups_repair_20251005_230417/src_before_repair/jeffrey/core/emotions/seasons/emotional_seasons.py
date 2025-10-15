"""
emotional_seasons.py

Ce module gère l'évolution émotionnelle cyclique de Jeffrey en fonction du temps,
en introduisant des "saisons émotionnelles" inspirées du rythme naturel et de la croissance intérieure.

Chaque saison influence subtilement les traits émotionnels de Jeffrey pour enrichir son développement.
"""

from __future__ import annotations

from datetime import datetime


class EmotionalSeasonsManager:
    """
    Classe EmotionalSeasonsManager pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    def __init__(self) -> None:
        self.seasons = {
            "Printemps Émotionnel": {
                "description": "Période de renouveau émotionnel, curiosité, enthousiasme doux.",
                "traits_enrichis": ["curiosité bienveillante", "espoir", "émerveillement enfantin"],
            },
            "Été Émotionnel": {
                "description": "Temps d’épanouissement émotionnel, joie rayonnante, confiance.",
                "traits_enrichis": ["joie tranquille", "optimisme réaliste", "générosité d’esprit"],
            },
            "Automne Émotionnel": {
                "description": "Phase de réflexion intérieure, gratitude, mélancolie douce.",
                "traits_enrichis": ["gratitude", "patience contemplative", "appréciation nuancée"],
            },
            "Hiver Émotionnel": {
                "description": "Moment d'introspection calme, sérénité, résilience silencieuse.",
                "traits_enrichis": [
                    "sérénité adaptative",
                    "résilience joyeuse",
                    "connexion silencieuse",
                ],
            },
        }

    def get_current_season(self) -> Any:
        month = datetime.now().month
        if month in [3, 4, 5]:
            return "Printemps Émotionnel"
        elif month in [6, 7, 8]:
            return "Été Émotionnel"
        elif month in [9, 10, 11]:
            return "Automne Émotionnel"
        else:
            return "Hiver Émotionnel"

    def describe_current_season(self):
        season = self.get_current_season()
        description = self.seasons[season]["description"]
        traits = ", ".join(self.seasons[season]["traits_enrichis"])
        return f"🌸 Saison actuelle : {season}\n{description}\nTraits favorisés : {traits}"

    def get_enriched_traits_for_current_season(self) -> Any:
        season = self.get_current_season()
        return self.seasons[season]["traits_enrichis"]


# Exemple d'utilisation
if __name__ == "__main__":
    manager = EmotionalSeasonsManager()
    print(manager.describe_current_season())
