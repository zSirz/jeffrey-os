#!/usr/bin/env python
"""
Moteur de personnalité dynamique.

Ce module implémente les fonctionnalités essentielles pour moteur de personnalité dynamique.
Il fournit une architecture robuste et évolutive intégrant les composants
nécessaires au fonctionnement optimal du système. L'implémentation suit
les principes de modularité et d'extensibilité pour faciliter l'évolution
future du système.

Le module gère l'initialisation, la configuration, le traitement des données,
la communication inter-composants, et la persistance des états. Il s'intègre
harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
une séparation claire des responsabilités.

L'architecture interne permet une évolution adaptative basée sur les interactions
et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
cohérente et authentique.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime


class AdaptivePersonalityEngine:
    """
    Classe AdaptivePersonalityEngine pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    def __init__(self) -> None:
        self.personality_profile = defaultdict(lambda: {"baseline": 0.5, "variation": [], "last_update": None})

    def register_emotional_event(self, trait: str, impact_score: float):
        """
        Enregistre un événement émotionnel affectant un trait de personnalité.

        Args:
            trait (str): Le trait affecté (ex: 'curiosité', 'confiance').
            impact_score (float): L'effet mesuré de l'événement (de -1.0 à +1.0).
        """
        if trait not in self.personality_profile:
            self.personality_profile[trait] = {
                "baseline": 0.5,
                "variation": [],
                "last_update": None,
            }

        self.personality_profile[trait]["variation"].append(
            {"timestamp": datetime.now().isoformat(), "impact": impact_score}
        )
        self.personality_profile[trait]["last_update"] = datetime.now().isoformat()

    def get_current_profile(self) -> Any:
        """
        Calcule l’état actuel des traits de personnalité.

        Returns:
            dict: Un dictionnaire avec les traits et leur score actuel.
        """
        profile = {}
        for trait, data in self.personality_profile.items():
            base = data["baseline"]
            variations = [v["impact"] for v in data["variation"][-20:]]  # On limite à 20 derniers événements
            variation_score = sum(variations) / len(variations) if variations else 0
            score = max(0.0, min(1.0, base + variation_score))
            profile[trait] = round(score, 3)
        return profile

    def reset_trait(self, trait: str):
        """
        Réinitialise un trait à son état de base.

        Args:
            trait (str): Le trait à réinitialiser.
        """
        if trait in self.personality_profile:
            self.personality_profile[trait]["variation"] = []
            self.personality_profile[trait]["last_update"] = datetime.now().isoformat()

    def export_profile(self):
        """
        Exporte le profil complet avec historique.

        Returns:
            dict: Le profil complet incluant baseline, variations et timestamps.
        """
        return dict(self.personality_profile)
