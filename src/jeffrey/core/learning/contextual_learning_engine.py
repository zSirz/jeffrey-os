"""
contextual_learning_engine.py

Ce module gère l’apprentissage contextuel de Jeffrey. Il lui permet de modifier ses comportements, réponses et priorités
en fonction du contexte, de l’environnement émotionnel, et de l’expérience passée.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime


class ContextualLearningEngine:
    """
    Classe ContextualLearningEngine pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    def __init__(self) -> None:
        # Structure : { contexte : { info : poids } }
        self.context_knowledge = defaultdict(lambda: defaultdict(float))
        self.context_history = []

    def observe_context(self, context_label: str, data: dict, weight: float = 1.0):
        """
        Enregistre une nouvelle observation contextuelle et renforce les connaissances associées.
        """
        """
        Enregistre une nouvelle observation contextuelle et renforce les connaissances associées.
        """
        timestamp = datetime.now()
        self.context_history.append((timestamp, context_label, data))

        for key, value in data.items():
            key_repr = f"{key}:{value}"
            self.context_knowledge[context_label][key_repr] += weight

    def get_relevant_context_info(self, context_label: str, top_n: int = 5):
        """
        Retourne les éléments les plus significatifs pour un contexte donné.
        """
        if context_label not in self.context_knowledge:
            return []

        sorted_items = sorted(self.context_knowledge[context_label].items(), key=lambda item: item[1], reverse=True)
        return sorted_items[:top_n]

    def adapt_behavior_from_context(self, context_label: str) -> dict:
        """
        Génère des recommandations comportementales à partir du contexte observé.
        """
        if context_label not in self.context_knowledge:
            return {}

        knowledge = self.context_knowledge[context_label]
        recommendations = {}

        for key_repr, weight in knowledge.items():
            key, value = key_repr.split(":", 1)
            if key not in recommendations:
                recommendations[key] = []
            recommendations[key].append((value, weight))

        # Agrégation simple : valeur avec poids le plus élevé
        final_recommendations = {}
        for key, values in recommendations.items():
            values.sort(key=lambda x: x[1], reverse=True)
            final_recommendations[key] = values[0][0]

        return final_recommendations

    def reset_context(self, context_label: str):
        """
        Réinitialise les données apprises pour un contexte spécifique.
        """
        if context_label in self.context_knowledge:
            del self.context_knowledge[context_label]
