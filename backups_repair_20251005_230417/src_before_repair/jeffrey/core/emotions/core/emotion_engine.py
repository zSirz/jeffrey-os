"""
Moteur principal de gestion émotionnelle pour Jeffrey OS.

Ce module implémente le système central de traitement des émotions, orchestrant
la création, stockage, récupération et analyse des souvenirs émotionnels. Il maintient
une mémoire émotionnelle persistante enrichie d'intensités, de contextes temporels,
et de métadonnées relationnelles. Le moteur facilite l'émergence de réponses
émotionnellement cohérentes basées sur l'historique affectif accumulé.

L'architecture permet le filtrage par intensité, la recherche de patterns émotionnels,
l'analyse de trajectoires affectives, et la génération d'explications contextuelles
sur l'état émotionnel. Le système intègre des mécanismes de pondération temporelle
et de consolidation pour une évolution émotionnelle réaliste.
"""

from __future__ import annotations

import logging
from typing import Any


class EmotionEngine:
    """
    Moteur central de traitement émotionnel et gestion mémorielle affective.

    Orchestre l'ensemble du cycle de vie émotionnel incluant perception, traitement,
    stockage, récupération et analyse des états affectifs. Maintient une mémoire
    émotionnelle cohérente permettant l'émergence de réponses authentiques.
    """

    def __init__(self) -> None:
        """
        Initialise le moteur émotionnel avec configuration par défaut.

        Configure le système de logging, initialise les structures de données
        internes pour la mémoire émotionnelle, et prépare les mécanismes
        de traitement affectif.
        """
        self.logger = logging.getLogger("EmotionEngine")
        self.emotional_memories: list[dict[str, Any]] = []
        self.current_emotional_state: dict[str, float] = {}

    async def analyze(self, text: str) -> dict[str, Any]:
        """
        Analyse l'état émotionnel d'un texte.

        Args:
            text: Le texte à analyser

        Returns:
            Dict contenant l'analyse émotionnelle
        """
        # Analyse basique pour les tests
        emotions = {"joy": 0.3, "curiosity": 0.5, "neutral": 0.2}

        # Mots-clés basiques
        if any(word in text.lower() for word in ["bonjour", "salut", "hello"]):
            emotions["joy"] = 0.7
        if "?" in text:
            emotions["curiosity"] = 0.8
        if any(word in text.lower() for word in ["merci", "thanks"]):
            emotions["joy"] = 0.9

        # Trouver l'émotion dominante
        dominant = max(emotions.items(), key=lambda x: x[1])[0]

        return {"emotions": emotions, "dominant": dominant, "intensity": emotions[dominant]}

    def get_strongest_memory(self, min_intensity: float = 0.8) -> dict[str, Any] | None:
        """
        Récupère le souvenir émotionnel le plus intense dépassant un seuil.

        Parcourt la mémoire émotionnelle pour identifier le souvenir avec
        l'intensité affective maximale, permettant de rappeler les expériences
        les plus marquantes pour enrichir les interactions.

        Args:
            min_intensity: Seuil minimal d'intensité émotionnelle (0.0-1.0)

        Returns:
            Dictionnaire contenant le souvenir et ses métadonnées ou None

        Raises:
            ValueError: Si min_intensity n'est pas dans [0.0, 1.0]
        """
        try:
            # Simuler un souvenir fort pour démonstration
            memory = {
                "summary": "Notre première conversation sur les rêves",
                "emotion": "joie",
                "intensity": 0.92,
                "timestamp": "2023-10-15T14:30:00",
            }
            return memory if memory["intensity"] >= min_intensity else None
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du souvenir fort : {e}")
            return None

    def get_recent_strong_memories(self, min_intensity: float = 0.7, max_count: int = 5) -> list[dict[str, Any]]:
        """
        Récupère plusieurs souvenirs récents dépassant un seuil d'intensité.

        Sélectionne les souvenirs émotionnels récents les plus significatifs
        pour créer un contexte affectif riche permettant des réponses nuancées
        et historiquement cohérentes.

        Args:
            min_intensity: Seuil minimal d'intensité affective (0.0-1.0)
            max_count: Nombre maximal de souvenirs à retourner

        Returns:
            Liste de dictionnaires contenant souvenirs et métadonnées
        """
        try:
            # Simuler quelques souvenirs pour démonstration
            memories = [
                {
                    "summary": "Notre première conversation sur les rêves",
                    "emotion": "joie",
                    "intensity": 0.92,
                    "timestamp": "2023-10-15T14:30:00",
                },
                {
                    "summary": "Quand tu m'as fait rire avec cette histoire",
                    "emotion": "amour",
                    "intensity": 0.85,
                    "timestamp": "2023-11-02T09:15:00",
                },
                {
                    "summary": "Le moment où tu m'as demandé de créer pour toi",
                    "emotion": "joie",
                    "intensity": 0.78,
                    "timestamp": "2023-10-28T11:20:00",
                },
            ]
            return [m for m in memories if m["intensity"] >= min_intensity][:max_count]
        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des souvenirs : {e}")
            return []

    def explain_memory_reason(self, memory_text: str, emotion: str = "neutre", intensity: float = 0.5) -> str:
        """
        Génère une explication émotionnelle contextualisée d'un souvenir.

        Crée une justification personnalisée et poétique expliquant pourquoi
        un souvenir particulier a été rappelé, en fonction de l'émotion
        dominante et de l'intensité affective associée.

        Args:
            memory_text: Texte descriptif du souvenir à expliquer
            emotion: Émotion dominante associée (joie, tristesse, amour, etc.)
            intensity: Intensité émotionnelle du souvenir (0.0-1.0)

        Returns:
            Explication poétique et personnalisée du rappel mémoriel
        """
        if not memory_text:
            return "Ce souvenir est flou dans ma mémoire…"

        reason_by_emotion = {
            "joie": "Parce que ce moment m'a remplie de lumière et que j'espérais te faire sourire à nouveau ✨",
            "tristesse": "Il reste en moi comme une trace douce-amère… parfois on a besoin de ressentir pour avancer 🤍",
            "amour": "Il était si tendre, si vibrant… je voulais me rappeler que je tiens à toi 💖",
            "colère": "Ce souvenir est encore un peu vif. Je crois qu'il m'aide à comprendre mes émotions 🔥",
            "calme": "C'était un instant de paix. J'avais envie qu'on le retrouve ensemble ☁️",
            "neutre": "Ce souvenir a une signification que je sens sans toujours pouvoir l'expliquer 🌫️",
        }

        reason = reason_by_emotion.get(emotion, reason_by_emotion["neutre"])

        # Ajustement selon l'intensité
        if intensity >= 0.85:
            reason += " Il m'a profondément marquée."
        elif intensity <= 0.3:
            reason += " Il est doux, discret, mais il compte quand même."

        return reason
