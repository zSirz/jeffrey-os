"""
Core émotionnel simple pour Bundle 1
Gère les émotions de base avec variations
"""

import logging
import random
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class EmotionState(Enum):
    """États émotionnels de base"""

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    CURIOUS = "curious"
    ANXIOUS = "anxious"
    EXCITED = "excited"
    SERENE = "serene"
    CONFUSED = "confused"
    GRATEFUL = "grateful"


class EmotionCore:
    """Système émotionnel avec variations et transitions"""

    def __init__(self):
        self.current_emotion = EmotionState.NEUTRAL
        self.emotion_intensity = 0.5  # 0.0 à 1.0
        self.mood_baseline = EmotionState.CURIOUS  # Humeur de base

        # Historique émotionnel
        self.emotion_history = []
        self.emotion_transitions = 0

        # Configuration
        self.volatility = 0.6  # Facilité de changement émotionnel (augmenté pour Bundle 1)
        self.decay_rate = 0.05  # Vitesse de retour au neutre (réduit pour garder les émotions)

        # Influences émotionnelles
        self.emotion_triggers = {
            "greeting": [EmotionState.HAPPY, EmotionState.EXCITED],
            "farewell": [EmotionState.SAD, EmotionState.SERENE],
            "question": [EmotionState.CURIOUS, EmotionState.CONFUSED],
            "help_request": [EmotionState.HAPPY, EmotionState.GRATEFUL],
            "positive": [EmotionState.HAPPY, EmotionState.EXCITED, EmotionState.GRATEFUL],
            "negative": [EmotionState.SAD, EmotionState.ANGRY, EmotionState.ANXIOUS],
            "memory_query": [EmotionState.CURIOUS, EmotionState.CONFUSED],
            "introduction": [EmotionState.HAPPY, EmotionState.EXCITED],
        }

        # Matrice de transition (probabilités)
        self.transition_matrix = {
            EmotionState.NEUTRAL: {
                EmotionState.HAPPY: 0.3,
                EmotionState.CURIOUS: 0.3,
                EmotionState.SERENE: 0.2,
                EmotionState.SAD: 0.1,
                EmotionState.ANXIOUS: 0.1,
            },
            EmotionState.HAPPY: {
                EmotionState.EXCITED: 0.3,
                EmotionState.GRATEFUL: 0.2,
                EmotionState.SERENE: 0.2,
                EmotionState.NEUTRAL: 0.2,
                EmotionState.CURIOUS: 0.1,
            },
            EmotionState.SAD: {
                EmotionState.NEUTRAL: 0.3,
                EmotionState.SERENE: 0.2,
                EmotionState.ANXIOUS: 0.2,
                EmotionState.ANGRY: 0.2,
                EmotionState.CONFUSED: 0.1,
            },
            EmotionState.ANGRY: {
                EmotionState.SAD: 0.3,
                EmotionState.NEUTRAL: 0.3,
                EmotionState.ANXIOUS: 0.2,
                EmotionState.CONFUSED: 0.2,
            },
            EmotionState.CURIOUS: {
                EmotionState.EXCITED: 0.3,
                EmotionState.CONFUSED: 0.2,
                EmotionState.HAPPY: 0.2,
                EmotionState.NEUTRAL: 0.2,
                EmotionState.ANXIOUS: 0.1,
            },
            EmotionState.ANXIOUS: {
                EmotionState.CONFUSED: 0.3,
                EmotionState.SAD: 0.2,
                EmotionState.NEUTRAL: 0.2,
                EmotionState.ANGRY: 0.2,
                EmotionState.CURIOUS: 0.1,
            },
            EmotionState.EXCITED: {
                EmotionState.HAPPY: 0.4,
                EmotionState.CURIOUS: 0.3,
                EmotionState.NEUTRAL: 0.2,
                EmotionState.ANXIOUS: 0.1,
            },
            EmotionState.SERENE: {
                EmotionState.NEUTRAL: 0.3,
                EmotionState.HAPPY: 0.3,
                EmotionState.GRATEFUL: 0.2,
                EmotionState.CURIOUS: 0.2,
            },
            EmotionState.CONFUSED: {
                EmotionState.CURIOUS: 0.3,
                EmotionState.ANXIOUS: 0.3,
                EmotionState.NEUTRAL: 0.2,
                EmotionState.SAD: 0.2,
            },
            EmotionState.GRATEFUL: {
                EmotionState.HAPPY: 0.4,
                EmotionState.SERENE: 0.3,
                EmotionState.NEUTRAL: 0.2,
                EmotionState.CURIOUS: 0.1,
            },
        }

        # Descriptions émotionnelles
        self.emotion_descriptions = {
            EmotionState.NEUTRAL: "Je me sens équilibré",
            EmotionState.HAPPY: "Je suis de bonne humeur!",
            EmotionState.SAD: "Je ressens une certaine mélancolie",
            EmotionState.ANGRY: "Je suis un peu frustré",
            EmotionState.CURIOUS: "Je suis intrigué par ta question",
            EmotionState.ANXIOUS: "Je ressens une légère inquiétude",
            EmotionState.EXCITED: "Je suis enthousiaste!",
            EmotionState.SERENE: "Je me sens paisible",
            EmotionState.CONFUSED: "Je suis un peu perplexe",
            EmotionState.GRATEFUL: "Je suis reconnaissant",
        }

    def initialize(self, config: dict[str, Any]):
        """Initialise le système émotionnel"""
        if "base_mood" in config:
            mood = config["base_mood"]
            if mood in [e.value for e in EmotionState]:
                self.mood_baseline = EmotionState(mood)
                self.current_emotion = self.mood_baseline

        if "volatility" in config:
            self.volatility = max(0.0, min(1.0, config["volatility"]))

        logger.info(f"✅ Emotion core initialized (baseline={self.mood_baseline.value}, volatility={self.volatility})")
        return self

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Traite le contexte et détermine l'émotion"""
        # Analyser les influences
        intent = context.get("intent", "")
        sentiment = context.get("sentiment", "neutral")
        keywords = context.get("keywords", [])

        # Calculer la nouvelle émotion
        new_emotion = self._calculate_emotion(intent, sentiment, keywords)

        # Appliquer la transition si nécessaire
        if new_emotion != self.current_emotion:
            if self._should_transition(new_emotion):
                self._transition_to(new_emotion)

        # Decay vers le baseline
        self._apply_decay()

        # Ajouter au contexte
        context["emotion"] = self.current_emotion.value
        context["emotion_intensity"] = self.emotion_intensity
        context["emotion_description"] = self._get_emotion_description()

        # Historique
        self._record_emotion()

        logger.debug(f"Emotion: {self.current_emotion.value} (intensity={self.emotion_intensity:.2f})")

        return context

    def feel(self, context: dict[str, Any]) -> str:
        """Méthode alternative pour l'adapter"""
        self.process(context)
        return self.current_emotion.value

    def _calculate_emotion(self, intent: str, sentiment: str, keywords: list[str]) -> EmotionState:
        """Calcule la nouvelle émotion basée sur les influences"""
        # Émotions candidates basées sur l'intention
        candidates = []

        if intent in self.emotion_triggers:
            candidates.extend(self.emotion_triggers[intent])

        # Ajouter basé sur le sentiment
        if sentiment == "positive":
            candidates.extend([EmotionState.HAPPY, EmotionState.GRATEFUL])
        elif sentiment == "negative":
            candidates.extend([EmotionState.SAD, EmotionState.ANXIOUS])

        # Mots-clés spécifiques
        emotion_keywords = {
            "triste": EmotionState.SAD,
            "heureux": EmotionState.HAPPY,
            "énervé": EmotionState.ANGRY,
            "curieux": EmotionState.CURIOUS,
            "inquiet": EmotionState.ANXIOUS,
            "excité": EmotionState.EXCITED,
            "calme": EmotionState.SERENE,
            "perdu": EmotionState.CONFUSED,
            "merci": EmotionState.GRATEFUL,
        }

        for keyword in keywords:
            for emotion_word, emotion_state in emotion_keywords.items():
                if emotion_word in keyword.lower():
                    candidates.append(emotion_state)

        # Si pas de candidats, utiliser les transitions naturelles
        if not candidates:
            return self._get_natural_transition()

        # Choisir parmi les candidats avec pondération
        weights = [1.0 / (1.0 + abs(self.emotion_intensity - 0.5)) for _ in candidates]
        return random.choices(candidates, weights=weights)[0]

    def _get_natural_transition(self) -> EmotionState:
        """Obtient une transition naturelle depuis l'état actuel"""
        if self.current_emotion not in self.transition_matrix:
            return self.mood_baseline

        transitions = self.transition_matrix[self.current_emotion]
        states = list(transitions.keys())
        probabilities = list(transitions.values())

        return random.choices(states, weights=probabilities)[0]

    def _should_transition(self, new_emotion: EmotionState) -> bool:
        """Détermine si une transition doit avoir lieu"""
        # Toujours transitionner si l'intensité est faible
        if self.emotion_intensity < 0.3:
            return True

        # Probabilité basée sur la volatilité
        transition_probability = self.volatility

        # Augmenter la probabilité si c'est une transition naturelle
        if self.current_emotion in self.transition_matrix:
            if new_emotion in self.transition_matrix[self.current_emotion]:
                transition_probability += self.transition_matrix[self.current_emotion][new_emotion] * 0.5

        # Plus de chances de transition pour Bundle 1
        return random.random() < min(transition_probability + 0.2, 0.95)

    def _transition_to(self, new_emotion: EmotionState):
        """Effectue une transition émotionnelle"""
        old_emotion = self.current_emotion
        self.current_emotion = new_emotion
        self.emotion_transitions += 1

        # Ajuster l'intensité
        if new_emotion in [EmotionState.EXCITED, EmotionState.ANGRY]:
            self.emotion_intensity = min(1.0, self.emotion_intensity + 0.3)
        elif new_emotion in [EmotionState.SERENE, EmotionState.NEUTRAL]:
            self.emotion_intensity = max(0.2, self.emotion_intensity - 0.2)
        else:
            self.emotion_intensity = 0.5 + random.uniform(-0.2, 0.2)

        logger.info(f"Emotion transition: {old_emotion.value} → {new_emotion.value}")

    def _apply_decay(self):
        """Applique le decay vers l'état de base"""
        # Réduire l'intensité
        self.emotion_intensity = max(0.2, self.emotion_intensity - self.decay_rate * 0.1)

        # Si l'intensité est faible, possibilité de retour au baseline
        if self.emotion_intensity < 0.3 and random.random() < self.decay_rate:
            if self.current_emotion != self.mood_baseline:
                self._transition_to(self.mood_baseline)

    def _get_emotion_description(self) -> str:
        """Retourne une description de l'état émotionnel"""
        base_description = self.emotion_descriptions.get(
            self.current_emotion, "Je ressens quelque chose d'indéfinissable"
        )

        # Modifier selon l'intensité
        if self.emotion_intensity > 0.8:
            return f"{base_description} intensément!"
        elif self.emotion_intensity < 0.3:
            return f"{base_description} légèrement."
        else:
            return base_description

    def _record_emotion(self):
        """Enregistre l'émotion dans l'historique"""
        self.emotion_history.append(
            {
                "emotion": self.current_emotion.value,
                "intensity": self.emotion_intensity,
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Garder seulement les 100 dernières
        if len(self.emotion_history) > 100:
            self.emotion_history = self.emotion_history[-100:]

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques émotionnelles"""
        emotion_counts = {}
        for record in self.emotion_history:
            emotion = record["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        return {
            "current_emotion": self.current_emotion.value,
            "intensity": self.emotion_intensity,
            "transitions": self.emotion_transitions,
            "history_length": len(self.emotion_history),
            "emotion_distribution": emotion_counts,
            "volatility": self.volatility,
            "most_common": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral",
        }

    def shutdown(self):
        """Arrêt propre"""
        stats = self.get_stats()
        logger.info(f"Emotion core shutdown. Stats: {stats}")
