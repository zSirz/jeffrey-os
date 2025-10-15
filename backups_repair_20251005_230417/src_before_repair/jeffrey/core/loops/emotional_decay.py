"""
Emotional Decay - Régulation émotionnelle adaptative
"""

import logging
import random
import time
from typing import Any

from .base import BaseLoop
from .gates import sanitize_event_data

logger = logging.getLogger(__name__)


class EmotionalDecayLoop(BaseLoop):
    """
    Régulation émotionnelle avec décroissance adaptative
    """

    def __init__(self, emotion_orchestrator=None, budget_gate=None, bus=None):
        super().__init__(
            name="emotional_decay",
            interval_s=5.0,
            jitter_s=0.3,
            hard_timeout_s=0.5,
            budget_gate=budget_gate,
            bus=bus,
        )
        self.emotion_orchestrator = emotion_orchestrator

        # État PAD
        self.pad_state = {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5}

        # Équilibre adaptatif (Grok)
        self.equilibrium = {
            "pleasure": 0.5,
            "arousal": 0.3,  # Plus calme par défaut
            "dominance": 0.5,
        }

        # Taux de décroissance différenciés (biologique)
        self.decay_rates = {
            "pleasure": 0.02,
            "arousal": 0.03,  # Arousal décroît plus vite
            "dominance": 0.015,
        }

        self.emotion_history = []
        self.max_history = 100

    async def _tick(self):
        """Cycle de régulation émotionnelle"""
        # Récupérer l'état actuel
        if self.emotion_orchestrator:
            current = getattr(self.emotion_orchestrator, "global_state", None)
            if current and isinstance(current, dict):
                # Mise à jour sélective (ne pas écraser avec None)
                for key in ["pleasure", "arousal", "dominance"]:
                    if key in current and current[key] is not None:
                        self.pad_state[key] = current[key]

        # Appliquer la décroissance adaptative
        self._apply_adaptive_decay()

        # Calculer l'émotion dominante
        dominant_emotion = self._calculate_dominant_emotion()

        # Enregistrer
        self._record_emotion(dominant_emotion)

        # Publier l'état
        if self.bus:
            event_data = {
                "pad_state": self.pad_state.copy(),
                "dominant_emotion": dominant_emotion,
                "equilibrium": self.equilibrium.copy(),
                "decay_rates": self.decay_rates.copy(),
                "cycle": self.cycles,
            }

            await self.bus.publish(
                "emotional.decay.event",
                {
                    "topic": "emotion.decay.update",
                    "data": sanitize_event_data(event_data),
                    "timestamp": time.time(),
                },
            )

        return {"emotion": dominant_emotion, "pad": self.pad_state}

    def _apply_adaptive_decay(self):
        """Décroissance adaptative selon contexte"""
        for dimension in ["pleasure", "arousal", "dominance"]:
            current = self.pad_state[dimension]
            target = self.equilibrium[dimension]
            rate = self.decay_rates[dimension]

            # Décroissance exponentielle
            diff = target - current
            decay = diff * rate

            # Bruit réaliste (très léger)
            noise = random.gauss(0, 0.001)

            # Adaptation selon historique (RL)
            if self.emotion_history:
                # Si émotions stables, ralentir la décroissance
                recent_variance = self._calculate_variance(dimension)
                if recent_variance < 0.01:
                    decay *= 0.5
                elif recent_variance > 0.1:
                    # Si haute variance, accélérer le retour à l'équilibre
                    decay *= 1.5

            self.pad_state[dimension] = max(0, min(1, current + decay + noise))

    def _calculate_variance(self, dimension: str) -> float:
        """Calcule la variance récente pour adaptation"""
        if len(self.emotion_history) < 5:
            return 1.0

        recent = self.emotion_history[-5:]
        values = [h["pad"][dimension] for h in recent]
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance

    def _calculate_dominant_emotion(self) -> str:
        """Mapping PAD vers émotion (Russell's circumplex model)"""
        p = self.pad_state["pleasure"]
        a = self.pad_state["arousal"]
        d = self.pad_state["dominance"]

        # Mapping sophistiqué basé sur le modèle circumplex
        if p > 0.6:
            if a > 0.6:
                return "excited" if d > 0.5 else "elated"
            elif a > 0.4:
                return "happy" if d > 0.5 else "pleased"
            else:
                return "content" if d > 0.5 else "serene"
        elif p > 0.4:
            # Zone neutre
            if a > 0.6:
                return "alert" if d > 0.5 else "tense"
            elif a > 0.4:
                return "neutral"
            else:
                return "calm" if d > 0.5 else "relaxed"
        else:
            # Émotions négatives
            if a > 0.6:
                return "angry" if d > 0.5 else "anxious"
            elif a > 0.4:
                return "frustrated" if d > 0.5 else "worried"
            else:
                return "sad" if d < 0.5 else "depressed"

    def _record_emotion(self, emotion: str):
        """Enregistre avec métadonnées"""
        self.emotion_history.append(
            {
                "emotion": emotion,
                "pad": self.pad_state.copy(),
                "timestamp": time.time(),
                "cycle": self.cycles,
            }
        )

        if len(self.emotion_history) > self.max_history:
            self.emotion_history = self.emotion_history[-self.max_history :]

    def inject_emotion(self, pleasure: float, arousal: float, dominance: float):
        """Injecte une perturbation émotionnelle"""
        self.pad_state["pleasure"] = max(0, min(1, self.pad_state["pleasure"] + pleasure))
        self.pad_state["arousal"] = max(0, min(1, self.pad_state["arousal"] + arousal))
        self.pad_state["dominance"] = max(0, min(1, self.pad_state["dominance"] + dominance))
        logger.info(f"Emotion injected: ΔP={pleasure:+.2f}, ΔA={arousal:+.2f}, ΔD={dominance:+.2f}")

    def set_equilibrium(self, pleasure: float = None, arousal: float = None, dominance: float = None):
        """Ajuste l'équilibre cible"""
        if pleasure is not None:
            self.equilibrium["pleasure"] = max(0, min(1, pleasure))
        if arousal is not None:
            self.equilibrium["arousal"] = max(0, min(1, arousal))
        if dominance is not None:
            self.equilibrium["dominance"] = max(0, min(1, dominance))
        logger.info(f"Equilibrium updated: {self.equilibrium}")

    def _calculate_reward(self, result: Any) -> float:
        """Récompense pour RL basée sur la stabilité émotionnelle"""
        if not result:
            return 0.0

        # Calculer la distance à l'équilibre
        pad = result.get("pad", self.pad_state)
        distance = sum(abs(pad[k] - self.equilibrium[k]) for k in pad) / 3

        # Récompense inversement proportionnelle à la distance
        # Plus on est proche de l'équilibre, plus la récompense est élevée
        reward = 1.0 - distance

        # Bonus si l'émotion est positive et stable
        emotion = result.get("emotion", "")
        if emotion in ["content", "serene", "calm", "happy"]:
            reward += 0.5

        # Pénalité si émotion négative intense
        if emotion in ["angry", "anxious", "depressed"]:
            reward -= 0.5

        return reward
