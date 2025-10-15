"""
Awareness Loop - Conscience avec pensée multi-vitesses
"""

import logging
import time
from typing import Any

from .base import BaseLoop
from .gates import sanitize_event_data

logger = logging.getLogger(__name__)


class AwarenessLoop(BaseLoop):
    """
    Boucle de conscience avec analyse rapide/profonde
    """

    def __init__(self, bus=None, cognitive_core=None, budget_gate=None):
        super().__init__(
            name="awareness",
            interval_s=10.0,
            jitter_s=0.5,
            hard_timeout_s=1.0,
            budget_gate=budget_gate,
            bus=bus,
        )
        self.cognitive_core = cognitive_core

        # État de conscience
        self.awareness_level = 0.5
        self.attention_focus = None
        self.introspection_depth = 0.3
        self.last_thought = None
        self.consciousness_events = []

        # Pensée multi-vitesses (Gemini)
        self.thinking_mode = "fast"  # fast | deep
        self.deep_thinking_threshold = 0.7

    async def _tick(self):
        """Cycle de conscience avec pensée adaptative"""
        # Observer l'état interne
        state = await self._observe_internal_state()

        # Décider du mode de pensée
        if self.awareness_level > self.deep_thinking_threshold:
            self.thinking_mode = "deep"
            patterns = await self._analyze_patterns_deep(state)
        else:
            self.thinking_mode = "fast"
            patterns = self._analyze_patterns_fast(state)

        # Générer une pensée
        thought = await self._generate_thought(state, patterns)

        # Ajuster le niveau de conscience
        self._adjust_awareness_level(patterns)

        # Publier l'état (avec sanitization)
        if self.bus:
            event_data = {
                "awareness_level": round(self.awareness_level, 2),
                "attention_focus": self.attention_focus,
                "introspection_depth": round(self.introspection_depth, 2),
                "thinking_mode": self.thinking_mode,
                "thought": thought,
                "patterns": patterns,
                "cycle": self.cycles,
            }

            await self.bus.publish(
                "awareness.event",
                {
                    "topic": "consciousness.awareness.update",
                    "data": sanitize_event_data(event_data),
                    "timestamp": time.time(),
                },
            )

        # Garde un historique local boundé (fix GPT)
        self.consciousness_events.append(
            {
                "awareness_level": self.awareness_level,
                "cycle": self.cycles,
                "timestamp": state["timestamp"],
                "thinking_mode": self.thinking_mode,
                "patterns_summary": patterns.get("activity_level", "unknown"),
            }
        )
        if len(self.consciousness_events) > 200:
            self.consciousness_events = self.consciousness_events[-200:]

        self.last_thought = thought
        return {"thought": thought, "patterns": patterns}

    async def _observe_internal_state(self) -> dict[str, Any]:
        """Observer l'état interne (fast)"""
        state = {
            "timestamp": time.time(),
            "memory_usage": 0,
            "emotional_state": None,
            "active_modules": [],
            "recent_interactions": self.cycles % 10,
        }

        if self.cognitive_core:
            # Pensée rapide : juste l'état actuel
            if hasattr(self.cognitive_core, "get_state"):
                core_state = self.cognitive_core.get_state()
                state.update(
                    {
                        "memory_usage": len(core_state.get("recent_memories", [])),
                        "emotional_state": core_state.get("current_emotion"),
                        "active_modules": core_state.get("active_modules", []),
                    }
                )

        return state

    def _analyze_patterns_fast(self, state: dict) -> dict[str, Any]:
        """Analyse rapide des patterns (50ms max)"""
        patterns = {
            "activity_level": "normal",
            "emotional_stability": "stable",
            "memory_activity": "low",
            "anomalies": [],
            "analysis_depth": "shallow",
        }

        # Analyse basique
        if state["recent_interactions"] > 7:
            patterns["activity_level"] = "high"
        elif state["recent_interactions"] < 3:
            patterns["activity_level"] = "low"

        # Vérifier l'état émotionnel
        if state.get("emotional_state"):
            if state["emotional_state"] in ["anxious", "angry", "excited"]:
                patterns["emotional_stability"] = "volatile"

        # Vérifier la mémoire
        if state["memory_usage"] > 50:
            patterns["memory_activity"] = "high"
        elif state["memory_usage"] > 20:
            patterns["memory_activity"] = "medium"

        return patterns

    async def _analyze_patterns_deep(self, state: dict) -> dict[str, Any]:
        """Analyse profonde des patterns (500ms max)"""
        patterns = self._analyze_patterns_fast(state)
        patterns["analysis_depth"] = "deep"

        # Analyses supplémentaires
        if self.consciousness_events:
            # Analyser les tendances
            recent_levels = [e.get("awareness_level", 0.5) for e in self.consciousness_events[-10:]]
            if recent_levels:
                trend = sum(recent_levels) / len(recent_levels)
                patterns["awareness_trend"] = "increasing" if trend > 0.5 else "decreasing"

            # Analyser la stabilité
            if len(recent_levels) > 2:
                variance = sum((x - trend) ** 2 for x in recent_levels) / len(recent_levels)
                if variance > 0.1:
                    patterns["anomalies"].append("high_variance")

        # Détecter les anomalies
        if state["memory_usage"] > 100:
            patterns["anomalies"].append("memory_overflow_risk")

        # Analyser la cohérence des modules actifs
        if len(state.get("active_modules", [])) > 20:
            patterns["anomalies"].append("too_many_active_modules")

        return patterns

    async def _generate_thought(self, state: dict, patterns: dict) -> str:
        """Générer une pensée basée sur l'analyse"""
        thoughts = []

        # Pensées selon l'activité
        if patterns["activity_level"] == "high":
            thoughts.append("I'm experiencing high activity")
        elif patterns["activity_level"] == "low":
            thoughts.append("Things are quiet, time to explore")

        # Pensées sur le mode
        if self.thinking_mode == "deep":
            thoughts.append("I'm thinking deeply about patterns")

        # Pensées sur les anomalies
        if patterns.get("anomalies"):
            thoughts.append(f"I notice {len(patterns['anomalies'])} unusual patterns")

        # Pensées sur la tendance
        if patterns.get("awareness_trend") == "increasing":
            thoughts.append("My awareness is expanding")
        elif patterns.get("awareness_trend") == "decreasing":
            thoughts.append("I'm becoming more focused")

        # Pensée par défaut
        if not thoughts:
            thoughts.append("I am aware and processing")

        return ". ".join(thoughts)

    def _adjust_awareness_level(self, patterns: dict):
        """Ajuste le niveau de conscience"""
        # Augmenter si activité haute
        if patterns["activity_level"] == "high":
            self.awareness_level = min(1.0, self.awareness_level + 0.1)

        # Augmenter si anomalies
        if patterns["anomalies"]:
            self.awareness_level = min(1.0, self.awareness_level + 0.2)

        # Ajuster selon stabilité émotionnelle
        if patterns.get("emotional_stability") == "volatile":
            self.awareness_level = min(1.0, self.awareness_level + 0.05)

        # Décroissance naturelle
        self.awareness_level = max(0.1, self.awareness_level - 0.01)

    def _calculate_reward(self, result: Any) -> float:
        """Récompense pour RL"""
        # Récompense si pensée profonde utile
        if result.get("patterns", {}).get("anomalies"):
            return 10.0  # Anomalie détectée = bon
        elif self.thinking_mode == "deep" and result.get("patterns", {}).get("activity_level") == "low":
            return -1.0  # Pensée profonde inutile
        else:
            return 1.0
