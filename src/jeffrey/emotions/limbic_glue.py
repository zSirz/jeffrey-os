"""
Module de système de traitement émotionnel pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de système de traitement émotionnel pour jeffrey os.
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

import logging

from jeffrey.core.neural_envelope import NeuralEnvelope


class LimbicSystem:
    """Wrapper pour ton module d'émotions existant"""

    def __init__(self) -> None:
        # Fallback si module manquant
        self.has_core = False
        try:
            from jeffrey.core.emotions.core.emotion_ml_enhancer import EmotionMLEnhancer

            self.core = EmotionMLEnhancer()  # MODULE EXISTANT
            self.has_core = True
        except (ImportError, SyntaxError) as e:
            logging.warning(f"EmotionMLEnhancer not available: {e} - using fallback")

    async def start(self, bus, registry):
        async def appraise(env: NeuralEnvelope):
            text = env.payload.get("text", "")

            try:
                if self.has_core and hasattr(self.core, "analyze_emotion"):
                    emotion_scores = await self.core.analyze_emotion(text)
                elif self.has_core and hasattr(self.core, "detect_emotion_enhanced"):
                    # Alternative method name
                    result = self.core.detect_emotion_enhanced(text)
                    emotion_scores = result.get("scores", {})
                else:
                    # Fallback simple
                    emotion_scores = _simple_emotion_detection(text)

                affect = {
                    "valence": emotion_scores.get("valence", 0.0),
                    "arousal": emotion_scores.get("arousal", 0.0),
                    "intensity": emotion_scores.get("dominance", 0.5),
                }

                # Enrichir l'envelope
                env.affect = affect
                return affect

            except Exception as e:
                logging.error(f"Emotion error: {e}")
                return {"valence": 0.0, "arousal": 0.0, "intensity": 0.1}

        def _simple_emotion_detection(text: str) -> dict:
            """Détection d'émotion basique en fallback"""
            text_lower = text.lower()

            # Simple heuristiques
            if any(word in text_lower for word in ["happy", "joy", "great", "excellent", "love"]):
                return {"valence": 0.8, "arousal": 0.6, "dominance": 0.7}
            elif any(word in text_lower for word in ["sad", "unhappy", "bad", "terrible", "hate"]):
                return {"valence": -0.8, "arousal": 0.4, "dominance": 0.3}
            elif any(word in text_lower for word in ["angry", "mad", "furious", "rage"]):
                return {"valence": -0.6, "arousal": 0.9, "dominance": 0.8}
            else:
                return {"valence": 0.0, "arousal": 0.5, "dominance": 0.5}

        bus.register_handler("affect.appraise", appraise)

        await registry.register("limbic_system", self, topics_in=["affect.appraise"], topics_out=[])
