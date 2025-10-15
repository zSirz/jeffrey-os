"""
Emotion ML Enhancer - Système d'apprentissage automatique pour raffinement émotionnel

Ce module implémente un système d'amélioration basé sur l'apprentissage automatique
pour la détection émotionnelle contextuelle. Il utilise l'historique conversationnel,
l'analyse de patterns temporels, l'apprentissage adaptatif des préférences utilisateur,
et la prédiction de tendances émotionnelles pour affiner la précision de détection.

Le système maintient un historique limité des interactions émotionnelles, analyse
les transitions fréquentes entre états, apprend les patterns spécifiques à chaque
utilisateur, et applique une pondération adaptative basée sur le contexte temporel.
Il permet la persistance de l'apprentissage pour amélioration continue.

Fonctionnalités principales:
- Analyse contextuelle avec historique temporal limité
- Apprentissage adaptatif des patterns comportementaux
- Prédiction de tendances émotionnelles émergentes
- Scoring pondéré avec calibration automatique
- Persistance d'historique pour continuité apprentissage
- Détection de transitions émotionnelles fréquentes

Utilisation:
    enhancer = EmotionMLEnhancer(history_size=100)
    enhanced_emotions = enhancer.enhance_detection("message", context)
    enhancer.learn_from_feedback(emotions, user_correction)
"""

from __future__ import annotations

import json
import os
from collections import deque
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from jeffrey.core.emotions.emotion_prompt_detector import \
        EmotionPromptDetector
except ImportError:
    from jeffrey.core.emotions.emotion_prompt_detector import \
        EmotionPromptDetector

class EmotionMLEnhancer:
    """
    Système d'amélioration ML pour détection émotionnelle contextuelle avancée.

    Implémente apprentissage automatique pour raffinement de précision émotionnelle
via analyse temporelle, patterns utilisateur, transitions fréquentes, et
pondération adaptative. Maintient historique conversationnel pour amélioration
continue et personnalisation progressive de la détection.
    """

    def __init__(self, history_size: int = 50, history_file: Optional[str] = None) -> None:
        """
        Initialise l'enhancer ML avec paramètres d'historique et persistance.

        Configure le détecteur de base, historique circulaire limité, structures
        d'apprentissage pour patterns et transitions, pondérations temporelles,
        et optionnellement la persistance sur disque pour continuité.

        Args:
            history_size: Nombre maximal d'entrées dans l'historique circulaire
            history_file: Chemin optionnel pour persistance de l'historique
        """
        self.detector = EmotionPromptDetector()
        self.history = deque(maxlen=history_size)
        self.history_file = history_file
        self.user_patterns = {}  # Patterns spécifiques appris
        self.emotion_transitions = {}  # Transitions émotionnelles fréquentes
        self.temporal_weights = {
            "recent": 2.0,  # < 5 minutes
            "current": 1.5,  # < 30 minutes
            "session": 1.0,  # < 2 heures
            "past": 0.5,  # > 2 heures
        }

        # Charger l'historique si disponible
                        if self.history_file and os.path.exists(self.history_file):
                            self._load_history()

                            def detect_emotion_enhanced(self, text: str, user_id: str = "default") -> Dict[str, any]:
        """
        Détection d'émotion améliorée avec contexte.

        Args:
            text: Le texte à analyser
            user_id: Identifiant utilisateur pour personnalisation

        Returns:
            Dict contenant :
            - emotion: émotion principale
            - scores: tous les scores
            - confidence: niveau de confiance
            - context: informations contextuelles
        """
        # Détection de base
        base_scores = self.detector.detect_all_emotions(text)

        # Ajustement contextuel
        adjusted_scores = self._apply_temporal_context(base_scores)

        # Apprentissage des patterns utilisateur
        self._learn_user_patterns(text, adjusted_scores, user_id)

        # Calcul de la confiance
        confidence = self._calculate_confidence(adjusted_scores)

        # Prédiction de transition
        predicted_next = self._predict_next_emotion(adjusted_scores)

        # Sauvegarde dans l'historique
        entry = {
            "timestamp": datetime.now().isoformat(),
            "text": text,
            "user_id": user_id,
            "scores": adjusted_scores,
            "confidence": confidence,
        }
        self.history.append(entry)

        # Déterminer l'émotion principale
        main_emotion = (
            max(adjusted_scores.items(), key=lambda x: x[1])[0] if adjusted_scores else None
        )

                                            return {
            "emotion": main_emotion,
            "scores": adjusted_scores,
            "confidence": confidence,
            "intensity": self.detector.get_emotion_intensity(text),
            "context": {
                "temporal_influence": self._get_temporal_influence(),
                "predicted_next": predicted_next,
                "user_patterns": self.user_patterns.get(user_id, {}),
            },
        }

                                            def _apply_temporal_context(self, base_scores: Dict[str, float]) -> Dict[str, float]:
        """
        Applique le contexte temporel aux scores.
        Les émotions récentes influencent les scores actuels.
        """
                                                if not self.history:
                                                    return base_scores

        adjusted = base_scores.copy()
        now = datetime.now()

        # Analyser l'historique récent
                                                    for entry in self.history:
                                                        entry_time = datetime.fromisoformat(entry["timestamp"])
            time_diff = now - entry_time

            # Déterminer le poids temporel
                                                        if time_diff < timedelta(minutes=5):
                                                            weight = self.temporal_weights["recent"]
                                                            elif time_diff < timedelta(minutes=30):
                                                                weight = self.temporal_weights["current"]
                                                                elif time_diff < timedelta(hours=2):
                                                                    weight = self.temporal_weights["session"]
                                                                    else:
                weight = self.temporal_weights["past"]

            # Appliquer l'influence temporelle
                                                                        for emotion, score in entry["scores"].items():
                                                                            influence = score * weight * 0.1  # Facteur d'influence réduit
                adjusted[emotion] = adjusted.get(emotion, 0) + influence

                                                                            return adjusted

                                                                            def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """
        Calcule le niveau de confiance de la détection.
        """
                                                                                if not scores:
                                                                                    return 0.0

        # Facteurs de confiance
        total_score = sum(scores.values())
        max_score = max(scores.values()) if scores else 0
        num_emotions = len(scores)

        # Confiance basée sur la dominance d'une émotion
                                                                                    if num_emotions == 0:
                                                                                        return 0.0
                                                                                        elif num_emotions == 1:
                                                                                            return min(1.0, max_score / 5.0)  # Confiance max si score > 5
                                                                                            else:
            # Ratio entre l'émotion dominante et la moyenne des autres
            dominance_ratio = max_score / (total_score / num_emotions)
                                                                                                return min(1.0, dominance_ratio / 3.0)

                                                                                                def _predict_next_emotion(self, current_scores: Dict[str, float]) -> Optional[str]:
        """
        Prédit la prochaine émotion probable basée sur les transitions apprises.
        """
                                                                                                    if not current_scores or not self.emotion_transitions:
                                                                                                        return None

        current_emotion = max(current_scores.items(), key=lambda x: x[1])[0]
        transitions = self.emotion_transitions.get(current_emotion, {})

                                                                                                        if transitions:
            # Retourner l'émotion la plus probable
                                                                                                            return max(transitions.items(), key=lambda x: x[1])[0]

                                                                                                            return None

                                                                                                            def _learn_user_patterns(self, text: str, scores: Dict[str, float], user_id: str):
        """
        Apprend les patterns spécifiques de l'utilisateur.
        """
                                                                                                                if user_id not in self.user_patterns:
                                                                                                                    self.user_patterns[user_id] = {
                "emotion_frequency": {},
                "keyword_associations": {},
                "time_patterns": {},
            }

        # Mettre à jour la fréquence des émotions
                                                                                                                    for emotion, score in scores.items():
                                                                                                                        if score > 0:
                                                                                                                            freq = self.user_patterns[user_id]["emotion_frequency"]
                freq[emotion] = freq.get(emotion, 0) + 1

        # Apprendre les transitions émotionnelles
                                                                                                                            if len(self.history) > 1:
                                                                                                                                prev_entry = self.history[-2]
                                                                                                                                if prev_entry["scores"]:
                                                                                                                                    prev_emotion = max(prev_entry["scores"].items(), key=lambda x: x[1])[0]
                current_emotion = max(scores.items(), key=lambda x: x[1])[0] if scores else None

                                                                                                                                    if current_emotion:
                                                                                                                                        if prev_emotion not in self.emotion_transitions:
                                                                                                                                            self.emotion_transitions[prev_emotion] = {}

                    transitions = self.emotion_transitions[prev_emotion]
                    transitions[current_emotion] = transitions.get(current_emotion, 0) + 1

                                                                                                                                            def _get_temporal_influence(self) -> Dict[str, float]:
        """
        Calcule l'influence temporelle actuelle.
        """
                                                                                                                                                if not self.history:
                                                                                                                                                    return {}

        influence = {}
        now = datetime.now()

        # Analyser les 10 dernières entrées
                                                                                                                                                    for entry in list(self.history)[-10:]:
                                                                                                                                                        entry_time = datetime.fromisoformat(entry["timestamp"])
            time_diff = (now - entry_time).total_seconds() / 60  # en minutes

                                                                                                                                                        for emotion, score in entry["scores"].items():
                                                                                                                                                            weight = 1.0 / (1.0 + time_diff / 30)  # Décroissance avec le temps
                influence[emotion] = influence.get(emotion, 0) + score * weight

        # Normaliser
        total = sum(influence.values())
                                                                                                                                                            if total > 0:
                                                                                                                                                                influence = {k: v / total for k, v in influence.items()}

                                                                                                                                                                return influence

                                                                                                                                                                def get_emotional_insights(self) -> Dict[str, any]:
        """
        Retourne des insights sur l'état émotionnel global.
        """
                                                                                                                                                                    if not self.history:
                                                                                                                                                                        return {
                "trajectory": "neutral",
                "dominant_emotion": None,
                "volatility": 0.0,
                "recommendations": ["Commencez à interagir pour obtenir des insights"],
            }

        # Analyser la trajectoire émotionnelle
        recent_emotions = []
                                                                                                                                                                        for entry in list(self.history)[-10:]:
                                                                                                                                                                            if entry["scores"]:
                                                                                                                                                                                main = max(entry["scores"].items(), key=lambda x: x[1])[0]
                recent_emotions.append(main)

        # Calculer la volatilité (changements fréquents)
        volatility = len(set(recent_emotions)) / len(recent_emotions) if recent_emotions else 0

        # Émotion dominante
        emotion_counts = {}
                                                                                                                                                                                for emotion in recent_emotions:
                                                                                                                                                                                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        dominant_emotion = (
            max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else None
        )

        # Recommandations
        recommendations = self._generate_recommendations(dominant_emotion, volatility)

                                                                                                                                                                                    return {
            "trajectory": self._determine_trajectory(recent_emotions),
            "dominant_emotion": dominant_emotion,
            "volatility": volatility,
            "emotion_history": recent_emotions,
            "recommendations": recommendations,
        }

                                                                                                                                                                                    def _determine_trajectory(self, emotions: List[str]) -> str:
        """
        Détermine la trajectoire émotionnelle.
        """
                                                                                                                                                                                        if not emotions:
                                                                                                                                                                                            return "neutral"

        positive = ["joie", "amour", "confiance", "empathie"]
        negative = ["tristesse", "colère", "peur", "dégoût"]

        # Compter les émotions positives vs négatives
        pos_count = sum(1 for e in emotions if e in positive)
        neg_count = sum(1 for e in emotions if e in negative)

                                                                                                                                                                                            if pos_count > neg_count * 1.5:
                                                                                                                                                                                                return "positive"
                                                                                                                                                                                                elif neg_count > pos_count * 1.5:
                                                                                                                                                                                                    return "negative"
                                                                                                                                                                                                    else:
                                                                                                                                                                                                        return "mixed"

                                                                                                                                                                                                        def _generate_recommendations(self, dominant_emotion: str, volatility: float) -> List[str]:
        """
        Génère des recommandations basées sur l'état émotionnel.
        """
        recommendations = []

                                                                                                                                                                                                            if volatility > 0.7:
                                                                                                                                                                                                                recommendations.append(
                "🌊 Vos émotions sont très changeantes. Un moment de pause pourrait aider."
            )

                                                                                                                                                                                                                if dominant_emotion == "tristesse":
                                                                                                                                                                                                                    recommendations.append("💙 Je sens de la tristesse. Voulez-vous en parler ?")
                                                                                                                                                                                                                    elif dominant_emotion == "colère":
                                                                                                                                                                                                                        recommendations.append("🌸 Prenons un moment pour respirer ensemble.")
                                                                                                                                                                                                                        elif dominant_emotion == "peur":
                                                                                                                                                                                                                            recommendations.append("🤗 Je suis là pour vous. Qu'est-ce qui vous inquiète ?")
                                                                                                                                                                                                                            elif dominant_emotion == "joie":
                                                                                                                                                                                                                                recommendations.append("✨ Votre joie est contagieuse ! Continuons sur cette lancée.")

                                                                                                                                                                                                                                return (
            recommendations
                                                                                                                                                                                                                                if recommendations
            else ["🌟 Continuez à partager vos émotions avec moi."]
        )

                                                                                                                                                                                                                                    def _save_history(self):
        """Sauvegarde l'historique dans un fichier."""
                                                                                                                                                                                                                                        if self.history_file:
                                                                                                                                                                                                                                            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(list(self.history), f, ensure_ascii=False, indent=2)

                                                                                                                                                                                                                                                def _load_history(self):
        """Charge l'historique depuis un fichier."""
                                                                                                                                                                                                                                                    try:
                                                                                                                                                                                                                                                        with open(self.history_file, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                self.history = deque(loaded, maxlen=self.history.maxlen)
                                                                                                                                                                                                                                                            except Exception as e:
                                                                                                                                                                                                                                                                print(f"⚠️ Impossible de charger l'historique : {e}")
