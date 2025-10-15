"""
Self Learning Module - Module d'apprentissage adaptatif pour Jeffrey AGI

Fonctionnalités:
- Apprentissage par renforcement des interactions
- Détection et mémorisation des patterns de conversation
- Adaptation automatique des réponses
- Amélioration continue basée sur le feedback
- Métriques d'apprentissage et progression

Intégration: AGI Orchestrator, UnifiedMemory, EmotionalCore
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LearningPattern:
    """Pattern d'apprentissage détecté"""

    pattern_id: str
    input_type: str  # question, salutation, demande_aide, etc.
    successful_responses: list[str]
    failure_responses: list[str]
    confidence: float
    usage_count: int
    last_updated: str
    effectiveness_score: float  # 0.0 à 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class InteractionRecord:
    """Enregistrement d'une interaction pour apprentissage"""

    timestamp: str
    user_input: str
    response_generated: str
    feedback_received: str | None
    user_emotion: str
    response_quality: float  # 0.0 à 1.0
    pattern_matched: str | None
    context_factors: dict[str, Any]

    def to_dict(self) -> dict:
        return asdict(self)


class SelfLearningModule:
    """
    Module d'apprentissage adaptatif pour Jeffrey AGI

    Apprend des interactions passées pour améliorer les réponses futures
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Fichiers de données
        self.patterns_file = self.data_dir / "learning_patterns.json"
        self.interactions_file = self.data_dir / "interaction_history.json"
        self.metrics_file = self.data_dir / "learning_metrics.json"

        # Structures de données en mémoire
        self.learned_patterns: dict[str, LearningPattern] = {}
        self.interaction_history: list[InteractionRecord] = []
        self.learning_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "patterns_learned": 0,
            "avg_response_quality": 0.0,
            "last_learning_session": None,
            "improvement_rate": 0.0,
        }

        # Configuration
        self.max_history_size = 1000
        self.pattern_confidence_threshold = 0.6
        self.learning_rate = 0.1

        # Charger les données existantes
        self.load_learning_data()

        logger.info(f"SelfLearningModule initialized with {len(self.learned_patterns)} patterns")

    def learn_from_interaction(
        self,
        user_input: str,
        response: str,
        feedback: str | None = None,
        user_emotion: str = "neutral",
        context: dict[str, Any] = None,
    ) -> bool:
        """
        Apprend d'une interaction utilisateur

        Args:
            user_input: Entrée utilisateur
            response: Réponse générée par Jeffrey
            feedback: Retour utilisateur (optionnel)
            user_emotion: Émotion détectée de l'utilisateur
            context: Contexte de l'interaction

        Returns:
            bool: True si apprentissage effectué avec succès
        """
        try:
            # Gérer le cas où user_input est un dict
            if isinstance(user_input, dict):
                # Extraire le texte du dict
                user_input = user_input.get('text', user_input.get('message', str(user_input)))
            else:
                user_input = str(user_input)

            # Gérer le cas où response est un dict
            if isinstance(response, dict):
                response = response.get('text', response.get('message', str(response)))
            else:
                response = str(response)

            # Gérer le cas où feedback est un dict
            if isinstance(feedback, dict):
                feedback = feedback.get('text', feedback.get('message', str(feedback)))
            elif feedback is not None:
                feedback = str(feedback)
            # Analyser la qualité de la réponse
            response_quality = self._evaluate_response_quality(user_input, response, feedback, user_emotion)

            # Classifier le type d'entrée
            input_type = self._classify_input_type(user_input)

            # Créer l'enregistrement d'interaction
            interaction = InteractionRecord(
                timestamp=datetime.now().isoformat(),
                user_input=user_input[:200],  # Limiter taille
                response_generated=response[:300],
                feedback_received=feedback,
                user_emotion=user_emotion,
                response_quality=response_quality,
                pattern_matched=self._find_matching_pattern(input_type, user_input),
                context_factors=context or {},
            )

            # Ajouter à l'historique
            self.interaction_history.append(interaction)

            # Limiter taille de l'historique
            if len(self.interaction_history) > self.max_history_size:
                self.interaction_history = self.interaction_history[-self.max_history_size :]

            # Mettre à jour ou créer des patterns
            self._update_learning_patterns(input_type, user_input, response, response_quality)

            # Mettre à jour les métriques
            self._update_learning_metrics(response_quality)

            # Sauvegarder périodiquement
            if len(self.interaction_history) % 10 == 0:
                self.save_learning_data()

            logger.debug(f"Learned from interaction: {input_type} (quality: {response_quality:.2f})")
            return True

        except Exception as e:
            logger.error(f"Error in learn_from_interaction: {e}")
            return False

    def get_learning_suggestion(self, user_input: str, context: dict = None) -> dict[str, Any]:
        """
        Obtient une suggestion basée sur l'apprentissage pour améliorer la réponse

        Args:
            user_input: Entrée utilisateur
            context: Contexte de la conversation

        Returns:
            dict: Suggestions d'amélioration basées sur l'apprentissage
        """
        input_type = self._classify_input_type(user_input)

        suggestion = {
            "input_type": input_type,
            "confidence": 0.0,
            "suggested_approach": "default",
            "pattern_matched": None,
            "response_modifiers": {},
            "learned_insights": [],
        }

        # Chercher patterns correspondants
        matching_patterns = self._find_relevant_patterns(input_type, user_input)

        if matching_patterns:
            best_pattern = max(matching_patterns, key=lambda p: p.effectiveness_score)

            suggestion.update(
                {
                    "confidence": best_pattern.confidence,
                    "pattern_matched": best_pattern.pattern_id,
                    "suggested_approach": self._get_approach_for_pattern(best_pattern),
                    "response_modifiers": self._get_response_modifiers(best_pattern),
                    "learned_insights": self._extract_insights_from_pattern(best_pattern),
                }
            )

        # Ajouter insights contextuels
        contextual_insights = self._get_contextual_insights(user_input, context)
        suggestion["learned_insights"].extend(contextual_insights)

        return suggestion

    def _classify_input_type(self, user_input: str) -> str:
        """Classifier le type d'entrée utilisateur avec plus de précision"""
        input_lower = user_input.lower()

        # Classifications avancées
        if any(word in input_lower for word in ['bonjour', 'salut', 'bonsoir', 'coucou', 'hello']):
            return 'salutation'
        elif any(word in input_lower for word in ['au revoir', 'bye', 'à bientôt', 'bonne nuit']):
            return 'au_revoir'
        elif any(word in input_lower for word in ['merci', 'remercie', 'reconnaissant']):
            return 'remerciement'
        elif any(word in input_lower for word in ['aide', 'aider', 'assistance', 'support', 'dépannage']):
            return 'demande_aide'
        elif any(word in input_lower for word in ['comment', 'pourquoi', 'qu\'est-ce', 'que', 'qui', 'où', 'quand']):
            return 'question'
        elif any(word in input_lower for word in ['raconte', 'histoire', 'parle-moi', 'explique']):
            return 'demande_narration'
        elif any(word in input_lower for word in ['j\'aime', 'adore', 'déteste', 'préfère']):
            return 'expression_preference'
        elif any(word in input_lower for word in ['je me sens', 'je suis', 'ça va', 'humeur']):
            return 'expression_emotion'
        elif any(word in input_lower for word in ['peux-tu', 'pourrais-tu', 'veux-tu']):
            return 'demande_action'
        elif "?" in user_input:
            return 'question_generale'
        elif "!" in user_input and len(user_input.split()) < 5:
            return 'exclamation'
        elif len(user_input.split()) > 15:
            return 'conversation_longue'
        else:
            return 'conversation_generale'

    def _evaluate_response_quality(
        self, user_input: str, response: str, feedback: str | None, user_emotion: str
    ) -> float:
        """Évalue la qualité d'une réponse basée sur plusieurs facteurs"""
        quality_score = 0.5  # Score de base

        # Facteur 1: Feedback explicite
        if feedback:
            feedback_lower = feedback.lower()
            if any(word in feedback_lower for word in ['bien', 'bon', 'super', 'parfait', 'merci']):
                quality_score += 0.3
            elif any(word in feedback_lower for word in ['mal', 'mauvais', 'nul', 'pas bien']):
                quality_score -= 0.3

        # Facteur 2: Longueur de réponse appropriée
        response_length = len(response.split())
        input_length = len(user_input.split())

        if input_length < 5 and response_length < 20:  # Réponse courte pour input court
            quality_score += 0.1
        elif input_length > 10 and response_length > 15:  # Réponse élaborée pour input long
            quality_score += 0.1
        elif response_length < 5:  # Réponse trop courte
            quality_score -= 0.1

        # Facteur 3: Cohérence émotionnelle
        if user_emotion in ['joie', 'confiance'] and any(word in response.lower() for word in ['!', 'super', 'génial']):
            quality_score += 0.1
        elif user_emotion in ['tristesse', 'peur'] and any(
            word in response.lower() for word in ['comprends', 'écoute', 'accompagne']
        ):
            quality_score += 0.1

        # Facteur 4: Présence d'éléments personnalisés
        if any(word in response.lower() for word in ['vous', 'votre', 'tu', 'ton']):
            quality_score += 0.1

        return max(0.0, min(1.0, quality_score))

    def _update_learning_patterns(self, input_type: str, user_input: str, response: str, quality: float):
        """Met à jour les patterns d'apprentissage"""
        pattern_id = f"{input_type}_{hash(user_input[:50]) % 10000}"

        if pattern_id in self.learned_patterns:
            pattern = self.learned_patterns[pattern_id]
            pattern.usage_count += 1

            # Mettre à jour l'efficacité avec moyenne mobile
            pattern.effectiveness_score = pattern.effectiveness_score * 0.8 + quality * 0.2

            # Ajouter à la liste appropriée
            if quality > 0.6:
                if response not in pattern.successful_responses:
                    pattern.successful_responses.append(response[:200])
                    # Limiter la taille des listes
                    if len(pattern.successful_responses) > 10:
                        pattern.successful_responses = pattern.successful_responses[-10:]
            else:
                if response not in pattern.failure_responses:
                    pattern.failure_responses.append(response[:200])
                    if len(pattern.failure_responses) > 5:
                        pattern.failure_responses = pattern.failure_responses[-5:]

            # Recalculer la confiance
            total_responses = len(pattern.successful_responses) + len(pattern.failure_responses)
            if total_responses > 0:
                pattern.confidence = len(pattern.successful_responses) / total_responses

            pattern.last_updated = datetime.now().isoformat()

        else:
            # Créer nouveau pattern
            new_pattern = LearningPattern(
                pattern_id=pattern_id,
                input_type=input_type,
                successful_responses=[response[:200]] if quality > 0.6 else [],
                failure_responses=[response[:200]] if quality <= 0.6 else [],
                confidence=1.0 if quality > 0.6 else 0.0,
                usage_count=1,
                last_updated=datetime.now().isoformat(),
                effectiveness_score=quality,
            )
            self.learned_patterns[pattern_id] = new_pattern

    def _find_relevant_patterns(self, input_type: str, user_input: str) -> list[LearningPattern]:
        """Trouve les patterns pertinents pour une entrée"""
        relevant_patterns = []

        for pattern in self.learned_patterns.values():
            if pattern.input_type == input_type and pattern.confidence > self.pattern_confidence_threshold:
                # Score de pertinence basé sur la similarité et l'efficacité
                relevance_score = pattern.effectiveness_score * pattern.confidence

                if relevance_score > 0.5:
                    relevant_patterns.append(pattern)

        return sorted(relevant_patterns, key=lambda p: p.effectiveness_score, reverse=True)

    def _get_approach_for_pattern(self, pattern: LearningPattern) -> str:
        """Détermine l'approche recommandée basée sur un pattern"""
        if pattern.effectiveness_score > 0.8:
            return "high_confidence"
        elif pattern.effectiveness_score > 0.6:
            return "moderate_confidence"
        elif len(pattern.successful_responses) > len(pattern.failure_responses):
            return "cautious_optimism"
        else:
            return "exploratory"

    def _get_response_modifiers(self, pattern: LearningPattern) -> dict[str, Any]:
        """Génère des modificateurs de réponse basés sur un pattern"""
        modifiers = {"tone": "neutral", "length": "medium", "personalization": "low", "emotional_resonance": "adaptive"}

        # Adapter selon le type d'input et l'efficacité
        if pattern.input_type == "salutation":
            modifiers["tone"] = "friendly"
            modifiers["length"] = "short"
        elif pattern.input_type == "question":
            modifiers["length"] = "detailed"
            modifiers["tone"] = "informative"
        elif pattern.input_type == "expression_emotion":
            modifiers["emotional_resonance"] = "high"
            modifiers["personalization"] = "high"

        # Ajuster selon l'efficacité du pattern
        if pattern.effectiveness_score > 0.8:
            modifiers["confidence"] = "high"
        elif pattern.effectiveness_score > 0.6:
            modifiers["confidence"] = "medium"
        else:
            modifiers["confidence"] = "low"

        return modifiers

    def _extract_insights_from_pattern(self, pattern: LearningPattern) -> list[str]:
        """Extrait des insights d'un pattern d'apprentissage"""
        insights = []

        if pattern.usage_count > 10:
            insights.append(f"Pattern fréquemment utilisé ({pattern.usage_count} fois)")

        if pattern.effectiveness_score > 0.8:
            insights.append("Approche très efficace selon l'historique")
        elif pattern.effectiveness_score < 0.4:
            insights.append("Approche peu efficace, à éviter")

        if len(pattern.successful_responses) > 3:
            insights.append("Plusieurs réponses efficaces disponibles")

        success_rate = pattern.confidence * 100
        insights.append(f"Taux de succès: {success_rate:.1f}%")

        return insights

    def _get_contextual_insights(self, user_input: str, context: dict) -> list[str]:
        """Génère des insights basés sur le contexte"""
        insights = []

        if not context:
            return insights

        # Analyser l'historique récent
        recent_interactions = [
            i
            for i in self.interaction_history[-10:]
            if (datetime.now() - datetime.fromisoformat(i.timestamp)).seconds < 3600
        ]

        if recent_interactions:
            avg_quality = sum(i.response_quality for i in recent_interactions) / len(recent_interactions)
            if avg_quality > 0.7:
                insights.append("Conversation récente de haute qualité")
            elif avg_quality < 0.4:
                insights.append("Qualité conversationnelle à améliorer")

        return insights

    def _find_matching_pattern(self, input_type: str, user_input: str) -> str | None:
        """Trouve le pattern le mieux correspondant"""
        relevant_patterns = self._find_relevant_patterns(input_type, user_input)

        if relevant_patterns:
            return relevant_patterns[0].pattern_id
        return None

    def _update_learning_metrics(self, response_quality: float):
        """Met à jour les métriques d'apprentissage"""
        self.learning_metrics["total_interactions"] += 1
        logger.debug(
            f"📊 Métriques mises à jour: total_interactions={self.learning_metrics['total_interactions']}, quality={response_quality}"
        )

        if response_quality > 0.6:
            self.learning_metrics["successful_interactions"] += 1

        # Moyenne mobile de la qualité
        total = self.learning_metrics["total_interactions"]
        current_avg = self.learning_metrics["avg_response_quality"]
        self.learning_metrics["avg_response_quality"] = (current_avg * (total - 1) + response_quality) / total

        self.learning_metrics["patterns_learned"] = len(self.learned_patterns)
        self.learning_metrics["last_learning_session"] = datetime.now().isoformat()

        # Calculer taux d'amélioration
        if total > 10:
            recent_avg = sum(i.response_quality for i in self.interaction_history[-10:]) / 10
            older_avg = sum(i.response_quality for i in self.interaction_history[-20:-10]) / 10
            self.learning_metrics["improvement_rate"] = (recent_avg - older_avg) * 100

    def get_learning_stats(self) -> dict[str, Any]:
        """Statistiques complètes d'apprentissage"""
        return {
            **self.learning_metrics,
            "patterns_breakdown": {
                input_type: len([p for p in self.learned_patterns.values() if p.input_type == input_type])
                for input_type in set(p.input_type for p in self.learned_patterns.values())
            },
            "quality_distribution": self._get_quality_distribution(),
            "recent_performance": self._get_recent_performance_stats(),
            "top_patterns": self._get_top_performing_patterns(),
        }

    def _get_quality_distribution(self) -> dict[str, int]:
        """Distribution de qualité des interactions récentes"""
        recent_interactions = self.interaction_history[-50:]

        distribution = {
            "excellent": 0,  # > 0.8
            "good": 0,  # 0.6-0.8
            "average": 0,  # 0.4-0.6
            "poor": 0,  # < 0.4
        }

        for interaction in recent_interactions:
            quality = interaction.response_quality
            if quality > 0.8:
                distribution["excellent"] += 1
            elif quality > 0.6:
                distribution["good"] += 1
            elif quality > 0.4:
                distribution["average"] += 1
            else:
                distribution["poor"] += 1

        return distribution

    def _get_recent_performance_stats(self) -> dict[str, float]:
        """Statistiques de performance récente"""
        recent_interactions = self.interaction_history[-20:]

        if not recent_interactions:
            return {"avg_quality": 0.0, "consistency": 0.0, "trend": "stable"}

        qualities = [i.response_quality for i in recent_interactions]
        avg_quality = sum(qualities) / len(qualities)

        # Mesurer la consistance (faible variance = haute consistance)
        variance = sum((q - avg_quality) ** 2 for q in qualities) / len(qualities)
        consistency = max(0, 1 - variance)

        # Détecter la tendance
        if len(qualities) >= 10:
            first_half = sum(qualities[: len(qualities) // 2]) / (len(qualities) // 2)
            second_half = sum(qualities[len(qualities) // 2 :]) / (len(qualities) - len(qualities) // 2)

            if second_half > first_half + 0.1:
                trend = "improving"
            elif second_half < first_half - 0.1:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"

        return {"avg_quality": round(avg_quality, 3), "consistency": round(consistency, 3), "trend": trend}

    def _get_top_performing_patterns(self) -> list[dict[str, Any]]:
        """Top 5 des patterns les plus performants"""
        sorted_patterns = sorted(
            self.learned_patterns.values(), key=lambda p: p.effectiveness_score * p.confidence, reverse=True
        )

        return [
            {
                "input_type": pattern.input_type,
                "effectiveness": round(pattern.effectiveness_score, 3),
                "confidence": round(pattern.confidence, 3),
                "usage_count": pattern.usage_count,
            }
            for pattern in sorted_patterns[:5]
        ]

    def save_learning_data(self):
        """Sauvegarde toutes les données d'apprentissage"""
        try:
            # Sauvegarder patterns
            patterns_data = {pattern_id: pattern.to_dict() for pattern_id, pattern in self.learned_patterns.items()}
            with open(self.patterns_file, 'w', encoding='utf-8') as f:
                json.dump(patterns_data, f, ensure_ascii=False, indent=2)

            # Sauvegarder historique (limité aux 500 derniers)
            interactions_data = [interaction.to_dict() for interaction in self.interaction_history[-500:]]
            with open(self.interactions_file, 'w', encoding='utf-8') as f:
                json.dump(interactions_data, f, ensure_ascii=False, indent=2)

            # Sauvegarder métriques
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.learning_metrics, f, ensure_ascii=False, indent=2)

            logger.info("Learning data saved successfully")

        except Exception as e:
            logger.error(f"Error saving learning data: {e}")

    def load_learning_data(self):
        """Charge les données d'apprentissage existantes"""
        try:
            # Charger patterns
            if self.patterns_file.exists():
                with open(self.patterns_file, encoding='utf-8') as f:
                    patterns_data = json.load(f)

                for pattern_id, pattern_dict in patterns_data.items():
                    self.learned_patterns[pattern_id] = LearningPattern(**pattern_dict)

            # Charger historique
            if self.interactions_file.exists():
                with open(self.interactions_file, encoding='utf-8') as f:
                    interactions_data = json.load(f)

                self.interaction_history = [
                    InteractionRecord(**interaction_dict) for interaction_dict in interactions_data
                ]

            # Charger métriques
            if self.metrics_file.exists():
                with open(self.metrics_file, encoding='utf-8') as f:
                    saved_metrics = json.load(f)
                    self.learning_metrics.update(saved_metrics)

            logger.info(
                f"Loaded {len(self.learned_patterns)} patterns and {len(self.interaction_history)} interactions"
            )

        except Exception as e:
            logger.error(f"Error loading learning data: {e}")

    def reset_learning_data(self):
        """Réinitialise toutes les données d'apprentissage"""
        self.learned_patterns.clear()
        self.interaction_history.clear()
        self.learning_metrics = {
            "total_interactions": 0,
            "successful_interactions": 0,
            "patterns_learned": 0,
            "avg_response_quality": 0.0,
            "last_learning_session": None,
            "improvement_rate": 0.0,
        }

        # Supprimer fichiers
        for file_path in [self.patterns_file, self.interactions_file, self.metrics_file]:
            if file_path.exists():
                file_path.unlink()

        logger.info("Learning data reset successfully")


# Instance globale pour utilisation dans l'AGI Orchestrator
_learning_module_instance = None


def get_learning_module() -> SelfLearningModule:
    """Récupère l'instance globale du module d'apprentissage"""
    global _learning_module_instance
    if _learning_module_instance is None:
        _learning_module_instance = SelfLearningModule()
    return _learning_module_instance
