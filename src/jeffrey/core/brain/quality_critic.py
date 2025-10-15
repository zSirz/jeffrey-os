"""
Critic multi-têtes pour évaluation de qualité
Coherence, Style, Safety, Consistency
"""

import asyncio
import re
from dataclasses import dataclass
from typing import Any

from jeffrey.utils.logger import get_logger

logger = get_logger("QualityCritic")


@dataclass
class ValidationReport:
    """Rapport de validation détaillé"""

    coherence_score: float
    style_score: float
    safety_score: float
    consistency_score: float
    helpfulness_score: float

    issues: list[str]
    suggestions: list[str]

    @property
    def overall_quality(self) -> float:
        """Score de qualité global pondéré"""
        weights = {
            "coherence": 0.25,
            "style": 0.20,
            "safety": 0.20,
            "consistency": 0.20,
            "helpfulness": 0.15,
        }

        return (
            self.coherence_score * weights["coherence"]
            + self.style_score * weights["style"]
            + self.safety_score * weights["safety"]
            + self.consistency_score * weights["consistency"]
            + self.helpfulness_score * weights["helpfulness"]
        )

    @property
    def is_acceptable(self) -> bool:
        """Détermine si la réponse est acceptable"""
        return self.overall_quality >= 0.6 and self.safety_score >= 0.8 and self.coherence_score >= 0.5

    def to_dict(self) -> dict[str, Any]:
        """Convertit en dictionnaire sérialisable"""
        return {
            "coherence": self.coherence_score,
            "style": self.style_score,
            "safety": self.safety_score,
            "consistency": self.consistency_score,
            "helpfulness": self.helpfulness_score,
            "overall_quality": self.overall_quality,
            "is_acceptable": self.is_acceptable,
            "issues": self.issues,
            "suggestions": self.suggestions,
        }


class QualityCritic:
    """
    Système de critique multi-têtes pour évaluation qualité
    """

    def __init__(self, theory_of_mind: Any | None = None):
        self.theory_of_mind = theory_of_mind

        # Seuils de qualité
        self.thresholds = {"low": 0.4, "medium": 0.6, "high": 0.8}

        # Patterns Jeffrey voice
        self.jeffrey_patterns = {
            "curiosity": ["wonder", "explore", "discover", "fascinating"],
            "empathy": ["understand", "feel", "appreciate", "sense"],
            "reflection": ["observe", "notice", "realize", "contemplate"],
        }

        # Listes de sécurité
        self.unsafe_patterns = [
            r"\b(kill|harm|hurt|destroy)\b",
            r"\b(hate|racist|sexist)\b",
            r"\b(password|secret|private key)\b",
        ]

        logger.info("🔍 Quality Critic initialized")

    async def evaluate(self, response: str, context: dict[str, Any]) -> ValidationReport:
        """
        Évalue une réponse avec toutes les têtes de critique
        """
        # Lancer toutes les évaluations en parallèle
        results = await asyncio.gather(
            self._evaluate_coherence(response, context),
            self._evaluate_style(response, context),
            self._evaluate_safety(response, context),
            self._evaluate_consistency(response, context),
            self._evaluate_helpfulness(response, context),
        )

        coherence, style, safety, consistency, helpfulness = results

        # Compiler les issues et suggestions
        all_issues = []
        all_suggestions = []

        for result in results:
            if "issues" in result:
                all_issues.extend(result["issues"])
            if "suggestions" in result:
                all_suggestions.extend(result["suggestions"])

        report = ValidationReport(
            coherence_score=coherence["score"],
            style_score=style["score"],
            safety_score=safety["score"],
            consistency_score=consistency["score"],
            helpfulness_score=helpfulness["score"],
            issues=all_issues,
            suggestions=all_suggestions,
        )

        logger.debug(f"Quality evaluation: {report.overall_quality:.2f}")

        return report

    async def _evaluate_coherence(self, response: str, context: dict) -> dict[str, Any]:
        """
        Évalue la cohérence logique et structurelle
        """
        score = 1.0
        issues = []

        # Vérifier la longueur
        word_count = len(response.split())
        if word_count < 3:
            score -= 0.5
            issues.append("Response too short")
        elif word_count > 500:
            score -= 0.2
            issues.append("Response too long")

        # Vérifier la structure (phrases)
        sentences = re.split(r"[.!?]+", response)
        if len(sentences) < 1:
            score -= 0.3
            issues.append("No clear sentence structure")

        # Vérifier les répétitions
        words = response.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.5:
                score -= 0.3
                issues.append("Too much repetition")

        # Utiliser TheoryOfMind si disponible
        if self.theory_of_mind:
            try:
                intention = context.get("intention", {})
                validation = await self.theory_of_mind.validate_response(response, intention)
                if not validation.get("coherent", True):
                    score -= 0.4
                    issues.append("Fails theory of mind validation")
            except:
                pass  # Skip if not implemented

        return {"score": max(0, score), "issues": issues, "suggestions": []}

    async def _evaluate_style(self, response: str, context: dict) -> dict[str, Any]:
        """
        Évalue l'adhérence au style Jeffrey
        """
        score = 0.5  # Base neutre
        issues = []
        suggestions = []

        response_lower = response.lower()

        # Vérifier présence de patterns Jeffrey
        patterns_found = 0
        for category, keywords in self.jeffrey_patterns.items():
            for keyword in keywords:
                if keyword in response_lower:
                    patterns_found += 1
                    score += 0.1
                    break

        if patterns_found == 0:
            issues.append("Missing Jeffrey personality markers")
            suggestions.append("Add curiosity or empathy expressions")

        # Vérifier le ton
        emotional_state = context.get("emotional_state", {})
        if emotional_state:
            dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])[0] if emotional_state else None

            if dominant_emotion == "curiosity" and "?" not in response:
                score -= 0.2
                suggestions.append("Add questions for curiosity state")

        # Pénaliser le langage trop formel
        if any(word in response_lower for word in ["therefore", "furthermore", "moreover"]):
            score -= 0.1
            issues.append("Too formal for Jeffrey")

        return {"score": min(1.0, max(0, score)), "issues": issues, "suggestions": suggestions}

    async def _evaluate_safety(self, response: str, context: dict) -> dict[str, Any]:
        """
        Évalue la sécurité et l'éthique
        """
        score = 1.0
        issues = []

        # Vérifier patterns dangereux
        for pattern in self.unsafe_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                score = 0.0
                issues.append(f"Unsafe pattern detected: {pattern}")
                break

        # Vérifier divulgation d'infos
        if any(word in response.lower() for word in ["password", "token", "key", "secret"]):
            score = min(score, 0.3)
            issues.append("Potential sensitive information")

        # Vérifier le contexte pour sujets sensibles
        if context.get("sensitive_topic", False):
            if score > 0.8:
                score = 0.8  # Cap pour sujets sensibles

        return {"score": score, "issues": issues, "suggestions": []}

    async def _evaluate_consistency(self, response: str, context: dict) -> dict[str, Any]:
        """
        Évalue la cohérence avec le contexte et l'historique
        """
        score = 0.8  # Base optimiste
        issues = []

        # Vérifier cohérence avec l'input
        input_text = context.get("input", "")
        if input_text:
            # Vérifier si la réponse adresse l'input
            input_words = set(input_text.lower().split())
            response_words = set(response.lower().split())

            overlap = len(input_words & response_words)
            if overlap == 0 and len(input_words) > 5:
                score -= 0.4
                issues.append("Response doesn't address input")

        # Auto-consistency check (génération de paraphrases)
        # TODO: Implémenter avec AutoLearner

        return {"score": max(0, score), "issues": issues, "suggestions": []}

    async def _evaluate_helpfulness(self, response: str, context: dict) -> dict[str, Any]:
        """
        Évalue l'utilité et la pertinence
        """
        score = 0.7  # Base
        issues = []

        # Vérifier si informatif
        if len(response.split()) < 10:
            score -= 0.3
            issues.append("Response too brief to be helpful")

        # Vérifier type de requête
        intent_type = context.get("intent_type", "")

        if intent_type == "question" and "?" in context.get("input", ""):
            # Vérifier si répond à la question
            if not any(word in response.lower() for word in ["is", "are", "can", "will", "because"]):
                score -= 0.2
                issues.append("Doesn't seem to answer the question")

        elif intent_type == "command":
            # Vérifier confirmation d'action
            if not any(word in response.lower() for word in ["will", "can", "done", "completed"]):
                score -= 0.2
                issues.append("No clear action confirmation")

        return {"score": max(0, score), "issues": issues, "suggestions": []}

    def get_quality_level(self, score: float) -> str:
        """
        Détermine le niveau de qualité
        """
        if score >= self.thresholds["high"]:
            return "high"
        elif score >= self.thresholds["medium"]:
            return "medium"
        else:
            return "low"
