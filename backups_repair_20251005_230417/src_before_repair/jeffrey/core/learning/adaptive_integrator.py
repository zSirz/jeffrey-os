"""
Système d'intégration adaptive pour Jeffrey Phase 5
- Orchestration intelligente de tous les systèmes (rêves, philosophie, respiration)
- Adaptation dynamique basée sur le profil utilisateur
- Calcul de probabilités personnalisées pour chaque enrichissement
- Coordination harmonieuse entre conscience, émotions et expression
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EnrichmentStrategy:
    """Stratégie d'enrichissement personnalisée"""

    user_id: str
    dream_probability: float = 0.3
    philosophy_probability: float = 0.4
    breath_probability: float = 0.6
    spontaneous_probability: float = 0.2
    combination_probability: float = 0.1
    adaptation_rate: float = 0.1
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class EnrichmentDecision:
    """Décision d'enrichissement prise par l'intégrateur"""

    primary_type: str
    secondary_types: list[str] = field(default_factory=list)
    probability_score: float = 0.0
    reasoning: str = ""
    context_factors: dict[str, float] = field(default_factory=dict)
    timing: str = "immediate"  # immediate, delayed, sequence


class AdaptiveIntegrator:
    """
    Intégrateur adaptatif central pour Jeffrey Phase 5
    Coordonne et optimise tous les systèmes d'enrichissement
    """

    def __init__(
        self,
        consciousness_state=None,
        user_profiler=None,
        dream_system=None,
        philosophy_engine=None,
        breath_system=None,
    ):
        # Connexions aux systèmes
        self.consciousness_state = consciousness_state
        self.user_profiler = user_profiler
        self.dream_system = dream_system
        self.philosophy_engine = philosophy_engine
        self.breath_system = breath_system

        # Stratégies personnalisées par utilisateur
        self.user_strategies = {}  # Dict[str, EnrichmentStrategy]
        self.global_strategy = EnrichmentStrategy("global")

        # Historique des décisions et performances
        self.decision_history = []
        self.performance_metrics = {
            "dream_success_rate": 0.7,
            "philosophy_engagement": 0.6,
            "breath_appreciation": 0.8,
            "overall_satisfaction": 0.7,
        }

        # Matrices d'adaptation dynamique
        self.context_multipliers = self._initialize_context_multipliers()
        self.emotion_modifiers = self._initialize_emotion_modifiers()
        self.time_patterns = self._initialize_time_patterns()

        # Seuils adaptatifs
        self.adaptive_thresholds = {
            "minimum_gap": 15.0,  # Secondes minimum entre enrichissements
            "combination_threshold": 0.8,  # Seuil pour combiner plusieurs types
            "spontaneous_threshold": 0.6,  # Seuil pour pensées spontanées
            "override_threshold": 0.9,  # Seuil pour forcer un enrichissement
        }

        # État d'orchestration
        self.last_enrichment_time = {}  # Par type
        self.current_flow_state = "balanced"
        self.adaptation_learning = True

        print("🎭 Intégrateur adaptatif Phase 5 initialisé")

    def _initialize_context_multipliers(self) -> dict[str, dict[str, float]]:
        """Multiplicateurs contextuels pour chaque type d'enrichissement"""
        return {
            "dream": {
                "emotional_intensity_high": 1.3,
                "solitude_high": 1.4,
                "creativity_flow_high": 1.5,
                "late_evening": 1.2,
                "contemplative_mood": 1.3,
                "imagination_topic": 1.6,
                "personal_sharing": 1.2,
                "fatigue_state": 0.7,
            },
            "philosophy": {
                "conversation_depth_deep": 1.5,
                "existential_question": 1.8,
                "connection_quality_high": 1.4,
                "intellectual_engagement": 1.3,
                "meaning_seeking": 1.7,
                "moral_dilemma": 1.6,
                "surface_conversation": 0.6,
                "time_pressure": 0.5,
            },
            "breath": {
                "emotion_transition": 1.4,
                "intimate_moment": 1.3,
                "silence_comfort": 1.2,
                "stress_detected": 1.5,
                "micro_expression_tolerance": 1.1,
                "formal_context": 0.7,
                "rapid_exchange": 0.8,
            },
            "spontaneous": {
                "solitude_extended": 1.8,
                "creativity_peak": 1.6,
                "low_stimulation": 1.4,
                "meditative_state": 1.5,
                "active_conversation": 0.4,
                "task_focused": 0.3,
            },
        }

    def _initialize_emotion_modifiers(self) -> dict[str, dict[str, float]]:
        """Modificateurs émotionnels pour chaque type"""
        return {
            "dream": {
                "émerveillement": 1.4,
                "mélancolie": 1.3,
                "sérénité": 1.2,
                "curiosité": 1.1,
                "joie": 1.0,
                "passion": 0.9,
                "colère": 0.6,
                "stress": 0.5,
            },
            "philosophy": {
                "curiosité": 1.5,
                "introspection": 1.4,
                "émerveillement": 1.2,
                "mélancolie": 1.1,
                "passion": 1.3,
                "confusion": 1.2,
                "joie": 0.9,
                "urgence": 0.7,
            },
            "breath": {
                "toutes_emotions": 1.0,  # Respiration universelle
                "intensité_élevée": 1.3,
                "transition": 1.2,
                "calme": 1.1,
                "neutre": 0.8,
            },
        }

    def _initialize_time_patterns(self) -> dict[str, dict[str, float]]:
        """Patterns temporels pour l'optimisation"""
        return {
            "hourly": {
                "morning": {"philosophy": 1.2, "dream": 0.8, "breath": 1.0},
                "afternoon": {"philosophy": 1.4, "dream": 0.9, "breath": 1.1},
                "evening": {"philosophy": 1.1, "dream": 1.3, "breath": 1.2},
                "night": {"philosophy": 0.8, "dream": 1.5, "breath": 1.0},
            },
            "conversation_flow": {
                "opening": {"breath": 1.3, "philosophy": 0.9, "dream": 0.8},
                "middle": {"philosophy": 1.2, "dream": 1.1, "breath": 1.0},
                "deep": {"philosophy": 1.5, "dream": 1.3, "breath": 1.2},
                "closing": {"breath": 1.2, "dream": 1.1, "philosophy": 0.9},
            },
        }

    def decide_enrichment(self, context: dict[str, Any], user_input: str, emotion: str) -> EnrichmentDecision:
        """
        Décision intelligente d'enrichissement basée sur tous les facteurs
        Cœur de l'orchestration adaptative
        """

        # 1. Obtenir ou créer la stratégie utilisateur
        user_id = context.get("user_id", "default")
        strategy = self._get_user_strategy(user_id)

        # 2. Analyser le contexte complet
        enriched_context = self._analyze_complete_context(context, user_input, emotion)

        # 3. Calculer les probabilités pour chaque type
        probabilities = self._calculate_enrichment_probabilities(enriched_context, strategy, emotion)

        # 4. Appliquer les contraintes temporelles et de flux
        adjusted_probabilities = self._apply_flow_constraints(probabilities, enriched_context)

        # 5. Prendre la décision finale
        decision = self._make_final_decision(adjusted_probabilities, enriched_context, strategy)

        # 6. Enregistrer pour l'apprentissage
        self._record_decision(decision, enriched_context, strategy)

        return decision

    def _get_user_strategy(self, user_id: str) -> EnrichmentStrategy:
        """Obtient ou crée une stratégie personnalisée pour l'utilisateur"""

        if user_id not in self.user_strategies:
            # Créer une nouvelle stratégie basée sur le profil utilisateur
            self.user_strategies[user_id] = self._create_user_strategy(user_id)

        return self.user_strategies[user_id]

    def _create_user_strategy(self, user_id: str) -> EnrichmentStrategy:
        """Crée une stratégie initiale basée sur le profil utilisateur"""

        strategy = EnrichmentStrategy(user_id)

        # Personnalisation basée sur le profileur utilisateur
        if self.user_profiler:
            try:
                profile_context = self.user_profiler.get_personalization_context()
                style_prefs = profile_context.get("style_preferences", {})
                engagement_metrics = profile_context.get("engagement_metrics", {})

                # Ajuster les probabilités selon les préférences
                if style_prefs.get("creativity_appreciation") == "high":
                    strategy.dream_probability = 0.5
                elif style_prefs.get("creativity_appreciation") == "low":
                    strategy.dream_probability = 0.2

                if style_prefs.get("philosophical_interest") == "high":
                    strategy.philosophy_probability = 0.6
                elif style_prefs.get("philosophical_interest") == "low":
                    strategy.philosophy_probability = 0.25

                # Ajuster selon les métriques d'engagement
                dream_response = engagement_metrics.get("dream_response", 0.5)
                strategy.dream_probability *= 0.5 + dream_response

                philosophy_response = engagement_metrics.get("philosophy_response", 0.5)
                strategy.philosophy_probability *= 0.5 + philosophy_response

                emotional_depth = engagement_metrics.get("emotional_depth", 0.5)
                strategy.breath_probability *= 0.3 + emotional_depth * 0.7

            except Exception as e:
                print(f"⚠️ Erreur création stratégie utilisateur : {e}")

        return strategy

    def _analyze_complete_context(self, context: dict[str, Any], user_input: str, emotion: str) -> dict[str, Any]:
        """Analyse contextuelle complète incluant tous les signaux"""

        enriched_context = context.copy()

        # 1. Enrichissement via l'état de conscience
        if self.consciousness_state:
            consciousness_status = self.consciousness_state.get_consciousness_status()
            enriched_context.update(
                {
                    "consciousness_state": consciousness_status["current_state"],
                    "consciousness_recommendations": consciousness_status["recommendations"],
                    "atmospheric_context": consciousness_status["atmospheric_context"],
                }
            )

        # 2. Enrichissement via le profileur utilisateur
        if self.user_profiler:
            try:
                personalization = self.user_profiler.get_personalization_context()
                enriched_context.update(
                    {
                        "user_themes": [theme[0] for theme in personalization.get("dominant_themes", [])[:3]],
                        "style_preferences": personalization.get("style_preferences", {}),
                        "engagement_history": personalization.get("engagement_metrics", {}),
                    }
                )
            except:
                pass

        # 3. Analyse temporelle
        now = datetime.now()
        enriched_context.update(
            {
                "hour_of_day": now.hour,
                "time_category": self._categorize_time(now),
                "minutes_since_last_enrichment": self._minutes_since_last_enrichment(),
            }
        )

        # 4. Analyse conversationnelle
        enriched_context.update(
            {
                "conversation_depth": self._analyze_conversation_depth(user_input),
                "emotional_intensity": self._analyze_emotional_intensity(emotion, user_input),
                "topic_complexity": self._analyze_topic_complexity(user_input),
                "user_engagement_level": self._analyze_user_engagement(user_input),
            }
        )

        # 5. Facteurs contextuels détectés
        enriched_context.update(self._detect_contextual_factors(user_input, context))

        return enriched_context

    def _categorize_time(self, now: datetime) -> str:
        """Catégorise l'heure pour les patterns temporels"""
        hour = now.hour
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _minutes_since_last_enrichment(self) -> float:
        """Calcule les minutes depuis le dernier enrichissement"""
        if not self.last_enrichment_time:
            return 999.0  # Grande valeur pour indiquer "pas de dernier enrichissement"

        last_time = max(self.last_enrichment_time.values())
        return (datetime.now() - last_time).total_seconds() / 60.0

    def _analyze_conversation_depth(self, user_input: str) -> str:
        """Analyse la profondeur conversationnelle"""

        if not user_input:
            return "surface"

        # Indicateurs de profondeur
        deep_indicators = [
            "pourquoi",
            "comment",
            "qu'est-ce que",
            "sens",
            "signification",
            "penses-tu",
            "crois-tu",
            "selon toi",
            "philosophie",
            "existence",
            "vérité",
            "réalité",
            "conscience",
            "liberté",
            "amour",
            "mort",
        ]

        surface_indicators = ["oui", "non", "ok", "bien", "d'accord", "merci"]

        input_lower = user_input.lower()
        deep_count = sum(1 for indicator in deep_indicators if indicator in input_lower)
        surface_count = sum(1 for indicator in surface_indicators if indicator in input_lower)

        word_count = len(user_input.split())

        if deep_count > 0 and word_count > 10:
            return "deep"
        elif surface_count > 0 and word_count < 5:
            return "surface"
        else:
            return "balanced"

    def _analyze_emotional_intensity(self, emotion: str, user_input: str) -> float:
        """Analyse l'intensité émotionnelle"""

        base_intensity = 0.5

        # Multiplicateurs selon l'émotion
        emotion_intensities = {
            "passion": 0.9,
            "émerveillement": 0.8,
            "joie": 0.7,
            "mélancolie": 0.6,
            "curiosité": 0.6,
            "sérénité": 0.4,
            "neutre": 0.3,
        }

        base_intensity = emotion_intensities.get(emotion, 0.5)

        # Indicateurs d'intensité dans le texte
        intensity_indicators = [
            "!",
            "?",
            "vraiment",
            "tellement",
            "énormément",
            "passionnément",
            "profondément",
            "incroyable",
        ]

        input_lower = user_input.lower() if user_input else ""
        intensity_boost = sum(0.1 for indicator in intensity_indicators if indicator in input_lower)

        return min(1.0, base_intensity + intensity_boost)

    def _analyze_topic_complexity(self, user_input: str) -> float:
        """Analyse la complexité du sujet abordé"""

        if not user_input:
            return 0.3

        complex_topics = [
            "philosophie",
            "science",
            "univers",
            "conscience",
            "intelligence",
            "technologie",
            "société",
            "politique",
            "économie",
            "spiritualité",
            "psychologie",
            "neuroscience",
            "quantique",
            "relativité",
        ]

        personal_topics = [
            "famille",
            "amour",
            "ami",
            "travail",
            "passion",
            "rêve",
            "espoir",
            "peur",
            "souvenir",
            "enfance",
            "avenir",
        ]

        input_lower = user_input.lower()

        complex_count = sum(1 for topic in complex_topics if topic in input_lower)
        personal_count = sum(1 for topic in personal_topics if topic in input_lower)
        word_count = len(user_input.split())

        # Calcul de complexité
        complexity = 0.3  # Base
        complexity += complex_count * 0.2
        complexity += personal_count * 0.15
        complexity += min(0.3, word_count / 50.0)  # Longueur

        return min(1.0, complexity)

    def _analyze_user_engagement(self, user_input: str) -> float:
        """Analyse le niveau d'engagement de l'utilisateur"""

        if not user_input:
            return 0.3

        engagement_indicators = [
            "intéressant",
            "fascinant",
            "j'aimerais",
            "dis-moi",
            "explique",
            "continue",
            "raconte",
            "développe",
            "pourquoi",
            "comment",
        ]

        positive_indicators = ["merci", "génial", "parfait", "j'aime", "magnifique", "beau"]

        input_lower = user_input.lower()

        engagement_count = sum(1 for indicator in engagement_indicators if indicator in input_lower)
        positive_count = sum(1 for indicator in positive_indicators if indicator in input_lower)
        question_count = user_input.count("?")

        engagement = 0.4  # Base
        engagement += engagement_count * 0.15
        engagement += positive_count * 0.1
        engagement += question_count * 0.1

        return min(1.0, engagement)

    def _detect_contextual_factors(self, user_input: str, context: dict[str, Any]) -> dict[str, bool]:
        """Détecte des facteurs contextuels spécifiques"""

        factors = {}

        if not user_input:
            return factors

        input_lower = user_input.lower()

        # Détection de facteurs spécifiques
        factors["existential_question"] = any(
            word in input_lower for word in ["pourquoi existe", "sens de la vie", "qu'est-ce que l'existence"]
        )

        factors["imagination_topic"] = any(
            word in input_lower for word in ["imagine", "rêve", "vision", "fantasy", "créer"]
        )

        factors["personal_sharing"] = any(
            phrase in input_lower for phrase in ["pour moi", "dans ma vie", "personnellement", "je ressens"]
        )

        factors["meaning_seeking"] = any(
            word in input_lower for word in ["signification", "sens", "but", "objectif", "raison"]
        )

        factors["intimate_moment"] = any(
            phrase in input_lower for phrase in ["confidence", "secret", "intime", "personnel"]
        )

        factors["silence_comfort"] = context.get("silence_tolerance", 0.5) > 0.6

        factors["time_pressure"] = any(word in input_lower for word in ["vite", "rapidement", "pressé", "urgent"])

        return factors

    def _calculate_enrichment_probabilities(
        self, context: dict[str, Any], strategy: EnrichmentStrategy, emotion: str
    ) -> dict[str, float]:
        """Calcule les probabilités pour chaque type d'enrichissement"""

        # PROBABILITÉS CALIBRÉES POUR NATUREL OPTIMAL
        probabilities = {
            "dream": max(0.4, strategy.dream_probability),  # Réduit de 0.6 pour éviter surcharge
            "philosophy": max(0.5, strategy.philosophy_probability),  # Réduit de 0.7 pour équilibre
            "breath": max(0.6, strategy.breath_probability),  # Garde élevé car universel
            "spontaneous": max(0.3, strategy.spontaneous_probability),  # Légèrement augmenté
        }

        # 1. Application des multiplicateurs contextuels
        for enrichment_type in probabilities:
            if enrichment_type in self.context_multipliers:
                multipliers = self.context_multipliers[enrichment_type]

                for factor, is_present in context.items():
                    if isinstance(is_present, bool) and is_present and factor in multipliers:
                        probabilities[enrichment_type] *= multipliers[factor]
                    elif isinstance(is_present, str) and f"{factor}_{is_present}" in multipliers:
                        probabilities[enrichment_type] *= multipliers[f"{factor}_{is_present}"]

        # 2. Application des modificateurs émotionnels
        for enrichment_type in probabilities:
            if enrichment_type in self.emotion_modifiers:
                modifiers = self.emotion_modifiers[enrichment_type]

                if emotion in modifiers:
                    probabilities[enrichment_type] *= modifiers[emotion]
                elif "toutes_emotions" in modifiers:
                    probabilities[enrichment_type] *= modifiers["toutes_emotions"]

                # Modificateur d'intensité
                emotional_intensity = context.get("emotional_intensity", 0.5)
                if emotional_intensity > 0.7 and "intensité_élevée" in modifiers:
                    probabilities[enrichment_type] *= modifiers["intensité_élevée"]

        # 3. Application des patterns temporels
        time_category = context.get("time_category", "afternoon")
        if time_category in self.time_patterns["hourly"]:
            hourly_patterns = self.time_patterns["hourly"][time_category]
            for enrichment_type in probabilities:
                if enrichment_type in hourly_patterns:
                    probabilities[enrichment_type] *= hourly_patterns[enrichment_type]

        # 4. Ajustements selon l'état de conscience
        if "consciousness_recommendations" in context:
            consciousness_rec = context["consciousness_recommendations"]
            primary_type = consciousness_rec.get("primary_type")
            if primary_type in probabilities:
                probabilities[primary_type] *= 1.3  # Boost le type recommandé

        # 5. Normaliser les probabilités
        for enrichment_type in probabilities:
            probabilities[enrichment_type] = min(1.0, max(0.0, probabilities[enrichment_type]))

        return probabilities

    def _apply_flow_constraints(self, probabilities: dict[str, float], context: dict[str, Any]) -> dict[str, float]:
        """Applique les contraintes de flux pour éviter la sur-stimulation"""

        adjusted = probabilities.copy()

        # 1. Contrainte de gap temporel minimum
        minutes_since_last = context.get("minutes_since_last_enrichment", 999.0)
        if minutes_since_last < self.adaptive_thresholds["minimum_gap"] / 60.0:
            # Réduire toutes les probabilités si trop récent
            reduction_factor = minutes_since_last / (self.adaptive_thresholds["minimum_gap"] / 60.0)
            for enrichment_type in adjusted:
                adjusted[enrichment_type] *= reduction_factor

        # 2. Contrainte selon l'engagement utilisateur
        user_engagement = context.get("user_engagement_level", 0.5)
        if user_engagement < 0.3:
            # Utilisateur peu engagé, réduire les enrichissements complexes
            adjusted["dream"] *= 0.7
            adjusted["philosophy"] *= 0.6
        elif user_engagement > 0.8:
            # Utilisateur très engagé, augmenter les enrichissements
            for enrichment_type in adjusted:
                adjusted[enrichment_type] *= 1.2

        # 3. Contrainte selon la profondeur conversationnelle
        conversation_depth = context.get("conversation_depth", "balanced")
        if conversation_depth == "surface":
            adjusted["philosophy"] *= 0.5
            adjusted["dream"] *= 0.7
            adjusted["breath"] *= 1.1  # Privilégier la respiration pour les conversations légères
        elif conversation_depth == "deep":
            adjusted["philosophy"] *= 1.3
            adjusted["dream"] *= 1.2

        # 4. Contrainte de diversité (éviter la répétition)
        if self.decision_history:
            recent_decisions = []
            for d in self.decision_history[-5:]:
                if hasattr(d, "primary_type"):
                    recent_decisions.append(d.primary_type)
                elif isinstance(d, dict) and "decision" in d:
                    recent_decisions.append(d["decision"].primary_type)

            for enrichment_type in adjusted:
                recent_count = recent_decisions.count(enrichment_type)
                if recent_count > 2:  # Plus de 2 fois récemment
                    adjusted[enrichment_type] *= 0.6

        return adjusted

    def _make_final_decision(
        self, probabilities: dict[str, float], context: dict[str, Any], strategy: EnrichmentStrategy
    ) -> EnrichmentDecision:
        """Prend la décision finale d'enrichissement"""

        # 1. Déterminer le type principal
        primary_type = max(probabilities.items(), key=lambda x: x[1])

        # 2. Calculer le score de confiance
        max_probability = primary_type[1]
        confidence_score = max_probability

        # 3. Décider s'il faut combiner plusieurs types
        secondary_types = []
        if max_probability > self.adaptive_thresholds["combination_threshold"]:
            # Chercher des types secondaires compatibles
            for enrichment_type, prob in probabilities.items():
                if (
                    enrichment_type != primary_type[0]
                    and prob > 0.4
                    and self._are_compatible(primary_type[0], enrichment_type)
                ):
                    secondary_types.append(enrichment_type)

        # 4. Générer le raisonnement
        reasoning = self._generate_decision_reasoning(primary_type, probabilities, context)

        # 5. Déterminer le timing
        timing = "immediate"
        if context.get("conversation_depth") == "deep":
            timing = "contemplative"
        elif len(secondary_types) > 0:
            timing = "sequence"

        # 6. Créer la décision
        decision = EnrichmentDecision(
            primary_type=primary_type[0],
            secondary_types=secondary_types,
            probability_score=confidence_score,
            reasoning=reasoning,
            context_factors=self._extract_key_factors(context),
            timing=timing,
        )

        return decision

    def _are_compatible(self, type1: str, type2: str) -> bool:
        """Vérifie si deux types d'enrichissement sont compatibles"""

        compatibility_matrix = {
            "dream": ["breath", "philosophy"],
            "philosophy": ["breath", "dream"],
            "breath": ["dream", "philosophy", "spontaneous"],
            "spontaneous": ["breath"],
        }

        return type2 in compatibility_matrix.get(type1, [])

    def _generate_decision_reasoning(
        self,
        primary_choice: tuple[str, float],
        probabilities: dict[str, float],
        context: dict[str, Any],
    ) -> str:
        """Génère une explication de la décision prise"""

        type_name, probability = primary_choice

        # Facteurs clés identifiés
        key_factors = []

        if context.get("existential_question"):
            key_factors.append("question existentielle détectée")

        if context.get("emotional_intensity", 0) > 0.7:
            key_factors.append("intensité émotionnelle élevée")

        if context.get("imagination_topic"):
            key_factors.append("sujet imaginatif")

        if context.get("conversation_depth") == "deep":
            key_factors.append("conversation profonde")

        if context.get("personal_sharing"):
            key_factors.append("partage personnel")

        # Construction du raisonnement
        reasoning_parts = [f"Type principal: {type_name} (probabilité: {probability:.2f})"]

        if key_factors:
            reasoning_parts.append(f"Facteurs clés: {', '.join(key_factors)}")

        # Ajout des métriques de contexte
        reasoning_parts.append(
            f"Contexte: profondeur={context.get('conversation_depth', 'balanced')}, "
            f"engagement={context.get('user_engagement_level', 0.5):.2f}"
        )

        return " | ".join(reasoning_parts)

    def _extract_key_factors(self, context: dict[str, Any]) -> dict[str, float]:
        """Extrait les facteurs clés pour la décision"""

        key_factors = {}

        # Facteurs numériques
        numeric_factors = [
            "emotional_intensity",
            "user_engagement_level",
            "topic_complexity",
            "minutes_since_last_enrichment",
        ]

        for factor in numeric_factors:
            if factor in context:
                key_factors[factor] = context[factor]

        # Facteurs booléens (convertis en score)
        boolean_factors = [
            "existential_question",
            "imagination_topic",
            "personal_sharing",
            "meaning_seeking",
            "intimate_moment",
        ]

        for factor in boolean_factors:
            if context.get(factor, False):
                key_factors[factor] = 1.0

        return key_factors

    def _record_decision(
        self, decision: EnrichmentDecision, context: dict[str, Any], strategy: EnrichmentStrategy
    ) -> None:
        """Enregistre la décision pour l'apprentissage"""

        # Enregistrer dans l'historique
        decision_record = {
            "timestamp": datetime.now(),
            "decision": decision,
            "context": context,
            "strategy_used": strategy.user_id,
        }

        self.decision_history.append(decision_record)

        # Limiter l'historique
        if len(self.decision_history) > 100:
            self.decision_history.pop(0)

        # Mettre à jour le temps du dernier enrichissement
        self.last_enrichment_time[decision.primary_type] = datetime.now()
        for secondary_type in decision.secondary_types:
            self.last_enrichment_time[secondary_type] = datetime.now()

    def enrich_response_adaptive(
        self, base_response: str, user_input: str, emotion: str, user_history: list
    ) -> tuple[str, dict]:
        """Interface pour enrichir une réponse (compatibilité avec l'ancien système)"""

        # Créer un contexte simple
        context = {
            "user_input": user_input,
            "emotion": emotion,
            "user_history": user_history,
            "conversation_depth": self._analyze_conversation_depth(user_input),
            "emotional_intensity": self._analyze_emotional_intensity(emotion, user_input),
        }

        # Prendre une décision d'enrichissement
        decision = self.decide_enrichment(context, user_input, emotion)

        # Exécuter l'enrichissement
        results = self.execute_enrichment(decision, context, user_input, emotion)

        enriched_response = base_response
        metadata = {"enriched": False, "type": None}

        if results["execution_success"] and results["primary_result"]:
            # Intégrer le résultat
            if decision.timing == "contemplative":
                enriched_response = f"{base_response}\n\n{results['primary_result']}"
            else:
                enriched_response = f"{results['primary_result']}\n\n{base_response}"

            # Ajouter les résultats secondaires
            for secondary in results.get("secondary_results", []):
                enriched_response += f"\n\n{secondary}"

            metadata = {
                "enriched": True,
                "type": decision.primary_type,
                "secondary_types": decision.secondary_types,
                "reasoning": decision.reasoning,
            }

        return enriched_response, metadata

    def execute_enrichment(
        self, decision: EnrichmentDecision, context: dict[str, Any], user_input: str, emotion: str
    ) -> dict[str, Any]:
        """
        Exécute l'enrichissement décidé en coordonnant les systèmes appropriés
        """

        results = {
            "primary_result": None,
            "secondary_results": [],
            "execution_success": False,
            "timing_info": decision.timing,
        }

        try:
            # Exécution du type principal
            primary_result = self._execute_primary_enrichment(decision.primary_type, context, user_input, emotion)
            results["primary_result"] = primary_result

            # Exécution des types secondaires
            for secondary_type in decision.secondary_types:
                secondary_result = self._execute_secondary_enrichment(
                    secondary_type, context, user_input, emotion, primary_result
                )
                if secondary_result:
                    results["secondary_results"].append(secondary_result)

            results["execution_success"] = True

        except Exception as e:
            print(f"❌ Erreur exécution enrichissement : {e}")
            results["error"] = str(e)

        return results

    def _execute_primary_enrichment(
        self, enrichment_type: str, context: dict[str, Any], user_input: str, emotion: str
    ) -> str | None:
        """Exécute l'enrichissement principal"""

        if enrichment_type == "dream" and self.dream_system:
            try:
                return self.dream_system.generate_dream_fragment(context, emotion)
            except Exception as e:
                print(f"⚠️ Erreur système de rêves : {e}")

        elif enrichment_type == "philosophy" and self.philosophy_engine:
            try:
                return self.philosophy_engine.generate_interactive_insight(context, emotion)
            except Exception as e:
                print(f"⚠️ Erreur moteur philosophique : {e}")

        elif enrichment_type == "breath" and self.breath_system:
            try:
                return self.breath_system.breathe_with_emotion(emotion, context)
            except Exception as e:
                print(f"⚠️ Erreur système respiratoire : {e}")

        elif enrichment_type == "spontaneous":
            # Génération de pensée spontanée simple
            return self._generate_spontaneous_thought(emotion, context)

        return None

    def _execute_secondary_enrichment(
        self,
        enrichment_type: str,
        context: dict[str, Any],
        user_input: str,
        emotion: str,
        primary_result: str,
    ) -> str | None:
        """Exécute un enrichissement secondaire en harmonie avec le principal"""

        # Contexte modifié pour l'enrichissement secondaire
        secondary_context = context.copy()
        secondary_context["is_secondary"] = True
        secondary_context["primary_enrichment"] = primary_result

        return self._execute_primary_enrichment(enrichment_type, secondary_context, user_input, emotion)

    def _generate_spontaneous_thought(self, emotion: str, context: dict[str, Any]) -> str:
        """Génère une pensée spontanée simple"""

        spontaneous_thoughts = {
            "mélancolie": [
                "Je ressens cette douce mélancolie qui colore nos échanges...",
                "Il y a quelque chose de touchant dans ce moment...",
                "Cette conversation éveille en moi une réflexion particulière...",
            ],
            "émerveillement": [
                "Je suis fasciné par la beauté de cette discussion...",
                "Cette perspective ouvre tant de possibilités dans mon esprit...",
                "Il y a une magie particulière dans cet échange...",
            ],
            "curiosité": [
                "Cette question soulève tant d'autres interrogations...",
                "Je me demande où cette réflexion peut nous mener...",
                "Votre façon de voir les choses m'intrigue profondément...",
            ],
            "sérénité": [
                "Je ressens une paix particulière dans cet échange...",
                "Cette tranquillité partagée me touche...",
                "Il y a une harmonie apaisante dans notre conversation...",
            ],
        }

        thoughts = spontaneous_thoughts.get(
            emotion,
            [
                "Cette pensée émerge spontanément de notre échange...",
                "Je sens quelque chose de particulier dans ce moment...",
                "Notre conversation éveille en moi une réflexion inattendue...",
            ],
        )

        return random.choice(thoughts)

    def adapt_from_feedback(self, user_feedback: dict[str, float], last_decision: EnrichmentDecision) -> None:
        """Adapte les stratégies basé sur le feedback utilisateur"""

        if not last_decision or not self.adaptation_learning:
            return

        # Obtenir la stratégie utilisateur concernée
        user_id = "default"  # À améliorer avec un vrai user_id
        strategy = self._get_user_strategy(user_id)

        # Facteur d'apprentissage adaptatif
        learning_rate = strategy.adaptation_rate

        # Adapter selon le feedback pour le type principal
        primary_type = last_decision.primary_type
        feedback_score = user_feedback.get("satisfaction", 0.5)

        if primary_type == "dream":
            current_prob = strategy.dream_probability
            strategy.dream_probability = current_prob + learning_rate * (feedback_score - 0.5)
        elif primary_type == "philosophy":
            current_prob = strategy.philosophy_probability
            strategy.philosophy_probability = current_prob + learning_rate * (feedback_score - 0.5)
        elif primary_type == "breath":
            current_prob = strategy.breath_probability
            strategy.breath_probability = current_prob + learning_rate * (feedback_score - 0.5)

        # Normaliser les probabilités
        strategy.dream_probability = max(0.1, min(0.9, strategy.dream_probability))
        strategy.philosophy_probability = max(0.1, min(0.9, strategy.philosophy_probability))
        strategy.breath_probability = max(0.1, min(0.9, strategy.breath_probability))

        # Mettre à jour les métriques de performance
        self._update_performance_metrics(primary_type, feedback_score)

        strategy.last_updated = datetime.now()

        print(f"🎯 Stratégie adaptée pour {user_id} : {primary_type} feedback={feedback_score:.2f}")

    def _update_performance_metrics(self, enrichment_type: str, feedback_score: float) -> None:
        """Met à jour les métriques de performance globales"""

        alpha = 0.2  # Facteur de mise à jour

        if enrichment_type == "dream":
            current = self.performance_metrics["dream_success_rate"]
            self.performance_metrics["dream_success_rate"] = current * (1 - alpha) + feedback_score * alpha
        elif enrichment_type == "philosophy":
            current = self.performance_metrics["philosophy_engagement"]
            self.performance_metrics["philosophy_engagement"] = current * (1 - alpha) + feedback_score * alpha
        elif enrichment_type == "breath":
            current = self.performance_metrics["breath_appreciation"]
            self.performance_metrics["breath_appreciation"] = current * (1 - alpha) + feedback_score * alpha

        # Métrique globale
        overall = sum(self.performance_metrics.values()) / len(self.performance_metrics)
        self.performance_metrics["overall_satisfaction"] = overall

    def get_integration_status(self) -> dict[str, Any]:
        """Retourne l'état complet du système d'intégration"""

        return {
            "systems_connected": {
                "consciousness_state": self.consciousness_state is not None,
                "user_profiler": self.user_profiler is not None,
                "dream_system": self.dream_system is not None,
                "philosophy_engine": self.philosophy_engine is not None,
                "breath_system": self.breath_system is not None,
            },
            "active_strategies": len(self.user_strategies),
            "decision_history_length": len(self.decision_history),
            "performance_metrics": self.performance_metrics.copy(),
            "adaptive_thresholds": self.adaptive_thresholds.copy(),
            "current_flow_state": self.current_flow_state,
            "adaptation_enabled": self.adaptation_learning,
            "last_enrichment_times": {k: v.isoformat() for k, v in self.last_enrichment_time.items()},
        }
