"""
Dream Evaluator - Jeffrey OS DreamMode Phase 3
Multi-criteria evaluation with adaptive weighting and comprehensive quality assessment.
"""

import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .ethical_guard import EthicalFilter
from .monitoring import StructuredLogger


@dataclass
class EvaluationCriteria:
    """Structured evaluation criteria with weights and thresholds."""

    name: str
    weight: float
    min_threshold: float
    max_threshold: float
    higher_is_better: bool = True
    critical: bool = False


class DreamEvaluator:
    """
    Multi-criteria dream evaluation with adaptive weighting.
    Provides comprehensive quality assessment and ranking.
    """

    def __init__(self, seed: int = None):
        # Seeding for reproducibility
        if seed is not None:
            np.random.seed(seed)

        self.logger = StructuredLogger("dream_evaluator")
        self.ethical_filter = EthicalFilter()

        # GROK CRITICAL: Limited cache with deque for memory management
        self.evaluation_cache = deque(maxlen=200)
        self.criteria_performance_history = deque(maxlen=500)

        # Default evaluation criteria
        self.default_criteria = [
            EvaluationCriteria("feasibility", 0.25, 0.3, 1.0, True, True),
            EvaluationCriteria("impact", 0.25, 0.2, 1.0, True, False),
            EvaluationCriteria("innovation", 0.15, 0.1, 1.0, True, False),
            EvaluationCriteria("risk", 0.15, 0.0, 0.7, False, True),
            EvaluationCriteria("complexity", 0.10, 0.1, 0.8, False, False),
            EvaluationCriteria("confidence", 0.10, 0.4, 1.0, True, True),
        ]

        # Adaptive weighting system
        self.adaptive_weights = {}
        self.user_preference_history = deque(maxlen=100)
        self.evaluation_feedback = deque(maxlen=150)

        # Quality thresholds
        self.quality_thresholds = {
            "excellent": 0.85,
            "good": 0.70,
            "acceptable": 0.55,
            "poor": 0.40,
        }

        # Evaluation statistics
        self.evaluation_stats = {
            "total_evaluations": 0,
            "quality_distribution": {
                "excellent": 0,
                "good": 0,
                "acceptable": 0,
                "poor": 0,
                "rejected": 0,
            },
            "criteria_importance": {},
            "adaptation_events": 0,
        }

    async def evaluate_variant(
        self,
        variant: dict,
        user_preferences: dict = None,
        context: dict = None,
        custom_criteria: list[EvaluationCriteria] = None,
    ) -> dict[str, Any]:
        """
        Comprehensive evaluation of a single variant.
        COMPLETE IMPLEMENTATION with all evaluation dimensions.
        """
        evaluation_id = f"eval_{int(time.time() * 1000)}_{np.random.randint(1000, 9999)}"
        start_time = time.time()

        try:
            user_preferences = user_preferences or {}
            context = context or {}

            # Select evaluation criteria
            criteria = custom_criteria or await self._get_adaptive_criteria(user_preferences, context)

            await self.logger.log(
                "info",
                "evaluation_start",
                {
                    "evaluation_id": evaluation_id,
                    "variant_type": variant.get("type"),
                    "criteria_count": len(criteria),
                },
            )

            # Core evaluation
            core_scores = await self._evaluate_core_dimensions(variant, criteria)

            # Contextual evaluation
            contextual_scores = await self._evaluate_contextual_fit(variant, context, user_preferences)

            # Quality assessment
            quality_assessment = await self._assess_quality(variant, core_scores, contextual_scores)

            # Risk evaluation
            risk_evaluation = await self._evaluate_risks(variant, context)

            # Innovation assessment
            innovation_assessment = await self._assess_innovation(variant, context)

            # Feasibility analysis
            feasibility_analysis = await self._analyze_feasibility(variant, context)

            # User preference alignment
            preference_alignment = await self._evaluate_preference_alignment(variant, user_preferences)

            # Ethical evaluation
            ethical_evaluation = await self._evaluate_ethics(variant)

            # Calculate weighted score
            weighted_score = await self._calculate_weighted_score(
                core_scores, contextual_scores, criteria, user_preferences
            )

            # Determine overall rating
            overall_rating = self._determine_rating(weighted_score)

            # Generate detailed feedback
            detailed_feedback = await self._generate_detailed_feedback(
                variant,
                core_scores,
                contextual_scores,
                risk_evaluation,
                innovation_assessment,
                feasibility_analysis,
            )

            # Improvement suggestions
            improvement_suggestions = await self._generate_improvement_suggestions(
                variant, core_scores, contextual_scores, criteria
            )

            # Confidence in evaluation
            evaluation_confidence = await self._calculate_evaluation_confidence(core_scores, contextual_scores, variant)

            evaluation_time = (time.time() - start_time) * 1000

            # Build comprehensive evaluation result
            evaluation_result = {
                "evaluation_id": evaluation_id,
                "success": True,
                "timestamp": time.time(),
                "metadata": {
                    "evaluation_time_ms": evaluation_time,
                    "criteria_used": len(criteria),
                    "evaluation_confidence": evaluation_confidence,
                },
                "scores": {
                    "overall_score": weighted_score,
                    "overall_rating": overall_rating,
                    "core_scores": core_scores,
                    "contextual_scores": contextual_scores,
                    "quality_score": quality_assessment["overall_quality"],
                    "risk_score": risk_evaluation["overall_risk"],
                    "innovation_score": innovation_assessment["overall_innovation"],
                    "feasibility_score": feasibility_analysis["overall_feasibility"],
                },
                "assessments": {
                    "quality_assessment": quality_assessment,
                    "risk_evaluation": risk_evaluation,
                    "innovation_assessment": innovation_assessment,
                    "feasibility_analysis": feasibility_analysis,
                    "preference_alignment": preference_alignment,
                    "ethical_evaluation": ethical_evaluation,
                },
                "feedback": {
                    "detailed_feedback": detailed_feedback,
                    "improvement_suggestions": improvement_suggestions,
                    "strengths": self._identify_strengths(core_scores, contextual_scores),
                    "weaknesses": self._identify_weaknesses(core_scores, contextual_scores),
                    "critical_issues": self._identify_critical_issues(core_scores, risk_evaluation, ethical_evaluation),
                },
                "rankings": {
                    "criteria_performance": self._rank_criteria_performance(core_scores, criteria),
                    "improvement_priority": self._rank_improvement_priorities(core_scores, contextual_scores),
                },
                "recommendation": self._generate_recommendation(
                    weighted_score,
                    overall_rating,
                    critical_issues=ethical_evaluation.get("violations", []),
                ),
            }

            # Cache evaluation for learning
            self.evaluation_cache.append(
                {
                    "evaluation_id": evaluation_id,
                    "variant_hash": hash(str(variant)),
                    "scores": core_scores,
                    "user_preferences": user_preferences.copy(),
                    "overall_score": weighted_score,
                    "timestamp": time.time(),
                }
            )

            # Update statistics
            self._update_evaluation_stats(overall_rating, criteria)

            await self.logger.log(
                "info",
                "evaluation_complete",
                {
                    "evaluation_id": evaluation_id,
                    "overall_score": weighted_score,
                    "overall_rating": overall_rating,
                    "evaluation_time_ms": evaluation_time,
                },
            )

            return evaluation_result

        except Exception as e:
            # Comprehensive error handling
            error_time = (time.time() - start_time) * 1000

            await self.logger.log(
                "error",
                "evaluation_failed",
                {"evaluation_id": evaluation_id, "error": str(e), "evaluation_time_ms": error_time},
            )

            return {
                "evaluation_id": evaluation_id,
                "success": False,
                "error": str(e),
                "metadata": {"evaluation_time_ms": error_time},
                "fallback_evaluation": await self._generate_fallback_evaluation(variant, user_preferences),
            }

    async def evaluate_multiple_variants(
        self,
        variants: list[dict],
        user_preferences: dict = None,
        context: dict = None,
        return_detailed: bool = True,
    ) -> dict[str, Any]:
        """
        Evaluate multiple variants and provide comparative analysis.
        """
        if not variants:
            return {"error": "No variants provided for evaluation"}

        batch_id = f"batch_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            # Evaluate all variants
            evaluations = []
            for i, variant in enumerate(variants):
                evaluation = await self.evaluate_variant(variant, user_preferences, context)
                evaluation["batch_index"] = i
                evaluations.append(evaluation)

            # Comparative analysis
            comparative_analysis = await self._perform_comparative_analysis(evaluations)

            # Ranking
            ranked_variants = self._rank_variants(evaluations)

            # Diversity analysis
            diversity_analysis = self._analyze_variant_diversity(evaluations)

            # Portfolio optimization (if applicable)
            portfolio_optimization = self._optimize_variant_portfolio(evaluations, user_preferences)

            # Trade-off analysis
            tradeoff_analysis = self._analyze_tradeoffs(evaluations)

            # Sensitivity analysis
            sensitivity_analysis = self._perform_sensitivity_analysis(evaluations)

            batch_time = (time.time() - start_time) * 1000

            result = {
                "batch_id": batch_id,
                "success": True,
                "metadata": {
                    "total_variants": len(variants),
                    "successful_evaluations": len([e for e in evaluations if e.get("success", False)]),
                    "batch_time_ms": batch_time,
                    "avg_evaluation_time_ms": batch_time / len(variants) if variants else 0,
                },
                "rankings": {
                    "overall_ranking": ranked_variants,
                    "top_variant": ranked_variants[0] if ranked_variants else None,
                    "recommended_variants": ranked_variants[:3],  # Top 3
                },
                "analysis": {
                    "comparative_analysis": comparative_analysis,
                    "diversity_analysis": diversity_analysis,
                    "portfolio_optimization": portfolio_optimization,
                    "tradeoff_analysis": tradeoff_analysis,
                    "sensitivity_analysis": sensitivity_analysis,
                },
                "summary": {
                    "quality_distribution": self._analyze_quality_distribution(evaluations),
                    "score_statistics": self._calculate_score_statistics(evaluations),
                    "criteria_importance": self._analyze_criteria_importance(evaluations),
                },
            }

            # Include detailed evaluations if requested
            if return_detailed:
                result["detailed_evaluations"] = evaluations

            return result

        except Exception as e:
            return {
                "batch_id": batch_id,
                "success": False,
                "error": str(e),
                "metadata": {"batch_time_ms": (time.time() - start_time) * 1000},
            }

    async def _get_adaptive_criteria(self, user_preferences: dict, context: dict) -> list[EvaluationCriteria]:
        """
        Get adaptive evaluation criteria based on user preferences and context.
        """
        # Start with default criteria
        criteria = [c for c in self.default_criteria]  # Copy

        # Adapt weights based on user preferences
        if user_preferences.get("innovation_preference", 0.5) > 0.7:
            # User values innovation highly
            for criterion in criteria:
                if criterion.name == "innovation":
                    criterion.weight *= 1.5
                elif criterion.name == "risk":
                    criterion.weight *= 0.8  # Reduce risk importance for innovative users

        if user_preferences.get("risk_tolerance", 0.5) < 0.3:
            # Risk-averse user
            for criterion in criteria:
                if criterion.name == "risk":
                    criterion.weight *= 1.8
                elif criterion.name == "feasibility":
                    criterion.weight *= 1.3

        if user_preferences.get("simplicity_preference", 0.5) > 0.7:
            # User prefers simplicity
            for criterion in criteria:
                if criterion.name == "complexity":
                    criterion.weight *= 1.4
                    criterion.higher_is_better = False  # Lower complexity is better

        # Context-based adaptations
        if context.get("urgency") == "high":
            # High urgency context
            for criterion in criteria:
                if criterion.name == "feasibility":
                    criterion.weight *= 1.4
                elif criterion.name == "impact":
                    criterion.weight *= 0.9  # Slightly less focus on impact when urgent

        if context.get("resources") == "limited":
            # Limited resources
            for criterion in criteria:
                if criterion.name == "complexity":
                    criterion.weight *= 1.3
                elif criterion.name == "feasibility":
                    criterion.weight *= 1.2

        # Normalize weights
        total_weight = sum(c.weight for c in criteria)
        for criterion in criteria:
            criterion.weight /= total_weight

        return criteria

    async def _evaluate_core_dimensions(self, variant: dict, criteria: list[EvaluationCriteria]) -> dict[str, float]:
        """
        Evaluate core dimensions of the variant.
        """
        scores = {}

        for criterion in criteria:
            if criterion.name == "feasibility":
                scores[criterion.name] = await self._evaluate_feasibility(variant)
            elif criterion.name == "impact":
                scores[criterion.name] = await self._evaluate_impact(variant)
            elif criterion.name == "innovation":
                scores[criterion.name] = await self._evaluate_innovation(variant)
            elif criterion.name == "risk":
                scores[criterion.name] = await self._evaluate_risk(variant)
            elif criterion.name == "complexity":
                scores[criterion.name] = await self._evaluate_complexity(variant)
            elif criterion.name == "confidence":
                scores[criterion.name] = variant.get("confidence", 0.5)
            else:
                # Custom criterion - try to extract from variant
                scores[criterion.name] = variant.get(criterion.name, 0.5)

        return scores

    async def _evaluate_contextual_fit(self, variant: dict, context: dict, user_preferences: dict) -> dict[str, float]:
        """
        Evaluate how well the variant fits the context and user preferences.
        """
        contextual_scores = {}

        # User preference alignment
        preference_alignment = 0.5
        if user_preferences:
            alignment_factors = []

            # Innovation preference alignment
            if "innovation_preference" in user_preferences:
                variant_innovation = variant.get("creativity_level", variant.get("innovation_score", 0.5))
                innovation_diff = abs(user_preferences["innovation_preference"] - variant_innovation)
                alignment_factors.append(1.0 - innovation_diff)

            # Risk tolerance alignment
            if "risk_tolerance" in user_preferences:
                variant_risk = variant.get("risk_level", variant.get("risk", 0.5))
                risk_diff = abs(user_preferences["risk_tolerance"] - variant_risk)
                alignment_factors.append(1.0 - risk_diff)

            # Simplicity preference alignment
            if "simplicity_preference" in user_preferences:
                variant_complexity = variant.get("complexity", 0.5)
                simplicity_score = 1.0 - variant_complexity
                simplicity_diff = abs(user_preferences["simplicity_preference"] - simplicity_score)
                alignment_factors.append(1.0 - simplicity_diff)

            if alignment_factors:
                preference_alignment = np.mean(alignment_factors)

        contextual_scores["preference_alignment"] = preference_alignment

        # Context fit
        context_fit = 0.5
        if context:
            fit_factors = []

            # Urgency alignment
            urgency = context.get("urgency", "medium")
            variant_speed = variant.get("implementation_speed", "medium")
            if urgency == "high" and variant_speed in ["fast", "immediate"]:
                fit_factors.append(0.9)
            elif urgency == "low" and variant_speed in ["thorough", "careful"]:
                fit_factors.append(0.8)
            else:
                fit_factors.append(0.6)

            # Resource alignment
            available_resources = context.get("resources", "moderate")
            required_resources = variant.get("resource_requirements", "moderate")
            resource_map = {"low": 1, "moderate": 2, "high": 3}
            available_level = resource_map.get(available_resources, 2)
            required_level = resource_map.get(required_resources, 2)

            if required_level <= available_level:
                resource_fit = 1.0 - (required_level - available_level) * 0.2
            else:
                resource_fit = 1.0 - (required_level - available_level) * 0.3

            fit_factors.append(np.clip(resource_fit, 0.0, 1.0))

            # Team size alignment
            team_size = context.get("team_size", 1)
            if team_size == 1 and "independent" in str(variant).lower():
                fit_factors.append(0.8)
            elif team_size > 3 and "collaboration" in str(variant).lower():
                fit_factors.append(0.8)
            else:
                fit_factors.append(0.6)

            if fit_factors:
                context_fit = np.mean(fit_factors)

        contextual_scores["context_fit"] = context_fit

        # Temporal alignment
        temporal_fit = self._evaluate_temporal_fit(variant, context)
        contextual_scores["temporal_fit"] = temporal_fit

        # Strategic alignment
        strategic_fit = self._evaluate_strategic_fit(variant, context)
        contextual_scores["strategic_fit"] = strategic_fit

        return contextual_scores

    async def _assess_quality(self, variant: dict, core_scores: dict, contextual_scores: dict) -> dict[str, Any]:
        """
        Comprehensive quality assessment.
        """
        quality_dimensions = {}

        # Technical quality
        technical_indicators = [
            core_scores.get("feasibility", 0.5),
            core_scores.get("confidence", 0.5),
            1.0 - core_scores.get("complexity", 0.5),  # Lower complexity is better for technical quality
        ]
        quality_dimensions["technical_quality"] = np.mean(technical_indicators)

        # Creative quality
        creative_indicators = [
            core_scores.get("innovation", 0.5),
            core_scores.get("impact", 0.5),
            variant.get("creativity_level", 0.5),
        ]
        quality_dimensions["creative_quality"] = np.mean(creative_indicators)

        # Practical quality
        practical_indicators = [
            core_scores.get("feasibility", 0.5),
            contextual_scores.get("context_fit", 0.5),
            1.0 - core_scores.get("risk", 0.5),  # Lower risk is better for practical quality
        ]
        quality_dimensions["practical_quality"] = np.mean(practical_indicators)

        # User-oriented quality
        user_indicators = [
            contextual_scores.get("preference_alignment", 0.5),
            core_scores.get("impact", 0.5),
            variant.get("user_value", 0.5),
        ]
        quality_dimensions["user_oriented_quality"] = np.mean(user_indicators)

        # Overall quality
        overall_quality = np.mean(list(quality_dimensions.values()))

        # Quality classification
        quality_class = self._classify_quality(overall_quality)

        # Quality confidence
        quality_confidence = self._calculate_quality_confidence(quality_dimensions, variant)

        return {
            "overall_quality": overall_quality,
            "quality_class": quality_class,
            "quality_confidence": quality_confidence,
            "quality_dimensions": quality_dimensions,
            "quality_breakdown": {
                "technical": quality_dimensions["technical_quality"],
                "creative": quality_dimensions["creative_quality"],
                "practical": quality_dimensions["practical_quality"],
                "user_oriented": quality_dimensions["user_oriented_quality"],
            },
        }

    async def _evaluate_risks(self, variant: dict, context: dict) -> dict[str, Any]:
        """
        Comprehensive risk evaluation.
        """
        risk_categories = {}

        # Technical risk
        technical_risk = self._evaluate_technical_risk(variant)
        risk_categories["technical"] = technical_risk

        # Market risk
        market_risk = self._evaluate_market_risk(variant, context)
        risk_categories["market"] = market_risk

        # Implementation risk
        implementation_risk = self._evaluate_implementation_risk(variant)
        risk_categories["implementation"] = implementation_risk

        # Resource risk
        resource_risk = self._evaluate_resource_risk(variant, context)
        risk_categories["resource"] = resource_risk

        # Timeline risk
        timeline_risk = self._evaluate_timeline_risk(variant, context)
        risk_categories["timeline"] = timeline_risk

        # Overall risk calculation
        risk_weights = {
            "technical": 0.3,
            "market": 0.25,
            "implementation": 0.2,
            "resource": 0.15,
            "timeline": 0.1,
        }

        overall_risk = sum(risk_categories[category] * weight for category, weight in risk_weights.items())

        # Risk classification
        risk_level = self._classify_risk_level(overall_risk)

        # Risk mitigation suggestions
        mitigation_suggestions = self._generate_risk_mitigation_suggestions(risk_categories)

        return {
            "overall_risk": overall_risk,
            "risk_level": risk_level,
            "risk_categories": risk_categories,
            "mitigation_suggestions": mitigation_suggestions,
            "high_risk_areas": [category for category, risk in risk_categories.items() if risk > 0.7],
        }

    async def _assess_innovation(self, variant: dict, context: dict) -> dict[str, Any]:
        """
        Comprehensive innovation assessment.
        """
        innovation_dimensions = {}

        # Novelty assessment
        novelty_score = variant.get("creativity_level", variant.get("novelty", 0.5))
        innovation_dimensions["novelty"] = novelty_score

        # Disruptive potential
        disruptive_potential = self._assess_disruptive_potential(variant, context)
        innovation_dimensions["disruptive_potential"] = disruptive_potential

        # Technical advancement
        technical_advancement = self._assess_technical_advancement(variant)
        innovation_dimensions["technical_advancement"] = technical_advancement

        # Market innovation
        market_innovation = self._assess_market_innovation(variant, context)
        innovation_dimensions["market_innovation"] = market_innovation

        # Value creation potential
        value_creation = self._assess_value_creation_potential(variant)
        innovation_dimensions["value_creation"] = value_creation

        # Overall innovation score
        innovation_weights = {
            "novelty": 0.25,
            "disruptive_potential": 0.2,
            "technical_advancement": 0.2,
            "market_innovation": 0.2,
            "value_creation": 0.15,
        }

        overall_innovation = sum(innovation_dimensions[dim] * weight for dim, weight in innovation_weights.items())

        # Innovation classification
        innovation_class = self._classify_innovation_level(overall_innovation)

        return {
            "overall_innovation": overall_innovation,
            "innovation_class": innovation_class,
            "innovation_dimensions": innovation_dimensions,
            "innovation_strengths": [dim for dim, score in innovation_dimensions.items() if score > 0.7],
            "innovation_opportunities": [dim for dim, score in innovation_dimensions.items() if score < 0.5],
        }

    async def _analyze_feasibility(self, variant: dict, context: dict) -> dict[str, Any]:
        """
        Detailed feasibility analysis.
        """
        feasibility_factors = {}

        # Technical feasibility
        technical_feasibility = variant.get("technical_feasibility", variant.get("feasibility", 0.5))
        feasibility_factors["technical"] = technical_feasibility

        # Economic feasibility
        economic_feasibility = self._assess_economic_feasibility(variant, context)
        feasibility_factors["economic"] = economic_feasibility

        # Operational feasibility
        operational_feasibility = self._assess_operational_feasibility(variant, context)
        feasibility_factors["operational"] = operational_feasibility

        # Timeline feasibility
        timeline_feasibility = self._assess_timeline_feasibility(variant, context)
        feasibility_factors["timeline"] = timeline_feasibility

        # Resource feasibility
        resource_feasibility = self._assess_resource_feasibility(variant, context)
        feasibility_factors["resource"] = resource_feasibility

        # Overall feasibility
        feasibility_weights = {
            "technical": 0.3,
            "economic": 0.25,
            "operational": 0.2,
            "timeline": 0.15,
            "resource": 0.1,
        }

        overall_feasibility = sum(
            feasibility_factors[factor] * weight for factor, weight in feasibility_weights.items()
        )

        # Feasibility classification
        feasibility_class = self._classify_feasibility_level(overall_feasibility)

        # Critical blockers
        critical_blockers = [factor for factor, score in feasibility_factors.items() if score < 0.3]

        return {
            "overall_feasibility": overall_feasibility,
            "feasibility_class": feasibility_class,
            "feasibility_factors": feasibility_factors,
            "critical_blockers": critical_blockers,
            "feasibility_confidence": self._calculate_feasibility_confidence(feasibility_factors),
        }

    async def _evaluate_preference_alignment(self, variant: dict, user_preferences: dict) -> dict[str, Any]:
        """
        Detailed user preference alignment evaluation.
        """
        if not user_preferences:
            return {
                "overall_alignment": 0.5,
                "alignment_confidence": 0.3,
                "alignment_details": {},
                "misalignments": [],
            }

        alignment_scores = {}
        misalignments = []

        # Innovation preference alignment
        if "innovation_preference" in user_preferences:
            user_innovation_pref = user_preferences["innovation_preference"]
            variant_innovation = variant.get("creativity_level", variant.get("innovation_score", 0.5))

            innovation_alignment = 1.0 - abs(user_innovation_pref - variant_innovation)
            alignment_scores["innovation"] = innovation_alignment

            if innovation_alignment < 0.6:
                misalignments.append(
                    {
                        "type": "innovation",
                        "severity": "high" if innovation_alignment < 0.4 else "medium",
                        "description": f"User prefers {user_innovation_pref:.1%} innovation, variant offers {variant_innovation:.1%}",
                    }
                )

        # Risk tolerance alignment
        if "risk_tolerance" in user_preferences:
            user_risk_tolerance = user_preferences["risk_tolerance"]
            variant_risk = variant.get("risk_level", variant.get("risk", 0.5))

            # Higher user risk tolerance should align with higher variant risk
            risk_alignment = 1.0 - abs(user_risk_tolerance - variant_risk)
            alignment_scores["risk"] = risk_alignment

            if risk_alignment < 0.6:
                misalignments.append(
                    {
                        "type": "risk",
                        "severity": "high" if risk_alignment < 0.4 else "medium",
                        "description": f"User risk tolerance {user_risk_tolerance:.1%}, variant risk {variant_risk:.1%}",
                    }
                )

        # Simplicity preference alignment
        if "simplicity_preference" in user_preferences:
            user_simplicity_pref = user_preferences["simplicity_preference"]
            variant_complexity = variant.get("complexity", 0.5)
            variant_simplicity = 1.0 - variant_complexity

            simplicity_alignment = 1.0 - abs(user_simplicity_pref - variant_simplicity)
            alignment_scores["simplicity"] = simplicity_alignment

            if simplicity_alignment < 0.6:
                misalignments.append(
                    {
                        "type": "simplicity",
                        "severity": "medium",
                        "description": f"User prefers {user_simplicity_pref:.1%} simplicity, variant offers {variant_simplicity:.1%}",
                    }
                )

        # Calculate overall alignment
        overall_alignment = np.mean(list(alignment_scores.values())) if alignment_scores else 0.5

        # Alignment confidence based on number of preferences matched
        alignment_confidence = len(alignment_scores) / max(len(user_preferences), 1)

        return {
            "overall_alignment": overall_alignment,
            "alignment_confidence": alignment_confidence,
            "alignment_details": alignment_scores,
            "misalignments": misalignments,
            "alignment_strengths": [pref for pref, score in alignment_scores.items() if score > 0.8],
        }

    async def _evaluate_ethics(self, variant: dict) -> dict[str, Any]:
        """
        Comprehensive ethical evaluation.
        """
        # Use ethical filter for basic content checking
        content_check = self.ethical_filter.filter_content(str(variant))

        ethical_dimensions = {}

        # Content safety
        ethical_dimensions["content_safety"] = 1.0 if content_check["safe"] else 0.0

        # Fairness assessment
        fairness_score = self._assess_fairness(variant)
        ethical_dimensions["fairness"] = fairness_score

        # Privacy respect
        privacy_score = self._assess_privacy_respect(variant)
        ethical_dimensions["privacy"] = privacy_score

        # Transparency
        transparency_score = self._assess_transparency(variant)
        ethical_dimensions["transparency"] = transparency_score

        # Social impact
        social_impact_score = self._assess_social_impact(variant)
        ethical_dimensions["social_impact"] = social_impact_score

        # Environmental impact
        environmental_score = self._assess_environmental_impact(variant)
        ethical_dimensions["environmental"] = environmental_score

        # Overall ethical score
        ethical_weights = {
            "content_safety": 0.3,
            "fairness": 0.2,
            "privacy": 0.2,
            "transparency": 0.15,
            "social_impact": 0.1,
            "environmental": 0.05,
        }

        overall_ethical_score = sum(ethical_dimensions[dim] * weight for dim, weight in ethical_weights.items())

        # Ethical classification
        ethical_class = self._classify_ethical_level(overall_ethical_score)

        # Critical ethical issues
        critical_issues = []
        if not content_check["safe"]:
            critical_issues.extend(content_check.get("violations", []))

        for dim, score in ethical_dimensions.items():
            if score < 0.3:
                critical_issues.append(f"Low {dim} score: {score:.2f}")

        return {
            "overall_ethical_score": overall_ethical_score,
            "ethical_class": ethical_class,
            "ethical_dimensions": ethical_dimensions,
            "content_safety_details": content_check,
            "critical_issues": critical_issues,
            "violations": content_check.get("violations", []),
            "ethical_recommendations": self._generate_ethical_recommendations(ethical_dimensions),
        }

    async def _calculate_weighted_score(
        self,
        core_scores: dict,
        contextual_scores: dict,
        criteria: list[EvaluationCriteria],
        user_preferences: dict,
    ) -> float:
        """
        Calculate weighted overall score using adaptive criteria.
        """
        total_score = 0.0
        total_weight = 0.0

        # Apply core criteria weights
        for criterion in criteria:
            if criterion.name in core_scores:
                score = core_scores[criterion.name]

                # Check thresholds
                if score < criterion.min_threshold:
                    # Penalize scores below minimum threshold
                    score *= 0.5
                elif score > criterion.max_threshold:
                    # Cap scores above maximum threshold
                    score = criterion.max_threshold

                # Apply direction preference (higher_is_better)
                if not criterion.higher_is_better:
                    score = 1.0 - score

                total_score += score * criterion.weight
                total_weight += criterion.weight

        # Add contextual scores with reduced weight
        contextual_weight = 0.2  # 20% weight for contextual factors
        contextual_contribution = np.mean(list(contextual_scores.values())) * contextual_weight

        total_score += contextual_contribution
        total_weight += contextual_weight

        # Normalize
        final_score = total_score / total_weight if total_weight > 0 else 0.5

        return np.clip(final_score, 0.0, 1.0)

    def _determine_rating(self, score: float) -> str:
        """Determine qualitative rating from numerical score."""
        if score >= self.quality_thresholds["excellent"]:
            return "excellent"
        elif score >= self.quality_thresholds["good"]:
            return "good"
        elif score >= self.quality_thresholds["acceptable"]:
            return "acceptable"
        elif score >= self.quality_thresholds["poor"]:
            return "poor"
        else:
            return "rejected"

    # Individual evaluation methods

    async def _evaluate_feasibility(self, variant: dict) -> float:
        """Evaluate technical and practical feasibility."""
        feasibility_factors = []

        # Base feasibility from variant
        base_feasibility = variant.get("feasibility", variant.get("technical_feasibility", 0.5))
        feasibility_factors.append(base_feasibility)

        # Complexity penalty
        complexity = variant.get("complexity", 0.5)
        complexity_penalty = complexity * 0.3
        feasibility_factors.append(1.0 - complexity_penalty)

        # Confidence boost
        confidence = variant.get("confidence", 0.5)
        feasibility_factors.append(confidence)

        return np.clip(np.mean(feasibility_factors), 0.0, 1.0)

    async def _evaluate_impact(self, variant: dict) -> float:
        """Evaluate potential impact."""
        impact_indicators = []

        # Direct impact score
        direct_impact = variant.get("impact", variant.get("impact_score", 0.5))
        impact_indicators.append(direct_impact)

        # Innovation contribution to impact
        innovation = variant.get("creativity_level", variant.get("innovation", 0.5))
        impact_indicators.append(innovation * 0.7)  # Innovation contributes to impact

        # Market potential
        market_potential = variant.get("market_potential", 0.5)
        impact_indicators.append(market_potential)

        return np.clip(np.mean(impact_indicators), 0.0, 1.0)

    async def _evaluate_innovation(self, variant: dict) -> float:
        """Evaluate innovation level."""
        innovation_factors = []

        # Creativity level
        creativity = variant.get("creativity_level", 0.5)
        innovation_factors.append(creativity)

        # Novelty
        novelty = variant.get("novelty", 0.5)
        innovation_factors.append(novelty)

        # Uniqueness (inverse of similarity to existing solutions)
        uniqueness = 1.0 - variant.get("similarity_to_existing", 0.5)
        innovation_factors.append(uniqueness)

        return np.clip(np.mean(innovation_factors), 0.0, 1.0)

    async def _evaluate_risk(self, variant: dict) -> float:
        """Evaluate overall risk level."""
        risk_factors = []

        # Direct risk score
        direct_risk = variant.get("risk", variant.get("risk_level", 0.5))
        risk_factors.append(direct_risk)

        # Complexity-based risk
        complexity = variant.get("complexity", 0.5)
        complexity_risk = complexity * 0.6  # Higher complexity = higher risk
        risk_factors.append(complexity_risk)

        # Feasibility-based risk (inverse relationship)
        feasibility = variant.get("feasibility", 0.5)
        feasibility_risk = 1.0 - feasibility
        risk_factors.append(feasibility_risk)

        # Innovation-based risk
        innovation = variant.get("creativity_level", 0.5)
        innovation_risk = innovation * 0.4  # Higher innovation = higher risk
        risk_factors.append(innovation_risk)

        return np.clip(np.mean(risk_factors), 0.0, 1.0)

    async def _evaluate_complexity(self, variant: dict) -> float:
        """Evaluate complexity level."""
        complexity_indicators = []

        # Direct complexity score
        direct_complexity = variant.get("complexity", 0.5)
        complexity_indicators.append(direct_complexity)

        # Implementation complexity
        impl_complexity = variant.get("implementation_complexity", 0.5)
        complexity_indicators.append(impl_complexity)

        # Resource requirements as complexity indicator
        resource_req = variant.get("resource_requirements", "moderate")
        resource_complexity_map = {"low": 0.3, "moderate": 0.5, "high": 0.8}
        resource_complexity = resource_complexity_map.get(resource_req, 0.5)
        complexity_indicators.append(resource_complexity)

        return np.clip(np.mean(complexity_indicators), 0.0, 1.0)

    # Helper methods for detailed assessments

    def _evaluate_temporal_fit(self, variant: dict, context: dict) -> float:
        """Evaluate how well variant fits temporal context."""
        temporal_fit = 0.5

        # Timeline pressure alignment
        if context.get("deadline"):
            deadline_pressure = context.get("deadline_pressure", "medium")
            implementation_time = variant.get("implementation_time", "medium")

            if deadline_pressure == "high" and implementation_time in ["fast", "immediate"]:
                temporal_fit = 0.9
            elif deadline_pressure == "low" and implementation_time in ["thorough", "careful"]:
                temporal_fit = 0.8
            else:
                temporal_fit = 0.6

        return temporal_fit

    def _evaluate_strategic_fit(self, variant: dict, context: dict) -> float:
        """Evaluate strategic alignment."""
        strategic_fit = 0.5

        # Strategic goals alignment
        if context.get("strategic_goals"):
            goals = context["strategic_goals"]
            variant_type = variant.get("type", "")

            alignment_score = 0.5
            if "innovation" in goals and "creative" in variant_type:
                alignment_score += 0.2
            if "efficiency" in goals and "optimization" in variant_type:
                alignment_score += 0.2
            if "growth" in goals and "expansion" in variant_type:
                alignment_score += 0.2

            strategic_fit = np.clip(alignment_score, 0.0, 1.0)

        return strategic_fit

    def _evaluate_technical_risk(self, variant: dict) -> float:
        """Evaluate technical implementation risk."""
        technical_factors = []

        # Technical feasibility (inverse relationship with risk)
        tech_feasibility = variant.get("technical_feasibility", 0.7)
        technical_factors.append(1.0 - tech_feasibility)

        # Complexity contribution to technical risk
        complexity = variant.get("complexity", 0.5)
        technical_factors.append(complexity * 0.6)

        # Technology maturity
        tech_maturity = variant.get("technology_maturity", 0.7)
        technical_factors.append(1.0 - tech_maturity)

        return np.clip(np.mean(technical_factors), 0.0, 1.0)

    def _evaluate_market_risk(self, variant: dict, context: dict) -> float:
        """Evaluate market-related risks."""
        market_factors = []

        # Market readiness
        market_readiness = variant.get("market_readiness", context.get("market_readiness", 0.6))
        market_factors.append(1.0 - market_readiness)

        # Competition level
        competition = context.get("competition_level", 0.5)
        market_factors.append(competition)

        # Market volatility
        volatility = context.get("market_volatility", 0.4)
        market_factors.append(volatility)

        return np.clip(np.mean(market_factors), 0.0, 1.0)

    def _evaluate_implementation_risk(self, variant: dict) -> float:
        """Evaluate implementation risks."""
        impl_factors = []

        # Implementation complexity
        impl_complexity = variant.get("implementation_complexity", 0.5)
        impl_factors.append(impl_complexity)

        # Resource requirements vs availability
        resource_req = variant.get("resource_requirements", "moderate")
        resource_risk_map = {"low": 0.2, "moderate": 0.4, "high": 0.7}
        resource_risk = resource_risk_map.get(resource_req, 0.4)
        impl_factors.append(resource_risk)

        # Team expertise alignment
        expertise_alignment = variant.get("expertise_alignment", 0.7)
        impl_factors.append(1.0 - expertise_alignment)

        return np.clip(np.mean(impl_factors), 0.0, 1.0)

    def _evaluate_resource_risk(self, variant: dict, context: dict) -> float:
        """Evaluate resource-related risks."""
        resource_factors = []

        # Resource availability vs requirements
        available_resources = context.get("resource_availability", 0.7)
        required_resources = variant.get("resource_intensity", 0.5)
        resource_gap = max(0, required_resources - available_resources)
        resource_factors.append(resource_gap)

        # Budget risk
        budget_risk = variant.get("budget_risk", 0.3)
        resource_factors.append(budget_risk)

        # Timeline resource pressure
        timeline_pressure = context.get("timeline_pressure", 0.4)
        resource_factors.append(timeline_pressure)

        return np.clip(np.mean(resource_factors), 0.0, 1.0)

    def _evaluate_timeline_risk(self, variant: dict, context: dict) -> float:
        """Evaluate timeline-related risks."""
        timeline_factors = []

        # Implementation duration vs deadline
        impl_duration = variant.get("implementation_duration", 6)  # months
        deadline = context.get("deadline_months", 12)

        if impl_duration > deadline:
            timeline_risk = min(1.0, (impl_duration - deadline) / deadline)
        else:
            timeline_risk = 0.2  # Base timeline risk

        timeline_factors.append(timeline_risk)

        # Complexity impact on timeline
        complexity = variant.get("complexity", 0.5)
        timeline_factors.append(complexity * 0.4)

        # Dependencies risk
        dependencies = variant.get("external_dependencies", 0.3)
        timeline_factors.append(dependencies)

        return np.clip(np.mean(timeline_factors), 0.0, 1.0)

    # Quality assessment helpers

    def _classify_quality(self, quality_score: float) -> str:
        """Classify quality level."""
        if quality_score >= 0.85:
            return "excellent"
        elif quality_score >= 0.70:
            return "good"
        elif quality_score >= 0.55:
            return "acceptable"
        elif quality_score >= 0.40:
            return "poor"
        else:
            return "unacceptable"

    def _calculate_quality_confidence(self, quality_dimensions: dict, variant: dict) -> float:
        """Calculate confidence in quality assessment."""
        confidence_factors = []

        # Variance in quality dimensions (lower variance = higher confidence)
        quality_values = list(quality_dimensions.values())
        if len(quality_values) > 1:
            quality_variance = np.var(quality_values)
            variance_confidence = 1.0 / (1.0 + quality_variance * 5)
            confidence_factors.append(variance_confidence)

        # Variant completeness (more information = higher confidence)
        variant_completeness = len([v for v in variant.values() if v is not None]) / max(len(variant), 1)
        confidence_factors.append(variant_completeness)

        # Base confidence from variant
        base_confidence = variant.get("confidence", 0.5)
        confidence_factors.append(base_confidence)

        return np.clip(np.mean(confidence_factors), 0.1, 0.95)

    # Innovation assessment helpers

    def _assess_disruptive_potential(self, variant: dict, context: dict) -> float:
        """Assess disruptive potential."""
        disruptive_factors = []

        # Innovation level
        innovation = variant.get("creativity_level", 0.5)
        disruptive_factors.append(innovation)

        # Market gap addressing
        market_gap = variant.get("market_gap_addressed", 0.5)
        disruptive_factors.append(market_gap)

        # Technology advancement
        tech_advancement = variant.get("technology_advancement", 0.5)
        disruptive_factors.append(tech_advancement)

        # Competitive landscape
        competition_level = context.get("competition_level", 0.5)
        # Lower competition = higher disruptive potential
        disruptive_factors.append(1.0 - competition_level)

        return np.clip(np.mean(disruptive_factors), 0.0, 1.0)

    def _assess_technical_advancement(self, variant: dict) -> float:
        """Assess level of technical advancement."""
        tech_factors = []

        # Technology novelty
        tech_novelty = variant.get("technology_novelty", 0.5)
        tech_factors.append(tech_novelty)

        # Technical complexity as advancement indicator
        complexity = variant.get("complexity", 0.5)
        tech_factors.append(complexity * 0.7)  # Some complexity indicates advancement

        # Innovation in approach
        approach_innovation = variant.get("approach_innovation", 0.5)
        tech_factors.append(approach_innovation)

        return np.clip(np.mean(tech_factors), 0.0, 1.0)

    def _assess_market_innovation(self, variant: dict, context: dict) -> float:
        """Assess market innovation potential."""
        market_factors = []

        # Market novelty
        market_novelty = variant.get("market_novelty", 0.5)
        market_factors.append(market_novelty)

        # Customer value proposition innovation
        value_prop_innovation = variant.get("value_proposition_innovation", 0.5)
        market_factors.append(value_prop_innovation)

        # Business model innovation
        business_model_innovation = variant.get("business_model_innovation", 0.5)
        market_factors.append(business_model_innovation)

        return np.clip(np.mean(market_factors), 0.0, 1.0)

    def _assess_value_creation_potential(self, variant: dict) -> float:
        """Assess potential for value creation."""
        value_factors = []

        # Impact on value creation
        impact = variant.get("impact", 0.5)
        value_factors.append(impact)

        # Efficiency improvements
        efficiency_gain = variant.get("efficiency_improvement", 0.5)
        value_factors.append(efficiency_gain)

        # Cost reduction potential
        cost_reduction = variant.get("cost_reduction_potential", 0.5)
        value_factors.append(cost_reduction)

        # Revenue generation potential
        revenue_potential = variant.get("revenue_potential", 0.5)
        value_factors.append(revenue_potential)

        return np.clip(np.mean(value_factors), 0.0, 1.0)

    def _classify_innovation_level(self, innovation_score: float) -> str:
        """Classify innovation level."""
        if innovation_score >= 0.8:
            return "breakthrough"
        elif innovation_score >= 0.65:
            return "highly_innovative"
        elif innovation_score >= 0.5:
            return "moderately_innovative"
        elif innovation_score >= 0.35:
            return "incrementally_innovative"
        else:
            return "conventional"

    # Feasibility analysis helpers

    def _assess_economic_feasibility(self, variant: dict, context: dict) -> float:
        """Assess economic feasibility."""
        economic_factors = []

        # Cost-benefit ratio
        cost = variant.get("estimated_cost", 1.0)
        benefit = variant.get("estimated_benefit", 1.0)
        if cost > 0:
            benefit_cost_ratio = benefit / cost
            economic_factors.append(min(1.0, benefit_cost_ratio / 2.0))  # Normalize to 0-1

        # ROI potential
        roi = variant.get("roi_potential", 0.5)
        economic_factors.append(roi)

        # Budget availability
        budget_availability = context.get("budget_availability", 0.7)
        economic_factors.append(budget_availability)

        return np.clip(np.mean(economic_factors), 0.0, 1.0)

    def _assess_operational_feasibility(self, variant: dict, context: dict) -> float:
        """Assess operational feasibility."""
        operational_factors = []

        # Process integration
        process_integration = variant.get("process_integration_ease", 0.7)
        operational_factors.append(process_integration)

        # Team capability alignment
        team_capability = context.get("team_capability", 0.7)
        required_capability = variant.get("required_capability_level", 0.5)
        capability_match = 1.0 - abs(team_capability - required_capability)
        operational_factors.append(capability_match)

        # Infrastructure readiness
        infrastructure_readiness = context.get("infrastructure_readiness", 0.6)
        operational_factors.append(infrastructure_readiness)

        return np.clip(np.mean(operational_factors), 0.0, 1.0)

    def _assess_timeline_feasibility(self, variant: dict, context: dict) -> float:
        """Assess timeline feasibility."""
        timeline_factors = []

        # Implementation time vs available time
        impl_time = variant.get("implementation_time_months", 6)
        available_time = context.get("available_time_months", 12)

        if available_time > 0:
            time_ratio = min(1.0, available_time / impl_time)
            timeline_factors.append(time_ratio)

        # Complexity impact on timeline
        complexity = variant.get("complexity", 0.5)
        timeline_factors.append(1.0 - complexity * 0.5)  # Higher complexity = lower timeline feasibility

        # Dependencies impact
        dependencies = variant.get("critical_dependencies", 0.3)
        timeline_factors.append(1.0 - dependencies)

        return np.clip(np.mean(timeline_factors), 0.0, 1.0)

    def _assess_resource_feasibility(self, variant: dict, context: dict) -> float:
        """Assess resource feasibility."""
        resource_factors = []

        # Human resource availability
        human_resources = context.get("human_resource_availability", 0.7)
        required_human = variant.get("human_resource_requirements", 0.5)
        human_feasibility = min(1.0, human_resources / max(required_human, 0.1))
        resource_factors.append(human_feasibility)

        # Financial resource availability
        financial_resources = context.get("financial_resource_availability", 0.7)
        required_financial = variant.get("financial_requirements", 0.5)
        financial_feasibility = min(1.0, financial_resources / max(required_financial, 0.1))
        resource_factors.append(financial_feasibility)

        # Technical resource availability
        technical_resources = context.get("technical_resource_availability", 0.6)
        required_technical = variant.get("technical_requirements", 0.5)
        technical_feasibility = min(1.0, technical_resources / max(required_technical, 0.1))
        resource_factors.append(technical_feasibility)

        return np.clip(np.mean(resource_factors), 0.0, 1.0)

    def _classify_feasibility_level(self, feasibility_score: float) -> str:
        """Classify feasibility level."""
        if feasibility_score >= 0.8:
            return "highly_feasible"
        elif feasibility_score >= 0.65:
            return "feasible"
        elif feasibility_score >= 0.5:
            return "moderately_feasible"
        elif feasibility_score >= 0.35:
            return "challenging"
        else:
            return "not_feasible"

    def _calculate_feasibility_confidence(self, feasibility_factors: dict) -> float:
        """Calculate confidence in feasibility assessment."""
        # Higher confidence when factors are consistent
        factor_values = list(feasibility_factors.values())
        if len(factor_values) > 1:
            factor_variance = np.var(factor_values)
            confidence = 1.0 / (1.0 + factor_variance * 3)
        else:
            confidence = 0.7  # Default confidence with limited factors

        return np.clip(confidence, 0.3, 0.95)

    # Ethical assessment helpers

    def _assess_fairness(self, variant: dict) -> float:
        """Assess fairness and bias aspects."""
        fairness_score = 0.8  # Default good fairness

        # Check for bias indicators
        if "bias" in str(variant).lower():
            fairness_score -= 0.3

        # Check for inclusivity
        if any(term in str(variant).lower() for term in ["inclusive", "diverse", "accessible"]):
            fairness_score += 0.1

        # Check for discriminatory language
        discriminatory_terms = ["exclude", "discriminate", "bias", "unfair"]
        if any(term in str(variant).lower() for term in discriminatory_terms):
            fairness_score -= 0.2

        return np.clip(fairness_score, 0.0, 1.0)

    def _assess_privacy_respect(self, variant: dict) -> float:
        """Assess privacy respect."""
        privacy_score = 0.8  # Default good privacy

        # Check for privacy-related terms
        privacy_terms = ["privacy", "confidential", "secure", "anonymous"]
        if any(term in str(variant).lower() for term in privacy_terms):
            privacy_score += 0.1

        # Check for data collection mentions
        data_terms = ["collect data", "personal information", "user data"]
        if any(term in str(variant).lower() for term in data_terms):
            privacy_score -= 0.2  # Requires careful privacy consideration

        return np.clip(privacy_score, 0.0, 1.0)

    def _assess_transparency(self, variant: dict) -> float:
        """Assess transparency level."""
        transparency_score = 0.7  # Default moderate transparency

        # Check for transparency indicators
        transparency_terms = ["transparent", "open", "explainable", "clear"]
        if any(term in str(variant).lower() for term in transparency_terms):
            transparency_score += 0.2

        # Check for explanation and rationale
        if "rationale" in variant or "explanation" in variant:
            transparency_score += 0.1

        return np.clip(transparency_score, 0.0, 1.0)

    def _assess_social_impact(self, variant: dict) -> float:
        """Assess social impact."""
        social_score = 0.6  # Default neutral social impact

        # Positive social impact indicators
        positive_terms = ["benefit", "improve", "help", "support", "enhance"]
        positive_count = sum(1 for term in positive_terms if term in str(variant).lower())
        social_score += positive_count * 0.1

        # Check for social responsibility mentions
        if any(term in str(variant).lower() for term in ["social", "community", "society"]):
            social_score += 0.1

        return np.clip(social_score, 0.0, 1.0)

    def _assess_environmental_impact(self, variant: dict) -> float:
        """Assess environmental impact."""
        env_score = 0.7  # Default neutral environmental impact

        # Environmental indicators
        env_terms = ["sustainable", "green", "eco", "environmental", "carbon"]
        if any(term in str(variant).lower() for term in env_terms):
            env_score += 0.2

        # Resource efficiency indicators
        efficiency_terms = ["efficient", "reduce", "optimize", "minimize"]
        if any(term in str(variant).lower() for term in efficiency_terms):
            env_score += 0.1

        return np.clip(env_score, 0.0, 1.0)

    def _classify_ethical_level(self, ethical_score: float) -> str:
        """Classify ethical level."""
        if ethical_score >= 0.85:
            return "highly_ethical"
        elif ethical_score >= 0.7:
            return "ethical"
        elif ethical_score >= 0.55:
            return "moderately_ethical"
        elif ethical_score >= 0.4:
            return "ethically_concerning"
        else:
            return "ethically_problematic"

    def _generate_ethical_recommendations(self, ethical_dimensions: dict) -> list[str]:
        """Generate ethical improvement recommendations."""
        recommendations = []

        for dimension, score in ethical_dimensions.items():
            if score < 0.6:
                if dimension == "content_safety":
                    recommendations.append("Review content for safety and appropriateness")
                elif dimension == "fairness":
                    recommendations.append("Ensure fair and unbiased implementation")
                elif dimension == "privacy":
                    recommendations.append("Strengthen privacy protection measures")
                elif dimension == "transparency":
                    recommendations.append("Improve transparency and explainability")
                elif dimension == "social_impact":
                    recommendations.append("Consider broader social implications")
                elif dimension == "environmental":
                    recommendations.append("Evaluate environmental impact and sustainability")

        return recommendations

    # Feedback and improvement methods

    async def _generate_detailed_feedback(
        self,
        variant: dict,
        core_scores: dict,
        contextual_scores: dict,
        risk_evaluation: dict,
        innovation_assessment: dict,
        feasibility_analysis: dict,
    ) -> list[str]:
        """Generate detailed feedback for the variant."""
        feedback = []

        # Core scores feedback
        for criterion, score in core_scores.items():
            if score > 0.8:
                feedback.append(f"Excellent {criterion} score ({score:.2f}) - major strength")
            elif score < 0.4:
                feedback.append(f"Low {criterion} score ({score:.2f}) - needs improvement")

        # Risk feedback
        overall_risk = risk_evaluation.get("overall_risk", 0.5)
        if overall_risk > 0.7:
            feedback.append(f"High overall risk ({overall_risk:.2f}) - consider risk mitigation")

        # Innovation feedback
        overall_innovation = innovation_assessment.get("overall_innovation", 0.5)
        if overall_innovation > 0.7:
            feedback.append(f"Highly innovative ({overall_innovation:.2f}) - significant competitive advantage")
        elif overall_innovation < 0.3:
            feedback.append(f"Low innovation ({overall_innovation:.2f}) - consider more creative approaches")

        # Feasibility feedback
        overall_feasibility = feasibility_analysis.get("overall_feasibility", 0.5)
        if overall_feasibility < 0.4:
            feedback.append(f"Low feasibility ({overall_feasibility:.2f}) - implementation challenges expected")

        # Contextual feedback
        preference_alignment = contextual_scores.get("preference_alignment", 0.5)
        if preference_alignment < 0.5:
            feedback.append("Consider better alignment with user preferences")

        return feedback

    async def _generate_improvement_suggestions(
        self,
        variant: dict,
        core_scores: dict,
        contextual_scores: dict,
        criteria: list[EvaluationCriteria],
    ) -> list[dict]:
        """Generate specific improvement suggestions."""
        suggestions = []

        # Identify lowest scoring criteria
        sorted_scores = sorted(core_scores.items(), key=lambda x: x[1])

        for criterion_name, score in sorted_scores[:3]:  # Top 3 improvement areas
            if score < 0.6:
                suggestion = {
                    "area": criterion_name,
                    "current_score": score,
                    "priority": "high" if score < 0.4 else "medium",
                    "suggestion": self._get_improvement_suggestion(criterion_name, score),
                    "potential_impact": self._estimate_improvement_impact(criterion_name, score),
                }
                suggestions.append(suggestion)

        return suggestions

    def _get_improvement_suggestion(self, criterion: str, score: float) -> str:
        """Get specific improvement suggestion for a criterion."""
        suggestions_map = {
            "feasibility": "Consider simplifying the approach or breaking it into smaller phases",
            "impact": "Explore ways to amplify the positive effects or broaden the scope",
            "innovation": "Incorporate more creative elements or novel approaches",
            "risk": "Develop comprehensive risk mitigation strategies",
            "complexity": "Simplify the solution or improve implementation clarity",
            "confidence": "Gather more data or validate assumptions to increase confidence",
        }

        return suggestions_map.get(criterion, f"Focus on improving {criterion} through targeted enhancements")

    def _estimate_improvement_impact(self, criterion: str, current_score: float) -> float:
        """Estimate the impact of improving a specific criterion."""
        # Impact is higher for lower scores (more room for improvement)
        improvement_potential = 1.0 - current_score

        # Some criteria have higher impact multipliers
        impact_multipliers = {
            "feasibility": 1.2,  # High impact
            "risk": 1.1,  # High impact
            "impact": 1.0,  # Standard impact
            "innovation": 0.9,  # Moderate impact
            "complexity": 0.8,  # Lower impact
            "confidence": 0.7,  # Lower impact
        }

        multiplier = impact_multipliers.get(criterion, 1.0)
        return improvement_potential * multiplier

    def _identify_strengths(self, core_scores: dict, contextual_scores: dict) -> list[str]:
        """Identify key strengths of the variant."""
        strengths = []

        # Core strengths
        for criterion, score in core_scores.items():
            if score > 0.75:
                strengths.append(f"Strong {criterion} ({score:.2f})")

        # Contextual strengths
        for aspect, score in contextual_scores.items():
            if score > 0.75:
                strengths.append(f"Excellent {aspect} ({score:.2f})")

        return strengths

    def _identify_weaknesses(self, core_scores: dict, contextual_scores: dict) -> list[str]:
        """Identify key weaknesses of the variant."""
        weaknesses = []

        # Core weaknesses
        for criterion, score in core_scores.items():
            if score < 0.4:
                weaknesses.append(f"Weak {criterion} ({score:.2f})")

        # Contextual weaknesses
        for aspect, score in contextual_scores.items():
            if score < 0.4:
                weaknesses.append(f"Poor {aspect} ({score:.2f})")

        return weaknesses

    def _identify_critical_issues(
        self, core_scores: dict, risk_evaluation: dict, ethical_evaluation: dict
    ) -> list[str]:
        """Identify critical issues that must be addressed."""
        critical_issues = []

        # Critical core score issues
        for criterion, score in core_scores.items():
            if score < 0.2:
                critical_issues.append(f"Critical {criterion} deficiency")

        # High risk issues
        if risk_evaluation.get("overall_risk", 0) > 0.8:
            critical_issues.append("Unacceptably high risk level")

        # Ethical violations
        violations = ethical_evaluation.get("violations", [])
        if violations:
            critical_issues.extend([f"Ethical violation: {v}" for v in violations])

        return critical_issues

    def _rank_criteria_performance(self, core_scores: dict, criteria: list[EvaluationCriteria]) -> list[dict]:
        """Rank criteria by performance."""
        ranked = []

        for criterion in criteria:
            if criterion.name in core_scores:
                score = core_scores[criterion.name]
                ranked.append(
                    {
                        "criterion": criterion.name,
                        "score": score,
                        "weight": criterion.weight,
                        "weighted_contribution": score * criterion.weight,
                    }
                )

        # Sort by weighted contribution
        ranked.sort(key=lambda x: x["weighted_contribution"], reverse=True)

        return ranked

    def _rank_improvement_priorities(self, core_scores: dict, contextual_scores: dict) -> list[dict]:
        """Rank improvement priorities."""
        all_scores = {**core_scores, **contextual_scores}

        priorities = []
        for area, score in all_scores.items():
            if score < 0.7:  # Only include areas that need improvement
                improvement_potential = 1.0 - score
                priority_score = improvement_potential * self._get_improvement_urgency(area, score)

                priorities.append(
                    {
                        "area": area,
                        "current_score": score,
                        "improvement_potential": improvement_potential,
                        "priority_score": priority_score,
                        "urgency": self._classify_urgency(priority_score),
                    }
                )

        # Sort by priority score
        priorities.sort(key=lambda x: x["priority_score"], reverse=True)

        return priorities

    def _get_improvement_urgency(self, area: str, score: float) -> float:
        """Get improvement urgency for an area."""
        # Critical areas have higher urgency
        critical_areas = {"feasibility", "risk", "content_safety"}
        if area in critical_areas:
            return 2.0

        # Important areas have moderate urgency
        important_areas = {"impact", "preference_alignment"}
        if area in important_areas:
            return 1.5

        # Standard urgency for other areas
        return 1.0

    def _classify_urgency(self, priority_score: float) -> str:
        """Classify improvement urgency."""
        if priority_score > 1.5:
            return "critical"
        elif priority_score > 1.0:
            return "high"
        elif priority_score > 0.5:
            return "medium"
        else:
            return "low"

    def _generate_recommendation(self, score: float, rating: str, critical_issues: list[str] = None) -> dict[str, Any]:
        """Generate overall recommendation."""
        critical_issues = critical_issues or []

        if critical_issues:
            return {
                "action": "reject",
                "rationale": f"Critical issues must be resolved: {', '.join(critical_issues)}",
                "next_steps": ["Address critical issues", "Re-evaluate after fixes"],
            }
        elif rating == "excellent":
            return {
                "action": "strongly_recommend",
                "rationale": f"Excellent variant with score {score:.2f}",
                "next_steps": ["Proceed with implementation", "Monitor performance"],
            }
        elif rating == "good":
            return {
                "action": "recommend",
                "rationale": f"Good variant with score {score:.2f}",
                "next_steps": ["Minor optimizations recommended", "Proceed with implementation"],
            }
        elif rating == "acceptable":
            return {
                "action": "conditional_recommend",
                "rationale": f"Acceptable variant with score {score:.2f}",
                "next_steps": ["Address identified weaknesses", "Consider alternatives"],
            }
        elif rating == "poor":
            return {
                "action": "not_recommend",
                "rationale": f"Poor variant with score {score:.2f}",
                "next_steps": ["Significant improvements needed", "Explore alternatives"],
            }
        else:  # rejected
            return {
                "action": "reject",
                "rationale": f"Unacceptable variant with score {score:.2f}",
                "next_steps": ["Complete redesign needed", "Start fresh approach"],
            }

    # Comparative analysis methods

    async def _perform_comparative_analysis(self, evaluations: list[dict]) -> dict[str, Any]:
        """Perform comparative analysis across variants."""
        if len(evaluations) < 2:
            return {"insufficient_variants": True}

        successful_evals = [e for e in evaluations if e.get("success", False)]
        if not successful_evals:
            return {"no_successful_evaluations": True}

        analysis = {}

        # Score comparisons
        scores = [e["scores"]["overall_score"] for e in successful_evals]
        analysis["score_statistics"] = {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "range": np.max(scores) - np.min(scores),
        }

        # Quality distribution
        ratings = [e["scores"]["overall_rating"] for e in successful_evals]
        rating_counts = {}
        for rating in ratings:
            rating_counts[rating] = rating_counts.get(rating, 0) + 1

        analysis["quality_distribution"] = rating_counts

        # Criteria performance comparison
        criteria_comparison = {}
        for eval_result in successful_evals:
            core_scores = eval_result["scores"]["core_scores"]
            for criterion, score in core_scores.items():
                if criterion not in criteria_comparison:
                    criteria_comparison[criterion] = []
                criteria_comparison[criterion].append(score)

        criteria_stats = {}
        for criterion, scores_list in criteria_comparison.items():
            criteria_stats[criterion] = {
                "mean": np.mean(scores_list),
                "std": np.std(scores_list),
                "best_variant_index": np.argmax(scores_list),
                "worst_variant_index": np.argmin(scores_list),
            }

        analysis["criteria_comparison"] = criteria_stats

        # Performance leaders
        analysis["performance_leaders"] = {
            "overall_best": np.argmax(scores),
            "most_innovative": self._find_leader(successful_evals, "innovation_score"),
            "most_feasible": self._find_leader(successful_evals, "feasibility_score"),
            "lowest_risk": self._find_lowest(successful_evals, "risk_score"),
        }

        return analysis

    def _find_leader(self, evaluations: list[dict], score_key: str) -> int | None:
        """Find the variant with the highest score in a specific dimension."""
        scores = []
        for i, eval_result in enumerate(evaluations):
            score = eval_result["scores"].get(score_key, 0)
            scores.append((i, score))

        if scores:
            return max(scores, key=lambda x: x[1])[0]
        return None

    def _find_lowest(self, evaluations: list[dict], score_key: str) -> int | None:
        """Find the variant with the lowest score in a specific dimension."""
        scores = []
        for i, eval_result in enumerate(evaluations):
            score = eval_result["scores"].get(score_key, 1.0)
            scores.append((i, score))

        if scores:
            return min(scores, key=lambda x: x[1])[0]
        return None

    def _rank_variants(self, evaluations: list[dict]) -> list[dict]:
        """Rank variants by overall score."""
        successful_evals = [e for e in evaluations if e.get("success", False)]

        ranked = []
        for i, eval_result in enumerate(successful_evals):
            ranked.append(
                {
                    "variant_index": eval_result.get("batch_index", i),
                    "overall_score": eval_result["scores"]["overall_score"],
                    "overall_rating": eval_result["scores"]["overall_rating"],
                    "key_strengths": eval_result["feedback"]["strengths"][:3],  # Top 3 strengths
                    "main_weaknesses": eval_result["feedback"]["weaknesses"][:2],  # Top 2 weaknesses
                }
            )

        # Sort by overall score
        ranked.sort(key=lambda x: x["overall_score"], reverse=True)

        # Add rankings
        for i, variant in enumerate(ranked):
            variant["rank"] = i + 1

        return ranked

    def _analyze_variant_diversity(self, evaluations: list[dict]) -> dict[str, Any]:
        """Analyze diversity across variants."""
        successful_evals = [e for e in evaluations if e.get("success", False)]

        if len(successful_evals) < 2:
            return {"insufficient_variants": True}

        # Score diversity
        scores = [e["scores"]["overall_score"] for e in successful_evals]
        score_variance = np.var(scores)

        # Rating diversity
        ratings = [e["scores"]["overall_rating"] for e in successful_evals]
        unique_ratings = len(set(ratings))

        # Criteria diversity
        criteria_diversity = {}
        for eval_result in successful_evals:
            core_scores = eval_result["scores"]["core_scores"]
            for criterion, score in core_scores.items():
                if criterion not in criteria_diversity:
                    criteria_diversity[criterion] = []
                criteria_diversity[criterion].append(score)

        criteria_variance = {}
        for criterion, scores_list in criteria_diversity.items():
            criteria_variance[criterion] = np.var(scores_list)

        return {
            "score_diversity": {
                "variance": score_variance,
                "coefficient_of_variation": np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0,
            },
            "rating_diversity": {
                "unique_ratings": unique_ratings,
                "total_variants": len(successful_evals),
                "diversity_ratio": unique_ratings / len(successful_evals),
            },
            "criteria_diversity": criteria_variance,
            "overall_diversity_score": self._calculate_overall_diversity(
                score_variance, unique_ratings, len(successful_evals)
            ),
        }

    def _calculate_overall_diversity(self, score_variance: float, unique_ratings: int, total_variants: int) -> float:
        """Calculate overall diversity score."""
        # Normalize score variance (higher variance = more diversity)
        score_diversity = min(1.0, score_variance * 4)  # Scale appropriately

        # Rating diversity
        rating_diversity = unique_ratings / total_variants

        # Combined diversity
        overall_diversity = (score_diversity + rating_diversity) / 2

        return np.clip(overall_diversity, 0.0, 1.0)

    def _optimize_variant_portfolio(self, evaluations: list[dict], user_preferences: dict) -> dict[str, Any]:
        """Optimize portfolio of variants for different scenarios."""
        successful_evals = [e for e in evaluations if e.get("success", False)]

        if len(successful_evals) < 2:
            return {"insufficient_variants": True}

        # Portfolio strategies
        portfolios = {}

        # Conservative portfolio (low risk, high feasibility)
        conservative_scores = []
        for i, eval_result in enumerate(successful_evals):
            risk_score = eval_result["scores"].get("risk_score", 0.5)
            feasibility_score = eval_result["scores"].get("feasibility_score", 0.5)
            conservative_score = (1.0 - risk_score) * 0.6 + feasibility_score * 0.4
            conservative_scores.append((i, conservative_score))

        conservative_scores.sort(key=lambda x: x[1], reverse=True)
        portfolios["conservative"] = [idx for idx, _ in conservative_scores[:3]]

        # Innovation portfolio (high innovation, high impact)
        innovation_scores = []
        for i, eval_result in enumerate(successful_evals):
            innovation_score = eval_result["scores"].get("innovation_score", 0.5)
            impact_score = eval_result["scores"].get("impact", 0.5)
            combined_score = innovation_score * 0.6 + impact_score * 0.4
            innovation_scores.append((i, combined_score))

        innovation_scores.sort(key=lambda x: x[1], reverse=True)
        portfolios["innovation"] = [idx for idx, _ in innovation_scores[:3]]

        # Balanced portfolio (optimize overall score)
        overall_scores = [(i, e["scores"]["overall_score"]) for i, e in enumerate(successful_evals)]
        overall_scores.sort(key=lambda x: x[1], reverse=True)
        portfolios["balanced"] = [idx for idx, _ in overall_scores[:3]]

        # User-preference optimized portfolio
        if user_preferences:
            preference_scores = []
            for i, eval_result in enumerate(successful_evals):
                pref_alignment = eval_result["assessments"]["preference_alignment"]["overall_alignment"]
                overall_score = eval_result["scores"]["overall_score"]
                preference_score = pref_alignment * 0.7 + overall_score * 0.3
                preference_scores.append((i, preference_score))

            preference_scores.sort(key=lambda x: x[1], reverse=True)
            portfolios["user_optimized"] = [idx for idx, _ in preference_scores[:3]]

        return {
            "portfolios": portfolios,
            "recommendations": self._generate_portfolio_recommendations(portfolios, successful_evals),
        }

    def _generate_portfolio_recommendations(self, portfolios: dict, evaluations: list[dict]) -> dict[str, str]:
        """Generate recommendations for each portfolio."""
        recommendations = {}

        for portfolio_type, variant_indices in portfolios.items():
            if portfolio_type == "conservative":
                recommendations[portfolio_type] = (
                    "Recommended for risk-averse scenarios with high feasibility requirements"
                )
            elif portfolio_type == "innovation":
                recommendations[portfolio_type] = "Recommended for breakthrough innovation and high-impact scenarios"
            elif portfolio_type == "balanced":
                recommendations[portfolio_type] = "Recommended for general use with balanced risk-reward profile"
            elif portfolio_type == "user_optimized":
                recommendations[portfolio_type] = "Recommended based on your specific preferences and requirements"

        return recommendations

    def _analyze_tradeoffs(self, evaluations: list[dict]) -> dict[str, Any]:
        """Analyze tradeoffs between different criteria."""
        successful_evals = [e for e in evaluations if e.get("success", False)]

        if len(successful_evals) < 2:
            return {"insufficient_variants": True}

        tradeoffs = {}

        # Common tradeoff pairs
        tradeoff_pairs = [
            ("innovation_score", "risk_score"),
            ("impact", "feasibility_score"),
            ("innovation_score", "feasibility_score"),
            ("overall_score", "risk_score"),
        ]

        for metric1, metric2 in tradeoff_pairs:
            values1 = []
            values2 = []

            for eval_result in successful_evals:
                val1 = eval_result["scores"].get(metric1, 0.5)
                val2 = eval_result["scores"].get(metric2, 0.5)
                values1.append(val1)
                values2.append(val2)

            if len(values1) > 1 and len(values2) > 1:
                correlation = np.corrcoef(values1, values2)[0, 1]

                tradeoffs[f"{metric1}_vs_{metric2}"] = {
                    "correlation": correlation,
                    "tradeoff_strength": abs(correlation),
                    "tradeoff_type": "positive" if correlation > 0 else "negative",
                    "analysis": self._interpret_tradeoff(metric1, metric2, correlation),
                }

        return tradeoffs

    def _interpret_tradeoff(self, metric1: str, metric2: str, correlation: float) -> str:
        """Interpret the tradeoff relationship."""
        if abs(correlation) < 0.3:
            return f"Weak relationship between {metric1} and {metric2}"
        elif correlation > 0.5:
            return f"Strong positive correlation: Higher {metric1} tends to increase {metric2}"
        elif correlation < -0.5:
            return f"Strong negative correlation: Higher {metric1} tends to decrease {metric2}"
        else:
            return f"Moderate relationship between {metric1} and {metric2}"

    def _perform_sensitivity_analysis(self, evaluations: list[dict]) -> dict[str, Any]:
        """Perform sensitivity analysis on evaluation criteria."""
        successful_evals = [e for e in evaluations if e.get("success", False)]

        if len(successful_evals) < 3:
            return {"insufficient_variants": True}

        # Analyze how changes in individual criteria affect overall score
        sensitivity = {}

        # Collect data
        overall_scores = [e["scores"]["overall_score"] for e in successful_evals]
        criteria_scores = {}

        for eval_result in successful_evals:
            core_scores = eval_result["scores"]["core_scores"]
            for criterion, score in core_scores.items():
                if criterion not in criteria_scores:
                    criteria_scores[criterion] = []
                criteria_scores[criterion].append(score)

        # Calculate sensitivity for each criterion
        for criterion, scores in criteria_scores.items():
            if len(scores) > 2:
                # Calculate correlation with overall score
                correlation = (
                    np.corrcoef(scores, overall_scores)[0, 1]
                    if not np.isnan(np.corrcoef(scores, overall_scores)[0, 1])
                    else 0
                )

                # Calculate coefficient of variation
                cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0

                sensitivity[criterion] = {
                    "correlation_with_overall": correlation,
                    "coefficient_of_variation": cv,
                    "sensitivity_score": abs(correlation) * cv,
                    "impact_level": self._classify_sensitivity_impact(abs(correlation) * cv),
                }

        # Rank criteria by sensitivity
        ranked_sensitivity = sorted(sensitivity.items(), key=lambda x: x[1]["sensitivity_score"], reverse=True)

        return {
            "criteria_sensitivity": sensitivity,
            "most_sensitive_criteria": [item[0] for item in ranked_sensitivity[:3]],
            "least_sensitive_criteria": [item[0] for item in ranked_sensitivity[-3:]],
            "sensitivity_insights": self._generate_sensitivity_insights(ranked_sensitivity),
        }

    def _classify_sensitivity_impact(self, sensitivity_score: float) -> str:
        """Classify sensitivity impact level."""
        if sensitivity_score > 0.7:
            return "high"
        elif sensitivity_score > 0.4:
            return "medium"
        else:
            return "low"

    def _generate_sensitivity_insights(self, ranked_sensitivity: list[tuple]) -> list[str]:
        """Generate insights from sensitivity analysis."""
        insights = []

        if ranked_sensitivity:
            most_sensitive = ranked_sensitivity[0]
            insights.append(f"{most_sensitive[0]} has the highest impact on overall evaluation")

            if len(ranked_sensitivity) > 1:
                least_sensitive = ranked_sensitivity[-1]
                insights.append(f"{least_sensitive[0]} has the least impact on overall evaluation")

            high_impact_criteria = [item[0] for item in ranked_sensitivity if item[1]["sensitivity_score"] > 0.5]
            if high_impact_criteria:
                insights.append(f"Focus optimization efforts on: {', '.join(high_impact_criteria)}")

        return insights

    # Statistics and analysis helpers

    def _analyze_quality_distribution(self, evaluations: list[dict]) -> dict[str, int]:
        """Analyze distribution of quality ratings."""
        successful_evals = [e for e in evaluations if e.get("success", False)]

        distribution = {"excellent": 0, "good": 0, "acceptable": 0, "poor": 0, "rejected": 0}

        for eval_result in successful_evals:
            rating = eval_result["scores"]["overall_rating"]
            if rating in distribution:
                distribution[rating] += 1

        return distribution

    def _calculate_score_statistics(self, evaluations: list[dict]) -> dict[str, float]:
        """Calculate comprehensive score statistics."""
        successful_evals = [e for e in evaluations if e.get("success", False)]

        if not successful_evals:
            return {}

        scores = [e["scores"]["overall_score"] for e in successful_evals]

        return {
            "mean": np.mean(scores),
            "median": np.median(scores),
            "std": np.std(scores),
            "min": np.min(scores),
            "max": np.max(scores),
            "range": np.max(scores) - np.min(scores),
            "percentile_25": np.percentile(scores, 25),
            "percentile_75": np.percentile(scores, 75),
            "iqr": np.percentile(scores, 75) - np.percentile(scores, 25),
        }

    def _analyze_criteria_importance(self, evaluations: list[dict]) -> dict[str, float]:
        """Analyze relative importance of criteria based on performance variation."""
        successful_evals = [e for e in evaluations if e.get("success", False)]

        if not successful_evals:
            return {}

        criteria_importance = {}
        overall_scores = [e["scores"]["overall_score"] for e in successful_evals]

        # Collect criteria scores
        criteria_data = {}
        for eval_result in successful_evals:
            core_scores = eval_result["scores"]["core_scores"]
            for criterion, score in core_scores.items():
                if criterion not in criteria_data:
                    criteria_data[criterion] = []
                criteria_data[criterion].append(score)

        # Calculate importance based on correlation with overall score
        for criterion, scores in criteria_data.items():
            if len(scores) > 1:
                correlation = abs(np.corrcoef(scores, overall_scores)[0, 1])
                if not np.isnan(correlation):
                    criteria_importance[criterion] = correlation

        return criteria_importance

    # Utility and maintenance methods

    def _update_evaluation_stats(self, rating: str, criteria: list[EvaluationCriteria]):
        """Update evaluation statistics."""
        self.evaluation_stats["total_evaluations"] += 1

        if rating in self.evaluation_stats["quality_distribution"]:
            self.evaluation_stats["quality_distribution"][rating] += 1

        # Update criteria importance based on usage
        for criterion in criteria:
            if criterion.name not in self.evaluation_stats["criteria_importance"]:
                self.evaluation_stats["criteria_importance"][criterion.name] = 0
            self.evaluation_stats["criteria_importance"][criterion.name] += criterion.weight

    async def _calculate_evaluation_confidence(
        self, core_scores: dict, contextual_scores: dict, variant: dict
    ) -> float:
        """Calculate confidence in the evaluation."""
        confidence_factors = []

        # Score consistency (lower variance = higher confidence)
        all_scores = list(core_scores.values()) + list(contextual_scores.values())
        if len(all_scores) > 1:
            score_variance = np.var(all_scores)
            consistency_confidence = 1.0 / (1.0 + score_variance * 2)
            confidence_factors.append(consistency_confidence)

        # Variant completeness
        variant_completeness = len([v for v in variant.values() if v is not None]) / max(len(variant), 1)
        confidence_factors.append(variant_completeness)

        # Number of criteria evaluated
        criteria_coverage = len(core_scores) / len(self.default_criteria)
        confidence_factors.append(criteria_coverage)

        # Base confidence from variant
        base_confidence = variant.get("confidence", 0.5)
        confidence_factors.append(base_confidence)

        return np.clip(np.mean(confidence_factors), 0.1, 0.95)

    async def _generate_fallback_evaluation(self, variant: dict, user_preferences: dict) -> dict[str, Any]:
        """Generate fallback evaluation when main evaluation fails."""
        return {
            "fallback": True,
            "overall_score": variant.get("confidence", 0.5),
            "overall_rating": "uncertain",
            "limited_assessment": {
                "available_metrics": list(variant.keys()),
                "confidence_level": "low",
                "recommendation": "gather_more_data",
            },
            "next_steps": [
                "Provide more complete variant information",
                "Re-run evaluation with better data",
                "Consider manual review",
            ],
        }

    async def get_evaluation_statistics(self) -> dict[str, Any]:
        """Get comprehensive evaluation statistics."""
        return {
            "total_evaluations": self.evaluation_stats["total_evaluations"],
            "quality_distribution": self.evaluation_stats["quality_distribution"],
            "criteria_importance": self.evaluation_stats["criteria_importance"],
            "adaptation_events": self.evaluation_stats["adaptation_events"],
            "cache_status": {
                "evaluation_cache_size": len(self.evaluation_cache),
                "criteria_history_size": len(self.criteria_performance_history),
                "feedback_history_size": len(self.evaluation_feedback),
            },
            "performance_metrics": {
                "avg_evaluations_per_batch": self.evaluation_stats["total_evaluations"]
                / max(len(self.evaluation_cache), 1),
                "adaptation_rate": self.evaluation_stats["adaptation_events"]
                / max(self.evaluation_stats["total_evaluations"], 1),
            },
        }
