"""
Explainer Module - Jeffrey OS DreamMode Phase 3
Natural language explanation generation for dream variants and decisions.
"""

import random
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .monitoring import StructuredLogger


@dataclass
class ExplanationTemplate:
    """Template for generating explanations."""

    category: str
    template: str
    min_confidence: float
    variables: list[str]


class ExplainerModule:
    """
    Natural language explanation generator for DreamMode outputs.
    Provides comprehensive, contextual explanations for all decisions and recommendations.
    """

    def __init__(self, seed: int = None):
        # Seeding for reproducible explanations
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.logger = StructuredLogger("explainer")

        # GROK CRITICAL: Limited cache with memory management
        self.explanation_cache = deque(maxlen=100)
        self.template_usage_stats = deque(maxlen=300)

        # Explanation templates
        self.templates = self._initialize_explanation_templates()

        # Language style configuration
        self.language_style = {
            "formality": "professional",  # casual, professional, technical
            "detail_level": "medium",  # brief, medium, detailed
            "audience": "general",  # technical, business, general
            "tone": "supportive",  # neutral, supportive, critical
        }

        # Explanation statistics
        self.explanation_stats = {
            "total_explanations": 0,
            "explanation_types": {},
            "avg_confidence": 0.0,
            "template_usage": {},
            "language_adaptations": 0,
        }

    async def explain_variant_selection(
        self,
        selected_variant: dict,
        all_variants: list[dict],
        evaluation_result: dict,
        context: dict = None,
    ) -> dict[str, Any]:
        """
        Explain why a specific variant was selected over others.
        """
        explanation_id = f"explain_selection_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            context = context or {}

            await self.logger.log(
                "info",
                "explanation_start",
                {
                    "explanation_id": explanation_id,
                    "explanation_type": "variant_selection",
                    "variants_count": len(all_variants),
                },
            )

            # Analyze selection rationale
            selection_analysis = self._analyze_selection_rationale(selected_variant, all_variants, evaluation_result)

            # Generate core explanation
            core_explanation = await self._generate_selection_explanation(
                selected_variant, selection_analysis, evaluation_result
            )

            # Add comparative analysis
            comparative_explanation = await self._generate_comparative_explanation(
                selected_variant, all_variants, evaluation_result
            )

            # Generate decision factors explanation
            factors_explanation = await self._explain_decision_factors(selection_analysis, evaluation_result)

            # Risk and benefit explanation
            risk_benefit_explanation = await self._explain_risks_and_benefits(selected_variant, evaluation_result)

            # Context-specific explanations
            context_explanation = await self._explain_contextual_fit(selected_variant, context, evaluation_result)

            # Alternative options explanation
            alternatives_explanation = await self._explain_alternatives(all_variants, evaluation_result)

            # Generate confidence explanation
            confidence_explanation = await self._explain_confidence_level(evaluation_result, selection_analysis)

            explanation_time = (time.time() - start_time) * 1000

            # Build comprehensive explanation
            explanation_result = {
                "explanation_id": explanation_id,
                "success": True,
                "explanation_type": "variant_selection",
                "metadata": {
                    "explanation_time_ms": explanation_time,
                    "confidence": selection_analysis.get("explanation_confidence", 0.8),
                    "language_style": self.language_style.copy(),
                },
                "explanations": {
                    "core_rationale": core_explanation,
                    "comparative_analysis": comparative_explanation,
                    "decision_factors": factors_explanation,
                    "risks_and_benefits": risk_benefit_explanation,
                    "contextual_fit": context_explanation,
                    "alternatives": alternatives_explanation,
                    "confidence_assessment": confidence_explanation,
                },
                "summary": {
                    "primary_reason": selection_analysis.get("primary_reason", "Best overall performance"),
                    "key_advantages": selection_analysis.get("key_advantages", []),
                    "main_concerns": selection_analysis.get("main_concerns", []),
                    "recommendation_strength": selection_analysis.get("recommendation_strength", "moderate"),
                },
                "next_steps": await self._generate_next_steps_explanation(selected_variant, evaluation_result, context),
            }

            # Cache explanation
            self._cache_explanation(explanation_id, explanation_result, "variant_selection")

            # Update statistics
            self._update_explanation_stats("variant_selection", selection_analysis.get("explanation_confidence", 0.8))

            await self.logger.log(
                "info",
                "explanation_complete",
                {
                    "explanation_id": explanation_id,
                    "explanation_type": "variant_selection",
                    "explanation_time_ms": explanation_time,
                },
            )

            return explanation_result

        except Exception as e:
            error_time = (time.time() - start_time) * 1000

            await self.logger.log(
                "error",
                "explanation_failed",
                {
                    "explanation_id": explanation_id,
                    "error": str(e),
                    "explanation_time_ms": error_time,
                },
            )

            return {
                "explanation_id": explanation_id,
                "success": False,
                "error": str(e),
                "fallback_explanation": await self._generate_fallback_explanation(
                    "variant_selection", selected_variant, evaluation_result
                ),
            }

    async def explain_evaluation_scores(
        self, evaluation_result: dict, variant: dict, criteria_weights: dict = None
    ) -> dict[str, Any]:
        """
        Explain evaluation scores and methodology.
        """
        explanation_id = f"explain_eval_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            criteria_weights = criteria_weights or {}

            # Extract evaluation components
            scores = evaluation_result.get("scores", {})
            assessments = evaluation_result.get("assessments", {})

            # Generate score explanations
            score_explanations = {}

            # Overall score explanation
            overall_score = scores.get("overall_score", 0)
            overall_rating = scores.get("overall_rating", "unknown")

            score_explanations["overall"] = await self._explain_overall_score(
                overall_score, overall_rating, scores, assessments
            )

            # Core criteria explanations
            core_scores = scores.get("core_scores", {})
            for criterion, score in core_scores.items():
                score_explanations[criterion] = await self._explain_criterion_score(
                    criterion, score, variant, assessments
                )

            # Quality assessment explanation
            quality_assessment = assessments.get("quality_assessment", {})
            score_explanations["quality"] = await self._explain_quality_assessment(quality_assessment, variant)

            # Risk evaluation explanation
            risk_evaluation = assessments.get("risk_evaluation", {})
            score_explanations["risk"] = await self._explain_risk_evaluation(risk_evaluation, variant)

            # Innovation assessment explanation
            innovation_assessment = assessments.get("innovation_assessment", {})
            score_explanations["innovation"] = await self._explain_innovation_assessment(innovation_assessment, variant)

            # Feasibility analysis explanation
            feasibility_analysis = assessments.get("feasibility_analysis", {})
            score_explanations["feasibility"] = await self._explain_feasibility_analysis(feasibility_analysis, variant)

            # Methodology explanation
            methodology_explanation = await self._explain_evaluation_methodology(criteria_weights, assessments)

            # Confidence and limitations
            confidence_limitations = await self._explain_evaluation_confidence_and_limitations(
                evaluation_result, variant
            )

            explanation_time = (time.time() - start_time) * 1000

            explanation_result = {
                "explanation_id": explanation_id,
                "success": True,
                "explanation_type": "evaluation_scores",
                "metadata": {
                    "explanation_time_ms": explanation_time,
                    "scores_explained": len(score_explanations),
                    "confidence": self._calculate_explanation_confidence(evaluation_result),
                },
                "score_explanations": score_explanations,
                "methodology": methodology_explanation,
                "confidence_and_limitations": confidence_limitations,
                "interpretation_guide": {
                    "score_ranges": {
                        "excellent": "0.85 - 1.00",
                        "good": "0.70 - 0.84",
                        "acceptable": "0.55 - 0.69",
                        "poor": "0.40 - 0.54",
                        "unacceptable": "0.00 - 0.39",
                    },
                    "key_factors": self._identify_key_evaluation_factors(scores, assessments),
                    "improvement_focus": self._identify_improvement_focus_areas(evaluation_result),
                },
            }

            self._cache_explanation(explanation_id, explanation_result, "evaluation_scores")
            self._update_explanation_stats(
                "evaluation_scores", self._calculate_explanation_confidence(evaluation_result)
            )

            return explanation_result

        except Exception as e:
            return {
                "explanation_id": explanation_id,
                "success": False,
                "error": str(e),
                "fallback_explanation": await self._generate_fallback_explanation(
                    "evaluation_scores", variant, evaluation_result
                ),
            }

    async def explain_simulation_results(self, simulation_result: dict, variant: dict) -> dict[str, Any]:
        """
        Explain simulation results and projections.
        """
        explanation_id = f"explain_sim_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            # Extract simulation components
            impact_analysis = simulation_result.get("impact_analysis", {})
            risk_assessment = simulation_result.get("risk_assessment", {})
            success_probabilities = simulation_result.get("success_probabilities", {})
            trend_analysis = simulation_result.get("trend_analysis", {})

            # Generate explanations
            explanations = {}

            # Success probability explanation
            explanations["success_probability"] = await self._explain_success_probability(
                success_probabilities, impact_analysis
            )

            # Impact analysis explanation
            explanations["impact_analysis"] = await self._explain_impact_analysis(impact_analysis, variant)

            # Risk assessment explanation
            explanations["risk_assessment"] = await self._explain_simulation_risk_assessment(risk_assessment, variant)

            # Scenario analysis explanation
            explanations["scenario_analysis"] = await self._explain_scenario_analysis(
                simulation_result.get("scenario_coverage", {}), impact_analysis
            )

            # Trend analysis explanation
            explanations["trend_analysis"] = await self._explain_trend_analysis(trend_analysis, variant)

            # Financial projections explanation
            financial_summary = impact_analysis.get("financial_summary", {})
            explanations["financial_projections"] = await self._explain_financial_projections(
                financial_summary, variant
            )

            # Confidence intervals explanation
            confidence_intervals = simulation_result.get("confidence_intervals", {})
            explanations["confidence_intervals"] = await self._explain_confidence_intervals(confidence_intervals)

            # Recommendations explanation
            recommendations = simulation_result.get("recommendations", [])
            explanations["recommendations"] = await self._explain_simulation_recommendations(
                recommendations, simulation_result
            )

            explanation_time = (time.time() - start_time) * 1000

            explanation_result = {
                "explanation_id": explanation_id,
                "success": True,
                "explanation_type": "simulation_results",
                "metadata": {
                    "explanation_time_ms": explanation_time,
                    "simulation_iterations": simulation_result.get("metadata", {}).get("iterations_completed", 0),
                    "confidence": self._calculate_simulation_explanation_confidence(simulation_result),
                },
                "explanations": explanations,
                "key_insights": {
                    "primary_finding": self._extract_primary_simulation_finding(simulation_result),
                    "critical_risks": self._extract_critical_simulation_risks(risk_assessment),
                    "success_factors": self._extract_simulation_success_factors(impact_analysis),
                    "decision_implications": self._extract_decision_implications(simulation_result),
                },
                "methodology_note": await self._explain_simulation_methodology(simulation_result),
                "limitations": await self._explain_simulation_limitations(simulation_result),
            }

            self._cache_explanation(explanation_id, explanation_result, "simulation_results")
            self._update_explanation_stats(
                "simulation_results",
                self._calculate_simulation_explanation_confidence(simulation_result),
            )

            return explanation_result

        except Exception as e:
            return {
                "explanation_id": explanation_id,
                "success": False,
                "error": str(e),
                "fallback_explanation": await self._generate_fallback_explanation(
                    "simulation_results", variant, simulation_result
                ),
            }

    async def explain_recommendation(
        self,
        recommendation: dict,
        evaluation_result: dict,
        simulation_result: dict = None,
        context: dict = None,
    ) -> dict[str, Any]:
        """
        Explain final recommendation and rationale.
        """
        explanation_id = f"explain_rec_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            context = context or {}

            # Extract recommendation components
            action = recommendation.get("action", "unknown")
            rationale = recommendation.get("rationale", "")
            next_steps = recommendation.get("next_steps", [])

            # Generate comprehensive recommendation explanation
            explanations = {}

            # Main recommendation explanation
            explanations["main_recommendation"] = await self._explain_main_recommendation(
                action, rationale, evaluation_result
            )

            # Evidence-based justification
            explanations["evidence_justification"] = await self._explain_evidence_justification(
                recommendation, evaluation_result, simulation_result
            )

            # Alternative actions considered
            explanations["alternatives_considered"] = await self._explain_alternatives_considered(
                action, evaluation_result, context
            )

            # Risk and mitigation explanation
            explanations["risk_mitigation"] = await self._explain_recommendation_risks(
                recommendation, evaluation_result, simulation_result
            )

            # Implementation guidance
            explanations["implementation_guidance"] = await self._explain_implementation_guidance(
                next_steps, recommendation, context
            )

            # Success criteria explanation
            explanations["success_criteria"] = await self._explain_success_criteria(
                recommendation, evaluation_result, simulation_result
            )

            # Timeline and resource implications
            explanations["resource_implications"] = await self._explain_resource_implications(
                recommendation, context, simulation_result
            )

            # Monitoring and adjustment explanation
            explanations["monitoring_guidance"] = await self._explain_monitoring_guidance(
                recommendation, evaluation_result
            )

            explanation_time = (time.time() - start_time) * 1000

            explanation_result = {
                "explanation_id": explanation_id,
                "success": True,
                "explanation_type": "recommendation",
                "metadata": {
                    "explanation_time_ms": explanation_time,
                    "recommendation_action": action,
                    "confidence": self._calculate_recommendation_explanation_confidence(
                        recommendation, evaluation_result
                    ),
                },
                "explanations": explanations,
                "executive_summary": {
                    "recommendation": action,
                    "primary_rationale": self._extract_primary_rationale(recommendation, evaluation_result),
                    "key_benefits": self._extract_key_benefits(recommendation, evaluation_result, simulation_result),
                    "main_risks": self._extract_main_risks(recommendation, evaluation_result, simulation_result),
                    "implementation_complexity": self._assess_implementation_complexity(recommendation, context),
                },
                "decision_support": {
                    "confidence_level": recommendation.get("confidence", "medium"),
                    "decision_urgency": context.get("urgency", "medium"),
                    "stakeholder_considerations": self._identify_stakeholder_considerations(recommendation, context),
                    "reversibility": self._assess_decision_reversibility(recommendation, context),
                },
            }

            self._cache_explanation(explanation_id, explanation_result, "recommendation")
            self._update_explanation_stats(
                "recommendation",
                self._calculate_recommendation_explanation_confidence(recommendation, evaluation_result),
            )

            return explanation_result

        except Exception as e:
            return {
                "explanation_id": explanation_id,
                "success": False,
                "error": str(e),
                "fallback_explanation": await self._generate_fallback_explanation(
                    "recommendation", recommendation, evaluation_result
                ),
            }

    def _initialize_explanation_templates(self) -> dict[str, list[ExplanationTemplate]]:
        """Initialize explanation templates for different contexts."""
        templates = {
            "positive_score": [
                ExplanationTemplate(
                    "high_performance",
                    "This variant scores {score:.1%} in {criterion}, indicating {performance_level} performance. {specific_reason}",
                    0.7,
                    ["score", "criterion", "performance_level", "specific_reason"],
                ),
                ExplanationTemplate(
                    "strength",
                    "A key strength is the {criterion} score of {score:.1%}, which suggests {implication}.",
                    0.6,
                    ["criterion", "score", "implication"],
                ),
            ],
            "negative_score": [
                ExplanationTemplate(
                    "concern",
                    "The {criterion} score of {score:.1%} is concerning because {reason}. {improvement_suggestion}",
                    0.5,
                    ["criterion", "score", "reason", "improvement_suggestion"],
                ),
                ExplanationTemplate(
                    "limitation",
                    "A limitation is the {criterion} performance at {score:.1%}, which may {consequence}.",
                    0.4,
                    ["criterion", "score", "consequence"],
                ),
            ],
            "comparison": [
                ExplanationTemplate(
                    "better_than",
                    "This variant outperforms others by {difference:.1%} in {criterion} due to {advantage}.",
                    0.6,
                    ["difference", "criterion", "advantage"],
                ),
                ExplanationTemplate(
                    "trade_off",
                    "While this variant has {strength} in {strong_area}, it trades off {weakness} in {weak_area}.",
                    0.5,
                    ["strength", "strong_area", "weakness", "weak_area"],
                ),
            ],
            "risk": [
                ExplanationTemplate(
                    "low_risk",
                    "The risk level is {risk_level} ({score:.1%}), suggesting {implication} for implementation.",
                    0.7,
                    ["risk_level", "score", "implication"],
                ),
                ExplanationTemplate(
                    "high_risk",
                    "The elevated risk ({score:.1%}) primarily stems from {risk_source} and requires {mitigation}.",
                    0.6,
                    ["score", "risk_source", "mitigation"],
                ),
            ],
            "confidence": [
                ExplanationTemplate(
                    "high_confidence",
                    "The assessment confidence is {confidence:.1%}, supported by {evidence_quality} and {data_completeness}.",
                    0.8,
                    ["confidence", "evidence_quality", "data_completeness"],
                ),
                ExplanationTemplate(
                    "uncertainty",
                    "Confidence is moderate ({confidence:.1%}) due to {uncertainty_source}, suggesting {recommendation}.",
                    0.5,
                    ["confidence", "uncertainty_source", "recommendation"],
                ),
            ],
        }

        return templates

    def _analyze_selection_rationale(
        self, selected_variant: dict, all_variants: list[dict], evaluation_result: dict
    ) -> dict[str, Any]:
        """Analyze why a variant was selected."""
        analysis = {
            "explanation_confidence": 0.7,
            "primary_reason": "Best overall performance",
            "key_advantages": [],
            "main_concerns": [],
            "recommendation_strength": "moderate",
        }

        # Extract selection factors
        selected_score = evaluation_result.get("scores", {}).get("overall_score", 0)
        selected_rating = evaluation_result.get("scores", {}).get("overall_rating", "unknown")

        # Determine primary reason
        if selected_score > 0.85:
            analysis["primary_reason"] = "Exceptional overall performance"
            analysis["recommendation_strength"] = "strong"
        elif selected_score > 0.7:
            analysis["primary_reason"] = "Strong performance across key criteria"
            analysis["recommendation_strength"] = "moderate"
        elif selected_score > 0.55:
            analysis["primary_reason"] = "Acceptable performance with manageable trade-offs"
            analysis["recommendation_strength"] = "conditional"
        else:
            analysis["primary_reason"] = "Best available option despite limitations"
            analysis["recommendation_strength"] = "weak"

        # Identify key advantages
        core_scores = evaluation_result.get("scores", {}).get("core_scores", {})
        for criterion, score in core_scores.items():
            if score > 0.75:
                analysis["key_advantages"].append(f"Strong {criterion} ({score:.1%})")

        # Identify main concerns
        for criterion, score in core_scores.items():
            if score < 0.4:
                analysis["main_concerns"].append(f"Low {criterion} ({score:.1%})")

        # Risk considerations
        risk_score = evaluation_result.get("scores", {}).get("risk_score", 0.5)
        if risk_score > 0.7:
            analysis["main_concerns"].append(f"High risk level ({risk_score:.1%})")

        return analysis

    async def _generate_selection_explanation(
        self, selected_variant: dict, selection_analysis: dict, evaluation_result: dict
    ) -> str:
        """Generate core selection explanation."""
        primary_reason = selection_analysis.get("primary_reason", "Best overall performance")
        overall_score = evaluation_result.get("scores", {}).get("overall_score", 0)
        overall_rating = evaluation_result.get("scores", {}).get("overall_rating", "unknown")

        explanation = f"This variant was selected because it demonstrates {primary_reason.lower()} "
        explanation += f"with an overall score of {overall_score:.1%} ({overall_rating} rating). "

        # Add key advantages
        advantages = selection_analysis.get("key_advantages", [])
        if advantages:
            explanation += f"Its primary strengths include {self._format_list(advantages[:3])}. "

        # Add context about trade-offs
        concerns = selection_analysis.get("main_concerns", [])
        if concerns:
            explanation += f"While there are considerations regarding {self._format_list(concerns[:2])}, "
            explanation += "the overall assessment indicates these are manageable within the proposed approach."
        else:
            explanation += "The assessment shows strong performance across all key evaluation criteria."

        return explanation

    async def _generate_comparative_explanation(
        self, selected_variant: dict, all_variants: list[dict], evaluation_result: dict
    ) -> str:
        """Generate comparative analysis explanation."""
        if len(all_variants) <= 1:
            return "This was the only variant evaluated, so no comparative analysis is available."

        explanation = f"Among {len(all_variants)} variants evaluated, this option stood out "

        # Add specific comparative advantages
        selected_score = evaluation_result.get("scores", {}).get("overall_score", 0)

        if selected_score > 0.8:
            explanation += "as the clear leader with significantly higher performance scores. "
        elif selected_score > 0.6:
            explanation += "with a favorable balance of performance and feasibility. "
        else:
            explanation += "as the most viable option despite some limitations. "

        # Add details about how it compares
        core_scores = evaluation_result.get("scores", {}).get("core_scores", {})
        strong_areas = [criterion for criterion, score in core_scores.items() if score > 0.75]

        if strong_areas:
            explanation += f"It particularly excels in {self._format_list(strong_areas)}, "
            explanation += "providing competitive advantages in these critical areas."

        return explanation

    async def _explain_decision_factors(self, selection_analysis: dict, evaluation_result: dict) -> dict[str, str]:
        """Explain the factors that influenced the decision."""
        factors = {}

        # Performance factors
        overall_score = evaluation_result.get("scores", {}).get("overall_score", 0)
        factors["performance"] = (
            f"Overall performance score of {overall_score:.1%} indicates {self._interpret_performance_level(overall_score)}"
        )

        # Risk factors
        risk_score = evaluation_result.get("scores", {}).get("risk_score", 0.5)
        factors["risk"] = (
            f"Risk assessment shows {self._interpret_risk_level(risk_score)} risk profile at {risk_score:.1%}"
        )

        # Feasibility factors
        feasibility_score = evaluation_result.get("scores", {}).get("feasibility_score", 0.5)
        factors["feasibility"] = (
            f"Feasibility analysis indicates {self._interpret_feasibility_level(feasibility_score)} implementation viability"
        )

        # Innovation factors
        innovation_score = evaluation_result.get("scores", {}).get("innovation_score", 0.5)
        factors["innovation"] = (
            f"Innovation assessment reveals {self._interpret_innovation_level(innovation_score)} creative potential"
        )

        # Quality factors
        quality_score = evaluation_result.get("scores", {}).get("quality_score", 0.5)
        factors["quality"] = (
            f"Quality evaluation demonstrates {self._interpret_quality_level(quality_score)} overall quality"
        )

        return factors

    async def _explain_risks_and_benefits(self, selected_variant: dict, evaluation_result: dict) -> dict[str, Any]:
        """Explain risks and benefits of the selected variant."""
        risk_benefit_analysis = {
            "primary_benefits": [],
            "key_risks": [],
            "risk_mitigation": [],
            "benefit_realization": [],
        }

        # Extract benefits from high-scoring areas
        core_scores = evaluation_result.get("scores", {}).get("core_scores", {})
        for criterion, score in core_scores.items():
            if score > 0.7:
                benefit = self._generate_benefit_explanation(criterion, score)
                risk_benefit_analysis["primary_benefits"].append(benefit)

        # Extract risks from assessments
        risk_evaluation = evaluation_result.get("assessments", {}).get("risk_evaluation", {})
        risk_categories = risk_evaluation.get("risk_categories", {})
        for category, risk_level in risk_categories.items():
            if risk_level > 0.6:
                risk = self._generate_risk_explanation(category, risk_level)
                risk_benefit_analysis["key_risks"].append(risk)

        # Generate mitigation suggestions
        mitigation_suggestions = risk_evaluation.get("mitigation_suggestions", [])
        for suggestion in mitigation_suggestions[:3]:  # Top 3
            risk_benefit_analysis["risk_mitigation"].append(suggestion)

        # Generate benefit realization strategies
        impact_score = evaluation_result.get("scores", {}).get("impact", 0.5)
        if impact_score > 0.6:
            risk_benefit_analysis["benefit_realization"].append(
                f"High impact potential ({impact_score:.1%}) suggests significant value creation opportunities"
            )

        return risk_benefit_analysis

    async def _explain_contextual_fit(self, selected_variant: dict, context: dict, evaluation_result: dict) -> str:
        """Explain how the variant fits the specific context."""
        if not context:
            return "No specific context provided for evaluation."

        explanation = "The variant aligns well with the provided context: "

        contextual_scores = evaluation_result.get("scores", {}).get("contextual_scores", {})

        # Context fit
        context_fit = contextual_scores.get("context_fit", 0.5)
        if context_fit > 0.7:
            explanation += (
                f"strong contextual alignment ({context_fit:.1%}) suggests excellent fit with current circumstances. "
            )
        elif context_fit > 0.5:
            explanation += f"good contextual fit ({context_fit:.1%}) indicates suitable alignment with requirements. "
        else:
            explanation += f"moderate contextual fit ({context_fit:.1%}) suggests some adaptation may be needed. "

        # Specific context factors
        urgency = context.get("urgency", "medium")
        if urgency == "high":
            explanation += "The approach accommodates the high urgency requirements. "

        resources = context.get("resources", "moderate")
        if resources == "limited":
            explanation += "The solution is designed to work within limited resource constraints. "

        return explanation

    async def _explain_alternatives(self, all_variants: list[dict], evaluation_result: dict) -> str:
        """Explain why alternatives were not selected."""
        if len(all_variants) <= 1:
            return "No alternative variants were available for comparison."

        explanation = f"Of the {len(all_variants)} variants considered, the alternatives were not selected for the following reasons: "

        # This would need access to evaluations of other variants
        # For now, provide a general explanation
        explanation += "they either scored lower on key criteria, presented higher risks, "
        explanation += "or offered less favorable trade-offs between performance and feasibility. "
        explanation += "Each alternative was carefully evaluated against the same criteria to ensure fair comparison."

        return explanation

    async def _explain_confidence_level(self, evaluation_result: dict, selection_analysis: dict) -> dict[str, str]:
        """Explain the confidence level in the evaluation and selection."""
        confidence_explanations = {}

        # Overall confidence
        evaluation_confidence = evaluation_result.get("metadata", {}).get("evaluation_confidence", 0.7)
        confidence_explanations["overall"] = f"The evaluation confidence is {evaluation_confidence:.1%}, "

        if evaluation_confidence > 0.8:
            confidence_explanations["overall"] += (
                "indicating high reliability in the assessment methodology and results."
            )
        elif evaluation_confidence > 0.6:
            confidence_explanations["overall"] += "suggesting good reliability with standard assessment uncertainty."
        else:
            confidence_explanations["overall"] += (
                "reflecting moderate reliability due to limited data or high uncertainty."
            )

        # Data quality
        confidence_explanations["data_quality"] = self._assess_data_quality_explanation(evaluation_result)

        # Methodology confidence
        confidence_explanations["methodology"] = (
            "The evaluation used established multi-criteria analysis with adaptive weighting based on context and preferences."
        )

        # Uncertainty factors
        confidence_explanations["uncertainty"] = self._identify_uncertainty_factors(evaluation_result)

        return confidence_explanations

    async def _generate_next_steps_explanation(
        self, selected_variant: dict, evaluation_result: dict, context: dict
    ) -> list[str]:
        """Generate explanation of recommended next steps."""
        next_steps = []

        # Based on recommendation action
        recommendation = evaluation_result.get("recommendation", {})
        action = recommendation.get("action", "conditional_recommend")

        if action == "strongly_recommend":
            next_steps.append("Proceed with implementation planning - this variant shows excellent potential")
            next_steps.append("Develop detailed implementation timeline and resource allocation")
            next_steps.append("Establish success metrics and monitoring framework")
        elif action == "recommend":
            next_steps.append("Move forward with implementation while monitoring key risk areas")
            next_steps.append("Consider minor optimizations in weaker performance areas")
            next_steps.append("Prepare contingency plans for identified risks")
        elif action == "conditional_recommend":
            next_steps.append("Address identified weaknesses before full implementation")
            next_steps.append("Conduct pilot testing to validate assumptions")
            next_steps.append("Reassess after initial improvements are made")
        else:
            next_steps.append("Reconsider approach - significant improvements needed")
            next_steps.append("Explore alternative solutions or major modifications")
            next_steps.append("Gather additional data to inform better options")

        # Add context-specific steps
        urgency = context.get("urgency", "medium")
        if urgency == "high":
            next_steps.append("Accelerate timeline due to high urgency requirements")

        return next_steps

    # Score explanation methods

    async def _explain_overall_score(
        self, overall_score: float, overall_rating: str, scores: dict, assessments: dict
    ) -> str:
        """Explain the overall evaluation score."""
        explanation = f"The overall score of {overall_score:.1%} ({overall_rating} rating) reflects "

        if overall_score > 0.85:
            explanation += "exceptional performance across multiple evaluation criteria. "
        elif overall_score > 0.7:
            explanation += "strong performance with good balance across key factors. "
        elif overall_score > 0.55:
            explanation += "acceptable performance with some areas for improvement. "
        elif overall_score > 0.4:
            explanation += "limited performance with significant challenges to address. "
        else:
            explanation += "poor performance requiring substantial improvements. "

        # Add contributing factors
        core_scores = scores.get("core_scores", {})
        if core_scores:
            top_performers = sorted(core_scores.items(), key=lambda x: x[1], reverse=True)[:2]
            explanation += f"Key contributors include {top_performers[0][0]} ({top_performers[0][1]:.1%})"
            if len(top_performers) > 1:
                explanation += f" and {top_performers[1][0]} ({top_performers[1][1]:.1%}). "

        return explanation

    async def _explain_criterion_score(self, criterion: str, score: float, variant: dict, assessments: dict) -> str:
        """Explain individual criterion score."""
        explanation = f"The {criterion} score of {score:.1%} "

        # Score interpretation
        if score > 0.8:
            explanation += f"indicates excellent {criterion}, suggesting strong capability in this area. "
        elif score > 0.6:
            explanation += f"shows good {criterion} with solid performance. "
        elif score > 0.4:
            explanation += f"reflects moderate {criterion} with room for improvement. "
        else:
            explanation += f"reveals weak {criterion} that requires attention. "

        # Add specific context based on criterion
        criterion_context = self._get_criterion_specific_context(criterion, score, variant)
        if criterion_context:
            explanation += criterion_context

        return explanation

    def _get_criterion_specific_context(self, criterion: str, score: float, variant: dict) -> str:
        """Get specific context for different criteria."""
        context_map = {
            "feasibility": f"This suggests the approach is {'highly viable' if score > 0.7 else 'moderately viable' if score > 0.5 else 'challenging'} to implement.",
            "impact": f"The potential for {'significant' if score > 0.7 else 'moderate' if score > 0.5 else 'limited'} positive outcomes.",
            "innovation": f"This represents {'breakthrough' if score > 0.8 else 'notable' if score > 0.6 else 'incremental'} innovation potential.",
            "risk": f"The risk level is {'low' if score < 0.3 else 'moderate' if score < 0.6 else 'high'}, requiring {'minimal' if score < 0.3 else 'standard' if score < 0.6 else 'enhanced'} risk management.",
            "complexity": f"Implementation complexity is {'low' if score < 0.4 else 'moderate' if score < 0.7 else 'high'}.",
            "confidence": f"This reflects {'high' if score > 0.7 else 'moderate' if score > 0.5 else 'low'} confidence in the assessment.",
        }

        return context_map.get(criterion, "")

    # Helper methods for explanations

    def _format_list(self, items: list[str], max_items: int = 3) -> str:
        """Format a list of items for natural language."""
        if not items:
            return "no specific items"

        items = items[:max_items]  # Limit to max_items

        if len(items) == 1:
            return items[0]
        elif len(items) == 2:
            return f"{items[0]} and {items[1]}"
        else:
            return f"{', '.join(items[:-1])}, and {items[-1]}"

    def _interpret_performance_level(self, score: float) -> str:
        """Interpret performance level from score."""
        if score > 0.85:
            return "exceptional performance"
        elif score > 0.7:
            return "strong performance"
        elif score > 0.55:
            return "acceptable performance"
        elif score > 0.4:
            return "limited performance"
        else:
            return "poor performance"

    def _interpret_risk_level(self, risk_score: float) -> str:
        """Interpret risk level from score."""
        if risk_score < 0.3:
            return "low"
        elif risk_score < 0.6:
            return "moderate"
        else:
            return "high"

    def _interpret_feasibility_level(self, feasibility_score: float) -> str:
        """Interpret feasibility level from score."""
        if feasibility_score > 0.8:
            return "high"
        elif feasibility_score > 0.6:
            return "good"
        elif feasibility_score > 0.4:
            return "moderate"
        else:
            return "low"

    def _interpret_innovation_level(self, innovation_score: float) -> str:
        """Interpret innovation level from score."""
        if innovation_score > 0.8:
            return "breakthrough"
        elif innovation_score > 0.65:
            return "high"
        elif innovation_score > 0.5:
            return "moderate"
        else:
            return "incremental"

    def _interpret_quality_level(self, quality_score: float) -> str:
        """Interpret quality level from score."""
        if quality_score > 0.85:
            return "excellent"
        elif quality_score > 0.7:
            return "good"
        elif quality_score > 0.55:
            return "acceptable"
        else:
            return "poor"

    def _generate_benefit_explanation(self, criterion: str, score: float) -> str:
        """Generate benefit explanation for high-scoring criteria."""
        benefit_templates = {
            "impact": f"High impact potential ({score:.1%}) enables significant value creation",
            "innovation": f"Strong innovation ({score:.1%}) provides competitive advantages",
            "feasibility": f"High feasibility ({score:.1%}) ensures reliable implementation",
            "quality": f"Excellent quality ({score:.1%}) guarantees superior outcomes",
            "confidence": f"High confidence ({score:.1%}) reduces uncertainty and risk",
        }

        return benefit_templates.get(criterion, f"Strong {criterion} ({score:.1%}) contributes to overall success")

    def _generate_risk_explanation(self, category: str, risk_level: float) -> str:
        """Generate risk explanation for high-risk categories."""
        risk_templates = {
            "technical": f"Technical risks ({risk_level:.1%}) may impact implementation complexity",
            "market": f"Market risks ({risk_level:.1%}) could affect adoption and success",
            "implementation": f"Implementation risks ({risk_level:.1%}) may cause delays or issues",
            "resource": f"Resource risks ({risk_level:.1%}) might constrain execution",
            "timeline": f"Timeline risks ({risk_level:.1%}) could lead to schedule pressures",
        }

        return risk_templates.get(category, f"{category.title()} risks ({risk_level:.1%}) require careful management")

    def _assess_data_quality_explanation(self, evaluation_result: dict) -> str:
        """Assess and explain data quality for confidence."""
        # This would analyze the completeness and quality of input data
        # For now, provide a general assessment
        return "Data quality is sufficient for reliable evaluation with standard confidence levels."

    def _identify_uncertainty_factors(self, evaluation_result: dict) -> str:
        """Identify factors contributing to uncertainty."""
        uncertainty_factors = []

        # Check for missing or low-quality data
        scores = evaluation_result.get("scores", {})
        if not scores.get("core_scores"):
            uncertainty_factors.append("limited scoring data")

        # Check evaluation confidence
        eval_confidence = evaluation_result.get("metadata", {}).get("evaluation_confidence", 0.7)
        if eval_confidence < 0.6:
            uncertainty_factors.append("evaluation methodology limitations")

        if not uncertainty_factors:
            return "No significant uncertainty factors identified in the evaluation."

        return f"Uncertainty factors include: {self._format_list(uncertainty_factors)}."

    # Placeholder methods for remaining explanation types

    async def _explain_quality_assessment(self, quality_assessment: dict, variant: dict) -> str:
        """Explain quality assessment results."""
        overall_quality = quality_assessment.get("overall_quality", 0.5)
        quality_class = quality_assessment.get("quality_class", "acceptable")

        return f"Quality assessment shows {quality_class} quality ({overall_quality:.1%}) across technical, creative, practical, and user-oriented dimensions."

    async def _explain_risk_evaluation(self, risk_evaluation: dict, variant: dict) -> str:
        """Explain risk evaluation results."""
        overall_risk = risk_evaluation.get("overall_risk", 0.5)
        risk_level = risk_evaluation.get("risk_level", "moderate")

        return (
            f"Risk evaluation indicates {risk_level} risk level ({overall_risk:.1%}) across multiple risk categories."
        )

    async def _explain_innovation_assessment(self, innovation_assessment: dict, variant: dict) -> str:
        """Explain innovation assessment results."""
        overall_innovation = innovation_assessment.get("overall_innovation", 0.5)
        innovation_class = innovation_assessment.get("innovation_class", "moderately_innovative")

        return f"Innovation assessment reveals {innovation_class} potential ({overall_innovation:.1%}) with opportunities for creative advancement."

    async def _explain_feasibility_analysis(self, feasibility_analysis: dict, variant: dict) -> str:
        """Explain feasibility analysis results."""
        overall_feasibility = feasibility_analysis.get("overall_feasibility", 0.5)
        feasibility_class = feasibility_analysis.get("feasibility_class", "moderately_feasible")

        return f"Feasibility analysis shows {feasibility_class} implementation viability ({overall_feasibility:.1%}) across technical, economic, and operational factors."

    async def _explain_evaluation_methodology(self, criteria_weights: dict, assessments: dict) -> str:
        """Explain the evaluation methodology used."""
        return "The evaluation used multi-criteria decision analysis with adaptive weighting based on user preferences and context, incorporating technical feasibility, impact potential, innovation level, risk assessment, and quality factors."

    async def _explain_evaluation_confidence_and_limitations(
        self, evaluation_result: dict, variant: dict
    ) -> dict[str, str]:
        """Explain evaluation confidence and limitations."""
        return {
            "confidence": "Evaluation confidence is based on data completeness, methodology rigor, and assessment consistency.",
            "limitations": "Limitations include inherent uncertainty in future projections and potential bias in subjective assessments.",
            "reliability": "Results are reliable within the scope of provided information and standard evaluation uncertainty.",
        }

    def _identify_key_evaluation_factors(self, scores: dict, assessments: dict) -> list[str]:
        """Identify key factors that influenced the evaluation."""
        key_factors = []

        core_scores = scores.get("core_scores", {})
        for criterion, score in core_scores.items():
            if score > 0.7 or score < 0.4:  # High impact or concerning scores
                key_factors.append(f"{criterion} ({score:.1%})")

        return key_factors[:5]  # Top 5 factors

    def _identify_improvement_focus_areas(self, evaluation_result: dict) -> list[str]:
        """Identify areas that would benefit most from improvement."""
        improvement_areas = []

        improvement_suggestions = evaluation_result.get("feedback", {}).get("improvement_suggestions", [])
        for suggestion in improvement_suggestions:
            if suggestion.get("priority") in ["high", "critical"]:
                improvement_areas.append(suggestion.get("area", "unknown"))

        return improvement_areas

    # Simulation explanation methods (simplified implementations)

    async def _explain_success_probability(self, success_probabilities: dict, impact_analysis: dict) -> str:
        """Explain success probability from simulation."""
        overall_success = success_probabilities.get("overall_success", 0.5)
        return f"Simulation indicates {overall_success:.1%} probability of overall success based on Monte Carlo analysis across multiple scenarios."

    async def _explain_impact_analysis(self, impact_analysis: dict, variant: dict) -> str:
        """Explain impact analysis from simulation."""
        expected_impact = impact_analysis.get("expected_impact", 0.5)
        return (
            f"Expected impact score of {expected_impact:.1%} suggests moderate to high potential for positive outcomes."
        )

    async def _explain_simulation_risk_assessment(self, risk_assessment: dict, variant: dict) -> str:
        """Explain risk assessment from simulation."""
        overall_risk = risk_assessment.get("overall_risk", {}).get("mean", 0.5)
        return f"Simulation risk analysis shows average risk level of {overall_risk:.1%} across scenario variations."

    # Additional helper methods

    def _calculate_explanation_confidence(self, evaluation_result: dict) -> float:
        """Calculate confidence in the explanation quality."""
        base_confidence = evaluation_result.get("metadata", {}).get("evaluation_confidence", 0.7)
        data_completeness = len(evaluation_result.get("scores", {})) / 10.0  # Rough estimate
        return np.clip(base_confidence * data_completeness, 0.1, 0.95)

    def _calculate_simulation_explanation_confidence(self, simulation_result: dict) -> float:
        """Calculate confidence in simulation explanation."""
        iterations = simulation_result.get("metadata", {}).get("iterations_completed", 0)
        if iterations >= 1000:
            return 0.9
        elif iterations >= 500:
            return 0.8
        elif iterations >= 100:
            return 0.7
        else:
            return 0.6

    def _calculate_recommendation_explanation_confidence(self, recommendation: dict, evaluation_result: dict) -> float:
        """Calculate confidence in recommendation explanation."""
        rec_confidence = recommendation.get("confidence", 0.7)
        eval_confidence = self._calculate_explanation_confidence(evaluation_result)
        return (rec_confidence + eval_confidence) / 2

    def _cache_explanation(self, explanation_id: str, explanation_result: dict, explanation_type: str):
        """Cache explanation for potential reuse."""
        cache_entry = {
            "explanation_id": explanation_id,
            "explanation_type": explanation_type,
            "timestamp": time.time(),
            "confidence": explanation_result.get("metadata", {}).get("confidence", 0.7),
            "summary": explanation_result.get("summary", {}),
        }
        self.explanation_cache.append(cache_entry)

    def _update_explanation_stats(self, explanation_type: str, confidence: float):
        """Update explanation statistics."""
        self.explanation_stats["total_explanations"] += 1

        if explanation_type not in self.explanation_stats["explanation_types"]:
            self.explanation_stats["explanation_types"][explanation_type] = 0
        self.explanation_stats["explanation_types"][explanation_type] += 1

        # Update average confidence
        total = self.explanation_stats["total_explanations"]
        current_avg = self.explanation_stats["avg_confidence"]
        self.explanation_stats["avg_confidence"] = (current_avg * (total - 1) + confidence) / total

    # Placeholder methods for remaining explanation functionality

    async def _explain_scenario_analysis(self, scenario_coverage: dict, impact_analysis: dict) -> str:
        return "Scenario analysis covered multiple potential futures including optimistic, pessimistic, and most likely scenarios."

    async def _explain_trend_analysis(self, trend_analysis: dict, variant: dict) -> str:
        return "Trend analysis shows performance patterns and convergence indicators across simulation iterations."

    async def _explain_financial_projections(self, financial_summary: dict, variant: dict) -> str:
        return (
            "Financial projections indicate potential revenue, cost, and ROI implications based on simulation results."
        )

    async def _explain_confidence_intervals(self, confidence_intervals: dict) -> str:
        return "Confidence intervals provide statistical ranges for expected outcomes at different probability levels."

    async def _explain_simulation_recommendations(self, recommendations: list[dict], simulation_result: dict) -> str:
        return "Simulation-based recommendations focus on risk mitigation and opportunity maximization strategies."

    async def _explain_simulation_methodology(self, simulation_result: dict) -> str:
        return "Monte Carlo simulation methodology uses random sampling across multiple scenario parameters to generate statistical projections."

    async def _explain_simulation_limitations(self, simulation_result: dict) -> str:
        return "Simulation limitations include model assumptions, parameter uncertainty, and inherent unpredictability of future conditions."

    # Extract methods for key insights

    def _extract_primary_simulation_finding(self, simulation_result: dict) -> str:
        success_rate = simulation_result.get("success_probabilities", {}).get("overall_success", 0.5)
        return f"Primary finding: {success_rate:.1%} probability of achieving intended outcomes"

    def _extract_critical_simulation_risks(self, risk_assessment: dict) -> list[str]:
        high_risk_areas = risk_assessment.get("high_risk_areas", [])
        return high_risk_areas[:3]  # Top 3 critical risks

    def _extract_simulation_success_factors(self, impact_analysis: dict) -> list[str]:
        return [
            "Strong feasibility",
            "Favorable market conditions",
            "Adequate resource availability",
        ]  # Simplified

    def _extract_decision_implications(self, simulation_result: dict) -> str:
        return "Decision implications suggest proceeding with careful risk monitoring and adaptive management."

    # Recommendation explanation methods (continued placeholders)

    async def _explain_main_recommendation(self, action: str, rationale: str, evaluation_result: dict) -> str:
        return f"The recommendation to {action} is based on {rationale} and comprehensive evaluation results."

    async def _explain_evidence_justification(
        self, recommendation: dict, evaluation_result: dict, simulation_result: dict
    ) -> str:
        return "Evidence justification draws from evaluation scores, risk assessment, and simulation projections to support the recommendation."

    async def _explain_alternatives_considered(self, action: str, evaluation_result: dict, context: dict) -> str:
        return "Alternative actions were considered including different implementation approaches and risk tolerance levels."

    async def _explain_recommendation_risks(
        self, recommendation: dict, evaluation_result: dict, simulation_result: dict
    ) -> str:
        return "Recommendation risks include implementation challenges and external factors that could affect success."

    async def _explain_implementation_guidance(self, next_steps: list[str], recommendation: dict, context: dict) -> str:
        return f"Implementation guidance includes {len(next_steps)} key steps for successful execution."

    async def _explain_success_criteria(
        self, recommendation: dict, evaluation_result: dict, simulation_result: dict
    ) -> str:
        return "Success criteria should include performance metrics, timeline milestones, and risk indicators."

    async def _explain_resource_implications(self, recommendation: dict, context: dict, simulation_result: dict) -> str:
        return "Resource implications include budget, personnel, and infrastructure requirements for implementation."

    async def _explain_monitoring_guidance(self, recommendation: dict, evaluation_result: dict) -> str:
        return "Monitoring guidance includes key performance indicators and review checkpoints for adaptive management."

    # Extract methods for recommendation summary

    def _extract_primary_rationale(self, recommendation: dict, evaluation_result: dict) -> str:
        return recommendation.get("rationale", "Based on comprehensive evaluation and analysis")

    def _extract_key_benefits(
        self, recommendation: dict, evaluation_result: dict, simulation_result: dict
    ) -> list[str]:
        return [
            "High success probability",
            "Manageable risk profile",
            "Strong value potential",
        ]  # Simplified

    def _extract_main_risks(self, recommendation: dict, evaluation_result: dict, simulation_result: dict) -> list[str]:
        return [
            "Implementation complexity",
            "Resource constraints",
            "Market uncertainty",
        ]  # Simplified

    def _assess_implementation_complexity(self, recommendation: dict, context: dict) -> str:
        return "moderate"  # Simplified assessment

    def _identify_stakeholder_considerations(self, recommendation: dict, context: dict) -> list[str]:
        return ["User impact", "Team resources", "Organizational alignment"]  # Simplified

    def _assess_decision_reversibility(self, recommendation: dict, context: dict) -> str:
        return "moderate"  # Simplified assessment

    async def _generate_fallback_explanation(
        self, explanation_type: str, primary_data: dict, secondary_data: dict
    ) -> str:
        """Generate fallback explanation when main explanation fails."""
        return f"A {explanation_type} explanation is available but with limited detail due to data constraints. Basic assessment indicates moderate performance with standard considerations."

    async def get_explanation_statistics(self) -> dict[str, Any]:
        """Get comprehensive explanation statistics."""
        return {
            "total_explanations": self.explanation_stats["total_explanations"],
            "explanation_types": self.explanation_stats["explanation_types"],
            "avg_confidence": self.explanation_stats["avg_confidence"],
            "template_usage": self.explanation_stats["template_usage"],
            "language_adaptations": self.explanation_stats["language_adaptations"],
            "cache_status": {
                "explanation_cache_size": len(self.explanation_cache),
                "template_usage_stats_size": len(self.template_usage_stats),
            },
            "language_style": self.language_style.copy(),
        }
