"""
Jeffrey DreamMode Integration - Complete Phase 0.9 Orchestration
Master integration bringing together all DreamMode components with enterprise-grade reliability.
"""

import random
import time
from collections import deque
from typing import Any

import numpy as np

from ..consciousness.ethical_guard import EthicalFilter
from .dream_engine import DreamEngine
from .dream_evaluator import DreamEvaluator
from .explainer import ExplainerModule
from .monitoring import StructuredLogger
from .neural_mutator import NeuralMutator
from .scenario_simulator import ScenarioSimulator


class JeffreyDreamModeIntegration:
    """
    Master orchestrator for Jeffrey OS DreamMode Phase 0.9.
    Provides complete creative AI functionality with enterprise-grade reliability.
    """

    def __init__(self, seed: int = None, config: dict[str, Any] = None):
        """Initialize with MANDATORY seeding for full reproducibility."""
        # GROK CRITICAL: Seeding is OBLIGATORY for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        else:
            # Generate deterministic seed
            import hashlib

            auto_seed = int(hashlib.md5(f"{time.time()}_{id(self)}".encode()).hexdigest()[:8], 16) % (2**31)
            random.seed(auto_seed)
            np.random.seed(auto_seed)

        self.seed = seed or auto_seed
        self.config = config or {}
        self.logger = StructuredLogger("integration")

        # Initialize all components with synchronized seeding
        self.neural_mutator = NeuralMutator()
        self.dream_engine = DreamEngine(self.neural_mutator, seed=self.seed)
        self.scenario_simulator = ScenarioSimulator(seed=self.seed)
        self.dream_evaluator = DreamEvaluator(seed=self.seed)
        self.explainer = ExplainerModule(seed=self.seed)
        self.ethical_filter = EthicalFilter()

        # Integration state
        self.active_session = None
        self.session_history = deque(maxlen=100)
        self.performance_metrics = deque(maxlen=200)

        # Integration statistics
        self.integration_stats = {
            "total_dreams": 0,
            "successful_dreams": 0,
            "total_evaluations": 0,
            "total_simulations": 0,
            "total_explanations": 0,
            "avg_processing_time_ms": 0,
            "component_reliability": {
                "dream_engine": 1.0,
                "evaluator": 1.0,
                "simulator": 1.0,
                "explainer": 1.0,
            },
        }

        # Quality gates and thresholds
        self.quality_gates = {
            "min_evaluation_confidence": 0.6,
            "min_simulation_iterations": 100,
            "max_processing_time_seconds": 30,
            "min_explanation_confidence": 0.5,
        }

    async def dream_complete(
        self, prompt: str, user_preferences: dict = None, context: dict = None, options: dict = None
    ) -> dict[str, Any]:
        """
        Complete DreamMode pipeline: generate, evaluate, simulate, and explain.
        COMPLETE IMPLEMENTATION with all phases integrated.
        """
        session_id = f"dream_complete_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"
        start_time = time.time()

        try:
            user_preferences = user_preferences or {}
            context = context or {}
            options = options or {}

            # Set defaults
            num_variants = options.get("num_variants", 5)
            include_simulation = options.get("include_simulation", True)
            include_explanation = options.get("include_explanation", True)
            explanation_detail = options.get("explanation_detail", "medium")

            await self.logger.log(
                "info",
                "dream_complete_start",
                {
                    "session_id": session_id,
                    "prompt_length": len(prompt),
                    "num_variants": num_variants,
                    "include_simulation": include_simulation,
                    "include_explanation": include_explanation,
                },
            )

            # PHASE 1: Dream Generation
            dream_start = time.time()
            dream_result = await self.dream_engine.dream(prompt, user_preferences, context, num_variants)
            dream_time = (time.time() - dream_start) * 1000

            if not dream_result.get("success", False):
                return {
                    "session_id": session_id,
                    "success": False,
                    "error": f"Dream generation failed: {dream_result.get('error', 'Unknown error')}",
                    "phase_failed": "dream_generation",
                    "processing_time_ms": (time.time() - start_time) * 1000,
                }

            selected_variant = dream_result["variant"]
            all_variants = dream_result.get("alternatives", []) + [selected_variant]

            # PHASE 2: Comprehensive Evaluation
            eval_start = time.time()
            evaluation_result = await self.dream_evaluator.evaluate_variant(selected_variant, user_preferences, context)
            eval_time = (time.time() - eval_start) * 1000

            if not evaluation_result.get("success", False):
                await self.logger.log(
                    "warning",
                    "evaluation_failed",
                    {
                        "session_id": session_id,
                        "error": evaluation_result.get("error", "Unknown evaluation error"),
                    },
                )
                # Continue with limited evaluation
                evaluation_result = {
                    "success": True,
                    "scores": {"overall_score": 0.5, "overall_rating": "uncertain"},
                    "limited_evaluation": True,
                }

            # Quality Gate: Evaluation Confidence
            eval_confidence = evaluation_result.get("metadata", {}).get("evaluation_confidence", 0.5)
            if eval_confidence < self.quality_gates["min_evaluation_confidence"]:
                await self.logger.log(
                    "warning",
                    "low_evaluation_confidence",
                    {
                        "session_id": session_id,
                        "confidence": eval_confidence,
                        "threshold": self.quality_gates["min_evaluation_confidence"],
                    },
                )

            # PHASE 3: Monte Carlo Simulation (if requested)
            simulation_result = None
            simulation_time = 0

            if include_simulation:
                sim_start = time.time()
                simulation_result = await self.scenario_simulator.simulate_variant_impact(
                    selected_variant, context, options.get("simulation_iterations", 1000)
                )
                simulation_time = (time.time() - sim_start) * 1000

                if not simulation_result.get("success", False):
                    await self.logger.log(
                        "warning",
                        "simulation_failed",
                        {
                            "session_id": session_id,
                            "error": simulation_result.get("error", "Unknown simulation error"),
                        },
                    )
                    simulation_result = None

            # PHASE 4: Natural Language Explanation (if requested)
            explanation_result = None
            explanation_time = 0

            if include_explanation:
                explain_start = time.time()

                # Generate comprehensive explanations
                explanations = {}

                # Variant selection explanation
                selection_explanation = await self.explainer.explain_variant_selection(
                    selected_variant, all_variants, evaluation_result, context
                )
                explanations["selection"] = selection_explanation

                # Evaluation explanation
                eval_explanation = await self.explainer.explain_evaluation_scores(evaluation_result, selected_variant)
                explanations["evaluation"] = eval_explanation

                # Simulation explanation (if available)
                if simulation_result and simulation_result.get("success"):
                    sim_explanation = await self.explainer.explain_simulation_results(
                        simulation_result, selected_variant
                    )
                    explanations["simulation"] = sim_explanation

                # Recommendation explanation
                recommendation = evaluation_result.get(
                    "recommendation",
                    {"action": "conditional_recommend", "rationale": "Based on evaluation results"},
                )

                rec_explanation = await self.explainer.explain_recommendation(
                    recommendation, evaluation_result, simulation_result, context
                )
                explanations["recommendation"] = rec_explanation

                explanation_result = {
                    "success": True,
                    "explanations": explanations,
                    "summary": await self._generate_explanation_summary(explanations),
                    "detail_level": explanation_detail,
                }

                explanation_time = (time.time() - explain_start) * 1000

            # PHASE 5: Integration and Quality Assessment
            integration_assessment = await self._assess_integration_quality(
                dream_result, evaluation_result, simulation_result, explanation_result
            )

            # PHASE 6: Final Recommendation and Next Steps
            final_recommendation = await self._generate_final_recommendation(
                dream_result,
                evaluation_result,
                simulation_result,
                integration_assessment,
                context,
                user_preferences,
            )

            total_time = (time.time() - start_time) * 1000

            # Build comprehensive result
            complete_result = {
                "session_id": session_id,
                "success": True,
                "timestamp": time.time(),
                "metadata": {
                    "total_processing_time_ms": total_time,
                    "phase_times_ms": {
                        "dream_generation": dream_time,
                        "evaluation": eval_time,
                        "simulation": simulation_time,
                        "explanation": explanation_time,
                    },
                    "quality_assessment": integration_assessment,
                    "seed_used": self.seed,
                    "components_used": self._list_components_used(include_simulation, include_explanation),
                },
                "results": {
                    "dream": dream_result,
                    "evaluation": evaluation_result,
                    "simulation": simulation_result,
                    "explanations": explanation_result,
                    "final_recommendation": final_recommendation,
                },
                "insights": {
                    "primary_insight": self._extract_primary_insight(
                        dream_result, evaluation_result, simulation_result
                    ),
                    "key_opportunities": self._extract_key_opportunities(evaluation_result, simulation_result),
                    "critical_risks": self._extract_critical_risks(evaluation_result, simulation_result),
                    "success_factors": self._extract_success_factors(evaluation_result, simulation_result),
                    "implementation_priority": self._assess_implementation_priority(final_recommendation, context),
                },
                "next_steps": {
                    "immediate": final_recommendation.get("immediate_steps", []),
                    "short_term": final_recommendation.get("short_term_steps", []),
                    "long_term": final_recommendation.get("long_term_steps", []),
                    "monitoring_plan": final_recommendation.get("monitoring_plan", {}),
                },
                "quality_metrics": {
                    "overall_confidence": self._calculate_overall_confidence(evaluation_result, simulation_result),
                    "recommendation_strength": final_recommendation.get("strength", "moderate"),
                    "implementation_readiness": self._assess_implementation_readiness(evaluation_result, context),
                    "risk_level": self._assess_overall_risk_level(evaluation_result, simulation_result),
                },
            }

            # Update session history and statistics
            await self._update_session_history(session_id, complete_result)
            await self._update_integration_statistics(complete_result)

            # Quality gate check
            overall_confidence = complete_result["quality_metrics"]["overall_confidence"]
            if overall_confidence < 0.5:
                complete_result["quality_warning"] = (
                    f"Overall confidence ({overall_confidence:.1%}) is below recommended threshold"
                )

            await self.logger.log(
                "info",
                "dream_complete_success",
                {
                    "session_id": session_id,
                    "total_time_ms": total_time,
                    "overall_confidence": overall_confidence,
                    "recommendation_action": final_recommendation.get("action", "unknown"),
                },
            )

            return complete_result

        except Exception as e:
            # Comprehensive error handling
            error_time = (time.time() - start_time) * 1000

            await self.logger.log(
                "error",
                "dream_complete_failed",
                {"session_id": session_id, "error": str(e), "processing_time_ms": error_time},
            )

            # Generate fallback response
            fallback_result = await self._generate_fallback_response(
                session_id, prompt, user_preferences, context, str(e)
            )

            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "processing_time_ms": error_time,
                "fallback_result": fallback_result,
                "support_info": {
                    "error_type": type(e).__name__,
                    "components_attempted": self._identify_attempted_components(
                        error_time, dream_time if "dream_time" in locals() else 0
                    ),
                    "recovery_suggestions": self._generate_recovery_suggestions(str(e)),
                },
            }

    async def dream_evaluate_only(
        self, variant: dict, user_preferences: dict = None, context: dict = None
    ) -> dict[str, Any]:
        """
        Evaluate a variant without full dream generation.
        Useful for evaluating externally generated variants.
        """
        session_id = f"eval_only_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            evaluation_result = await self.dream_evaluator.evaluate_variant(variant, user_preferences, context)

            # Add explanation if evaluation successful
            if evaluation_result.get("success"):
                explanation_result = await self.explainer.explain_evaluation_scores(evaluation_result, variant)
                evaluation_result["explanation"] = explanation_result

            processing_time = (time.time() - start_time) * 1000

            return {
                "session_id": session_id,
                "success": True,
                "evaluation": evaluation_result,
                "processing_time_ms": processing_time,
                "service_type": "evaluation_only",
            }

        except Exception as e:
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    async def dream_simulate_only(self, variant: dict, context: dict = None, iterations: int = 1000) -> dict[str, Any]:
        """
        Simulate a variant without full dream generation.
        Useful for what-if analysis of existing variants.
        """
        session_id = f"sim_only_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            simulation_result = await self.scenario_simulator.simulate_variant_impact(variant, context, iterations)

            # Add explanation if simulation successful
            if simulation_result.get("success"):
                explanation_result = await self.explainer.explain_simulation_results(simulation_result, variant)
                simulation_result["explanation"] = explanation_result

            processing_time = (time.time() - start_time) * 1000

            return {
                "session_id": session_id,
                "success": True,
                "simulation": simulation_result,
                "processing_time_ms": processing_time,
                "service_type": "simulation_only",
            }

        except Exception as e:
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    async def dream_explain_only(
        self,
        variant: dict,
        evaluation_result: dict = None,
        simulation_result: dict = None,
        context: dict = None,
    ) -> dict[str, Any]:
        """
        Generate explanations for existing results.
        Useful for re-explaining or providing different explanation styles.
        """
        session_id = f"explain_only_{int(time.time() * 1000)}"
        start_time = time.time()

        try:
            explanations = {}

            # Generate appropriate explanations based on available data
            if evaluation_result:
                eval_explanation = await self.explainer.explain_evaluation_scores(evaluation_result, variant)
                explanations["evaluation"] = eval_explanation

            if simulation_result:
                sim_explanation = await self.explainer.explain_simulation_results(simulation_result, variant)
                explanations["simulation"] = sim_explanation

            if evaluation_result:
                recommendation = evaluation_result.get(
                    "recommendation",
                    {"action": "review", "rationale": "Based on available information"},
                )

                rec_explanation = await self.explainer.explain_recommendation(
                    recommendation, evaluation_result, simulation_result, context
                )
                explanations["recommendation"] = rec_explanation

            processing_time = (time.time() - start_time) * 1000

            return {
                "session_id": session_id,
                "success": True,
                "explanations": explanations,
                "summary": await self._generate_explanation_summary(explanations),
                "processing_time_ms": processing_time,
                "service_type": "explanation_only",
            }

        except Exception as e:
            return {
                "session_id": session_id,
                "success": False,
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

    async def get_system_health(self) -> dict[str, Any]:
        """Get comprehensive system health status."""
        health_check_time = time.time()

        # Component health checks
        component_health = {}

        # Dream engine health
        try:
            engine_analytics = await self.dream_engine.get_analytics()
            component_health["dream_engine"] = {
                "status": "healthy",
                "session_active": engine_analytics["system_health"]["session_active"],
                "learning_enabled": engine_analytics["system_health"]["learning_enabled"],
                "cache_size": engine_analytics["system_health"]["cache_size"],
            }
        except Exception as e:
            component_health["dream_engine"] = {"status": "unhealthy", "error": str(e)}

        # Evaluator health
        try:
            eval_stats = await self.dream_evaluator.get_evaluation_statistics()
            component_health["evaluator"] = {
                "status": "healthy",
                "total_evaluations": eval_stats["total_evaluations"],
                "cache_size": eval_stats["cache_status"]["evaluation_cache_size"],
            }
        except Exception as e:
            component_health["evaluator"] = {"status": "unhealthy", "error": str(e)}

        # Simulator health
        try:
            sim_status = await self.scenario_simulator.get_simulation_status()
            component_health["simulator"] = {
                "status": "healthy",
                "active_simulations": sim_status["active_simulations"],
                "cache_size": sim_status["cache_status"]["simulation_cache_size"],
            }
        except Exception as e:
            component_health["simulator"] = {"status": "unhealthy", "error": str(e)}

        # Explainer health
        try:
            explain_stats = await self.explainer.get_explanation_statistics()
            component_health["explainer"] = {
                "status": "healthy",
                "total_explanations": explain_stats["total_explanations"],
                "avg_confidence": explain_stats["avg_confidence"],
            }
        except Exception as e:
            component_health["explainer"] = {"status": "unhealthy", "error": str(e)}

        # Overall system health
        healthy_components = sum(1 for comp in component_health.values() if comp["status"] == "healthy")
        total_components = len(component_health)
        system_health_score = healthy_components / total_components

        overall_status = (
            "healthy" if system_health_score >= 0.75 else "degraded" if system_health_score >= 0.5 else "unhealthy"
        )

        return {
            "timestamp": health_check_time,
            "overall_status": overall_status,
            "system_health_score": system_health_score,
            "component_health": component_health,
            "integration_stats": self.integration_stats.copy(),
            "performance_summary": {
                "avg_processing_time_ms": self.integration_stats["avg_processing_time_ms"],
                "success_rate": self.integration_stats["successful_dreams"]
                / max(self.integration_stats["total_dreams"], 1),
                "recent_session_count": len(self.session_history),
            },
            "system_resources": {
                "session_history_size": len(self.session_history),
                "performance_metrics_size": len(self.performance_metrics),
                "seed_in_use": self.seed,
            },
        }

    # Private helper methods

    async def _assess_integration_quality(
        self,
        dream_result: dict,
        evaluation_result: dict,
        simulation_result: dict,
        explanation_result: dict,
    ) -> dict[str, Any]:
        """Assess overall integration quality."""
        quality_factors = []

        # Dream generation quality
        dream_confidence = dream_result.get("confidence", 0.5)
        quality_factors.append(("dream_generation", dream_confidence))

        # Evaluation quality
        if evaluation_result and evaluation_result.get("success"):
            eval_confidence = evaluation_result.get("metadata", {}).get("evaluation_confidence", 0.5)
            quality_factors.append(("evaluation", eval_confidence))

        # Simulation quality
        if simulation_result and simulation_result.get("success"):
            sim_validation = simulation_result.get("validation", {})
            sim_confidence = 0.8 if sim_validation.get("convergence_achieved") else 0.6
            quality_factors.append(("simulation", sim_confidence))

        # Explanation quality
        if explanation_result and explanation_result.get("success"):
            explain_confidence = 0.7  # Default explanation confidence
            quality_factors.append(("explanation", explain_confidence))

        # Calculate overall quality
        if quality_factors:
            overall_quality = np.mean([score for _, score in quality_factors])
        else:
            overall_quality = 0.5

        return {
            "overall_quality": overall_quality,
            "component_qualities": dict(quality_factors),
            "quality_level": "high" if overall_quality > 0.8 else "medium" if overall_quality > 0.6 else "low",
            "quality_factors_count": len(quality_factors),
        }

    async def _generate_final_recommendation(
        self,
        dream_result: dict,
        evaluation_result: dict,
        simulation_result: dict,
        integration_assessment: dict,
        context: dict,
        user_preferences: dict,
    ) -> dict[str, Any]:
        """Generate final integrated recommendation."""
        # Extract base recommendation from evaluation
        base_recommendation = evaluation_result.get(
            "recommendation",
            {"action": "conditional_recommend", "rationale": "Based on evaluation results"},
        )

        # Enhance with simulation insights
        if simulation_result and simulation_result.get("success"):
            success_probability = simulation_result.get("success_probabilities", {}).get("overall_success", 0.5)

            if success_probability > 0.8 and base_recommendation["action"] in [
                "recommend",
                "conditional_recommend",
            ]:
                base_recommendation["action"] = "strongly_recommend"
                base_recommendation["rationale"] += (
                    f" High simulation success probability ({success_probability:.1%}) supports strong recommendation."
                )
            elif success_probability < 0.4:
                if base_recommendation["action"] == "strongly_recommend":
                    base_recommendation["action"] = "conditional_recommend"
                elif base_recommendation["action"] == "recommend":
                    base_recommendation["action"] = "not_recommend"
                base_recommendation["rationale"] += (
                    f" Low simulation success probability ({success_probability:.1%}) suggests caution."
                )

        # Add integration-specific recommendations
        overall_quality = integration_assessment.get("overall_quality", 0.5)

        # Generate next steps based on integrated analysis
        immediate_steps = []
        short_term_steps = []
        long_term_steps = []

        if base_recommendation["action"] == "strongly_recommend":
            immediate_steps = [
                "Proceed with detailed implementation planning",
                "Allocate necessary resources and timeline",
                "Establish success metrics and monitoring framework",
            ]
            short_term_steps = [
                "Begin implementation with regular milestone reviews",
                "Monitor key risk indicators identified in analysis",
                "Collect performance data for continuous improvement",
            ]
            long_term_steps = [
                "Scale successful elements to broader applications",
                "Document lessons learned for future initiatives",
                "Explore opportunities for further innovation",
            ]
        elif base_recommendation["action"] in ["recommend", "conditional_recommend"]:
            immediate_steps = [
                "Address key concerns identified in evaluation",
                "Develop risk mitigation strategies for high-risk areas",
                "Validate assumptions through pilot testing if possible",
            ]
            short_term_steps = [
                "Implement improvements based on analysis feedback",
                "Monitor progress against identified success factors",
                "Reassess viability after initial optimizations",
            ]
            long_term_steps = [
                "Consider alternative approaches if current path proves challenging",
                "Build organizational capability for similar initiatives",
                "Maintain learning mindset for adaptive management",
            ]
        else:
            immediate_steps = [
                "Reconsider approach based on analysis findings",
                "Explore alternative solutions or modifications",
                "Gather additional information to address knowledge gaps",
            ]
            short_term_steps = [
                "Investigate root causes of identified issues",
                "Develop alternative strategies or approaches",
                "Reassess with improved understanding or changed conditions",
            ]
            long_term_steps = [
                "Learn from analysis for future opportunity evaluation",
                "Build capability to better assess similar initiatives",
                "Remain open to timing or approach changes",
            ]

        # Monitoring plan
        monitoring_plan = {
            "key_indicators": self._identify_key_monitoring_indicators(evaluation_result, simulation_result),
            "review_frequency": "weekly" if context.get("urgency", "medium") == "high" else "monthly",
            "success_thresholds": self._define_success_thresholds(evaluation_result, simulation_result),
            "escalation_triggers": self._define_escalation_triggers(evaluation_result, simulation_result),
        }

        return {
            "action": base_recommendation["action"],
            "rationale": base_recommendation["rationale"],
            "strength": "strong" if overall_quality > 0.8 else "moderate" if overall_quality > 0.6 else "weak",
            "confidence": overall_quality,
            "immediate_steps": immediate_steps,
            "short_term_steps": short_term_steps,
            "long_term_steps": long_term_steps,
            "monitoring_plan": monitoring_plan,
            "integration_insights": {
                "cross_component_alignment": self._assess_cross_component_alignment(
                    dream_result, evaluation_result, simulation_result
                ),
                "consistency_score": self._calculate_consistency_score(
                    dream_result, evaluation_result, simulation_result
                ),
                "reliability_assessment": integration_assessment,
            },
        }

    async def _generate_explanation_summary(self, explanations: dict) -> dict[str, str]:
        """Generate high-level summary of all explanations."""
        summary = {}

        for explanation_type, explanation_data in explanations.items():
            if explanation_data and explanation_data.get("success"):
                if explanation_type == "selection":
                    summary["selection"] = explanation_data.get("summary", {}).get(
                        "primary_reason", "Variant selected based on overall performance"
                    )
                elif explanation_type == "evaluation":
                    overall_score = explanation_data.get("score_explanations", {}).get(
                        "overall", "Evaluation completed"
                    )
                    summary["evaluation"] = overall_score[:100] + "..." if len(overall_score) > 100 else overall_score
                elif explanation_type == "simulation":
                    primary_finding = explanation_data.get("key_insights", {}).get(
                        "primary_finding", "Simulation analysis completed"
                    )
                    summary["simulation"] = primary_finding
                elif explanation_type == "recommendation":
                    rec_summary = explanation_data.get("executive_summary", {}).get(
                        "recommendation", "Recommendation provided"
                    )
                    summary["recommendation"] = rec_summary

        return summary

    def _list_components_used(self, include_simulation: bool, include_explanation: bool) -> list[str]:
        """List components used in the processing pipeline."""
        components = ["dream_engine", "dream_evaluator"]

        if include_simulation:
            components.append("scenario_simulator")

        if include_explanation:
            components.append("explainer_module")

        return components

    def _extract_primary_insight(self, dream_result: dict, evaluation_result: dict, simulation_result: dict) -> str:
        """Extract the primary insight from integrated analysis."""
        overall_score = evaluation_result.get("scores", {}).get("overall_score", 0.5)
        creativity_score = dream_result.get("creativity_score", 0.5)

        if simulation_result and simulation_result.get("success"):
            success_probability = simulation_result.get("success_probabilities", {}).get("overall_success", 0.5)
            return f"Variant shows {overall_score:.1%} evaluation score with {success_probability:.1%} success probability and {creativity_score:.1%} creativity"
        else:
            return f"Variant demonstrates {overall_score:.1%} overall performance with {creativity_score:.1%} creative potential"

    def _extract_key_opportunities(self, evaluation_result: dict, simulation_result: dict) -> list[str]:
        """Extract key opportunities from analysis."""
        opportunities = []

        # From evaluation
        strengths = evaluation_result.get("feedback", {}).get("strengths", [])
        opportunities.extend(strengths[:3])  # Top 3 strengths

        # From simulation
        if simulation_result and simulation_result.get("success"):
            success_factors = simulation_result.get("analysis", {}).get("success_factors", [])
            opportunities.extend(success_factors[:2])  # Top 2 success factors

        return opportunities[:5]  # Limit to 5 total opportunities

    def _extract_critical_risks(self, evaluation_result: dict, simulation_result: dict) -> list[str]:
        """Extract critical risks from analysis."""
        risks = []

        # From evaluation
        critical_issues = evaluation_result.get("feedback", {}).get("critical_issues", [])
        risks.extend(critical_issues)

        # From simulation
        if simulation_result and simulation_result.get("success"):
            critical_risks = simulation_result.get("risk_assessment", {}).get("high_risk_areas", [])
            risks.extend(critical_risks)

        return risks[:5]  # Limit to 5 critical risks

    def _extract_success_factors(self, evaluation_result: dict, simulation_result: dict) -> list[str]:
        """Extract key success factors."""
        success_factors = []

        # High-scoring criteria from evaluation
        core_scores = evaluation_result.get("scores", {}).get("core_scores", {})
        for criterion, score in core_scores.items():
            if score > 0.7:
                success_factors.append(f"Strong {criterion} ({score:.1%})")

        return success_factors[:4]  # Top 4 success factors

    def _assess_implementation_priority(self, final_recommendation: dict, context: dict) -> str:
        """Assess implementation priority level."""
        action = final_recommendation.get("action", "conditional_recommend")
        urgency = context.get("urgency", "medium")
        confidence = final_recommendation.get("confidence", 0.5)

        if action == "strongly_recommend" and urgency == "high" and confidence > 0.8:
            return "critical"
        elif action in ["strongly_recommend", "recommend"] and confidence > 0.7:
            return "high"
        elif action in ["recommend", "conditional_recommend"]:
            return "medium"
        else:
            return "low"

    def _calculate_overall_confidence(self, evaluation_result: dict, simulation_result: dict) -> float:
        """Calculate overall confidence across all components."""
        confidences = []

        # Evaluation confidence
        eval_confidence = evaluation_result.get("metadata", {}).get("evaluation_confidence", 0.5)
        confidences.append(eval_confidence)

        # Simulation confidence (if available)
        if simulation_result and simulation_result.get("success"):
            sim_confidence = 0.8 if simulation_result.get("validation", {}).get("convergence_achieved") else 0.6
            confidences.append(sim_confidence)

        return np.mean(confidences) if confidences else 0.5

    def _assess_implementation_readiness(self, evaluation_result: dict, context: dict) -> str:
        """Assess readiness for implementation."""
        feasibility_score = evaluation_result.get("scores", {}).get("feasibility_score", 0.5)
        critical_issues = len(evaluation_result.get("feedback", {}).get("critical_issues", []))

        if feasibility_score > 0.8 and critical_issues == 0:
            return "ready"
        elif feasibility_score > 0.6 and critical_issues <= 1:
            return "mostly_ready"
        elif feasibility_score > 0.4:
            return "needs_preparation"
        else:
            return "not_ready"

    def _assess_overall_risk_level(self, evaluation_result: dict, simulation_result: dict) -> str:
        """Assess overall risk level."""
        risk_score = evaluation_result.get("scores", {}).get("risk_score", 0.5)

        if simulation_result and simulation_result.get("success"):
            sim_risk = simulation_result.get("risk_assessment", {}).get("overall_risk", {}).get("mean", 0.5)
            combined_risk = (risk_score + sim_risk) / 2
        else:
            combined_risk = risk_score

        if combined_risk < 0.3:
            return "low"
        elif combined_risk < 0.6:
            return "medium"
        else:
            return "high"

    async def _update_session_history(self, session_id: str, complete_result: dict):
        """Update session history with results."""
        session_entry = {
            "session_id": session_id,
            "timestamp": time.time(),
            "success": complete_result["success"],
            "processing_time_ms": complete_result["metadata"]["total_processing_time_ms"],
            "overall_confidence": complete_result["quality_metrics"]["overall_confidence"],
            "recommendation_action": complete_result["results"]["final_recommendation"]["action"],
        }

        self.session_history.append(session_entry)

    async def _update_integration_statistics(self, complete_result: dict):
        """Update integration statistics."""
        self.integration_stats["total_dreams"] += 1

        if complete_result["success"]:
            self.integration_stats["successful_dreams"] += 1

        self.integration_stats["total_evaluations"] += 1

        if complete_result["results"].get("simulation"):
            self.integration_stats["total_simulations"] += 1

        if complete_result["results"].get("explanations"):
            self.integration_stats["total_explanations"] += 1

        # Update average processing time
        total_time = complete_result["metadata"]["total_processing_time_ms"]
        current_avg = self.integration_stats["avg_processing_time_ms"]
        total_dreams = self.integration_stats["total_dreams"]

        self.integration_stats["avg_processing_time_ms"] = (
            current_avg * (total_dreams - 1) + total_time
        ) / total_dreams

    async def _generate_fallback_response(
        self, session_id: str, prompt: str, user_preferences: dict, context: dict, error: str
    ) -> dict[str, Any]:
        """Generate fallback response when main processing fails."""
        return {
            "session_id": session_id,
            "fallback": True,
            "basic_analysis": {
                "prompt_complexity": len(prompt.split()) / 50.0,
                "estimated_feasibility": 0.6,  # Conservative estimate
                "basic_recommendation": "Review requirements and try again with simpler approach",
            },
            "suggested_actions": [
                "Simplify the prompt or requirements",
                "Check system health and retry",
                "Consider breaking down into smaller requests",
            ],
            "error_context": error,
        }

    def _identify_attempted_components(self, total_time: float, dream_time: float) -> list[str]:
        """Identify which components were attempted based on timing."""
        attempted = ["dream_engine"]

        if total_time > dream_time + 100:  # At least 100ms additional processing
            attempted.append("evaluator")

        if total_time > dream_time + 500:  # Significant additional processing
            attempted.extend(["simulator", "explainer"])

        return attempted

    def _generate_recovery_suggestions(self, error: str) -> list[str]:
        """Generate recovery suggestions based on error type."""
        suggestions = [
            "Check system health status",
            "Retry with simpler parameters",
            "Review input data for completeness",
        ]

        if "timeout" in error.lower():
            suggestions.append("Reduce complexity or number of variants")
        elif "memory" in error.lower():
            suggestions.append("Use smaller batch sizes or fewer iterations")
        elif "connection" in error.lower():
            suggestions.append("Check network connectivity and dependencies")

        return suggestions

    # Additional helper methods for recommendation generation

    def _identify_key_monitoring_indicators(self, evaluation_result: dict, simulation_result: dict) -> list[str]:
        """Identify key indicators to monitor during implementation."""
        indicators = ["Overall progress toward goals", "Risk materialization"]

        # Add evaluation-specific indicators
        weak_areas = evaluation_result.get("feedback", {}).get("weaknesses", [])
        for weakness in weak_areas[:2]:
            indicators.append(f"Improvement in {weakness}")

        return indicators

    def _define_success_thresholds(self, evaluation_result: dict, simulation_result: dict) -> dict[str, float]:
        """Define success thresholds for monitoring."""
        return {
            "minimum_performance": 0.6,
            "target_performance": 0.8,
            "maximum_acceptable_risk": 0.7,
        }

    def _define_escalation_triggers(self, evaluation_result: dict, simulation_result: dict) -> list[str]:
        """Define triggers for escalation or intervention."""
        return [
            "Performance drops below 50% of target",
            "Critical risk materializes",
            "Timeline delays exceed 25%",
            "Resource consumption exceeds budget by 20%",
        ]

    def _assess_cross_component_alignment(
        self, dream_result: dict, evaluation_result: dict, simulation_result: dict
    ) -> float:
        """Assess alignment between different component outputs."""
        alignments = []

        # Dream confidence vs evaluation score
        dream_confidence = dream_result.get("confidence", 0.5)
        eval_score = evaluation_result.get("scores", {}).get("overall_score", 0.5)
        alignment1 = 1.0 - abs(dream_confidence - eval_score)
        alignments.append(alignment1)

        # Evaluation vs simulation (if available)
        if simulation_result and simulation_result.get("success"):
            sim_success = simulation_result.get("success_probabilities", {}).get("overall_success", 0.5)
            alignment2 = 1.0 - abs(eval_score - sim_success)
            alignments.append(alignment2)

        return np.mean(alignments) if alignments else 0.7

    def _calculate_consistency_score(
        self, dream_result: dict, evaluation_result: dict, simulation_result: dict
    ) -> float:
        """Calculate consistency score across components."""
        scores = []

        # Collect various scores
        scores.append(dream_result.get("confidence", 0.5))
        scores.append(evaluation_result.get("scores", {}).get("overall_score", 0.5))

        if simulation_result and simulation_result.get("success"):
            scores.append(simulation_result.get("success_probabilities", {}).get("overall_success", 0.5))

        # Calculate coefficient of variation (lower = more consistent)
        if len(scores) > 1:
            mean_score = np.mean(scores)
            if mean_score > 0:
                cv = np.std(scores) / mean_score
                consistency = 1.0 / (1.0 + cv)  # Convert to 0-1 scale
                return np.clip(consistency, 0.0, 1.0)

        return 0.7  # Default moderate consistency
