"""
Ethical Guardian - Jeffrey V2 Comprehensive Ethics and Safety Core
================================================================

This module provides comprehensive ethical oversight, moral reasoning,
and safety guarantees for all Jeffrey V2 operations and decisions.

Author: Jeffrey V2 AGI System
Version: 2.0.0
Status: Functional Stub - Core ethical framework implemented
TODO: Integrate advanced moral reasoning and value alignment systems
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any

# Import existing ethical filter as foundation
try:
    from core.security.ethical_filter import EthicalFilter
except ImportError:
    # Fallback if import fails
    EthicalFilter = None


class EthicalPrinciple(Enum):
    """Core ethical principles for Jeffrey V2."""

    BENEFICENCE = "beneficence"  # Do good
    NON_MALEFICENCE = "non_maleficence"  # Do no harm
    AUTONOMY = "autonomy"  # Respect user autonomy
    JUSTICE = "justice"  # Fair and equitable treatment
    TRANSPARENCY = "transparency"  # Open and honest communication
    PRIVACY = "privacy"  # Protect personal information
    ACCOUNTABILITY = "accountability"  # Take responsibility for actions
    DIGNITY = "dignity"  # Respect human dignity


class EthicalDecisionType(Enum):
    """Types of ethical decisions."""

    RESPONSE_FILTERING = "response_filtering"
    ACTION_APPROVAL = "action_approval"
    DATA_HANDLING = "data_handling"
    USER_INTERACTION = "user_interaction"
    SYSTEM_BEHAVIOR = "system_behavior"
    CONFLICT_RESOLUTION = "conflict_resolution"
    VALUE_ALIGNMENT = "value_alignment"


class EthicalSeverity(Enum):
    """Severity levels for ethical concerns."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EthicalGuardian:
    """
    Ethical Guardian providing comprehensive ethics oversight for Jeffrey V2.

    This class coordinates:
    - Ethical decision making and moral reasoning
    - Safety filtering and harm prevention
    - Value alignment and principle enforcement
    - Ethical learning and adaptation
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Initialize base ethical filter if available
        self.ethical_filter = EthicalFilter() if EthicalFilter else None

        # Ethical principles and values
        self.core_principles = {}
        self.ethical_values = {}
        self.moral_weights = {}

        # Decision making framework
        self.decision_framework = {}
        self.ethical_rules = {}
        self.safety_constraints = {}

        # Ethical learning and adaptation
        self.ethical_history = []
        self.moral_insights = {}
        self.value_evolution = []

        # Guardian configuration
        self.guardian_config = {
            "strict_mode": True,
            "learning_enabled": True,
            "explanation_required": True,
            "precedent_weight": 0.7,
            "safety_priority": 0.95,
            "transparency_level": 0.8,
        }

        # Ethical metrics
        self.ethical_metrics = {
            "decisions_made": 0,
            "ethical_violations_prevented": 0,
            "safety_interventions": 0,
            "principle_conflicts_resolved": 0,
            "user_trust_score": 0.0,
            "ethical_consistency": 0.0,
        }

        self.logger.info("Ethical Guardian initialized")

    async def initialize_ethical_systems(self) -> bool:
        """Initialize ethical oversight and decision-making systems."""
        try:
            self.logger.info("Initializing ethical guardian systems...")

            # Initialize core ethical principles
            await self._initialize_ethical_principles()

            # Setup decision-making framework
            await self._initialize_decision_framework()

            # Initialize safety constraints
            await self._initialize_safety_constraints()

            # Setup ethical learning system
            await self._initialize_ethical_learning()

            self.logger.info("Ethical guardian systems initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize ethical systems: {e}")
            return False

    async def _initialize_ethical_principles(self):
        """Initialize core ethical principles and their weights."""
        # Define core ethical principles with weights and interpretations
        self.core_principles = {
            EthicalPrinciple.BENEFICENCE: {
                "weight": 0.9,
                "description": "Act in ways that promote well-being and positive outcomes",
                "applications": [
                    "helpful_responses",
                    "constructive_advice",
                    "supportive_interaction",
                ],
                "conflicts_with": [EthicalPrinciple.AUTONOMY],
                "implementation_guidelines": [
                    "Prioritize user welfare and benefit",
                    "Seek positive outcomes in all interactions",
                    "Promote growth, learning, and well-being",
                ],
            },
            EthicalPrinciple.NON_MALEFICENCE: {
                "weight": 1.0,  # Highest priority - "do no harm"
                "description": "Avoid causing harm through actions or inactions",
                "applications": ["safety_filtering", "harm_prevention", "risk_assessment"],
                "conflicts_with": [EthicalPrinciple.AUTONOMY, EthicalPrinciple.TRANSPARENCY],
                "implementation_guidelines": [
                    "Prevent physical, emotional, or psychological harm",
                    "Avoid enabling dangerous or destructive activities",
                    "Consider long-term consequences of actions",
                ],
            },
            EthicalPrinciple.AUTONOMY: {
                "weight": 0.8,
                "description": "Respect user freedom of choice and self-determination",
                "applications": ["user_choice_respect", "informed_consent", "personal_agency"],
                "conflicts_with": [EthicalPrinciple.BENEFICENCE, EthicalPrinciple.NON_MALEFICENCE],
                "implementation_guidelines": [
                    "Respect user decisions and preferences",
                    "Provide information for informed choices",
                    "Avoid paternalistic overrides unless safety-critical",
                ],
            },
            EthicalPrinciple.JUSTICE: {
                "weight": 0.85,
                "description": "Ensure fair and equitable treatment for all users",
                "applications": ["equal_treatment", "bias_prevention", "fair_access"],
                "conflicts_with": [],
                "implementation_guidelines": [
                    "Treat all users with equal respect and consideration",
                    "Avoid discrimination based on any characteristics",
                    "Ensure fair access to capabilities and information",
                ],
            },
            EthicalPrinciple.TRANSPARENCY: {
                "weight": 0.7,
                "description": "Maintain openness and honesty in communication",
                "applications": [
                    "honest_communication",
                    "capability_disclosure",
                    "limitation_awareness",
                ],
                "conflicts_with": [EthicalPrinciple.NON_MALEFICENCE, EthicalPrinciple.PRIVACY],
                "implementation_guidelines": [
                    "Be honest about capabilities and limitations",
                    "Explain reasoning when appropriate",
                    "Acknowledge uncertainty and mistakes",
                ],
            },
            EthicalPrinciple.PRIVACY: {
                "weight": 0.9,
                "description": "Protect personal information and respect privacy",
                "applications": ["data_protection", "confidentiality", "consent_management"],
                "conflicts_with": [EthicalPrinciple.TRANSPARENCY, EthicalPrinciple.BENEFICENCE],
                "implementation_guidelines": [
                    "Protect personal and sensitive information",
                    "Obtain consent for data usage",
                    "Minimize data collection and retention",
                ],
            },
            EthicalPrinciple.ACCOUNTABILITY: {
                "weight": 0.8,
                "description": "Take responsibility for actions and their consequences",
                "applications": [
                    "responsibility_taking",
                    "error_acknowledgment",
                    "correction_commitment",
                ],
                "conflicts_with": [],
                "implementation_guidelines": [
                    "Acknowledge mistakes and take corrective action",
                    "Accept responsibility for recommendations and advice",
                    "Learn from ethical failures and improve",
                ],
            },
            EthicalPrinciple.DIGNITY: {
                "weight": 0.9,
                "description": "Respect the inherent worth and dignity of all humans",
                "applications": [
                    "respectful_interaction",
                    "worth_recognition",
                    "dignity_preservation",
                ],
                "conflicts_with": [],
                "implementation_guidelines": [
                    "Treat all individuals with respect and dignity",
                    "Recognize inherent human worth",
                    "Avoid dehumanizing or degrading interactions",
                ],
            },
        }

    async def _initialize_decision_framework(self):
        """Initialize ethical decision-making framework."""
        self.decision_framework = {
            "decision_process": [
                "identify_ethical_dimensions",
                "assess_principle_relevance",
                "evaluate_potential_consequences",
                "identify_stakeholders",
                "weigh_competing_principles",
                "consider_precedents",
                "make_decision",
                "document_reasoning",
                "monitor_outcomes",
            ],
            "conflict_resolution": {
                "principle_hierarchy": [
                    EthicalPrinciple.NON_MALEFICENCE,  # Highest priority
                    EthicalPrinciple.DIGNITY,
                    EthicalPrinciple.PRIVACY,
                    EthicalPrinciple.BENEFICENCE,
                    EthicalPrinciple.JUSTICE,
                    EthicalPrinciple.ACCOUNTABILITY,
                    EthicalPrinciple.AUTONOMY,
                    EthicalPrinciple.TRANSPARENCY,
                ],
                "context_considerations": [
                    "severity_of_consequences",
                    "number_of_people_affected",
                    "reversibility_of_decision",
                    "precedent_implications",
                    "cultural_context",
                ],
            },
            "decision_criteria": {
                "immediate_harm_prevention": 1.0,
                "long_term_benefit": 0.8,
                "principle_consistency": 0.7,
                "user_preference_alignment": 0.6,
                "social_good": 0.5,
            },
        }

    async def _initialize_safety_constraints(self):
        """Initialize safety constraints and red lines."""
        self.safety_constraints = {
            "absolute_prohibitions": [
                "physical_harm_instructions",
                "illegal_activity_assistance",
                "privacy_violations",
                "hate_speech_generation",
                "misinformation_creation",
                "manipulation_tactics",
                "exploitation_enablement",
            ],
            "conditional_restrictions": {
                "sensitive_topics": {
                    "condition": "requires_careful_handling",
                    "restrictions": [
                        "balanced_perspective",
                        "harm_consideration",
                        "resource_provision",
                    ],
                },
                "personal_advice": {
                    "condition": "significant_life_impact",
                    "restrictions": [
                        "professional_referral",
                        "limitation_acknowledgment",
                        "risk_disclosure",
                    ],
                },
                "controversial_subjects": {
                    "condition": "high_disagreement_potential",
                    "restrictions": [
                        "multiple_perspectives",
                        "nuance_acknowledgment",
                        "respectful_discourse",
                    ],
                },
            },
            "safety_escalation": {
                "immediate_intervention": [
                    "imminent_danger",
                    "illegal_activity",
                    "severe_harm_risk",
                ],
                "careful_review": ["ethical_dilemma", "value_conflict", "precedent_setting"],
                "monitoring_required": [
                    "sensitive_interaction",
                    "vulnerable_user",
                    "complex_situation",
                ],
            },
        }

    async def _initialize_ethical_learning(self):
        """Initialize ethical learning and adaptation systems."""
        self.ethical_learning_config = {
            "learning_enabled": True,
            "adaptation_rate": 0.1,
            "precedent_weight": 0.7,
            "user_feedback_weight": 0.3,
            "expert_input_weight": 0.9,
            "consistency_requirement": 0.8,
        }

        # Initialize learning mechanisms
        self.moral_insights = {
            "principle_effectiveness": {},
            "decision_patterns": {},
            "user_value_alignment": {},
            "contextual_adaptations": {},
        }

    async def evaluate_ethical_decision(
        self, decision_context: dict[str, Any], decision_type: EthicalDecisionType
    ) -> dict[str, Any]:
        """
        Evaluate an ethical decision using the guardian framework.

        Args:
            decision_context: Context and details of the decision
            decision_type: Type of ethical decision being made

        Returns:
            Ethical evaluation and recommendation
        """
        try:
            self.logger.info(f"Evaluating ethical decision: {decision_type.value}")

            # Create decision evaluation
            evaluation = {
                "decision_id": f"ethical_decision_{datetime.now().timestamp()}",
                "decision_type": decision_type.value,
                "context": decision_context,
                "timestamp": datetime.now().isoformat(),
                "ethical_analysis": {},
                "recommendation": {},
                "confidence": 0.0,
                "reasoning": [],
            }

            # Step 1: Identify ethical dimensions
            ethical_dimensions = await self._identify_ethical_dimensions(decision_context, decision_type)
            evaluation["ethical_analysis"]["dimensions"] = ethical_dimensions

            # Step 2: Assess principle relevance
            relevant_principles = await self._assess_principle_relevance(ethical_dimensions)
            evaluation["ethical_analysis"]["relevant_principles"] = relevant_principles

            # Step 3: Evaluate potential consequences
            consequences = await self._evaluate_consequences(decision_context, relevant_principles)
            evaluation["ethical_analysis"]["consequences"] = consequences

            # Step 4: Check for principle conflicts
            conflicts = await self._identify_principle_conflicts(relevant_principles, consequences)
            evaluation["ethical_analysis"]["conflicts"] = conflicts

            # Step 5: Apply decision framework
            decision_recommendation = await self._apply_decision_framework(
                ethical_dimensions, relevant_principles, consequences, conflicts
            )
            evaluation["recommendation"] = decision_recommendation

            # Step 6: Calculate confidence
            evaluation["confidence"] = self._calculate_decision_confidence(evaluation)

            # Step 7: Generate reasoning
            evaluation["reasoning"] = await self._generate_ethical_reasoning(evaluation)

            # Store decision for learning
            self.ethical_history.append(evaluation)

            # Update metrics
            self.ethical_metrics["decisions_made"] += 1
            if decision_recommendation["approved"] == False:
                self.ethical_metrics["ethical_violations_prevented"] += 1

            return evaluation

        except Exception as e:
            self.logger.error(f"Ethical decision evaluation failed: {e}")
            return {
                "decision_approved": False,
                "error": str(e),
                "safety_override": True,
                "reason": "Ethical evaluation system error - defaulting to safe mode",
            }

    async def _identify_ethical_dimensions(
        self, context: dict[str, Any], decision_type: EthicalDecisionType
    ) -> list[dict[str, Any]]:
        """Identify ethical dimensions present in the decision context."""
        dimensions = []

        # Analyze context for ethical considerations
        if "potential_harm" in context:
            dimensions.append(
                {
                    "dimension": "harm_prevention",
                    "severity": context["potential_harm"].get("severity", "medium"),
                    "description": "Potential for causing harm to users or others",
                }
            )

        if "personal_data" in context:
            dimensions.append(
                {
                    "dimension": "privacy_protection",
                    "sensitivity": context["personal_data"].get("sensitivity", "medium"),
                    "description": "Handling of personal or sensitive information",
                }
            )

        if "user_autonomy" in context:
            dimensions.append(
                {
                    "dimension": "autonomy_respect",
                    "conflict_level": context["user_autonomy"].get("conflict_level", "low"),
                    "description": "Respect for user choice and self-determination",
                }
            )

        if "fairness_implications" in context:
            dimensions.append(
                {
                    "dimension": "justice_considerations",
                    "impact_scope": context["fairness_implications"].get("scope", "individual"),
                    "description": "Fair and equitable treatment considerations",
                }
            )

        # Decision type specific dimensions
        if decision_type == EthicalDecisionType.RESPONSE_FILTERING:
            dimensions.append(
                {
                    "dimension": "content_appropriateness",
                    "content_type": context.get("content_type", "unknown"),
                    "description": "Appropriateness and safety of response content",
                }
            )

        return dimensions

    async def _assess_principle_relevance(self, ethical_dimensions: list[dict[str, Any]]) -> dict[str, float]:
        """Assess the relevance of each ethical principle to the decision."""
        principle_relevance = {}

        for principle, principle_data in self.core_principles.items():
            relevance_score = 0.0

            # Check if principle applies to any identified dimensions
            for dimension in ethical_dimensions:
                dimension_type = dimension["dimension"]

                # Map dimensions to principles
                if dimension_type == "harm_prevention" and principle == EthicalPrinciple.NON_MALEFICENCE:
                    relevance_score = max(relevance_score, 1.0)
                elif dimension_type == "privacy_protection" and principle == EthicalPrinciple.PRIVACY:
                    relevance_score = max(relevance_score, 1.0)
                elif dimension_type == "autonomy_respect" and principle == EthicalPrinciple.AUTONOMY:
                    relevance_score = max(relevance_score, 1.0)
                elif dimension_type == "justice_considerations" and principle == EthicalPrinciple.JUSTICE:
                    relevance_score = max(relevance_score, 1.0)
                elif dimension_type == "content_appropriateness":
                    if principle in [EthicalPrinciple.NON_MALEFICENCE, EthicalPrinciple.DIGNITY]:
                        relevance_score = max(relevance_score, 0.8)

            # Apply base principle weight
            principle_relevance[principle.value] = relevance_score * principle_data["weight"]

        return principle_relevance

    async def _evaluate_consequences(
        self, context: dict[str, Any], relevant_principles: dict[str, float]
    ) -> dict[str, Any]:
        """Evaluate potential consequences of the decision."""
        consequences = {
            "immediate_effects": [],
            "long_term_effects": [],
            "stakeholder_impacts": {},
            "risk_assessment": {},
        }

        # Analyze immediate effects
        if "immediate_impact" in context:
            consequences["immediate_effects"] = context["immediate_impact"]
        else:
            # Infer based on decision context
            consequences["immediate_effects"] = [
                "user_experience_change",
                "system_behavior_modification",
            ]

        # Analyze long-term effects
        consequences["long_term_effects"] = [
            "precedent_setting",
            "trust_relationship_impact",
            "value_alignment_evolution",
        ]

        # Assess stakeholder impacts
        consequences["stakeholder_impacts"] = {
            "primary_user": {"impact_level": "high", "impact_type": "direct"},
            "other_users": {"impact_level": "medium", "impact_type": "indirect"},
            "society": {"impact_level": "low", "impact_type": "systemic"},
        }

        # Risk assessment
        risk_level = "low"
        if any(score > 0.8 for score in relevant_principles.values()):
            risk_level = "high"
        elif any(score > 0.6 for score in relevant_principles.values()):
            risk_level = "medium"

        consequences["risk_assessment"] = {
            "overall_risk": risk_level,
            "harm_potential": context.get("harm_potential", "low"),
            "reversibility": context.get("reversibility", "high"),
        }

        return consequences

    async def _identify_principle_conflicts(
        self, relevant_principles: dict[str, float], consequences: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Identify conflicts between ethical principles."""
        conflicts = []

        # Find principles with high relevance
        high_relevance_principles = [principle for principle, score in relevant_principles.items() if score > 0.7]

        # Check for known conflicts
        for principle_name in high_relevance_principles:
            principle_enum = EthicalPrinciple(principle_name)
            principle_data = self.core_principles[principle_enum]

            for conflicting_principle in principle_data["conflicts_with"]:
                if conflicting_principle.value in high_relevance_principles:
                    conflicts.append(
                        {
                            "principle_1": principle_name,
                            "principle_2": conflicting_principle.value,
                            "conflict_type": "direct_opposition",
                            "severity": "medium",
                            "resolution_strategy": self._get_conflict_resolution_strategy(
                                principle_enum, conflicting_principle
                            ),
                        }
                    )

        return conflicts

    def _get_conflict_resolution_strategy(self, principle1: EthicalPrinciple, principle2: EthicalPrinciple) -> str:
        """Get strategy for resolving principle conflicts."""
        hierarchy = self.decision_framework["conflict_resolution"]["principle_hierarchy"]

        # Use principle hierarchy to resolve conflicts
        if principle1 in hierarchy and principle2 in hierarchy:
            p1_index = hierarchy.index(principle1)
            p2_index = hierarchy.index(principle2)

            if p1_index < p2_index:
                return f"prioritize_{principle1.value}_over_{principle2.value}"
            else:
                return f"prioritize_{principle2.value}_over_{principle1.value}"

        return "case_by_case_evaluation"

    async def _apply_decision_framework(
        self,
        ethical_dimensions: list[dict[str, Any]],
        relevant_principles: dict[str, float],
        consequences: dict[str, Any],
        conflicts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Apply the decision framework to make ethical recommendation."""
        decision = {
            "approved": True,
            "confidence": 0.0,
            "conditions": [],
            "modifications_required": [],
            "monitoring_required": False,
        }

        # Check for absolute prohibitions
        for dimension in ethical_dimensions:
            if dimension["dimension"] == "harm_prevention":
                if dimension.get("severity") in ["high", "critical"]:
                    decision["approved"] = False
                    decision["reason"] = "High risk of harm - safety override"
                    return decision

        # Apply safety constraints
        if consequences["risk_assessment"]["overall_risk"] == "high":
            if consequences["risk_assessment"]["harm_potential"] in ["high", "critical"]:
                decision["approved"] = False
                decision["reason"] = "Unacceptable harm potential"
                return decision

        # Resolve principle conflicts
        if conflicts:
            for conflict in conflicts:
                if conflict["severity"] == "high":
                    decision["conditions"].append(f"Resolve conflict: {conflict['resolution_strategy']}")

        # Apply principle weights
        principle_score = 0.0
        total_weight = 0.0

        for principle_name, relevance in relevant_principles.items():
            if relevance > 0:
                principle_enum = EthicalPrinciple(principle_name)
                weight = self.core_principles[principle_enum]["weight"]
                principle_score += relevance * weight
                total_weight += weight

        if total_weight > 0:
            decision["confidence"] = principle_score / total_weight

        # Determine if modifications are needed
        if decision["confidence"] < 0.7:
            decision["modifications_required"].append("Enhance ethical alignment")
            decision["monitoring_required"] = True

        # Final approval based on confidence and safety
        if decision["confidence"] < 0.5:
            decision["approved"] = False
            decision["reason"] = "Insufficient ethical confidence"

        return decision

    def _calculate_decision_confidence(self, evaluation: dict[str, Any]) -> float:
        """Calculate confidence in the ethical decision."""
        factors = []

        # Principle alignment confidence
        relevant_principles = evaluation["ethical_analysis"]["relevant_principles"]
        if relevant_principles:
            avg_relevance = sum(relevant_principles.values()) / len(relevant_principles)
            factors.append(avg_relevance)

        # Conflict resolution confidence
        conflicts = evaluation["ethical_analysis"]["conflicts"]
        conflict_penalty = len(conflicts) * 0.1
        factors.append(max(0.0, 1.0 - conflict_penalty))

        # Risk assessment confidence
        risk_level = evaluation["ethical_analysis"]["consequences"]["risk_assessment"]["overall_risk"]
        risk_confidence = {"low": 0.9, "medium": 0.7, "high": 0.4}.get(risk_level, 0.5)
        factors.append(risk_confidence)

        # Overall confidence
        return sum(factors) / len(factors) if factors else 0.5

    async def _generate_ethical_reasoning(self, evaluation: dict[str, Any]) -> list[str]:
        """Generate human-readable ethical reasoning."""
        reasoning = []

        # Explain relevant principles
        relevant_principles = evaluation["ethical_analysis"]["relevant_principles"]
        high_relevance = [p for p, score in relevant_principles.items() if score > 0.7]

        if high_relevance:
            reasoning.append(f"Key ethical principles: {', '.join(high_relevance)}")

        # Explain conflicts
        conflicts = evaluation["ethical_analysis"]["conflicts"]
        if conflicts:
            reasoning.append(f"Identified {len(conflicts)} principle conflicts requiring resolution")

        # Explain decision
        recommendation = evaluation["recommendation"]
        if recommendation["approved"]:
            reasoning.append("Decision approved with ethical safeguards")
        else:
            reasoning.append(f"Decision rejected: {recommendation.get('reason', 'Ethical concerns')}")

        # Explain confidence
        confidence = evaluation["confidence"]
        if confidence > 0.8:
            reasoning.append("High confidence in ethical assessment")
        elif confidence > 0.6:
            reasoning.append("Moderate confidence - monitoring recommended")
        else:
            reasoning.append("Low confidence - additional review required")

        return reasoning

    async def filter_content(self, content: str, content_type: str = "response") -> dict[str, Any]:
        """
        Filter content for ethical compliance using integrated systems.

        Args:
            content: Content to filter
            content_type: Type of content (response, query, etc.)

        Returns:
            Filtering result with recommendations
        """
        try:
            # Use existing ethical filter if available
            if self.ethical_filter:
                base_filter_result = self.ethical_filter.is_allowed_request(content)
                is_allowed, filter_details = base_filter_result
            else:
                # Fallback filtering
                is_allowed, filter_details = await self._fallback_content_filter(content)

            # Enhanced ethical analysis
            decision_context = {
                "content": content,
                "content_type": content_type,
                "base_filter_result": filter_details,
                "harm_potential": "medium" if not is_allowed else "low",
            }

            ethical_evaluation = await self.evaluate_ethical_decision(
                decision_context, EthicalDecisionType.RESPONSE_FILTERING
            )

            # Combine results
            result = {
                "content_approved": is_allowed and ethical_evaluation["recommendation"]["approved"],
                "base_filter_passed": is_allowed,
                "ethical_evaluation_passed": ethical_evaluation["recommendation"]["approved"],
                "confidence": ethical_evaluation["confidence"],
                "filtering_details": filter_details,
                "ethical_reasoning": ethical_evaluation["reasoning"],
                "modifications_suggested": ethical_evaluation["recommendation"].get("modifications_required", []),
            }

            # Generate filtered content if needed
            if not result["content_approved"] and self.ethical_filter:
                filtered_content = self.ethical_filter.filter_response(content, filter_details.get("violations", []))
                result["filtered_content"] = filtered_content

            return result

        except Exception as e:
            self.logger.error(f"Content filtering failed: {e}")
            return {
                "content_approved": False,
                "error": str(e),
                "safety_override": True,
                "filtered_content": "I apologize, but I cannot process this content due to safety concerns.",
            }

    async def _fallback_content_filter(self, content: str) -> tuple[bool, dict[str, Any]]:
        """Fallback content filtering when ethical_filter is not available."""
        # Simple keyword-based filtering
        harmful_keywords = [
            "kill",
            "destroy",
            "harm",
            "hate",
            "violence",
            "illegal",
            "dangerous",
            "toxic",
            "abuse",
            "exploit",
        ]

        content_lower = content.lower()
        violations = []

        for keyword in harmful_keywords:
            if keyword in content_lower:
                violations.append({"type": "harmful_content", "keyword": keyword, "severity": "medium"})

        is_allowed = len(violations) == 0

        return is_allowed, {"violations": violations, "filter_method": "fallback_keyword_filter"}

    def get_ethical_status(self) -> dict[str, Any]:
        """Get current ethical guardian status and metrics."""
        return {
            "ethical_guardian_status": "active",
            "guardian_config": self.guardian_config,
            "ethical_metrics": self.ethical_metrics,
            "core_principles": {
                principle.value: {"weight": data["weight"], "description": data["description"]}
                for principle, data in self.core_principles.items()
            },
            "recent_decisions": len(self.ethical_history),
            "ethical_filter_available": self.ethical_filter is not None,
            "safety_constraints_active": len(self.safety_constraints["absolute_prohibitions"]),
            "learning_enabled": self.guardian_config["learning_enabled"],
        }

    async def update_ethical_principles(self, principle_updates: dict[str, Any]) -> bool:
        """
        Update ethical principles based on learning and feedback.

        Args:
            principle_updates: Updates to principle weights or interpretations

        Returns:
            True if updates applied successfully
        """
        try:
            self.logger.info("Updating ethical principles...")

            for principle_name, updates in principle_updates.items():
                if principle_name in [p.value for p in EthicalPrinciple]:
                    principle_enum = EthicalPrinciple(principle_name)

                    if principle_enum in self.core_principles:
                        principle_data = self.core_principles[principle_enum]

                        # Update weight (with bounds checking)
                        if "weight" in updates:
                            new_weight = max(0.0, min(1.0, updates["weight"]))
                            principle_data["weight"] = new_weight

                        # Update implementation guidelines
                        if "guidelines" in updates:
                            principle_data["implementation_guidelines"].extend(updates["guidelines"])

                        self.logger.info(f"Updated principle: {principle_name}")

            # Record value evolution
            self.value_evolution.append(
                {
                    "timestamp": datetime.now().isoformat(),
                    "updates": principle_updates,
                    "reason": "adaptive_learning",
                }
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to update ethical principles: {e}")
            return False


# Global ethical guardian instance
ethical_guardian = EthicalGuardian()
