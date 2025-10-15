"""
Jeffrey OS Phase 0.8 - Dream Suggester
Proactive reformulation system that suggests improvements to proposals
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Optional dependencies with fallbacks
try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize, word_tokenize

    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Import Jeffrey OS components
from feedback.models import Decision, Proposal, ProposalType, RiskLevel, VerdictType


@dataclass
class RejectionPattern:
    """Pattern analysis for common rejection reasons"""

    reason: str
    keywords: list[str]
    frequency: int
    severity: float
    languages: list[str] = field(default_factory=lambda: ["en"])


@dataclass
class ReformulationSuggestion:
    """Suggestion for reformulating a proposal"""

    original_issue: str
    suggested_change: str
    confidence: float
    reasoning: str
    impact_prediction: float
    alternative_approaches: list[str] = field(default_factory=list)
    language: str = "en"


@dataclass
class ProactiveIntervention:
    """Proactive intervention recommendation"""

    trigger_reason: str
    intervention_type: str
    suggested_actions: list[str]
    urgency_level: str
    success_probability: float
    resources_needed: list[str] = field(default_factory=list)


class RejectionAnalyzer:
    """Analyzes rejection patterns to identify common failure modes"""

    def __init__(self):
        self.rejection_patterns = {}
        self.keyword_mappings = {
            "complexity": {
                "en": ["complex", "complicated", "difficult", "hard", "intricate"],
                "fr": ["complexe", "compliqué", "difficile", "dur", "intriqué"],
                "es": ["complejo", "complicado", "difícil", "duro", "intrincado"],
                "de": ["komplex", "kompliziert", "schwierig", "hart", "verschachtelt"],
                "it": ["complesso", "complicato", "difficile", "duro", "intricato"],
                "pt": ["complexo", "complicado", "difícil", "duro", "intrincado"],
            },
            "resources": {
                "en": ["expensive", "costly", "budget", "resource", "time", "money"],
                "fr": ["cher", "coûteux", "budget", "ressource", "temps", "argent"],
                "es": ["caro", "costoso", "presupuesto", "recurso", "tiempo", "dinero"],
                "de": ["teuer", "kostspielig", "Budget", "Ressource", "Zeit", "Geld"],
                "it": ["caro", "costoso", "budget", "risorsa", "tempo", "denaro"],
                "pt": ["caro", "custoso", "orçamento", "recurso", "tempo", "dinheiro"],
            },
            "risk": {
                "en": ["risky", "dangerous", "unsafe", "security", "vulnerability"],
                "fr": ["risqué", "dangereux", "pas sûr", "sécurité", "vulnérabilité"],
                "es": ["riesgoso", "peligroso", "inseguro", "seguridad", "vulnerabilidad"],
                "de": ["riskant", "gefährlich", "unsicher", "Sicherheit", "Verwundbarkeit"],
                "it": ["rischioso", "pericoloso", "insicuro", "sicurezza", "vulnerabilità"],
                "pt": ["arriscado", "perigoso", "inseguro", "segurança", "vulnerabilidade"],
            },
            "timing": {
                "en": ["too early", "too late", "wrong time", "schedule", "deadline"],
                "fr": ["trop tôt", "trop tard", "mauvais moment", "horaire", "échéance"],
                "es": ["muy temprano", "muy tarde", "mal momento", "horario", "fecha límite"],
                "de": ["zu früh", "zu spät", "falsche Zeit", "Zeitplan", "Stichtag"],
                "it": [
                    "troppo presto",
                    "troppo tardi",
                    "momento sbagliato",
                    "programma",
                    "scadenza",
                ],
                "pt": ["muito cedo", "muito tarde", "momento errado", "cronograma", "prazo"],
            },
            "unclear": {
                "en": ["unclear", "vague", "ambiguous", "confusing", "incomplete"],
                "fr": ["pas clair", "vague", "ambigu", "confus", "incomplet"],
                "es": ["poco claro", "vago", "ambiguo", "confuso", "incompleto"],
                "de": ["unklar", "vage", "mehrdeutig", "verwirrend", "unvollständig"],
                "it": ["poco chiaro", "vago", "ambiguo", "confuso", "incompleto"],
                "pt": ["pouco claro", "vago", "ambíguo", "confuso", "incompleto"],
            },
        }

    def analyze_rejection_reasons(
        self,
        proposals: list[Proposal],
        decisions: list[Decision],
        rationales: list[str],
        languages: list[str],
    ) -> list[RejectionPattern]:
        """Analyze rejection patterns from historical data"""
        rejection_data = []

        for proposal, decision, rationale, language in zip(proposals, decisions, rationales, languages, strict=False):
            if decision.verdict == VerdictType.REJECT:
                rejection_data.append({"proposal": proposal, "rationale": rationale, "language": language})

        # Extract patterns
        patterns = {}

        for data in rejection_data:
            rationale = data["rationale"].lower()
            language = data["language"]

            # Check for each pattern category
            for category, lang_keywords in self.keyword_mappings.items():
                keywords = lang_keywords.get(language, lang_keywords["en"])

                score = sum(1 for keyword in keywords if keyword in rationale)
                if score > 0:
                    pattern_key = f"{category}_{language}"

                    if pattern_key not in patterns:
                        patterns[pattern_key] = RejectionPattern(
                            reason=category,
                            keywords=keywords,
                            frequency=0,
                            severity=0.0,
                            languages=[language],
                        )

                    patterns[pattern_key].frequency += 1
                    patterns[pattern_key].severity += score / len(keywords)

        # Normalize severity scores
        for pattern in patterns.values():
            if pattern.frequency > 0:
                pattern.severity = pattern.severity / pattern.frequency

        return list(patterns.values())

    def predict_rejection_probability(self, proposal: Proposal, language: str = "en") -> dict[str, float]:
        """Predict probability of rejection for different reasons"""
        description = proposal.description.lower()
        detailed_plan = proposal.detailed_plan.lower()
        text = f"{description} {detailed_plan}"

        rejection_probs = {}

        for category, lang_keywords in self.keyword_mappings.items():
            keywords = lang_keywords.get(language, lang_keywords["en"])

            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text)

            # Calculate probability based on matches and historical patterns
            base_prob = matches / len(keywords) if keywords else 0

            # Apply risk level modifier
            risk_modifier = {
                RiskLevel.LOW: 0.8,
                RiskLevel.MEDIUM: 1.0,
                RiskLevel.HIGH: 1.3,
                RiskLevel.CRITICAL: 1.6,
            }.get(proposal.risk_level, 1.0)

            # Apply proposal type modifier
            type_modifier = {
                ProposalType.SECURITY: 0.7,  # Security proposals less likely to be rejected
                ProposalType.BUGFIX: 0.8,
                ProposalType.OPTIMIZATION: 1.0,
                ProposalType.FEATURE: 1.2,
            }.get(proposal.type, 1.0)

            rejection_probs[category] = min(base_prob * risk_modifier * type_modifier, 1.0)

        return rejection_probs


class ReformulationEngine:
    """Engine for generating reformulation suggestions"""

    def __init__(self):
        self.reformulation_templates = {
            "complexity": {
                "en": {
                    "analysis": "The proposal appears complex due to: {issues}",
                    "suggestions": [
                        "Break down into smaller, sequential phases",
                        "Provide clearer step-by-step implementation plan",
                        "Add specific success criteria for each phase",
                        "Include risk mitigation strategies",
                    ],
                },
                "fr": {
                    "analysis": "La proposition semble complexe en raison de: {issues}",
                    "suggestions": [
                        "Diviser en phases plus petites et séquentielles",
                        "Fournir un plan d'implémentation étape par étape plus clair",
                        "Ajouter des critères de succès spécifiques pour chaque phase",
                        "Inclure des stratégies d'atténuation des risques",
                    ],
                },
            },
            "resources": {
                "en": {
                    "analysis": "Resource concerns identified: {issues}",
                    "suggestions": [
                        "Provide detailed cost-benefit analysis",
                        "Suggest alternative low-cost approaches",
                        "Propose phased implementation to spread costs",
                        "Identify potential cost savings in other areas",
                    ],
                },
                "fr": {
                    "analysis": "Préoccupations concernant les ressources identifiées: {issues}",
                    "suggestions": [
                        "Fournir une analyse coûts-bénéfices détaillée",
                        "Suggérer des approches alternatives à faible coût",
                        "Proposer une implémentation par phases pour répartir les coûts",
                        "Identifier les économies potentielles dans d'autres domaines",
                    ],
                },
            },
            "risk": {
                "en": {
                    "analysis": "Risk factors detected: {issues}",
                    "suggestions": [
                        "Add comprehensive risk assessment",
                        "Include rollback procedures",
                        "Propose pilot testing approach",
                        "Detail monitoring and alerting systems",
                    ],
                },
                "fr": {
                    "analysis": "Facteurs de risque détectés: {issues}",
                    "suggestions": [
                        "Ajouter une évaluation complète des risques",
                        "Inclure des procédures de retour en arrière",
                        "Proposer une approche de test pilote",
                        "Détailler les systèmes de surveillance et d'alerte",
                    ],
                },
            },
            "timing": {
                "en": {
                    "analysis": "Timing concerns: {issues}",
                    "suggestions": [
                        "Provide detailed timeline with milestones",
                        "Justify why this timing is optimal",
                        "Consider alternative scheduling options",
                        "Address resource availability constraints",
                    ],
                },
                "fr": {
                    "analysis": "Préoccupations de timing: {issues}",
                    "suggestions": [
                        "Fournir un calendrier détaillé avec des jalons",
                        "Justifier pourquoi ce timing est optimal",
                        "Considérer des options de planification alternatives",
                        "Aborder les contraintes de disponibilité des ressources",
                    ],
                },
            },
            "unclear": {
                "en": {
                    "analysis": "Clarity issues found: {issues}",
                    "suggestions": [
                        "Add more specific technical details",
                        "Include concrete examples and use cases",
                        "Define all technical terms and acronyms",
                        "Provide visual diagrams if applicable",
                    ],
                },
                "fr": {
                    "analysis": "Problèmes de clarté trouvés: {issues}",
                    "suggestions": [
                        "Ajouter des détails techniques plus spécifiques",
                        "Inclure des exemples concrets et des cas d'usage",
                        "Définir tous les termes techniques et acronymes",
                        "Fournir des diagrammes visuels si applicable",
                    ],
                },
            },
        }

        self.sentiment_analyzer = None
        if TEXTBLOB_AVAILABLE:
            self.sentiment_analyzer = TextBlob

        self.text_generator = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.text_generator = pipeline("text-generation", model="gpt2")
            except:
                pass

    def generate_suggestions(
        self, proposal: Proposal, rejection_probs: dict[str, float], language: str = "en"
    ) -> list[ReformulationSuggestion]:
        """Generate reformulation suggestions based on rejection probabilities"""
        suggestions = []

        # Sort rejection probabilities by likelihood
        sorted_probs = sorted(rejection_probs.items(), key=lambda x: x[1], reverse=True)

        for reason, probability in sorted_probs:
            if probability > 0.3:  # Only suggest for significant risks
                suggestion = self._create_reformulation_suggestion(proposal, reason, probability, language)
                suggestions.append(suggestion)

        return suggestions

    def _create_reformulation_suggestion(
        self, proposal: Proposal, reason: str, probability: float, language: str
    ) -> ReformulationSuggestion:
        """Create a specific reformulation suggestion"""
        templates = self.reformulation_templates.get(reason, {})
        lang_templates = templates.get(language, templates.get("en", {}))

        if not lang_templates:
            # Fallback generic suggestion
            return ReformulationSuggestion(
                original_issue=f"Potential {reason} concerns",
                suggested_change=f"Address {reason} factors in the proposal",
                confidence=0.5,
                reasoning=f"Based on {reason} analysis",
                impact_prediction=0.3,
                language=language,
            )

        # Identify specific issues
        issues = self._identify_specific_issues(proposal, reason, language)

        # Select appropriate suggestions
        suggestions = lang_templates.get("suggestions", [])
        selected_suggestions = suggestions[:2]  # Top 2 suggestions

        # Calculate confidence based on probability and specificity
        confidence = min(probability * 0.8, 0.9)

        # Predict impact of applying suggestion
        impact_prediction = self._predict_suggestion_impact(proposal, reason, probability)

        return ReformulationSuggestion(
            original_issue=lang_templates.get("analysis", "").format(issues=", ".join(issues)),
            suggested_change="; ".join(selected_suggestions),
            confidence=confidence,
            reasoning=f"Analysis shows {probability:.1%} probability of {reason}-related rejection",
            impact_prediction=impact_prediction,
            alternative_approaches=suggestions[2:] if len(suggestions) > 2 else [],
            language=language,
        )

    def _identify_specific_issues(self, proposal: Proposal, reason: str, language: str) -> list[str]:
        """Identify specific issues in the proposal"""
        issues = []

        text = f"{proposal.description} {proposal.detailed_plan}".lower()

        # Look for specific indicators based on reason
        if reason == "complexity":
            if "multiple" in text or "several" in text:
                issues.append("multiple components")
            if "integration" in text:
                issues.append("system integration")
            if "dependency" in text or "dependencies" in text:
                issues.append("external dependencies")

        elif reason == "resources":
            if "new" in text and ("hire" in text or "team" in text):
                issues.append("additional personnel needed")
            if "hardware" in text or "infrastructure" in text:
                issues.append("infrastructure requirements")
            if "license" in text or "software" in text:
                issues.append("software licensing costs")

        elif reason == "risk":
            if "data" in text and ("delete" in text or "modify" in text):
                issues.append("data modification risks")
            if "production" in text or "live" in text:
                issues.append("production environment risks")
            if "security" in text:
                issues.append("security implications")

        elif reason == "timing":
            if "urgent" in text or "asap" in text:
                issues.append("rushed timeline")
            if "deadline" in text:
                issues.append("tight deadlines")
            if "busy" in text or "overloaded" in text:
                issues.append("resource availability")

        elif reason == "unclear":
            if len(proposal.description.split()) < 20:
                issues.append("insufficient detail")
            if "?" in text:
                issues.append("unresolved questions")
            if "tbd" in text or "todo" in text:
                issues.append("incomplete planning")

        return issues if issues else ["general concerns"]

    def _predict_suggestion_impact(self, proposal: Proposal, reason: str, probability: float) -> float:
        """Predict impact of applying suggestion"""
        # Base impact inversely related to rejection probability
        base_impact = 1.0 - (probability * 0.7)

        # Adjust based on proposal characteristics
        if proposal.type == ProposalType.SECURITY:
            base_impact *= 1.2  # Security improvements have higher impact
        elif proposal.type == ProposalType.FEATURE:
            base_impact *= 0.9  # Feature requests more variable

        # Adjust based on risk level
        risk_adjustment = {
            RiskLevel.LOW: 1.0,
            RiskLevel.MEDIUM: 0.9,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 0.7,
        }.get(proposal.risk_level, 1.0)

        return min(base_impact * risk_adjustment, 1.0)

    def enhance_proposal_text(
        self, proposal: Proposal, suggestions: list[ReformulationSuggestion], language: str = "en"
    ) -> str:
        """Generate enhanced proposal text incorporating suggestions"""
        enhanced_sections = []

        # Original description
        enhanced_sections.append(f"**Original Description:**\n{proposal.description}")

        # Add enhanced sections based on suggestions
        for suggestion in suggestions:
            if suggestion.confidence > 0.6:
                enhanced_sections.append(f"\n**Addressing {suggestion.original_issue}:**")
                enhanced_sections.append(suggestion.suggested_change)

        # Add implementation timeline if timing concerns
        timing_suggestions = [s for s in suggestions if "timing" in s.original_issue.lower()]
        if timing_suggestions:
            enhanced_sections.append("\n**Implementation Timeline:**")
            enhanced_sections.append("Phase 1: Planning and preparation (2 weeks)")
            enhanced_sections.append("Phase 2: Core implementation (4 weeks)")
            enhanced_sections.append("Phase 3: Testing and validation (2 weeks)")
            enhanced_sections.append("Phase 4: Deployment and monitoring (1 week)")

        # Add risk mitigation if risk concerns
        risk_suggestions = [s for s in suggestions if "risk" in s.original_issue.lower()]
        if risk_suggestions:
            enhanced_sections.append("\n**Risk Mitigation:**")
            enhanced_sections.append("- Comprehensive testing in staging environment")
            enhanced_sections.append("- Rollback procedures documented and tested")
            enhanced_sections.append("- Monitoring and alerting systems in place")
            enhanced_sections.append("- Regular checkpoints and go/no-go decisions")

        return "\n".join(enhanced_sections)


class DreamSuggester:
    """
    Proactive system for reformulating proposals to increase acceptance probability
    """

    def __init__(self, data_dir: str = "data/learning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.rejection_analyzer = RejectionAnalyzer()
        self.reformulation_engine = ReformulationEngine()

        # Learning components
        self.historical_patterns = {}
        self.success_tracking = {}

        # Database for persistent storage
        self.db_path = self.data_dir / "dream_suggestions.db"
        self._init_database()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize database for suggestion tracking"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Suggestions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS suggestions (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT,
                    original_issue TEXT,
                    suggested_change TEXT,
                    confidence REAL,
                    impact_prediction REAL,
                    language TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    applied BOOLEAN DEFAULT FALSE,
                    outcome TEXT
                )
            """
            )

            # Interventions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS interventions (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT,
                    trigger_reason TEXT,
                    intervention_type TEXT,
                    urgency_level TEXT,
                    success_probability REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    executed BOOLEAN DEFAULT FALSE,
                    result TEXT
                )
            """
            )

            # Pattern learning table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT,
                    pattern_data TEXT,
                    success_rate REAL,
                    usage_count INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()

    def learn_from_history(
        self,
        proposals: list[Proposal],
        decisions: list[Decision],
        rationales: list[str],
        languages: list[str],
    ):
        """Learn rejection patterns from historical data"""
        self.logger.info(f"Learning from {len(proposals)} historical proposals")

        # Analyze rejection patterns
        patterns = self.rejection_analyzer.analyze_rejection_reasons(proposals, decisions, rationales, languages)

        # Store patterns
        for pattern in patterns:
            self.historical_patterns[pattern.reason] = pattern

        # Store in database
        self._store_learned_patterns(patterns)

        self.logger.info(f"Learned {len(patterns)} rejection patterns")

    def suggest_reformulation(self, proposal: Proposal, language: str = "en") -> list[ReformulationSuggestion]:
        """
        Suggest reformulations for a proposal to increase acceptance probability
        """
        self.logger.info(f"Generating reformulation suggestions for proposal {proposal.id}")

        # Predict rejection probabilities
        rejection_probs = self.rejection_analyzer.predict_rejection_probability(proposal, language)

        # Generate suggestions
        suggestions = self.reformulation_engine.generate_suggestions(proposal, rejection_probs, language)

        # Store suggestions in database
        self._store_suggestions(proposal.id, suggestions)

        # Filter by confidence threshold
        high_confidence_suggestions = [s for s in suggestions if s.confidence > 0.5]

        self.logger.info(f"Generated {len(high_confidence_suggestions)} high-confidence suggestions")

        return high_confidence_suggestions

    def proactive_intervention(
        self, proposal: Proposal, predicted_rejection_reason: str, language: str = "en"
    ) -> ProactiveIntervention:
        """
        Generate proactive intervention when rejection is likely
        """
        self.logger.info(f"Generating proactive intervention for {predicted_rejection_reason}")

        # Determine intervention type
        intervention_type = self._determine_intervention_type(predicted_rejection_reason)

        # Generate specific actions
        actions = self._generate_intervention_actions(proposal, predicted_rejection_reason, language)

        # Calculate success probability
        success_prob = self._calculate_intervention_success_probability(proposal, predicted_rejection_reason)

        # Determine urgency
        urgency = self._determine_urgency(proposal, predicted_rejection_reason)

        # Identify needed resources
        resources = self._identify_needed_resources(intervention_type, proposal)

        intervention = ProactiveIntervention(
            trigger_reason=predicted_rejection_reason,
            intervention_type=intervention_type,
            suggested_actions=actions,
            urgency_level=urgency,
            success_probability=success_prob,
            resources_needed=resources,
        )

        # Store intervention
        self._store_intervention(proposal.id, intervention)

        return intervention

    def _determine_intervention_type(self, rejection_reason: str) -> str:
        """Determine the type of intervention needed"""
        intervention_map = {
            "complexity": "simplification",
            "resources": "cost_optimization",
            "risk": "risk_mitigation",
            "timing": "schedule_adjustment",
            "unclear": "clarification",
        }

        return intervention_map.get(rejection_reason, "general_improvement")

    def _generate_intervention_actions(self, proposal: Proposal, reason: str, language: str) -> list[str]:
        """Generate specific intervention actions"""
        action_templates = {
            "complexity": {
                "en": [
                    "Break proposal into smaller, independent phases",
                    "Create detailed technical specification document",
                    "Identify and document all dependencies",
                    "Propose prototype or proof-of-concept first",
                ],
                "fr": [
                    "Diviser la proposition en phases plus petites et indépendantes",
                    "Créer un document de spécification technique détaillé",
                    "Identifier et documenter toutes les dépendances",
                    "Proposer d'abord un prototype ou une preuve de concept",
                ],
            },
            "resources": {
                "en": [
                    "Prepare detailed cost-benefit analysis",
                    "Identify alternative low-cost solutions",
                    "Propose phased implementation to spread costs",
                    "Research potential funding sources or budget reallocation",
                ],
                "fr": [
                    "Préparer une analyse coûts-bénéfices détaillée",
                    "Identifier des solutions alternatives à faible coût",
                    "Proposer une implémentation par phases pour répartir les coûts",
                    "Rechercher des sources de financement ou réallocation budgétaire",
                ],
            },
            "risk": {
                "en": [
                    "Conduct comprehensive risk assessment",
                    "Develop detailed rollback procedures",
                    "Create monitoring and alerting systems",
                    "Establish clear success/failure criteria",
                ],
                "fr": [
                    "Effectuer une évaluation complète des risques",
                    "Développer des procédures de retour en arrière détaillées",
                    "Créer des systèmes de surveillance et d'alerte",
                    "Établir des critères de succès/échec clairs",
                ],
            },
            "timing": {
                "en": [
                    "Create detailed project timeline with milestones",
                    "Identify critical path and potential bottlenecks",
                    "Prepare contingency plans for delays",
                    "Coordinate with stakeholders on resource availability",
                ],
                "fr": [
                    "Créer un calendrier de projet détaillé avec jalons",
                    "Identifier le chemin critique et les goulots d'étranglement potentiels",
                    "Préparer des plans de contingence pour les retards",
                    "Coordonner avec les parties prenantes sur la disponibilité des ressources",
                ],
            },
            "unclear": {
                "en": [
                    "Expand technical details and specifications",
                    "Add concrete examples and use cases",
                    "Create visual diagrams and flowcharts",
                    "Define all technical terms and acronyms",
                ],
                "fr": [
                    "Étendre les détails techniques et spécifications",
                    "Ajouter des exemples concrets et cas d'usage",
                    "Créer des diagrammes visuels et organigrammes",
                    "Définir tous les termes techniques et acronymes",
                ],
            },
        }

        templates = action_templates.get(reason, {})
        actions = templates.get(language, templates.get("en", []))

        return actions

    def _calculate_intervention_success_probability(self, proposal: Proposal, reason: str) -> float:
        """Calculate probability of intervention success"""
        # Base success rate varies by reason
        base_rates = {
            "complexity": 0.7,
            "resources": 0.6,
            "risk": 0.8,
            "timing": 0.5,
            "unclear": 0.9,
        }

        base_rate = base_rates.get(reason, 0.6)

        # Adjust based on proposal characteristics
        if proposal.type == ProposalType.SECURITY:
            base_rate *= 1.2  # Security proposals easier to justify
        elif proposal.type == ProposalType.FEATURE:
            base_rate *= 0.9  # Feature requests more challenging

        # Adjust based on risk level
        risk_adjustment = {
            RiskLevel.LOW: 1.1,
            RiskLevel.MEDIUM: 1.0,
            RiskLevel.HIGH: 0.9,
            RiskLevel.CRITICAL: 0.8,
        }.get(proposal.risk_level, 1.0)

        return min(base_rate * risk_adjustment, 1.0)

    def _determine_urgency(self, proposal: Proposal, reason: str) -> str:
        """Determine intervention urgency level"""
        # Risk-based urgency
        if proposal.risk_level == RiskLevel.CRITICAL:
            return "high"
        elif proposal.risk_level == RiskLevel.HIGH:
            return "medium"

        # Reason-based urgency
        urgent_reasons = ["risk", "timing"]
        if reason in urgent_reasons:
            return "medium"

        return "low"

    def _identify_needed_resources(self, intervention_type: str, proposal: Proposal) -> list[str]:
        """Identify resources needed for intervention"""
        resource_map = {
            "simplification": ["technical_writer", "architect", "time"],
            "cost_optimization": ["financial_analyst", "project_manager", "research_time"],
            "risk_mitigation": ["security_expert", "testing_resources", "monitoring_tools"],
            "schedule_adjustment": ["project_manager", "stakeholder_time", "planning_tools"],
            "clarification": ["technical_writer", "domain_expert", "documentation_tools"],
        }

        return resource_map.get(intervention_type, ["project_manager", "time"])

    def _store_suggestions(self, proposal_id: str, suggestions: list[ReformulationSuggestion]):
        """Store suggestions in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for suggestion in suggestions:
                cursor.execute(
                    """
                    INSERT INTO suggestions (
                        id, proposal_id, original_issue, suggested_change,
                        confidence, impact_prediction, language
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        f"sug_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{proposal_id}",
                        proposal_id,
                        suggestion.original_issue,
                        suggestion.suggested_change,
                        suggestion.confidence,
                        suggestion.impact_prediction,
                        suggestion.language,
                    ),
                )

            conn.commit()

    def _store_intervention(self, proposal_id: str, intervention: ProactiveIntervention):
        """Store intervention in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO interventions (
                    id, proposal_id, trigger_reason, intervention_type,
                    urgency_level, success_probability
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    f"int_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{proposal_id}",
                    proposal_id,
                    intervention.trigger_reason,
                    intervention.intervention_type,
                    intervention.urgency_level,
                    intervention.success_probability,
                ),
            )

            conn.commit()

    def _store_learned_patterns(self, patterns: list[RejectionPattern]):
        """Store learned patterns in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for pattern in patterns:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO learned_patterns (
                        id, pattern_type, pattern_data, success_rate
                    ) VALUES (?, ?, ?, ?)
                """,
                    (
                        f"pat_{pattern.reason}_{datetime.now().strftime('%Y%m%d')}",
                        pattern.reason,
                        json.dumps(
                            {
                                "keywords": pattern.keywords,
                                "frequency": pattern.frequency,
                                "severity": pattern.severity,
                                "languages": pattern.languages,
                            }
                        ),
                        1.0 - (pattern.severity * 0.5),  # Inverse relationship
                    ),
                )

            conn.commit()

    def get_suggestion_effectiveness(self) -> dict[str, Any]:
        """Get effectiveness metrics for suggestions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get suggestion statistics
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_suggestions,
                    SUM(CASE WHEN applied THEN 1 ELSE 0 END) as applied_suggestions,
                    AVG(confidence) as avg_confidence,
                    AVG(impact_prediction) as avg_impact
                FROM suggestions
            """
            )

            suggestion_stats = cursor.fetchone()

            # Get intervention statistics
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_interventions,
                    SUM(CASE WHEN executed THEN 1 ELSE 0 END) as executed_interventions,
                    AVG(success_probability) as avg_success_prob
                FROM interventions
            """
            )

            intervention_stats = cursor.fetchone()

            return {
                "total_suggestions": suggestion_stats[0] if suggestion_stats else 0,
                "applied_suggestions": suggestion_stats[1] if suggestion_stats else 0,
                "suggestion_application_rate": (
                    suggestion_stats[1] / suggestion_stats[0] if suggestion_stats and suggestion_stats[0] > 0 else 0
                ),
                "average_confidence": suggestion_stats[2] if suggestion_stats else 0,
                "average_impact_prediction": suggestion_stats[3] if suggestion_stats else 0,
                "total_interventions": intervention_stats[0] if intervention_stats else 0,
                "executed_interventions": intervention_stats[1] if intervention_stats else 0,
                "intervention_execution_rate": (
                    intervention_stats[1] / intervention_stats[0]
                    if intervention_stats and intervention_stats[0] > 0
                    else 0
                ),
                "average_success_probability": intervention_stats[2] if intervention_stats else 0,
            }

    def update_suggestion_outcome(self, suggestion_id: str, outcome: str, applied: bool = True):
        """Update outcome of a suggestion"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                UPDATE suggestions
                SET applied = ?, outcome = ?
                WHERE id = ?
            """,
                (applied, outcome, suggestion_id),
            )

            conn.commit()

    def dream_mode_analysis(self, proposal: Proposal, language: str = "en") -> dict[str, Any]:
        """
        Complete dream mode analysis combining all capabilities
        """
        self.logger.info(f"Running dream mode analysis for proposal {proposal.id}")

        # Step 1: Predict rejection probabilities
        rejection_probs = self.rejection_analyzer.predict_rejection_probability(proposal, language)

        # Step 2: Generate reformulation suggestions
        suggestions = self.reformulation_engine.generate_suggestions(proposal, rejection_probs, language)

        # Step 3: Identify highest risk factor
        highest_risk = max(rejection_probs, key=rejection_probs.get)
        highest_risk_prob = rejection_probs[highest_risk]

        # Step 4: Generate intervention if needed
        intervention = None
        if highest_risk_prob > 0.6:  # High risk threshold
            intervention = self.proactive_intervention(proposal, highest_risk, language)

        # Step 5: Generate enhanced proposal text
        enhanced_text = self.reformulation_engine.enhance_proposal_text(proposal, suggestions, language)

        # Step 6: Calculate overall acceptance probability improvement
        improvement_estimate = self._calculate_acceptance_improvement(rejection_probs, suggestions)

        return {
            "rejection_probabilities": rejection_probs,
            "reformulation_suggestions": suggestions,
            "proactive_intervention": intervention,
            "enhanced_proposal_text": enhanced_text,
            "acceptance_probability_improvement": improvement_estimate,
            "highest_risk_factor": highest_risk,
            "recommendation": self._generate_recommendation(
                highest_risk_prob, len(suggestions), intervention is not None
            ),
        }

    def _calculate_acceptance_improvement(
        self, rejection_probs: dict[str, float], suggestions: list[ReformulationSuggestion]
    ) -> float:
        """Calculate estimated improvement in acceptance probability"""
        total_rejection_risk = sum(rejection_probs.values()) / len(rejection_probs)

        # Calculate mitigation from suggestions
        mitigation_effect = 0
        for suggestion in suggestions:
            mitigation_effect += suggestion.confidence * suggestion.impact_prediction

        # Normalize mitigation effect
        mitigation_effect = min(mitigation_effect / len(suggestions) if suggestions else 0, 0.7)

        # Calculate improvement
        improvement = (1 - total_rejection_risk) * mitigation_effect

        return min(improvement, 0.8)  # Cap at 80% improvement

    def _generate_recommendation(self, highest_risk_prob: float, suggestion_count: int, has_intervention: bool) -> str:
        """Generate overall recommendation"""
        if highest_risk_prob > 0.8:
            return "HIGH RISK: Significant reformulation strongly recommended before submission"
        elif highest_risk_prob > 0.6:
            return "MEDIUM RISK: Apply suggested improvements to increase acceptance probability"
        elif highest_risk_prob > 0.4:
            return "LOW RISK: Minor improvements suggested, but proposal likely acceptable"
        else:
            return "GOOD: Proposal appears well-structured with high acceptance probability"
