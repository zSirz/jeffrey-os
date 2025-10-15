"""
Implémentation de la théorie de l'esprit.

Ce module implémente les fonctionnalités essentielles pour implémentation de la théorie de l'esprit.
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

import json
import logging
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Optional dependencies
try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import LatentDirichletAllocation
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Import Jeffrey OS components
# from feedback.models import Decision, Proposal, ProposalType, RiskLevel, VerdictType


@dataclass
class ConceptCluster:
    """Represents a discovered concept cluster"""

    name: str
    keywords: list[str]
    frequency: int
    associated_verdicts: dict[str, int]
    emotional_valence: float
    confidence: float
    languages: set[str] = field(default_factory=set)
    examples: list[str] = field(default_factory=list)


@dataclass
class DecisionPattern:
    """Represents a decision-making pattern"""

    pattern_id: str
    trigger_conditions: dict[str, Any]
    typical_response: VerdictType
    confidence_level: float
    temporal_factors: dict[str, Any]
    contextual_factors: dict[str, Any]
    frequency: int


@dataclass
class MentalState:
    """Represents inferred mental state"""

    complexity_tolerance: float
    timing_preference: str
    risk_profile: str
    cognitive_load: float
    emotional_state: str
    decision_fatigue: float
    confidence_level: float
    likely_friction_points: list[str]


class ConceptExtractor:
    """Extracts fundamental concepts from rationales"""

    def __init__(self) -> None:
        self.concept_patterns = {
            "complexity": {
                "en": ["complex", "complicated", "difficult", "hard", "challenging", "intricate"],
                "fr": ["complexe", "compliqué", "difficile", "dur", "délicat", "intricaqué"],
                "es": ["complejo", "complicado", "difícil", "duro", "desafiante"],
                "de": ["komplex", "kompliziert", "schwierig", "hart", "herausfordernd"],
                "it": ["complesso", "complicato", "difficile", "duro", "impegnativo"],
                "pt": ["complexo", "complicado", "difícil", "duro", "desafiador"],
            },
            "timing": {
                "en": ["time", "deadline", "urgent", "quick", "slow", "delay", "schedule"],
                "fr": ["temps", "délai", "urgent", "rapide", "lent", "retard", "horaire"],
                "es": ["tiempo", "plazo", "urgente", "rápido", "lento", "retraso"],
                "de": ["Zeit", "Frist", "dringend", "schnell", "langsam", "Verzögerung"],
                "it": ["tempo", "scadenza", "urgente", "veloce", "lento", "ritardo"],
                "pt": ["tempo", "prazo", "urgente", "rápido", "lento", "atraso"],
            },
            "risk": {
                "en": ["risk", "dangerous", "safe", "secure", "unsafe", "hazard", "threat"],
                "fr": ["risque", "dangereux", "sûr", "sécurisé", "dangereux", "menace"],
                "es": ["riesgo", "peligroso", "seguro", "seguro", "peligroso", "amenaza"],
                "de": ["Risiko", "gefährlich", "sicher", "sicher", "unsicher", "Bedrohung"],
                "it": ["rischio", "pericoloso", "sicuro", "sicuro", "insicuro", "minaccia"],
                "pt": ["risco", "perigoso", "seguro", "seguro", "inseguro", "ameaça"],
            },
            "resources": {
                "en": ["cost", "expensive", "cheap", "budget", "money", "resource", "effort"],
                "fr": ["coût", "cher", "pas cher", "budget", "argent", "ressource", "effort"],
                "es": ["costo", "caro", "barato", "presupuesto", "dinero", "recurso", "esfuerzo"],
                "de": ["Kosten", "teuer", "billig", "Budget", "Geld", "Ressource", "Aufwand"],
                "it": ["costo", "costoso", "economico", "budget", "denaro", "risorsa", "sforzo"],
                "pt": ["custo", "caro", "barato", "orçamento", "dinheiro", "recurso", "esforço"],
            },
            "impact": {
                "en": ["impact", "benefit", "advantage", "improvement", "gain", "value"],
                "fr": ["impact", "bénéfice", "avantage", "amélioration", "gain", "valeur"],
                "es": ["impacto", "beneficio", "ventaja", "mejora", "ganancia", "valor"],
                "de": ["Auswirkung", "Nutzen", "Vorteil", "Verbesserung", "Gewinn", "Wert"],
                "it": ["impatto", "beneficio", "vantaggio", "miglioramento", "guadagno", "valore"],
                "pt": ["impacto", "benefício", "vantagem", "melhoria", "ganho", "valor"],
            },
        }

        # Emotional valence indicators
        self.emotional_indicators = {
            "positive": {
                "en": ["good", "great", "excellent", "love", "like", "happy", "excited"],
                "fr": ["bon", "super", "excellent", "aimer", "content", "heureux", "excité"],
                "es": ["bueno", "genial", "excelente", "amor", "gustar", "feliz", "emocionado"],
                "de": ["gut", "toll", "ausgezeichnet", "lieben", "mögen", "glücklich", "aufgeregt"],
                "it": [
                    "buono",
                    "fantastico",
                    "eccellente",
                    "amare",
                    "piacere",
                    "felice",
                    "eccitato",
                ],
                "pt": ["bom", "ótimo", "excelente", "amar", "gostar", "feliz", "animado"],
            },
            "negative": {
                "en": ["bad", "terrible", "awful", "hate", "dislike", "worried", "concerned"],
                "fr": ["mauvais", "terrible", "affreux", "détester", "inquiet", "préoccupé"],
                "es": ["malo", "terrible", "horrible", "odiar", "preocupado", "preocupado"],
                "de": ["schlecht", "schrecklich", "furchtbar", "hassen", "besorgt", "besorgt"],
                "it": ["cattivo", "terribile", "orribile", "odiare", "preoccupato", "preoccupato"],
                "pt": ["ruim", "terrível", "horrível", "odiar", "preocupado", "preocupado"],
            },
        }

    def extract_concepts(self, rationale: str, language: str = "en") -> dict[str, float]:
        """Extract concepts from rationale text"""
        concepts = {}
        rationale_lower = rationale.lower()

        for concept, patterns in self.concept_patterns.items():
            lang_patterns = patterns.get(language, patterns.get("en", []))

            # Count occurrences of concept keywords
            count = sum(1 for keyword in lang_patterns if keyword in rationale_lower)
            concepts[concept] = count / len(lang_patterns) if lang_patterns else 0

        return concepts

    def analyze_emotional_valence(self, rationale: str, language: str = "en") -> float:
        """Analyze emotional valence of rationale"""
        rationale_lower = rationale.lower()

        positive_patterns = self.emotional_indicators["positive"].get(language, [])
        negative_patterns = self.emotional_indicators["negative"].get(language, [])

        positive_count = sum(1 for keyword in positive_patterns if keyword in rationale_lower)
        negative_count = sum(1 for keyword in negative_patterns if keyword in rationale_lower)

        total_count = positive_count + negative_count
        if total_count == 0:
            return 0.0

        return (positive_count - negative_count) / total_count


class PatternAnalyzer:
    """Analyzes decision patterns and behaviors"""

    def __init__(self) -> None:
        self.temporal_patterns = {}
        self.contextual_patterns = {}
        self.sequence_patterns = {}

    def analyze_temporal_patterns(self, decisions: list[Decision]) -> dict[str, Any]:
        """Analyze temporal decision patterns"""
        patterns = {
            "hour_preferences": defaultdict(list),
            "day_preferences": defaultdict(list),
            "speed_patterns": defaultdict(list),
            "fatigue_indicators": [],
        }

        for decision in decisions:
            hour = decision.timestamp.hour
            day = decision.timestamp.weekday()

            patterns["hour_preferences"][hour].append(decision.verdict.value)
            patterns["day_preferences"][day].append(decision.verdict.value)
            patterns["speed_patterns"][decision.verdict.value].append(decision.review_time_seconds)

            # Detect potential fatigue (very fast decisions)
            if decision.review_time_seconds < 10:
                patterns["fatigue_indicators"].append(
                    {
                        "timestamp": decision.timestamp,
                        "review_time": decision.review_time_seconds,
                        "verdict": decision.verdict.value,
                    }
                )

        return patterns

    def analyze_contextual_patterns(self, proposals: list[Proposal], decisions: list[Decision]) -> dict[str, Any]:
        """Analyze contextual decision patterns"""
        patterns = {
            "type_preferences": defaultdict(list),
            "risk_tolerance": defaultdict(list),
            "complexity_handling": defaultdict(list),
            "impact_sensitivity": defaultdict(list),
        }

        for proposal, decision in zip(proposals, decisions):
            verdict = decision.verdict.value

            patterns["type_preferences"][proposal.type.value].append(verdict)
            patterns["risk_tolerance"][proposal.risk_level.value].append(verdict)
            patterns["complexity_handling"][len(proposal.description)].append(verdict)
            patterns["impact_sensitivity"][proposal.impact_score].append(verdict)

        return patterns

    def detect_decision_sequences(self, decisions: list[Decision]) -> list[dict[str, Any]]:
        """Detect sequential decision patterns"""
        sequences = []

        # Look for sequences of similar decisions
        current_sequence = []

        for i, decision in enumerate(decisions):
            if not current_sequence:
                current_sequence.append(decision)
            elif decision.verdict == current_sequence[-1].verdict:
                current_sequence.append(decision)
            else:
                # End of sequence
                if len(current_sequence) >= 3:
                    sequences.append(
                        {
                            "verdict": current_sequence[0].verdict.value,
                            "length": len(current_sequence),
                            "start_time": current_sequence[0].timestamp,
                            "end_time": current_sequence[-1].timestamp,
                            "avg_review_time": np.mean([d.review_time_seconds for d in current_sequence]),
                        }
                    )

                current_sequence = [decision]

        # Handle final sequence
        if len(current_sequence) >= 3:
            sequences.append(
                {
                    "verdict": current_sequence[0].verdict.value,
                    "length": len(current_sequence),
                    "start_time": current_sequence[0].timestamp,
                    "end_time": current_sequence[-1].timestamp,
                    "avg_review_time": np.mean([d.review_time_seconds for d in current_sequence]),
                }
            )

        return sequences


class TheoryOfMind:
    """
    Models human reasoning patterns and mental states
    """

    def __init__(self, data_dir: str = "data/learning") -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.concept_extractor = ConceptExtractor()
        self.pattern_analyzer = PatternAnalyzer()

        # Learned models
        self.concept_clusters = {}
        self.decision_patterns = {}
        self.mental_models = {}
        self.reasoning_chains = {}

        # Analysis results
        self.temporal_patterns = {}
        self.contextual_patterns = {}
        self.personality_profile = {}

        # Database for persistent storage
        self.db_path = self.data_dir / "theory_of_mind.db"
        self._init_database()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize database for theory of mind storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Concept clusters table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS concept_clusters (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    keywords TEXT,
                    frequency INTEGER,
                    associated_verdicts TEXT,
                    emotional_valence REAL,
                    confidence REAL,
                    languages TEXT,
                    examples TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Decision patterns table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_id TEXT NOT NULL,
                    trigger_conditions TEXT,
                    typical_response TEXT,
                    confidence_level REAL,
                    temporal_factors TEXT,
                    contextual_factors TEXT,
                    frequency INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Mental states table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS mental_states (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT,
                    complexity_tolerance REAL,
                    timing_preference TEXT,
                    risk_profile TEXT,
                    cognitive_load REAL,
                    emotional_state TEXT,
                    decision_fatigue REAL,
                    confidence_level REAL,
                    friction_points TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Reasoning chains table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS reasoning_chains (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT,
                    reasoning_steps TEXT,
                    confidence_evolution TEXT,
                    key_factors TEXT,
                    final_verdict TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()

    def build_mental_model(self, rationales: list[str], decisions: list[Decision], languages: list[str]) -> float:
        """Build comprehensive mental model from rationales and decisions"""
        self.logger.info(f"Building mental model from {len(rationales)} rationales")

        # Phase 1: Extract concepts
        self._extract_and_cluster_concepts(rationales, decisions, languages)

        # Phase 2: Analyze patterns
        self._analyze_decision_patterns(decisions)

        # Phase 3: Build personality profile
        self._build_personality_profile(rationales, decisions, languages)

        # Phase 4: Create reasoning chains
        self._create_reasoning_chains(rationales, decisions)

        # Phase 5: Validate model
        accuracy = self._validate_mental_model(decisions)

        self.logger.info(f"Mental model built with {accuracy:.2f} accuracy")
        return accuracy

    def _extract_and_cluster_concepts(self, rationales: list[str], decisions: list[Decision], languages: list[str]):
        """Extract and cluster fundamental concepts"""
        self.logger.info("Extracting and clustering concepts...")

        # Extract concepts from all rationales
        all_concepts = []
        concept_data = []

        for rationale, decision, language in zip(rationales, decisions, languages):
            concepts = self.concept_extractor.extract_concepts(rationale, language)
            emotional_valence = self.concept_extractor.analyze_emotional_valence(rationale, language)

            concept_data.append(
                {
                    "concepts": concepts,
                    "verdict": decision.verdict.value,
                    "rationale": rationale,
                    "language": language,
                    "emotional_valence": emotional_valence,
                    "confidence": decision.confidence_score,
                }
            )

            all_concepts.extend(concepts.keys())

        # Cluster concepts if sklearn is available
        if SKLEARN_AVAILABLE and len(rationales) > 10:
            self._cluster_concepts_sklearn(concept_data)
        else:
            self._cluster_concepts_simple(concept_data)

    def _cluster_concepts_sklearn(self, concept_data: list[dict[str, Any]]):
        """Cluster concepts using sklearn"""
        try:
            # Prepare text data
            texts = [item["rationale"] for item in concept_data]

            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
            vectors = vectorizer.fit_transform(texts)

            # Perform clustering
            n_clusters = min(10, max(3, len(texts) // 10))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(vectors)

            # Create concept clusters
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                clusters[label].append(concept_data[i])

            # Build cluster objects
            feature_names = vectorizer.get_feature_names_out()

            for cluster_id, items in clusters.items():
                # Get top keywords for this cluster
                cluster_center = kmeans.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices]

                # Aggregate verdict statistics
                verdicts = [item["verdict"] for item in items]
                verdict_counts = Counter(verdicts)

                # Calculate emotional valence
                valences = [item["emotional_valence"] for item in items]
                avg_valence = np.mean(valences) if valences else 0

                # Create cluster
                cluster_name = f"cluster_{cluster_id}"
                self.concept_clusters[cluster_name] = ConceptCluster(
                    name=cluster_name,
                    keywords=keywords,
                    frequency=len(items),
                    associated_verdicts=dict(verdict_counts),
                    emotional_valence=avg_valence,
                    confidence=0.8,
                    languages=set(item["language"] for item in items),
                    examples=[item["rationale"][:100] for item in items[:3]],
                )

        except Exception as e:
            self.logger.warning(f"Sklearn clustering failed: {e}")
            self._cluster_concepts_simple(concept_data)

    def _cluster_concepts_simple(self, concept_data: list[dict[str, Any]]):
        """Simple concept clustering fallback"""
        # Group by dominant concept
        concept_groups = defaultdict(list)

        for item in concept_data:
            # Find dominant concept
            concepts = item["concepts"]
            if concepts:
                dominant_concept = max(concepts, key=concepts.get)
                concept_groups[dominant_concept].append(item)

        # Create clusters
        for concept, items in concept_groups.items():
            if len(items) >= 2:  # Minimum cluster size
                # Aggregate statistics
                verdicts = [item["verdict"] for item in items]
                verdict_counts = Counter(verdicts)

                valences = [item["emotional_valence"] for item in items]
                avg_valence = np.mean(valences) if valences else 0

                # Extract keywords
                keywords = [concept]
                for item in items:
                    words = item["rationale"].lower().split()
                    keywords.extend([w for w in words if len(w) > 4])

                # Keep top keywords
                keyword_counts = Counter(keywords)
                top_keywords = [k for k, _ in keyword_counts.most_common(10)]

                self.concept_clusters[concept] = ConceptCluster(
                    name=concept,
                    keywords=top_keywords,
                    frequency=len(items),
                    associated_verdicts=dict(verdict_counts),
                    emotional_valence=avg_valence,
                    confidence=0.6,
                    languages=set(item["language"] for item in items),
                    examples=[item["rationale"][:100] for item in items[:3]],
                )

    def _analyze_decision_patterns(self, decisions: list[Decision]):
        """Analyze decision patterns"""
        self.logger.info("Analyzing decision patterns...")

        # Temporal patterns
        self.temporal_patterns = self.pattern_analyzer.analyze_temporal_patterns(decisions)

        # Sequential patterns
        sequences = self.pattern_analyzer.detect_decision_sequences(decisions)

        # Create decision pattern objects
        for i, seq in enumerate(sequences):
            pattern_id = f"sequence_{i}"
            self.decision_patterns[pattern_id] = DecisionPattern(
                pattern_id=pattern_id,
                trigger_conditions={"sequence_type": "repetitive"},
                typical_response=VerdictType(seq["verdict"]),
                confidence_level=0.7,
                temporal_factors={"avg_review_time": seq["avg_review_time"]},
                contextual_factors={"sequence_length": seq["length"]},
                frequency=seq["length"],
            )

        # Analyze hour preferences
        hour_patterns = self.temporal_patterns.get("hour_preferences", {})
        for hour, verdicts in hour_patterns.items():
            if len(verdicts) >= 3:
                verdict_counts = Counter(verdicts)
                dominant_verdict = verdict_counts.most_common(1)[0][0]

                pattern_id = f"hour_{hour}"
                self.decision_patterns[pattern_id] = DecisionPattern(
                    pattern_id=pattern_id,
                    trigger_conditions={"hour": hour},
                    typical_response=VerdictType(dominant_verdict),
                    confidence_level=verdict_counts[dominant_verdict] / len(verdicts),
                    temporal_factors={"hour": hour},
                    contextual_factors={},
                    frequency=len(verdicts),
                )

    def _build_personality_profile(self, rationales: list[str], decisions: list[Decision], languages: list[str]):
        """Build personality profile"""
        self.logger.info("Building personality profile...")

        # Analyze risk tolerance
        risk_scores = []
        for decision in decisions:
            if "risk" in decision.rationale.lower():
                if decision.verdict == VerdictType.ACCEPT:
                    risk_scores.append(0.7)  # Risk-taking
                else:
                    risk_scores.append(0.3)  # Risk-averse

        # Analyze complexity handling
        complexity_scores = []
        for rationale, decision in zip(rationales, decisions):
            if any(word in rationale.lower() for word in ["complex", "complicated", "simple"]):
                if decision.verdict == VerdictType.ACCEPT:
                    complexity_scores.append(0.8)
                else:
                    complexity_scores.append(0.2)

        # Analyze decision speed
        review_times = [d.review_time_seconds for d in decisions if d.review_time_seconds > 0]
        avg_review_time = np.mean(review_times) if review_times else 30

        # Build profile
        self.personality_profile = {
            "risk_tolerance": np.mean(risk_scores) if risk_scores else 0.5,
            "complexity_comfort": np.mean(complexity_scores) if complexity_scores else 0.5,
            "decision_speed": 1 / (1 + avg_review_time / 60),  # Normalized speed
            "confidence_level": np.mean([d.confidence_score for d in decisions]),
            "emotional_stability": 1 - np.std([d.confidence_score for d in decisions]),
            "language_diversity": len(set(languages)) / len(languages) if languages else 0,
            "consistency": self._calculate_consistency(decisions),
        }

    def _calculate_consistency(self, decisions: list[Decision]) -> float:
        """Calculate decision consistency"""
        if len(decisions) < 2:
            return 1.0

        # Group decisions by similar conditions and see how consistent they are
        consistency_scores = []

        # Look for decisions made in similar time periods
        for i in range(len(decisions) - 1):
            time_diff = abs((decisions[i].timestamp - decisions[i + 1].timestamp).total_seconds())
            if time_diff < 3600:  # Within 1 hour
                if decisions[i].verdict == decisions[i + 1].verdict:
                    consistency_scores.append(1.0)
                else:
                    consistency_scores.append(0.0)

        return np.mean(consistency_scores) if consistency_scores else 0.5

    def _create_reasoning_chains(self, rationales: list[str], decisions: list[Decision]):
        """Create reasoning chains"""
        self.logger.info("Creating reasoning chains...")

        for i, (rationale, decision) in enumerate(zip(rationales, decisions)):
            # Extract reasoning steps from rationale
            sentences = rationale.split(".")
            reasoning_steps = []

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:  # Ignore very short sentences
                    reasoning_steps.append(sentence)

            # Store reasoning chain
            chain_id = f"chain_{i}"
            self.reasoning_chains[chain_id] = {
                "proposal_id": decision.proposal_id,
                "reasoning_steps": reasoning_steps,
                "confidence_evolution": [decision.confidence_score],  # Simplified
                "key_factors": list(self.concept_extractor.extract_concepts(rationale).keys()),
                "final_verdict": decision.verdict.value,
            }

    def _validate_mental_model(self, decisions: list[Decision]) -> float:
        """Validate mental model accuracy"""
        if len(decisions) < 10:
            return 0.5  # Not enough data for validation

        # Test on subset of decisions
        test_decisions = decisions[-len(decisions) // 4 :]  # Use last 25% for testing
        correct_predictions = 0

        for decision in test_decisions:
            # Simple validation: check if our patterns would predict this decision
            predicted_verdict = self._predict_verdict_simple(decision)
            if predicted_verdict == decision.verdict:
                correct_predictions += 1

        return correct_predictions / len(test_decisions) if test_decisions else 0.5

    def _predict_verdict_simple(self, decision: Decision) -> VerdictType:
        """Simple prediction for validation"""
        # Use hour patterns if available
        hour = decision.timestamp.hour
        hour_pattern = f"hour_{hour}"

        if hour_pattern in self.decision_patterns:
            return self.decision_patterns[hour_pattern].typical_response

        # Fall back to most common verdict
        if hasattr(self, "_most_common_verdict"):
            return self._most_common_verdict

        return VerdictType.ACCEPT  # Default

    def infer_state(self, proposal: Proposal) -> MentalState:
        """Infer mental state for a given proposal"""
        # Analyze proposal characteristics
        description_length = len(proposal.description)
        complexity_score = description_length / 1000  # Normalize

        # Infer from personality profile
        complexity_tolerance = self.personality_profile.get("complexity_comfort", 0.5)
        risk_tolerance = self.personality_profile.get("risk_tolerance", 0.5)

        # Assess timing factors
        current_hour = datetime.now().hour
        timing_preference = self._assess_timing_preference(current_hour)

        # Determine risk profile
        risk_profile = self._determine_risk_profile(proposal, risk_tolerance)

        # Calculate cognitive load
        cognitive_load = min(1.0, complexity_score * (1 - complexity_tolerance))

        # Infer emotional state
        emotional_state = self._infer_emotional_state(proposal)

        # Assess decision fatigue (simplified)
        decision_fatigue = 0.5  # Would need session context

        # Identify friction points
        friction_points = self._identify_friction_points(proposal)

        return MentalState(
            complexity_tolerance=complexity_tolerance,
            timing_preference=timing_preference,
            risk_profile=risk_profile,
            cognitive_load=cognitive_load,
            emotional_state=emotional_state,
            decision_fatigue=decision_fatigue,
            confidence_level=self.personality_profile.get("confidence_level", 0.8),
            likely_friction_points=friction_points,
        )

    def _assess_timing_preference(self, hour: int) -> str:
        """Assess timing preference based on hour"""
        if 9 <= hour <= 11:
            return "morning_focused"
        elif 14 <= hour <= 16:
            return "afternoon_productive"
        elif 19 <= hour <= 21:
            return "evening_relaxed"
        else:
            return "neutral"

    def _determine_risk_profile(self, proposal: Proposal, risk_tolerance: float) -> str:
        """Determine risk profile"""
        risk_level_score = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0,
        }.get(proposal.risk_level, 0.5)

        if risk_tolerance > 0.7:
            return "risk_seeking"
        elif risk_tolerance < 0.3:
            return "risk_averse"
        else:
            return "balanced"

    def _infer_emotional_state(self, proposal: Proposal) -> str:
        """Infer emotional state from proposal"""
        # Simplified emotional state inference
        if proposal.impact_score > 0.8:
            return "excited"
        elif proposal.risk_level == RiskLevel.HIGH:
            return "concerned"
        elif proposal.type == ProposalType.SECURITY:
            return "focused"
        else:
            return "neutral"

    def _identify_friction_points(self, proposal: Proposal) -> list[str]:
        """Identify potential friction points"""
        friction_points = []

        # Check complexity
        if len(proposal.description) > 500:
            friction_points.append("high_complexity")

        # Check risk
        if proposal.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            friction_points.append("high_risk")

        # Check resource implications
        if "cost" in proposal.description.lower() or "expensive" in proposal.description.lower():
            friction_points.append("resource_concerns")

        # Check timing
        if "urgent" in proposal.description.lower():
            friction_points.append("timing_pressure")

        return friction_points

    def get_concept_insights(self) -> dict[str, Any]:
        """Get insights about discovered concepts"""
        insights = {
            "total_concepts": len(self.concept_clusters),
            "top_concepts": [],
            "emotional_patterns": {},
            "language_patterns": {},
            "verdict_associations": {},
        }

        # Top concepts by frequency
        sorted_concepts = sorted(self.concept_clusters.items(), key=lambda x: x[1].frequency, reverse=True)

        insights["top_concepts"] = [
            {
                "name": name,
                "frequency": cluster.frequency,
                "keywords": cluster.keywords[:5],
                "emotional_valence": cluster.emotional_valence,
            }
            for name, cluster in sorted_concepts[:10]
        ]

        # Emotional patterns
        for name, cluster in self.concept_clusters.items():
            if cluster.emotional_valence != 0:
                insights["emotional_patterns"][name] = cluster.emotional_valence

        # Language patterns
        for name, cluster in self.concept_clusters.items():
            if len(cluster.languages) > 1:
                insights["language_patterns"][name] = list(cluster.languages)

        # Verdict associations
        for name, cluster in self.concept_clusters.items():
            if cluster.associated_verdicts:
                dominant_verdict = max(cluster.associated_verdicts, key=cluster.associated_verdicts.get)
                insights["verdict_associations"][name] = {
                    "dominant_verdict": dominant_verdict,
                    "distribution": cluster.associated_verdicts,
                }

        return insights

    def get_personality_summary(self) -> dict[str, Any]:
        """Get personality profile summary"""
        if not self.personality_profile:
            return {"message": "No personality profile available"}

        summary = {
            "traits": {},
            "behavioral_patterns": {},
            "preferences": {},
            "strengths": [],
            "areas_for_attention": [],
        }

        # Categorize traits
        for trait, score in self.personality_profile.items():
            if score > 0.7:
                summary["traits"][trait] = "high"
                summary["strengths"].append(trait)
            elif score < 0.3:
                summary["traits"][trait] = "low"
                summary["areas_for_attention"].append(trait)
            else:
                summary["traits"][trait] = "moderate"

        # Behavioral patterns
        if self.decision_patterns:
            summary["behavioral_patterns"] = {
                "total_patterns": len(self.decision_patterns),
                "most_frequent": (
                    max(self.decision_patterns.items(), key=lambda x: x[1].frequency)[0]
                    if self.decision_patterns
                    else None
                ),
            }

        # Preferences
        if self.temporal_patterns:
            summary["preferences"] = {
                "active_hours": list(self.temporal_patterns.get("hour_preferences", {}).keys()),
                "decision_speed": (
                    "fast" if self.personality_profile.get("decision_speed", 0.5) > 0.7 else "thoughtful"
                ),
            }

        return summary

    def export_mental_model(self, filepath: str):
        """Export mental model to file"""
        model_data = {
            "concept_clusters": {
                name: {
                    "name": cluster.name,
                    "keywords": cluster.keywords,
                    "frequency": cluster.frequency,
                    "associated_verdicts": cluster.associated_verdicts,
                    "emotional_valence": cluster.emotional_valence,
                    "confidence": cluster.confidence,
                    "languages": list(cluster.languages),
                    "examples": cluster.examples,
                }
                for name, cluster in self.concept_clusters.items()
            },
            "decision_patterns": {
                name: {
                    "pattern_id": pattern.pattern_id,
                    "trigger_conditions": pattern.trigger_conditions,
                    "typical_response": pattern.typical_response.value,
                    "confidence_level": pattern.confidence_level,
                    "temporal_factors": pattern.temporal_factors,
                    "contextual_factors": pattern.contextual_factors,
                    "frequency": pattern.frequency,
                }
                for name, pattern in self.decision_patterns.items()
            },
            "personality_profile": self.personality_profile,
            "temporal_patterns": self.temporal_patterns,
            "reasoning_chains": self.reasoning_chains,
            "export_timestamp": datetime.now().isoformat(),
            "version": "0.8.0",
        }

        with open(filepath, "w") as f:
            json.dump(model_data, f, indent=2)

        self.logger.info(f"Mental model exported to {filepath}")

    def load_mental_model(self, filepath: str):
        """Load mental model from file"""
        with open(filepath) as f:
            model_data = json.load(f)

        # Reconstruct concept clusters
        self.concept_clusters = {}
        for name, data in model_data.get("concept_clusters", {}).items():
            self.concept_clusters[name] = ConceptCluster(
                name=data["name"],
                keywords=data["keywords"],
                frequency=data["frequency"],
                associated_verdicts=data["associated_verdicts"],
                emotional_valence=data["emotional_valence"],
                confidence=data["confidence"],
                languages=set(data["languages"]),
                examples=data["examples"],
            )

        # Reconstruct decision patterns
        self.decision_patterns = {}
        for name, data in model_data.get("decision_patterns", {}).items():
            self.decision_patterns[name] = DecisionPattern(
                pattern_id=data["pattern_id"],
                trigger_conditions=data["trigger_conditions"],
                typical_response=VerdictType(data["typical_response"]),
                confidence_level=data["confidence_level"],
                temporal_factors=data["temporal_factors"],
                contextual_factors=data["contextual_factors"],
                frequency=data["frequency"],
            )

        # Load other data
        self.personality_profile = model_data.get("personality_profile", {})
        self.temporal_patterns = model_data.get("temporal_patterns", {})
        self.reasoning_chains = model_data.get("reasoning_chains", {})

        self.logger.info(f"Mental model loaded from {filepath}")

    def get_decision_explanation(self, proposal: Proposal, predicted_verdict: VerdictType) -> str:
        """Generate explanation for decision based on mental model"""
        mental_state = self.infer_state(proposal)

        explanations = []

        # Explain based on mental state
        if mental_state.cognitive_load > 0.7:
            explanations.append(f"High cognitive load ({mental_state.cognitive_load:.1f}) suggests complexity concerns")

        if mental_state.risk_profile == "risk_averse" and proposal.risk_level == RiskLevel.HIGH:
            explanations.append("Risk-averse profile conflicts with high-risk proposal")

        if mental_state.emotional_state == "excited" and predicted_verdict == VerdictType.ACCEPT:
            explanations.append("Positive emotional state supports acceptance")

        # Explain based on patterns
        for pattern_name, pattern in self.decision_patterns.items():
            if pattern.typical_response == predicted_verdict and pattern.confidence_level > 0.7:
                explanations.append(f"Pattern '{pattern_name}' suggests {predicted_verdict.value}")

        # Explain based on concepts
        for concept_name, cluster in self.concept_clusters.items():
            if predicted_verdict.value in cluster.associated_verdicts:
                strength = cluster.associated_verdicts[predicted_verdict.value]
                if strength > 2:
                    explanations.append(f"Concept '{concept_name}' historically leads to {predicted_verdict.value}")

        if not explanations:
            explanations.append("Decision based on general patterns and preferences")

        return "; ".join(explanations[:3])  # Limit to top 3 explanations
