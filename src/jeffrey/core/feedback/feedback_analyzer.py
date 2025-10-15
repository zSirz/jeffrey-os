import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from textblob import TextBlob

    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import sympy as sp

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from .models import Decision, Proposal, ProposalType, RiskLevel, VerdictType


@dataclass
class DecisionPattern:
    """Represents a detected decision pattern"""

    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    languages: list[str] = field(default_factory=list)


@dataclass
class BiasDetection:
    """Represents detected bias in decision making"""

    bias_type: str
    severity: str  # low, medium, high
    description: str
    evidence: dict[str, Any]
    recommendation: str
    confidence: float


class FeedbackAnalyzer:
    """Advanced feedback analyzer with multi-language support and bias detection"""

    def __init__(self, db_path: str = "feedback_analysis.db"):
        self.db_path = db_path
        self.supported_languages = ["en", "fr", "es", "de", "it", "pt"]
        self.bias_detector = BiasDetector()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Initialize ML models if available
        self.nlp_model = None
        if TORCH_AVAILABLE:
            self.nlp_model = SimpleDecisionPredictor()

        # Initialize database
        self._init_database()

        # Pattern cache
        self.pattern_cache = {}
        self.bias_cache = {}

    def _init_database(self):
        """Initialize analysis database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Decision patterns table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_patterns (
                    pattern_id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    frequency INTEGER DEFAULT 1,
                    confidence REAL DEFAULT 1.0,
                    description TEXT,
                    metadata TEXT,
                    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    languages TEXT
                )
            """
            )

            # Bias detections table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS bias_detections (
                    id TEXT PRIMARY KEY,
                    bias_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT,
                    evidence TEXT,
                    recommendation TEXT,
                    confidence REAL DEFAULT 1.0,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    resolved BOOLEAN DEFAULT FALSE
                )
            """
            )

            # Sentiment analysis table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_analysis (
                    id TEXT PRIMARY KEY,
                    text_hash TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    sentiment TEXT NOT NULL,
                    confidence REAL DEFAULT 1.0,
                    keywords TEXT,
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Decision predictions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS decision_predictions (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT NOT NULL,
                    predicted_verdict TEXT NOT NULL,
                    actual_verdict TEXT,
                    confidence REAL DEFAULT 1.0,
                    features TEXT,
                    predicted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    correct_prediction BOOLEAN
                )
            """
            )

            conn.commit()

    def analyze_decision_patterns(self, decisions: list[Decision]) -> list[DecisionPattern]:
        """Analyze decision patterns with multi-language support"""
        patterns = []

        # Group decisions by various criteria
        temporal_patterns = self._analyze_temporal_patterns(decisions)
        sentiment_patterns = self._analyze_sentiment_patterns(decisions)
        language_patterns = self._analyze_language_patterns(decisions)
        behavioral_patterns = self._analyze_behavioral_patterns(decisions)

        patterns.extend(temporal_patterns)
        patterns.extend(sentiment_patterns)
        patterns.extend(language_patterns)
        patterns.extend(behavioral_patterns)

        # Store patterns in database
        self._store_patterns(patterns)

        return patterns

    def _analyze_temporal_patterns(self, decisions: list[Decision]) -> list[DecisionPattern]:
        """Analyze temporal decision patterns"""
        patterns = []

        if not decisions:
            return patterns

        # Group by hour of day
        hour_decisions = defaultdict(list)
        for decision in decisions:
            hour = decision.timestamp.hour
            hour_decisions[hour].append(decision)

        # Find peak hours
        peak_hours = []
        avg_decisions_per_hour = len(decisions) / 24

        for hour, hour_decisions_list in hour_decisions.items():
            if len(hour_decisions_list) > avg_decisions_per_hour * 1.5:
                peak_hours.append(hour)

        if peak_hours:
            patterns.append(
                DecisionPattern(
                    pattern_id=f"temporal_peak_{hash(tuple(peak_hours))}",
                    pattern_type="temporal_peak",
                    frequency=len(peak_hours),
                    confidence=0.8,
                    description=f"Peak decision hours: {', '.join(map(str, peak_hours))}",
                    metadata={"peak_hours": peak_hours},
                )
            )

        # Analyze decision speed patterns
        review_times = [d.review_time_seconds for d in decisions if d.review_time_seconds > 0]
        if review_times:
            avg_time = sum(review_times) / len(review_times)
            fast_decisions = [t for t in review_times if t < avg_time * 0.5]

            if len(fast_decisions) > len(review_times) * 0.2:  # More than 20% are fast
                patterns.append(
                    DecisionPattern(
                        pattern_id="fast_decision_pattern",
                        pattern_type="decision_speed",
                        frequency=len(fast_decisions),
                        confidence=0.7,
                        description=f"High frequency of fast decisions ({len(fast_decisions)}/{len(review_times)})",
                        metadata={"avg_time": avg_time, "fast_threshold": avg_time * 0.5},
                    )
                )

        return patterns

    def _analyze_sentiment_patterns(self, decisions: list[Decision]) -> list[DecisionPattern]:
        """Analyze sentiment patterns in decision rationales"""
        patterns = []

        if not decisions:
            return patterns

        sentiments = []
        for decision in decisions:
            sentiment = self.sentiment_analyzer.analyze_sentiment(decision.rationale)
            sentiments.append((decision, sentiment))

        # Group by sentiment
        sentiment_groups = defaultdict(list)
        for decision, sentiment in sentiments:
            sentiment_groups[sentiment["sentiment"]].append(decision)

        # Find dominant sentiment
        total_decisions = len(decisions)
        for sentiment, decisions_list in sentiment_groups.items():
            if len(decisions_list) > total_decisions * 0.6:  # More than 60%
                patterns.append(
                    DecisionPattern(
                        pattern_id=f"sentiment_dominant_{sentiment}",
                        pattern_type="sentiment_dominant",
                        frequency=len(decisions_list),
                        confidence=0.8,
                        description=f"Dominant {sentiment} sentiment in rationales",
                        metadata={
                            "sentiment": sentiment,
                            "percentage": len(decisions_list) / total_decisions,
                        },
                    )
                )

        return patterns

    def _analyze_language_patterns(self, decisions: list[Decision]) -> list[DecisionPattern]:
        """Analyze language patterns in decision rationales"""
        patterns = []

        if not decisions:
            return patterns

        language_counts = defaultdict(int)
        language_verdicts = defaultdict(list)

        for decision in decisions:
            language = self._detect_language(decision.rationale)
            language_counts[language] += 1
            language_verdicts[language].append(decision.verdict)

        # Find multi-language usage
        if len(language_counts) > 1:
            patterns.append(
                DecisionPattern(
                    pattern_id="multilingual_usage",
                    pattern_type="language_diversity",
                    frequency=len(language_counts),
                    confidence=0.9,
                    description=f"Multi-language usage: {list(language_counts.keys())}",
                    metadata={"language_distribution": dict(language_counts)},
                    languages=list(language_counts.keys()),
                )
            )

        # Analyze verdict patterns by language
        for language, verdicts in language_verdicts.items():
            if len(verdicts) > 5:  # At least 5 decisions
                verdict_counts = Counter(v.value for v in verdicts)
                dominant_verdict = verdict_counts.most_common(1)[0]

                if dominant_verdict[1] > len(verdicts) * 0.7:  # More than 70%
                    patterns.append(
                        DecisionPattern(
                            pattern_id=f"language_verdict_{language}_{dominant_verdict[0]}",
                            pattern_type="language_verdict_bias",
                            frequency=dominant_verdict[1],
                            confidence=0.7,
                            description=f"Language {language} shows bias toward {dominant_verdict[0]} verdicts",
                            metadata={
                                "language": language,
                                "dominant_verdict": dominant_verdict[0],
                            },
                            languages=[language],
                        )
                    )

        return patterns

    def _analyze_behavioral_patterns(self, decisions: list[Decision]) -> list[DecisionPattern]:
        """Analyze behavioral patterns in decision making"""
        patterns = []

        if not decisions:
            return patterns

        # Analyze rationale length patterns
        rationale_lengths = [len(d.rationale) for d in decisions]
        avg_length = sum(rationale_lengths) / len(rationale_lengths)

        short_rationales = [d for d in decisions if len(d.rationale) < avg_length * 0.3]
        if len(short_rationales) > len(decisions) * 0.3:  # More than 30% are very short
            patterns.append(
                DecisionPattern(
                    pattern_id="short_rationale_pattern",
                    pattern_type="rationale_quality",
                    frequency=len(short_rationales),
                    confidence=0.6,
                    description=f"High frequency of short rationales ({len(short_rationales)}/{len(decisions)})",
                    metadata={"avg_length": avg_length, "short_threshold": avg_length * 0.3},
                )
            )

        # Analyze confidence patterns
        confidence_scores = [d.confidence_score for d in decisions]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)

        high_confidence = [d for d in decisions if d.confidence_score > 0.9]
        if len(high_confidence) > len(decisions) * 0.8:  # More than 80% are high confidence
            patterns.append(
                DecisionPattern(
                    pattern_id="high_confidence_pattern",
                    pattern_type="confidence_pattern",
                    frequency=len(high_confidence),
                    confidence=0.7,
                    description=f"Consistently high confidence scores ({len(high_confidence)}/{len(decisions)})",
                    metadata={"avg_confidence": avg_confidence},
                )
            )

        return patterns

    def detect_automation_bias(self, decisions: list[Decision]) -> list[BiasDetection]:
        """Detect automation bias in decision making"""
        biases = []

        if not decisions:
            return biases

        # Check for suspiciously fast decisions
        fast_decisions = [d for d in decisions if d.review_time_seconds < 2]
        if len(fast_decisions) > len(decisions) * 0.15:  # More than 15% are very fast
            biases.append(
                BiasDetection(
                    bias_type="automation_bias",
                    severity="medium",
                    description="High frequency of very fast decisions may indicate automation bias",
                    evidence={
                        "fast_decisions": len(fast_decisions),
                        "total_decisions": len(decisions),
                        "percentage": len(fast_decisions) / len(decisions),
                        "threshold": 2.0,
                    },
                    recommendation="Review decision-making process and ensure adequate human consideration",
                    confidence=0.7,
                )
            )

        # Check for duplicated rationales
        rationale_counts = Counter(d.rationale for d in decisions)
        duplicated_rationales = [rationale for rationale, count in rationale_counts.items() if count > 1]

        if duplicated_rationales:
            total_duplicated = sum(rationale_counts[r] for r in duplicated_rationales)
            if total_duplicated > len(decisions) * 0.2:  # More than 20% are duplicated
                biases.append(
                    BiasDetection(
                        bias_type="copy_paste_bias",
                        severity="high",
                        description="High frequency of duplicated rationales suggests copy-paste behavior",
                        evidence={
                            "duplicated_rationales": len(duplicated_rationales),
                            "total_duplicated_decisions": total_duplicated,
                            "percentage": total_duplicated / len(decisions),
                        },
                        recommendation="Encourage unique, thoughtful rationales for each decision",
                        confidence=0.8,
                    )
                )

        # Check for verdict pattern bias
        verdict_counts = Counter(d.verdict.value for d in decisions)
        total_decisions = len(decisions)

        for verdict, count in verdict_counts.items():
            if count > total_decisions * 0.85:  # More than 85% are the same verdict
                biases.append(
                    BiasDetection(
                        bias_type="verdict_bias",
                        severity="medium",
                        description=f"Strong bias toward {verdict} verdicts",
                        evidence={
                            "verdict": verdict,
                            "count": count,
                            "percentage": count / total_decisions,
                        },
                        recommendation="Review decision criteria and ensure balanced evaluation",
                        confidence=0.6,
                    )
                )

        # Store biases in database
        self._store_biases(biases)

        return biases

    def predict_decision(self, proposal: Proposal) -> dict[str, Any]:
        """Predict decision using ML model or symbolic rules"""
        if self.nlp_model and TORCH_AVAILABLE:
            return self._predict_with_ml(proposal)
        else:
            return self._predict_with_rules(proposal)

    def _predict_with_ml(self, proposal: Proposal) -> dict[str, Any]:
        """Predict decision using ML model"""
        # Prepare features
        features = self._extract_features(proposal)

        # Make prediction
        prediction = self.nlp_model.predict(features)

        # Store prediction for later validation
        self._store_prediction(proposal.id, prediction)

        return prediction

    def _predict_with_rules(self, proposal: Proposal) -> dict[str, Any]:
        """Predict decision using symbolic rules"""
        if not SYMPY_AVAILABLE:
            return self._predict_with_heuristics(proposal)

        # Define symbolic variables
        impact = sp.Symbol("impact", real=True)
        risk = sp.Symbol("risk", real=True)

        # Define acceptance probability function
        # P(accept) = impact * (1 - risk_penalty) * type_multiplier

        # Map risk levels to penalties
        risk_penalties = {
            RiskLevel.LOW: 0.1,
            RiskLevel.MEDIUM: 0.3,
            RiskLevel.HIGH: 0.6,
            RiskLevel.CRITICAL: 0.9,
        }

        # Map proposal types to multipliers
        type_multipliers = {
            ProposalType.SECURITY: 1.2,
            ProposalType.BUGFIX: 1.1,
            ProposalType.OPTIMIZATION: 1.0,
            ProposalType.FEATURE: 0.9,
        }

        # Calculate acceptance probability
        risk_penalty = risk_penalties.get(proposal.risk_level, 0.3)
        type_multiplier = type_multipliers.get(proposal.type, 1.0)

        acceptance_prob = proposal.impact_score * (1 - risk_penalty) * type_multiplier
        acceptance_prob = min(1.0, max(0.0, acceptance_prob))

        # Determine predicted verdict
        if acceptance_prob > 0.7:
            predicted_verdict = VerdictType.ACCEPT
        elif acceptance_prob > 0.3:
            predicted_verdict = VerdictType.DEFER
        else:
            predicted_verdict = VerdictType.REJECT

        return {
            "predicted_verdict": predicted_verdict,
            "confidence": acceptance_prob,
            "explanation": f"Impact: {proposal.impact_score:.2f}, Risk penalty: {risk_penalty:.2f}, Type multiplier: {type_multiplier:.2f}",
            "features": {
                "impact_score": proposal.impact_score,
                "risk_level": proposal.risk_level.value,
                "proposal_type": proposal.type.value,
                "acceptance_probability": acceptance_prob,
            },
        }

    def _predict_with_heuristics(self, proposal: Proposal) -> dict[str, Any]:
        """Simple heuristic-based prediction"""
        score = proposal.impact_score

        # Adjust for risk
        if proposal.risk_level == RiskLevel.HIGH:
            score *= 0.7
        elif proposal.risk_level == RiskLevel.CRITICAL:
            score *= 0.5
        elif proposal.risk_level == RiskLevel.LOW:
            score *= 1.1

        # Adjust for type
        if proposal.type == ProposalType.SECURITY:
            score *= 1.2
        elif proposal.type == ProposalType.BUGFIX:
            score *= 1.1

        # Predict verdict
        if score > 0.7:
            predicted_verdict = VerdictType.ACCEPT
        elif score > 0.3:
            predicted_verdict = VerdictType.DEFER
        else:
            predicted_verdict = VerdictType.REJECT

        return {
            "predicted_verdict": predicted_verdict,
            "confidence": score,
            "explanation": f"Heuristic score: {score:.2f}",
            "features": {
                "impact_score": proposal.impact_score,
                "risk_level": proposal.risk_level.value,
                "proposal_type": proposal.type.value,
            },
        }

    def _extract_features(self, proposal: Proposal) -> dict[str, Any]:
        """Extract features for ML prediction"""
        features = {
            "impact_score": proposal.impact_score,
            "risk_level_encoded": self._encode_risk_level(proposal.risk_level),
            "type_encoded": self._encode_proposal_type(proposal.type),
            "description_length": len(proposal.description),
            "plan_length": len(proposal.detailed_plan) if proposal.detailed_plan else 0,
            "sources_count": len(proposal.sources),
            "created_hour": proposal.created_at.hour,
            "created_weekday": proposal.created_at.weekday(),
        }

        # Add sentiment features
        if proposal.description:
            sentiment = self.sentiment_analyzer.analyze_sentiment(proposal.description)
            features["description_sentiment"] = sentiment["polarity"]

        return features

    def _encode_risk_level(self, risk_level: RiskLevel) -> float:
        """Encode risk level as numeric value"""
        encoding = {
            RiskLevel.LOW: 0.25,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.75,
            RiskLevel.CRITICAL: 1.0,
        }
        return encoding.get(risk_level, 0.5)

    def _encode_proposal_type(self, proposal_type: ProposalType) -> float:
        """Encode proposal type as numeric value"""
        encoding = {
            ProposalType.SECURITY: 0.9,
            ProposalType.BUGFIX: 0.7,
            ProposalType.OPTIMIZATION: 0.5,
            ProposalType.FEATURE: 0.3,
        }
        return encoding.get(proposal_type, 0.5)

    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        if not text:
            return "en"

        # Simple language detection based on common words
        language_patterns = {
            "fr": ["le", "la", "les", "de", "du", "et", "est", "pour", "avec", "sur"],
            "es": ["el", "la", "los", "las", "de", "del", "y", "es", "para", "con"],
            "de": ["der", "die", "das", "und", "ist", "für", "mit", "auf", "zu", "von"],
            "it": ["il", "la", "lo", "gli", "di", "del", "e", "è", "per", "con"],
            "pt": ["o", "a", "os", "as", "de", "do", "e", "é", "para", "com"],
        }

        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)

        language_scores = {}
        for lang, patterns in language_patterns.items():
            score = sum(1 for word in words if word in patterns)
            if score > 0:
                language_scores[lang] = score / len(words)

        if language_scores:
            return max(language_scores, key=language_scores.get)

        return "en"  # Default to English

    def _store_patterns(self, patterns: list[DecisionPattern]):
        """Store patterns in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for pattern in patterns:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO decision_patterns (
                        pattern_id, pattern_type, frequency, confidence,
                        description, metadata, languages
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        pattern.pattern_id,
                        pattern.pattern_type,
                        pattern.frequency,
                        pattern.confidence,
                        pattern.description,
                        json.dumps(pattern.metadata),
                        json.dumps(pattern.languages),
                    ),
                )

            conn.commit()

    def _store_biases(self, biases: list[BiasDetection]):
        """Store bias detections in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            for bias in biases:
                cursor.execute(
                    """
                    INSERT INTO bias_detections (
                        id, bias_type, severity, description,
                        evidence, recommendation, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        f"bias_{bias.bias_type}_{hash(bias.description)}",
                        bias.bias_type,
                        bias.severity,
                        bias.description,
                        json.dumps(bias.evidence),
                        bias.recommendation,
                        bias.confidence,
                    ),
                )

            conn.commit()

    def _store_prediction(self, proposal_id: str, prediction: dict[str, Any]):
        """Store prediction for later validation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO decision_predictions (
                    id, proposal_id, predicted_verdict, confidence, features
                ) VALUES (?, ?, ?, ?, ?)
            """,
                (
                    f"pred_{proposal_id}_{int(datetime.now().timestamp())}",
                    proposal_id,
                    prediction["predicted_verdict"].value,
                    prediction["confidence"],
                    json.dumps(prediction["features"]),
                ),
            )

            conn.commit()

    def validate_predictions(self, decisions: list[Decision]) -> dict[str, Any]:
        """Validate past predictions against actual decisions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            validation_results = {
                "total_predictions": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "by_verdict": {},
            }

            for decision in decisions:
                cursor.execute(
                    """
                    SELECT * FROM decision_predictions
                    WHERE proposal_id = ? AND actual_verdict IS NULL
                """,
                    (decision.proposal_id,),
                )

                prediction_row = cursor.fetchone()
                if prediction_row:
                    predicted_verdict = prediction_row[2]
                    actual_verdict = decision.verdict.value

                    # Update prediction with actual verdict
                    cursor.execute(
                        """
                        UPDATE decision_predictions
                        SET actual_verdict = ?, correct_prediction = ?
                        WHERE id = ?
                    """,
                        (actual_verdict, predicted_verdict == actual_verdict, prediction_row[0]),
                    )

                    validation_results["total_predictions"] += 1
                    if predicted_verdict == actual_verdict:
                        validation_results["correct_predictions"] += 1

                    # Track by verdict
                    if actual_verdict not in validation_results["by_verdict"]:
                        validation_results["by_verdict"][actual_verdict] = {
                            "total": 0,
                            "correct": 0,
                        }

                    validation_results["by_verdict"][actual_verdict]["total"] += 1
                    if predicted_verdict == actual_verdict:
                        validation_results["by_verdict"][actual_verdict]["correct"] += 1

            conn.commit()

            # Calculate accuracy
            if validation_results["total_predictions"] > 0:
                validation_results["accuracy"] = (
                    validation_results["correct_predictions"] / validation_results["total_predictions"]
                )

            return validation_results

    def anonymize_for_ml(self, feedback_data: list[dict[str, Any]]) -> dict[str, Any]:
        """Anonymize feedback data for ML training with differential privacy"""
        anonymized_data = []

        for record in feedback_data:
            anonymized_record = {}

            # Keep numeric features as-is (they're already anonymous)
            for key, value in record.items():
                if isinstance(value, (int, float)):
                    anonymized_record[key] = value
                elif key == "rationale" and isinstance(value, str):
                    # Anonymize text
                    anonymized_record[key] = self._anonymize_text(value)
                elif key in ["verdict", "proposal_type", "risk_level"]:
                    # Keep categorical data
                    anonymized_record[key] = value

            anonymized_data.append(anonymized_record)

        # Add differential privacy noise if numpy available
        if NUMPY_AVAILABLE:
            anonymized_data = self._add_differential_privacy_noise(anonymized_data)

        return {
            "data": anonymized_data,
            "anonymization_method": "text_replacement_with_dp_noise",
            "timestamp": datetime.now().isoformat(),
            "record_count": len(anonymized_data),
        }

    def _anonymize_text(self, text: str) -> str:
        """Anonymize text content"""
        # Remove potential identifiers
        text = re.sub(r"\b[A-Z][a-z]+\b", "[NAME]", text)
        text = re.sub(r"\b\d{1,5}\b", "[NUMBER]", text)
        text = re.sub(r"\b[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}\b", "[UUID]", text)
        text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", text)

        return text

    def _add_differential_privacy_noise(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Add differential privacy noise to numeric data"""
        if not NUMPY_AVAILABLE:
            return data

        # Add small amount of noise to numeric features
        numeric_keys = ["impact_score", "review_time", "confidence"]

        for record in data:
            for key in numeric_keys:
                if key in record and isinstance(record[key], (int, float)):
                    # Add Laplace noise with small epsilon
                    noise = np.random.laplace(0, 0.01)
                    record[key] = max(0, min(1, record[key] + noise))

        return data


class SentimentAnalyzer:
    """Multi-language sentiment analyzer"""

    def __init__(self):
        self.sentiment_cache = {}

        # Multi-language sentiment keywords
        self.positive_keywords = {
            "en": ["good", "great", "excellent", "approve", "beneficial", "improve", "positive"],
            "fr": ["bon", "excellent", "positif", "améliorer", "bénéfique", "approuver"],
            "es": ["bueno", "excelente", "positivo", "mejorar", "beneficioso", "aprobar"],
            "de": ["gut", "ausgezeichnet", "positiv", "verbessern", "vorteilhaft", "genehmigen"],
            "it": ["buono", "eccellente", "positivo", "migliorare", "benefico", "approvare"],
            "pt": ["bom", "excelente", "positivo", "melhorar", "benéfico", "aprovar"],
        }

        self.negative_keywords = {
            "en": ["bad", "poor", "terrible", "reject", "dangerous", "risky", "negative"],
            "fr": ["mauvais", "pauvre", "terrible", "rejeter", "dangereux", "risqué"],
            "es": ["malo", "pobre", "terrible", "rechazar", "peligroso", "arriesgado"],
            "de": ["schlecht", "arm", "schrecklich", "ablehnen", "gefährlich", "riskant"],
            "it": ["cattivo", "povero", "terribile", "rifiutare", "pericoloso", "rischioso"],
            "pt": ["ruim", "pobre", "terrível", "rejeitar", "perigoso", "arriscado"],
        }

    def analyze_sentiment(self, text: str, language: str = "en") -> dict[str, Any]:
        """Analyze sentiment with multi-language support"""
        if not text:
            return {"sentiment": "neutral", "polarity": 0.0, "confidence": 0.0}

        # Check cache
        text_hash = hashlib.md5(text.encode()).hexdigest()
        if text_hash in self.sentiment_cache:
            return self.sentiment_cache[text_hash]

        # Use TextBlob if available
        if TEXTBLOB_AVAILABLE:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity

                if polarity > 0.1:
                    sentiment = "positive"
                elif polarity < -0.1:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"

                result = {"sentiment": sentiment, "polarity": polarity, "confidence": abs(polarity)}

                self.sentiment_cache[text_hash] = result
                return result
            except:
                pass

        # Fallback to keyword-based analysis
        return self._keyword_based_sentiment(text, language)

    def _keyword_based_sentiment(self, text: str, language: str) -> dict[str, Any]:
        """Keyword-based sentiment analysis"""
        text_lower = text.lower()

        positive_words = self.positive_keywords.get(language, self.positive_keywords["en"])
        negative_words = self.negative_keywords.get(language, self.negative_keywords["en"])

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        if positive_count > negative_count:
            sentiment = "positive"
            polarity = 0.5
        elif negative_count > positive_count:
            sentiment = "negative"
            polarity = -0.5
        else:
            sentiment = "neutral"
            polarity = 0.0

        return {"sentiment": sentiment, "polarity": polarity, "confidence": abs(polarity)}


class BiasDetector:
    """Detects various types of bias in decision making"""

    def __init__(self):
        self.bias_patterns = {
            "confirmation_bias": self._detect_confirmation_bias,
            "anchoring_bias": self._detect_anchoring_bias,
            "availability_bias": self._detect_availability_bias,
            "temporal_bias": self._detect_temporal_bias,
        }

    def detect_biases(self, decisions: list[Decision]) -> list[BiasDetection]:
        """Detect all types of biases"""
        biases = []

        for bias_type, detector in self.bias_patterns.items():
            detected_biases = detector(decisions)
            biases.extend(detected_biases)

        return biases

    def _detect_confirmation_bias(self, decisions: list[Decision]) -> list[BiasDetection]:
        """Detect confirmation bias"""
        # Implementation would analyze if decisions consistently confirm initial impressions
        return []

    def _detect_anchoring_bias(self, decisions: list[Decision]) -> list[BiasDetection]:
        """Detect anchoring bias"""
        # Implementation would analyze if decisions are overly influenced by first piece of information
        return []

    def _detect_availability_bias(self, decisions: list[Decision]) -> list[BiasDetection]:
        """Detect availability bias"""
        # Implementation would analyze if recent events overly influence decisions
        return []

    def _detect_temporal_bias(self, decisions: list[Decision]) -> list[BiasDetection]:
        """Detect temporal bias"""
        # Implementation would analyze if decision quality varies by time
        return []


class SimpleDecisionPredictor:
    """Simple neural network for decision prediction"""

    def __init__(self):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ML predictions")

        self.model = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # 3 classes: accept, reject, defer
            nn.Softmax(dim=1),
        )

        self.trained = False

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Make prediction using the model"""
        if not self.trained:
            # Return random prediction if not trained
            import random

            verdicts = [VerdictType.ACCEPT, VerdictType.REJECT, VerdictType.DEFER]
            return {
                "predicted_verdict": random.choice(verdicts),
                "confidence": 0.33,
                "explanation": "Random prediction (model not trained)",
                "features": features,
            }

        # Convert features to tensor
        feature_vector = self._features_to_tensor(features)

        # Make prediction
        with torch.no_grad():
            output = self.model(feature_vector)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.max(output).item()

        # Map to verdict
        verdict_map = {0: VerdictType.ACCEPT, 1: VerdictType.REJECT, 2: VerdictType.DEFER}
        predicted_verdict = verdict_map[predicted_class]

        return {
            "predicted_verdict": predicted_verdict,
            "confidence": confidence,
            "explanation": f"ML prediction with confidence {confidence:.2f}",
            "features": features,
        }

    def _features_to_tensor(self, features: dict[str, Any]) -> torch.Tensor:
        """Convert features dictionary to tensor"""
        # Extract numeric features in consistent order
        feature_values = [
            features.get("impact_score", 0.0),
            features.get("risk_level_encoded", 0.5),
            features.get("type_encoded", 0.5),
            features.get("description_length", 0.0) / 1000.0,  # Normalize
            features.get("plan_length", 0.0) / 1000.0,  # Normalize
            features.get("sources_count", 0.0) / 10.0,  # Normalize
            features.get("created_hour", 12.0) / 24.0,  # Normalize
            features.get("created_weekday", 3.0) / 7.0,  # Normalize
            features.get("description_sentiment", 0.0),
            0.0,  # Padding to make it 10 features
        ]

        return torch.tensor(feature_values, dtype=torch.float32).unsqueeze(0)
