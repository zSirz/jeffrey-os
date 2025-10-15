"""
Jeffrey OS Phase 0.8 - Meta-Learner
Hybrid brain that combines classification with causal understanding
"""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Optional dependencies with fallbacks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import sympy as sp

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

# Import Jeffrey OS components
from feedback.models import Decision, Proposal, ProposalType, RiskLevel


@dataclass
class MetaLearningData:
    """Container for meta-learning training data"""

    proposals: list[Proposal]
    decisions: list[Decision]
    rationales: list[str]
    languages: list[str]
    timestamps: list[datetime]
    contexts: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UnderstandingResult:
    """Result of meta-learning prediction with understanding"""

    verdict_probability: dict[str, float]
    primary_reason: str
    causal_factors: dict[str, float]
    mental_model: dict[str, Any]
    confidence: float
    explanation: str
    symbolic_equation: str | None = None
    natural_language: str = ""


class EmotionalStateTracker:
    """Tracks emotional patterns in decision-making"""

    def __init__(self):
        self.emotional_patterns = {
            "enthusiasm": ["excited", "great", "excellent", "fantastic"],
            "concern": ["worried", "concerned", "risky", "dangerous"],
            "fatigue": ["tired", "overwhelming", "too much", "complex"],
            "confidence": ["sure", "certain", "definitely", "clearly"],
            "uncertainty": ["maybe", "perhaps", "unsure", "unclear"],
        }

        self.language_patterns = {
            "fr": {
                "enthusiasm": ["formidable", "excellent", "fantastique"],
                "concern": ["inquiet", "risqué", "dangereux"],
                "fatigue": ["fatigué", "trop", "complexe"],
                "confidence": ["sûr", "certain", "clairement"],
                "uncertainty": ["peut-être", "incertain", "pas sûr"],
            },
            "es": {
                "enthusiasm": ["excelente", "fantástico", "genial"],
                "concern": ["preocupado", "riesgoso", "peligroso"],
                "fatigue": ["cansado", "demasiado", "complejo"],
                "confidence": ["seguro", "cierto", "claramente"],
                "uncertainty": ["tal vez", "incierto", "no seguro"],
            },
        }

    def analyze_emotional_state(self, rationale: str, language: str = "en") -> dict[str, float]:
        """Analyze emotional state from rationale"""
        patterns = self.emotional_patterns.copy()
        if language in self.language_patterns:
            patterns.update(self.language_patterns[language])

        rationale_lower = rationale.lower()
        emotions = {}

        for emotion, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in rationale_lower)
            emotions[emotion] = score / len(keywords) if keywords else 0

        return emotions


class CausalFactorExtractor:
    """Extracts causal factors from proposals and rationales"""

    def __init__(self):
        self.causal_keywords = {
            "complexity": ["complex", "complicated", "difficult", "hard"],
            "timing": ["time", "deadline", "urgent", "quick", "slow"],
            "risk": ["risk", "dangerous", "safe", "secure", "unsafe"],
            "resource": ["cost", "expensive", "cheap", "resource", "budget"],
            "impact": ["impact", "benefit", "improvement", "advantage"],
            "technical": ["technical", "implementation", "architecture", "design"],
            "user": ["user", "customer", "experience", "interface"],
            "maintenance": ["maintain", "support", "update", "legacy"],
        }

        self.multilang_keywords = {
            "fr": {
                "complexity": ["complexe", "compliqué", "difficile"],
                "timing": ["temps", "délai", "urgent", "rapide"],
                "risk": ["risque", "dangereux", "sûr", "sécurisé"],
                "resource": ["coût", "cher", "budget", "ressource"],
                "impact": ["impact", "bénéfice", "amélioration"],
                "technical": ["technique", "implémentation", "architecture"],
                "user": ["utilisateur", "client", "expérience"],
                "maintenance": ["maintenir", "support", "mise à jour"],
            }
        }

    def extract_factors(self, proposal: Proposal, rationale: str, language: str = "en") -> dict[str, float]:
        """Extract causal factors from proposal and rationale"""
        factors = {}

        # Use appropriate keywords for language
        keywords = self.causal_keywords.copy()
        if language in self.multilang_keywords:
            for category, words in self.multilang_keywords[language].items():
                keywords[category].extend(words)

        # Analyze proposal description
        description_lower = proposal.description.lower()
        rationale_lower = rationale.lower()

        for factor, words in keywords.items():
            # Score based on proposal description
            desc_score = sum(1 for word in words if word in description_lower)
            # Score based on rationale
            rationale_score = sum(1 for word in words if word in rationale_lower)

            # Combine scores with rationale weighted higher
            factors[factor] = (desc_score * 0.3 + rationale_score * 0.7) / len(words)

        # Add proposal-specific factors
        factors["impact_score"] = proposal.impact_score
        factors["risk_level"] = self._encode_risk_level(proposal.risk_level)
        factors["proposal_type"] = self._encode_proposal_type(proposal.type)
        factors["sources_count"] = len(proposal.sources)

        return factors

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


class NeuralClassifier(nn.Module):
    """Neural network for decision classification"""

    def __init__(self, input_size: int, hidden_sizes: list[int] = [128, 64, 32]):
        super(NeuralClassifier, self).__init__()

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([nn.Linear(prev_size, hidden_size), nn.ReLU(), nn.Dropout(0.3)])
            prev_size = hidden_size

        layers.append(nn.Linear(prev_size, 3))  # accept, reject, defer
        layers.append(nn.Softmax(dim=1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MetaLearner:
    """
    Hybrid meta-learner that combines classification with causal understanding
    and theory of mind modeling
    """

    def __init__(self, data_dir: str = "data/learning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.classifier = None
        self.causal_model = None
        self.mind_theory = None

        # Analysis components
        self.emotional_tracker = EmotionalStateTracker()
        self.causal_extractor = CausalFactorExtractor()

        # Training data
        self.training_data = None
        self.feature_names = []

        # Model performance tracking
        self.performance_history = []
        self.confidence_calibration = {}

        # Database for persistent storage
        self.db_path = self.data_dir / "meta_learning.db"
        self._init_database()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize database for meta-learning storage"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Training sessions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS training_sessions (
                    id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    data_size INTEGER,
                    accuracy REAL,
                    causal_accuracy REAL,
                    model_version TEXT,
                    config TEXT
                )
            """
            )

            # Prediction history table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT,
                    predicted_verdict TEXT,
                    actual_verdict TEXT,
                    confidence REAL,
                    primary_reason TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    correct_prediction BOOLEAN
                )
            """
            )

            # Causal factors table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS causal_factors (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT,
                    factor_name TEXT,
                    factor_value REAL,
                    importance REAL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()

    def load_training_data(
        self,
        proposals: list[Proposal],
        decisions: list[Decision],
        rationales: list[str],
        languages: list[str],
    ) -> MetaLearningData:
        """Load and prepare training data"""
        self.logger.info(f"Loading training data: {len(proposals)} proposals")

        # Validate data consistency
        assert len(proposals) == len(decisions) == len(rationales) == len(languages), (
            "All data lists must have the same length"
        )

        # Create training data container
        self.training_data = MetaLearningData(
            proposals=proposals,
            decisions=decisions,
            rationales=rationales,
            languages=languages,
            timestamps=[d.timestamp for d in decisions],
        )

        # Extract contexts and features
        for i, (proposal, decision, rationale, language) in enumerate(zip(proposals, decisions, rationales, languages)):
            context = {
                "emotional_state": self.emotional_tracker.analyze_emotional_state(rationale, language),
                "causal_factors": self.causal_extractor.extract_factors(proposal, rationale, language),
                "decision_time": decision.review_time_seconds,
                "confidence": decision.confidence_score,
                "language": language,
            }
            self.training_data.contexts.append(context)

        self.logger.info("Training data loaded successfully")
        return self.training_data

    def train(
        self,
        proposals: list[Proposal],
        decisions: list[Decision],
        rationales: list[str],
        languages: list[str],
    ) -> dict[str, float]:
        """Train the meta-learner with multi-phase approach"""
        self.logger.info("Starting meta-learning training...")

        # Phase 1: Load and prepare data
        training_data = self.load_training_data(proposals, decisions, rationales, languages)

        # Phase 2: Train basic classifier
        classifier_accuracy = self._train_classifier(training_data)

        # Phase 3: Train causal model
        causal_accuracy = self._train_causal_model(training_data)

        # Phase 4: Build theory of mind
        mind_accuracy = self._build_mind_theory(training_data)

        # Phase 5: Integrate models
        integration_score = self._integrate_models(training_data)

        # Record training session
        self._record_training_session(len(proposals), classifier_accuracy, causal_accuracy)

        results = {
            "classifier_accuracy": classifier_accuracy,
            "causal_accuracy": causal_accuracy,
            "mind_accuracy": mind_accuracy,
            "integration_score": integration_score,
            "total_samples": len(proposals),
        }

        self.logger.info(f"Training completed: {results}")
        return results

    def _train_classifier(self, data: MetaLearningData) -> float:
        """Train the basic verdict classifier"""
        self.logger.info("Training classifier...")

        # Prepare features
        X = []
        y = []

        for i, (proposal, decision, context) in enumerate(zip(data.proposals, data.decisions, data.contexts)):
            # Combine all features
            features = []
            features.extend(context["causal_factors"].values())
            features.extend(context["emotional_state"].values())
            features.extend(
                [
                    context["decision_time"],
                    context["confidence"],
                    1 if context["language"] == "en" else 0,  # Language encoding
                    proposal.created_at.hour / 24.0,  # Time of day
                    proposal.created_at.weekday() / 7.0,  # Day of week
                ]
            )

            X.append(features)
            y.append(decision.verdict.value)

        # Store feature names for later use
        self.feature_names = (
            list(data.contexts[0]["causal_factors"].keys())
            + list(data.contexts[0]["emotional_state"].keys())
            + ["decision_time", "confidence", "is_english", "hour", "weekday"]
        )

        X = np.array(X)
        y = np.array(y)

        # Train with best available method
        if TORCH_AVAILABLE and len(X) > 100:
            accuracy = self._train_neural_classifier(X, y)
        elif SKLEARN_AVAILABLE:
            accuracy = self._train_sklearn_classifier(X, y)
        else:
            accuracy = self._train_simple_classifier(X, y)

        return accuracy

    def _train_neural_classifier(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train neural network classifier"""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        # Prepare data
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train)
        y_test_tensor = torch.LongTensor(y_test)

        # Create model
        self.classifier = NeuralClassifier(X.shape[1])
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)

        # Training loop
        self.classifier.train()
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = self.classifier(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()

        # Evaluate
        self.classifier.eval()
        with torch.no_grad():
            outputs = self.classifier(X_test_tensor)
            _, predicted = torch.max(outputs.data, 1)
            accuracy = (predicted == y_test_tensor).float().mean().item()

        # Store label encoder for later use
        self.label_encoder = le

        return accuracy

    def _train_sklearn_classifier(self, X: np.ndarray, y: np.ndarray) -> float:
        """Train scikit-learn classifier"""
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)

        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        return accuracy

    def _train_simple_classifier(self, X: np.ndarray, y: np.ndarray) -> float:
        """Simple fallback classifier"""
        # Simple majority class baseline
        from collections import Counter

        most_common = Counter(y).most_common(1)[0][0]

        self.classifier = lambda x: most_common

        # Return baseline accuracy
        return Counter(y)[most_common] / len(y)

    def _train_causal_model(self, data: MetaLearningData) -> float:
        """Train causal relationship model"""
        self.logger.info("Training causal model...")

        # Extract causal relationships
        causal_relationships = {}

        for i, (proposal, decision, context) in enumerate(zip(data.proposals, data.decisions, data.contexts)):
            verdict = decision.verdict.value
            rationale = data.rationales[i]

            # Analyze which factors are mentioned in rationale
            for factor, value in context["causal_factors"].items():
                if factor not in causal_relationships:
                    causal_relationships[factor] = {"accept": [], "reject": [], "defer": []}

                causal_relationships[factor][verdict].append(value)

        # Build causal model
        self.causal_model = {}
        total_accuracy = 0

        for factor, verdicts in causal_relationships.items():
            # Calculate mean factor value for each verdict
            factor_means = {}
            for verdict, values in verdicts.items():
                if values:
                    factor_means[verdict] = np.mean(values)
                else:
                    factor_means[verdict] = 0

            self.causal_model[factor] = factor_means

            # Simple accuracy estimate
            if len(set(factor_means.values())) > 1:
                total_accuracy += 1

        accuracy = total_accuracy / len(causal_relationships) if causal_relationships else 0

        self.logger.info(f"Causal model trained with {len(causal_relationships)} factors")
        return accuracy

    def _build_mind_theory(self, data: MetaLearningData) -> float:
        """Build theory of mind model"""
        self.logger.info("Building theory of mind...")

        # Import theory of mind component
        from .theory_of_mind import TheoryOfMind

        self.mind_theory = TheoryOfMind()

        # Build mental model from rationales
        accuracy = self.mind_theory.build_mental_model(data.rationales, data.decisions, data.languages)

        return accuracy

    def _integrate_models(self, data: MetaLearningData) -> float:
        """Integrate all models into unified system"""
        self.logger.info("Integrating models...")

        # Test integration on sample data
        correct_predictions = 0
        total_predictions = min(len(data.proposals), 50)  # Test on subset

        for i in range(total_predictions):
            proposal = data.proposals[i]
            actual_decision = data.decisions[i]

            # Make integrated prediction
            try:
                prediction = self.predict_with_understanding(proposal)

                # Check if prediction matches actual
                predicted_verdict = max(prediction.verdict_probability, key=prediction.verdict_probability.get)

                if predicted_verdict == actual_decision.verdict.value:
                    correct_predictions += 1

            except Exception as e:
                self.logger.warning(f"Integration test failed for proposal {i}: {e}")

        integration_score = correct_predictions / total_predictions if total_predictions > 0 else 0

        self.logger.info(f"Integration score: {integration_score:.2f}")
        return integration_score

    def predict_with_understanding(
        self, proposal: Proposal, context: dict[str, Any] | None = None
    ) -> UnderstandingResult:
        """
        Make prediction with full understanding of reasoning
        """
        if not self.classifier or not self.causal_model:
            raise ValueError("Model not trained. Call train() first.")

        # Extract features
        if context is None:
            context = {
                "causal_factors": self.causal_extractor.extract_factors(proposal, "", "en"),
                "emotional_state": self.emotional_tracker.analyze_emotional_state("", "en"),
                "decision_time": 30.0,  # Default
                "confidence": 0.8,  # Default
                "language": "en",
            }

        # Prepare feature vector
        features = []
        features.extend(context["causal_factors"].values())
        features.extend(context["emotional_state"].values())
        features.extend(
            [
                context["decision_time"],
                context["confidence"],
                1 if context["language"] == "en" else 0,
                proposal.created_at.hour / 24.0,
                proposal.created_at.weekday() / 7.0,
            ]
        )

        # Make prediction
        if TORCH_AVAILABLE and hasattr(self.classifier, "forward"):
            # Neural network prediction
            feature_tensor = torch.FloatTensor([features])
            with torch.no_grad():
                outputs = self.classifier(feature_tensor)
                probs = outputs.numpy()[0]

            verdict_probs = {
                "accept": float(probs[0]),
                "reject": float(probs[1]),
                "defer": float(probs[2]),
            }

        elif SKLEARN_AVAILABLE and hasattr(self.classifier, "predict_proba"):
            # Scikit-learn prediction
            probs = self.classifier.predict_proba([features])[0]
            classes = self.classifier.classes_

            verdict_probs = {classes[i]: float(probs[i]) for i in range(len(classes))}

        else:
            # Simple fallback
            verdict_probs = {"accept": 0.33, "reject": 0.33, "defer": 0.34}

        # Identify primary causal factor
        causal_factors = context["causal_factors"]
        primary_factor = max(causal_factors, key=causal_factors.get)

        # Get mental model prediction
        mental_model = {}
        if self.mind_theory:
            mental_model = self.mind_theory.infer_state(proposal)

        # Calculate confidence
        confidence = max(verdict_probs.values())

        # Generate explanation
        explanation = self._generate_explanation(verdict_probs, primary_factor, causal_factors)

        return UnderstandingResult(
            verdict_probability=verdict_probs,
            primary_reason=primary_factor,
            causal_factors=causal_factors,
            mental_model=mental_model,
            confidence=confidence,
            explanation=explanation,
            natural_language=self._generate_natural_explanation(verdict_probs, primary_factor, context["language"]),
        )

    def _generate_explanation(
        self, verdict_probs: dict[str, float], primary_factor: str, causal_factors: dict[str, float]
    ) -> str:
        """Generate explanation for prediction"""
        max_verdict = max(verdict_probs, key=verdict_probs.get)
        max_prob = verdict_probs[max_verdict]

        explanation = f"Predicted {max_verdict} with {max_prob:.1%} confidence. "
        explanation += f"Primary factor: {primary_factor} (strength: {causal_factors[primary_factor]:.2f}). "

        # Add top contributing factors
        sorted_factors = sorted(causal_factors.items(), key=lambda x: x[1], reverse=True)
        top_factors = sorted_factors[:3]

        if len(top_factors) > 1:
            explanation += f"Contributing factors: {', '.join([f'{f}({v:.2f})' for f, v in top_factors])}"

        return explanation

    def _generate_natural_explanation(self, verdict_probs: dict[str, float], primary_factor: str, language: str) -> str:
        """Generate natural language explanation"""
        max_verdict = max(verdict_probs, key=verdict_probs.get)
        max_prob = verdict_probs[max_verdict]

        templates = {
            "en": {
                "accept": "I recommend accepting this proposal because the {factor} factor is favorable.",
                "reject": "I recommend rejecting this proposal due to concerns about {factor}.",
                "defer": "I suggest deferring this proposal to address {factor} considerations.",
            },
            "fr": {
                "accept": "Je recommande d'accepter cette proposition car le facteur {factor} est favorable.",
                "reject": "Je recommande de rejeter cette proposition en raison de préoccupations concernant {factor}.",
                "defer": "Je suggère de reporter cette proposition pour aborder les considérations {factor}.",
            },
            "es": {
                "accept": "Recomiendo aceptar esta propuesta porque el factor {factor} es favorable.",
                "reject": "Recomiendo rechazar esta propuesta debido a preocupaciones sobre {factor}.",
                "defer": "Sugiero diferir esta propuesta para abordar las consideraciones {factor}.",
            },
        }

        lang_templates = templates.get(language, templates["en"])
        template = lang_templates.get(max_verdict, lang_templates["accept"])

        return template.format(factor=primary_factor)

    def _record_training_session(self, data_size: int, classifier_accuracy: float, causal_accuracy: float):
        """Record training session in database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            cursor.execute(
                """
                INSERT INTO training_sessions (
                    id, data_size, accuracy, causal_accuracy, model_version, config
                ) VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    data_size,
                    classifier_accuracy,
                    causal_accuracy,
                    "0.8.0",
                    json.dumps(
                        {
                            "torch_available": TORCH_AVAILABLE,
                            "sklearn_available": SKLEARN_AVAILABLE,
                            "sympy_available": SYMPY_AVAILABLE,
                        }
                    ),
                ),
            )

            conn.commit()

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics and statistics"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Get training history
            cursor.execute(
                """
                SELECT * FROM training_sessions
                ORDER BY timestamp DESC
                LIMIT 10
            """
            )

            training_history = cursor.fetchall()

            # Get prediction accuracy
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN correct_prediction THEN 1 ELSE 0 END) as correct_predictions,
                    AVG(confidence) as avg_confidence
                FROM prediction_history
            """
            )

            prediction_stats = cursor.fetchone()

            return {
                "training_sessions": len(training_history),
                "latest_accuracy": training_history[0][3] if training_history else 0,
                "latest_causal_accuracy": training_history[0][4] if training_history else 0,
                "total_predictions": prediction_stats[0] if prediction_stats else 0,
                "prediction_accuracy": (
                    prediction_stats[1] / prediction_stats[0] if prediction_stats and prediction_stats[0] > 0 else 0
                ),
                "average_confidence": prediction_stats[2] if prediction_stats else 0,
                "feature_count": len(self.feature_names),
                "causal_factors": len(self.causal_model) if self.causal_model else 0,
            }

    def detect_concept_drift(self, recent_data: MetaLearningData) -> dict[str, Any]:
        """Detect concept drift in recent data"""
        if not self.training_data:
            return {"drift_detected": False, "message": "No baseline data available"}

        # Compare recent data with training data
        baseline_verdicts = [d.verdict.value for d in self.training_data.decisions]
        recent_verdicts = [d.verdict.value for d in recent_data.decisions]

        # Calculate distribution differences
        from collections import Counter

        baseline_dist = Counter(baseline_verdicts)
        recent_dist = Counter(recent_verdicts)

        # Simple drift detection using distribution difference
        drift_score = 0
        for verdict in ["accept", "reject", "defer"]:
            baseline_freq = baseline_dist.get(verdict, 0) / len(baseline_verdicts)
            recent_freq = recent_dist.get(verdict, 0) / len(recent_verdicts)
            drift_score += abs(baseline_freq - recent_freq)

        drift_detected = drift_score > 0.3  # Threshold for drift detection

        return {
            "drift_detected": drift_detected,
            "drift_score": drift_score,
            "baseline_distribution": dict(baseline_dist),
            "recent_distribution": dict(recent_dist),
            "recommendation": "Retrain model" if drift_detected else "No action needed",
        }

    def export_model(self, filepath: str):
        """Export trained model to file"""
        import pickle

        model_data = {
            "classifier": self.classifier,
            "causal_model": self.causal_model,
            "mind_theory": self.mind_theory,
            "feature_names": self.feature_names,
            "training_timestamp": datetime.now().isoformat(),
            "version": "0.8.0",
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        self.logger.info(f"Model exported to {filepath}")

    def load_model(self, filepath: str):
        """Load trained model from file"""
        import pickle

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.classifier = model_data["classifier"]
        self.causal_model = model_data["causal_model"]
        self.mind_theory = model_data["mind_theory"]
        self.feature_names = model_data["feature_names"]

        self.logger.info(f"Model loaded from {filepath}")
