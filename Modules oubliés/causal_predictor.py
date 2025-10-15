"""
Jeffrey OS Phase 0.8 - Causal Predictor
Prediction with causal explanations and symbolic reasoning
"""

import json
import logging
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Optional dependencies
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    import sympy as sp
    from sympy import diff, latex, simplify, solve, symbols

    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

try:
    from sklearn.inspection import permutation_importance
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import pandas as pd

    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# Import Jeffrey OS components
from feedback.models import Decision, Proposal, VerdictType

from .feature_extractor import FeatureVector


@dataclass
class CausalFactor:
    """Represents a causal factor in decision making"""

    name: str
    influence: float
    direction: str  # positive, negative, neutral
    confidence: float
    evidence: list[str] = field(default_factory=list)
    symbolic_expression: str | None = None
    correlation: float = 0.0
    causal_strength: float = 0.0


@dataclass
class CausalExplanation:
    """Complete causal explanation for a prediction"""

    prediction: VerdictType
    confidence: float
    primary_cause: CausalFactor
    contributing_factors: list[CausalFactor]
    symbolic_equation: str | None = None
    natural_language: str = ""
    causal_chain: list[str] = field(default_factory=list)
    counterfactual: str | None = None
    intervention_suggestions: list[str] = field(default_factory=list)


class CausalGraph:
    """Represents causal relationships between variables"""

    def __init__(self):
        self.graph = nx.DiGraph() if NETWORKX_AVAILABLE else None
        self.adjacency_matrix = {}
        self.node_attributes = {}
        self.edge_weights = {}

    def add_node(self, node_id: str, attributes: dict[str, Any]):
        """Add node to causal graph"""
        if self.graph:
            self.graph.add_node(node_id, **attributes)

        self.node_attributes[node_id] = attributes

    def add_edge(self, from_node: str, to_node: str, weight: float, causal_type: str = "direct"):
        """Add causal edge to graph"""
        if self.graph:
            self.graph.add_edge(from_node, to_node, weight=weight, causal_type=causal_type)

        if from_node not in self.adjacency_matrix:
            self.adjacency_matrix[from_node] = {}

        self.adjacency_matrix[from_node][to_node] = {"weight": weight, "causal_type": causal_type}

        self.edge_weights[(from_node, to_node)] = weight

    def get_parents(self, node_id: str) -> list[str]:
        """Get parent nodes (causes) of a node"""
        if self.graph:
            return list(self.graph.predecessors(node_id))

        parents = []
        for from_node, targets in self.adjacency_matrix.items():
            if node_id in targets:
                parents.append(from_node)

        return parents

    def get_children(self, node_id: str) -> list[str]:
        """Get child nodes (effects) of a node"""
        if self.graph:
            return list(self.graph.successors(node_id))

        return self.adjacency_matrix.get(node_id, {}).keys()

    def get_causal_path(self, from_node: str, to_node: str) -> list[str]:
        """Get causal path between two nodes"""
        if self.graph and nx.has_path(self.graph, from_node, to_node):
            return nx.shortest_path(self.graph, from_node, to_node)

        # Simple path finding for non-networkx case
        return self._find_simple_path(from_node, to_node)

    def _find_simple_path(self, start: str, end: str, visited: set | None = None) -> list[str]:
        """Simple path finding without networkx"""
        if visited is None:
            visited = set()

        if start == end:
            return [start]

        if start in visited:
            return []

        visited.add(start)

        for neighbor in self.adjacency_matrix.get(start, {}):
            path = self._find_simple_path(neighbor, end, visited.copy())
            if path:
                return [start] + path

        return []

    def calculate_total_effect(self, from_node: str, to_node: str) -> float:
        """Calculate total causal effect between nodes"""
        path = self.get_causal_path(from_node, to_node)

        if len(path) < 2:
            return 0.0

        # Multiply edge weights along path
        total_effect = 1.0
        for i in range(len(path) - 1):
            edge_weight = self.edge_weights.get((path[i], path[i + 1]), 0.0)
            total_effect *= edge_weight

        return total_effect

    def export_structure(self) -> dict[str, Any]:
        """Export graph structure"""
        return {
            "nodes": self.node_attributes,
            "edges": self.adjacency_matrix,
            "edge_weights": self.edge_weights,
        }


class SymbolicReasoningEngine:
    """Symbolic reasoning for causal relationships"""

    def __init__(self):
        self.symbols = {}
        self.equations = {}
        self.constraints = {}

        if SYMPY_AVAILABLE:
            self.sp = sp
        else:
            self.sp = None

    def create_symbol(self, name: str, description: str = "") -> Any:
        """Create symbolic variable"""
        if self.sp:
            symbol = self.sp.Symbol(name, real=True)
            self.symbols[name] = {"symbol": symbol, "description": description, "constraints": []}
            return symbol
        else:
            # Fallback representation
            self.symbols[name] = {"symbol": name, "description": description, "constraints": []}
            return name

    def create_equation(self, equation_name: str, expression: str, variables: list[str]) -> Any:
        """Create symbolic equation"""
        if not self.sp:
            # Store string representation
            self.equations[equation_name] = {
                "expression": expression,
                "variables": variables,
                "symbolic": None,
            }
            return expression

        # Create symbols for variables
        var_symbols = {}
        for var in variables:
            if var not in self.symbols:
                var_symbols[var] = self.create_symbol(var)
            else:
                var_symbols[var] = self.symbols[var]["symbol"]

        # Parse expression
        try:
            # Replace variable names with symbols
            expr = expression
            for var, symbol in var_symbols.items():
                expr = expr.replace(var, str(symbol))

            symbolic_expr = self.sp.sympify(expr)

            self.equations[equation_name] = {
                "expression": expression,
                "variables": variables,
                "symbolic": symbolic_expr,
                "var_symbols": var_symbols,
            }

            return symbolic_expr

        except Exception as e:
            logging.warning(f"Failed to create symbolic equation: {e}")
            self.equations[equation_name] = {
                "expression": expression,
                "variables": variables,
                "symbolic": None,
            }
            return expression

    def evaluate_equation(self, equation_name: str, variable_values: dict[str, float]) -> float:
        """Evaluate equation with given variable values"""
        if equation_name not in self.equations:
            return 0.0

        equation = self.equations[equation_name]

        if equation["symbolic"] and self.sp:
            # Substitute values
            substitutions = {}
            for var, value in variable_values.items():
                if var in equation["var_symbols"]:
                    substitutions[equation["var_symbols"][var]] = value

            try:
                result = float(equation["symbolic"].subs(substitutions))
                return result
            except:
                return 0.0

        # Fallback evaluation
        return self._evaluate_string_expression(equation["expression"], variable_values)

    def _evaluate_string_expression(self, expression: str, values: dict[str, float]) -> float:
        """Evaluate string expression (fallback)"""
        try:
            # Simple substitution
            expr = expression
            for var, value in values.items():
                expr = expr.replace(var, str(value))

            # Basic evaluation (limited for security)
            if re.match(r"^[0-9+\-*/().\s]+$", expr):
                return eval(expr)

            return 0.0
        except:
            return 0.0

    def differentiate(self, equation_name: str, variable: str) -> Any:
        """Calculate partial derivative"""
        if not self.sp or equation_name not in self.equations:
            return None

        equation = self.equations[equation_name]
        if not equation["symbolic"]:
            return None

        if variable not in equation["var_symbols"]:
            return None

        try:
            derivative = self.sp.diff(equation["symbolic"], equation["var_symbols"][variable])
            return derivative
        except:
            return None

    def solve_equation(self, equation_name: str, target_variable: str) -> list[Any]:
        """Solve equation for target variable"""
        if not self.sp or equation_name not in self.equations:
            return []

        equation = self.equations[equation_name]
        if not equation["symbolic"]:
            return []

        if target_variable not in equation["var_symbols"]:
            return []

        try:
            solutions = self.sp.solve(equation["symbolic"], equation["var_symbols"][target_variable])
            return solutions
        except:
            return []

    def simplify_expression(self, expression: Any) -> Any:
        """Simplify symbolic expression"""
        if self.sp and hasattr(expression, "simplify"):
            return self.sp.simplify(expression)
        return expression

    def to_latex(self, expression: Any) -> str:
        """Convert expression to LaTeX"""
        if self.sp and hasattr(expression, "latex"):
            return self.sp.latex(expression)
        return str(expression)


class CausalPredictor:
    """
    Causal predictor with symbolic reasoning and explanations
    """

    def __init__(self, data_dir: str = "data/causal"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Core components
        self.causal_graph = CausalGraph()
        self.symbolic_engine = SymbolicReasoningEngine()

        # Learned relationships
        self.causal_factors = {}
        self.factor_weights = {}
        self.interaction_effects = {}

        # Models
        self.base_model = None
        self.causal_model = None

        # Database
        self.db_path = self.data_dir / "causal_predictions.db"
        self._init_database()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """Initialize database for causal predictions"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Causal factors table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS causal_factors (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    influence REAL NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence TEXT,
                    symbolic_expression TEXT,
                    correlation REAL,
                    causal_strength REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Causal relationships table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS causal_relationships (
                    id TEXT PRIMARY KEY,
                    from_factor TEXT NOT NULL,
                    to_factor TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    strength REAL NOT NULL,
                    evidence TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Predictions table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS causal_predictions (
                    id TEXT PRIMARY KEY,
                    proposal_id TEXT,
                    predicted_verdict TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    primary_cause TEXT,
                    contributing_factors TEXT,
                    symbolic_equation TEXT,
                    natural_language TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            # Explanations table
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS explanations (
                    id TEXT PRIMARY KEY,
                    prediction_id TEXT,
                    explanation_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL,
                    language TEXT DEFAULT 'en',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            conn.commit()

    def build_causal_graph(
        self,
        proposals: list[Proposal],
        decisions: list[Decision],
        feature_vectors: list[FeatureVector],
    ) -> CausalGraph:
        """Build causal graph from training data"""
        self.logger.info("Building causal graph from training data")

        # Extract features and outcomes
        all_features = {}
        outcomes = []

        for i, (proposal, decision, features) in enumerate(zip(proposals, decisions, feature_vectors)):
            # Collect all features
            feature_dict = {}
            feature_dict.update(features.basic_features)
            feature_dict.update(features.causal_features)
            feature_dict.update(features.temporal_features)
            feature_dict.update(features.emotional_features)
            feature_dict.update(features.contextual_features)

            for feature_name, value in feature_dict.items():
                if feature_name not in all_features:
                    all_features[feature_name] = []
                all_features[feature_name].append(value)

            outcomes.append(decision.verdict.value)

        # Add nodes to graph
        for feature_name in all_features.keys():
            self.causal_graph.add_node(
                feature_name,
                {
                    "type": "feature",
                    "mean": np.mean(all_features[feature_name]),
                    "std": np.std(all_features[feature_name]),
                },
            )

        # Add outcome node
        self.causal_graph.add_node("verdict", {"type": "outcome", "classes": list(set(outcomes))})

        # Learn causal relationships
        self._learn_causal_relationships(all_features, outcomes)

        return self.causal_graph

    def _learn_causal_relationships(self, features: dict[str, list[float]], outcomes: list[str]):
        """Learn causal relationships between features and outcomes"""
        if not SKLEARN_AVAILABLE:
            self.logger.warning("Scikit-learn not available, using simplified causal learning")
            return self._learn_simple_relationships(features, outcomes)

        # Prepare data
        feature_names = list(features.keys())
        X = np.array([features[name] for name in feature_names]).T

        # Encode outcomes
        outcome_map = {"accept": 0, "reject": 1, "defer": 2}
        y = np.array([outcome_map.get(outcome, 0) for outcome in outcomes])

        # Train interpretable model
        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        # Get feature importances
        importances = np.abs(model.coef_[0])

        # Create causal edges
        for i, feature_name in enumerate(feature_names):
            importance = importances[i]

            if importance > 0.1:  # Threshold for significance
                # Determine direction
                direction = "positive" if model.coef_[0][i] > 0 else "negative"

                # Add edge to graph
                self.causal_graph.add_edge(feature_name, "verdict", weight=importance, causal_type="direct")

                # Store causal factor
                self.causal_factors[feature_name] = CausalFactor(
                    name=feature_name,
                    influence=importance,
                    direction=direction,
                    confidence=min(1.0, importance * 2),
                    evidence=[f"Logistic regression coefficient: {model.coef_[0][i]:.3f}"],
                    correlation=importance,
                    causal_strength=importance,
                )

        # Store base model
        self.base_model = model

        # Create symbolic equations
        self._create_symbolic_equations(feature_names, model)

    def _learn_simple_relationships(self, features: dict[str, list[float]], outcomes: list[str]):
        """Simple causal learning fallback"""
        # Calculate correlations
        outcome_numeric = [1 if o == "accept" else 0 for o in outcomes]

        for feature_name, values in features.items():
            if len(values) > 1:
                # Simple correlation
                correlation = np.corrcoef(values, outcome_numeric)[0, 1]

                if abs(correlation) > 0.3:  # Threshold
                    direction = "positive" if correlation > 0 else "negative"

                    self.causal_graph.add_edge(
                        feature_name, "verdict", weight=abs(correlation), causal_type="correlation"
                    )

                    self.causal_factors[feature_name] = CausalFactor(
                        name=feature_name,
                        influence=abs(correlation),
                        direction=direction,
                        confidence=abs(correlation),
                        evidence=[f"Correlation: {correlation:.3f}"],
                        correlation=correlation,
                        causal_strength=abs(correlation),
                    )

    def _create_symbolic_equations(self, feature_names: list[str], model: Any):
        """Create symbolic equations from trained model"""
        if not SYMPY_AVAILABLE:
            return

        # Create symbols for features
        for feature_name in feature_names:
            self.symbolic_engine.create_symbol(feature_name, f"Feature: {feature_name}")

        # Create logistic regression equation
        # P(accept) = 1 / (1 + exp(-(w0 + w1*x1 + w2*x2 + ...)))

        # Build linear combination
        linear_expr = f"{model.intercept_[0]}"
        for i, feature_name in enumerate(feature_names):
            coef = model.coef_[0][i]
            if coef != 0:
                linear_expr += f" + {coef} * {feature_name}"

        # Create logistic equation
        logistic_expr = f"1 / (1 + exp(-({linear_expr})))"

        self.symbolic_engine.create_equation("acceptance_probability", logistic_expr, feature_names)

        # Create decision equation
        decision_expr = f"1 if {logistic_expr} > 0.5 else 0"
        self.symbolic_engine.create_equation("decision", decision_expr, feature_names)

    def predict_with_explanation(self, proposal: Proposal, features: FeatureVector) -> CausalExplanation:
        """Make prediction with full causal explanation"""
        # Extract feature values
        feature_values = {}
        feature_values.update(features.basic_features)
        feature_values.update(features.causal_features)
        feature_values.update(features.temporal_features)
        feature_values.update(features.emotional_features)
        feature_values.update(features.contextual_features)

        # Make base prediction
        if self.base_model and SKLEARN_AVAILABLE:
            prediction, confidence = self._predict_with_model(feature_values)
        else:
            prediction, confidence = self._predict_simple(feature_values)

        # Find primary cause
        primary_cause = self._identify_primary_cause(feature_values)

        # Find contributing factors
        contributing_factors = self._identify_contributing_factors(feature_values)

        # Generate symbolic equation
        symbolic_equation = self._generate_symbolic_explanation(feature_values)

        # Generate natural language explanation
        natural_language = self._generate_natural_explanation(
            prediction, primary_cause, contributing_factors, features.metadata.get("language", "en")
        )

        # Generate causal chain
        causal_chain = self._generate_causal_chain(primary_cause, contributing_factors)

        # Generate counterfactual
        counterfactual = self._generate_counterfactual(feature_values, prediction)

        # Generate intervention suggestions
        interventions = self._generate_interventions(feature_values, prediction)

        explanation = CausalExplanation(
            prediction=prediction,
            confidence=confidence,
            primary_cause=primary_cause,
            contributing_factors=contributing_factors,
            symbolic_equation=symbolic_equation,
            natural_language=natural_language,
            causal_chain=causal_chain,
            counterfactual=counterfactual,
            intervention_suggestions=interventions,
        )

        # Store prediction
        self._store_prediction(proposal.id, explanation)

        return explanation

    def _predict_with_model(self, feature_values: dict[str, float]) -> tuple[VerdictType, float]:
        """Make prediction using trained model"""
        # Prepare feature vector
        feature_names = list(self.causal_factors.keys())
        X = np.array([[feature_values.get(name, 0.0) for name in feature_names]])

        # Predict
        proba = self.base_model.predict_proba(X)[0]
        prediction_idx = np.argmax(proba)
        confidence = proba[prediction_idx]

        # Map to verdict
        verdict_map = {0: VerdictType.ACCEPT, 1: VerdictType.REJECT, 2: VerdictType.DEFER}
        prediction = verdict_map.get(prediction_idx, VerdictType.ACCEPT)

        return prediction, confidence

    def _predict_simple(self, feature_values: dict[str, float]) -> tuple[VerdictType, float]:
        """Simple prediction fallback"""
        # Calculate weighted sum of causal factors
        score = 0.0
        total_weight = 0.0

        for factor_name, factor in self.causal_factors.items():
            value = feature_values.get(factor_name, 0.0)
            weight = factor.influence

            if factor.direction == "positive":
                score += value * weight
            else:
                score -= value * weight

            total_weight += weight

        # Normalize
        if total_weight > 0:
            score /= total_weight

        # Map to verdict
        if score > 0.2:
            return VerdictType.ACCEPT, min(1.0, score)
        elif score < -0.2:
            return VerdictType.REJECT, min(1.0, abs(score))
        else:
            return VerdictType.DEFER, 0.5

    def _identify_primary_cause(self, feature_values: dict[str, float]) -> CausalFactor:
        """Identify primary causal factor"""
        if not self.causal_factors:
            return CausalFactor(name="unknown", influence=0.0, direction="neutral", confidence=0.0)

        # Find factor with highest influence * value
        best_factor = None
        best_score = 0.0

        for factor_name, factor in self.causal_factors.items():
            value = feature_values.get(factor_name, 0.0)
            score = factor.influence * abs(value)

            if score > best_score:
                best_score = score
                best_factor = factor

        return best_factor or list(self.causal_factors.values())[0]

    def _identify_contributing_factors(self, feature_values: dict[str, float]) -> list[CausalFactor]:
        """Identify contributing causal factors"""
        factors = []

        for factor_name, factor in self.causal_factors.items():
            value = feature_values.get(factor_name, 0.0)
            contribution = factor.influence * abs(value)

            if contribution > 0.1:  # Threshold
                factors.append(factor)

        # Sort by influence
        factors.sort(key=lambda f: f.influence, reverse=True)

        return factors[:5]  # Top 5 factors

    def _generate_symbolic_explanation(self, feature_values: dict[str, float]) -> str:
        """Generate symbolic explanation"""
        if not SYMPY_AVAILABLE or "acceptance_probability" not in self.symbolic_engine.equations:
            return self._generate_simple_equation(feature_values)

        # Evaluate symbolic equation
        try:
            probability = self.symbolic_engine.evaluate_equation("acceptance_probability", feature_values)

            # Get the symbolic expression
            equation = self.symbolic_engine.equations["acceptance_probability"]
            if equation["symbolic"]:
                return f"P(accept) = {equation['symbolic']} = {probability:.3f}"
            else:
                return f"P(accept) = {probability:.3f}"

        except Exception as e:
            self.logger.warning(f"Failed to generate symbolic explanation: {e}")
            return self._generate_simple_equation(feature_values)

    def _generate_simple_equation(self, feature_values: dict[str, float]) -> str:
        """Generate simple equation explanation"""
        terms = []

        for factor_name, factor in self.causal_factors.items():
            value = feature_values.get(factor_name, 0.0)
            if abs(value) > 0.01:  # Threshold
                sign = "+" if factor.direction == "positive" else "-"
                terms.append(f"{sign}{factor.influence:.2f}*{factor_name}")

        if terms:
            return f"Score = {' '.join(terms)}"
        else:
            return "Score = 0.5 (baseline)"

    def _generate_natural_explanation(
        self,
        prediction: VerdictType,
        primary_cause: CausalFactor,
        contributing_factors: list[CausalFactor],
        language: str = "en",
    ) -> str:
        """Generate natural language explanation"""
        templates = {
            "en": {
                "accept": "I recommend accepting this proposal primarily because {primary_cause} is favorable. {contributing_factors}",
                "reject": "I recommend rejecting this proposal primarily due to concerns about {primary_cause}. {contributing_factors}",
                "defer": "I suggest deferring this proposal to further evaluate {primary_cause}. {contributing_factors}",
            },
            "fr": {
                "accept": "Je recommande d'accepter cette proposition principalement parce que {primary_cause} est favorable. {contributing_factors}",
                "reject": "Je recommande de rejeter cette proposition principalement en raison de préoccupations concernant {primary_cause}. {contributing_factors}",
                "defer": "Je suggère de reporter cette proposition pour évaluer davantage {primary_cause}. {contributing_factors}",
            },
            "es": {
                "accept": "Recomiendo aceptar esta propuesta principalmente porque {primary_cause} es favorable. {contributing_factors}",
                "reject": "Recomiendo rechazar esta propuesta principalmente debido a preocupaciones sobre {primary_cause}. {contributing_factors}",
                "defer": "Sugiero diferir esta propuesta para evaluar más {primary_cause}. {contributing_factors}",
            },
        }

        # Get template
        lang_templates = templates.get(language, templates["en"])
        template = lang_templates.get(prediction.value, lang_templates["accept"])

        # Format contributing factors
        contrib_text = ""
        if contributing_factors:
            contrib_names = [f.name for f in contributing_factors[:3]]
            if language == "en":
                contrib_text = f"Additionally, {', '.join(contrib_names)} are also influential factors."
            elif language == "fr":
                contrib_text = f"De plus, {', '.join(contrib_names)} sont aussi des facteurs influents."
            elif language == "es":
                contrib_text = f"Además, {', '.join(contrib_names)} también son factores influyentes."

        return template.format(primary_cause=primary_cause.name, contributing_factors=contrib_text)

    def _generate_causal_chain(
        self, primary_cause: CausalFactor, contributing_factors: list[CausalFactor]
    ) -> list[str]:
        """Generate causal chain of reasoning"""
        chain = []

        # Start with primary cause
        chain.append(f"Primary cause: {primary_cause.name} (influence: {primary_cause.influence:.2f})")

        # Add contributing factors
        for factor in contributing_factors[:3]:
            chain.append(f"Contributing factor: {factor.name} (influence: {factor.influence:.2f})")

        # Add causal paths if available
        if primary_cause.name in self.causal_graph.node_attributes:
            path = self.causal_graph.get_causal_path(primary_cause.name, "verdict")
            if len(path) > 1:
                chain.append(f"Causal path: {' -> '.join(path)}")

        return chain

    def _generate_counterfactual(self, feature_values: dict[str, float], prediction: VerdictType) -> str:
        """Generate counterfactual explanation"""
        if not self.causal_factors:
            return "No counterfactual available"

        # Find the most influential factor
        primary_factor = self._identify_primary_cause(feature_values)

        # Generate counterfactual
        if prediction == VerdictType.ACCEPT:
            return f"If {primary_factor.name} were lower, the proposal would likely be rejected."
        elif prediction == VerdictType.REJECT:
            return f"If {primary_factor.name} were higher, the proposal would likely be accepted."
        else:
            return f"If {primary_factor.name} were more decisive, the proposal would have a clearer outcome."

    def _generate_interventions(self, feature_values: dict[str, float], prediction: VerdictType) -> list[str]:
        """Generate intervention suggestions"""
        interventions = []

        if prediction == VerdictType.REJECT:
            # Suggest ways to improve acceptance
            for factor_name, factor in self.causal_factors.items():
                if factor.direction == "positive" and factor.influence > 0.3:
                    interventions.append(f"Improve {factor_name} to increase acceptance probability")
                elif factor.direction == "negative" and factor.influence > 0.3:
                    interventions.append(f"Reduce {factor_name} to increase acceptance probability")

        elif prediction == VerdictType.DEFER:
            # Suggest ways to make decision clearer
            interventions.append("Provide more detailed information to clarify the decision")
            interventions.append("Address key concerns in the proposal")

        return interventions[:3]  # Top 3 interventions

    def _store_prediction(self, proposal_id: str, explanation: CausalExplanation):
        """Store prediction and explanation"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            prediction_id = f"pred_{proposal_id}_{int(datetime.now().timestamp())}"

            cursor.execute(
                """
                INSERT INTO causal_predictions (
                    id, proposal_id, predicted_verdict, confidence, primary_cause,
                    contributing_factors, symbolic_equation, natural_language
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    prediction_id,
                    proposal_id,
                    explanation.prediction.value,
                    explanation.confidence,
                    explanation.primary_cause.name,
                    json.dumps([f.name for f in explanation.contributing_factors]),
                    explanation.symbolic_equation,
                    explanation.natural_language,
                ),
            )

            conn.commit()

    def explain_feature_importance(self, feature_name: str) -> dict[str, Any]:
        """Explain importance of a specific feature"""
        if feature_name not in self.causal_factors:
            return {"error": f"Feature {feature_name} not found in causal model"}

        factor = self.causal_factors[feature_name]

        explanation = {
            "feature_name": feature_name,
            "influence": factor.influence,
            "direction": factor.direction,
            "confidence": factor.confidence,
            "evidence": factor.evidence,
            "causal_paths": [],
            "interactions": [],
        }

        # Find causal paths
        if feature_name in self.causal_graph.node_attributes:
            path = self.causal_graph.get_causal_path(feature_name, "verdict")
            explanation["causal_paths"] = [path] if path else []

        # Find interactions
        for other_feature, other_factor in self.causal_factors.items():
            if other_feature != feature_name:
                interaction_key = (feature_name, other_feature)
                if interaction_key in self.interaction_effects:
                    explanation["interactions"].append(
                        {"with": other_feature, "effect": self.interaction_effects[interaction_key]}
                    )

        return explanation

    def analyze_decision_boundary(self, feature_vectors: list[FeatureVector]) -> dict[str, Any]:
        """Analyze decision boundary and feature sensitivity"""
        if not feature_vectors:
            return {"error": "No feature vectors provided"}

        # Extract feature values
        all_features = []
        for features in feature_vectors:
            feature_dict = {}
            feature_dict.update(features.basic_features)
            feature_dict.update(features.causal_features)
            feature_dict.update(features.temporal_features)
            feature_dict.update(features.emotional_features)
            feature_dict.update(features.contextual_features)
            all_features.append(feature_dict)

        # Analyze sensitivity
        sensitivity_analysis = {}

        for feature_name in self.causal_factors.keys():
            # Calculate feature range
            values = [f.get(feature_name, 0.0) for f in all_features]
            if values:
                sensitivity_analysis[feature_name] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "influence": self.causal_factors[feature_name].influence,
                }

        return {
            "sensitivity_analysis": sensitivity_analysis,
            "most_sensitive_features": sorted(
                sensitivity_analysis.keys(),
                key=lambda x: sensitivity_analysis[x]["influence"],
                reverse=True,
            )[:5],
        }

    def generate_what_if_analysis(self, base_features: FeatureVector, changes: dict[str, float]) -> dict[str, Any]:
        """Generate what-if analysis by changing feature values"""
        # Original prediction
        original_explanation = self.predict_with_explanation(Proposal(id="what_if_original"), base_features)

        # Create modified features
        modified_features = base_features

        # Apply changes
        for feature_name, new_value in changes.items():
            if feature_name in modified_features.basic_features:
                modified_features.basic_features[feature_name] = new_value
            elif feature_name in modified_features.causal_features:
                modified_features.causal_features[feature_name] = new_value
            elif feature_name in modified_features.temporal_features:
                modified_features.temporal_features[feature_name] = new_value
            elif feature_name in modified_features.emotional_features:
                modified_features.emotional_features[feature_name] = new_value
            elif feature_name in modified_features.contextual_features:
                modified_features.contextual_features[feature_name] = new_value

        # New prediction
        new_explanation = self.predict_with_explanation(Proposal(id="what_if_modified"), modified_features)

        return {
            "original_prediction": original_explanation.prediction.value,
            "original_confidence": original_explanation.confidence,
            "new_prediction": new_explanation.prediction.value,
            "new_confidence": new_explanation.confidence,
            "changes_made": changes,
            "prediction_changed": original_explanation.prediction != new_explanation.prediction,
            "confidence_delta": new_explanation.confidence - original_explanation.confidence,
            "explanation_comparison": {
                "original": original_explanation.natural_language,
                "modified": new_explanation.natural_language,
            },
        }

    def export_causal_model(self, filepath: str):
        """Export causal model to file"""
        export_data = {
            "causal_factors": {
                name: {
                    "name": factor.name,
                    "influence": factor.influence,
                    "direction": factor.direction,
                    "confidence": factor.confidence,
                    "evidence": factor.evidence,
                    "correlation": factor.correlation,
                    "causal_strength": factor.causal_strength,
                }
                for name, factor in self.causal_factors.items()
            },
            "causal_graph": self.causal_graph.export_structure(),
            "symbolic_equations": {
                name: {"expression": eq["expression"], "variables": eq["variables"]}
                for name, eq in self.symbolic_engine.equations.items()
            },
            "export_timestamp": datetime.now().isoformat(),
            "version": "0.8.0",
        }

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Causal model exported to {filepath}")

    def load_causal_model(self, filepath: str):
        """Load causal model from file"""
        with open(filepath) as f:
            data = json.load(f)

        # Load causal factors
        self.causal_factors = {}
        for name, factor_data in data.get("causal_factors", {}).items():
            self.causal_factors[name] = CausalFactor(
                name=factor_data["name"],
                influence=factor_data["influence"],
                direction=factor_data["direction"],
                confidence=factor_data["confidence"],
                evidence=factor_data["evidence"],
                correlation=factor_data.get("correlation", 0.0),
                causal_strength=factor_data.get("causal_strength", 0.0),
            )

        # Load causal graph structure
        graph_data = data.get("causal_graph", {})
        self.causal_graph = CausalGraph()

        for node_id, attributes in graph_data.get("nodes", {}).items():
            self.causal_graph.add_node(node_id, attributes)

        for from_node, edges in graph_data.get("edges", {}).items():
            for to_node, edge_data in edges.items():
                self.causal_graph.add_edge(
                    from_node, to_node, edge_data["weight"], edge_data.get("causal_type", "direct")
                )

        # Load symbolic equations
        equations_data = data.get("symbolic_equations", {})
        for name, eq_data in equations_data.items():
            self.symbolic_engine.create_equation(name, eq_data["expression"], eq_data["variables"])

        self.logger.info(f"Causal model loaded from {filepath}")

    def get_model_statistics(self) -> dict[str, Any]:
        """Get causal model statistics"""
        stats = {
            "causal_factors_count": len(self.causal_factors),
            "causal_graph_nodes": len(self.causal_graph.node_attributes),
            "causal_graph_edges": len(self.causal_graph.edge_weights),
            "symbolic_equations": len(self.symbolic_engine.equations),
            "top_causal_factors": [],
            "model_complexity": 0.0,
        }

        # Top causal factors
        if self.causal_factors:
            sorted_factors = sorted(self.causal_factors.items(), key=lambda x: x[1].influence, reverse=True)

            stats["top_causal_factors"] = [
                {
                    "name": name,
                    "influence": factor.influence,
                    "direction": factor.direction,
                    "confidence": factor.confidence,
                }
                for name, factor in sorted_factors[:10]
            ]

        # Model complexity
        stats["model_complexity"] = (
            len(self.causal_factors) * 0.1
            + len(self.causal_graph.edge_weights) * 0.05
            + len(self.symbolic_engine.equations) * 0.2
        )

        return stats
