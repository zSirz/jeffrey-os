"""
Variant Generator for Jeffrey OS DreamMode
Creative variant generation with ML mutations and epsilon-greedy exploration.
"""

import random
from typing import Any

import numpy as np
import torch

from .neural_mutator import NeuralMutator


class VariantGenerator:
    """
    Générateur de variantes créatives avec mutations ML.
    Inclut exploration epsilon-greedy pour éviter convergence.
    """

    def __init__(self, neural_mutator: NeuralMutator, seed: int = None, exploration_rate: float = 0.1):
        self.neural_mutator = neural_mutator
        self.causal_predictor = None  # Lazy load
        self.exploration_rate = exploration_rate  # Epsilon for exploration

        # Synchroniser TOUS les seeds pour reproductibilité totale
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        self.mutation_strategies = {
            "decompose": self._decompose_strategy,
            "inverse": self._inverse_strategy,
            "combine": self._combine_strategy,
            "simplify": self._simplify_strategy,
            "amplify": self._amplify_strategy,
            "lateral": self._lateral_strategy,
            "neural": self._neural_strategy,
            "symbolic": self._symbolic_strategy,  # Nouveau avec sympy
            "probabilistic": self._probabilistic_strategy,  # Nouveau pour incertitude
        }

        # Strategy performance tracking
        self.strategy_performance = {name: {"successes": 0, "attempts": 0} for name in self.mutation_strategies.keys()}

    async def generate(
        self, base: dict, preferences: dict, num_variants: int = 5, user_state: dict = None
    ) -> list[dict]:
        """
        Génère des variantes créatives avec mix de stratégies et ML.
        Inclut exploration pour éviter biais.
        """
        # Lazy load du causal predictor avec validation
        if not self.causal_predictor:
            try:
                from ..learning.causal_predictor import CausalPredictor

                self.causal_predictor = CausalPredictor()
            except ImportError:
                # Fallback si module pas disponible
                self.causal_predictor = None

        variants = []

        # Analyser pourquoi la base a été rejetée (si applicable)
        rejection_reasons = await self._analyze_rejection_reasons(base)

        # Exploration vs Exploitation with performance weighting
        if random.random() < self.exploration_rate:
            # Mode exploration : stratégies aléatoires
            selected_strategies = random.sample(
                list(self.mutation_strategies.keys()),
                min(num_variants, len(self.mutation_strategies)),
            )
        else:
            # Mode exploitation : stratégies basées sur performance + contexte
            selected_strategies = self._select_best_strategies(rejection_reasons, preferences, num_variants)

        # Appliquer les stratégies avec error handling robuste
        for strategy_name in selected_strategies:
            try:
                strategy = self.mutation_strategies[strategy_name]
                variant = await strategy(base, rejection_reasons, preferences, user_state)

                if variant:
                    variant.update(
                        {
                            "strategy": strategy_name,
                            "exploration_mode": random.random() < self.exploration_rate,
                            "strategy_confidence": self._calculate_strategy_confidence(strategy_name),
                            "generation_timestamp": np.datetime64("now").astype(str),
                        }
                    )
                    variants.append(variant)

                # Track strategy attempt
                self.strategy_performance[strategy_name]["attempts"] += 1

            except Exception as e:
                # Log error but continue with other strategies
                print(f"Strategy {strategy_name} failed: {e}")
                continue

        # Toujours inclure au moins une variante neurale si pas déjà présente
        neural_strategies = [v["strategy"] for v in variants if v.get("strategy") == "neural"]
        if not neural_strategies and len(variants) < num_variants:
            try:
                neural_variant = await self._neural_strategy(base, rejection_reasons, preferences, user_state)
                if neural_variant:
                    neural_variant.update({"strategy": "neural", "fallback_generation": True})
                    variants.append(neural_variant)
            except Exception as e:
                print(f"Neural fallback strategy failed: {e}")

        # Diversité post-processing
        variants = self._ensure_diversity(variants, min_diversity=0.3)

        return variants[:num_variants]

    async def _neural_strategy(self, base: dict, reasons: dict, prefs: dict, state: dict) -> dict:
        """
        Utilise le réseau de neurones pour générer une variante créative.
        """
        # Convertir la base en embedding
        base_embedding = self._create_embedding(base, prefs, state)

        # Déterminer le niveau de créativité basé sur l'état utilisateur
        creativity_level = self._calculate_creativity_level(state)

        # Ajouter randomness en mode exploration
        if random.random() < self.exploration_rate:
            creativity_level = min(1.0, creativity_level + random.uniform(0.1, 0.3))

        # Générer variante via neural mutator
        variant_embedding = self.neural_mutator.generate_variant(base_embedding, creativity_level=creativity_level)

        # Décoder l'embedding en proposition
        variant = self._decode_embedding(variant_embedding, base)

        # Enrichir avec métadonnées
        variant.update(
            {
                "type": "neural_generated",
                "creativity_level": creativity_level,
                "confidence": self._calculate_neural_confidence(variant_embedding),
                "learned_from": await self._find_similar_successes(variant_embedding),
                "embedding_distance": float(np.linalg.norm(base_embedding - variant_embedding)),
            }
        )

        return variant

    async def _symbolic_strategy(self, base: dict, reasons: dict, prefs: dict, state: dict) -> dict:
        """
        Utilise la manipulation symbolique pour mutations créatives.
        Particulièrement utile pour formules, équations, règles logiques.
        """
        try:
            import sympy as sp
        except ImportError:
            # Fallback to simplified algebraic manipulations
            return await self._algebraic_fallback_strategy(base, reasons, prefs, state)

        # Extraire les éléments symboliques
        symbolic_elements = self._extract_symbolic_elements(base)

        if not symbolic_elements:
            # Fallback si pas d'éléments symboliques
            return await self._lateral_strategy(base, reasons, prefs, state)

        # Appliquer transformations symboliques intelligentes
        transformations = [
            ("simplify", lambda x: sp.simplify(x)),
            ("expand", lambda x: sp.expand(x)),
            ("factor", lambda x: sp.factor(x)),
            ("collect", lambda x: sp.collect(x, sp.symbols("x y z"))),
            ("trigsimp", lambda x: sp.trigsimp(x)),
        ]

        # Choisir transformation basée sur le contexte
        transform_name, transform_func = self._select_symbolic_transform(transformations, symbolic_elements, prefs)

        variant = base.copy()
        transformed_count = 0

        for key, expr_str in symbolic_elements.items():
            try:
                # Parse expression
                expr = sp.sympify(expr_str)
                transformed = transform_func(expr)

                # Only use if transformation actually changed something
                if str(transformed) != str(expr):
                    variant[key] = str(transformed)
                    transformed_count += 1

            except Exception as e:
                # Log but continue with other expressions
                print(f"Symbolic transformation failed for {key}: {e}")
                continue

        if transformed_count == 0:
            # No successful transformations, fallback
            return await self._lateral_strategy(base, reasons, prefs, state)

        variant.update(
            {
                "type": "symbolic_mutation",
                "transformation": transform_name,
                "transformed_elements": transformed_count,
                "confidence": min(0.9, 0.5 + transformed_count * 0.2),
            }
        )

        return variant

    async def _probabilistic_strategy(self, base: dict, reasons: dict, prefs: dict, state: dict) -> dict:
        """
        Génération probabiliste avec gestion d'incertitude.
        """
        variant = base.copy()

        # Identifier les paramètres numériques
        numeric_params = self._extract_numeric_parameters(base)

        if not numeric_params:
            return await self._lateral_strategy(base, reasons, prefs, state)

        # Appliquer variations probabilistes
        for key, value in numeric_params.items():
            if isinstance(value, (int, float)):
                # Calculer variance basée sur l'incertitude contextuelle
                uncertainty = self._calculate_parameter_uncertainty(key, value, state)

                # Distribution normale avec variance adaptative
                std_dev = abs(value) * uncertainty if value != 0 else uncertainty
                new_value = np.random.normal(value, std_dev)

                # Contraintes réalistes
                if value > 0:
                    new_value = max(0.1 * value, new_value)  # Pas trop petit

                variant[key] = new_value

        # Ajouter métadonnées d'incertitude
        variant.update(
            {
                "type": "probabilistic_mutation",
                "uncertainty_level": np.mean(
                    [
                        self._calculate_parameter_uncertainty(k, v, state)
                        for k, v in numeric_params.items()
                        if isinstance(v, (int, float))
                    ]
                ),
                "parameters_modified": list(numeric_params.keys()),
                "confidence": 0.7,  # Moderate confidence pour variations probabilistes
            }
        )

        return variant

    async def _decompose_strategy(self, base: dict, reasons: dict, prefs: dict, state: dict) -> dict:
        """Décompose une proposition complexe en étapes plus simples."""
        variant = base.copy()

        # Identifier la complexité
        complexity = base.get("complexity", 0.5)
        if complexity < 0.4:
            # Déjà simple, essayer autre chose
            return await self._amplify_strategy(base, reasons, prefs, state)

        # Décomposer selon le type
        if "steps" in base:
            # Décomposer les étapes existantes
            original_steps = base["steps"]
            new_steps = []

            for step in original_steps:
                if isinstance(step, str) and len(step) > 50:
                    # Diviser les étapes longues
                    sub_steps = self._split_step(step)
                    new_steps.extend(sub_steps)
                else:
                    new_steps.append(step)

            variant["steps"] = new_steps
            variant["complexity"] = max(0.1, complexity - 0.3)

        elif "description" in base:
            # Créer des étapes depuis la description
            description = base["description"]
            steps = self._extract_steps_from_description(description)
            variant["steps"] = steps
            variant["complexity"] = max(0.2, complexity - 0.2)

        variant.update(
            {
                "type": "decomposed_approach",
                "confidence": 0.8,
                "rationale": "Broken down into simpler, manageable steps",
            }
        )

        return variant

    async def _inverse_strategy(self, base: dict, reasons: dict, prefs: dict, state: dict) -> dict:
        """Approche inverse du problème."""
        variant = base.copy()

        # Inverser la logique principale
        if "goal" in base:
            original_goal = base["goal"]
            variant["approach"] = f"Instead of directly {original_goal}, work backwards from the desired outcome"
            variant["methodology"] = "reverse_engineering"

        # Inverser les contraintes
        if "constraints" in base:
            constraints = base["constraints"]
            variant["opportunities"] = [f"Leverage: {c}" for c in constraints]

        # Inverser les priorités
        if "priority" in base:
            priority_map = {"high": "low", "low": "high", "medium": "medium"}
            variant["priority"] = priority_map.get(base["priority"], base["priority"])

        variant.update(
            {
                "type": "inverse_approach",
                "confidence": 0.75,
                "rationale": "Approached from opposite direction for fresh perspective",
            }
        )

        return variant

    async def _simplify_strategy(self, base: dict, reasons: dict, prefs: dict, state: dict) -> dict:
        """Simplifie l'approche."""
        variant = base.copy()

        # Réduire la complexité
        if "complexity" in base:
            variant["complexity"] = max(0.1, base["complexity"] - 0.4)

        # Simplifier les étapes
        if "steps" in base:
            steps = base["steps"]
            # Garder seulement les étapes essentielles
            essential_steps = steps[:3] if len(steps) > 3 else steps
            variant["steps"] = essential_steps

        # Simplifier la description
        if "description" in base:
            desc = base["description"]
            # Extraire l'essentiel
            simplified = self._extract_core_concept(desc)
            variant["description"] = simplified

        variant.update(
            {
                "type": "simplified_approach",
                "confidence": 0.85,
                "rationale": "Focused on core essentials for clarity and feasibility",
            }
        )

        return variant

    async def _amplify_strategy(self, base: dict, reasons: dict, prefs: dict, state: dict) -> dict:
        """Amplifie les aspects positifs."""
        variant = base.copy()

        # Identifier les points forts
        strengths = []
        if base.get("impact", 0) > 0.6:
            strengths.append("high_impact")
        if base.get("feasibility", 0) > 0.7:
            strengths.append("feasible")
        if base.get("innovation", 0) > 0.6:
            strengths.append("innovative")

        # Amplifier chaque force
        for strength in strengths:
            if strength == "high_impact":
                variant["impact"] = min(1.0, base.get("impact", 0.5) + 0.2)
                variant["scope"] = "expanded"
            elif strength == "feasible":
                variant["implementation_speed"] = "accelerated"
                variant["resource_efficiency"] = "optimized"
            elif strength == "innovative":
                variant["novelty"] = "breakthrough"
                variant["disruption_potential"] = "high"

        variant.update(
            {
                "type": "amplified_approach",
                "amplified_aspects": strengths,
                "confidence": 0.8,
                "rationale": f'Enhanced existing strengths: {", ".join(strengths)}',
            }
        )

        return variant

    async def _lateral_strategy(self, base: dict, reasons: dict, prefs: dict, state: dict) -> dict:
        """Pensée latérale créative."""
        variant = base.copy()

        # Techniques de pensée latérale
        lateral_techniques = [
            "random_word_association",
            "metaphor_thinking",
            "role_reversal",
            "assumption_challenging",
            "constraint_removal",
        ]

        technique = random.choice(lateral_techniques)

        if technique == "random_word_association":
            random_words = ["bridge", "music", "garden", "storm", "mirror", "dance", "mountain"]
            word = random.choice(random_words)
            variant["inspiration"] = f"Inspired by concept of '{word}'"
            variant["approach"] = f"Apply {word}-like principles to the solution"

        elif technique == "metaphor_thinking":
            metaphors = ["ecosystem", "symphony", "architecture", "recipe", "journey"]
            metaphor = random.choice(metaphors)
            variant["metaphor"] = metaphor
            variant["approach"] = f"Structure solution like a {metaphor}"

        elif technique == "role_reversal":
            variant["perspective"] = "reversed_roles"
            variant["approach"] = "Consider what the opposite stakeholder would do"

        elif technique == "assumption_challenging":
            variant["approach"] = "Challenge core assumptions and do the opposite"
            variant["contrarian"] = True

        elif technique == "constraint_removal":
            variant["approach"] = "Imagine unlimited resources and work backwards"
            variant["idealistic"] = True

        variant.update(
            {
                "type": "lateral_thinking",
                "technique": technique,
                "confidence": 0.6,  # Plus d'incertitude mais potentiel de créativité
                "rationale": f'Applied {technique.replace("_", " ")} for creative breakthrough',
            }
        )

        return variant

    async def _combine_strategy(self, base: dict, reasons: dict, prefs: dict, state: dict) -> dict:
        """Combine plusieurs approches."""
        variant = base.copy()

        # Combiner avec d'autres stratégies
        combination_strategies = ["decompose", "simplify", "amplify"]
        chosen_strategies = random.sample(combination_strategies, 2)

        # Appliquer séquentiellement
        for strategy_name in chosen_strategies:
            if strategy_name in self.mutation_strategies:
                strategy_func = self.mutation_strategies[strategy_name]
                temp_variant = await strategy_func(variant, reasons, prefs, state)
                if temp_variant:
                    # Merger les améliorations
                    variant.update(temp_variant)

        variant.update(
            {
                "type": "hybrid_approach",
                "combined_strategies": chosen_strategies,
                "confidence": 0.75,
                "rationale": f'Combined {" and ".join(chosen_strategies)} approaches',
            }
        )

        return variant

    # Helper methods

    def _create_embedding(self, base: dict, prefs: dict, state: dict) -> np.ndarray:
        """Crée un embedding vectoriel à partir des données."""
        features = []

        # Features de la base
        features.extend(
            [
                base.get("complexity", 0.5),
                base.get("impact", 0.5),
                base.get("feasibility", 0.5),
                len(base.get("steps", [])) / 10.0,  # Normaliser
            ]
        )

        # Features des préférences
        for pref_key in ["simplicity", "innovation", "risk_tolerance"]:
            features.append(prefs.get(pref_key, 0.5))

        # Features de l'état utilisateur
        if state:
            features.extend(
                [
                    state.get("mood_score", 0.5),
                    state.get("engagement_level", 0.5),
                    state.get("fatigue_level", 0.0),
                ]
            )
        else:
            features.extend([0.5, 0.5, 0.0])

        # Pad to expected dimension
        while len(features) < 64:
            features.append(0.0)

        return np.array(features[:64], dtype=np.float32)

    def _calculate_creativity_level(self, state: dict) -> float:
        """Calcule le niveau de créativité approprié."""
        if not state:
            return 0.3

        # Base level
        creativity = 0.3

        # Adjust based on user state
        mood = state.get("mood_score", 0.5)
        engagement = state.get("engagement_level", 0.5)
        fatigue = state.get("fatigue_level", 0.0)

        # Higher mood and engagement = more creativity
        creativity += (mood - 0.5) * 0.3
        creativity += (engagement - 0.5) * 0.2

        # Fatigue reduces creativity
        creativity -= fatigue * 0.3

        return np.clip(creativity, 0.1, 0.9)

    def _decode_embedding(self, embedding: np.ndarray, base: dict) -> dict:
        """Décode un embedding en proposition."""
        # Simple decoding - update base with embedding-derived changes
        variant = base.copy()

        # Extract key features from embedding
        complexity = float(embedding[0])
        impact = float(embedding[1])
        feasibility = float(embedding[2])

        variant.update(
            {
                "complexity": np.clip(complexity, 0.0, 1.0),
                "impact": np.clip(impact, 0.0, 1.0),
                "feasibility": np.clip(feasibility, 0.0, 1.0),
                "embedding_derived": True,
            }
        )

        return variant

    def _calculate_neural_confidence(self, embedding: np.ndarray) -> float:
        """Calcule la confiance basée sur l'embedding."""
        # Simple heuristic: variance in embedding indicates uncertainty
        variance = float(np.var(embedding))
        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + variance * 10)
        return np.clip(confidence, 0.1, 0.95)

    async def _find_similar_successes(self, embedding: np.ndarray) -> str | None:
        """Trouve des succès similaires pour apprentissage."""
        # Placeholder - would use actual success history
        return f"pattern_{int(np.sum(embedding) * 1000) % 100}"

    def _select_best_strategies(self, reasons: dict, prefs: dict, num_variants: int) -> list[str]:
        """Sélectionne les meilleures stratégies basées sur performance."""
        # Calculate strategy scores
        strategy_scores = {}

        for name, perf in self.strategy_performance.items():
            if perf["attempts"] > 0:
                success_rate = perf["successes"] / perf["attempts"]
                # Combine success rate with strategy appropriateness
                context_bonus = self._calculate_strategy_context_bonus(name, reasons, prefs)
                strategy_scores[name] = success_rate * 0.7 + context_bonus * 0.3
            else:
                # No history, use context bonus only
                strategy_scores[name] = self._calculate_strategy_context_bonus(name, reasons, prefs)

        # Select top strategies
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        return [name for name, score in sorted_strategies[:num_variants]]

    def _calculate_strategy_context_bonus(self, strategy_name: str, reasons: dict, prefs: dict) -> float:
        """Calcule un bonus de contexte pour une stratégie."""
        bonus = 0.5  # Base score

        # Strategy-specific context bonuses
        if strategy_name == "simplify" and prefs.get("simplicity", 0.5) > 0.7:
            bonus += 0.3
        elif strategy_name == "neural" and prefs.get("innovation", 0.5) > 0.6:
            bonus += 0.2
        elif strategy_name == "decompose" and reasons.get("complexity_issue", False):
            bonus += 0.4
        elif strategy_name == "amplify" and reasons.get("low_impact", False):
            bonus += 0.3

        return min(bonus, 1.0)

    def _ensure_diversity(self, variants: list[dict], min_diversity: float = 0.3) -> list[dict]:
        """Assure la diversité dans les variantes."""
        if len(variants) <= 1:
            return variants

        # Simple diversity check based on strategy types
        strategies_used = [v.get("strategy", "unknown") for v in variants]
        unique_strategies = len(set(strategies_used))
        diversity_ratio = unique_strategies / len(variants)

        if diversity_ratio < min_diversity and len(variants) > 2:
            # Remove least diverse variants (same strategy)
            strategy_counts = {}
            for v in variants:
                strategy = v.get("strategy", "unknown")
                strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

            # Keep one of each strategy, remove duplicates
            seen_strategies = set()
            diverse_variants = []

            for v in variants:
                strategy = v.get("strategy", "unknown")
                if strategy not in seen_strategies or strategy_counts[strategy] == 1:
                    diverse_variants.append(v)
                    seen_strategies.add(strategy)

            return diverse_variants

        return variants

    async def _analyze_rejection_reasons(self, base: dict) -> dict:
        """Analyse pourquoi une proposition pourrait être rejetée."""
        reasons = {}

        # Analyze common rejection factors
        if base.get("complexity", 0.5) > 0.8:
            reasons["complexity_issue"] = True
        if base.get("impact", 0.5) < 0.3:
            reasons["low_impact"] = True
        if base.get("feasibility", 0.5) < 0.4:
            reasons["feasibility_concern"] = True
        if base.get("risk", 0.5) > 0.7:
            reasons["high_risk"] = True

        return reasons

    def record_strategy_success(self, strategy_name: str, success: bool):
        """Enregistre le succès d'une stratégie pour l'apprentissage."""
        if strategy_name in self.strategy_performance:
            if success:
                self.strategy_performance[strategy_name]["successes"] += 1

    def get_strategy_stats(self) -> dict[str, dict]:
        """Retourne les statistiques de performance des stratégies."""
        return self.strategy_performance.copy()

    # Additional helper methods for specific strategies

    def _extract_symbolic_elements(self, base: dict) -> dict[str, str]:
        """Extrait les éléments symboliques d'une proposition."""
        symbolic = {}
        for key, value in base.items():
            if isinstance(value, str):
                # Look for mathematical expressions
                if any(symbol in value for symbol in ["=", "+", "-", "*", "/", "^", "x", "y"]):
                    symbolic[key] = value
        return symbolic

    def _select_symbolic_transform(self, transformations, elements, prefs):
        """Sélectionne une transformation symbolique appropriée."""
        # Simple selection - could be more sophisticated
        return random.choice(transformations)

    def _extract_numeric_parameters(self, base: dict) -> dict[str, Any]:
        """Extrait les paramètres numériques."""
        numeric = {}
        for key, value in base.items():
            if isinstance(value, (int, float)):
                numeric[key] = value
        return numeric

    def _calculate_parameter_uncertainty(self, key: str, value: Any, state: dict) -> float:
        """Calcule l'incertitude d'un paramètre."""
        # Base uncertainty
        uncertainty = 0.1

        # Increase uncertainty based on context
        if state and state.get("fatigue_level", 0) > 0.5:
            uncertainty += 0.2
        if "critical" in key.lower():
            uncertainty -= 0.05  # Less uncertainty for critical params
        if "experimental" in key.lower():
            uncertainty += 0.3  # More uncertainty for experimental params

        return np.clip(uncertainty, 0.05, 0.5)

    def _calculate_strategy_confidence(self, strategy_name: str) -> float:
        """Calcule la confiance dans une stratégie."""
        perf = self.strategy_performance.get(strategy_name, {"successes": 0, "attempts": 0})
        if perf["attempts"] == 0:
            return 0.5  # Default confidence
        return perf["successes"] / perf["attempts"]
