"""
Cost metric collector for AI benchmarking.

Collects and simulates cost metrics for AI operations.
"""

from __future__ import annotations

import random
from typing import Any

from config.config_loader import get_value


class CostCollector:
    """
    Collector for cost-related metrics of AI operations.

    Calculates or simulates the cost of AI operations based on token usage.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the cost collector.

        Args:
            seed: Optional random seed for reproducibility
        """
        self.random = random.Random(seed)

        # Load model pricing from config
        self.model_pricing = self._load_model_pricing()

    def _load_model_pricing(self) -> dict[str, float]:
        """
        Load model pricing from configuration.

        Returns:
            Dict[str, float]: Dictionary of model name to cost per 1k tokens
        """
        pricing = {}

        # Get all pricing entries from config
        models_pricing = get_value("models.pricing", {})

        # Extract cost_per_1k_tokens for each model
        for model_name, model_info in models_pricing.items():
            pricing[model_name] = model_info.get("cost_per_1k_tokens", 0.01)
            pricing[model_info.get("model", model_name)] = model_info.get("cost_per_1k_tokens", 0.01)

        # Add default entry for unknown models
        pricing["custom"] = 0.01

        return pricing

    def collect(self, model: str, input_tokens: int, output_tokens: int) -> dict[str, Any]:
        """
        Collect cost metrics for an AI operation.

        Args:
            model: The model used for generation
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Dict with cost metrics
        """
        # Get model price or use custom if not found
        model_price = self.model_pricing.get(model, self.model_pricing["custom"])

        # Calculate costs for input and output separately (some models charge differently)
        input_cost = (input_tokens / 1000) * model_price
        output_cost = (output_tokens / 1000) * model_price * 2  # Output often costs more

        total_cost = input_cost + output_cost
        cost_per_1k_tokens = model_price  # Base cost per 1k tokens

        return {
            "total_cost": round(total_cost, 6),
            "cost_per_1k_tokens": round(cost_per_1k_tokens, 6),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def simulate(self, model_tier: str = "medium", prompt_length: int = 100, verbose: bool = False) -> dict[str, Any]:
        """
        Simulate cost metrics for testing without actual AI operations.

        Args:
            model_tier: Tier of the model ("low", "medium", "high")
            prompt_length: Length of the prompt in characters
            verbose: Whether to return verbose information

        Returns:
            Dict with simulated cost metrics
        """
        # Map tiers to approximate costs per 1k tokens
        tier_costs = {
            "low": 0.002,  # Like GPT-3.5
            "medium": 0.01,  # Mid-tier models
            "high": 0.05,  # Like GPT-4
        }

        # Get cost for the specified tier or default to medium
        cost_per_1k = tier_costs.get(model_tier, tier_costs["medium"])

        # Add slight randomness to costs
        cost_per_1k = cost_per_1k * (1 + self.random.uniform(-0.1, 0.1))

        # Estimate tokens from prompt length (rough approximation)
        estimated_input_tokens = max(1, int(prompt_length / 4))

        # Estimate output tokens (typically proportional to input with some randomness)
        output_multiplier = self.random.uniform(1.5, 3.0)
        estimated_output_tokens = int(estimated_input_tokens * output_multiplier)

        # Calculate total cost
        input_cost = (estimated_input_tokens / 1000) * cost_per_1k
        output_cost = (estimated_output_tokens / 1000) * cost_per_1k * 1.5  # Output costs more
        total_cost = input_cost + output_cost

        result = {"total_cost": round(total_cost, 6), "cost_per_1k_tokens": round(cost_per_1k, 6)}

        # Add more details if verbose
        if verbose:
            result.update(
                {
                    "input_tokens": estimated_input_tokens,
                    "output_tokens": estimated_output_tokens,
                    "input_cost": round(input_cost, 6),
                    "output_cost": round(output_cost, 6),
                }
            )

        return result
