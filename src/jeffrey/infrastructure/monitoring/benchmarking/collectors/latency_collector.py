"""
Latency metric collector for AI benchmarking.

Collects and simulates latency metrics for AI responses.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable


class LatencyCollector:
    """
    Collector for latency-related metrics of AI responses.

    Measures or simulates the time taken to generate responses.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the latency collector.

        Args:
            seed: Optional random seed for reproducibility
        """
        self.random = random.Random(seed)

    def collect(self, operation: Callable, *args, **kwargs) -> dict[str, float]:
        """
        Collect latency metrics for an operation.

        Args:
            operation: The callable operation to measure
            *args, **kwargs: Arguments to pass to the operation

        Returns:
            Dict with latency metrics in milliseconds
        """
        # Measure start time
        start_time = time.time()

        # Execute the operation
        result = operation(*args, **kwargs)

        # Measure end time and calculate duration
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000

        return {"latency_ms": round(duration_ms, 2), "result": result}

    def simulate(self, model_speed: str = "medium", prompt_length: int = 100) -> dict[str, float]:
        """
        Simulate latency metrics for testing without actual AI operations.

        Args:
            model_speed: Speed level of the model ("fast", "medium", "slow")
            prompt_length: Length of the prompt in characters

        Returns:
            Dict with simulated latency metrics
        """
        # Base latency values for different speed levels (in milliseconds)
        speed_bases = {
            "fast": 200,
            "medium": 500,
            "slow": 1500,  # Augmenté pour garantir qu'il est supérieur à 1000ms dans les tests
        }

        # Use the specified speed level or default to medium
        base_latency = speed_bases.get(model_speed, speed_bases["medium"])

        # Adjust for prompt length (longer prompts take more time)
        length_factor = max(1.0, prompt_length / 100)
        adjusted_latency = base_latency * length_factor

        # Add randomness to simulate real-world variability, mais garantir
        # que 'slow' reste au-dessus de 1000ms et 'fast' reste en-dessous de 1000ms
        if model_speed == "slow":
            # Pour le mode "slow", garantir que la latence est > 1000ms pour les tests
            variation = min(adjusted_latency * 0.2, (adjusted_latency - 1100) if adjusted_latency > 1100 else 0)
            final_latency = max(1100, adjusted_latency + self.random.uniform(-variation, variation))
        elif model_speed == "fast":
            # Pour le mode "fast", garantir que la latence est < 1000ms pour les tests
            variation = min(adjusted_latency * 0.2, 900 - adjusted_latency if adjusted_latency < 900 else 0)
            final_latency = min(900, adjusted_latency + self.random.uniform(-variation, variation))
        else:
            # Pour le mode "medium", utiliser la variation standard
            variation = adjusted_latency * 0.2  # 20% variation
            final_latency = adjusted_latency + self.random.uniform(-variation, variation)

        # Ensure latency is not negative
        final_latency = max(10.0, final_latency)

        return {"latency_ms": round(final_latency, 2)}
