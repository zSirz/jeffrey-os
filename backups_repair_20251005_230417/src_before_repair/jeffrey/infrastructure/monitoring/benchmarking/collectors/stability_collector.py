"""
Stability metric collector for AI benchmarking.

Collects and simulates stability metrics for AI operations.
"""

from __future__ import annotations

import random
import statistics
from typing import Any


class StabilityCollector:
    """
    Collector for stability-related metrics of AI operations.

    Measures or simulates the reliability and consistency of AI responses.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the stability collector.

        Args:
            seed: Optional random seed for reproducibility
        """
        self.random = random.Random(seed)
        self.operation_history = []

    def collect(self, success: bool, response_times: list[float], consistency_scores: list[float]) -> dict[str, Any]:
        """
        Collect stability metrics from operation data.

        Args:
            success: Whether the operation succeeded
            response_times: List of response times from multiple runs
            consistency_scores: List of consistency scores from multiple runs

        Returns:
            Dict with stability metrics
        """
        # Update operation history
        self.operation_history.append(
            {
                "success": success,
                "response_time": statistics.mean(response_times) if response_times else 0,
                "consistency_score": (statistics.mean(consistency_scores) if consistency_scores else 0),
            }
        )

        # Calculate error rate from history
        total_operations = len(self.operation_history)
        successful_operations = sum(1 for op in self.operation_history if op["success"])
        error_rate = 0 if total_operations == 0 else (total_operations - successful_operations) / total_operations

        # Calculate variance in response times
        if len(response_times) > 1:
            time_variance = statistics.variance(response_times) / (max(response_times) ** 2)  # Normalized variance
        else:
            time_variance = 0

        # Calculate variance in consistency scores
        if len(consistency_scores) > 1:
            consistency_variance = statistics.variance(consistency_scores)
        else:
            consistency_variance = 0

        # Combined variance (weighted average)
        combined_variance = time_variance * 0.6 + consistency_variance * 0.4

        return {
            "error_rate": round(error_rate, 4),
            "variance": round(combined_variance, 4),
            "total_operations": total_operations,
            "success_rate": round(1 - error_rate, 4),
        }

    def simulate(self, model_stability: str = "medium", num_operations: int = 10) -> dict[str, Any]:
        """
        Simulate stability metrics for testing without actual AI operations.

        Args:
            model_stability: Stability level of the model ("low", "medium", "high")
            num_operations: Number of simulated operations

        Returns:
            Dict with simulated stability metrics
        """
        # Base error rates for different stability levels
        stability_bases = {
            "low": {"error_rate": 0.15, "variance_base": 0.3},
            "medium": {"error_rate": 0.05, "variance_base": 0.15},
            "high": {"error_rate": 0.01, "variance_base": 0.05},
        }

        # Use the specified stability level or default to medium
        base = stability_bases.get(model_stability, stability_bases["medium"])

        # Simulate multiple operations
        simulated_success = []
        simulated_times = []
        simulated_consistency = []

        for _ in range(num_operations):
            # Simulate success/failure
            success = self.random.random() > base["error_rate"]
            simulated_success.append(success)

            # Simulate response time with variance
            time_base = 500  # 500ms base
            time_variance = base["variance_base"] * time_base
            response_time = time_base + self.random.uniform(-time_variance, time_variance)
            simulated_times.append(max(10, response_time))  # Ensure positive time

            # Simulate consistency score (0-1)
            consistency_base = 0.8 if success else 0.4
            consistency_variance = base["variance_base"]
            consistency = consistency_base + self.random.uniform(-consistency_variance, consistency_variance)
            simulated_consistency.append(max(0, min(1, consistency)))  # Clamp between 0 and 1

        # Calculate actual error rate from simulations
        error_rate = 1 - (sum(simulated_success) / num_operations)

        # Calculate variances
        time_variance = statistics.variance(simulated_times) / (max(simulated_times) ** 2) if simulated_times else 0
        consistency_variance = statistics.variance(simulated_consistency) if simulated_consistency else 0

        # Combined variance (weighted average)
        combined_variance = time_variance * 0.6 + consistency_variance * 0.4

        return {
            "error_rate": round(error_rate, 4),
            "variance": round(combined_variance, 4),
            "total_operations": num_operations,
            "success_rate": round(1 - error_rate, 4),
        }
