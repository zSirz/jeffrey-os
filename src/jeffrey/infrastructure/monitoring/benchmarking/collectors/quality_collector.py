"""
Quality metric collector for AI benchmarking.

Collects and simulates quality-related metrics for AI responses.
"""

from __future__ import annotations

import random


class QualityCollector:
    """
    Collector for quality-related metrics of AI responses.

    Simulates the collection of relevance, accuracy, and completeness scores
    for AI-generated responses.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Initialize the quality collector.

        Args:
            seed: Optional random seed for reproducibility
        """
        self.random = random.Random(seed)

    def collect(self, prompt: str, response: str, reference: str | None = None) -> dict[str, float]:
        """
        Collect quality metrics for a given prompt and response.

        In a real implementation, this would use actual evaluation methods.
        This simulation uses prompt/response characteristics and randomness.

        Args:
            prompt: The input prompt
            response: The AI-generated response
            reference: Optional reference answer for comparison

        Returns:
            Dict with quality metrics (relevance, accuracy, completeness)
        """
        # Simulated metrics based on characteristics and randomness
        # In a real implementation, these would use NLP evaluation techniques

        # Simulate relevance based on keyword matching (simplified)
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        keyword_overlap = len(prompt_words.intersection(response_words))
        base_relevance = min(10, (keyword_overlap / max(1, len(prompt_words))) * 10)
        relevance = base_relevance * 0.7 + self.random.uniform(0, 3)

        # Simulate accuracy (would use fact-checking in real implementation)
        if reference:
            ref_words = set(reference.lower().split())
            ref_overlap = len(response_words.intersection(ref_words))
            base_accuracy = min(10, (ref_overlap / max(1, len(ref_words))) * 10)
            accuracy = base_accuracy * 0.6 + self.random.uniform(0, 4)
        else:
            # Without reference, use simulated values
            accuracy = 5 + self.random.uniform(0, 5)

        # Simulate completeness based on response length and structure
        response_length = len(response)
        length_factor = min(1.0, response_length / 200)  # Normalize for typical response
        completeness = length_factor * 8 + self.random.uniform(0, 2)

        return {
            "relevance": round(relevance, 2),
            "accuracy": round(accuracy, 2),
            "completeness": round(completeness, 2),
        }

    def simulate(self, model_quality: str = "medium") -> dict[str, float]:
        """
        Simulate quality metrics for testing without actual AI responses.

        Args:
            model_quality: Quality level of the model ("low", "medium", "high")

        Returns:
            Dict with simulated quality metrics
        """
        # Base values for different quality levels
        quality_bases = {
            "low": {"relevance": 4.0, "accuracy": 3.0, "completeness": 5.0},
            "medium": {"relevance": 7.0, "accuracy": 6.0, "completeness": 7.0},
            "high": {"relevance": 8.5, "accuracy": 8.0, "completeness": 9.0},
        }

        # Use the specified quality level or default to medium
        base = quality_bases.get(model_quality, quality_bases["medium"])

        # Add randomness
        relevance = base["relevance"] + self.random.uniform(-1.0, 1.0)
        accuracy = base["accuracy"] + self.random.uniform(-1.0, 1.0)
        completeness = base["completeness"] + self.random.uniform(-1.0, 1.0)

        # Ensure values are within 0-10 range
        relevance = max(0, min(10, relevance))
        accuracy = max(0, min(10, accuracy))
        completeness = max(0, min(10, completeness))

        return {
            "relevance": round(relevance, 2),
            "accuracy": round(accuracy, 2),
            "completeness": round(completeness, 2),
        }
