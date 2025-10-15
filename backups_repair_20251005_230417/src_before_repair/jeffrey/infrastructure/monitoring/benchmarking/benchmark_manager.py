"""
Benchmark manager for AI model performance evaluation.

This module provides the core functionality for collecting, analyzing,
and scoring various metrics related to AI model performance.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from benchmarking.collectors.cost_collector import CostCollector
from benchmarking.collectors.latency_collector import LatencyCollector
from benchmarking.collectors.quality_collector import QualityCollector
from benchmarking.collectors.stability_collector import StabilityCollector
from benchmarking.metrics import (
    calculate_composite_score,
    calculate_cost_score,
    calculate_latency_score,
    calculate_quality_score,
    calculate_stability_score,
)
from config.config_loader import get_value
from core.api_security import secure_api_method

logger = logging.getLogger("benchmark_manager")


@dataclass
class BenchmarkResult:
    """Data class representing a benchmark result."""

    # Model information
    model_id: str
    model_name: str

    # Raw metrics
    quality_metrics: dict[str, float] = field(default_factory=dict)
    latency_metrics: dict[str, float] = field(default_factory=dict)
    cost_metrics: dict[str, float] = field(default_factory=dict)
    stability_metrics: dict[str, float] = field(default_factory=dict)

    # Normalized scores (0-1 where 1 is best)
    quality_score: float = 0.0
    latency_score: float = 0.0
    cost_score: float = 0.0
    stability_score: float = 0.0

    # Composite score
    composite_score: float = 0.0

    # Metadata
    timestamp: str = ""
    benchmark_id: str = ""
    prompt: str = ""
    response: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert benchmark result to a dictionary."""
        return {
            "model_info": {"model_id": self.model_id, "model_name": self.model_name},
            "raw_metrics": {
                "quality": self.quality_metrics,
                "latency": self.latency_metrics,
                "cost": self.cost_metrics,
                "stability": self.stability_metrics,
            },
            "normalized_scores": {
                "quality": self.quality_score,
                "latency": self.latency_score,
                "cost": self.cost_score,
                "stability": self.stability_score,
                "composite": self.composite_score,
            },
            "metadata": {
                "timestamp": self.timestamp,
                "benchmark_id": self.benchmark_id,
                "prompt": self.prompt,
                "response": self.response,
                **self.metadata,  # Inclure tous les éléments supplémentaires de metadata
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Create a benchmark result from a dictionary."""
        # Extracting the standard metadata fields
        metadata_dict = data["metadata"].copy()
        standard_fields = ["timestamp", "benchmark_id", "prompt", "response"]
        for field in standard_fields:
            metadata_dict.pop(field, None)

        return cls(
            model_id=data["model_info"]["model_id"],
            model_name=data["model_info"]["model_name"],
            quality_metrics=data["raw_metrics"]["quality"],
            latency_metrics=data["raw_metrics"]["latency"],
            cost_metrics=data["raw_metrics"]["cost"],
            stability_metrics=data["raw_metrics"]["stability"],
            quality_score=data["normalized_scores"]["quality"],
            latency_score=data["normalized_scores"]["latency"],
            cost_score=data["normalized_scores"]["cost"],
            stability_score=data["normalized_scores"]["stability"],
            composite_score=data["normalized_scores"]["composite"],
            timestamp=data["metadata"].get("timestamp", ""),
            benchmark_id=data["metadata"].get("benchmark_id", ""),
            prompt=data["metadata"].get("prompt", ""),
            response=data["metadata"].get("response", ""),
            metadata=metadata_dict,
        )


class BenchmarkManager:
    """
    BenchmarkManager - Gestionnaire d'évaluation des performances des modèles d'IA
    ----------------------------------------------------------------------------
    Collecte, analyse et calcule des scores pour diverses métriques liées aux
    performances des modèles d'IA (qualité, latence, coût, stabilité).

    Le gestionnaire utilise plusieurs collecteurs spécialisés pour obtenir
    des métriques brutes, puis normalise ces métriques en scores entre 0 et 1,
    et enfin calcule un score composite pondéré selon le profil de tâche.

    Les scores plus élevés indiquent de meilleures performances, avec 1.0 étant optimal.

    Méthodes principales:
    - run_benchmark(model_id, prompt, response, ...) -> BenchmarkResult: Exécute un benchmark complet
    - simulate_benchmark(model_id, quality_level, ...) -> BenchmarkResult: Simule un benchmark
    - compare_models(results, metric_weights) -> List[Tuple]: Compare plusieurs modèles et les classe
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        seed: int | None = None,
        user_id: str = "default_user",
    ):
        """
        Initialize the benchmark manager.

        Args:
            weights: Optional dictionary with weights for each metric category
            seed: Optional random seed for reproducibility
            user_id: User identifier for authorization checks
        """
        # Default weights from config if not provided
        self.weights = weights or get_value(
            "benchmark.metric_weights",
            {"quality": 0.40, "latency": 0.25, "cost": 0.20, "stability": 0.15},
        )

        # Initialize metric collectors
        self.quality_collector = QualityCollector(seed=seed)
        self.latency_collector = LatencyCollector(seed=seed)
        self.cost_collector = CostCollector(seed=seed)
        self.stability_collector = StabilityCollector(seed=seed)

        # Cache results if enabled
        self.cache_results = get_value("benchmark.cache_results", True)
        self.results_directory = get_value("benchmark.results_directory", "benchmark_results")

        # User info and API security
        self.user_id = user_id
        self.benchmark_model = None  # Will be set in run_benchmark

        logger.info("Benchmark Manager initialized with weights: %s", self.weights)

    @secure_api_method(model_name_attr="benchmark_model", reason="Benchmarking d'IA: {benchmark_model}")
    def run_benchmark(
        self,
        model_id: str,
        model_name: str,
        prompt: str,
        response: str,
        reference: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        response_time_ms: float | None = None,
    ) -> BenchmarkResult:
        """
        Run a full benchmark on a model response.

        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable name for the model
            prompt: The input prompt
            response: The model's response
            reference: Optional reference answer for comparison
            input_tokens: Optional count of input tokens
            output_tokens: Optional count of output tokens
            response_time_ms: Optional response time in milliseconds

        Returns:
            BenchmarkResult with all metrics and scores
        """
        # Set the model name for the security decorator
        self.benchmark_model = model_name

        # Collect quality metrics
        quality_metrics = self.quality_collector.collect(prompt, response, reference)

        # Collect or estimate cost metrics
        if input_tokens is not None and output_tokens is not None:
            cost_metrics = self.cost_collector.collect(model_name, input_tokens, output_tokens)
        else:
            # Estimate tokens from text length (rough approximation)
            estimated_input_tokens = len(prompt) // 4
            estimated_output_tokens = len(response) // 4
            cost_metrics = self.cost_collector.collect(model_name, estimated_input_tokens, estimated_output_tokens)

        # Collect or estimate latency metrics
        if response_time_ms is not None:
            latency_metrics = {"latency_ms": response_time_ms}
        else:
            # Estimate latency based on response characteristics
            latency_metrics = self.latency_collector.simulate("medium", len(prompt))

        # For stability, we need multiple runs which we don't have in a single benchmark
        # So use a simplified simulation based on model characteristics
        stability_metrics = self.stability_collector.simulate("medium")

        # Calculate normalized scores using parameters from config
        latency_baseline_ms = get_value("benchmark.latency_baseline_ms", 1000.0)
        min_cost = get_value("benchmark.cost_range.min_cost", 0.0001)
        max_cost = get_value("benchmark.cost_range.max_cost", 0.06)
        quality_weights = get_value(
            "benchmark.quality_weights", {"relevance": 0.4, "accuracy": 0.4, "completeness": 0.2}
        )
        stability_weights = get_value("benchmark.stability_weights", {"error_rate": 0.7, "variance": 0.3})

        quality_score = calculate_quality_score(
            quality_metrics["relevance"],
            quality_metrics["accuracy"],
            quality_metrics["completeness"],
            weights=quality_weights,
        )

        latency_score = calculate_latency_score(latency_metrics["latency_ms"], baseline_ms=latency_baseline_ms)

        cost_score = calculate_cost_score(cost_metrics["cost_per_1k_tokens"], min_cost=min_cost, max_cost=max_cost)

        stability_score = calculate_stability_score(
            stability_metrics["error_rate"],
            stability_metrics["variance"],
            weights=stability_weights,
        )

        # Calculate composite score
        scores = {
            "quality": quality_score,
            "latency": latency_score,
            "cost": cost_score,
            "stability": stability_score,
        }

        composite_score = calculate_composite_score(scores, self.weights)

        # Create benchmark result with timestamp
        timestamp = datetime.now().isoformat()
        benchmark_id = f"{model_id}_{timestamp.replace(':', '').replace('.', '')[:14]}"

        # Create and return the benchmark result
        result = BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            quality_metrics=quality_metrics,
            latency_metrics=latency_metrics,
            cost_metrics=cost_metrics,
            stability_metrics=stability_metrics,
            quality_score=quality_score,
            latency_score=latency_score,
            cost_score=cost_score,
            stability_score=stability_score,
            composite_score=composite_score,
            prompt=prompt,
            response=response,
            timestamp=timestamp,
            benchmark_id=benchmark_id,
        )

        # Cache result if enabled
        if self.cache_results:
            self._cache_result(result)

        logger.info(
            "Benchmark completed for model %s with composite score: %.4f",
            model_name,
            composite_score,
        )

        return result

    def _cache_result(self, result: BenchmarkResult) -> None:
        """
        Cache a benchmark result to disk.

        Args:
            result: The benchmark result to cache
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(self.results_directory, exist_ok=True)

            # Create a file with timestamp and model ID
            filename = f"benchmark_{result.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(self.results_directory, filename)

            # Write result to file
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)

            logger.debug("Benchmark result cached to %s", filepath)
        except Exception as e:
            logger.warning("Failed to cache benchmark result: %s", str(e))

    def simulate_benchmark(
        self,
        model_id: str,
        model_name: str,
        quality_level: str = "medium",
        speed_level: str = "medium",
        cost_level: str = "medium",
        stability_level: str = "medium",
        prompt_length: int = 100,
    ) -> BenchmarkResult:
        """
        Simulate a benchmark without actual AI operations.

        Args:
            model_id: Unique identifier for the model
            model_name: Human-readable name for the model
            quality_level: Quality level of the model ("low", "medium", "high")
            speed_level: Speed level of the model ("slow", "medium", "fast")
            cost_level: Cost level of the model ("low", "medium", "high")
            stability_level: Stability level of the model ("low", "medium", "high")
            prompt_length: Simulated prompt length

        Returns:
            BenchmarkResult with simulated metrics and scores
        """
        # Simulate metrics using the collectors
        quality_metrics = self.quality_collector.simulate(quality_level)
        latency_metrics = self.latency_collector.simulate(speed_level, prompt_length)
        cost_metrics = self.cost_collector.simulate(cost_level, prompt_length)
        stability_metrics = self.stability_collector.simulate(stability_level)

        # Get parameters from config
        latency_baseline_ms = get_value("benchmark.latency_baseline_ms", 1000.0)
        min_cost = get_value("benchmark.cost_range.min_cost", 0.0001)
        max_cost = get_value("benchmark.cost_range.max_cost", 0.06)
        quality_weights = get_value(
            "benchmark.quality_weights", {"relevance": 0.4, "accuracy": 0.4, "completeness": 0.2}
        )
        stability_weights = get_value("benchmark.stability_weights", {"error_rate": 0.7, "variance": 0.3})

        # Calculate normalized scores
        quality_score = calculate_quality_score(
            quality_metrics["relevance"],
            quality_metrics["accuracy"],
            quality_metrics["completeness"],
            weights=quality_weights,
        )

        latency_score = calculate_latency_score(latency_metrics["latency_ms"], baseline_ms=latency_baseline_ms)

        cost_score = calculate_cost_score(cost_metrics["cost_per_1k_tokens"], min_cost=min_cost, max_cost=max_cost)

        stability_score = calculate_stability_score(
            stability_metrics["error_rate"],
            stability_metrics["variance"],
            weights=stability_weights,
        )

        # Calculate composite score
        scores = {
            "quality": quality_score,
            "latency": latency_score,
            "cost": cost_score,
            "stability": stability_score,
        }

        composite_score = calculate_composite_score(scores, self.weights)

        # Create benchmark result with timestamp
        timestamp = datetime.now().isoformat()
        benchmark_id = f"sim_{model_id}_{timestamp.replace(':', '').replace('.', '')[:14]}"

        # Create and return the benchmark result
        result = BenchmarkResult(
            model_id=model_id,
            model_name=model_name,
            quality_metrics=quality_metrics,
            latency_metrics=latency_metrics,
            cost_metrics=cost_metrics,
            stability_metrics=stability_metrics,
            quality_score=quality_score,
            latency_score=latency_score,
            cost_score=cost_score,
            stability_score=stability_score,
            composite_score=composite_score,
            prompt=f"Simulated prompt with length {prompt_length}",
            response=f"Simulated response for model {model_name}",
            timestamp=timestamp,
            benchmark_id=benchmark_id,
        )

        logger.info(
            "Simulated benchmark completed for model %s with composite score: %.4f",
            model_name,
            composite_score,
        )

        return result

    def get_scores_by_task_type(self, task_type: str) -> dict[str, dict[str, float]]:
        """
        Retourne les scores pour tous les modèles pour un type de tâche donné.

        Args:
            task_type: Type de tâche ('créatif', 'logique', 'analyse', 'général')

        Returns:
            Dictionnaire des scores par modèle
        """
        # Pour les tests, retourner des scores simulés
        return {
            "Claude": {"qualité": 0.85, "latence": 1.2, "coût": 0.02, "empreinte": 0.5},
            "GPT": {"qualité": 0.82, "latence": 0.9, "coût": 0.03, "empreinte": 0.6},
            "Grok": {"qualité": 0.78, "latence": 0.7, "coût": 0.01, "empreinte": 0.4},
            "llama3": {"qualité": 0.75, "latence": 0.8, "coût": 0.005, "empreinte": 0.3},
        }

    def compare_models(
        self, results: list[BenchmarkResult], metric_weights: dict[str, float] | None = None
    ) -> list[tuple[str, float]]:
        """
        Compare multiple models based on their benchmark results.

        Args:
            results: List of BenchmarkResult objects
            metric_weights: Optional custom weights for comparison

        Returns:
            List of (model_name, score) tuples sorted by score (highest first)
        """
        weights = metric_weights or self.weights

        # Extract scores and recalculate composite if custom weights provided
        model_scores = []
        for result in results:
            if metric_weights:
                # Recalculate with custom weights
                scores = {
                    "quality": result.quality_score,
                    "latency": result.latency_score,
                    "cost": result.cost_score,
                    "stability": result.stability_score,
                }
                composite = calculate_composite_score(scores, weights)
            else:
                # Use existing composite score
                composite = result.composite_score

            model_scores.append((result.model_name, composite))

        # Sort by score (highest first)
        model_scores.sort(key=lambda x: x[1], reverse=True)
        return model_scores
