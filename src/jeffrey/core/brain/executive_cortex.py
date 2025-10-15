"""
Executive Cortex - Décision intelligente par bandit contextuel
Choisit optimalement entre Cache/Autonome/LLM
"""

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np

from jeffrey.utils.logger import get_logger

logger = get_logger("ExecutiveCortex")


@dataclass
class ArmStats:
    """Statistiques pour un bras du bandit"""

    pulls: int = 0
    rewards: float = 0.0
    squared_rewards: float = 0.0

    @property
    def mean_reward(self) -> float:
        return self.rewards / max(self.pulls, 1)

    @property
    def variance(self) -> float:
        if self.pulls < 2:
            return 1.0
        mean_sq = self.squared_rewards / self.pulls
        return max(0, mean_sq - self.mean_reward**2)

    @property
    def std_dev(self) -> float:
        return np.sqrt(self.variance)


class ExecutiveCortex:
    """
    Cortex exécutif utilisant un bandit contextuel
    pour décisions optimales
    """

    def __init__(self, exploration_rate: float = 0.1):
        # Bras disponibles
        self.arms = ["cache", "autonomous", "llm"]

        # Stats par contexte et par bras
        self.stats: dict[str, dict[str, ArmStats]] = defaultdict(lambda: {arm: ArmStats() for arm in self.arms})

        # Paramètres du bandit
        self.exploration_rate = exploration_rate
        self.c = 2.0  # Paramètre UCB

        # Circuit breaker
        self.failures = defaultdict(int)
        self.breaker_threshold = 3
        self.breaker_timeout = 600  # 10 minutes
        self.breaker_reset_times = {}

        # Métriques
        self.total_decisions = 0
        self.arm_selections = defaultdict(int)
        self.total_reward = 0.0

        logger.info("🎯 Executive Cortex initialized with bandit strategy")

    def _extract_context_key(self, context: dict[str, Any]) -> str:
        """
        Extrait une clé de contexte pour grouper les décisions similaires
        """
        # Features importantes pour la décision
        features = [
            context.get("intent_type", "unknown"),
            str(int(context.get("complexity", 0.5) * 10)),  # Quantifié 0-10
            str(int(context.get("familiarity", 0.5) * 10)),
            context.get("domain", "general"),
        ]

        return "|".join(features)

    async def decide(self, context: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """
        Décide quel bras utiliser selon le contexte
        Retourne (arm_selected, metadata)
        """
        self.total_decisions += 1

        # Extraire la clé de contexte
        ctx_key = self._extract_context_key(context)

        # Vérifier circuit breaker
        broken_arms = self._check_circuit_breakers()
        available_arms = [arm for arm in self.arms if arm not in broken_arms]

        if not available_arms:
            logger.warning("⚠️ All arms broken, forcing cache")
            return "cache", {"reason": "circuit_breaker", "context": ctx_key}

        # Sélection du bras
        if np.random.random() < self.exploration_rate:
            # Exploration : choix aléatoire
            arm = np.random.choice(available_arms)
            strategy = "exploration"
        else:
            # Exploitation : UCB ou Thompson
            arm = self._select_best_arm(ctx_key, available_arms)
            strategy = "exploitation"

        # Mise à jour des compteurs
        self.arm_selections[arm] += 1

        metadata = {
            "context": ctx_key,
            "strategy": strategy,
            "available_arms": available_arms,
            "decision_number": self.total_decisions,
            "mean_rewards": {arm: self.stats[ctx_key][arm].mean_reward for arm in available_arms},
        }

        logger.debug(f"Decision #{self.total_decisions}: {arm} ({strategy})")

        return arm, metadata

    def _select_best_arm(self, ctx_key: str, available_arms: list[str]) -> str:
        """
        Sélectionne le meilleur bras selon UCB (Upper Confidence Bound)
        """
        stats = self.stats[ctx_key]

        # Calcul UCB pour chaque bras
        ucb_scores = {}
        total_pulls = sum(stats[arm].pulls for arm in available_arms)

        for arm in available_arms:
            arm_stats = stats[arm]

            if arm_stats.pulls == 0:
                # Bras jamais tiré : priorité maximale
                ucb_scores[arm] = float("inf")
            else:
                # UCB = mean + c * sqrt(2 * ln(n) / n_i)
                exploitation = arm_stats.mean_reward
                exploration = self.c * np.sqrt(2 * np.log(max(total_pulls, 1)) / arm_stats.pulls)
                ucb_scores[arm] = exploitation + exploration

        # Sélectionner le bras avec le meilleur score UCB
        return max(ucb_scores.items(), key=lambda x: x[1])[0]

    async def reward(
        self,
        arm: str,
        context: dict[str, Any],
        quality_score: float,
        latency_ms: float,
        success: bool = True,
    ):
        """
        Enregistre la récompense pour un bras
        """
        ctx_key = self._extract_context_key(context)

        # Calculer la récompense composite
        # Récompense = qualité * (1 - latence_normalisée) * succès
        latency_normalized = min(latency_ms / 1000.0, 1.0)  # Cap à 1s
        reward = quality_score * (1 - latency_normalized * 0.3)  # 30% poids latence

        if not success:
            reward = -1.0  # Pénalité forte pour échec
            self.failures[arm] += 1
        else:
            # Reset failures sur succès
            self.failures[arm] = 0

        # Mise à jour des stats
        arm_stats = self.stats[ctx_key][arm]
        arm_stats.pulls += 1
        arm_stats.rewards += reward
        arm_stats.squared_rewards += reward**2

        self.total_reward += reward

        logger.debug(f"Reward for {arm}: {reward:.3f} (quality={quality_score:.2f}, latency={latency_ms:.0f}ms)")

    def _check_circuit_breakers(self) -> list[str]:
        """
        Vérifie les circuit breakers et retourne les bras cassés
        """
        broken = []
        current_time = time.time()

        for arm in self.arms:
            # Vérifier si en timeout
            if arm in self.breaker_reset_times:
                if current_time < self.breaker_reset_times[arm]:
                    broken.append(arm)
                else:
                    # Timeout expiré, reset
                    del self.breaker_reset_times[arm]
                    self.failures[arm] = 0
                    logger.info(f"🔧 Circuit breaker reset for {arm}")

            # Vérifier si doit casser
            elif self.failures[arm] >= self.breaker_threshold:
                self.breaker_reset_times[arm] = current_time + self.breaker_timeout
                broken.append(arm)
                logger.warning(f"⚡ Circuit breaker triggered for {arm} ({self.failures[arm]} failures)")

        return broken

    def get_stats(self) -> dict[str, Any]:
        """
        Retourne les statistiques complètes
        """
        # Agréger les stats par bras
        global_stats = {arm: ArmStats() for arm in self.arms}

        for ctx_stats in self.stats.values():
            for arm, arm_stats in ctx_stats.items():
                global_stats[arm].pulls += arm_stats.pulls
                global_stats[arm].rewards += arm_stats.rewards
                global_stats[arm].squared_rewards += arm_stats.squared_rewards

        return {
            "total_decisions": self.total_decisions,
            "arm_selections": dict(self.arm_selections),
            "mean_reward": self.total_reward / max(self.total_decisions, 1),
            "arm_stats": {
                arm: {
                    "pulls": stats.pulls,
                    "mean_reward": stats.mean_reward,
                    "std_dev": stats.std_dev,
                }
                for arm, stats in global_stats.items()
            },
            "contexts_tracked": len(self.stats),
            "circuit_breakers": list(self.breaker_reset_times.keys()),
        }
