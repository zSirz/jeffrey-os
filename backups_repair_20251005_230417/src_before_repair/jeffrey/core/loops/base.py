"""
BaseLoop - Socle robuste pour toutes les boucles autonomes
Gère : cancellation, jitter, backoff, timeouts, budget, métriques
"""

import asyncio
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class BaseLoop(ABC):
    """
    Base commune pour toutes les boucles avec gestion robuste
    """

    def __init__(
        self,
        name: str,
        interval_s: float,
        *,
        jitter_s: float = 0.2,
        hard_timeout_s: float = 2.0,
        budget_gate: Callable[[], bool] | None = None,
        bus=None,
        privacy_guard=None,
    ):
        self.name = name
        self.interval_s = interval_s
        self.jitter_s = jitter_s
        self.hard_timeout_s = hard_timeout_s
        self.budget_gate = budget_gate
        self.bus = bus
        self.privacy_guard = privacy_guard

        # État
        self._task: asyncio.Task | None = None
        self.running = False
        self._err_streak = 0
        self.cycles = 0
        self._latencies_ms: list[float] = []

        # RL Adaptive (Grok) with Replay Buffer (Phase 2.3)
        self.q_table = {}  # État -> Valeur pour Q-learning
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1

        # Replay Buffer for stable learning
        try:
            from ..rl import ReplayBuffer

            self.replay_buffer = ReplayBuffer(capacity=5000)
            self.batch_size = 32
            self.update_frequency = 10  # Update every N steps
            self.update_counter = 0
            self.has_replay_buffer = True
        except ImportError:
            self.replay_buffer = None
            self.has_replay_buffer = False
            logger.debug(f"ReplayBuffer not available for {self.name}")

        # Bus dropped counter (back-ref set by LoopManager)
        self.loop_manager = None
        self.total_errors = 0
        self.bus_dropped_count = 0  # NOUVEAU : Compteur de drops
        self.bus_timeout = 0.5  # Default timeout for publishes

        # Trace propagation (Phase 3)
        self._current_trace_id: str | None = None

        # Default interval for backpressure management
        self.default_interval_s = interval_s

    async def start(self):
        """Démarre la boucle avec gestion propre"""
        if self.running:
            return

        # Reset compteurs au start (GPT fix #6)
        self.bus_dropped_count = 0
        self.total_errors = 0
        self.cycles = 0

        self.running = True
        self._task = asyncio.create_task(self._run(), name=f"loop:{self.name}")
        logger.info(f"▶️ {self.name} loop started")

    async def stop(self):
        """Arrête proprement la boucle avec sauvegarde RL"""
        self.running = False

        # CRITIQUE : Sauvegarder replay buffer
        if hasattr(self, "replay_buffer") and self.replay_buffer:
            try:
                self.replay_buffer.save()
                logger.info(f"{self.name}: Replay buffer saved ({len(self.replay_buffer)} experiences)")
            except Exception as e:
                logger.error(f"{self.name}: Failed to save replay buffer: {e}")

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info(f"⏹️ {self.name} loop stopped")

    async def safe_publish(self, event: str, data: dict, timeout: float = None):
        """Publish sécurisé avec trace propagation et compteur de drops"""
        if timeout is None:
            timeout = self.bus_timeout

        # Ajouter trace_id au data pour correlation
        if not self._current_trace_id:
            self._current_trace_id = str(uuid.uuid4())
        data["_trace_id"] = self._current_trace_id

        # Log avec trace pour correlation
        logger.debug(f"[{self._current_trace_id}] {self.name} publishing {event}")

        try:
            if self.bus:  # Using self.bus not event_bus
                await asyncio.wait_for(
                    self.bus.safe_publish(event, data, timeout)
                    if hasattr(self.bus, "safe_publish")
                    else self.bus.publish(event, data),
                    timeout=timeout,
                )
        except TimeoutError:
            self.bus_dropped_count += 1
            logger.warning(f"[{self._current_trace_id}] {self.name}: Event dropped (timeout) → {event}")
        except Exception as e:
            self.bus_dropped_count += 1
            logger.error(f"[{self._current_trace_id}] {self.name}: Event dropped (error) → {event}: {e}")

    async def _run(self):
        """Boucle principale avec toutes les protections"""
        while self.running:
            try:
                # Gate de ressources (GPT)
                if self.budget_gate and not self.budget_gate():
                    await asyncio.sleep(self._sleep_with_jitter())
                    continue

                start = time.perf_counter()

                # Hard timeout sur un tick
                try:
                    await asyncio.wait_for(self._tick_with_learning(), timeout=self.hard_timeout_s)
                    self._err_streak = 0
                except TimeoutError:
                    self._err_streak += 1
                    logger.warning(f"{self.name} tick timeout after {self.hard_timeout_s}s")

                # Métriques
                dt_ms = (time.perf_counter() - start) * 1000
                self._latencies_ms.append(dt_ms)
                if len(self._latencies_ms) > 200:
                    self._latencies_ms = self._latencies_ms[-200:]
                self.cycles += 1

                # RL : Ajuster les paramètres selon performance
                self._adapt_parameters(dt_ms)

                await asyncio.sleep(self._sleep_with_jitter())

            except Exception as e:
                self._err_streak += 1
                logger.error(f"{self.name} loop error: {e}")
                # Backoff exponentiel limité
                await asyncio.sleep(min(5.0, (2 ** min(self._err_streak, 5)) * 0.1))

    def _sleep_with_jitter(self) -> float:
        """Sleep avec jitter pour éviter la résonance"""
        j = random.uniform(-self.jitter_s, self.jitter_s)
        return max(0.01, self.interval_s + j)

    async def _tick_with_learning(self):
        """Tick avec apprentissage RL et nouveau trace par cycle"""
        # Nouveau trace pour chaque cycle
        self._current_trace_id = str(uuid.uuid4())

        # État actuel pour Q-learning
        state = self._get_current_state()

        # Exploration vs Exploitation
        if random.random() < self.exploration_rate:
            action = self._explore_action()
        else:
            action = self._exploit_action(state)

        # Exécuter le tick
        result = await self._tick()

        # Calculer la récompense
        reward = self._calculate_reward(result)

        # Mettre à jour Q-table
        self._update_q_table(state, action, reward)

        return result

    @abstractmethod
    async def _tick(self):
        """À implémenter dans les sous-classes"""
        raise NotImplementedError

    def _get_current_state(self) -> str:
        """État actuel pour RL (à override si besoin)"""
        return f"cycles:{self.cycles // 10},errors:{min(self._err_streak, 5)}"

    def _explore_action(self) -> str:
        """Action d'exploration (random)"""
        return random.choice(["normal", "fast", "slow"])

    def _exploit_action(self, state: str) -> str:
        """Meilleure action connue pour l'état"""
        return max(
            self.q_table.get(state, {"normal": 0}),
            key=lambda k: self.q_table.get(state, {}).get(k, 0),
        )

    def _calculate_reward(self, result: Any) -> float:
        """Calcule la récompense (à override)"""
        return -self._err_streak if self._err_streak > 0 else 1.0

    def _get_state(self) -> str:
        """Alias pour _get_current_state pour compatibilité"""
        return self._get_current_state()

    def _get_action(self, state: str, epsilon: float = 0.1) -> str:
        """Get action (for RL compatibility in tests)"""
        if random.random() < epsilon:
            return self._explore_action()
        else:
            return self._exploit_action(state)

    def _update_q_table(self, state: str, action: str, reward: float):
        """Met à jour la Q-table avec replay buffer si disponible"""
        # Calculer next_state (simple heuristique)
        next_state = self._get_state()

        # Si replay buffer disponible, l'utiliser
        if self.has_replay_buffer and self.replay_buffer:
            # Ajouter la transition au buffer
            self.replay_buffer.add(state, action, reward, next_state, done=False)
            self.update_counter += 1

            # Update périodique depuis le buffer
            if self.update_counter % self.update_frequency == 0 and len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)

                for s, a, r, ns, _ in batch:
                    if s not in self.q_table:
                        self.q_table[s] = {}

                    old_value = self.q_table[s].get(a, 0)
                    next_max = max(self.q_table.get(ns, {}).values(), default=0)
                    new_value = old_value + self.learning_rate * (r + self.discount_factor * next_max - old_value)
                    self.q_table[s][a] = new_value

            # Sauvegarder périodiquement
            if self.update_counter % 100 == 0:
                self.replay_buffer.save()
        else:
            # Fallback : mise à jour directe sans buffer
            if state not in self.q_table:
                self.q_table[state] = {}

            old_value = self.q_table[state].get(action, 0)
            new_value = old_value + self.learning_rate * (
                reward + self.discount_factor * max(self.q_table.get(next_state, {}).values(), default=0) - old_value
            )
            self.q_table[state][action] = new_value

    def _adapt_parameters(self, latency_ms: float):
        """Adapte les paramètres selon la performance (RL)"""
        if latency_ms > self.hard_timeout_s * 500:  # Trop lent
            self.interval_s = min(60, self.interval_s * 1.1)
        elif latency_ms < 100:  # Très rapide
            self.interval_s = max(1, self.interval_s * 0.95)

    def p95_latency_ms(self) -> float:
        """Calcule la latence P95"""
        if not self._latencies_ms:
            return 0.0
        arr = sorted(self._latencies_ms)
        return arr[int(0.95 * (len(arr) - 1))]

    def p99_latency_ms(self) -> float:
        """Calcule la latence P99"""
        if not self._latencies_ms or len(self._latencies_ms) < 10:
            return 0.0
        arr = sorted(self._latencies_ms)
        return arr[int(0.99 * (len(arr) - 1))]

    def _percentile(self, sorted_vals: list, pct: float) -> float:
        """Calcul percentile sans numpy"""
        if not sorted_vals:
            return 0
        k = max(0, min(len(sorted_vals) - 1, int(round((pct / 100.0) * (len(sorted_vals) - 1)))))
        return sorted_vals[k]

    def get_metrics(self) -> dict[str, Any]:
        """Métriques enrichies avec P95/P99 sans numpy"""
        metrics = {
            "name": self.name,
            "cycles": self.cycles,
            "errors": self.total_errors,
            "error_rate": self.total_errors / max(self.cycles, 1),
            "consecutive_errors": self._err_streak,
            "interval": self.interval_s,
            "running": self.running,
            "bus_dropped": self.bus_dropped_count,  # NOUVEAU
            "status": "running" if self.running else "stopped",
        }

        # Latency metrics sans numpy
        if self._latencies_ms:
            sorted_lat = sorted(self._latencies_ms)
            p50 = self._percentile(sorted_lat, 50)
            p95 = self._percentile(sorted_lat, 95)
            p99 = self._percentile(sorted_lat, 99) if len(sorted_lat) > 50 else p95

            metrics.update(
                {
                    "avg_latency_ms": sum(self._latencies_ms) / len(self._latencies_ms),
                    "p50_latency_ms": p50,
                    "p95_latency_ms": p95,
                    "p99_latency_ms": p99,
                    "max_latency_ms": max(self._latencies_ms),
                }
            )
        else:
            metrics.update(
                {
                    "avg_latency_ms": 0,
                    "p50_latency_ms": 0,
                    "p95_latency_ms": 0,
                    "p99_latency_ms": 0,
                    "max_latency_ms": 0,
                }
            )

        # RL metrics
        metrics["q_table_size"] = len(self.q_table)
        metrics["exploration_rate"] = self.exploration_rate

        # Replay buffer metrics if available
        if self.has_replay_buffer and self.replay_buffer:
            metrics["replay_buffer"] = self.replay_buffer.get_stats()

        return metrics
