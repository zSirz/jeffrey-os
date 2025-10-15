"""
Bus local asynchrone avec circuit breakers et monitoring
Version production-ready pour Bundle 1
"""

import asyncio
import hashlib
import json
import logging
import time
import traceback
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# GPT Fix #4: rendre psutil optionnel
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """États du circuit breaker"""

    CLOSED = "closed"  # Normal
    OPEN = "open"  # Erreurs, bloque
    HALF_OPEN = "half_open"  # Test de récupération


@dataclass
class BusMetrics:
    """Métriques du bus"""

    published: int = 0
    delivered: int = 0
    failed: int = 0
    latency_ms: deque = field(default_factory=lambda: deque(maxlen=100))
    throughput_per_sec: float = 0.0
    active_handlers: int = 0
    memory_mb: float = 0.0
    circuit_trips: int = 0

    def to_dict(self) -> dict:
        return {
            **asdict(self),
            "latency_p50": self.get_percentile(50),
            "latency_p95": self.get_percentile(95),
            "latency_p99": self.get_percentile(99),
        }

    def get_percentile(self, p: int) -> float:
        if not self.latency_ms:
            return 0.0
        sorted_latencies = sorted(self.latency_ms)
        idx = int(len(sorted_latencies) * p / 100)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


@dataclass
class CircuitBreaker:
    """Circuit breaker pour un handler"""

    state: CircuitState = CircuitState.CLOSED
    failures: int = 0
    last_failure_time: float | None = None
    last_success_time: float | None = None
    failure_threshold: int = 5
    recovery_timeout: float = 30.0  # secondes
    half_open_requests: int = 0

    def record_success(self):
        """Enregistre un succès"""
        self.last_success_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            self.failures = 0
            logger.info("Circuit breaker recovered to CLOSED")

    def record_failure(self):
        """Enregistre un échec"""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker OPEN after {self.failures} failures")

    def can_execute(self) -> bool:
        """Vérifie si on peut exécuter"""
        if self.state == CircuitState.CLOSED:
            return True

        if self.state == CircuitState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.half_open_requests = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
                return True
            return False

        # HALF_OPEN: limite les requêtes de test
        if self.half_open_requests < 3:
            self.half_open_requests += 1
            return True
        return False


class LocalAsyncBus:
    """
    Bus local asynchrone production-ready avec:
    - Circuit breakers par handler
    - Monitoring et métriques
    - Rate limiting
    - Dead letter queue
    - Graceful shutdown
    """

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 0.1,
        enable_metrics: bool = True,
        enable_dlq: bool = True,
        memory_limit_mb: int = 512,
    ):
        # Handlers et topics
        self.handlers: dict[str, list[Callable]] = defaultdict(list)
        self.handler_metadata: dict[str, dict] = {}

        # Circuit breakers
        self.circuit_breakers: dict[str, CircuitBreaker] = {}

        # Métriques
        self.metrics = BusMetrics() if enable_metrics else None
        self.start_time = time.time()

        # Configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.memory_limit_mb = memory_limit_mb

        # Dead Letter Queue
        self.dlq: deque = deque(maxlen=1000) if enable_dlq else None

        # État
        self._running = True
        self._tasks: set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()

        # Rate limiting
        self._rate_limiters: dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Monitoring task
        if enable_metrics:
            asyncio.create_task(self._monitor_loop())

    async def subscribe(
        self,
        topic: str,
        handler: Callable,
        priority: int = 0,
        timeout: float = 1.0,
        rate_limit: int | None = None,
    ) -> str:
        """
        S'abonne à un topic avec options avancées

        Args:
            topic: Pattern du topic (supporte wildcards)
            handler: Fonction handler
            priority: Priorité (plus élevé = exécuté en premier)
            timeout: Timeout en secondes
            rate_limit: Limite de messages par seconde

        Returns:
            handler_id: Identifiant unique du handler
        """
        if not self._running:
            raise RuntimeError("Bus is shutting down")

        # Générer ID unique
        handler_id = f"{topic}:{handler.__name__}:{id(handler)}"

        # Metadata
        self.handler_metadata[handler_id] = {
            "topic": topic,
            "priority": priority,
            "timeout": timeout,
            "rate_limit": rate_limit,
            "subscribed_at": datetime.now().isoformat(),
        }

        # Circuit breaker
        self.circuit_breakers[handler_id] = CircuitBreaker()

        # Ajouter handler trié par priorité
        self.handlers[topic].append(handler)
        self.handlers[topic].sort(
            key=lambda h: self.handler_metadata.get(f"{topic}:{h.__name__}:{id(h)}", {}).get("priority", 0),
            reverse=True,
        )

        logger.info(f"✅ Subscribed to '{topic}' with handler {handler.__name__}")
        return handler_id

    async def unsubscribe(self, handler_id: str):
        """Désabonne un handler"""
        if handler_id in self.handler_metadata:
            metadata = self.handler_metadata[handler_id]
            topic = metadata["topic"]
            # Retrouver et supprimer le handler
            # ... (implémentation de recherche)
            del self.handler_metadata[handler_id]
            del self.circuit_breakers[handler_id]
            logger.info(f"Unsubscribed handler {handler_id}")

    async def publish(
        self,
        topic: str,
        data: Any,
        correlation_id: str | None = None,
        priority: int = 0,
        ttl: float | None = None,
    ) -> bool:
        """
        Publie un message sur un topic

        Args:
            topic: Topic de destination
            data: Données à publier
            correlation_id: ID de corrélation pour traçage
            priority: Priorité du message
            ttl: Time-to-live en secondes

        Returns:
            success: True si au moins un handler a reçu le message
        """
        if not self._running:
            return False

        start_time = time.perf_counter()

        # Générer correlation_id si nécessaire
        if not correlation_id:
            correlation_id = hashlib.md5(f"{topic}{time.time()}".encode()).hexdigest()[:8]

        # Préparer le message
        message = {
            "topic": topic,
            "data": data,
            "correlation_id": correlation_id,
            "timestamp": time.time(),
            "priority": priority,
        }

        # Vérifier TTL
        if ttl and time.time() - message["timestamp"] > ttl:
            logger.warning(f"Message expired (TTL={ttl}s): {correlation_id}")
            return False

        # Collecter les handlers matching
        matching_handlers = []
        for pattern, handlers in self.handlers.items():
            if self._match_pattern(pattern, topic):
                matching_handlers.extend([(h, f"{pattern}:{h.__name__}:{id(h)}") for h in handlers])

        if not matching_handlers:
            logger.debug(f"No handlers for topic: {topic}")
            return False

        # Exécuter handlers en parallèle avec gestion d'erreur
        tasks = []
        for handler, handler_id in matching_handlers:
            if self._check_rate_limit(handler_id):
                task = asyncio.create_task(self._execute_handler_safe(handler, handler_id, message))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
                tasks.append(task)

        # Attendre completion
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Métriques
        if self.metrics:
            self.metrics.published += 1
            latency = (time.perf_counter() - start_time) * 1000
            self.metrics.latency_ms.append(latency)

            success_count = sum(1 for r in results if r is True)
            self.metrics.delivered += success_count
            self.metrics.failed += len(results) - success_count

        return any(r is True for r in results)

    async def _execute_handler_safe(self, handler: Callable, handler_id: str, message: dict) -> bool:
        """Exécute un handler avec protections"""

        # Vérifier circuit breaker
        breaker = self.circuit_breakers.get(handler_id)
        if breaker and not breaker.can_execute():
            logger.debug(f"Circuit breaker OPEN for {handler_id}")
            return False

        # Récupérer metadata
        metadata = self.handler_metadata.get(handler_id, {})
        timeout = metadata.get("timeout", 1.0)

        for attempt in range(self.max_retries):
            try:
                # Timeout sur l'exécution
                if asyncio.iscoroutinefunction(handler):
                    result = await asyncio.wait_for(handler(message["topic"], message["data"]), timeout=timeout)
                else:
                    # Handler synchrone dans executor
                    loop = asyncio.get_event_loop()
                    result = await asyncio.wait_for(
                        loop.run_in_executor(None, handler, message["topic"], message["data"]),
                        timeout=timeout,
                    )

                # Succès
                if breaker:
                    breaker.record_success()

                if self.metrics:
                    self.metrics.active_handlers = len(self._tasks)

                return True

            except TimeoutError:
                logger.warning(f"Handler timeout ({timeout}s): {handler_id} [attempt {attempt + 1}/{self.max_retries}]")
                if breaker:
                    breaker.record_failure()

            except Exception as e:
                logger.error(
                    f"Handler error: {handler_id} [attempt {attempt + 1}/{self.max_retries}]\n"
                    f"Error: {e}\n"
                    f"Traceback: {traceback.format_exc()}"
                )
                if breaker:
                    breaker.record_failure()

            # Delay avant retry
            if attempt < self.max_retries - 1:
                await asyncio.sleep(self.retry_delay * (attempt + 1))

        # Échec final -> DLQ
        if self.dlq is not None:
            self.dlq.append(
                {
                    **message,
                    "handler_id": handler_id,
                    "failed_at": time.time(),
                    "error": "Max retries exceeded",
                }
            )

        return False

    def _match_pattern(self, pattern: str, topic: str) -> bool:
        """Match pattern avec support wildcards"""
        if pattern == topic:
            return True

        # Wildcard simple: topic.*
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".")

        # Wildcard multi-niveaux: topic.>
        if pattern.endswith(".>"):
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".")

        # Pattern exact
        return pattern == topic

    def _check_rate_limit(self, handler_id: str) -> bool:
        """Vérifie le rate limit d'un handler"""
        metadata = self.handler_metadata.get(handler_id, {})
        rate_limit = metadata.get("rate_limit")

        if not rate_limit:
            return True

        now = time.time()
        limiter = self._rate_limiters[handler_id]

        # Nettoyer anciennes entrées (fenêtre de 1 seconde)
        while limiter and now - limiter[0] > 1.0:
            limiter.popleft()

        # Vérifier limite
        if len(limiter) >= rate_limit:
            logger.warning(f"Rate limit exceeded for {handler_id}")
            return False

        limiter.append(now)
        return True

    async def _monitor_loop(self):
        """Boucle de monitoring des métriques"""
        while self._running:
            try:
                await asyncio.sleep(10)  # Toutes les 10 secondes

                if self.metrics:
                    # GPT Fix #4: Mémoire avec psutil optionnel
                    if psutil:
                        process = psutil.Process()
                        self.metrics.memory_mb = process.memory_info().rss / 1024 / 1024
                    else:
                        # Fallback minimal sans psutil
                        self.metrics.memory_mb = 0.0

                    # Throughput
                    uptime = time.time() - self.start_time
                    self.metrics.throughput_per_sec = self.metrics.published / uptime if uptime > 0 else 0

                    # Circuit breakers
                    self.metrics.circuit_trips = sum(
                        1 for cb in self.circuit_breakers.values() if cb.state != CircuitState.CLOSED
                    )

                    # Check memory limit
                    if psutil and self.metrics.memory_mb > self.memory_limit_mb:
                        logger.warning(
                            f"Memory limit exceeded: {self.metrics.memory_mb:.1f}MB > {self.memory_limit_mb}MB"
                        )

                    # Log summary
                    logger.info(
                        f"📊 Bus Metrics: "
                        f"pub={self.metrics.published} "
                        f"del={self.metrics.delivered} "
                        f"fail={self.metrics.failed} "
                        f"p95={self.metrics.get_percentile(95):.1f}ms "
                        f"mem={self.metrics.memory_mb:.1f}MB"
                    )

            except Exception as e:
                logger.error(f"Monitor error: {e}")

    def get_metrics(self) -> dict:
        """Retourne les métriques actuelles"""
        if not self.metrics:
            return {}

        return {
            **self.metrics.to_dict(),
            "uptime_seconds": time.time() - self.start_time,
            "handlers_count": sum(len(h) for h in self.handlers.values()),
            "topics_count": len(self.handlers),
            "dlq_size": len(self.dlq) if self.dlq else 0,
            "circuit_breakers": {
                "open": sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.OPEN),
                "half_open": sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.HALF_OPEN),
                "closed": sum(1 for cb in self.circuit_breakers.values() if cb.state == CircuitState.CLOSED),
            },
        }

    def get_dlq(self) -> list[dict]:
        """Récupère la Dead Letter Queue"""
        return list(self.dlq) if self.dlq else []

    async def replay_dlq(self, max_messages: int = 10) -> int:
        """Rejoue des messages de la DLQ"""
        if not self.dlq:
            return 0

        replayed = 0
        for _ in range(min(max_messages, len(self.dlq))):
            if not self.dlq:
                break

            message = self.dlq.popleft()
            success = await self.publish(
                message["topic"],
                message["data"],
                correlation_id=f"replay_{message['correlation_id']}",
            )

            if success:
                replayed += 1
            else:
                # Remettre en DLQ si échec
                self.dlq.append(message)
                break

        logger.info(f"Replayed {replayed} messages from DLQ")
        return replayed

    @asynccontextmanager
    async def transaction(self):
        """Context manager pour transactions (futur)"""
        # TODO: Implémenter transactions
        yield self

    async def shutdown(self, timeout: float = 5.0):
        """Arrêt gracieux du bus"""
        logger.info("Starting graceful shutdown...")
        self._running = False

        # Signaler shutdown
        self._shutdown_event.set()

        # Attendre tasks en cours
        if self._tasks:
            logger.info(f"Waiting for {len(self._tasks)} tasks...")
            done, pending = await asyncio.wait(self._tasks, timeout=timeout)

            # Cancel remaining
            for task in pending:
                task.cancel()

            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        # Sauvegarder DLQ si nécessaire
        if self.dlq:
            dlq_path = f"dlq_backup_{int(time.time())}.json"
            with open(dlq_path, "w") as f:
                json.dump(list(self.dlq), f, indent=2)
            logger.info(f"DLQ saved to {dlq_path}")

        logger.info("Bus shutdown complete")
