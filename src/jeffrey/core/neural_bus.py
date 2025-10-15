#!/usr/bin/env python3
"""
NeuralBus - Système nerveux central de Jeffrey OS avec Redis
Corrigé : Race condition, wildcard handlers, filtres async/sync
"""

import asyncio
import inspect
import json
import logging
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

# Redis pour scalabilité (Grok suggestion)
try:
    from redis.asyncio import Redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available, using in-memory fallback")

logger = logging.getLogger(__name__)


class EventPriority(Enum):
    """Priorités des événements neuronaux"""

    CRITICAL = 1  # Réflexes, sécurité
    HIGH = 2  # Conscience, décisions
    NORMAL = 3  # Mémoire, traitement
    LOW = 4  # Background, maintenance


@dataclass
class NeuralEnvelope:
    """Enveloppe standard pour tous les messages"""

    topic: str
    payload: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    priority: EventPriority = EventPriority.NORMAL
    source: str | None = None
    correlation_id: str | None = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["priority"] = self.priority.value
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "NeuralEnvelope":
        data["priority"] = EventPriority(data.get("priority", 3))
        return cls(**data)


class NeuralConnector:
    """Connecteur pour simplifier l'accès au bus depuis un module"""

    def __init__(self, bus: "NeuralBus", module_name: str):
        self.bus = bus
        self.module_name = module_name

    async def emit(
        self,
        topic: str,
        payload: dict[str, Any],
        priority: EventPriority = EventPriority.NORMAL,
        wait_for_response: bool = False,
        timeout: float = 5.0,
    ) -> Any | None:
        """Émet un événement sur le bus"""
        envelope = NeuralEnvelope(topic=topic, payload=payload, priority=priority, source=self.module_name)
        return await self.bus.publish(envelope, wait_for_response, timeout)

    def subscribe(self, topic: str, handler: Callable) -> None:
        """S'abonne à un topic"""
        self.bus.register_handler(topic, handler)


class NeuralBus:
    """
    Bus de messages neuronal avec support Redis
    Corrections GPT appliquées + optimisations Grok
    """

    def __init__(self, redis_url: str | None = "redis://localhost:6379"):
        # Redis pour scalabilité (Grok)
        self.redis: Redis | None = None
        self.redis_url = redis_url
        self._use_redis = REDIS_AVAILABLE and redis_url

        # Handlers par topic
        self._handlers: dict[str, list[Callable]] = {}

        # Files de priorité in-memory (fallback si pas Redis)
        self._queues: dict[EventPriority, asyncio.Queue] = {priority: asyncio.Queue() for priority in EventPriority}

        # Redis pub/sub
        self._pubsub = None
        self._redis_listeners: dict[str, asyncio.Task] = {}

        # Métriques
        self._metrics = {
            "events_published": 0,
            "events_processed": 0,
            "events_failed": 0,
            "by_topic": {},
            "latencies": [],
        }

        # État
        self._running = False
        self._workers: list[asyncio.Task] = []

        # Filtres (support sync et async - correction GPT)
        self._filters: list[Callable] = []

        # Historique
        self._history: list[NeuralEnvelope] = []
        self._history_limit = 1000

    async def start(self, num_workers: int = 4):
        """Démarre le bus avec Redis si disponible"""
        if self._running:
            return

        self._running = True
        logger.info(f"🧠 NeuralBus starting with {num_workers} workers")

        # Connecter Redis si disponible
        if self._use_redis:
            try:
                self.redis = Redis.from_url(self.redis_url)
                await self.redis.ping()
                self._pubsub = self.redis.pubsub()
                logger.info("✅ Redis connected for distributed messaging")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}, using in-memory")
                self._use_redis = False
                self.redis = None

        # Créer les workers
        for priority in EventPriority:
            for i in range(max(1, num_workers // len(EventPriority))):
                worker = asyncio.create_task(self._worker(priority, f"{priority.name}-{i}"))
                self._workers.append(worker)

        logger.info("✅ NeuralBus started successfully")

    async def stop(self):
        """Arrête proprement le bus"""
        self._running = False

        # Arrêter les listeners Redis
        for task in self._redis_listeners.values():
            task.cancel()

        # Fermer Redis
        if self.redis:
            await self.redis.close()

        # Arrêter les workers
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
            self._workers.clear()

        logger.info("🛑 NeuralBus stopped")

    def register_handler(self, topic: str, handler: Callable) -> None:
        """Enregistre un handler pour un topic"""
        if topic not in self._handlers:
            self._handlers[topic] = []

        self._handlers[topic].append(handler)

        # Si Redis, s'abonner au topic
        if self._use_redis and self._pubsub and topic != "*":
            asyncio.create_task(self._subscribe_redis(topic))

    def subscribe(self, topic: str, handler: Callable) -> None:
        """Alias for register_handler to maintain compatibility"""
        self.register_handler(topic, handler)

        logger.debug(f"Handler registered for topic: {topic}")

    async def _subscribe_redis(self, topic: str):
        """S'abonne à un topic Redis"""
        if topic not in self._redis_listeners:
            await self._pubsub.subscribe(topic)

            async def listen():
                async for message in self._pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            envelope = NeuralEnvelope.from_dict(data)
                            await self._process_event(envelope, f"redis-{topic}")
                        except Exception as e:
                            logger.error(f"Redis message error: {e}")

            self._redis_listeners[topic] = asyncio.create_task(listen())

    def add_filter(self, filter_func: Callable) -> None:
        """Ajoute un filtre (support sync et async)"""
        self._filters.append(filter_func)

    async def publish(
        self, envelope: NeuralEnvelope, wait_for_response: bool = False, timeout: float = 5.0
    ) -> Any | None:
        """
        Publie un événement (correction race condition GPT)
        """
        # Appliquer les filtres (support sync/async - correction GPT)
        for filter_func in self._filters:
            if inspect.iscoroutinefunction(filter_func):
                ok = await filter_func(envelope)
            else:
                ok = filter_func(envelope)

            if not ok:
                logger.warning(f"Event filtered out: {envelope.topic}")
                return None

        # Métriques
        self._metrics["events_published"] += 1
        if envelope.topic not in self._metrics["by_topic"]:
            self._metrics["by_topic"][envelope.topic] = {"published": 0, "processed": 0}
        self._metrics["by_topic"][envelope.topic]["published"] += 1

        # Historique
        self._history.append(envelope)
        if len(self._history) > self._history_limit:
            self._history.pop(0)

        # CORRECTION GPT : Créer le future AVANT l'enqueue
        response_future = None
        if wait_for_response:
            response_future = asyncio.Future()
            envelope.meta["_response_future"] = response_future

        # Publier selon le mode
        if self._use_redis and self.redis:
            # Mode Redis distribué
            await self.redis.publish(envelope.topic, json.dumps(envelope.to_dict()))
        else:
            # Mode in-memory
            queue = self._queues[envelope.priority]
            await queue.put(envelope)

        # Attendre la réponse si demandé
        if wait_for_response and response_future:
            try:
                return await asyncio.wait_for(response_future, timeout)
            except TimeoutError:
                logger.error(f"Timeout waiting for response on topic: {envelope.topic}")
                return None

        return None

    async def _worker(self, priority: EventPriority, worker_id: str):
        """Worker qui traite les événements"""
        queue = self._queues[priority]

        while self._running:
            try:
                envelope = await asyncio.wait_for(queue.get(), timeout=1.0)
                await self._process_event(envelope, worker_id)
            except TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")

    async def _process_event(self, envelope: NeuralEnvelope, worker_id: str):
        """
        Traite un événement (correction wildcard GPT)
        """
        start_time = datetime.now()

        # CORRECTION GPT : Éviter les doublons avec un set
        seen_handlers = set()
        handlers = []

        # Handlers directs
        direct_handlers = self._handlers.get(envelope.topic, [])
        for handler in direct_handlers:
            if id(handler) not in seen_handlers:
                handlers.append(handler)
                seen_handlers.add(id(handler))

        # Handlers wildcard (correction pattern matching)
        for topic_pattern, pattern_handlers in self._handlers.items():
            if topic_pattern != envelope.topic and self._matches_pattern(envelope.topic, topic_pattern):
                for handler in pattern_handlers:
                    if id(handler) not in seen_handlers:
                        handlers.append(handler)
                        seen_handlers.add(id(handler))

        if not handlers:
            logger.debug(f"No handlers for topic: {envelope.topic}")
            return

        # Exécuter les handlers
        results = []
        for handler in handlers:
            try:
                result = await handler(envelope)
                results.append(result)

                # Stocker la première réponse pour wait_for_response
                if "_response_future" in envelope.meta:
                    future = envelope.meta["_response_future"]
                    if not future.done():
                        future.set_result(result)

            except Exception as e:
                logger.error(f"Handler error for topic {envelope.topic}: {e}")
                self._metrics["events_failed"] += 1

        # Métriques
        self._metrics["events_processed"] += 1
        if envelope.topic in self._metrics["by_topic"]:
            self._metrics["by_topic"][envelope.topic]["processed"] += 1

        latency = (datetime.now() - start_time).total_seconds()
        self._metrics["latencies"].append(latency)
        if len(self._metrics["latencies"]) > 100:
            self._metrics["latencies"].pop(0)

        logger.debug(f"[{worker_id}] Processed {envelope.topic} in {latency:.3f}s")

    def _matches_pattern(self, topic: str, pattern: str) -> bool:
        """
        CORRECTION GPT : Support correct du wildcard
        """
        if pattern == "*":  # Match tout
            return True
        if pattern == topic:  # Match exact
            return True
        if pattern.endswith(".*"):  # Prefix match
            prefix = pattern[:-2]
            return topic.startswith(prefix + ".")
        return False

    def get_metrics(self) -> dict[str, Any]:
        """Retourne les métriques du bus"""
        avg_latency = (
            sum(self._metrics["latencies"]) / len(self._metrics["latencies"]) if self._metrics["latencies"] else 0
        )

        return {
            **self._metrics,
            "avg_latency": avg_latency,
            "queue_sizes": {priority.name: queue.qsize() for priority, queue in self._queues.items()},
            "num_handlers": sum(len(h) for h in self._handlers.values()),
            "topics": list(self._handlers.keys()),
            "redis_connected": self.redis is not None,
        }

    def get_history(self, topic: str | None = None, limit: int = 100) -> list[dict]:
        """Retourne l'historique des événements"""
        history = self._history
        if topic:
            history = [e for e in history if e.topic == topic]

        return [e.to_dict() for e in history[-limit:]]


# Singleton pour toute l'application
_neural_bus_instance: NeuralBus | None = None


def get_neural_bus(redis_url: str | None = None) -> NeuralBus:
    """Retourne l'instance singleton du bus"""
    global _neural_bus_instance
    if _neural_bus_instance is None:
        _neural_bus_instance = NeuralBus(redis_url)
    return _neural_bus_instance
