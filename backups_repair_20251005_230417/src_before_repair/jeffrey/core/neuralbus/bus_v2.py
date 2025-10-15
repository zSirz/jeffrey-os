"""
NeuralBus V2 - Bus de messages minimal avec RPC et gestion robuste
Compatible avec FastAPI et le control plane existant
"""

import asyncio
import fnmatch
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

logger = logging.getLogger(__name__)

# Types
Envelope = dict[str, Any]
Handler = Callable[[Envelope], Awaitable[None]]


class Subscription:
    """Handle de subscription avec méthode unsubscribe"""

    def __init__(self, bus: "NeuralBusV2", pattern: str, handler: Handler):
        self.bus = bus
        self.pattern = pattern
        self.handler = handler
        self.active = True

    def unsubscribe(self):
        """Désabonne le handler proprement"""
        if self.active:
            self.bus._remove_subscription(self)
            self.active = False


class NeuralBusV2:
    """Bus de messages neuronal minimal avec RPC"""

    def __init__(self, maxsize: int = 1000, workers: int = 2):
        self.queue: asyncio.Queue[Envelope] = asyncio.Queue(maxsize=maxsize)
        self.subscriptions: list[Subscription] = []
        self.workers: list[asyncio.Task] = []
        self.running = False
        self.worker_count = workers
        self.stats = {"published": 0, "delivered": 0, "handlers": 0, "errors": 0, "queue_size": 0}

    async def start(self):
        """Démarre le bus et les workers"""
        if self.running:
            return

        self.running = True
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)

        logger.info(f"NeuralBus V2 started with {self.worker_count} workers")

    async def stop(self):
        """Arrête proprement le bus"""
        if not self.running:
            return

        self.running = False

        # Signal shutdown aux workers
        for _ in range(self.worker_count):
            await self.queue.put({"type": "__shutdown__"})

        # Attendre l'arrêt des workers
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

        logger.info("NeuralBus V2 stopped")

    async def subscribe(self, pattern: str, handler: Handler) -> Subscription:
        """
        S'abonne à un pattern de messages
        Retourne un handle avec méthode unsubscribe()
        """
        sub = Subscription(self, pattern, handler)
        self.subscriptions.append(sub)
        self.stats["handlers"] = len(self.subscriptions)

        logger.debug(f"Subscribed to {pattern}")
        return sub

    def _remove_subscription(self, sub: Subscription):
        """Retire une subscription (appelé par unsubscribe)"""
        try:
            self.subscriptions.remove(sub)
            self.stats["handlers"] = len(self.subscriptions)
        except ValueError:
            pass

    async def publish(self, message: Envelope) -> str:
        """
        Publie un message sur le bus
        Retourne l'ID du message
        """
        # Enrichir le message
        message.setdefault("data", {})
        message.setdefault("meta", {})
        message.setdefault("correlation_id", None)
        message.setdefault("trace_id", str(uuid.uuid4()))
        message["meta"]["timestamp"] = time.time()
        message["id"] = str(uuid.uuid4())

        # Publier
        await self.queue.put(message)
        self.stats["published"] += 1
        self.stats["queue_size"] = self.queue.qsize()

        return message["id"]

    async def ask(
        self, request_type: str, data: dict[str, Any], response_type: str, timeout: float = 3.0
    ) -> Envelope | None:
        """
        Pattern Request/Response avec correlation
        Envoie une requête et attend la réponse corrélée
        """
        correlation_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()

        # Handler temporaire pour capturer la réponse
        async def capture_response(envelope: Envelope):
            if envelope.get("correlation_id") == correlation_id:
                if not future.done():
                    future.set_result(envelope)

        # S'abonner temporairement
        sub = await self.subscribe(response_type, capture_response)

        try:
            # Publier la requête
            await self.publish({"type": request_type, "data": data, "correlation_id": correlation_id})

            # Attendre la réponse
            result = await asyncio.wait_for(future, timeout=timeout)
            return result

        except TimeoutError:
            logger.warning(f"Timeout waiting for {response_type}")
            return None

        finally:
            # IMPORTANT: Se désabonner pour éviter les fuites
            sub.unsubscribe()

    async def _worker(self, worker_name: str):
        """Worker qui traite les messages"""
        logger.debug(f"{worker_name} started")

        while True:
            try:
                # Récupérer un message
                envelope = await self.queue.get()

                # Signal shutdown
                if envelope.get("type") == "__shutdown__":
                    break

                # Distribuer aux handlers
                delivered = 0
                for sub in list(self.subscriptions):  # Copie pour éviter modification pendant itération
                    if sub.active and fnmatch.fnmatch(envelope["type"], sub.pattern):
                        try:
                            await sub.handler(envelope)
                            delivered += 1
                        except Exception as e:
                            logger.error(f"Handler error for {envelope['type']}: {e}")
                            self.stats["errors"] += 1

                self.stats["delivered"] += delivered
                self.stats["queue_size"] = self.queue.qsize()

            except Exception as e:
                logger.error(f"{worker_name} error: {e}")
                self.stats["errors"] += 1

        logger.debug(f"{worker_name} stopped")

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques du bus"""
        return dict(self.stats)
