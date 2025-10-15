"""
Adapter minimal pour compatibilité loops → NeuralBus
Permet connexion sans modifier les loops existantes
"""

import asyncio
import logging
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


class NeuroBusAdapter:
    """Wrapper de compatibilité pour NeuralBus existant"""

    def __init__(self, neural_bus):
        self._bus = neural_bus  # NeuralBus V2 existant
        self._subscriptions = {}
        self.bus_dropped_count = 0  # Compteur de messages perdus
        self.dlq_count = 0  # Compteur de messages envoyés au DLQ
        self.corrupted_count = 0  # Compteur de messages corrompus
        self._published = 0  # Compteur de messages publiés avec succès

    async def connect(self):
        """
        Initialise le bus si nécessaire (no-op si déjà fait)
        """
        if not self._bus:
            logger.warning("No NeuralBus instance to connect")
            return

        # Vérifier si déjà initialisé
        if hasattr(self._bus, "initialized") and self._bus.initialized:
            logger.debug("NeuralBus already initialized")
            return

        # Essayer d'initialiser
        if hasattr(self._bus, "initialize"):
            await self._bus.start()
            logger.info("NeuralBus initialized via initialize()")
        elif hasattr(self._bus, "connect"):
            await self._bus.connect()
            logger.info("NeuralBus connected via connect()")
        else:
            logger.debug("NeuralBus has no initialize/connect method")

    async def publish(self, event: str, data: dict, timeout: float = 0.5) -> str:
        """
        Compatible avec safe_publish() des loops
        Map vers format attendu par NeuralBusV2
        """
        # Validation des données AVANT publication
        is_corrupted, corruption_reason = self._validate_data(data)

        if is_corrupted:
            self.corrupted_count += 1
            self.dlq_count += 1

            # Routage vers DLQ avec raison de corruption
            dlq_event = f"dlq.{event}"
            dlq_data = {
                "original_event": event,
                "original_data": str(data)[:1000],  # Limiter la taille
                "corruption_reason": corruption_reason,
                "timestamp": asyncio.get_event_loop().time(),
            }

            # Publier vers DLQ sans validation
            try:
                if hasattr(self._bus, "publish"):
                    event_dict = {
                        "type": dlq_event,
                        "data": dlq_data,
                        "meta": {
                            "type": dlq_event,
                            "trace_id": str(uuid.uuid4()),
                            "priority": "low",
                            "tenant_id": "jeffrey-os",
                        },
                    }
                    await self._bus.publish(event_dict)
                    logger.warning(f"Corrupted message routed to DLQ: {corruption_reason}")
            except Exception as e:
                logger.error(f"Failed to route to DLQ: {e}")

            # Retourner un ID même pour les messages corrompus
            return str(uuid.uuid4())

        # Extraction du trace_id si présent
        trace_id = data.pop("_trace_id", str(uuid.uuid4()))

        # Détection automatique de la priorité
        priority = "normal"
        if "awareness" in event or "emotional" in event:
            priority = "high"  # Loops critiques
        elif "curiosity" in event:
            priority = "low"  # Peut attendre

        # Create event dict directly - NeuralBusV2 expects a dict
        event_dict = {
            "type": event,
            "data": data,
            "meta": {
                "type": event,
                "trace_id": trace_id,
                "priority": priority,
                "tenant_id": "jeffrey-os",
            },
        }

        # Publication via NeuralBus
        if hasattr(self._bus, "publish"):
            result = await self._bus.publish(event_dict)
            self._published += 1  # Increment success counter
            return result
        else:
            logger.warning(f"Bus has no publish method, dropping: {event}")
            self.bus_dropped_count += 1
            return str(uuid.uuid4())

    async def subscribe(self, topic: str, handler: Callable) -> str:
        """
        Subscribe avec gestion robuste des différentes APIs du NeuralBus
        """

        async def adapted_handler(msg):
            """Adapte le format du message pour les loops"""
            # Gestion robuste des différents formats possibles
            if hasattr(msg, "type"):  # CloudEvent
                message = {
                    "topic": msg.type,
                    "payload": msg.data if hasattr(msg, "data") else msg.payload,
                    "headers": {
                        "trace_id": getattr(msg.meta, "trace_id", None) if hasattr(msg, "meta") else None,
                        "event_id": getattr(msg, "id", str(uuid.uuid4())),
                    },
                }
            elif hasattr(msg, "payload"):  # Message NATS
                message = {
                    "topic": topic,
                    "payload": msg.payload,
                    "headers": getattr(msg, "headers", {}),
                }
            else:  # Format inconnu, on passe tel quel
                message = {"topic": topic, "payload": msg, "headers": {}}

            return await handler(message)

        # Essayer différentes méthodes selon l'API du NeuralBus
        sub_id = None

        # Méthode 1 : subscribe direct
        if hasattr(self._bus, "subscribe"):
            sub_id = await self._bus.subscribe(topic, adapted_handler)
            logger.info(f"Subscribed via subscribe(): {topic}")

        # Méthode 2 : register_handler
        elif hasattr(self._bus, "register_handler"):
            sub_id = await self._bus.register_handler(topic, adapted_handler)
            logger.info(f"Subscribed via register_handler(): {topic}")

        # Méthode 3 : create_consumer avec callback manuel (avec await si nécessaire)
        elif hasattr(self._bus, "create_consumer"):
            # Check si c'est une coroutine
            create_consumer_func = getattr(self._bus, "create_consumer")
            if asyncio.iscoroutinefunction(create_consumer_func):
                consumer = await create_consumer_func(name=f"loop_{topic.replace('.', '_')}", subject_filter=topic)
            else:
                consumer = create_consumer_func(name=f"loop_{topic.replace('.', '_')}", subject_filter=topic)

            # Attacher le handler au consumer
            if hasattr(consumer, "set_handler"):
                consumer.set_handler(adapted_handler)
            elif hasattr(consumer, "handler"):
                consumer.handler = adapted_handler
            elif hasattr(consumer, "on_message"):
                consumer.on_message = adapted_handler
            else:
                # Fallback : enregistrer dans un registre global
                if not hasattr(self._bus, "_handlers"):
                    self._bus._handlers = {}
                self._bus._handlers[topic] = adapted_handler

            sub_id = consumer.name if hasattr(consumer, "name") else f"consumer_{topic}"
            logger.info(f"Subscribed via create_consumer(): {topic}")

        else:
            raise RuntimeError(
                f"NeuralBus API incompatible: no subscribe/register_handler/create_consumer found. "
                f"Available methods: {dir(self._bus)}"
            )

        if not sub_id:
            sub_id = f"sub_{topic}_{id(handler)}"

        self._subscriptions[sub_id] = adapted_handler
        logger.debug(f"Subscription registered: {sub_id} -> {topic}")
        return sub_id

    async def safe_publish(self, event: str, data: dict, timeout: float = 0.5):
        """Alias pour compatibilité avec loops"""
        try:
            # Utilise un timeout réel avec asyncio
            return await asyncio.wait_for(self.publish(event, data, timeout), timeout=timeout)
        except TimeoutError:
            logger.error(f"Timeout publishing {event} after {timeout}s")
            self.bus_dropped_count += 1
        except Exception as e:
            logger.error(f"Failed to publish {event}: {e}")
            self.bus_dropped_count += 1

    def _validate_data(self, data: Any) -> tuple[bool, str]:
        """
        Valide les données avant publication
        Retourne (is_corrupted, reason)
        """
        # Vérifier si None ou pas un dict
        if data is None:
            return True, "null_data"

        if not isinstance(data, dict):
            return True, "not_a_dict"

        # Vérifier la taille (>1MB)
        try:
            import sys

            if sys.getsizeof(str(data)) > 1024 * 1024:
                return True, "oversized"
        except:
            pass

        # Vérifier structure minimum
        if "CORRUPTED" in str(data):
            return True, "test_corruption"

        # Vérifier types critiques
        for key, value in data.items():
            if key == "timestamp" and not isinstance(value, (int, float)):
                return True, "invalid_timestamp_type"
            if key == "cycle" and not isinstance(value, int):
                return True, "invalid_cycle_type"

        return False, "valid"

    def get_metrics(self) -> dict[str, Any]:
        """
        Expose métriques avec mapping robuste des noms
        """
        if not hasattr(self._bus, "get_metrics"):
            return {
                "published": self._published,
                "consumed": 0,
                "dropped": self.bus_dropped_count,
                "dlq_count": self.dlq_count,
                "corrupted_count": self.corrupted_count,
                "p95_latency_ms": 0,
                "p99_latency_ms": 0,
                "pending_messages": 0,
                "compressed_count": 0,
                "adapter_dropped": self.bus_dropped_count,
            }

        m = self._bus.get_metrics() or {}

        # Mapping intelligent des noms possibles
        return {
            "published": self._published or m.get("published") or m.get("pub") or m.get("events_published") or 0,
            "consumed": m.get("consumed") or m.get("recv") or m.get("events_consumed") or 0,
            "dropped": (m.get("dropped") or m.get("drops") or m.get("events_dropped") or 0) + self.bus_dropped_count,
            "dlq_count": self.dlq_count,
            "corrupted_count": self.corrupted_count,
            "p95_latency_ms": m.get("p95_latency_ms") or m.get("p95") or m.get("p95_ms") or 0,
            "p99_latency_ms": m.get("p99_latency_ms") or m.get("p99") or m.get("p99_ms") or 0,
            "pending_messages": m.get("pending_messages") or m.get("pending") or m.get("backlog") or 0,
            "compressed_count": m.get("compressed_count") or m.get("compressed") or 0,
            # Ajouter les métriques spécifiques si présentes
            "throughput": m.get("throughput") or 0,
            "error_rate": m.get("error_rate") or 0,
            "adapter_dropped": self.bus_dropped_count,
        }
