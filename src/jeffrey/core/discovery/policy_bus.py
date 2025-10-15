"""
Module système pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module système pour jeffrey os.
Il fournit une architecture robuste et évolutive intégrant les composants
nécessaires au fonctionnement optimal du système. L'implémentation suit
les principes de modularité et d'extensibilité pour faciliter l'évolution
future du système.

Le module gère l'initialisation, la configuration, le traitement des données,
la communication inter-composants, et la persistance des états. Il s'intègre
harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
une séparation claire des responsabilités.

L'architecture interne permet une évolution adaptative basée sur les interactions
et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
cohérente et authentique.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from enum import Enum
from typing import Any


class Domain(Enum):
    """
    Classe Domain pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    BRAIN = "brain"
    BRIDGE = "bridge"
    AVATAR = "avatar"
    SKILL = "skill"


class PolicyBus:
    """Wrapper sécurisé pour NeuralBus existant"""

    def __init__(self, neural_bus, firewall) -> None:
        self.bus = neural_bus
        self.firewall = firewall
        self.logger = logging.getLogger("policy.bus")

        # Détection des API disponibles
        self._has_subscribe = hasattr(self.bus, "subscribe")
        self._has_emit = hasattr(self.bus, "emit")
        self._has_register = hasattr(self.bus, "register_handler")
        self._has_publish = hasattr(self.bus, "publish")

        # Topics EXISTANTS (pas de préfixe jeffrey.)
        self.brain_topics = [
            "percept.text",
            "percept.visual",
            "percept.audio",
            "plan.slow",
            "plan.fast",
            "plan.execute",
            "mem.store",
            "mem.recall",
            "mem.consolidate",
            "consciousness.broadcast",
            "consciousness.state",
            "emotion.state",
            "emotion.trigger",
            "system.health",
            "system.ready",
        ]

    def make_handler(self, instance: Any, method_name: str) -> Callable:
        """Créer un handler compatible avec NeuralBus - reste ASYNC"""
        method = getattr(instance, method_name, None)

        if method is None:
            # Chercher méthodes alternatives
            for fallback in ["process", "handle_request", "execute", "run"]:
                method = getattr(instance, fallback, None)
                if method:
                    break

        if method is None:
            self.logger.warning(f"No handler found for {instance.__class__.__name__}")

            async def noop(_):
                return None

            return noop

        # Garder async ou wrapper sync en async
        if asyncio.iscoroutinefunction(method):
            return method  # laisser async tel quel
        else:

            async def wrapper(env):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, method, env)

            return wrapper

    def subscribe_guarded(self, domain: Domain, topic: str, handler: Callable):
        """Subscribe avec vérification firewall - compatible multi-API"""

        allowed, reason = self.firewall.validate_subscription(domain, topic)
        if not allowed:
            self.logger.warning(f"Subscribe blocked: {topic} - {reason}")
            return

        try:
            if self._has_subscribe:
                self.bus.subscribe(topic, handler)
            elif self._has_register:
                self.bus.register_handler(topic, handler)
            else:
                raise RuntimeError("NeuralBus has no subscribe/register_handler API")

            self.logger.info(f"✅ {domain.value} subscribed to {topic}")
        except Exception as e:
            self.logger.error(f"Failed to subscribe {topic}: {e}")

    async def emit_guarded(self, source_domain: Domain, topic: str, envelope: Any):
        """Emit avec filtrage - compatible multi-API"""

        allowed, filtered_data = self.firewall.validate_emission(
            source_domain,
            topic,
            envelope.payload if hasattr(envelope, "payload") else envelope,
        )

        if not allowed:
            self.logger.warning(f"Emit blocked: {topic}")
            return

        # Modifier payload sans remplacer envelope
        if hasattr(envelope, "payload") and filtered_data is not None:
            envelope.payload = filtered_data

        try:
            if self._has_emit:
                if asyncio.iscoroutinefunction(self.bus.emit):
                    await self.bus.emit(topic, envelope)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.bus.emit, topic, envelope)
            elif self._has_publish:
                if asyncio.iscoroutinefunction(self.bus.publish):
                    await self.bus.publish(topic, envelope)
                else:
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, self.bus.publish, topic, envelope)
            else:
                raise RuntimeError("NeuralBus has no emit/publish API")
        except Exception as e:
            self.logger.error(f"Failed to emit {topic}: {e}")
