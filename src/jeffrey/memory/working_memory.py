"""
Mémoire de travail pour traitement temps réel.

Ce module implémente les fonctionnalités essentielles pour mémoire de travail pour traitement temps réel.
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

from datetime import datetime

from jeffrey.core.neural_envelope import NeuralEnvelope


class WorkingMemory:
    """Mémoire de travail (7±2 slots comme l'humain)"""

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self.slots = {}  # {namespace: {key: (value, expires_at)}}

    async def start(self, bus, registry):
        async def put(env: NeuralEnvelope):
            ns = env.ns
            key = env.payload.get("key")
            value = env.payload.get("value")
            ttl = env.payload.get("ttl", 60)

            if ns not in self.slots:
                self.slots[ns] = {}

            # Limite de capacité
            if len(self.slots[ns]) >= self.capacity:
                # Évincer le plus ancien
                oldest = min(self.slots[ns], key=lambda k: self.slots[ns][k][1])
                del self.slots[ns][oldest]

            expires = datetime.utcnow().timestamp() + ttl
            self.slots[ns][key] = (value, expires)
            return {"ok": True}

        async def get(env: NeuralEnvelope):
            ns = env.ns
            key = env.payload.get("key")

            if ns in self.slots and key in self.slots[ns]:
                value, expires = self.slots[ns][key]
                if datetime.utcnow().timestamp() < expires:
                    return {"value": value}

            return {"value": None}

        async def snapshot(env: NeuralEnvelope):
            ns = env.ns
            context = {}

            if ns in self.slots:
                now = datetime.utcnow().timestamp()
                for key, (value, expires) in self.slots[ns].items():
                    if now < expires:
                        context[key] = value

            return {"context": context}

        bus.register_handler("mem.working.put", put)
        bus.register_handler("mem.working.get", get)
        bus.register_handler("mem.working.snapshot", snapshot)

        await registry.register(
            "working_memory",
            self,
            topics_in=["mem.working.put", "mem.working.get", "mem.working.snapshot"],
            topics_out=[],
        )
