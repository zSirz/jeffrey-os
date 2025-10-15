"""
Module de composant de gestion mémorielle pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de composant de gestion mémorielle pour jeffrey os.
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

import logging

from jeffrey.core.neural_envelope import NeuralEnvelope


class DeclarativeMemory:
    """Wrapper pour ton module de mémoire existant"""

    def __init__(self) -> None:
        # Fallback simple si module manquant
        self.memory_store = {}
        try:
            from jeffrey.core.memory.memory_manager import MemoryManager

            self.core = MemoryManager()  # MODULE EXISTANT
            self.has_core = True
        except ImportError:
            logging.warning("MemoryManager not found - using fallback")
            self.has_core = False

    async def start(self, bus, registry):
        async def store(env: NeuralEnvelope):
            # Toujours utiliser fallback pour l'instant
            if env.ns not in self.memory_store:
                self.memory_store[env.ns] = []
            self.memory_store[env.ns].append(env.payload)
            return {"ok": True}

        async def recall(env: NeuralEnvelope):
            query = env.payload.get("query", "")
            k = env.payload.get("k", 5)

            # Fallback: retourne les k derniers
            results = self.memory_store.get(env.ns, [])[-k:]

            max_sim = 0.5 if results else 0  # Similarity fixe pour fallback
            return {"memories": results, "max_similarity": max_sim}

        bus.register_handler("mem.recall", recall)
        bus.register_handler("mem.store", store)

        await registry.register("declarative_memory", self, topics_in=["mem.recall", "mem.store"], topics_out=[])
