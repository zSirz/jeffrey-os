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
import heapq
import logging
from collections import deque
from datetime import datetime


class GlobalWorkspace:
    """Conscience - Espace de travail global (20Hz)"""

    def __init__(self, capacity: int = 7) -> None:
        self.capacity = capacity
        self.proposals = []
        self.spotlight = None
        self.stage = deque(maxlen=capacity)

    async def start(self, bus, registry):
        self.bus = bus

        # Démarrer boucle de conscience
        asyncio.create_task(self._consciousness_loop())

        await registry.register(
            "global_workspace",
            self,
            topics_in=["workspace.propose"],
            topics_out=["consciousness.broadcast", "plan.slow"],
        )

    async def propose(self, env, salience, source):
        """Proposer pour accès conscient"""
        heapq.heappush(self.proposals, (-salience, datetime.utcnow(), env))

    async def _consciousness_loop(self):
        """Boucle à 20Hz comme le cerveau"""
        while True:
            try:
                if self.proposals:
                    _, _, winner = heapq.heappop(self.proposals)

                    self.stage.append(winner)
                    self.spotlight = winner

                    winner.add_to_path("workspace_broadcast")

                    # Diffusion globale
                    await self.bus.emit("consciousness.broadcast", winner)

                    # Déclencher S2
                    await self.bus.emit(f"{winner.ns}.plan.slow", winner)

                await asyncio.sleep(0.05)  # 20Hz

            except Exception as e:
                logging.error(f"Consciousness error: {e}")
