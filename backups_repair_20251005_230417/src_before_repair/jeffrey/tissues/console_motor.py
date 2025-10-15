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

import json

from jeffrey.core.neural_envelope import NeuralEnvelope


class ConsoleMotor:
    """Module de sortie console pour afficher les réponses"""

    async def start(self, bus, registry):
        async def speak(env: NeuralEnvelope):
            """Affiche la réponse dans la console"""
            msg = env.payload
            if isinstance(msg, dict):
                if "text" in msg:
                    msg = msg["text"]
                elif "response" in msg:
                    msg = msg["response"]
                else:
                    msg = json.dumps(msg, ensure_ascii=False)
            elif not isinstance(msg, str):
                msg = str(msg)

            print(f"\n[JEFFREY][{env.ns}] {msg}\n")

        bus.subscribe("act.speak", speak)

        await registry.register("console_motor", self, topics_in=["act.speak"], topics_out=[])
