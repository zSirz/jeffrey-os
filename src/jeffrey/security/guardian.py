"""
Module de module de sécurité pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de module de sécurité pour jeffrey os.
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

import re

from jeffrey.core.neural_envelope import NeuralEnvelope


class Guardian:
    """
    CORRECTION CRITIQUE : Guardian check AVANT thalamus
    Protège PII et sécurité
    """

    def __init__(self) -> None:
        self.pii_patterns = [
            (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),
            (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
            (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "credit_card"),
        ]

    async def start(self, bus, registry):
        async def check(env: NeuralEnvelope):
            text = str(env.payload.get("text", ""))

            # Détection PII
            for pattern, pii_type in self.pii_patterns:
                if re.search(pattern, text):
                    env.tags.append(f"pii:{pii_type}")

            # Redaction si nécessaire
            if any("pii" in tag for tag in env.tags):
                for pattern, _ in self.pii_patterns:
                    text = re.sub(pattern, "[REDACTED]", text)
                env.payload["text"] = text
                env.tags.append("redacted")

            return {"ok": True, "tags": env.tags}

        bus.register_handler("guardian.check", check)

        await registry.register("guardian", self, topics_in=["guardian.check"], topics_out=[])
