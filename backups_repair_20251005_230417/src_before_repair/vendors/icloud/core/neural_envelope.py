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

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class NeuralEnvelope:
    """Message nerveux universel - Comme un potentiel d'action neuronal"""

    ns: str  # Namespace (ex: avatar.001)
    topic: str  # Type de signal
    payload: dict[str, Any]  # Données principales

    # Métadonnées cognitives
    affect: dict[str, float] | None = None  # État émotionnel
    salience: float = 0.5  # Importance (0-1)
    confidence: float = 0.5  # Certitude (0-1)
    urgency: float = 0.5  # Urgence (0-1)

    # Traçabilité
    path: list[str] = field(default_factory=list)
    synapses_strength: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)  # ["pii", "sensitive", etc]

    # Identifiants
    cid: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: datetime = field(default_factory=datetime.utcnow)

    def add_to_path(self, component: str):
        """Ajoute un composant au chemin avec horodatage précis"""
        self.path.append(f"{component}@{datetime.utcnow().isoformat(timespec='milliseconds')}")
