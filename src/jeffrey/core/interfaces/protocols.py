"""
Interfaces unifiées pour tous les modules
Utilise Protocol pour le duck typing avec vérification runtime
"""

import hashlib
import re
from collections.abc import Awaitable
from typing import Any, Protocol, Union, runtime_checkable

# Types
AwaitableOrValue = Union[Awaitable[Any], Any]


@runtime_checkable
class MemoryModule(Protocol):
    """Interface standard pour tous les modules de mémoire"""

    def capabilities(self) -> list[str]:
        """Retourne les capacités supportées"""
        ...

    async def store(self, payload: dict[str, Any]) -> bool:
        """Stocke une information"""
        ...

    async def recall(self, user_id: str, limit: int = 5) -> list[dict]:
        """Rappelle des souvenirs récents"""
        ...

    async def search(self, query: str, user_id: str | None = None) -> list[dict]:
        """Recherche dans les mémoires"""
        ...

    async def consolidate(self) -> bool:
        """Consolide/optimise les mémoires"""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques"""
        ...


@runtime_checkable
class EmotionModule(Protocol):
    """Interface standard pour tous les modules d'émotions"""

    async def analyze(self, text: str) -> dict[str, float]:
        """Analyse l'émotion d'un texte"""
        ...

    async def update_state(self, state: dict[str, float]) -> None:
        """Met à jour l'état émotionnel"""
        ...

    def get_current_state(self) -> dict[str, float]:
        """Retourne l'état actuel"""
        ...

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques"""
        ...


# Utilitaires
def normalize_text(text: str) -> str:
    """Normalise un texte pour la déduplication"""
    return re.sub(r"\s+", " ", text.strip().lower())


def memory_hash(text: str, user_id: str, role: str) -> str:
    """Génère un hash stable pour la déduplication"""
    normalized = normalize_text(text) + user_id + role
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def trimmed_mean(values: list[float], trim: float = 0.2) -> float:
    """Moyenne avec trimming des outliers"""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = int(len(sorted_values) * trim)
    if len(sorted_values) > 2 * k:
        sorted_values = sorted_values[k : len(sorted_values) - k]
    return sum(sorted_values) / len(sorted_values) if sorted_values else 0.0
