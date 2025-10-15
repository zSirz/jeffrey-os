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
import json
import logging
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types d'événements émis par les gardiens"""

    # EthicsGuardian
    BIAS_DETECTED = "bias_detected"
    ETHICS_VIOLATION = "ethics_violation"
    ETHICS_PASSED = "ethics_passed"

    # ResourceZen
    COST_THRESHOLD = "cost_threshold"
    LIMIT_EXCEEDED = "limit_exceeded"
    USAGE_SPIKE = "usage_spike"
    PREDICTION_UPDATE = "prediction_update"

    # JeffreyAuditor
    COMPLEXITY_ALERT = "complexity_alert"
    QUALITY_DROP = "quality_drop"
    DOC_MISSING = "doc_missing"
    TEST_COVERAGE_LOW = "test_coverage_low"

    # DocZen
    DOC_GENERATED = "doc_generated"
    DOC_UPDATE_NEEDED = "doc_update_needed"

    # Symphony
    CROSS_CORRELATION = "cross_correlation"
    INSIGHT_GENERATED = "insight_generated"
    IMPROVEMENT_PROPOSED = "improvement_proposed"


@dataclass
class GuardianEvent:
    """Structure d'un événement émis par un gardien"""

    id: str
    source: str  # ethics_guardian, resource_zen, etc.
    event_type: EventType
    severity: float  # 0-1
    data: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: str | None = None

    def __lt__(self, other):
        """Pour PriorityQueue - priorité par sévérité"""
        return self.severity > other.severity

    def to_dict(self) -> dict[str, Any]:
        """Convertit l'événement en dictionnaire"""
        return {
            "id": self.id,
            "source": self.source,
            "event_type": self.event_type.value,
            "severity": self.severity,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        }


class EventBus:
    """
    Bus d'événements central pour communication inter-gardiens
    Thread-safe et supporte les callbacks synchrones et asynchrones
    """

    def __init__(self, persist_events: bool = True, log_dir: str = "logs/events") -> None:
        self._subscribers: dict[EventType, list[weakref.ref]] = {}
        self._async_subscribers: dict[EventType, list[weakref.ref]] = {}
        self._event_history: list[GuardianEvent] = []
        self._lock = threading.Lock()
        self.persist_events = persist_events
        self.log_dir = Path(log_dir)

        if persist_events:
            self.log_dir.mkdir(parents=True, exist_ok=True)

        # Statistiques
        self.stats = {"events_published": 0, "events_by_type": {}, "events_by_source": {}}

        logger.info("EventBus initialized")

    def subscribe(self, event_type: EventType, callback: Callable, weak: bool = True):
        """
        Abonne une fonction à un type d'événement

        Args:
            event_type: Type d'événement à écouter
            callback: Fonction à appeler
            weak: Utiliser une référence faible (évite les fuites mémoire)
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []

            ref = weakref.ref(callback) if weak else lambda: callback
            self._subscribers[event_type].append(ref)

        logger.info(f"Subscribed to {event_type.value}")

    def subscribe_async(self, event_type: EventType, callback: Callable, weak: bool = True):
        """Abonne une fonction asynchrone à un type d'événement"""
        with self._lock:
            if event_type not in self._async_subscribers:
                self._async_subscribers[event_type] = []

            ref = weakref.ref(callback) if weak else lambda: callback
            self._async_subscribers[event_type].append(ref)

        logger.info(f"Async subscribed to {event_type.value}")

    def publish(self, event: GuardianEvent):
        """
        Publie un événement de manière synchrone

        Args:
            event: L'événement à publier
        """
        # Statistiques
        self.stats["events_published"] += 1
        self.stats["events_by_type"][event.event_type.value] = (
            self.stats["events_by_type"].get(event.event_type.value, 0) + 1
        )
        self.stats["events_by_source"][event.source] = self.stats["events_by_source"].get(event.source, 0) + 1

        # Historique
        with self._lock:
            self._event_history.append(event)
            if len(self._event_history) > 1000:  # Limiter la mémoire
                self._event_history = self._event_history[-500:]

        # Persister si activé
        if self.persist_events:
            self._persist_event(event)

        # Notifier les abonnés synchrones
        if event.event_type in self._subscribers:
            dead_refs = []

            with self._lock:
                subscribers = self._subscribers[event.event_type].copy()

            for ref in subscribers:
                callback = ref()
                if callback:
                    try:
                        callback(event)
                    except Exception as e:
                        logger.error(f"Error in callback: {e}")
                else:
                    dead_refs.append(ref)

            # Nettoyer les références mortes
            if dead_refs:
                with self._lock:
                    for ref in dead_refs:
                        if ref in self._subscribers[event.event_type]:
                            self._subscribers[event.event_type].remove(ref)

        logger.debug(f"Published event: {event.event_type.value} from {event.source}")

    async def publish_async(self, event: GuardianEvent):
        """Publie un événement de manière asynchrone"""
        # Publier en synchrone d'abord
        self.publish(event)

        # Notifier les abonnés asynchrones
        if event.event_type in self._async_subscribers:
            tasks = []
            dead_refs = []

            with self._lock:
                subscribers = self._async_subscribers[event.event_type].copy()

            for ref in subscribers:
                callback = ref()
                if callback:
                    await tasks.append(asyncio.create_task(self._safe_async_call(callback, event)))
                else:
                    dead_refs.append(ref)

            # Nettoyer les références mortes
            if dead_refs:
                with self._lock:
                    for ref in dead_refs:
                        if ref in self._async_subscribers[event.event_type]:
                            self._async_subscribers[event.event_type].remove(ref)

            # Attendre toutes les tâches
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _safe_async_call(self, callback: Callable, event: GuardianEvent):
        """Appel sécurisé d'un callback asynchrone"""
        try:
            await callback(event)
        except Exception as e:
            logger.error(f"Error in async callback: {e}")

    def get_recent_events(
        self, event_type: EventType | None = None, source: str | None = None, limit: int = 100
    ) -> list[GuardianEvent]:
        """
        Récupère les événements récents

        Args:
            event_type: Filtrer par type
            source: Filtrer par source
            limit: Nombre maximum d'événements

        Returns:
            Liste des événements filtrés
        """
        with self._lock:
            events = self._event_history.copy()

        # Filtrer
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source:
            events = [e for e in events if e.source == source]

        # Limiter et retourner les plus récents
        return events[-limit:]

    def get_event_stats(self) -> dict[str, Any]:
        """Retourne les statistiques du bus"""
        return {
            "total_events": self.stats["events_published"],
            "events_by_type": dict(self.stats["events_by_type"]),
            "events_by_source": dict(self.stats["events_by_source"]),
            "subscribers_count": sum(len(subs) for subs in self._subscribers.values()),
            "async_subscribers_count": sum(len(subs) for subs in self._async_subscribers.values()),
            "history_size": len(self._event_history),
        }

    def _persist_event(self, event: GuardianEvent):
        """Persiste un événement sur disque"""
        try:
            # Fichier du jour
            today = datetime.now().strftime("%Y%m%d")
            log_file = self.log_dir / f"events_{today}.jsonl"

            # Ajouter l'événement
            with open(log_file, "a") as f:
                json.dump(event.to_dict(), f)
                f.write("\n")

        except Exception as e:
            logger.error(f"Failed to persist event: {e}")

    def clear_history(self):
        """Efface l'historique des événements"""
        with self._lock:
            self._event_history.clear()
        logger.info("Event history cleared")


# Instance globale du bus
guardian_bus = EventBus()
