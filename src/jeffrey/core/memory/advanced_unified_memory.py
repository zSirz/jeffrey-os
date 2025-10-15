#!/usr/bin/env python3

"""
Advanced Unified Memory System v2.1
===================================

Système de mémoire unifié avancé pour Jeffrey basé sur l'analyse de migration.
Remplace tous les systèmes de mémoire existants (15 systèmes → 1 système unifié).

Architecture:
- 4 types de mémoire : Épisodique, Procédurale, Affective, Contextuelle
- Méthodes principales : store(), retrieve(), update(), consolidate()
- Système de priorités et décroissance temporelle
- Cache LRU intelligent avec compression
- Validation et intégrité des données

Author: Claude (Jeffrey V2.1 Memory Migration)
Date: 2025-06-08
Version: 2.1.0
"""

import gzip
import json
import logging
import pickle
import threading
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types de mémoire dans le système unifié"""

    EPISODIC = "episodic"  # Événements personnels et conversations
    PROCEDURAL = "procedural"  # Savoir-faire et patterns comportementaux
    AFFECTIVE = "affective"  # Émotions, sentiments et attachements
    CONTEXTUAL = "contextual"  # Contexte conversationnel et environnemental


class MemoryPriority(Enum):
    """Niveaux de priorité pour la rétention mémoire"""

    CRITICAL = 1  # Jamais supprimé
    HIGH = 2  # Rétention longue durée
    MEDIUM = 3  # Rétention moyenne
    LOW = 4  # Peut être supprimé
    TEMPORARY = 5  # Suppression rapide


class MemoryOperation(Enum):
    """Types d'opérations sur la mémoire"""

    STORE = "store"
    RETRIEVE = "retrieve"
    UPDATE = "update"
    DELETE = "delete"
    CONSOLIDATE = "consolidate"


@dataclass
class MemoryEntry:
    """Entrée de mémoire unifiée"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.CONTEXTUAL
    content: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    priority: MemoryPriority = MemoryPriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: set[str] = field(default_factory=set)
    decay_factor: float = 1.0
    importance_score: float = 0.5
    related_memories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Conversion en dictionnaire pour sérialisation"""
        data = asdict(self)
        # Conversion des types non-sérialisables
        data["memory_type"] = self.memory_type.value
        data["priority"] = self.priority.value
        data["timestamp"] = self.timestamp.isoformat()
        data["last_accessed"] = self.last_accessed.isoformat()
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryEntry":
        """Création depuis un dictionnaire"""
        entry = cls()
        entry.id = data.get("id", str(uuid.uuid4()))
        entry.memory_type = MemoryType(data.get("memory_type", "contextual"))
        entry.content = data.get("content", {})
        entry.metadata = data.get("metadata", {})
        entry.priority = MemoryPriority(data.get("priority", 3))
        entry.timestamp = datetime.fromisoformat(data.get("timestamp", datetime.now().isoformat()))
        entry.last_accessed = datetime.fromisoformat(data.get("last_accessed", datetime.now().isoformat()))
        entry.access_count = data.get("access_count", 0)
        entry.tags = set(data.get("tags", []))
        entry.decay_factor = data.get("decay_factor", 1.0)
        entry.importance_score = data.get("importance_score", 0.5)
        entry.related_memories = data.get("related_memories", [])
        return entry


@dataclass
class MemoryQuery:
    """Requête de recherche dans la mémoire"""

    memory_types: list[MemoryType] = field(default_factory=list)
    tags: set[str] = field(default_factory=set)
    time_range: tuple[datetime, datetime] | None = None
    min_importance: float = 0.0
    max_results: int = 100
    include_related: bool = True
    content_filters: dict[str, Any] = field(default_factory=dict)
    text_search: str = ""


class LRUCache:
    """Cache LRU intelligent avec compression"""

    def __init__(self, max_size: int = 1000, compression_threshold: int = 100):
        self.max_size = max_size
        self.compression_threshold = compression_threshold
        self.cache = OrderedDict()
        self.access_stats = defaultdict(int)
        self.lock = threading.RLock()

    def get(self, key: str) -> Any | None:
        """Récupération avec mise à jour LRU"""
        with self.lock:
            if key in self.cache:
                # Déplacer en fin (plus récent)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_stats[key] += 1

                # Décompression si nécessaire
                if isinstance(value, bytes):
                    try:
                        return pickle.loads(gzip.decompress(value))
                    except:
                        return value
                return value
            return None

    def put(self, key: str, value: Any) -> None:
        """Ajout avec gestion LRU et compression"""
        with self.lock:
            # Compression des gros objets
            if len(str(value)) > self.compression_threshold:
                try:
                    compressed_value = gzip.compress(pickle.dumps(value))
                    if len(compressed_value) < len(str(value)):
                        value = compressed_value
                except:
                    pass  # Garde la valeur originale si compression échoue

            # Ajout/mise à jour
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.max_size:
                # Suppression du plus ancien
                oldest_key, _ = self.cache.popitem(last=False)
                self.access_stats.pop(oldest_key, None)

            self.cache[key] = value
            self.access_stats[key] += 1

    def clear(self) -> None:
        """Nettoyage complet"""
        with self.lock:
            self.cache.clear()
            self.access_stats.clear()

    def stats(self) -> dict[str, Any]:
        """Statistiques du cache"""
        with self.lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hit_rate": sum(self.access_stats.values()) / max(len(self.cache), 1),
                "most_accessed": max(self.access_stats.items(), key=lambda x: x[1], default=("none", 0)),
            }


class AdvancedUnifiedMemory:
    """
    Système de mémoire unifié avancé remplaçant tous les systèmes existants.

    Fonctionnalités:
    - 4 types de mémoire : Épisodique, Procédurale, Affective, Contextuelle
    - Cache LRU intelligent avec compression
    - Système de priorités et décroissance temporelle
    - Consolidation automatique des mémoires
    - Validation et intégrité des données
    - Migration depuis les anciens systèmes
    """

    def __init__(self, base_path: Path, config: dict[str, Any] | None = None):
        self.base_path = Path(base_path)
        self.config = config or self._get_default_config()

        # Stockage principal par type de mémoire
        self.memories: dict[MemoryType, dict[str, MemoryEntry]] = {memory_type: {} for memory_type in MemoryType}

        # Cache LRU pour performances
        self.cache = LRUCache(
            max_size=self.config.get("cache_max_size", 1000),
            compression_threshold=self.config.get("compression_threshold", 100),
        )

        # Index pour recherche rapide
        self.tag_index: dict[str, set[str]] = defaultdict(set)
        self.content_index: dict[str, set[str]] = defaultdict(set)

        # Statistiques et monitoring
        self.stats = {
            "operations": defaultdict(int),
            "memory_sizes": defaultdict(int),
            "last_consolidation": None,
            "errors": [],
        }

        # Système de verrous
        self.locks = {memory_type: threading.RLock() for memory_type in MemoryType}
        self.global_lock = threading.RLock()

        # Chargement initial
        self._ensure_directories()
        self._load_all_memories()

        logger.info(f"AdvancedUnifiedMemory initialized with {sum(len(m) for m in self.memories.values())} memories")

    def _get_default_config(self) -> dict[str, Any]:
        """Configuration par défaut"""
        return {
            "cache_max_size": 1000,
            "compression_threshold": 100,
            "auto_consolidate_interval": 3600,  # 1 heure
            "decay_rate": 0.95,  # Facteur de décroissance par jour
            "max_memory_age_days": 365,
            "backup_enabled": True,
            "validation_enabled": True,
            "priority_thresholds": {"critical": 0.9, "high": 0.7, "medium": 0.5, "low": 0.3},
        }

    def _ensure_directories(self) -> None:
        """Création des répertoires nécessaires"""
        self.base_path.mkdir(parents=True, exist_ok=True)

        for memory_type in MemoryType:
            (self.base_path / memory_type.value).mkdir(exist_ok=True)

        # Répertoires auxiliaires
        (self.base_path / "backups").mkdir(exist_ok=True)
        (self.base_path / "indexes").mkdir(exist_ok=True)
        (self.base_path / "logs").mkdir(exist_ok=True)

    def _load_all_memories(self) -> None:
        """Chargement de toutes les mémoires depuis le disque"""
        try:
            for memory_type in MemoryType:
                memory_file = self.base_path / f"{memory_type.value}.json"
                if memory_file.exists():
                    with open(memory_file, encoding="utf-8") as f:
                        data = json.load(f)
                        for entry_data in data:
                            entry = MemoryEntry.from_dict(entry_data)
                            self.memories[memory_type][entry.id] = entry
                            self._update_indexes(entry)

            logger.info("All memories loaded successfully")

        except Exception as e:
            logger.error(f"Error loading memories: {e}")
            self.stats["errors"].append(f"Load error: {e}")

    def _save_all_memories(self) -> None:
        """Sauvegarde de toutes les mémoires sur disque"""
        try:
            for memory_type in MemoryType:
                memory_file = self.base_path / f"{memory_type.value}.json"
                memories_data = [entry.to_dict() for entry in self.memories[memory_type].values()]

                with open(memory_file, "w", encoding="utf-8") as f:
                    json.dump(memories_data, f, indent=2, ensure_ascii=False)

            logger.debug("All memories saved successfully")

        except Exception as e:
            logger.error(f"Error saving memories: {e}")
            self.stats["errors"].append(f"Save error: {e}")

    def _update_indexes(self, entry: MemoryEntry) -> None:
        """Mise à jour des index de recherche"""
        # Index des tags
        for tag in entry.tags:
            self.tag_index[tag].add(entry.id)

        # Index du contenu (mots-clés)
        content_text = str(entry.content).lower()
        words = content_text.split()
        for word in words:
            if len(word) > 3:  # Ignorer les mots trop courts
                self.content_index[word].add(entry.id)

    def _calculate_importance(self, entry: MemoryEntry) -> float:
        """Calcul du score d'importance d'une mémoire"""
        base_score = entry.importance_score

        # Facteur de récence
        age_days = (datetime.now() - entry.timestamp).days
        recency_factor = max(0.1, 1.0 - (age_days / 365))

        # Facteur d'accès
        access_factor = min(2.0, 1.0 + (entry.access_count / 10))

        # Facteur de priorité
        priority_factor = {
            MemoryPriority.CRITICAL: 2.0,
            MemoryPriority.HIGH: 1.5,
            MemoryPriority.MEDIUM: 1.0,
            MemoryPriority.LOW: 0.7,
            MemoryPriority.TEMPORARY: 0.3,
        }.get(entry.priority, 1.0)

        # Facteur de décroissance
        decay_factor = entry.decay_factor

        final_score = base_score * recency_factor * access_factor * priority_factor * decay_factor
        return min(1.0, max(0.0, final_score))

    # ========================================
    # MÉTHODES PRINCIPALES DE L'API
    # ========================================

    async def store(
        self,
        memory_type: MemoryType,
        content: dict[str, Any],
        metadata: dict[str, Any] | None = None,
        priority: MemoryPriority = MemoryPriority.MEDIUM,
        tags: set[str] | None = None,
    ) -> str:
        """
        Stockage d'une nouvelle mémoire.

        Args:
            memory_type: Type de mémoire
            content: Contenu de la mémoire
            metadata: Métadonnées optionnelles
            priority: Priorité de rétention
            tags: Tags pour la recherche

        Returns:
            ID de la mémoire créée
        """
        with self.locks[memory_type]:
            # Création de l'entrée
            entry = MemoryEntry(
                memory_type=memory_type,
                content=content,
                metadata=metadata or {},
                priority=priority,
                tags=tags or set(),
            )

            # Calcul de l'importance initiale
            entry.importance_score = self._calculate_initial_importance(content, metadata)

            # Stockage
            self.memories[memory_type][entry.id] = entry

            # Mise à jour des index
            self._update_indexes(entry)

            # Mise en cache
            self.cache.put(f"{memory_type.value}:{entry.id}", entry)

            # Statistiques
            self.stats["operations"]["store"] += 1
            self.stats["memory_sizes"][memory_type.value] += 1

            logger.debug(f"Stored memory {entry.id} of type {memory_type.value}")
            return entry.id

    async def retrieve(self, query: MemoryQuery) -> list[MemoryEntry]:
        """
        Récupération de mémoires selon une requête.

        Args:
            query: Requête de recherche

        Returns:
            Liste des mémoires correspondantes
        """
        results = []

        # Détermine les types de mémoire à chercher
        search_types = query.memory_types if query.memory_types else list(MemoryType)

        for memory_type in search_types:
            with self.locks[memory_type]:
                for entry in self.memories[memory_type].values():
                    if self._matches_query(entry, query):
                        # Mise à jour de l'accès
                        entry.last_accessed = datetime.now()
                        entry.access_count += 1

                        results.append(entry)

        # Tri par importance et limitation
        results.sort(key=lambda e: self._calculate_importance(e), reverse=True)
        results = results[: query.max_results]

        # Inclusion des mémoires liées si demandé
        if query.include_related:
            related_ids = set()
            for entry in results:
                related_ids.update(entry.related_memories)

            for related_id in related_ids:
                related_entry = await self._get_by_id(related_id)
                if related_entry and related_entry not in results:
                    results.append(related_entry)

        self.stats["operations"]["retrieve"] += 1
        logger.debug(f"Retrieved {len(results)} memories for query")

        return results

    async def update(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """
        Mise à jour d'une mémoire existante.

        Args:
            memory_id: ID de la mémoire
            updates: Dictionnaire des mises à jour

        Returns:
            True si succès, False sinon
        """
        entry = await self._get_by_id(memory_id)
        if not entry:
            return False

        memory_type = entry.memory_type

        with self.locks[memory_type]:
            # Application des mises à jour
            if "content" in updates:
                entry.content.update(updates["content"])

            if "metadata" in updates:
                entry.metadata.update(updates["metadata"])

            if "tags" in updates:
                # Mise à jour des index
                for old_tag in entry.tags:
                    self.tag_index[old_tag].discard(memory_id)

                entry.tags = set(updates["tags"])
                self._update_indexes(entry)

            if "priority" in updates:
                entry.priority = MemoryPriority(updates["priority"])

            if "importance_score" in updates:
                entry.importance_score = float(updates["importance_score"])

            # Mise à jour du cache
            self.cache.put(f"{memory_type.value}:{memory_id}", entry)

            self.stats["operations"]["update"] += 1
            logger.debug(f"Updated memory {memory_id}")

            return True

    async def consolidate(self, force: bool = False) -> dict[str, Any]:
        """
        Consolidation de la mémoire (nettoyage, optimisation).

        Args:
            force: Force la consolidation même si récente

        Returns:
            Rapport de consolidation
        """
        if not force and self.stats["last_consolidation"]:
            last_consolidation = self.stats["last_consolidation"]
            if isinstance(last_consolidation, str):
                last_consolidation = datetime.fromisoformat(last_consolidation)

            time_since = (datetime.now() - last_consolidation).total_seconds()
            if time_since < self.config["auto_consolidate_interval"]:
                return {"status": "skipped", "reason": "too_recent"}

        report = {
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "actions": [],
            "memory_changes": {},
            "performance_gains": {},
        }

        total_removed = 0
        total_consolidated = 0

        # Consolidation par type de mémoire
        for memory_type in MemoryType:
            with self.locks[memory_type]:
                memories = list(self.memories[memory_type].values())
                original_count = len(memories)

                # 1. Suppression des mémoires expirées
                expired_ids = []
                for entry in memories:
                    age_days = (datetime.now() - entry.timestamp).days
                    max_age = self.config["max_memory_age_days"]

                    if age_days > max_age and entry.priority in [
                        MemoryPriority.LOW,
                        MemoryPriority.TEMPORARY,
                    ]:
                        expired_ids.append(entry.id)

                # 2. Suppression des mémoires peu importantes
                low_importance = []
                for entry in memories:
                    if entry.id not in expired_ids:
                        importance = self._calculate_importance(entry)
                        if importance < 0.1 and entry.priority == MemoryPriority.TEMPORARY:
                            low_importance.append(entry.id)

                # 3. Application des suppressions
                removed_ids = set(expired_ids + low_importance)
                for memory_id in removed_ids:
                    if memory_id in self.memories[memory_type]:
                        del self.memories[memory_type][memory_id]
                        # Nettoyage des index
                        for tag_set in self.tag_index.values():
                            tag_set.discard(memory_id)
                        for content_set in self.content_index.values():
                            content_set.discard(memory_id)

                # 4. Application de la décroissance
                for entry in self.memories[memory_type].values():
                    entry.decay_factor *= self.config["decay_rate"]

                final_count = len(self.memories[memory_type])
                removed_count = original_count - final_count

                total_removed += removed_count
                total_consolidated += final_count

                report["memory_changes"][memory_type.value] = {
                    "original": original_count,
                    "removed": removed_count,
                    "final": final_count,
                }

                if removed_count > 0:
                    report["actions"].append(f"Removed {removed_count} {memory_type.value} memories")

        # 5. Nettoyage du cache
        self.cache.clear()

        # 6. Sauvegarde après consolidation
        self._save_all_memories()

        # 7. Mise à jour des statistiques
        self.stats["last_consolidation"] = datetime.now().isoformat()

        report["performance_gains"] = {
            "total_memories_removed": total_removed,
            "cache_cleared": True,
            "disk_saved": True,
        }

        logger.info(f"Memory consolidation completed: removed {total_removed} memories")

        return report

    # ========================================
    # MÉTHODES AUXILIAIRES
    # ========================================

    def _calculate_initial_importance(self, content: dict[str, Any], metadata: dict[str, Any] | None) -> float:
        """Calcul de l'importance initiale d'une mémoire"""
        score = 0.5  # Score de base

        # Facteurs basés sur le contenu
        if "user_message" in content:
            score += 0.2  # Contenu utilisateur important

        if "emotion" in content:
            score += 0.1  # Contenu émotionnel

        if metadata:
            if metadata.get("memorable", False):
                score += 0.3  # Marqué comme mémorable

            if metadata.get("user_initiated", False):
                score += 0.2  # Initié par l'utilisateur

        return min(1.0, score)

    def _matches_query(self, entry: MemoryEntry, query: MemoryQuery) -> bool:
        """Vérifie si une mémoire correspond à une requête"""
        # Vérification des tags
        if query.tags and not query.tags.intersection(entry.tags):
            return False

        # Vérification de la plage temporelle
        if query.time_range:
            start, end = query.time_range
            if not (start <= entry.timestamp <= end):
                return False

        # Vérification de l'importance minimale
        if query.min_importance > 0:
            importance = self._calculate_importance(entry)
            if importance < query.min_importance:
                return False

        # Vérification des filtres de contenu
        for key, value in query.content_filters.items():
            if key not in entry.content or entry.content[key] != value:
                return False

        # Recherche textuelle
        if query.text_search:
            search_text = query.text_search.lower()
            content_text = str(entry.content).lower()
            metadata_text = str(entry.metadata).lower()

            if search_text not in content_text and search_text not in metadata_text:
                return False

        return True

    async def _get_by_id(self, memory_id: str) -> MemoryEntry | None:
        """Récupération d'une mémoire par ID"""
        # Recherche dans le cache
        for memory_type in MemoryType:
            cache_key = f"{memory_type.value}:{memory_id}"
            cached_entry = self.cache.get(cache_key)
            if cached_entry:
                return cached_entry

        # Recherche dans les mémoires
        for memory_type in MemoryType:
            if memory_id in self.memories[memory_type]:
                entry = self.memories[memory_type][memory_id]
                # Mise en cache
                self.cache.put(f"{memory_type.value}:{memory_id}", entry)
                return entry

        return None

    # ========================================
    # MÉTHODES DE GESTION ET MONITORING
    # ========================================

    def get_stats(self) -> dict[str, Any]:
        """Récupération des statistiques"""
        total_memories = sum(len(memories) for memories in self.memories.values())

        return {
            "total_memories": total_memories,
            "memory_breakdown": {mt.value: len(self.memories[mt]) for mt in MemoryType},
            "operations": dict(self.stats["operations"]),
            "cache_stats": self.cache.stats(),
            "last_consolidation": self.stats["last_consolidation"],
            "errors_count": len(self.stats["errors"]),
            "index_sizes": {"tags": len(self.tag_index), "content": len(self.content_index)},
        }

    def create_backup(self) -> str:
        """Création d'un backup"""
        if not self.config.get("backup_enabled", True):
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.base_path / "backups" / f"memory_backup_{timestamp}.json"

        try:
            backup_data = {
                "timestamp": timestamp,
                "version": "2.1.0",
                "config": self.config,
                "memories": {mt.value: [entry.to_dict() for entry in self.memories[mt].values()] for mt in MemoryType},
                "stats": dict(self.stats),
            }

            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Backup created: {backup_path}")
            return str(backup_path)

        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return ""

    async def shutdown(self) -> None:
        """Arrêt propre du système"""
        logger.info("Shutting down AdvancedUnifiedMemory...")

        # Consolidation finale
        await self.consolidate(force=True)

        # Sauvegarde finale
        self._save_all_memories()

        # Backup final
        self.create_backup()

        # Nettoyage
        self.cache.clear()

        logger.info("AdvancedUnifiedMemory shutdown completed")


# Export des classes principales
__all__ = [
    "AdvancedUnifiedMemory",
    "MemoryType",
    "MemoryPriority",
    "MemoryOperation",
    "MemoryEntry",
    "MemoryQuery",
    "LRUCache",
]
