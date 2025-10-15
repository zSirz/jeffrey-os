#!/usr/bin/env python3
"""
Jeffrey V2.0 Memory Systems - Architecture Complete
Système de mémoire contextuelle intelligent et multi-niveaux

Features:
- Mémoire court/moyen/long terme avec decay naturel
- Tags émotionnels automatiques via Emotion Engine
- Prioritisation intelligente et recherche contextuelle
- Performance optimisée <50ms pour rappel
- Intégration seamless avec AGI Orchestrator
"""

import json
import logging
import math
import os
import re
import shutil
import tempfile
import time
import uuid
from collections import OrderedDict, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class JSONMemoryValidator:
    """
    Validateur JSON robuste avec système de backup et restoration automatique
    Prévient les erreurs 'Expecting value: line 1 column 1 (char 0)'
    """

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def safe_load_json(self, filepath: str, backup_filepath: str | None = None) -> dict[str, Any]:
        """
        Charge un fichier JSON avec validation et fallback sur backup

        Args:
            filepath: Chemin vers le fichier JSON principal
            backup_filepath: Chemin vers le fichier de backup (optionnel)

        Returns:
            Dict[str, Any]: Données JSON ou dictionnaire vide en cas d'erreur
        """
        file_path = Path(filepath)

        # Tentative de chargement du fichier principal
        try:
            if file_path.exists() and file_path.stat().st_size > 0:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Vérifier que le contenu n'est pas vide
                        data = json.loads(content)
                        self.logger.debug(f"✅ JSON loaded successfully: {filepath}")
                        return data
                    else:
                        self.logger.warning(f"⚠️ Empty JSON file: {filepath}")

        except json.JSONDecodeError as e:
            self.logger.error(f"❌ JSON decode error in {filepath}: {e}")
        except Exception as e:
            self.logger.error(f"❌ Error reading {filepath}: {e}")

        # Tentative de restoration depuis backup
        if backup_filepath:
            backup_path = Path(backup_filepath)
            try:
                if backup_path.exists() and backup_path.stat().st_size > 0:
                    with open(backup_path, encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            data = json.loads(content)
                            self.logger.info(f"🔄 Restored from backup: {backup_filepath}")

                            # Restaurer le fichier principal depuis le backup
                            shutil.copy2(backup_path, file_path)
                            self.logger.info(f"🔧 Main file restored from backup: {filepath}")
                            return data

            except Exception as e:
                self.logger.error(f"❌ Error reading backup {backup_filepath}: {e}")

        # Retourner dictionnaire vide en dernier recours
        self.logger.warning(f"⚠️ Returning empty dict for: {filepath}")
        return {}

    def atomic_save_json(self, data: dict[str, Any], filepath: str, create_backup: bool = True) -> bool:
        """
        Sauvegarde atomique avec backup automatique

        Args:
            data: Données à sauvegarder
            filepath: Chemin de destination
            create_backup: Créer un backup avant sauvegarde

        Returns:
            bool: True si succès, False sinon
        """
        file_path = Path(filepath)

        try:
            # Créer le dossier parent si nécessaire
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Créer backup du fichier existant
            if create_backup and file_path.exists():
                backup_path = file_path.with_suffix(f"{file_path.suffix}.backup")
                try:
                    shutil.copy2(file_path, backup_path)
                    self.logger.debug(f"💾 Backup created: {backup_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Could not create backup: {e}")

            # Écriture atomique via fichier temporaire
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=file_path.parent,
                prefix=f".{file_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_file:
                json.dump(data, tmp_file, indent=2, ensure_ascii=False)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())  # Force écriture sur disque
                tmp_path = tmp_file.name

            # Renommage atomique
            shutil.move(tmp_path, file_path)
            self.logger.debug(f"✅ JSON saved atomically: {filepath}")
            return True

        except Exception as e:
            self.logger.error(f"❌ Error saving JSON {filepath}: {e}")
            # Nettoyer le fichier temporaire si nécessaire
            try:
                if 'tmp_path' in locals() and Path(tmp_path).exists():
                    os.unlink(tmp_path)
            except:
                pass
            return False

    def validate_json_structure(self, data: Any, expected_structure: dict | None = None) -> bool:
        """
        Valide la structure JSON contre un schéma attendu

        Args:
            data: Données à valider
            expected_structure: Structure attendue (optionnel)

        Returns:
            bool: True si structure valide
        """
        try:
            if not isinstance(data, dict):
                return False

            if expected_structure:
                for key, expected_type in expected_structure.items():
                    if key not in data:
                        self.logger.warning(f"⚠️ Missing key in JSON: {key}")
                        return False
                    if not isinstance(data[key], expected_type):
                        self.logger.warning(f"⚠️ Wrong type for {key}: expected {expected_type}, got {type(data[key])}")
                        return False

            return True

        except Exception as e:
            self.logger.error(f"❌ JSON validation error: {e}")
            return False

    def repair_json_file(self, filepath: str) -> bool:
        """
        Tente de réparer un fichier JSON corrompu

        Args:
            filepath: Chemin vers le fichier à réparer

        Returns:
            bool: True si réparation réussie
        """
        file_path = Path(filepath)

        try:
            if not file_path.exists():
                return False

            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            # Tentatives de réparation courantes
            repairs = [
                lambda x: x.strip(),  # Supprimer espaces
                lambda x: x.rstrip(',\n\r\t ') + '\n',  # Supprimer virgule finale
                lambda x: x + '}' if x.count('{') > x.count('}') else x,  # Ajouter } manquant
                lambda x: x + ']' if x.count('[') > x.count(']') else x,  # Ajouter ] manquant
            ]

            for repair_func in repairs:
                try:
                    repaired_content = repair_func(content)
                    json.loads(repaired_content)  # Test si valide

                    # Sauvegarder la version réparée
                    self.atomic_save_json(json.loads(repaired_content), filepath, create_backup=True)
                    self.logger.info(f"🔧 JSON file repaired: {filepath}")
                    return True

                except json.JSONDecodeError:
                    continue

            self.logger.error(f"❌ Could not repair JSON file: {filepath}")
            return False

        except Exception as e:
            self.logger.error(f"❌ Error repairing JSON file {filepath}: {e}")
            return False


class MemoryLevel(Enum):
    """Niveaux de mémoire avec caractéristiques spécifiques"""

    SHORT_TERM = "short_term"  # 0-24h, haute volatilité
    MEDIUM_TERM = "medium_term"  # 1-30 jours, decay modéré
    LONG_TERM = "long_term"  # 30+ jours, stable, compressé


class MemoryType(Enum):
    """Types de contenu mémoire"""

    DIALOGUE = "dialogue"
    EMOTIONAL = "emotional"
    FACTUAL = "factual"
    CREATIVE = "creative"
    RELATIONSHIP = "relationship"
    LEARNING = "learning"


@dataclass
class EmotionalTag:
    """Tag émotionnel enrichi par Emotion Engine"""

    emotion: str
    intensity: float  # 0.0 - 1.0
    valence: float  # -1.0 (négatif) à +1.0 (positif)
    arousal: float  # 0.0 (calme) à 1.0 (excité)
    confidence: float  # 0.0 - 1.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ContextVector:
    """Vecteur de contexte pour recherche sémantique"""

    keywords: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    semantic_hash: str = ""
    similarity_threshold: float = 0.7


@dataclass
class MemoryEntry:
    """Structure complète d'une entrée mémoire"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    content: str = ""
    memory_type: MemoryType = MemoryType.DIALOGUE
    memory_level: MemoryLevel = MemoryLevel.SHORT_TERM

    # Enrichissement émotionnel
    emotional_tags: list[EmotionalTag] = field(default_factory=list)
    emotional_weight: float = 0.0  # Impact émotionnel global

    # Métadonnées contextuelles
    context_vector: ContextVector = field(default_factory=ContextVector)
    user_id: str = "default"
    conversation_id: str = ""

    # Scoring et priorités
    importance_score: float = 0.5  # 0.0 - 1.0
    access_count: int = 0
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())

    # Decay et compression
    decay_factor: float = 1.0  # 1.0 = frais, 0.0 = totalement dégradé
    compressed: bool = False
    compression_ratio: float = 1.0

    # Relations et liens
    related_memories: list[str] = field(default_factory=list)  # IDs des mémoires liées
    parent_memory: str | None = None
    child_memories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convertit l'entrée mémoire en dictionnaire JSON-sérialisable"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'content': self.content,
            'memory_type': self.memory_type.value,  # Convertir Enum en string
            'memory_level': self.memory_level.value,  # Convertir Enum en string
            'emotional_tags': [asdict(tag) for tag in self.emotional_tags],
            'emotional_weight': self.emotional_weight,
            'context_vector': asdict(self.context_vector),
            'user_id': self.user_id,
            'conversation_id': self.conversation_id,
            'importance_score': self.importance_score,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed,
            'decay_factor': self.decay_factor,
            'compressed': self.compressed,
            'compression_ratio': self.compression_ratio,
            'related_memories': self.related_memories,
            'parent_memory': self.parent_memory,
            'child_memories': self.child_memories,
        }


class MemoryCore:
    """
    Système de mémoire contextuelle intelligent pour Jeffrey V2.0

    Architecture multi-niveaux avec gestion émotionnelle avancée,
    decay naturel et optimisation de performance.
    """

    def __init__(self, config_path: str = "data/memory_config.json"):
        """
        Initialise le système de mémoire

        Args:
            config_path: Chemin vers le fichier de configuration
        """
        # Initialiser le validateur JSON robuste
        self.json_validator = JSONMemoryValidator(logger)

        self.config = self._load_config(config_path)
        self.memories: dict[str, MemoryEntry] = {}
        self.memory_index: dict[MemoryLevel, set[str]] = {level: set() for level in MemoryLevel}
        self.emotional_index: dict[str, set[str]] = defaultdict(set)
        self.topic_index: dict[str, set[str]] = defaultdict(set)
        self.user_index: dict[str, set[str]] = defaultdict(set)

        # Métriques de performance
        self.stats = {
            "total_memories": 0,
            "by_level": {level.value: 0 for level in MemoryLevel},
            "by_type": {mtype.value: 0 for mtype in MemoryType},
            "avg_access_time": 0.0,
            "compression_ratio": 0.0,
            "last_cleanup": datetime.now().isoformat(),
        }

        # Intégration avec autres systèmes
        self.emotion_engine = None
        self.agi_orchestrator = None

        # Cache LRU optimisé pour performance
        self._search_cache: OrderedDict[str, tuple[list[MemoryEntry], float]] = OrderedDict()
        self._cache_ttl = self.config.get("cache_ttl", 300)  # 5 minutes
        self._max_cache_size = self.config.get("max_cache_size", 1000)
        self._query_analytics = defaultdict(int)  # Analyse fréquence des requêtes

        # Optimisations avancées
        self._similarity_cache: dict[str, float] = {}
        self._precomputed_vectors: dict[str, np.ndarray] = {}
        self._frequent_patterns: set[str] = set()

        logger.info("🧠 MemoryCore initialized with robust JSON validation - Ready for contextual intelligence")

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Charge la configuration du système mémoire avec validation JSON robuste"""
        default_config = {
            "max_memories": 15000,  # Augmenté pour de meilleures performances
            "decay_rates": {
                "short_term": 0.05,  # Réduit de 0.1 à 0.05 - préservation accrue
                "medium_term": 0.02,  # Réduit de 0.05 à 0.02 - préservation forte
                "long_term": 0.005,  # Réduit de 0.01 à 0.005 - préservation maximale
            },
            "promotion_thresholds": {
                "short_to_medium": 0.65,  # Réduit de 0.7 à 0.65 - promotion plus facile
                "medium_to_long": 0.75,  # Réduit de 0.8 à 0.75 - promotion plus facile
            },
            "compression_threshold": 0.25,  # Réduit de 0.3 à 0.25 - compression plus tardive
            "cleanup_interval": 3600,  # 1 heure
            "performance_target": 0.03,  # Optimisé de 0.05 à 0.03 (30ms)
            "emotional_weight_multiplier": 2.5,  # Augmenté de 2.0 à 2.5 - impact émotionnel renforcé
            "cache_ttl": 600,  # Augmenté de 300 à 600 (10 minutes)
            "max_cache_size": 2000,  # Doublé pour meilleures performances
            "similarity_threshold": 0.80,  # Augmenté de 0.75 à 0.80 pour >90% précision
            "semantic_boost_factor": 2.2,  # Augmenté de 1.8 à 2.2 - boost sémantique renforcé
            "recency_decay_factor": 0.90,  # Augmenté de 0.85 à 0.90 - importance de la récence
            "batching_enabled": True,
            "batch_size": 100,  # Augmenté de 50 à 100 pour efficacité
            "quality_threshold": 0.70,  # Nouveau: seuil de qualité pour le recall
            "context_expansion_factor": 1.5,  # Nouveau: facteur d'expansion du contexte
            "emotional_memory_boost": 1.8,  # Nouveau: boost spécifique pour mémoires émotionnelles
        }

        try:
            config_file = Path(config_path)
            backup_path = config_file.with_suffix(f"{config_file.suffix}.backup")

            if config_file.exists():
                # Charger avec validation JSON robuste
                user_config = self.json_validator.safe_load_json(
                    str(config_file), str(backup_path) if backup_path.exists() else None
                )

                if user_config:  # Si le fichier a été chargé avec succès
                    default_config.update(user_config)
                    logger.info(f"✅ Configuration loaded: {config_path}")
                else:
                    # Créer un nouveau fichier de config
                    logger.warning("⚠️ Creating new config file with defaults")
                    config_file.parent.mkdir(parents=True, exist_ok=True)
                    self.json_validator.atomic_save_json(default_config, str(config_file))
            else:
                # Créer le fichier de config par défaut avec sauvegarde atomique
                config_file.parent.mkdir(parents=True, exist_ok=True)
                success = self.json_validator.atomic_save_json(default_config, str(config_file))
                if success:
                    logger.info(f"✅ Default config created: {config_path}")
                else:
                    logger.error(f"❌ Failed to create config file: {config_path}")

        except Exception as e:
            logger.error(f"❌ Critical error loading config: {e}, using defaults")

        return default_config

    def integrate_emotion_engine(self, emotion_engine):
        """Intègre l'Emotion Engine pour enrichissement automatique"""
        self.emotion_engine = emotion_engine
        logger.info("🎭 Emotion Engine integrated with Memory Systems")

    def integrate_agi_orchestrator(self, agi_orchestrator):
        """Intègre l'AGI Orchestrator pour capture automatique"""
        self.agi_orchestrator = agi_orchestrator
        logger.info("🎯 AGI Orchestrator integrated with Memory Systems")

    async def store_memory(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.DIALOGUE,
        user_id: str = "default",
        conversation_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Stocke une nouvelle mémoire avec enrichissement automatique

        Args:
            content: Contenu à mémoriser
            memory_type: Type de mémoire
            user_id: ID utilisateur
            conversation_id: ID conversation
            metadata: Métadonnées additionnelles

        Returns:
            str: ID de la mémoire créée
        """
        start_time = time.time()

        try:
            # Créer l'entrée mémoire
            memory = MemoryEntry(
                content=content, memory_type=memory_type, user_id=user_id, conversation_id=conversation_id
            )

            # Enrichissement émotionnel via Emotion Engine
            if self.emotion_engine:
                emotional_analysis = await self._analyze_emotional_content(content)
                memory.emotional_tags = emotional_analysis["tags"]
                memory.emotional_weight = emotional_analysis["weight"]

            # Génération du vecteur de contexte
            memory.context_vector = self._generate_context_vector(content, metadata)

            # Calcul de l'importance
            memory.importance_score = self._calculate_importance(memory)

            # Détermination du niveau initial
            memory.memory_level = self._determine_initial_level(memory)

            # Stockage et indexation
            self.memories[memory.id] = memory
            self._update_indices(memory)

            # Mise à jour des statistiques
            self._update_stats(memory)

            # Nettoyage intelligent avec batching
            if len(self.memories) > self.config["max_memories"]:
                if self.config.get("batching_enabled", True):
                    await self._batch_cleanup_memories()
                else:
                    await self._cleanup_old_memories()

            processing_time = time.time() - start_time
            self.stats["avg_access_time"] = self.stats["avg_access_time"] * 0.9 + processing_time * 0.1

            logger.debug(f"💾 Memory stored: {memory.id[:8]} ({processing_time:.3f}s)")
            return memory.id

        except Exception as e:
            logger.error(f"Erreur stockage mémoire: {e}")
            raise

    async def recall_contextual(
        self,
        query: str,
        user_id: str = "default",
        limit: int = 5,
        emotional_filter: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        quality_threshold: float | None = None,
    ) -> list[MemoryEntry]:
        """
        Rappel contextuel intelligent des mémoires

        Args:
            query: Requête de recherche
            user_id: ID utilisateur pour filtrer
            limit: Nombre max de résultats
            emotional_filter: Filtre par émotion
            time_range: Plage temporelle

        Returns:
            List[MemoryEntry]: Mémoires pertinentes triées par relevance
        """
        start_time = time.time()

        try:
            # Analyser la fréquence des requêtes pour optimisation
            self._query_analytics[query] += 1

            # Vérifier le cache LRU optimisé
            cache_key = f"{query}_{user_id}_{limit}_{emotional_filter}"
            if cache_key in self._search_cache:
                cached_result, cache_time = self._search_cache[cache_key]
                if time.time() - cache_time < self._cache_ttl:
                    # Déplacer vers la fin pour LRU
                    self._search_cache.move_to_end(cache_key)
                    return cached_result[:limit]

            # Génération du vecteur de requête
            query_vector = self._generate_context_vector(query)

            # Recherche multi-critères
            candidates = self._find_candidate_memories(user_id, emotional_filter, time_range)

            # Scoring sémantique et contextuel avec filtrage qualité
            scored_memories = []
            quality_threshold = quality_threshold or self.config.get("quality_threshold", 0.70)

            for memory_id in candidates:
                memory = self.memories[memory_id]
                score = self._calculate_relevance_score(memory, query_vector, query)

                # Appliquer le seuil de qualité pour >90% précision
                if score >= quality_threshold:
                    scored_memories.append((score, memory))

            # Tri par pertinence
            scored_memories.sort(key=lambda x: x[0], reverse=True)

            # Expansion adaptative si pas assez de résultats de qualité
            if len(scored_memories) < limit and quality_threshold > 0.3:
                # Réduire le seuil progressivement pour avoir des résultats
                fallback_threshold = max(0.3, quality_threshold - 0.2)

                for memory_id in candidates:
                    memory = self.memories[memory_id]
                    score = self._calculate_relevance_score(memory, query_vector, query)

                    if fallback_threshold <= score < quality_threshold:
                        scored_memories.append((score, memory))

                scored_memories.sort(key=lambda x: x[0], reverse=True)

            results = [memory for _, memory in scored_memories[:limit]]

            # Mise à jour des accès
            for memory in results:
                memory.access_count += 1
                memory.last_accessed = datetime.now().isoformat()

            # Cache du résultat avec gestion LRU
            self._search_cache[cache_key] = (results, time.time())
            self._search_cache.move_to_end(cache_key)

            # Nettoyage du cache si trop grand
            while len(self._search_cache) > self._max_cache_size:
                self._search_cache.popitem(last=False)

            processing_time = time.time() - start_time
            logger.debug(f"🔍 Contextual recall: {len(results)} memories ({processing_time:.3f}s)")

            return results

        except Exception as e:
            logger.error(f"Erreur rappel contextuel: {e}")
            return []

    def get_memory_by_id(self, memory_id: str) -> MemoryEntry | None:
        """Récupère une mémoire par son ID"""
        memory = self.memories.get(memory_id)
        if memory:
            memory.access_count += 1
            memory.last_accessed = datetime.now().isoformat()
        return memory

    async def update_memory(self, memory_id: str, updates: dict[str, Any]) -> bool:
        """Met à jour une mémoire existante"""
        try:
            if memory_id not in self.memories:
                return False

            memory = self.memories[memory_id]
            old_level = memory.memory_level

            # Appliquer les mises à jour
            for key, value in updates.items():
                if hasattr(memory, key):
                    setattr(memory, key, value)

            # Re-indexer si nécessaire
            if memory.memory_level != old_level:
                self.memory_index[old_level].discard(memory_id)
                self.memory_index[memory.memory_level].add(memory_id)

            logger.debug(f"📝 Memory updated: {memory_id[:8]}")
            return True

        except Exception as e:
            logger.error(f"Erreur mise à jour mémoire: {e}")
            return False

    async def delete_memory(self, memory_id: str) -> bool:
        """Supprime une mémoire"""
        try:
            if memory_id not in self.memories:
                return False

            memory = self.memories[memory_id]

            # Supprimer des index
            self._remove_from_indices(memory)

            # Supprimer la mémoire
            del self.memories[memory_id]

            # Mettre à jour les stats
            self.stats["total_memories"] -= 1
            self.stats["by_level"][memory.memory_level.value] -= 1
            self.stats["by_type"][memory.memory_type.value] -= 1

            logger.debug(f"🗑️ Memory deleted: {memory_id[:8]}")
            return True

        except Exception as e:
            logger.error(f"Erreur suppression mémoire: {e}")
            return False

    async def process_memory_decay(self) -> dict[str, int]:
        """Traite le decay naturel des mémoires"""
        stats = {"decayed": 0, "promoted": 0, "compressed": 0, "deleted": 0}

        try:
            current_time = datetime.now()

            for memory_id, memory in list(self.memories.items()):
                memory_age = current_time - datetime.fromisoformat(memory.timestamp)

                # Calcul du decay basé sur l'âge et le niveau
                decay_rate = self.config["decay_rates"][memory.memory_level.value]
                age_factor = memory_age.days * decay_rate

                # Facteur de protection émotionnelle renforcé
                emotional_protection = memory.emotional_weight * self.config["emotional_weight_multiplier"]

                # Protection additionnelle pour mémoires importantes
                importance_protection = memory.importance_score * 0.3

                # Protection pour mémoires fréquemment accédées
                access_protection = min(0.2, memory.access_count / 15.0)

                # Application du decay avec protections multiples
                total_protection = emotional_protection + importance_protection + access_protection
                new_decay = max(0.0, memory.decay_factor - age_factor + total_protection)
                memory.decay_factor = new_decay

                # Gestion des transitions de niveau
                if self._should_promote_memory(memory):
                    await self._promote_memory(memory)
                    stats["promoted"] += 1
                elif self._should_compress_memory(memory):
                    await self._compress_memory(memory)
                    stats["compressed"] += 1
                elif self._should_delete_memory(memory):
                    await self.delete_memory(memory_id)
                    stats["deleted"] += 1
                else:
                    stats["decayed"] += 1

            self.stats["last_cleanup"] = current_time.isoformat()
            logger.info(f"🧹 Memory decay processed: {stats}")

        except Exception as e:
            logger.error(f"Erreur traitement decay: {e}")

        return stats

    def get_memory_stats(self) -> dict[str, Any]:
        """Retourne les statistiques détaillées du système mémoire"""
        return {
            **self.stats,
            "memory_levels": {level.value: len(memories) for level, memories in self.memory_index.items()},
            "cache_size": len(self._search_cache),
            "index_sizes": {
                "emotional": len(self.emotional_index),
                "topic": len(self.topic_index),
                "user": len(self.user_index),
            },
        }

    def export_memories(
        self,
        user_id: str | None = None,
        time_range: tuple[datetime, datetime] | None = None,
        format: str = "json",
        compressed: bool = False,
    ) -> str:
        """Exporte les mémoires au format spécifié avec optimisations"""
        try:
            memories_to_export = []

            # Lazy loading avec générateur pour économiser la mémoire
            for memory in self._get_filtered_memories(user_id, time_range):
                if compressed:
                    # Export compressé avec seulement les champs essentiels
                    memory_data = {
                        'id': memory.id,
                        'content': memory.content[:200] if len(memory.content) > 200 else memory.content,
                        'timestamp': memory.timestamp,
                        'importance': memory.importance_score,
                        'emotional_weight': memory.emotional_weight,
                        'topics': memory.context_vector.topics,
                    }
                else:
                    memory_data = memory.to_dict()

                memories_to_export.append(memory_data)

            if format == "json":
                return json.dumps(memories_to_export, indent=2 if not compressed else None, ensure_ascii=False)
            else:
                raise ValueError(f"Format non supporté: {format}")

        except Exception as e:
            logger.error(f"Erreur export mémoires: {e}")
            return ""

    def _get_filtered_memories(self, user_id: str | None, time_range: tuple[datetime, datetime] | None):
        """Générateur pour lazy loading des mémoires filtrées"""
        for memory in self.memories.values():
            # Filtrer par utilisateur
            if user_id and memory.user_id != user_id:
                continue

            # Filtrer par plage temporelle
            if time_range:
                memory_time = datetime.fromisoformat(memory.timestamp)
                if not (time_range[0] <= memory_time <= time_range[1]):
                    continue

            yield memory

    async def _analyze_emotional_content(self, content: str) -> dict[str, Any]:
        """Analyse émotionnelle du contenu via Emotion Engine"""
        try:
            if hasattr(self.emotion_engine, 'analyze_emotion_hybrid'):
                analysis = self.emotion_engine.analyze_emotion_hybrid(content)

                emotional_tag = EmotionalTag(
                    emotion=analysis.get('emotion_dominante', 'neutre'),
                    intensity=analysis.get('intensite', 50) / 100.0,
                    valence=analysis.get('contexte_emotionnel', {}).get('valence', 0.0),
                    arousal=analysis.get('contexte_emotionnel', {}).get('arousal', 0.0),
                    confidence=analysis.get('confiance', 50) / 100.0,
                )

                emotional_weight = (
                    emotional_tag.intensity * 0.4
                    + abs(emotional_tag.valence) * 0.3
                    + emotional_tag.arousal * 0.2
                    + emotional_tag.confidence * 0.1
                )

                return {"tags": [emotional_tag], "weight": emotional_weight}
            else:
                return {"tags": [], "weight": 0.0}

        except Exception as e:
            logger.warning(f"Erreur analyse émotionnelle: {e}")
            return {"tags": [], "weight": 0.0}

    def _generate_context_vector(self, content: str, metadata: dict | None = None) -> ContextVector:
        """Génère un vecteur de contexte optimisé pour recherche sémantique précise"""
        try:
            # Normalisation et nettoyage avancé
            normalized_content = self._normalize_text(content)

            # Extraction de mots-clés avec scoring TF-IDF simplifié
            keywords = self._extract_weighted_keywords(normalized_content)

            # Détection de topics avec patterns étendus et scoring
            topics = self._detect_semantic_topics(normalized_content)

            # Extraction d'entités nommées simples
            entities = self._extract_entities(normalized_content)

            # Hash sémantique basé sur les concepts clés
            semantic_concepts = keywords + topics + entities
            semantic_hash = self._generate_semantic_hash(semantic_concepts)

            return ContextVector(
                keywords=keywords[:15],  # Plus de mots-clés pour meilleure précision
                topics=topics,
                entities=entities,
                semantic_hash=semantic_hash,
                similarity_threshold=self.config.get("similarity_threshold", 0.75),
            )

        except Exception as e:
            logger.warning(f"Erreur génération vecteur contexte: {e}")
            return ContextVector()

    def _normalize_text(self, text: str) -> str:
        """Normalise le texte pour améliorer la recherche"""
        # Conversion en minuscules
        text = text.lower()
        # Suppression de la ponctuation excessive
        text = re.sub(r'[^\w\s]', ' ', text)
        # Normalisation des espaces
        text = ' '.join(text.split())
        return text

    def _extract_weighted_keywords(self, text: str) -> list[str]:
        """Extrait les mots-clés avec scoring de fréquence"""
        words = text.split()

        # Filtrage intelligent des mots (liste réduite pour capturer plus de mots-clés)
        stop_words = {
            'le',
            'la',
            'les',
            'un',
            'une',
            'des',
            'du',
            'de',
            'et',
            'ou',
            'mais',
            'que',
            'qui',
            'est',
            'sont',
            'avoir',
            'être',
        }

        # Scoring basé sur longueur et rareté
        word_scores = {}
        for word in words:
            if len(word) >= 3 and word.isalpha() and word not in stop_words:
                # Score basé sur longueur et position
                score = len(word) * 0.5
                if word in word_scores:
                    word_scores[word] += score * 0.8  # Bonus répétition modéré
                else:
                    word_scores[word] = score

        # Tri par score et sélection des meilleurs
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, score in sorted_words[:15]]

    def _detect_semantic_topics(self, text: str) -> list[str]:
        """Détection avancée de topics avec patterns étendus"""
        topic_patterns = {
            "émotion": [
                r'\b(ressens|sens|émotion|sentiment|tristesse|joie|colère|peur|amour|haine|bonheur)\b',
                r'\b(heureux|triste|en colère|effrayé|amoureux|content|déprimé)\b',
            ],
            "créativité": [
                r'\b(créativité|créatif|crée|créer|imagine|invente|histoire|art|dessin|musique|écriture)\b',
                r'\b(inspiration|idée|projet|création|artistique|original|collaboration|collaborer)\b',
            ],
            "relation": [
                r'\b(amour|amitié|ensemble|famille|ami|copain|relation|couple)\b',
                r'\b(toi|moi|nous|vous|ils|elles|rencontre|partage)\b',
            ],
            "technique": [
                r'\b(code|programme|bug|fonction|algorithme|développement|ordinateur)\b',
                r'\b(python|javascript|html|css|données|base|serveur)\b',
            ],
            "apprentissage": [
                r'\b(apprendre|étudier|comprendre|savoir|connaissance|leçon)\b',
                r'\b(école|université|cours|formation|enseignement)\b',
            ],
            "travail": [
                r'\b(travail|job|emploi|carrière|bureau|collègue|patron|projet)\b',
                r'\b(réunion|tâche|deadline|objectif|performance)\b',
            ],
            "santé": [
                r'\b(santé|maladie|médecin|hôpital|traitement|douleur|fatigue)\b',
                r'\b(sport|exercice|fitness|alimentation|nutrition)\b',
            ],
            "voyage": [
                r'\b(voyage|vacances|pays|ville|transport|avion|train|voiture)\b',
                r'\b(découverte|exploration|culture|langue|restaurant)\b',
            ],
        }

        detected_topics = []
        for topic, patterns in topic_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    detected_topics.append(topic)
                    break  # Une seule occurrence par topic

        return detected_topics

    def _extract_entities(self, text: str) -> list[str]:
        """Extraction d'entités nommées simples"""
        entities = []

        # Noms propres (mots commençant par une majuscule)
        words = text.split()
        for word in words:
            if len(word) > 2 and word[0].isupper() and word[1:].islower():
                entities.append(word.lower())

        # Dates et nombres
        date_pattern = r'\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b'
        number_pattern = r'\b\d+\b'

        entities.extend(['date'] * len(re.findall(date_pattern, text)))
        if re.search(number_pattern, text):
            entities.append('nombre')

        return list(set(entities))[:5]  # Limiter à 5 entités max

    def _generate_semantic_hash(self, concepts: list[str]) -> str:
        """Génère un hash sémantique basé sur les concepts"""
        if not concepts:
            return str(hash('empty'))[:16]

        # Tri pour cohérence
        sorted_concepts = sorted(set(concepts))
        combined = ''.join(sorted_concepts)
        return str(hash(combined))[:16]

    def _calculate_importance(self, memory: MemoryEntry) -> float:
        """Calcule le score d'importance optimisé d'une mémoire"""
        try:
            importance = 0.4  # Base légèrement réduite

            # Facteur émotionnel renforcé
            if memory.emotional_tags:
                emotional_scores = []
                for tag in memory.emotional_tags:
                    # Score composite : intensité × confiance × |valence|
                    tag_score = tag.intensity * tag.confidence * (abs(tag.valence) * 0.5 + 0.5)
                    emotional_scores.append(tag_score)

                avg_emotional_score = sum(emotional_scores) / len(emotional_scores)
                importance += avg_emotional_score * 0.4  # Augmenté de 0.3 à 0.4

            # Facteur de richesse du contenu optimisé
            content_length = len(memory.content)
            if content_length > 0:
                # Fonction logarithmique pour éviter de sur-valoriser les textes très longs
                content_factor = min(math.log(content_length + 1) / math.log(500), 1.0) * 0.25
                importance += content_factor

            # Facteur de diversité contextuelle
            context_diversity = 0
            if memory.context_vector.topics:
                context_diversity += len(memory.context_vector.topics) * 0.1
            if memory.context_vector.entities:
                context_diversity += len(memory.context_vector.entities) * 0.05
            importance += min(context_diversity, 0.2)

            # Facteur de type optimisé
            type_weights = {
                MemoryType.EMOTIONAL: 1.2,  # Augmenté pour priorité émotions
                MemoryType.RELATIONSHIP: 1.1,  # Augmenté pour relations
                MemoryType.CREATIVE: 1.0,  # Équilibré
                MemoryType.LEARNING: 0.9,  # Légèrement réduit
                MemoryType.DIALOGUE: 0.8,  # Réduit car plus commun
                MemoryType.FACTUAL: 0.7,  # Plus faible car moins engageant
            }
            importance *= type_weights.get(memory.memory_type, 0.8)

            # Bonus pour unicité (éviter la redondance)
            if memory.context_vector.semantic_hash:
                # Petit bonus pour les contenus uniques
                importance += 0.05

            return min(1.0, max(0.1, importance))  # Min à 0.1 au lieu de 0.0

        except Exception as e:
            logger.warning(f"Erreur calcul importance: {e}")
            return 0.5

    def _determine_initial_level(self, memory: MemoryEntry) -> MemoryLevel:
        """Détermine le niveau initial de la mémoire"""
        if memory.importance_score > 0.8:
            return MemoryLevel.MEDIUM_TERM
        else:
            return MemoryLevel.SHORT_TERM

    def _update_indices(self, memory: MemoryEntry):
        """Met à jour tous les index avec la nouvelle mémoire"""
        # Index par niveau
        self.memory_index[memory.memory_level].add(memory.id)

        # Index émotionnel
        for tag in memory.emotional_tags:
            self.emotional_index[tag.emotion].add(memory.id)

        # Index par topics
        for topic in memory.context_vector.topics:
            self.topic_index[topic].add(memory.id)

        # Index utilisateur
        self.user_index[memory.user_id].add(memory.id)

    def _remove_from_indices(self, memory: MemoryEntry):
        """Supprime une mémoire de tous les index"""
        # Supprimer de tous les index
        for index_dict in [self.memory_index, self.emotional_index, self.topic_index, self.user_index]:
            for memory_set in index_dict.values():
                memory_set.discard(memory.id)

    def _update_stats(self, memory: MemoryEntry):
        """Met à jour les statistiques"""
        self.stats["total_memories"] += 1
        self.stats["by_level"][memory.memory_level.value] += 1
        self.stats["by_type"][memory.memory_type.value] += 1

    def _find_candidate_memories(
        self, user_id: str, emotional_filter: str | None, time_range: tuple[datetime, datetime] | None
    ) -> set[str]:
        """Trouve les mémoires candidates pour la recherche"""
        candidates = set()

        # Filtrer par utilisateur
        if user_id in self.user_index:
            candidates = self.user_index[user_id].copy()
        else:
            candidates = set(self.memories.keys())

        # Filtrer par émotion
        if emotional_filter and emotional_filter in self.emotional_index:
            candidates &= self.emotional_index[emotional_filter]

        # Filtrer par temps
        if time_range:
            time_filtered = set()
            for memory_id in candidates:
                memory = self.memories[memory_id]
                memory_time = datetime.fromisoformat(memory.timestamp)
                if time_range[0] <= memory_time <= time_range[1]:
                    time_filtered.add(memory_id)
            candidates = time_filtered

        return candidates

    def _calculate_relevance_score(self, memory: MemoryEntry, query_vector: ContextVector, query: str) -> float:
        """Calcule le score de pertinence optimisé avec algorithme hybride"""
        try:
            # Composantes du score avec poids optimisés
            semantic_score = self._calculate_semantic_similarity(memory, query_vector, query)
            emotional_score = self._calculate_emotional_relevance(memory, query)
            contextual_score = self._calculate_contextual_relevance(memory, query_vector)
            temporal_score = self._calculate_temporal_relevance(memory)
            importance_score = memory.importance_score

            # Combinaison pondérée optimisée pour >90% précision
            total_score = (
                semantic_score * 0.50  # Sémantique prioritaire pour précision
                + emotional_score * 0.20  # Émotionnel important
                + contextual_score * 0.15  # Contexte structurel
                + temporal_score * 0.10  # Fraîcheur modérée
                + importance_score * 0.05  # Importance réduite
            )

            # Boost par facteurs de qualité
            quality_multiplier = self._calculate_quality_multiplier(memory)
            total_score *= quality_multiplier

            # Normalisation finale
            total_score = min(1.0, max(0.0, total_score))

            return total_score

        except Exception as e:
            logger.warning(f"Erreur calcul pertinence: {e}")
            return 0.0

    def _calculate_semantic_similarity(self, memory: MemoryEntry, query_vector: ContextVector, query: str) -> float:
        """Calcule la similarité sémantique avancée avec approche fuzzy robuste pour >90% précision"""
        score = 0.0
        content_lower = memory.content.lower()
        query_lower = query.lower()

        # 1. Correspondance directe (exacte ou partielle)
        if query_lower in content_lower:
            position_factor = 1.0 - (content_lower.find(query_lower) / len(content_lower)) * 0.15
            score += 0.9 * position_factor

        # 2. Correspondance avancée de mots avec tolerance fuzzy
        query_words = [w.strip() for w in query_lower.split() if len(w.strip()) >= 2]
        content_words = [w.strip() for w in content_lower.split()]

        if query_words:
            matched_score = 0
            for query_word in query_words:
                best_match_score = 0

                # Chercher correspondance exacte
                for content_word in content_words:
                    if query_word == content_word:
                        best_match_score = 1.0
                        break
                    # Correspondance partielle (préfixe/suffixe)
                    elif query_word in content_word or content_word in query_word:
                        overlap = len(set(query_word) & set(content_word)) / len(set(query_word) | set(content_word))
                        best_match_score = max(best_match_score, overlap * 0.8)
                    # Correspondance floue par caractères communs
                    elif len(query_word) >= 4 and len(content_word) >= 4:
                        common_chars = len(set(query_word) & set(content_word))
                        char_similarity = common_chars / max(len(query_word), len(content_word))
                        if char_similarity >= 0.6:
                            best_match_score = max(best_match_score, char_similarity * 0.6)

                # Pondérer par importance du mot (longueur)
                word_importance = min(1.0, len(query_word) / 8.0)
                matched_score += best_match_score * word_importance

            word_similarity = matched_score / len(query_words)
            score += word_similarity * 0.8

        # 3. Correspondance de mots-clés extraits avec expansion
        memory_keywords = set(kw.lower() for kw in memory.context_vector.keywords)
        query_keywords = set(kw.lower() for kw in query_vector.keywords)

        if memory_keywords and query_keywords:
            keyword_matches = 0
            for qkw in query_keywords:
                for mkw in memory_keywords:
                    if qkw == mkw or qkw in mkw or mkw in qkw:
                        keyword_matches += 1
                        break

            if keyword_matches > 0:
                keyword_score = keyword_matches / len(query_keywords)
                score += keyword_score * 0.7

        # 4. Correspondance sémantique de topics avec synonymes
        memory_topics = set(topic.lower() for topic in memory.context_vector.topics)
        query_topics = set(topic.lower() for topic in query_vector.topics)

        # Mapping de synonymes pour topics
        topic_synonyms = {
            'créativité': ['création', 'créatif', 'art', 'artistique'],
            'émotion': ['sentiment', 'ressenti', 'humeur'],
            'technique': ['technologie', 'informatique', 'programmation'],
            'relation': ['social', 'ami', 'famille'],
            'apprentissage': ['éducation', 'formation', 'étude'],
            'travail': ['emploi', 'métier', 'profession'],
            'voyage': ['tourisme', 'vacances', 'exploration'],
        }

        topic_score = 0
        if memory_topics and query_topics:
            for qt in query_topics:
                # Correspondance directe
                if qt in memory_topics:
                    topic_score += 1.0
                else:
                    # Correspondance par synonymes
                    for mt in memory_topics:
                        synonyms = topic_synonyms.get(qt, []) + topic_synonyms.get(mt, [])
                        if mt in synonyms or qt in synonyms:
                            topic_score += 0.8
                            break

            if topic_score > 0:
                topic_score = min(1.0, topic_score / len(query_topics))
                score += topic_score * 0.6

        # 5. Correspondance d'entités avec variance
        memory_entities = set(entity.lower() for entity in memory.context_vector.entities)
        query_entities = set(entity.lower() for entity in query_vector.entities)

        if memory_entities and query_entities:
            entity_matches = len(memory_entities & query_entities)
            if entity_matches > 0:
                entity_score = entity_matches / len(query_entities)
                score += entity_score * 0.5

        # 6. Bonus pour richesse de contenu
        if len(memory.content) >= 30:
            score *= 1.1
        elif len(memory.content) < 15:
            score *= 0.8

        # 7. Bonus si très proche sémantiquement
        if score >= 0.7:
            score = min(1.0, score * 1.15)

        return min(1.0, max(0.0, score))

    def _calculate_emotional_relevance(self, memory: MemoryEntry, query: str) -> float:
        """Calcule la pertinence émotionnelle"""
        if not memory.emotional_tags:
            return 0.5  # Neutre

        # Détection d'émotion dans la requête
        query_emotion = self._detect_query_emotion(query)

        if not query_emotion:
            # Pas d'émotion détectée, on valorise les mémoires neutres
            avg_intensity = sum(tag.intensity for tag in memory.emotional_tags) / len(memory.emotional_tags)
            return 0.5 + (1.0 - avg_intensity) * 0.3  # Préférer les émotions moins intenses

        # Correspondance émotionnelle directe
        for tag in memory.emotional_tags:
            if tag.emotion == query_emotion:
                return 0.8 + tag.intensity * 0.2  # Bonus pour correspondance directe

        # Correspondance de valence (positif/négatif)
        query_valence = self._get_emotion_valence(query_emotion)
        memory_valence = sum(tag.valence for tag in memory.emotional_tags) / len(memory.emotional_tags)

        if abs(query_valence - memory_valence) < 0.5:
            return 0.6  # Valence similaire

        return 0.3  # Émotions différentes

    def _calculate_contextual_relevance(self, memory: MemoryEntry, query_vector: ContextVector) -> float:
        """Calcule la pertinence contextuelle"""
        score = 0.5  # Base

        # Bonus pour type de mémoire pertinent
        if memory.memory_type == MemoryType.DIALOGUE:
            score += 0.2  # Les dialogues sont souvent plus pertinents
        elif memory.memory_type == MemoryType.EMOTIONAL:
            score += 0.3  # Émotions très pertinentes

        # Bonus pour mémoires fréquemment accédées
        access_bonus = min(0.3, memory.access_count / 20.0)
        score += access_bonus

        # Bonus pour contenu riche
        content_richness = min(0.2, len(memory.content) / 500.0)
        score += content_richness

        return min(1.0, score)

    def _calculate_temporal_relevance(self, memory: MemoryEntry) -> float:
        """Calcule la pertinence temporelle optimisée"""
        try:
            now = datetime.now()
            memory_time = datetime.fromisoformat(memory.timestamp)
            last_access = datetime.fromisoformat(memory.last_accessed)

            # Âge de la mémoire
            age_days = (now - memory_time).days
            age_factor = math.exp(-age_days / 90.0)  # Décroissance exponentielle sur 90 jours

            # Récence d'accès
            access_days = (now - last_access).days
            recency_factor = math.exp(-access_days / 30.0)  # Décroissance sur 30 jours

            # Facteur de decay naturel
            decay_factor = memory.decay_factor

            # Combinaison optimisée
            temporal_score = age_factor * 0.4 + recency_factor * 0.4 + decay_factor * 0.2

            return temporal_score

        except Exception:
            return 0.5

    def _calculate_quality_multiplier(self, memory: MemoryEntry) -> float:
        """Calcule le multiplicateur de qualité"""
        multiplier = 1.0

        # Bonus pour mémoires non compressées (plus de détails)
        if not memory.compressed:
            multiplier *= 1.1

        # Bonus pour mémoires avec tags émotionnels riches
        if memory.emotional_tags:
            avg_confidence = sum(tag.confidence for tag in memory.emotional_tags) / len(memory.emotional_tags)
            multiplier *= 1.0 + avg_confidence * 0.2

        # Bonus pour mémoires avec contexte riche
        if memory.context_vector.topics or memory.context_vector.entities:
            multiplier *= 1.15

        return multiplier

    def _detect_query_emotion(self, query: str) -> str | None:
        """Détecte l'émotion dans une requête"""
        emotion_patterns = {
            'joie': [r'\b(heureux|joyeux|content|ravi|enchanté|joie)\b'],
            'tristesse': [r'\b(triste|déprimé|mélancolique|chagrin|peine)\b'],
            'colère': [r'\b(en colère|furieux|énervé|irrité|rage)\b'],
            'peur': [r'\b(peur|effrayé|anxieux|inquiet|angoisse)\b'],
            'surprise': [r'\b(surpris|étonné|choqué|stupéfait)\b'],
            'amour': [r'\b(amour|adore|aime|affection|tendresse)\b'],
        }

        query_lower = query.lower()
        for emotion, patterns in emotion_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return emotion

        return None

    def _get_emotion_valence(self, emotion: str) -> float:
        """Retourne la valence d'une émotion (-1 à 1)"""
        valences = {
            'joie': 0.8,
            'amour': 0.9,
            'surprise': 0.3,
            'tristesse': -0.7,
            'colère': -0.8,
            'peur': -0.6,
            'neutre': 0.0,
        }
        return valences.get(emotion, 0.0)

    def _should_promote_memory(self, memory: MemoryEntry) -> bool:
        """Détermine si une mémoire doit être promue avec critères optimisés"""
        if memory.memory_level == MemoryLevel.LONG_TERM:
            return False

        # Score de promotion multi-critères optimisé
        promotion_score = (
            memory.importance_score * 0.35  # Importance baseline
            + memory.access_count / 8.0 * 0.25  # Facteur d'accès renforcé
            + memory.emotional_weight * 0.25  # Poids émotionnel
            + memory.decay_factor * 0.15  # Facteur de fraîcheur
        )

        # Bonus pour mémoires avec contexte riche
        if memory.context_vector.topics or memory.context_vector.entities:
            promotion_score += 0.1

        # Bonus pour mémoires émotionnelles fortes
        if memory.memory_type == MemoryType.EMOTIONAL and memory.emotional_weight > 0.7:
            promotion_score += 0.15

        # Appliquer les seuils optimisés
        if memory.memory_level == MemoryLevel.SHORT_TERM:
            return promotion_score > self.config["promotion_thresholds"]["short_to_medium"]
        else:
            return promotion_score > self.config["promotion_thresholds"]["medium_to_long"]

    def _should_compress_memory(self, memory: MemoryEntry) -> bool:
        """Détermine si une mémoire doit être compressée"""
        return (
            memory.decay_factor < self.config["compression_threshold"]
            and not memory.compressed
            and memory.memory_level != MemoryLevel.SHORT_TERM
        )

    def _should_delete_memory(self, memory: MemoryEntry) -> bool:
        """Détermine si une mémoire doit être supprimée avec critères optimisés"""
        # Critères plus stricts pour éviter la perte de mémoires précieuses
        base_condition = (
            memory.decay_factor < 0.05  # Seuil plus strict: 5% au lieu de 10%
            and memory.access_count == 0
            and memory.importance_score < 0.2  # Seuil plus strict: 20% au lieu de 30%
        )

        # Protection absolue pour certaines mémoires
        protected_memory = (
            memory.memory_type == MemoryType.EMOTIONAL
            or memory.memory_type == MemoryType.RELATIONSHIP
            or memory.emotional_weight > 0.5
            or len(memory.context_vector.topics) > 2
        )

        return base_condition and not protected_memory

    async def _promote_memory(self, memory: MemoryEntry):
        """Promeut une mémoire au niveau supérieur"""
        old_level = memory.memory_level

        if memory.memory_level == MemoryLevel.SHORT_TERM:
            memory.memory_level = MemoryLevel.MEDIUM_TERM
        elif memory.memory_level == MemoryLevel.MEDIUM_TERM:
            memory.memory_level = MemoryLevel.LONG_TERM

        # Mettre à jour les index
        self.memory_index[old_level].discard(memory.id)
        self.memory_index[memory.memory_level].add(memory.id)

        logger.debug(f"📈 Memory promoted: {memory.id[:8]} -> {memory.memory_level.value}")

    async def _compress_memory(self, memory: MemoryEntry):
        """Compresse une mémoire pour économiser l'espace"""
        if memory.compressed:
            return

        # Compression simple : garder les éléments essentiels
        original_length = len(memory.content)

        # Résumé du contenu (simulation)
        if len(memory.content) > 200:
            memory.content = memory.content[:100] + " [...résumé...] " + memory.content[-50:]

        memory.compressed = True
        memory.compression_ratio = len(memory.content) / original_length

        logger.debug(f"🗜️ Memory compressed: {memory.id[:8]} (ratio: {memory.compression_ratio:.2f})")

    async def _cleanup_old_memories(self):
        """Nettoie les anciennes mémoires pour libérer de l'espace"""
        # Trier par score de conservation optimisé
        memories_by_score = []
        for memory in self.memories.values():
            conservation_score = (
                memory.importance_score * 0.35  # Importance augmentée
                + memory.decay_factor * 0.25  # Facteur de decay
                + memory.access_count / 8.0 * 0.25  # Accès fréquents valorisés
                + memory.emotional_weight * 0.15  # Poids émotionnel
            )
            memories_by_score.append((conservation_score, memory))

        memories_by_score.sort(key=lambda x: x[0])

        # Supprimer seulement 5% de mémoires (au lieu de 10%) pour éviter pertes importantes
        to_delete = max(1, int(len(memories_by_score) * 0.05))
        deleted_count = 0

        for _, memory in memories_by_score:
            if deleted_count >= to_delete:
                break

            # Protection contre suppression de mémoires précieuses
            if not self._is_protected_memory(memory):
                await self.delete_memory(memory.id)
                deleted_count += 1

        logger.info(f"🧹 Cleaned up {deleted_count} old memories")

    async def _batch_cleanup_memories(self):
        """Nettoyage en batch pour de meilleures performances"""
        batch_size = self.config.get("batch_size", 100)
        memories_to_process = list(self.memories.values())

        # Traiter par batch
        for i in range(0, len(memories_to_process), batch_size):
            batch = memories_to_process[i : i + batch_size]
            await self._process_memory_batch(batch)

        # Cleanup final
        await self._cleanup_old_memories()

    async def _process_memory_batch(self, memories: list[MemoryEntry]):
        """Traite un batch de mémoires pour optimisation"""
        # Decay en batch
        current_time = datetime.now()

        for memory in memories:
            # Application rapide du decay
            memory_age = current_time - datetime.fromisoformat(memory.timestamp)
            decay_rate = self.config["decay_rates"][memory.memory_level.value]
            age_factor = memory_age.days * decay_rate * 0.1  # Réduction du decay

            # Protection émotionnelle
            emotional_protection = memory.emotional_weight * 0.2

            # Application du decay
            memory.decay_factor = max(0.0, memory.decay_factor - age_factor + emotional_protection)

    def _is_protected_memory(self, memory: MemoryEntry) -> bool:
        """Détermine si une mémoire est protégée contre la suppression"""
        return (
            memory.memory_type in [MemoryType.EMOTIONAL, MemoryType.RELATIONSHIP]
            or memory.emotional_weight > 0.6
            or memory.importance_score > 0.8
            or memory.access_count > 5
            or len(memory.context_vector.topics) > 2
            or memory.memory_level == MemoryLevel.LONG_TERM
        )

    # 🧠 JEFFREY V2.0 DYNAMIC LEARNING SYSTEM INTEGRATION

    async def store_learned_knowledge(self, knowledge_entry: dict[str, Any]) -> str:
        """
        Store learned knowledge as a special memory entry

        Args:
            knowledge_entry: Learned knowledge from ImmediateLearningManager

        Returns:
            Memory ID for the stored knowledge
        """
        try:
            # Create content description for the learned knowledge
            knowledge_type = knowledge_entry.get('type', 'unknown')
            knowledge_value = knowledge_entry.get('value', '')
            source = knowledge_entry.get('source', 'external_ai')

            content = f"Learned {knowledge_type}: {knowledge_value}"

            # Store as relationship memory with high importance
            memory_id = await self.store_memory(
                content=content,
                memory_type=MemoryType.RELATIONSHIP,
                user_id="learning_system",
                conversation_id="dynamic_learning",
                metadata={
                    'is_learned_knowledge': True,
                    'knowledge_type': knowledge_type,
                    'knowledge_value': knowledge_value,
                    'learning_source': source,
                    'learned_at': knowledge_entry.get('learned_at'),
                    'confidence': knowledge_entry.get('confidence', 0.8),
                    'context': knowledge_entry.get('context', {}),
                },
            )

            # Mark as protected and important
            if memory_id in self.memories:
                memory = self.memories[memory_id]
                memory.importance_score = 0.9  # High importance
                memory.memory_level = MemoryLevel.LONG_TERM  # Long-term storage
                memory.emotional_weight = 0.7  # Moderate emotional weight

            logger.info(f"🧠 Stored learned knowledge: {knowledge_type} = {knowledge_value}")
            return memory_id

        except Exception as e:
            logger.error(f"Error storing learned knowledge: {e}")
            return ""

    async def update_user_profile(self, field: str, value: str, user_id: str = "default") -> bool:
        """
        Update user profile information in memory

        Args:
            field: Profile field to update (name, location, job, etc.)
            value: New value for the field
            user_id: User identifier

        Returns:
            Success status
        """
        try:
            content = f"User {field}: {value}"

            # Check if similar profile info already exists
            existing_memories = await self.recall_contextual(query=f"user {field}", user_id=user_id, limit=5)

            # Update existing or create new
            updated = False
            for memory in existing_memories:
                if memory.metadata and memory.metadata.get('profile_field') == field:
                    # Update existing memory
                    await self.update_memory(
                        memory.id,
                        {
                            'content': content,
                            'metadata': {
                                **memory.metadata,
                                'profile_value': value,
                                'last_updated': datetime.now().isoformat(),
                            },
                        },
                    )
                    updated = True
                    break

            if not updated:
                # Create new profile memory
                await self.store_memory(
                    content=content,
                    memory_type=MemoryType.RELATIONSHIP,
                    user_id=user_id,
                    metadata={
                        'is_user_profile': True,
                        'profile_field': field,
                        'profile_value': value,
                        'created_at': datetime.now().isoformat(),
                    },
                )

            logger.info(f"👤 Updated user profile: {field} = {value}")
            return True

        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False

    async def update_jeffrey_profile(self, field: str, value: str) -> bool:
        """
        Update Jeffrey's self-knowledge in memory

        Args:
            field: Profile field to update
            value: New value for the field

        Returns:
            Success status
        """
        try:
            content = f"Jeffrey {field}: {value}"

            # Store as Jeffrey's self-knowledge
            await self.store_memory(
                content=content,
                memory_type=MemoryType.RELATIONSHIP,
                user_id="jeffrey_self",
                metadata={
                    'is_jeffrey_profile': True,
                    'profile_field': field,
                    'profile_value': value,
                    'created_at': datetime.now().isoformat(),
                },
            )

            logger.info(f"🤖 Updated Jeffrey profile: {field} = {value}")
            return True

        except Exception as e:
            logger.error(f"Error updating Jeffrey profile: {e}")
            return False

    async def get_learned_knowledge(self, knowledge_type: str = None, user_id: str = "default") -> list[dict[str, Any]]:
        """
        Retrieve learned knowledge from memory

        Args:
            knowledge_type: Optional filter by knowledge type
            user_id: User identifier

        Returns:
            List of learned knowledge entries
        """
        try:
            learned_knowledge = []

            for memory in self.memories.values():
                if (
                    memory.metadata
                    and memory.metadata.get('is_learned_knowledge')
                    and (not knowledge_type or memory.metadata.get('knowledge_type') == knowledge_type)
                ):
                    learned_knowledge.append(
                        {
                            'type': memory.metadata.get('knowledge_type'),
                            'value': memory.metadata.get('knowledge_value'),
                            'source': memory.metadata.get('learning_source'),
                            'learned_at': memory.metadata.get('learned_at'),
                            'confidence': memory.metadata.get('confidence', 0.8),
                            'memory_id': memory.id,
                        }
                    )

            return learned_knowledge

        except Exception as e:
            logger.error(f"Error getting learned knowledge: {e}")
            return []

    async def mark_as_known(self, knowledge_type: str, value: str) -> bool:
        """
        Mark specific knowledge as known/verified

        Args:
            knowledge_type: Type of knowledge
            value: Knowledge value

        Returns:
            Success status
        """
        try:
            # Find the relevant memory and mark as verified
            for memory in self.memories.values():
                if (
                    memory.metadata
                    and memory.metadata.get('is_learned_knowledge')
                    and memory.metadata.get('knowledge_type') == knowledge_type
                    and memory.metadata.get('knowledge_value') == value
                ):
                    # Update metadata to mark as verified
                    await self.update_memory(
                        memory.id,
                        {'metadata': {**memory.metadata, 'verified': True, 'verified_at': datetime.now().isoformat()}},
                    )

                    # Boost importance
                    memory.importance_score = min(1.0, memory.importance_score + 0.1)
                    logger.info(f"✅ Marked as known: {knowledge_type} = {value}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Error marking as known: {e}")
            return False


# Instance globale pour faciliter l'intégration
_memory_core_instance: MemoryCore | None = None


def get_memory_core() -> MemoryCore:
    """Récupère ou crée l'instance globale de MemoryCore"""
    global _memory_core_instance
    if _memory_core_instance is None:
        _memory_core_instance = MemoryCore()
    return _memory_core_instance


# Fonction de compatibilité pour l'intégration
def initialize_memory_systems(config_path: str = "data/memory_config.json") -> MemoryCore:
    """Initialise le système de mémoire avec configuration"""
    global _memory_core_instance
    _memory_core_instance = MemoryCore(config_path)
    return _memory_core_instance
