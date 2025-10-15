"""
Memory Consolidation - Organisation intelligente des souvenirs
"""

import hashlib
import hmac
import json
import logging
import os
import sys
import time
import zlib
from collections import deque
from typing import Any

from .base import BaseLoop
from .gates import sanitize_event_data

logger = logging.getLogger(__name__)


class MemoryConsolidationLoop(BaseLoop):
    """
    Consolidation mémoire avec clustering ML
    """

    def __init__(self, memory_federation=None, budget_gate=None, bus=None):
        super().__init__(
            name="memory_consolidation",
            interval_s=60.0,
            jitter_s=5.0,
            hard_timeout_s=5.0,
            budget_gate=budget_gate,
            bus=bus,
        )
        self.memory_federation = memory_federation

        # Seuils adaptatifs
        self.short_term_threshold = 300  # 5 minutes
        self.long_term_threshold = 3600  # 1 heure

        # Stats
        self.consolidation_count = 0
        self.memories_processed = 0
        self.memories_archived = 0
        self.memories_pruned = 0

        # Clé pour HMAC (fix GPT)
        self._get_hmac_key()

        # ML clustering (si disponible)
        self.use_ml_clustering = False
        try:
            from sklearn.cluster import DBSCAN

            self.clustering_model = DBSCAN(eps=0.3, min_samples=2)
            self.use_ml_clustering = True
            logger.info("ML clustering enabled for memory consolidation")
        except ImportError:
            logger.info("ML clustering unavailable, using hash-based consolidation")

        # Compression anti-OOM (Phase 2.3)
        self.short_term = deque(maxlen=100)  # Max 100 items
        self.compressed_history = []  # Historique compressé
        self.compression_count = 0

    def _get_hmac_key(self):
        """Récupère la clé HMAC de manière sécurisée"""
        key = os.environ.get("JEFFREY_SEARCH_KEY") or os.environ.get("JEFFREY_ENCRYPTION_KEY")
        if key and len(key) == 64:
            # Clé hex
            self.hmac_key = bytes.fromhex(key)
        elif key:
            # Clé string
            self.hmac_key = key.encode()
        else:
            # Clé par défaut (dev mode)
            self.hmac_key = b"jeffrey_memory_consolidation_default_key"

    async def _tick(self):
        """Cycle de consolidation"""
        start_time = time.perf_counter()

        # Identifier les souvenirs
        memories = await self._identify_memories_safe()

        # Consolider
        if self.use_ml_clustering and len(memories) > 10:
            consolidated = await self._consolidate_with_ml(memories)
        else:
            consolidated = await self._consolidate_with_hash(memories)

        # Archiver
        archived = await self._archive_old_memories()

        # Nettoyer
        pruned = await self._prune_redundant()

        # Stats
        self.consolidation_count += 1
        self.memories_processed += len(memories)
        self.memories_archived += archived
        self.memories_pruned += pruned

        cycle_time = (time.perf_counter() - start_time) * 1000

        # Compress old memories if needed (Phase 2.3)
        if len(self.short_term) > 50:
            self._compress_old_memories()
            saved_mb = self._estimate_saved_memory()
            if saved_mb > 0:
                logger.debug(f"Compressed memories, saved ~{saved_mb:.1f}MB")

        # Publier métriques (sans contenu)
        if self.bus:
            event_data = {
                "cycle": self.consolidation_count,
                "processed": len(memories),
                "consolidated": len(consolidated),
                "archived": archived,
                "pruned": pruned,
                "duration_ms": round(cycle_time, 1),
                "method": "ml" if self.use_ml_clustering else "hash",
            }

            await self.bus.publish(
                "memory.consolidation.event",
                {
                    "topic": "memory.consolidation.complete",
                    "data": sanitize_event_data(event_data),
                    "timestamp": time.time(),
                },
            )

        logger.info(
            f"Memory consolidation #{self.consolidation_count}: "
            f"processed={len(memories)}, consolidated={len(consolidated)}, "
            f"archived={archived}, pruned={pruned}, time={cycle_time:.1f}ms"
        )

        return {"consolidated": len(consolidated), "pruned": pruned}

    async def _identify_memories_safe(self) -> list[dict]:
        """Identifie sans exposer le contenu"""
        memories = []

        if self.memory_federation:
            try:
                # Essai avec différentes signatures d'API pour compatibilité
                if hasattr(self.memory_federation, "recall_from_all"):
                    try:
                        # Essai API nouvelle avec metadata_only et limit
                        recent = await self.memory_federation.recall_from_all(query="", metadata_only=True, limit=100)
                    except TypeError:
                        # Fallback API ancienne avec max_results
                        recent = await self.memory_federation.recall_from_all(query="", max_results=100)
                else:
                    recent = []

                now = time.time()
                for memory in recent:
                    # Utiliser HMAC pour hacher de manière sécurisée (fix GPT)
                    raw = str(memory.get("text", ""))

                    # Ne JAMAIS logguer le contenu raw
                    memory_hash = hmac.new(self.hmac_key, raw.encode(), hashlib.sha256).hexdigest()[:16]

                    timestamp = memory.get("timestamp", now)
                    age = now - timestamp

                    memories.append(
                        {
                            "hash": memory_hash,
                            "age": age,
                            "tier": self._classify_tier(age),
                            "metadata": memory.get("metadata", {}),
                            "user_id": memory.get("user_id", "system"),
                            "role": memory.get("role", "assistant"),
                            "source": memory.get("source", "unknown"),
                        }
                    )
            except Exception as e:
                logger.error(f"Error identifying memories: {e}")

        return memories

    def _classify_tier(self, age: float) -> str:
        """Classifie le tier de mémoire"""
        if age < self.short_term_threshold:
            return "short_term"
        elif age < self.long_term_threshold:
            return "medium_term"
        else:
            return "long_term"

    async def _consolidate_with_hash(self, memories: list[dict]) -> list[dict]:
        """Consolidation basée sur hash"""
        groups = {}

        for memory in memories:
            # Clé composite pour grouper
            # Utilise seulement les 8 premiers chars du hash pour grouper
            key = f"{memory['user_id']}|{memory['role']}|{memory['hash'][:8]}"

            if key not in groups:
                groups[key] = []
            groups[key].append(memory)

        # Fusionner les groupes
        consolidated = []
        for key, group in groups.items():
            if len(group) > 1:
                # Garder le plus récent avec count
                merged = min(group, key=lambda x: x["age"]).copy()
                merged["consolidation_count"] = len(group)
                merged["consolidated_hashes"] = [m["hash"] for m in group]
                consolidated.append(merged)
            else:
                consolidated.append(group[0])

        return consolidated

    async def _consolidate_with_ml(self, memories: list[dict]) -> list[dict]:
        """Consolidation avec clustering ML"""
        try:
            # Créer des features simples pour clustering
            features = []
            for m in memories:
                # Features: [age_normalized, tier_encoded, role_encoded]
                age_norm = min(m["age"] / self.long_term_threshold, 1.0)
                tier_map = {"short_term": 0, "medium_term": 0.5, "long_term": 1}
                tier_enc = tier_map.get(m["tier"], 0.5)
                role_enc = 0 if m["role"] == "user" else 1
                features.append([age_norm, tier_enc, role_enc])

            # Clustering
            clusters = self.clustering_model.fit_predict(features)

            # Grouper par cluster
            groups = {}
            for idx, cluster_id in enumerate(clusters):
                if cluster_id not in groups:
                    groups[cluster_id] = []
                groups[cluster_id].append(memories[idx])

            # Consolider chaque cluster
            consolidated = []
            for cluster_id, group in groups.items():
                if cluster_id == -1:  # Noise points
                    consolidated.extend(group)
                elif len(group) > 1:
                    # Fusionner le cluster
                    merged = min(group, key=lambda x: x["age"]).copy()
                    merged["cluster_size"] = len(group)
                    merged["ml_consolidated"] = True
                    consolidated.append(merged)
                else:
                    consolidated.append(group[0])

            return consolidated

        except Exception as e:
            logger.error(f"ML consolidation failed: {e}, falling back to hash")
            return await self._consolidate_with_hash(memories)

    async def _archive_old_memories(self) -> int:
        """Archive les vieux souvenirs"""
        # En production, déplacer vers stockage froid (S3, etc.)
        # Pour l'instant, juste compter
        archived = 0

        if self.memory_federation and hasattr(self.memory_federation, "archive_old"):
            try:
                archived = await self.memory_federation.archive_old(
                    older_than=time.time() - self.long_term_threshold * 24  # 24h
                )
            except:
                pass

        return archived

    async def _prune_redundant(self) -> int:
        """Nettoie les duplicates"""
        pruned = 0

        if self.memory_federation:
            # Nettoyer les hash vus
            if hasattr(self.memory_federation, "seen_hashes"):
                before = len(self.memory_federation.seen_hashes)
                if before > 10000:
                    # Garder les 5000 plus récents
                    self.memory_federation.seen_hashes = set(list(self.memory_federation.seen_hashes)[-5000:])
                    pruned = before - 5000

            # Nettoyer les vieilles métadonnées
            if hasattr(self.memory_federation, "prune_metadata"):
                try:
                    pruned += await self.memory_federation.prune_metadata()
                except:
                    pass

        return pruned

    def _calculate_reward(self, result: Any) -> float:
        """Récompense pour RL basée sur l'efficacité de consolidation"""
        if not result:
            return 0.0

        consolidated = result.get("consolidated", 0)
        pruned = result.get("pruned", 0)

        # Récompense basée sur le ratio de consolidation
        if self.memories_processed > 0:
            consolidation_ratio = consolidated / self.memories_processed
            # Bon si on consolide 20-50% des souvenirs
            if 0.2 <= consolidation_ratio <= 0.5:
                reward = 5.0
            else:
                reward = 1.0
        else:
            reward = 0.0

        # Bonus pour le pruning efficace
        if pruned > 100:
            reward += 2.0

        return reward

    def _compress_old_memories(self):
        """Compresse les vieilles mémoires pour économiser RAM (Phase 2.3)"""
        if len(self.short_term) > 50:
            # Prendre les 25 plus vieux
            to_compress = []
            for _ in range(min(25, len(self.short_term))):
                if self.short_term:
                    to_compress.append(self.short_term.popleft())

            if to_compress:
                # Compresser en batch
                data = json.dumps(to_compress).encode("utf-8")
                compressed = zlib.compress(data, level=6)
                self.compressed_history.append(
                    {
                        "timestamp": time.time(),
                        "count": len(to_compress),
                        "data": compressed,
                        "size_ratio": len(compressed) / len(data),
                    }
                )

                # Limiter historique compressé
                if len(self.compressed_history) > 20:
                    self.compressed_history.pop(0)

                self.compression_count += 1
                logger.debug(f"Compressed {len(to_compress)} memories, ratio: {len(compressed) / len(data):.2f}")

    def _estimate_saved_memory(self) -> float:
        """Estime mémoire économisée en MB"""
        if not self.compressed_history:
            return 0.0
        total_ratio = sum(h.get("size_ratio", 1) for h in self.compressed_history)
        avg_ratio = total_ratio / len(self.compressed_history)
        return (1 - avg_ratio) * len(self.compressed_history) * 0.1

    def _get_memory_usage(self) -> float:
        """Calcule usage mémoire approximatif en MB"""
        total = sys.getsizeof(self.short_term)
        total += sum(sys.getsizeof(item["data"]) for item in self.compressed_history)
        return total / (1024 * 1024)
