from typing import List, Dict, Optional
from datetime import datetime
from collections import OrderedDict
import json
import logging
import numpy as np
from sqlalchemy import select, update, text
from jeffrey.memory.memory_store import MemoryStore
from jeffrey.models.memory import Memory, EmotionEvent
from jeffrey.db.session import AsyncSessionLocal, with_adaptive_retry
from jeffrey.ml.embeddings_service import embeddings_service

logger = logging.getLogger(__name__)

class HybridMemoryStore(MemoryStore):
    """Store hybride avec cache local LRU et persistence PostgreSQL"""

    def __init__(self, cache_size: int = 1000):
        self.cache = OrderedDict()
        self.cache_size = cache_size
        self.fallback_buffer = []
        self._force_fallback = False  # For testing purposes

    def _update_cache(self, key: str, value: Dict):
        """Met à jour le cache LRU"""
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value

        # Éviction LRU si dépassement
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    @with_adaptive_retry
    async def store(self, memory_data: Dict) -> str:
        """Store avec cache local et persistence DB"""
        # Check force fallback flag for testing
        if self._force_fallback:
            self.fallback_buffer.append({
                **memory_data,
                "timestamp": datetime.utcnow().isoformat()
            })
            return None

        try:
            # Coerce metadata → meta si nécessaire
            if "metadata" in memory_data and "meta" not in memory_data:
                memory_data["meta"] = memory_data.pop("metadata")

            # Generate embedding for text
            text = memory_data.get('text', '')
            embedding = None

            if text:
                try:
                    embedding = await embeddings_service.generate_embedding(text)
                    if embedding is not None:
                        logger.debug(f"Generated embedding of shape: {embedding.shape}")
                except Exception as e:
                    logger.warning(f"Failed to generate embedding: {e}")

            async with AsyncSessionLocal() as session:
                memory = Memory(
                    **{k: v for k, v in memory_data.items() if k != 'embedding'},
                    embedding=embedding.tolist() if embedding is not None else None
                )
                session.add(memory)
                await session.commit()

                memory_id = str(memory.id)

                # Mise à jour du cache
                self._update_cache(memory_id, {
                    'id': memory_id,
                    'text': memory.text,
                    'emotion': memory.emotion,
                    'confidence': memory.confidence,
                    'timestamp': memory.timestamp.isoformat(),
                    'meta': memory.meta or {}
                })

                return memory_id

        except Exception as e:
            logger.error(f"DB store failed, using fallback buffer: {e}")
            self.fallback_buffer.append(memory_data)

            # Stockage temporaire dans le cache uniquement
            temp_id = f"temp_{datetime.utcnow().timestamp()}"
            memory_data['id'] = temp_id
            self._update_cache(temp_id, memory_data)

            raise

    async def get_recent(self, since: datetime, limit: int = 100) -> List[Dict]:
        """Récupère les mémoires récentes avec fallback cache"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Memory)
                    .where(Memory.timestamp >= since)
                    .order_by(Memory.timestamp.desc())
                    .limit(limit)
                )
                memories = result.scalars().all()

                # Mise à jour du cache avec les résultats
                for m in memories:
                    self._update_cache(str(m.id), self._to_dict(m))

                return [self._to_dict(m) for m in memories]

        except Exception as e:
            logger.warning(f"DB unavailable, using cache fallback: {e}")

            # Fallback vers le cache local
            cached_memories = []
            for mem_id, mem_data in self.cache.items():
                try:
                    mem_time = datetime.fromisoformat(mem_data['timestamp'])
                    if mem_time >= since:
                        cached_memories.append(mem_data)
                except:
                    pass

            # Trier par timestamp et limiter
            cached_memories.sort(key=lambda x: x['timestamp'], reverse=True)
            return cached_memories[:limit]

    async def search(self, query: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Recherche avec fallback cache"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Memory)
                    .order_by(Memory.timestamp.desc())
                    .offset(offset)
                    .limit(limit)
                )
                memories = result.scalars().all()

                # Filtrage naïf pour l'instant
                query_lower = query.lower().strip()
                if query_lower:
                    memories = [m for m in memories if query_lower in (m.text or "").lower()]

                return [self._to_dict(m) for m in memories]

        except Exception:
            # Fallback cache search
            results = []
            query_lower = query.lower().strip()

            for mem_data in list(self.cache.values())[offset:]:
                if not query_lower or query_lower in mem_data.get('text', '').lower():
                    results.append(mem_data)
                    if len(results) >= limit:
                        break

            return results

    async def retrieve(self, memory_id: str) -> Optional[Dict]:
        """Récupère une mémoire par ID avec cache check"""
        # Check cache first
        if memory_id in self.cache:
            self.cache.move_to_end(memory_id)  # LRU update
            return self.cache[memory_id]

        # Fallback to DB
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(
                    select(Memory).where(Memory.id == memory_id)
                )
                memory = result.scalar_one_or_none()

                if memory:
                    mem_dict = self._to_dict(memory)
                    self._update_cache(memory_id, mem_dict)
                    return mem_dict
        except:
            pass

        return None

    async def sync_fallback_buffer(self):
        """Synchronise le buffer fallback vers la DB"""
        if not self.fallback_buffer:
            return 0

        synced = 0
        pending = list(self.fallback_buffer)
        self.fallback_buffer.clear()

        for memory_data in pending:
            try:
                await self.store(memory_data)
                synced += 1
            except Exception as e:
                logger.error(f"Failed to sync buffer entry: {e}")
                self.fallback_buffer.append(memory_data)

        return synced

    async def mark_processed(self, memory_ids: List[str]):
        """Marque les mémoires comme traitées par DreamEngine"""
        try:
            async with AsyncSessionLocal() as session:
                await session.execute(
                    update(Memory)
                    .where(Memory.id.in_(memory_ids))
                    .values(processed=True)
                )
                await session.commit()
        except Exception as e:
            logger.error(f"Failed to mark memories as processed: {e}")

    async def get_stats(self) -> Dict:
        """Statistiques du store hybride"""
        try:
            async with AsyncSessionLocal() as session:
                result = await session.execute(select(Memory))
                all_memories = result.scalars().all()

                total_memories = len(all_memories)

                # Analyser les émotions
                emotions = {}
                for memory in all_memories:
                    emotion = memory.emotion or 'unknown'
                    emotions[emotion] = emotions.get(emotion, 0) + 1

                return {
                    "store_type": "hybrid",
                    "total_memories": total_memories,
                    "cache_size": len(self.cache),
                    "fallback_buffer_size": len(self.fallback_buffer),
                    "emotions_distribution": emotions,
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "store_type": "hybrid_cache_only",
                "cache_size": len(self.cache),
                "fallback_buffer_size": len(self.fallback_buffer),
                "error": str(e)
            }

    async def semantic_search(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        threshold: float = 0.5
    ) -> List[Dict]:
        """Search memories using vector similarity with pgvector"""
        try:
            async with AsyncSessionLocal() as session:
                # Convert numpy array to list for SQL
                embedding_list = query_embedding.tolist()

                # Use pgvector's cosine distance operator with proper SQLAlchemy binding
                query_sql = """
                    SELECT
                        id,
                        text,
                        emotion,
                        confidence,
                        timestamp,
                        meta,
                        1 - (embedding <=> :query_embedding) as similarity
                    FROM memories
                    WHERE embedding IS NOT NULL
                        AND 1 - (embedding <=> :query_embedding) >= :threshold
                    ORDER BY embedding <=> :query_embedding
                    LIMIT :limit
                """

                result = await session.execute(
                    text(query_sql),
                    {
                        "query_embedding": str(embedding_list),
                        "threshold": threshold,
                        "limit": limit
                    }
                )

                rows = result.fetchall()

                return [
                    {
                        "id": str(row.id),
                        "text": row.text,
                        "emotion": row.emotion,
                        "confidence": row.confidence,
                        "timestamp": row.timestamp,
                        "meta": row.meta or {},
                        "similarity": float(row.similarity)
                    }
                    for row in rows
                ]

        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []

    def _to_dict(self, memory: Memory) -> Dict:
        return {
            'id': str(memory.id),
            'text': memory.text,
            'emotion': memory.emotion,
            'confidence': memory.confidence,
            'timestamp': memory.timestamp.isoformat(),
            'meta': memory.meta or {},
            'processed': memory.processed
        }