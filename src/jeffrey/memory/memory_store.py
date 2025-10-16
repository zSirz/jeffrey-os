"""
Memory Store Interface - À implémenter

Ce module définit l'interface pour le stockage des mémoires de Jeffrey OS.
Il inclut une implémentation simple en mémoire pour les tests et le développement.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from datetime import datetime
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class MemoryStore(ABC):
    """Interface pour le stockage des mémoires"""

    @abstractmethod
    async def store(self, memory: Dict) -> str:
        """
        Stocker une nouvelle mémoire

        Args:
            memory: Dictionnaire contenant les données de la mémoire

        Returns:
            ID unique de la mémoire stockée
        """
        pass

    @abstractmethod
    async def retrieve(self, memory_id: str) -> Optional[Dict]:
        """
        Récupérer une mémoire par ID

        Args:
            memory_id: ID unique de la mémoire

        Returns:
            Dictionnaire de la mémoire ou None si non trouvée
        """
        pass

    @abstractmethod
    async def search(self, query: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Rechercher des mémoires

        Args:
            query: Terme de recherche
            limit: Nombre maximum de résultats
            offset: Décalage pour pagination

        Returns:
            Liste de mémoires correspondantes
        """
        pass

    @abstractmethod
    async def get_recent(self, since: datetime, limit: int = 100) -> List[Dict]:
        """
        Récupérer les mémoires récentes

        Args:
            since: Date limite (incluse)
            limit: Nombre maximum de résultats

        Returns:
            Liste de mémoires récentes
        """
        pass

    @abstractmethod
    async def get_stats(self) -> Dict:
        """
        Obtenir les statistiques du store

        Returns:
            Dictionnaire avec les métriques du store
        """
        pass


class InMemoryStore(MemoryStore):
    """
    Implémentation simple en mémoire pour tests et développement

    Cette implémentation stocke tout en RAM et est donc volatile.
    À utiliser uniquement pour les tests et le développement local.
    """

    def __init__(self):
        self.memories = {}
        self.counter = 0
        self.created_at = datetime.now()
        logger.info("InMemoryStore initialized")

    async def store(self, memory: Dict) -> str:
        """Stocker une mémoire en RAM"""
        self.counter += 1
        memory_id = f"mem_{self.counter:06d}"

        # Enrichir la mémoire avec métadonnées
        enriched_memory = {
            **memory,
            "id": memory_id,
            "stored_at": datetime.now().isoformat(),
            "store_type": "in_memory"
        }

        self.memories[memory_id] = enriched_memory
        logger.debug(f"Stored memory {memory_id}")
        return memory_id

    async def retrieve(self, memory_id: str) -> Optional[Dict]:
        """Récupérer une mémoire par ID"""
        memory = self.memories.get(memory_id)
        if memory:
            logger.debug(f"Retrieved memory {memory_id}")
        else:
            logger.debug(f"Memory {memory_id} not found")
        return memory

    async def search(self, query: str, limit: int = 100, offset: int = 0) -> List[Dict]:
        """
        Recherche simple par mot-clé dans le contenu des mémoires

        Recherche dans les champs: text, content, description, emotion
        """
        if not query.strip():
            # Retourner toutes les mémoires si query vide
            all_memories = list(self.memories.values())
            return all_memories[offset:offset + limit]

        results = []
        query_lower = query.lower()

        for memory in self.memories.values():
            # Recherche dans plusieurs champs
            searchable_text = " ".join([
                str(memory.get('text', '')),
                str(memory.get('content', '')),
                str(memory.get('description', '')),
                str(memory.get('emotion', '')),
                str(memory.get('tags', []))
            ]).lower()

            if query_lower in searchable_text:
                results.append(memory)

        # Trier par date de stockage (plus récent en premier)
        results.sort(key=lambda x: x.get('stored_at', ''), reverse=True)

        # Appliquer pagination
        paginated_results = results[offset:offset + limit]
        logger.debug(f"Search '{query}' returned {len(paginated_results)} results")
        return paginated_results

    async def get_recent(self, since: datetime, limit: int = 100) -> List[Dict]:
        """Récupérer les mémoires récentes"""
        recent_memories = []
        since_iso = since.isoformat()

        for memory in self.memories.values():
            stored_at = memory.get('stored_at', '')
            if stored_at >= since_iso:
                recent_memories.append(memory)

        # Trier par date de stockage (plus récent en premier)
        recent_memories.sort(key=lambda x: x.get('stored_at', ''), reverse=True)

        # Limiter les résultats
        limited_results = recent_memories[:limit]
        logger.debug(f"Retrieved {len(limited_results)} recent memories since {since_iso}")
        return limited_results

    async def get_stats(self) -> Dict:
        """Statistiques du store en mémoire"""
        now = datetime.now()
        uptime = (now - self.created_at).total_seconds()

        # Analyser les émotions
        emotions = {}
        for memory in self.memories.values():
            emotion = memory.get('emotion', 'unknown')
            emotions[emotion] = emotions.get(emotion, 0) + 1

        # Calculer la taille moyenne des textes
        text_lengths = []
        for memory in self.memories.values():
            text = memory.get('text', '')
            if text:
                text_lengths.append(len(text))

        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0

        return {
            "store_type": "in_memory",
            "total_memories": len(self.memories),
            "uptime_seconds": round(uptime, 2),
            "memory_usage_bytes": self._estimate_memory_usage(),
            "emotions_distribution": emotions,
            "avg_text_length": round(avg_text_length, 2),
            "oldest_memory": min(
                (m.get('stored_at', '') for m in self.memories.values()),
                default=None
            ),
            "newest_memory": max(
                (m.get('stored_at', '') for m in self.memories.values()),
                default=None
            )
        }

    def _estimate_memory_usage(self) -> int:
        """Estimation simple de l'usage mémoire en bytes"""
        try:
            # Sérialiser toutes les mémoires pour estimer la taille
            serialized = json.dumps(self.memories)
            return len(serialized.encode('utf-8'))
        except Exception:
            # Fallback en cas d'erreur de sérialisation
            return len(self.memories) * 1000  # 1KB par mémoire (estimation)

    async def clear(self) -> int:
        """Effacer toutes les mémoires (pour tests)"""
        count = len(self.memories)
        self.memories.clear()
        self.counter = 0
        logger.info(f"Cleared {count} memories")
        return count

    def __len__(self) -> int:
        """Nombre de mémoires stockées"""
        return len(self.memories)

    def __repr__(self) -> str:
        return f"InMemoryStore(memories={len(self.memories)})"


# Adaptateur pour compatibilité avec l'ancien système
class SyncMemoryAdapter:
    """
    Adaptateur pour utiliser un MemoryStore async de façon synchrone

    Permet de migrer progressivement vers l'API async sans casser
    l'existant comme DreamEngine qui appelle de façon synchrone.
    """

    def __init__(self, async_store: MemoryStore):
        self.async_store = async_store

    def search(self, query: str = "", limit: int = 100, offset: int = 0) -> List[Dict]:
        """Version synchrone de search"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(
                self.async_store.search(query, limit, offset)
            )
        except RuntimeError:
            # Pas de loop actif, en créer un nouveau
            return asyncio.run(self.async_store.search(query, limit, offset))

    def store(self, memory: Dict) -> str:
        """Version synchrone de store"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.async_store.store(memory))
        except RuntimeError:
            return asyncio.run(self.async_store.store(memory))

    def get_stats(self) -> Dict:
        """Version synchrone de get_stats"""
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.async_store.get_stats())
        except RuntimeError:
            return asyncio.run(self.async_store.get_stats())


# Factory pour créer les stores
def create_memory_store(store_type: str = "in_memory") -> MemoryStore:
    """
    Factory pour créer des instances de MemoryStore

    Args:
        store_type: Type de store ("in_memory", "postgres", "mongodb", etc.)

    Returns:
        Instance de MemoryStore
    """
    if store_type == "in_memory":
        return InMemoryStore()
    elif store_type == "postgres":
        # TODO: Implémenter PostgreSQLStore
        raise NotImplementedError("PostgreSQL store not implemented yet")
    elif store_type == "mongodb":
        # TODO: Implémenter MongoDBStore
        raise NotImplementedError("MongoDB store not implemented yet")
    else:
        raise ValueError(f"Unknown store type: {store_type}")


# Instance globale pour testing/development
_default_store = None


def get_default_store() -> MemoryStore:
    """Obtenir l'instance par défaut du store"""
    global _default_store
    if _default_store is None:
        _default_store = create_memory_store("in_memory")
    return _default_store


def get_sync_adapter() -> SyncMemoryAdapter:
    """Obtenir un adaptateur synchrone pour l'usage legacy"""
    return SyncMemoryAdapter(get_default_store())