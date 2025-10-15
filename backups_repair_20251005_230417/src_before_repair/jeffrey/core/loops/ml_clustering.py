"""
ML Clustering for memory consolidation
Phase 2.2 - Real clustering with embeddings (with numpy fix)
"""

import hashlib
import logging
import time
from collections import defaultdict
from typing import Any

# Import new AdaptiveMemoryClusterer as well
try:
    from ..ml.memory_clusterer import AdaptiveMemoryClusterer
except ImportError:
    AdaptiveMemoryClusterer = None

# Import numpy independently
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Import sklearn separately
try:
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# Import transformers separately
try:
    from sentence_transformers import SentenceTransformer

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)


class MemoryClusterer:
    """
    ML-based memory clustering with embeddings
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.embeddings_cache = {}
        self.cluster_centers = {}

        # Initialize model if available
        if HAS_TRANSFORMERS:
            try:
                self.model = SentenceTransformer(model_name)
                logger.info(f"Loaded embedding model: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load model {model_name}: {e}")
                self.model = None
        else:
            logger.info("sentence-transformers not available, using hash-based clustering")

    async def cluster_memories(self, memories: list[dict], eps: float = 0.3, min_samples: int = 2) -> list[list[dict]]:
        """
        Cluster memories using embeddings or fallback to hash-based
        """
        if not memories:
            return []

        # Get embeddings
        embeddings = await self._get_embeddings(memories)

        if embeddings is None or not HAS_SKLEARN:
            # Fallback to simple hash-based clustering
            logger.debug("Using hash-based clustering (ML not available)")
            return self._hash_based_clustering(memories)

        # Perform DBSCAN clustering
        try:
            clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
            clusters = clusterer.fit_predict(embeddings)

            # Group memories by cluster
            clustered_memories = defaultdict(list)
            for idx, cluster_id in enumerate(clusters):
                clustered_memories[cluster_id].append(memories[idx])

            # Convert to list of clusters
            result = list(clustered_memories.values())

            logger.info(f"Clustered {len(memories)} memories into {len(result)} clusters")
            return result

        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            return self._hash_based_clustering(memories)

    async def _get_embeddings(self, memories: list[dict]) -> Any | None:
        """
        Get embeddings for memories
        """
        if not self.model or not HAS_NUMPY:
            return None

        try:
            texts = []
            for memory in memories:
                # Extract text content
                if "text" in memory:
                    text = memory["text"]
                elif "content" in memory:
                    text = memory["content"]
                else:
                    text = str(memory.get("metadata", ""))

                texts.append(text[:512])  # Limit length

            # Generate embeddings
            embeddings = self.model.encode(texts, show_progress_bar=False)

            # Cache embeddings
            for i, memory in enumerate(memories):
                memory_hash = self._get_memory_hash(memory)
                self.embeddings_cache[memory_hash] = embeddings[i]

            # Limit cache size
            if len(self.embeddings_cache) > 10000:
                # Remove oldest entries
                keys = list(self.embeddings_cache.keys())
                for key in keys[:5000]:
                    del self.embeddings_cache[key]

            return embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            return None

    def _hash_based_clustering(self, memories: list[dict]) -> list[list[dict]]:
        """
        Fallback clustering based on content hashes
        """
        clusters = defaultdict(list)

        for memory in memories:
            # Generate hash-based cluster ID
            cluster_id = self._get_cluster_hash(memory)
            clusters[cluster_id].append(memory)

        return list(clusters.values())

    def _get_memory_hash(self, memory: dict) -> str:
        """Get unique hash for memory"""
        content = str(memory.get("text", "")) + str(memory.get("metadata", ""))
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_cluster_hash(self, memory: dict) -> str:
        """Get cluster ID based on content similarity (simple hash prefix)"""
        content = str(memory.get("text", ""))[:100]  # First 100 chars
        # Use first 4 chars of hash for clustering
        return hashlib.sha256(content.encode()).hexdigest()[:4]

    async def find_similar(self, query: str, memories: list[dict], top_k: int = 5) -> list[dict]:
        """
        Find similar memories to query
        """
        if not self.model or not HAS_NUMPY:
            # Fallback to keyword matching
            return self._keyword_search(query, memories, top_k)

        try:
            # Get query embedding
            query_embedding = self.model.encode([query], show_progress_bar=False)[0]

            # Get memory embeddings
            embeddings = await self._get_embeddings(memories)
            if embeddings is None:
                return self._keyword_search(query, memories, top_k)

            # Calculate similarities using numpy
            similarities = []
            for i, embedding in enumerate(embeddings):
                # Cosine similarity
                sim = np.dot(query_embedding, embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(embedding))
                similarities.append((sim, memories[i]))

            # Sort by similarity
            similarities.sort(reverse=True, key=lambda x: x[0])

            # Return top k
            return [mem for _, mem in similarities[:top_k]]

        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return self._keyword_search(query, memories, top_k)

    def _keyword_search(self, query: str, memories: list[dict], top_k: int) -> list[dict]:
        """
        Simple keyword-based search fallback
        """
        query_words = set(query.lower().split())
        scores = []

        for memory in memories:
            text = str(memory.get("text", "")) + str(memory.get("metadata", ""))
            text_words = set(text.lower().split())

            # Calculate overlap
            overlap = len(query_words & text_words)
            scores.append((overlap, memory))

        # Sort by score
        scores.sort(reverse=True, key=lambda x: x[0])

        # Return top k
        return [mem for _, mem in scores[:top_k]]

    def get_cluster_summary(self, cluster: list[dict]) -> dict[str, Any]:
        """
        Generate summary for a cluster of memories
        """
        if not cluster:
            return {}

        # Find most representative memory (longest for now)
        representative = max(cluster, key=lambda m: len(str(m.get("text", ""))))

        # Extract common themes (simple word frequency)
        word_counts = defaultdict(int)
        for memory in cluster:
            text = str(memory.get("text", ""))
            for word in text.lower().split():
                if len(word) > 4:  # Skip short words
                    word_counts[word] += 1

        # Top themes
        themes = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "size": len(cluster),
            "representative": representative,
            "themes": [word for word, _ in themes],
            "timestamp_range": {
                "earliest": min(m.get("timestamp", time.time()) for m in cluster),
                "latest": max(m.get("timestamp", time.time()) for m in cluster),
            },
        }
