"""
Adaptive Memory Clusterer with auto-tuning
"""

import logging
import re
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

# Check for optional dependencies
HAS_SKLEARN = False
HAS_SENTENCE_TRANSFORMER = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score

    HAS_SKLEARN = True
except ImportError:
    logger.info("scikit-learn not available, using fallback clustering")

try:
    from sentence_transformers import SentenceTransformer

    HAS_SENTENCE_TRANSFORMER = True
except ImportError:
    logger.info("sentence-transformers not available, using hash-based embeddings")

try:
    from scipy.stats import entropy

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.info("scipy not available, using simple entropy calculation")


class AdaptiveMemoryClusterer:
    """Clustering avec auto-tune des paramètres"""

    def __init__(self):
        self.eps = 0.3  # Initial
        self.min_samples = 2  # Initial
        self.quality_history = deque(maxlen=100)
        self.learning_rate = 0.01

        # Initialize model if available
        if HAS_SENTENCE_TRANSFORMER:
            try:
                self.model = SentenceTransformer("all-MiniLM-L6-v2")
                logger.info("Sentence transformer loaded")
            except Exception as e:
                logger.error(f"Could not load sentence transformer: {e}")
                self.model = None
        else:
            self.model = None

    def _tune_parameters(self, X, current_quality: float):
        """Auto-ajuste eps et min_samples via gradient"""
        # Si qualité baisse, explorer
        if len(self.quality_history) > 10:
            avg_quality = sum(self.quality_history) / len(self.quality_history)

            if current_quality < avg_quality * 0.9:
                # Explorer : augmenter epsilon
                self.eps = min(1.0, self.eps * 1.1)
                logger.debug(f"Clustering explore: eps={self.eps:.3f}")

            elif current_quality > avg_quality * 1.1:
                # Exploiter : diminuer epsilon
                self.eps = max(0.05, self.eps * 0.95)
                logger.debug(f"Clustering exploit: eps={self.eps:.3f}")

        self.quality_history.append(current_quality)

    def cluster_memories(self, memories: list[dict]) -> list[dict]:
        """Cluster avec auto-tune et monitoring entropie"""
        if len(memories) < 3:
            return memories

        # Embeddings avec privacy check
        embeddings = self._get_safe_embeddings(memories)

        # Clustering DBSCAN
        if HAS_SKLEARN:
            clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            labels = clustering.fit_predict(embeddings)

            # Calculer qualité (silhouette score)
            if len(set(labels)) > 1 and -1 not in labels:
                quality = silhouette_score(embeddings, labels)
                self._tune_parameters(embeddings, quality)
            else:
                quality = 0.0

            # Calculer entropie des clusters
            entropy_val = self._calculate_entropy(labels)

            # Log si entropie faible (risque de bias)
            if entropy_val < 1.5:
                logger.warning(f"Low cluster entropy: {entropy_val:.2f} - risk of bias")

            return self._merge_clusters(memories, labels, entropy_val)
        else:
            # Fallback: hash-based clustering
            return self._hash_based_clustering(memories)

    def _get_safe_embeddings(self, memories: list[dict]) -> np.ndarray:
        """Embeddings avec privacy sanitization"""
        texts = []

        for mem in memories:
            text = mem.get("content", mem.get("text", ""))
            # PRIVACY : Masquer PII basique
            text = self._sanitize_pii(text)
            texts.append(text)

        if HAS_SENTENCE_TRANSFORMER and self.model:
            try:
                return self.model.encode(texts)
            except Exception as e:
                logger.error(f"Embedding failed: {e}")
                return self._hash_based_embeddings(texts)
        else:
            # Fallback: hash-based
            return self._hash_based_embeddings(texts)

    def _sanitize_pii(self, text: str) -> str:
        """Masque PII basique (emails, phones, etc)"""
        if not text:
            return ""

        # Email
        text = re.sub(r"\S+@\S+", "[EMAIL]", text)
        # Phone
        text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
        # SSN-like
        text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[ID]", text)
        # Credit card-like
        text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[CARD]", text)

        return text

    def _hash_based_embeddings(self, texts: list[str]) -> np.ndarray:
        """Fallback: embeddings basés sur hash"""
        import hashlib

        embeddings = []
        for text in texts:
            # Simple hash embedding (8D vector)
            h = hashlib.sha256(text.encode()).hexdigest()
            # Convert hash to vector
            vec = []
            for i in range(0, 64, 8):
                chunk = h[i : i + 8]
                val = int(chunk, 16) / (2**32)  # Normalize to [0,1]
                vec.append(val)
            embeddings.append(vec)

        return np.array(embeddings)

    def _hash_based_clustering(self, memories: list[dict]) -> list[dict]:
        """Fallback clustering sans sklearn"""
        # Simple clustering par similarité de hash
        clusters = {}

        for mem in memories:
            text = str(mem.get("content", mem.get("text", "")))
            # Hash court pour regrouper
            h = hashlib.sha256(text.encode()).hexdigest()[:8]

            if h not in clusters:
                clusters[h] = []
            clusters[h].append(mem)

        # Merger les clusters
        result = []
        for cluster_id, cluster_mems in clusters.items():
            if len(cluster_mems) > 1:
                # Prendre le plus récent
                merged = cluster_mems[0].copy()
                merged["cluster_size"] = len(cluster_mems)
                result.append(merged)
            else:
                result.extend(cluster_mems)

        return result

    def _calculate_entropy(self, labels: np.ndarray) -> float:
        """Calcule entropie de Shannon des clusters"""
        if HAS_SCIPY:
            from scipy.stats import entropy

            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            if len(unique) <= 1:
                return 0.0

            probs = counts / counts.sum()
            return entropy(probs, base=2)
        else:
            # Simple entropy calculation
            unique, counts = np.unique(labels[labels >= 0], return_counts=True)
            if len(unique) <= 1:
                return 0.0

            total = counts.sum()
            entropy_val = 0
            for count in counts:
                p = count / total
                if p > 0:
                    entropy_val -= p * np.log2(p)
            return entropy_val

    def _merge_clusters(self, memories: list[dict], labels: np.ndarray, entropy_val: float) -> list[dict]:
        """Fusionne les mémoires selon les clusters"""
        result = []
        clusters = {}

        for i, label in enumerate(labels):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(memories[i])

        for cluster_id, cluster_mems in clusters.items():
            if cluster_id == -1:  # Noise points
                result.extend(cluster_mems)
            elif len(cluster_mems) > 1:
                # Fusionner le cluster
                merged = cluster_mems[0].copy()
                merged["cluster_size"] = len(cluster_mems)
                merged["ml_consolidated"] = True
                merged["cluster_entropy"] = entropy_val
                result.append(merged)
            else:
                result.append(cluster_mems[0])

        return result
