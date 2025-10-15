"""
Clustering léger pour regroupement thématique des souvenirs.
Utilise MiniBatchKMeans si scikit-learn disponible, sinon no-op.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ClusterEngine:
    """
    Moteur de clustering pour découverte thématique.

    Features:
    - Auto-détection de scikit-learn (graceful degradation)
    - Nombre de clusters adaptatif: k = max(k_min, min(k_max, sqrt(n)))
    - Thèmes approximatifs basés sur TF-IDF
    - Best-effort (ne bloque jamais les opérations normales)
    - Support du français (stop_words=None)
    """

    def __init__(self, k_min: int = 4, k_max: int = 12):
        """
        Args:
            k_min: Nombre minimum de clusters
            k_max: Nombre maximum de clusters
        """
        self.k_min = k_min
        self.k_max = k_max
        self.enabled = False

        # Tenter d'importer sklearn
        try:
            from sklearn.cluster import MiniBatchKMeans  # noqa
            from sklearn.feature_extraction.text import TfidfVectorizer  # noqa

            self.enabled = True
            logger.info("[ClusterEngine] Enabled: scikit-learn detected")
        except ImportError as e:
            logger.info(f"[ClusterEngine] Disabled: {e}")

    def fit_user(self, memories: list[dict[str, Any]]) -> tuple[dict[str, int], dict[int, str]]:
        """
        Clustering des souvenirs d'un utilisateur.

        Args:
            memories: Liste de souvenirs (doit contenir 'id', 'content', 'tags')

        Returns:
            (mem_id -> cluster_id, cluster_id -> theme)
            Si disabled ou échec: ({}, {})
        """
        # Fallback si disabled ou trop peu de données
        if not self.enabled or len(memories) < 4:
            return {}, {}

        try:
            from sklearn.cluster import MiniBatchKMeans
            from sklearn.feature_extraction.text import TfidfVectorizer

            # Préparer les textes
            texts = []
            ids = []
            for m in memories:
                mem_id = m["id"]
                content = m.get("content", "")
                tags = m.get("tags", []) or []
                text = f'{content} {" ".join(tags)}'
                texts.append(text)
                ids.append(mem_id)

            # TF-IDF vectorization (support français)
            vectorizer = TfidfVectorizer(
                max_features=4096,
                stop_words=None,  # Support du français
                ngram_range=(1, 2),
            )
            X = vectorizer.fit_transform(texts)

            # Nombre de clusters adaptatif
            n = len(memories)
            k = max(self.k_min, min(self.k_max, int(n**0.5)))

            # Clustering avec seed déterministe
            kmeans = MiniBatchKMeans(
                n_clusters=k,
                random_state=42,  # Déterminisme
                batch_size=256,
                max_iter=100,
            )
            labels = kmeans.fit_predict(X)

            # Créer mapping mem_id -> cluster_id
            mapping = {ids[i]: int(labels[i]) for i in range(len(ids))}

            # Générer thèmes approximatifs
            themes = {}
            feature_names = vectorizer.get_feature_names_out()

            for cid in range(k):
                # Indices des mémoires dans ce cluster
                cluster_indices = [i for i, label in enumerate(labels) if label == cid]

                if not cluster_indices:
                    themes[cid] = f"cluster_{cid}"
                    continue

                # Prendre les top mots du centroid
                centroid = kmeans.cluster_centers_[cid]
                top_indices = centroid.argsort()[-3:][::-1]  # Top 3 mots
                top_words = [feature_names[i] for i in top_indices]

                # Thème = concaténation des top mots
                themes[cid] = " ".join(top_words)

            logger.info(f"[ClusterEngine] Created {k} clusters for {n} memories")
            return mapping, themes

        except Exception as e:
            logger.warning(f"[ClusterEngine] fit_user failed: {e}")
            return {}, {}
