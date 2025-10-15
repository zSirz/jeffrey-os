"""
Déduplication sémantique ML - OPTIONNEL, désactivé par défaut
Active avec : memory.semantic_dedup: true
"""

import asyncio
import logging

import numpy as np

logger = logging.getLogger(__name__)


class SemanticDeduplicator:
    """Dédup ML avec lazy loading et normalisation"""

    def __init__(self, threshold: float = 0.95):  # Seuil ÉLEVÉ par défaut
        self.threshold = threshold
        self.model = None
        self.index = None
        self.embeddings = []
        self.texts = []
        self._lock = asyncio.Lock()  # Thread safety
        self._initialized = False

    async def _lazy_init(self):
        """Charge le modèle SEULEMENT si utilisé"""
        async with self._lock:
            if self._initialized:
                return

            try:
                logger.info("Loading semantic model (this may take 30s)...")

                # Import seulement si nécessaire
                import faiss
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")

                # Index FAISS avec normalisation
                self.index = faiss.IndexFlatIP(384)  # Inner product pour cosine

                self._initialized = True
                logger.info("Semantic model ready")

            except ImportError:
                logger.warning("Semantic dedup unavailable (install sentence-transformers faiss-cpu)")
                self._initialized = False
            except Exception as e:
                logger.error(f"Semantic model failed: {e}")
                self._initialized = False

    async def is_duplicate(self, text: str, context: str = "") -> tuple[bool, str | None]:
        """Check semantic similarity (async safe)"""

        # Si pas initialisé ou échec, skip
        if not self._initialized:
            await self._lazy_init()

        if not self._initialized or not self.model:
            return False, None

        try:
            # Normalisation importante
            full_text = f"{context} {text}" if context else text

            # Encoder avec normalisation
            embedding = self.model.encode(
                [full_text],
                normalize_embeddings=True,  # CRITIQUE pour cosine
            )[0]

            # Chercher similaires
            async with self._lock:
                if len(self.embeddings) > 0:
                    # Recherche par similarité cosine
                    embeddings_array = np.array(self.embeddings)

                    # Normaliser les embeddings stockés aussi
                    norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
                    embeddings_array = embeddings_array / norms

                    # Calcul similarité
                    similarities = np.dot(embeddings_array, embedding)
                    max_sim_idx = np.argmax(similarities)
                    max_sim = similarities[max_sim_idx]

                    if max_sim > self.threshold:
                        return True, self.texts[max_sim_idx][:50]

                # Ajouter si nouveau
                self.embeddings.append(embedding)
                self.texts.append(text[:100])

                # Pruning
                if len(self.embeddings) > 500:
                    self.embeddings = self.embeddings[-250:]
                    self.texts = self.texts[-250:]

            return False, None

        except Exception as e:
            logger.debug(f"Semantic check failed: {e}")
            return False, None
