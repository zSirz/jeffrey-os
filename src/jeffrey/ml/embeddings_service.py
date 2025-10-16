import logging
from typing import List, Optional, Union
import numpy as np
import os
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

class EmbeddingsService:
    """Service for generating semantic embeddings from text"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize with a lightweight but effective model
        all-MiniLM-L6-v2: 384 dimensions, fast, good for semantic similarity
        """
        self.model_name = model_name
        self.model = None
        self._lock = asyncio.Lock()
        self.enabled = os.getenv("ENABLE_EMBEDDINGS", "false").lower() == "true"

    @lru_cache(maxsize=1)
    def _get_model(self):
        """Lazy load the model (singleton pattern)"""
        if not self.enabled:
            logger.warning("Embeddings disabled via ENABLE_EMBEDDINGS flag")
            return None

        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.model_name}")
                self.model = SentenceTransformer(self.model_name)
                # Warmup
                _ = self.model.encode("warmup", convert_to_numpy=True)
                logger.info(f"Model loaded: {self.model.get_sentence_embedding_dimension()} dimensions")
            except ImportError as e:
                logger.error(f"sentence-transformers not available: {e}")
                return None
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                return None
        return self.model

    async def generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for a single text"""
        if not self.enabled or not text.strip():
            return None

        async with self._lock:
            loop = asyncio.get_event_loop()
            try:
                embedding = await loop.run_in_executor(
                    None,
                    self._generate_sync,
                    text
                )
                return embedding
            except Exception as e:
                logger.error(f"Failed to generate embedding: {e}")
                return None

    async def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for multiple texts efficiently"""
        if not self.enabled:
            return [None] * len(texts)

        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not valid_texts:
            return [None] * len(texts)

        async with self._lock:
            loop = asyncio.get_event_loop()
            try:
                embeddings = await loop.run_in_executor(
                    None,
                    self._generate_batch_sync,
                    [text for _, text in valid_texts]
                )

                # Map back to original indices
                result = [None] * len(texts)
                for (original_idx, _), embedding in zip(valid_texts, embeddings):
                    result[original_idx] = embedding

                return result
            except Exception as e:
                logger.error(f"Failed to generate batch embeddings: {e}")
                return [None] * len(texts)

    def _generate_sync(self, text: str) -> Optional[np.ndarray]:
        """Synchronous embedding generation"""
        model = self._get_model()
        if model is None:
            return None

        try:
            # Normalize for better cosine similarity
            embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return None

    def _generate_batch_sync(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """Batch processing for efficiency"""
        model = self._get_model()
        if model is None:
            return [None] * len(texts)

        try:
            embeddings = model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                batch_size=32,
                show_progress_bar=False
            )
            return [emb.astype(np.float32) for emb in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [None] * len(texts)

# Global service instance
embeddings_service = EmbeddingsService()