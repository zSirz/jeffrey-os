import logging
import os
from typing import Dict, List
import asyncio
from functools import lru_cache

logger = logging.getLogger(__name__)

# Feature flag check
ENABLE_REAL_ML = os.getenv("ENABLE_REAL_ML", "false").lower() == "true"

class RealEmotionClassifier:
    """Real ML-based emotion classifier using HuggingFace"""

    def __init__(self):
        self.classifier = None
        self._lock = asyncio.Lock()

    @lru_cache(maxsize=1)
    def _get_classifier(self):
        """Lazy load the model (singleton pattern)"""
        if self.classifier is None:
            logger.info("Loading emotion classification model...")
            try:
                from transformers import pipeline
                self.classifier = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=-1  # CPU, use 0 for GPU
                )
                logger.info("Model loaded successfully")
            except ImportError as e:
                logger.error(f"transformers not available: {e}")
                raise
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                raise
        return self.classifier

    async def detect_emotion(self, text: str) -> Dict:
        """Detect emotion with real ML model"""
        if not ENABLE_REAL_ML:
            raise RuntimeError("Real ML is disabled via feature flag")

        async with self._lock:
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._classify_text,
                text
            )

            # Model returns list of all emotions with scores
            # Convert to our format
            top_emotion = result[0]
            all_scores = {r['label'].lower(): r['score'] for r in result}

            return {
                'emotion': top_emotion['label'].lower(),
                'confidence': top_emotion['score'],
                'all_scores': all_scores,
                'method': 'distilroberta_ml'
            }

    def _classify_text(self, text: str) -> List[Dict]:
        """Internal classification method"""
        classifier = self._get_classifier()
        # Get all scores, not just top 1
        return classifier(text, top_k=None)

# Singleton instance - but only if enabled
if ENABLE_REAL_ML:
    try:
        emotion_classifier = RealEmotionClassifier()
    except Exception as e:
        logger.warning(f"Failed to initialize ML classifier: {e}")
        emotion_classifier = None
else:
    emotion_classifier = None