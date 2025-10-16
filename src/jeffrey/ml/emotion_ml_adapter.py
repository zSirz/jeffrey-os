"""
Adapter ML pour détection d'émotions - Production Ready
Intègre toutes les bonnes pratiques : async, thread-safety, config, monitoring
"""
import asyncio
import logging
import os
import threading
import time
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from collections import OrderedDict
from typing import Optional, Tuple

# Monitoring (optionnel si prometheus installé)
try:
    from prometheus_client import Counter, Gauge, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from jeffrey.core.emotion_backend import ProtoEmotionDetector, RegexEmotionDetector

logger = logging.getLogger(__name__)

class EmotionMLAdapter:
    """
    Adapter production-ready pour ML emotion detection.

    Features:
    - Thread-safe lazy loading avec double-check locking
    - Async/await avec ThreadPoolExecutor pour ne pas bloquer
    - Configuration via variables d'environnement
    - Fallback automatique vers regex si ML fail
    - Monitoring avec métriques Prometheus (si disponible)
    - Logging structuré JSON pour observabilité
    """

    # Singleton avec thread safety
    _instance = None
    _instance_lock = threading.Lock()
    _init_lock = threading.Lock()

    def __new__(cls):
        """Singleton pattern thread-safe"""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialisation avec configuration flexible"""
        # Éviter réinitialisation du singleton
        if hasattr(self, '_initialized_singleton'):
            return

        # Configuration via env vars
        self.use_ml = os.getenv("JEFFREY_EMO_ML", "true").lower() == "true"
        self.model_name = os.getenv("EMO_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
        self.data_dir = Path(os.getenv("EMO_DATA_DIR", "data"))
        self.warmup_enabled = os.getenv("EMO_WARMUP", "true").lower() == "true"
        self.warmup_text = os.getenv("EMO_WARMUP_TEXT", "test")
        self.timeout_ms = float(os.getenv("EMO_TIMEOUT_MS", "200.0"))  # GPT OPTIMIZATION: Reduced from 250ms
        self.max_text_length = int(os.getenv("EMO_MAX_TEXT_LENGTH", "2000"))

        # GPT OPTIMIZATION: Cache configuration
        self.cache_enabled = os.getenv("EMO_CACHE_ENABLED", "true").lower() == "true"
        self.cache_size = int(os.getenv("EMO_CACHE_SIZE", "1000"))
        self.cache_ttl_sec = int(os.getenv("EMO_CACHE_TTL", "300"))  # 5 minutes

        # Thread pool pour opérations CPU-bound - GPT OPTIMIZATION: Increased workers
        self.executor = ThreadPoolExecutor(
            max_workers=int(os.getenv("EMO_THREADS", "8")),  # Increased from 4
            thread_name_prefix="emo_ml"
        )

        # GPT OPTIMIZATION: Concurrency semaphore for better control
        self.semaphore = asyncio.Semaphore(int(os.getenv("EMO_MAX_CONCURRENT", "16")))

        # GPT OPTIMIZATION: LRU Cache with TTL for predictions
        self._cache = OrderedDict()  # LRU cache
        self._cache_timestamps = {}  # TTL tracking
        self._cache_lock = threading.Lock()  # Thread-safe cache

        # Composants ML
        self._ml_detector: ProtoEmotionDetector | None = None
        self._regex_head: RegexEmotionDetector | None = None
        self._initialized = False

        # Stats internes - GPT OPTIMIZATION: Added cache metrics
        self._stats = {
            "total_predictions": 0,
            "ml_predictions": 0,
            "fallback_predictions": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_latency_ms": 0.0,
            "avg_latency_ms": 0.0,
            "failures": 0,
            "timeouts": 0,
            "concurrent_requests": 0,
            "max_concurrent_requests": 0
        }

        # Métriques Prometheus si disponible
        if PROMETHEUS_AVAILABLE:
            self._init_prometheus_metrics()

        self._initialized_singleton = True
        logger.info("✅ EmotionMLAdapter singleton initialized")

    def _init_prometheus_metrics(self):
        """Initialise les métriques Prometheus"""
        self._metrics = {
            "predictions_total": Counter(
                "jeffrey_emotion_ml_predictions_total",
                "Total emotion predictions",
                ["method", "emotion"]
            ),
            "latency_ms": Histogram(
                "jeffrey_emotion_ml_latency_ms",
                "Prediction latency in milliseconds",
                buckets=[10, 25, 50, 100, 250, 500, 1000]
            ),
            "failures_total": Counter(
                "jeffrey_emotion_ml_failures_total",
                "Total prediction failures",
                ["reason"]
            ),
            "confidence": Histogram(
                "jeffrey_emotion_ml_confidence",
                "Prediction confidence scores",
                buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
            )
        }
        logger.info("📊 Prometheus metrics initialized")

    def _ensure_initialized(self):
        """Thread-safe lazy loading avec double-check locking"""
        if self._initialized:
            return

        with self._init_lock:
            # Double-check après acquisition du lock
            if self._initialized:
                return

            try:
                start = time.perf_counter()
                logger.info("🧠 Initializing ML emotion detection components...")

                if self.use_ml:
                    # Charger le détecteur ML existant
                    self._ml_detector = ProtoEmotionDetector()

                    # Warmup optionnel
                    if self.warmup_enabled:
                        try:
                            logger.debug("Warming up ML model...")
                            _ = self._ml_detector.predict_label(self.warmup_text)
                        except Exception as e:
                            logger.warning(f"Warmup failed (non-critical): {e}")

                    logger.info("✅ ML components loaded successfully")

                # Toujours charger regex comme fallback
                self._regex_head = RegexEmotionDetector()

                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(f"🎉 Emotion system initialized in {elapsed_ms:.1f}ms")
                self._initialized = True

            except Exception as e:
                logger.error(f"❌ Failed to initialize ML components: {e}", exc_info=True)
                # Fallback sur regex uniquement
                self.use_ml = False
                self._regex_head = RegexEmotionDetector()
                self._initialized = True
                raise RuntimeError(f"ML initialization failed, using regex fallback: {e}")

    def _predict_sync(self, text: str) -> tuple[str, float, dict[str, float]]:
        """Prédiction synchrone (pour ThreadPoolExecutor)"""
        if self.use_ml and self._ml_detector:
            # ML prediction avec le système existant
            emotion = self._ml_detector.predict_label(text)
            all_scores, _ = self._ml_detector.predict_proba(text)
            confidence = all_scores.get(emotion, 0.0)
            return emotion, float(confidence), all_scores
        else:
            # Regex fallback
            emotion = self._regex_head.predict_label(text)
            all_scores = self._regex_head.predict_proba(text)
            confidence = all_scores.get(emotion, 0.8)
            return emotion, confidence, all_scores

    def _validate_input(self, text: str) -> dict | None:
        """Valide l'input et retourne une erreur si invalide"""
        if not text:
            return {
                "success": False,
                "error": "Empty text",
                "emotion": "neutral",
                "confidence": 0.0,
                "method": "validation_failed"
            }

        if not isinstance(text, str):
            return {
                "success": False,
                "error": f"Invalid type: {type(text).__name__}",
                "emotion": "neutral",
                "confidence": 0.0,
                "method": "validation_failed"
            }

        if len(text) > self.max_text_length:
            logger.warning(f"Text truncated from {len(text)} to {self.max_text_length} chars")
            text = text[:self.max_text_length]

        return None  # Input valide

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text - GPT OPTIMIZATION"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]

    def _get_from_cache(self, cache_key: str) -> Optional[Tuple[str, float, dict]]:
        """Get cached result if valid - GPT OPTIMIZATION"""
        if not self.cache_enabled:
            return None

        with self._cache_lock:
            if cache_key not in self._cache:
                return None

            # Check TTL
            timestamp = self._cache_timestamps.get(cache_key, 0)
            if time.time() - timestamp > self.cache_ttl_sec:
                # Expired, remove
                del self._cache[cache_key]
                del self._cache_timestamps[cache_key]
                return None

            # Move to end (LRU)
            result = self._cache[cache_key]
            self._cache.move_to_end(cache_key)
            self._stats["cache_hits"] += 1
            return result

    def _put_in_cache(self, cache_key: str, result: Tuple[str, float, dict]) -> None:
        """Store result in cache - GPT OPTIMIZATION"""
        if not self.cache_enabled:
            return

        with self._cache_lock:
            # Remove oldest if cache full
            while len(self._cache) >= self.cache_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                del self._cache_timestamps[oldest_key]

            # Store new result
            self._cache[cache_key] = result
            self._cache_timestamps[cache_key] = time.time()
            self._stats["cache_misses"] += 1

    async def detect_emotion(self, text: str) -> dict:
        """
        Détecte l'émotion de manière asynchrone avec cache et semaphore - GPT OPTIMIZATION

        Args:
            text: Texte à analyser

        Returns:
            Dict avec emotion, confidence, all_scores, method, latency_ms, success
        """
        # Validation
        validation_error = self._validate_input(text)
        if validation_error:
            return validation_error

        # Ensure initialized
        self._ensure_initialized()

        # GPT OPTIMIZATION: Check cache first
        text_cleaned = text.strip()
        cache_key = self._get_cache_key(text_cleaned)
        cached_result = self._get_from_cache(cache_key)

        if cached_result:
            emotion, confidence, all_scores = cached_result
            return {
                "success": True,
                "emotion": emotion,
                "confidence": float(confidence),
                "all_scores": all_scores,
                "method": "cache_hit",
                "latency_ms": 0.1,  # Minimal cache latency
                "text_length": len(text)
            }

        # GPT OPTIMIZATION: Semaphore-controlled processing
        async with self.semaphore:
            # Track concurrent requests
            self._stats["concurrent_requests"] += 1
            self._stats["max_concurrent_requests"] = max(
                self._stats["max_concurrent_requests"],
                self._stats["concurrent_requests"]
            )

            try:
                start = time.perf_counter()
                loop = asyncio.get_running_loop()

                # Prédiction avec timeout
                emotion, confidence, all_scores = await asyncio.wait_for(
                    loop.run_in_executor(
                        self.executor,
                        self._predict_sync,
                        text_cleaned
                    ),
                    timeout=self.timeout_ms / 1000.0
                )

                method = "ml_linear_head" if self.use_ml else "regex_fallback"
                success = True

                # GPT OPTIMIZATION: Store in cache for future requests
                self._put_in_cache(cache_key, (emotion, confidence, all_scores))

            except TimeoutError:
                logger.warning(f"ML prediction timeout after {self.timeout_ms}ms, using fallback")
                self._stats["timeouts"] += 1

                # Fallback to regex
                emotion = self._regex_head.predict_label(text_cleaned)
                all_scores = self._regex_head.predict_proba(text_cleaned)
                confidence = all_scores.get(emotion, 0.8)
                method = "regex_timeout_fallback"
                success = True

                if PROMETHEUS_AVAILABLE:
                    self._metrics["failures_total"].labels(reason="timeout").inc()

            except Exception as e:
                logger.error(f"Prediction failed: {e}", exc_info=True)
                self._stats["failures"] += 1

                # Fallback to regex
                try:
                    emotion = self._regex_head.predict_label(text_cleaned)
                    all_scores = self._regex_head.predict_proba(text_cleaned)
                    confidence = all_scores.get(emotion, 0.8)
                    method = "regex_error_fallback"
                    success = True
                except Exception as e2:
                    logger.error(f"Regex fallback also failed: {e2}")
                    return {
                        "success": False,
                        "error": str(e2),
                        "emotion": "neutral",
                        "confidence": 0.0,
                        "all_scores": {},
                        "method": "all_failed",
                        "latency_ms": 0.0
                    }

                if PROMETHEUS_AVAILABLE:
                    self._metrics["failures_total"].labels(reason="exception").inc()

            finally:
                # GPT OPTIMIZATION: Decrement concurrent request counter
                self._stats["concurrent_requests"] = max(0, self._stats["concurrent_requests"] - 1)

            # Calcul latence
            elapsed_ms = (time.perf_counter() - start) * 1000

        # Update stats
        self._stats["total_predictions"] += 1
        if "ml" in method:
            self._stats["ml_predictions"] += 1
        else:
            self._stats["fallback_predictions"] += 1
        self._stats["total_latency_ms"] += elapsed_ms
        self._stats["avg_latency_ms"] = (
            self._stats["total_latency_ms"] / self._stats["total_predictions"]
        )

        # Prometheus metrics
        if PROMETHEUS_AVAILABLE:
            self._metrics["predictions_total"].labels(method=method, emotion=emotion).inc()
            self._metrics["latency_ms"].observe(elapsed_ms)
            self._metrics["confidence"].observe(confidence)

        # Logging structuré pour monitoring
        result = {
            "success": success,
            "emotion": emotion,
            "confidence": float(confidence),
            "all_scores": all_scores,
            "method": method,
            "latency_ms": elapsed_ms,
            "text_length": len(text)
        }

        # Log JSONL pour pipeline existant
        try:
            from jeffrey.utils.monitoring import log_prediction
            log_prediction(
                route="linear_head" if "ml" in method else "regex",
                primary=emotion,
                confidence=confidence,
                latency_ms=elapsed_ms,
                rule_applied="",
                text_preview=text[:120]
            )
        except ImportError:
            # Si le module monitoring n'existe pas encore
            pass

        logger.debug(
            "🎯 Emotion detected",
            extra={
                "emotion_detection": result,
                "structured": True
            }
        )

        return result

    def get_stats(self) -> dict:
        """Retourne les statistiques détaillées - GPT OPTIMIZATION with cache metrics"""
        total_cache_ops = self._stats["cache_hits"] + self._stats["cache_misses"]
        return {
            **self._stats,
            "ml_enabled": self.use_ml,
            "cache_enabled": self.cache_enabled,
            "cache_size": len(self._cache),
            "cache_max_size": self.cache_size,
            "ml_percentage": (
                self._stats["ml_predictions"] / max(1, self._stats["total_predictions"]) * 100
                if self._stats["total_predictions"] > 0 else 0
            ),
            "fallback_percentage": (
                self._stats["fallback_predictions"] / max(1, self._stats["total_predictions"]) * 100
                if self._stats["total_predictions"] > 0 else 0
            ),
            "cache_hit_rate": (
                self._stats["cache_hits"] / max(1, total_cache_ops) * 100
                if total_cache_ops > 0 else 0
            ),
            "failure_rate": (
                self._stats["failures"] / max(1, self._stats["total_predictions"]) * 100
                if self._stats["total_predictions"] > 0 else 0
            ),
            "timeout_rate": (
                self._stats["timeouts"] / max(1, self._stats["total_predictions"]) * 100
                if self._stats["total_predictions"] > 0 else 0
            ),
            "concurrency_utilization": (
                self._stats["max_concurrent_requests"] / int(os.getenv("EMO_MAX_CONCURRENT", "16")) * 100
            )
        }

    @classmethod
    async def get_instance(cls) -> "EmotionMLAdapter":
        """Factory method async pour obtenir l'instance singleton"""
        return cls()

    def __del__(self):
        """Cleanup du ThreadPoolExecutor"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
