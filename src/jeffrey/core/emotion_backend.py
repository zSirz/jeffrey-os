"""
Backend émotionnel unifié Jeffrey - Production Ready
Synthèse feedbacks : GPT (simplicité) + Gemini (robustesse) + Grok (monitoring)

Backends supportés :
- 'proto' : ProtoClassifier ML (F1 LOSO = 0.537)
- 'regex' : Fallback regex simple

Feature flag: JEFFREY_EMOTION_BACKEND (default: proto)
"""

from __future__ import annotations

import logging
import os
import re
import time

# Linear head support (priority over prototypes)
from pathlib import Path

import joblib
import numpy as np

from jeffrey.ml.encoder import create_default_encoder
from jeffrey.ml.proto import ProtoClassifier

# Monitoring support
from jeffrey.utils.monitoring import get_monitor

# Initialize logger
logger = logging.getLogger(__name__)

LINEAR_HEAD_PATH = Path("data/linear_head.joblib")
_linear_head_bundle = None


# ========== TIE-BREAK CONFIGURATION ==========
def _parse_bool(value: str) -> bool:
    """Parse boolean from string (supports 1/0, true/false, yes/no)."""
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# Feature flag to enable/disable tie-break rule entirely
TIEBREAK_ENABLED = _parse_bool(os.getenv("JEFFREY_TIEBREAK_ENABLED", "true"))

# Delta threshold for considering anger/frustration as "close"
TIEBREAK_DELTA = float(os.getenv("JEFFREY_TIEBREAK_DELTA", "0.05"))

# Maximum delta for extended tie-break (when emotion in top-2)
TIEBREAK_EXTENDED_DELTA = float(os.getenv("JEFFREY_TIEBREAK_EXTENDED_DELTA", "0.15"))

logger.info(
    f"🎯 Tie-break config: ENABLED={TIEBREAK_ENABLED}, DELTA={TIEBREAK_DELTA}, EXTENDED={TIEBREAK_EXTENDED_DELTA}"
)


def _apply_anger_tiebreak(
    text: str, scores: dict[str, float]
) -> tuple[str | None, dict[str, float] | None, str | None]:
    """
    Apply tie-break rule when anger and frustration are close.

    Can be disabled via JEFFREY_TIEBREAK_ENABLED=false env var.

    Returns:
        (new_primary, new_scores, rule_name) if override applied
        (None, None, None) otherwise
    """
    # Check if feature is enabled (GPT improvement)
    if not TIEBREAK_ENABLED:
        return (None, None, None)

    # Check if both emotions exist in scores
    if "anger" not in scores or "frustration" not in scores:
        return (None, None, None)

    s_anger = scores["anger"]
    s_frustration = scores["frustration"]

    # Check if close enough for tie-break
    gap = abs(s_anger - s_frustration)

    # Standard tie-break: very close scores
    if gap < TIEBREAK_DELTA:
        pass  # Continue to keyword check

    # Extended tie-break: if either is in top-2, allow larger gap (GPT improvement)
    elif gap < TIEBREAK_EXTENDED_DELTA:
        all_emotions = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top2_emotions = [emotion for emotion, _ in all_emotions[:2]]

        if "anger" not in top2_emotions and "frustration" not in top2_emotions:
            return (None, None, None)
    else:
        # Gap too large, no tie-break
        return (None, None, None)

    # Lowercase for matching
    text_lower = text.lower()

    # Strong anger keywords (EN) - expanded list
    en_anger_keywords = r'\b(angry|furious|rage|outraged|livid|fuming|enraged|so\s+angry|really\s+angry|very\s+angry)\b'
    # Strong anger keywords (FR)
    fr_anger_keywords = (
        r'\b(en\s*rage|furax|furieuse?|vénère|très\s*énervé[e]?|hors\s*de\s*moi|vraiment\s+en\s*colère)\b'
    )

    # Check for explicit anger expressions
    has_en_anger = re.search(en_anger_keywords, text_lower) is not None
    has_fr_anger = re.search(fr_anger_keywords, text_lower) is not None

    if has_en_anger or has_fr_anger:
        # Create new scores with anger boosted
        new_scores = dict(scores)

        # Boost anger slightly above the max of the two
        target_score = max(s_anger, s_frustration) + 0.02
        new_scores["anger"] = target_score

        # Renormalize to sum to 1.0
        total = sum(new_scores.values())
        if total > 0:
            new_scores = {k: v / total for k, v in new_scores.items()}

        # Determine rule type based on threshold used
        rule_type = "extended" if gap >= TIEBREAK_DELTA else "standard"
        return ("anger", new_scores, f"tiebreak_anger_keyword_{rule_type}")

    return (None, None, None)


logger = logging.getLogger(__name__)

# Core-8 emotions officielles
CORE_EMOTIONS: list[str] = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "frustration"]

# Métriques monitoring (compteurs globaux)
_METRICS = {
    "proto_success": 0,
    "proto_failures": 0,
    "proto_dimension_mismatch": 0,  # NEW
    "linear_head_success": 0,
    "linear_head_failures": 0,
    "encoder_mismatch_warnings": 0,  # NEW
    "fallback_triggered": 0,
    "total_predictions": 0,
}


def _load_linear_head():
    """Load linear head model if available (with encoder alignment check)."""
    global _linear_head_bundle
    if _linear_head_bundle is None and LINEAR_HEAD_PATH.exists():
        try:
            _linear_head_bundle = joblib.load(LINEAR_HEAD_PATH)

            # CRITICAL: Check encoder alignment
            bundle_encoder = _linear_head_bundle.get("encoder", "unknown")
            logger.info(f"✅ Loaded linear head from {LINEAR_HEAD_PATH}")
            logger.info(f"   Bundle encoder: {bundle_encoder}")

            # Store encoder name for validation
            _linear_head_bundle["_encoder_validated"] = False

        except Exception as e:
            logger.warning(f"⚠️ Failed to load linear head: {e}")
            _linear_head_bundle = False  # Mark as attempted but failed
    return _linear_head_bundle if _linear_head_bundle else None


def _predict_with_linear_head(text: str, encoder):
    """Predict emotion using linear head (if available) with encoder validation."""
    bundle = _load_linear_head()
    if bundle is None:
        return None  # Linear head not available

    try:
        # Validate encoder alignment (once)
        if not bundle.get("_encoder_validated", False):
            bundle_encoder = bundle.get("encoder", "unknown")
            runtime_encoder = getattr(encoder, 'model_name', 'unknown')

            if bundle_encoder != runtime_encoder:
                logger.warning("⚠️ ENCODER MISMATCH DETECTED!")
                logger.warning(f"   Linear head trained with: {bundle_encoder}")
                logger.warning(f"   Runtime encoder: {runtime_encoder}")
                logger.warning("   Predictions may be suboptimal.")
                _METRICS["encoder_mismatch_warnings"] = _METRICS.get("encoder_mismatch_warnings", 0) + 1
            else:
                logger.info(f"✅ Encoder alignment validated: {runtime_encoder}")

            bundle["_encoder_validated"] = True
        # Preprocess (light mode)
        from scripts.preprocess_text import preprocess_light

        text_preprocessed = preprocess_light(text)

        # Encode with E5 prefix
        text_prefixed = f"query: {text_preprocessed.strip()}"
        embeddings = encoder.encode([text_prefixed])

        # Handle encoder output format correctly
        if len(embeddings.shape) == 1:
            # Si encode([text]) retourne directement le vecteur (1024,)
            embedding = embeddings
        else:
            # Si encode([text]) retourne (1, 1024)
            embedding = embeddings[0]

        # L2 normalize
        import numpy as np

        embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

        # Predict with calibrated classifier
        clf = bundle["clf"]
        probs = clf.predict_proba([embedding])[0]
        pred_idx = np.argmax(probs)

        classes = bundle["classes"]
        primary_emotion = classes[pred_idx]
        confidence = float(probs[pred_idx])

        # Build result
        all_scores = {classes[i]: float(probs[i]) for i in range(len(classes))}

        # Apply tie-break rule if applicable
        override_primary, override_scores, rule_applied = _apply_anger_tiebreak(text, all_scores)

        if override_primary:
            primary_emotion = override_primary
            all_scores = override_scores
            confidence = all_scores[primary_emotion]
            logger.debug(f"🎯 Linear head tie-break rule applied: {rule_applied} → {primary_emotion}")

        _METRICS["linear_head_success"] += 1

        return {
            "primary": primary_emotion,
            "confidence": confidence,
            "all_scores": all_scores,
            "method": "linear_head",
            "rule_applied": rule_applied,
        }
    except Exception as e:
        logger.error(f"Error in linear head prediction: {e}")
        _METRICS["linear_head_failures"] += 1
        return None


class RegexEmotionDetector:
    """
    Backend regex simple (fallback déterministe).
    Version épurée (GPT) avec patterns essentiels.
    """

    def __init__(self):
        import re

        self._re = re

        # Patterns minimaux mais efficaces
        self.patterns = {
            "joy": [r"\b(happy|joy|love|great|amazing|excellent)\b"],
            "sadness": [r"\b(sad|depressed|down|cry|tears|disappointed)\b"],
            "anger": [r"\b(angry|mad|furious|hate|pissed)\b"],
            "fear": [r"\b(afraid|scared|fear|anxious|panic|worried)\b"],
            "surprise": [r"\b(surprised|shocked|wow|omg)\b"],
            "disgust": [r"\b(disgust|gross|nasty|yuck)\b"],
            "neutral": [r"\b(ok|fine|normal|anyway|actually)\b"],
            "frustration": [r"\b(frustrat|annoyed|fed up|ugh)\b"],
        }

        self._compiled = {
            cls: [self._re.compile(p, self._re.I) for p in patterns] for cls, patterns in self.patterns.items()
        }

        logger.info("✅ RegexEmotionDetector initialized (fallback mode)")

    def predict_label(self, text: str) -> str:
        """Prédit via regex (déterministe)."""
        if not text:
            return "neutral"

        scores = {
            cls: sum(1 for pattern in patterns if pattern.search(text)) for cls, patterns in self._compiled.items()
        }

        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "neutral"

    def predict_proba(self, text: str) -> dict[str, float]:
        """Distribution one-hot sur classe prédite."""
        predicted = self.predict_label(text)
        return {cls: 1.0 if cls == predicted else 0.0 for cls in CORE_EMOTIONS}


class ProtoEmotionDetector:
    """
    Backend ProtoClassifier ML avancé - VERSION OPTIMISÉE.

    Améliorations:
    - Lazy loading des prototypes (Gemini)
    - Validation métadonnées (GPT)
    - Chemin configurable (GPT)
    - Normalisation L2 (GPT)
    - Fallback runtime robuste
    - Métriques intégrées (Grok)
    """

    def __init__(self):
        logger.info("🔄 Initializing ProtoEmotionDetector (ML backend optimized v2.4.2)...")

        try:
            # Encoder + ProtoClassifier (dépendances minimales - GPT)
            self.encoder = create_default_encoder()

            # GPT micro-adjustment: Cache encoder dimension
            self._encoder_dim = None
            if hasattr(self.encoder.model, 'get_sentence_embedding_dimension'):
                self._encoder_dim = self.encoder.model.get_sentence_embedding_dimension()
            elif hasattr(self.encoder, 'get_sentence_embedding_dimension'):
                self._encoder_dim = self.encoder.get_sentence_embedding_dimension()
            else:
                # Fallback: single test encode
                self._encoder_dim = self.encoder.encode(["query: test"]).shape[-1]

            logger.info(f"   Cached encoder dimension: {self._encoder_dim}")

            # ✨ Restreindre aux émotions core-8 (FIX ProtoClassifier mismatch)
            self.proto = ProtoClassifier(allowed_labels=CORE_EMOTIONS)

            # Fallback regex pour runtime failures
            self._fallback = RegexEmotionDetector()

            # ✨ Lazy loading : prototypes chargés au premier predict
            self._prototypes_loaded = False
            self._prototypes_enabled = True  # New: can be disabled by dimension guard
            self._prototypes_metadata = None

            logger.info("✅ ProtoEmotionDetector initialized (lazy loading enabled)")

        except Exception as e:
            logger.error(f"❌ Failed to initialize ProtoEmotionDetector: {e}")
            raise  # On remonte l'erreur pour que make_emotion_backend puisse fallback

    def _ensure_prototypes_loaded(self):
        """
        Lazy loading des prototypes (Gemini recommendation).
        Charge seulement au premier predict pour optimiser init.
        """
        if self._prototypes_loaded:
            return

        self._load_prototypes()
        self._prototypes_loaded = True

    def _load_prototypes(self):
        """
        Charge les prototypes avec priorité mE5-large + validation métadonnées (GPT).
        """
        import json
        from pathlib import Path

        import numpy as np

        data_dir = Path("data")

        # Priority: mE5-large prototypes (1024-dim) > legacy (768-dim)
        proto_path_new = data_dir / "prototypes_mE5large.npz"
        meta_path_new = data_dir / "prototypes_mE5large.meta.json"

        if proto_path_new.exists() and meta_path_new.exists():
            proto_path = proto_path_new
            meta_path = meta_path_new
            logger.info(f"✅ Using mE5-large prototypes (1024-dim): {proto_path}")
        else:
            # Fallback to legacy prototypes
            proto_path = data_dir / "prototypes.npz"
            meta_path = data_dir / "prototypes.meta.json"
            if proto_path.exists():
                logger.warning(f"⚠️ Using legacy prototypes (768-dim): {proto_path}")
            else:
                logger.warning("⚠️ No prototypes found (neither mE5-large nor legacy)")
                logger.warning("   Run: python scripts/train_prototypes_mE5large.py")
                return

        logger.info(f"📂 Loading prototypes from {proto_path}...")

        try:
            # Charger métadonnées (GPT recommendation + FIX GPT #4)
            meta = None
            if meta_path.exists():
                try:
                    with open(meta_path, encoding='utf-8') as f:
                        meta = json.load(f)

                    logger.info("📝 Metadata loaded:")
                    logger.info(f"   Version: {meta.get('version', 'unknown')}")
                    logger.info(f"   Method: {meta.get('method', 'unknown')}")
                    logger.info(f"   Validation: {meta.get('validation_method', 'unknown')}")
                    logger.info(f"   F1 LOSO: {meta.get('validation', {}).get('f1_macro', 'N/A')}")

                    # ✨ Validation dimension (GPT recommendation + FIX GPT #4)
                    expected_dim = meta.get("embedding_dim")
                    if expected_dim:
                        # GPT MICRO-ADJUSTMENT: Use cached dimension to avoid repeated "test" encode calls
                        current_dim = self._encoder_dim

                        if int(current_dim) != int(expected_dim):
                            logger.error("❌ Embedding dimension mismatch!")
                            logger.error(f"   Expected: {expected_dim} (from prototypes)")
                            logger.error(f"   Current: {current_dim} (from encoder)")
                            logger.error("   Fallback will be triggered")
                            return

                    self._prototypes_metadata = meta

                except Exception as e:
                    logger.warning(f"⚠️  Failed to load metadata: {e}")

            # Charger prototypes
            data = np.load(proto_path)

            # ✨ Créer ProtoEmotion objects (FIX ProtoClassifier API)
            from jeffrey.ml.proto import ProtoEmotion

            prototypes = {}
            loaded_vectors = {}

            # GPT micro-adjustment: Handle both proto_ prefix and legacy format
            for key in data.files:
                if key.startswith("proto_"):
                    emotion = key.replace("proto_", "")
                    vec = data[key]
                elif key in CORE_EMOTIONS:
                    # Legacy format without prefix
                    emotion = key
                    vec = data[key]
                else:
                    continue

                # Handle both single vectors and k-medoids (k, dim) arrays
                if len(vec.shape) == 1:
                    # Single vector (legacy format)
                    vec = vec / (np.linalg.norm(vec) + 1e-12)
                else:
                    # K-medoids array (k, dim) - normalize each medoid
                    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)

                loaded_vectors[emotion] = vec

            # CRITICAL: Dimension mismatch guard (GPT micro-adjustment: use cached dim)
            if loaded_vectors:
                # Get prototype dimensions
                proto_dims = {
                    emotion: vec.shape[-1] if len(vec.shape) == 1 else vec.shape[1]
                    for emotion, vec in loaded_vectors.items()
                }

                # Check if all proto dims match encoder dim (using cached value)
                encoder_dim = self._encoder_dim
                mismatched_emotions = [emotion for emotion, dim in proto_dims.items() if dim != encoder_dim]

                if mismatched_emotions:
                    logger.error("🚨 DIMENSION MISMATCH DETECTED!")
                    logger.error(f"   Encoder dimension: {encoder_dim}")
                    logger.error(f"   Mismatched emotions: {mismatched_emotions}")
                    logger.error(f"   Proto dimensions: {proto_dims}")
                    logger.error("   Disabling prototype fallback for safety.")

                    # Disable prototypes fallback
                    self._prototypes_enabled = False
                    _METRICS["proto_dimension_mismatch"] = 1

                    # Set fallback to regex-only
                    return  # Exit early, no prototypes loaded

                # If we reach here, dimensions are OK
                logger.info(
                    f"✅ Dimension check passed: encoder={encoder_dim}, protos={proto_dims[list(proto_dims.keys())[0]]}"
                )
                self._prototypes_enabled = True

            # Vérifier qu'on a toutes les émotions core-8 (FIX GPT #6)
            missing_emotions = set(CORE_EMOTIONS) - set(loaded_vectors.keys())
            if missing_emotions:
                logger.warning(f"⚠️  Missing prototypes for: {missing_emotions}")
                # Stratégie: copier neutral ou créer vecteurs moyens
                if "neutral" in loaded_vectors:
                    for emotion in missing_emotions:
                        loaded_vectors[emotion] = loaded_vectors["neutral"].copy()
                        logger.info(f"   📋 {emotion}: copied from neutral")

            # Créer ProtoEmotion objects pour chaque émotion
            if loaded_vectors:
                first_vector = list(loaded_vectors.values())[0]
                current_dim = first_vector.shape[0] if len(first_vector.shape) == 1 else first_vector.shape[1]
            else:
                # Fallback: tester encoder
                test_emb = self.encoder.encode(["test"])
                current_dim = test_emb.shape[0] if len(test_emb.shape) == 1 else test_emb.shape[1]

            for emotion, vector in loaded_vectors.items():
                if len(vector.shape) == 1:
                    # Single vector (legacy format)
                    k = 1
                    proto_emotion = ProtoEmotion(emotion, current_dim, alpha=0.05, k=k)
                    proto_emotion.centroids[0] = vector
                    proto_emotion.n_samples[0] = 1
                    proto_emotion.centroid = vector  # Compatibilité API
                else:
                    # K-medoids format (k, dim)
                    k = vector.shape[0]
                    proto_emotion = ProtoEmotion(emotion, current_dim, alpha=0.05, k=k)

                    # Load all k medoids
                    for i in range(k):
                        proto_emotion.centroids[i] = vector[i]
                        proto_emotion.n_samples[i] = 1

                    # Compatibility API (first medoid)
                    proto_emotion.centroid = vector[0]

                prototypes[emotion] = proto_emotion

            # Assigner au ProtoClassifier
            self.proto.prototypes = prototypes

            logger.info(f"✅ Loaded {len(prototypes)} prototypes: {list(prototypes.keys())}")

            if meta:
                validation = meta.get("validation", {})
                f1 = validation.get("f1_macro", 0)
                if f1 > 0:
                    logger.info(f"🎯 Expected F1: {f1:.3f}")

        except Exception as e:
            logger.error(f"❌ Failed to load prototypes: {e}")
            logger.warning("   ProtoClassifier will use fallback")

    def predict_proba(self, text: str) -> tuple[dict[str, float], bool]:
        """
        Prédit distribution avec linear head priority + fallback prototypes + monitoring.

        Returns:
            (probs_dict, used_fallback) : Probabilités + flag fallback
        """
        start_time = time.time()
        _METRICS["total_predictions"] += 1

        # Cas spécial : texte vide
        if not text:
            return ({cls: 1.0 if cls == "neutral" else 0.0 for cls in CORE_EMOTIONS}, False)

        # PRIORITY 1: Try linear head first (if available)
        linear_result = _predict_with_linear_head(text, self.encoder)
        if linear_result:
            # Log prediction with monitoring
            try:
                latency_ms = (time.time() - start_time) * 1000
                monitor = get_monitor()
                monitor.log_prediction(
                    text=text,
                    route="linear_head",
                    primary_emotion=linear_result["primary"],
                    confidence=linear_result["confidence"],
                    all_scores=linear_result["all_scores"],
                    latency_ms=latency_ms,
                    encoder_name=getattr(self.encoder, 'model_name', 'unknown'),
                    version="2.4.2",
                    low_confidence=linear_result["confidence"] < 0.4,
                    rule_applied=linear_result.get("rule_applied"),
                )
            except Exception as e:
                logger.error(f"Monitoring error: {e}")

            return (linear_result["all_scores"], False)

        # PRIORITY 2: Fallback to prototypes
        # ✨ Lazy loading
        self._ensure_prototypes_loaded()

        # Check if prototypes are enabled (not disabled by dimension guard)
        if not getattr(self, '_prototypes_enabled', True):
            logger.warning("⚠️ Prototypes disabled due to dimension mismatch, using regex fallback")
            _METRICS["fallback_triggered"] += 1
            probs_dict = self._fallback.predict_proba(text)
            return (probs_dict, True)

        # Tentative prédiction ML
        try:
            start_time = time.time()

            # GPT micro-adjustment: Apply consistent preprocessing (light mode) + query: prefix
            from scripts.preprocess_text import preprocess_light

            text_prep = preprocess_light(text)
            text_prefixed = f"query: {text_prep.strip()}"

            # Encodage
            embeddings = self.encoder.encode([text_prefixed])
            # Fix: Handle encoder output format correctly
            if len(embeddings.shape) == 1:
                # Si encode([text]) retourne directement le vecteur (1024,)
                embedding = embeddings
            else:
                # Si encode([text]) retourne (1, 1024)
                embedding = embeddings[0]

            # ✨ Normalisation L2 à l'inférence (GPT + FIX GPT #2)
            embedding = embedding / (np.linalg.norm(embedding) + 1e-12)

            # Prédiction ProtoClassifier
            prediction = self.proto.predict(embedding, text=text, return_all=True)

            # Extraire probas
            probs_dict = prediction.all_scores

            # Safeguards
            if not probs_dict:
                raise ValueError("ProtoClassifier returned empty scores (model not trained)")

            if not all(np.isfinite(list(probs_dict.values()))):
                raise ValueError("Probs contain NaN or Inf")

            # Normalisation - assurer toutes émotions core-8 (FIX GPT #6)
            normalized_probs = {}
            for emotion in CORE_EMOTIONS:
                if emotion in probs_dict:
                    normalized_probs[emotion] = probs_dict[emotion]
                else:
                    # Si émotion manquante, utiliser 0.0 (sera normalisé)
                    normalized_probs[emotion] = 0.0
                    logger.debug(f"Missing emotion in prediction: {emotion}")

            # Normalisation finale
            prob_sum = sum(normalized_probs.values())
            if prob_sum <= 0:
                raise ValueError("Probs sum to zero or negative")

            normalized_probs = {k: v / prob_sum for k, v in normalized_probs.items()}

            # Apply tie-break rule if applicable
            override_primary, override_scores, rule_applied = _apply_anger_tiebreak(text, normalized_probs)

            if override_primary:
                primary_emotion = override_primary
                normalized_probs = override_scores
                confidence = normalized_probs[primary_emotion]
                logger.debug(f"🎯 Prototypes tie-break rule applied: {rule_applied} → {primary_emotion}")
            else:
                primary_emotion = max(normalized_probs, key=normalized_probs.get)
                confidence = normalized_probs[primary_emotion]

            # Métriques success
            latency = time.time() - start_time
            _METRICS["proto_success"] += 1

            if latency > 1.0:
                logger.warning(f"⚠️  Slow prediction: {latency:.2f}s")

            # Log prediction with monitoring
            try:
                latency_ms = latency * 1000

                monitor = get_monitor()
                monitor.log_prediction(
                    text=text,
                    route="prototypes",
                    primary_emotion=primary_emotion,
                    confidence=confidence,
                    all_scores=normalized_probs,
                    latency_ms=latency_ms,
                    encoder_name=getattr(self.encoder, 'model_name', 'unknown'),
                    version="2.4.2",
                    low_confidence=confidence < 0.4,
                    rule_applied=rule_applied,
                )
            except Exception as e:
                logger.error(f"Monitoring error (prototypes): {e}")

            return (normalized_probs, False)

        except Exception as e:
            # Runtime fallback (Gemini) : comportement déterministe
            logger.error(f"❌ ProtoClassifier runtime error: {e}")
            _METRICS["proto_failures"] += 1
            _METRICS["fallback_triggered"] += 1

            logger.info("🔄 Falling back to RegexEmotionDetector")
            fallback_probs = self._fallback.predict_proba(text)

            # Log regex fallback with monitoring
            try:
                latency_ms = (time.time() - start_time) * 1000
                primary_emotion = max(fallback_probs, key=fallback_probs.get)
                confidence = fallback_probs[primary_emotion]

                monitor = get_monitor()
                monitor.log_prediction(
                    text=text,
                    route="regex",
                    primary_emotion=primary_emotion,
                    confidence=confidence,
                    all_scores=fallback_probs,
                    latency_ms=latency_ms,
                    encoder_name="regex_fallback",
                    version="2.4.2",
                    low_confidence=confidence < 0.4,
                    rule_applied=None,
                )
            except Exception as e:
                logger.error(f"Monitoring error (regex): {e}")

            return (fallback_probs, True)

    def predict_label(self, text: str) -> str:
        """Prédit label (argmax de predict_proba)."""
        probs, used_fallback = self.predict_proba(text)

        if used_fallback:
            logger.debug("⚠️ Prediction used fallback")

        return max(probs, key=probs.get)


# Cache singleton (GPT)
_CACHED_BACKENDS: dict[str, object] = {}


def make_emotion_backend(kind: str | None = None) -> object:
    """
    Factory pour créer backend émotionnel avec singleton.

    Args:
        kind: Type backend ('proto' ou 'regex')
              Si None, lit JEFFREY_EMOTION_BACKEND env var

    Returns:
        Instance de ProtoEmotionDetector ou RegexEmotionDetector
    """
    # Déterminer le type de backend
    if kind is None:
        kind = os.getenv("JEFFREY_EMOTION_BACKEND", "proto")

    kind = (kind or "").strip().lower()

    # Singleton : retourner instance cachée si existe
    if kind in _CACHED_BACKENDS:
        logger.debug(f"♻️ Reusing cached {kind} backend")
        return _CACHED_BACKENDS[kind]

    # Créer nouvelle instance
    logger.info(f"🔨 Creating emotion backend: {kind}")

    if kind == "proto":
        try:
            inst = ProtoEmotionDetector()
        except Exception as e:
            logger.error(f"❌ Failed to init ProtoEmotionDetector: {e}")
            logger.warning("🔄 Falling back to RegexEmotionDetector")
            inst = RegexEmotionDetector()
    elif kind == "regex":
        inst = RegexEmotionDetector()
    else:
        logger.warning(f"⚠️ Unknown backend '{kind}', defaulting to proto")
        return make_emotion_backend("proto")

    # Cache l'instance
    _CACHED_BACKENDS[kind] = inst
    return inst


def get_metrics() -> dict[str, int]:
    """Retourne métriques monitoring (Grok)."""
    return _METRICS.copy()


def reset_metrics() -> None:
    """Reset compteurs (pour tests)."""
    global _METRICS
    _METRICS = {k: 0 for k in _METRICS}


# Export public
__all__ = [
    "CORE_EMOTIONS",
    "make_emotion_backend",
    "get_metrics",
    "reset_metrics",
    "ProtoEmotionDetector",
    "RegexEmotionDetector",
]
