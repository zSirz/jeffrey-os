"""
Endpoint simple pour emotion/detect avec métriques Prometheus

Ce module fournit un endpoint de détection d'émotion basique pour les tests
et le développement, avec intégration complète des métriques Prometheus.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List
import random
import time
import logging
from datetime import datetime

# Métriques Prometheus - correctif GPT #1
from prometheus_client import Counter, Histogram
from jeffrey.memory.hybrid_store import HybridMemoryStore

logger = logging.getLogger(__name__)

# Memory store pour sauvegarder les emotions
memory_store = HybridMemoryStore()

# Métriques Prometheus pour emotion endpoint (noms différents pour éviter conflit)
emotion_confidence_sum = Counter(
    "jeffrey_emotion_endpoint_confidence_sum",
    "Sum of predicted emotion confidences from endpoint"
)

emotion_confidence_count = Counter(
    "jeffrey_emotion_endpoint_confidence_count",
    "Count of emotion predictions from endpoint"
)

emotion_requests_total = Counter(
    "jeffrey_emotion_requests_total",
    "Total emotion detection requests",
    ["emotion", "method"]
)

emotion_detection_duration = Histogram(
    "jeffrey_emotion_detection_duration_seconds",
    "Time spent on emotion detection",
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)

router = APIRouter(prefix="/api/v1/emotion", tags=["emotion"])


class EmotionRequest(BaseModel):
    """Requête de détection d'émotion"""
    text: str = Field(..., min_length=1, max_length=2000, description="Text to analyze for emotion")

    class Config:
        example = {
            "text": "I am very happy about this new project!"
        }


class EmotionResponse(BaseModel):
    """Réponse de détection d'émotion enrichie"""
    emotion: str = Field(..., description="Detected primary emotion")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    text: str = Field(..., description="Original input text")
    method: str = Field(..., description="Detection method used")
    timestamp: str = Field(..., description="Detection timestamp")
    all_scores: Dict[str, float] = Field(..., description="All emotion scores")


@router.post("/detect", response_model=EmotionResponse)
async def detect_emotion(request: EmotionRequest) -> EmotionResponse:
    """
    Détection d'émotion simple pour tests et développement

    Cette implémentation utilise des règles basiques et de la randomisation
    pour simuler un système de détection d'émotion. En production, cet
    endpoint devrait être remplacé par un vrai modèle ML.

    Métriques Prometheus intégrées pour monitoring.
    """
    start_time = time.time()

    try:
        # Simulation simple pour tests
        emotions = ["joy", "sadness", "anger", "fear", "surprise", "curiosity", "neutral"]

        # Analyse basique basée sur mots-clés
        text_lower = request.text.lower()

        # Dictionnaire de mots-clés par émotion
        emotion_keywords = {
            "joy": ["happy", "joy", "excited", "great", "wonderful", "amazing", "love", "perfect", "fantastic"],
            "sadness": ["sad", "depressed", "unhappy", "terrible", "awful", "disappointed", "crying", "grief"],
            "anger": ["angry", "mad", "furious", "hate", "irritated", "annoyed", "frustrated", "rage"],
            "fear": ["scared", "afraid", "terrified", "worried", "anxious", "nervous", "panic", "frightened"],
            "surprise": ["surprised", "shocked", "unexpected", "amazed", "astonished", "wow", "incredible"],
            "curiosity": ["curious", "wonder", "interesting", "question", "explore", "discover", "learn"]
        }

        # Calculer scores pour chaque émotion
        all_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 0.15  # Boost per keyword match

            # Ajouter du bruit aléatoire
            score += random.uniform(0.0, 0.3)
            all_scores[emotion] = min(score, 1.0)

        # Ajouter score neutral
        all_scores["neutral"] = random.uniform(0.1, 0.4)

        # Trouver l'émotion dominante
        detected_emotion = max(all_scores.items(), key=lambda x: x[1])
        emotion = detected_emotion[0]
        confidence = detected_emotion[1]

        # Normaliser les scores pour qu'ils soient plus réalistes
        if confidence < 0.5:
            confidence = random.uniform(0.5, 0.7)
            all_scores[emotion] = confidence

        # Méthode utilisée
        method = "keyword_matching_v1"

        # Créer la réponse
        response = EmotionResponse(
            emotion=emotion,
            confidence=confidence,
            text=request.text,
            method=method,
            timestamp=datetime.now().isoformat(),
            all_scores=all_scores
        )

        # Métriques Prometheus - correctif GPT #1
        emotion_confidence_sum.inc(confidence)
        emotion_confidence_count.inc()
        emotion_requests_total.labels(emotion=emotion, method=method).inc()

        # Mesurer la durée
        duration = time.time() - start_time
        emotion_detection_duration.observe(duration)

        # Sauvegarder en mémoire
        try:
            memory_data = {
                'text': request.text,
                'emotion': emotion,
                'confidence': confidence,
                'meta': {
                    'source': 'emotion_detection',
                    'method': method,
                    'all_scores': all_scores,
                    'processing_time_ms': duration * 1000
                }
            }
            await memory_store.store(memory_data)
            logger.info(f"✅ Emotion saved to memory: {emotion} ({confidence:.2f})")
        except Exception as e:
            logger.warning(f"Failed to save emotion to memory: {e}")

        logger.info(f"Detected emotion '{emotion}' with confidence {confidence:.2f} in {duration:.3f}s")

        return response

    except Exception as e:
        # Mesurer les erreurs aussi
        duration = time.time() - start_time
        emotion_detection_duration.observe(duration)
        emotion_requests_total.labels(emotion="error", method="error").inc()

        logger.error(f"Emotion detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Emotion detection failed: {str(e)}")


@router.get("/stats")
async def get_emotion_stats() -> Dict:
    """
    Statistiques de détection d'émotion

    Retourne des métriques sur l'usage de l'endpoint emotion/detect
    """
    try:
        # Ici on pourrait calculer des stats plus sophistiquées
        # Pour l'instant, on retourne des infos basiques

        return {
            "service": "emotion_detection",
            "version": "1.0.0-simple",
            "method": "keyword_matching",
            "status": "operational",
            "features": [
                "keyword_based_detection",
                "prometheus_metrics",
                "multi_emotion_scoring"
            ],
            "supported_emotions": [
                "joy", "sadness", "anger", "fear",
                "surprise", "curiosity", "neutral"
            ],
            "metrics_available": [
                "jeffrey_emotion_endpoint_confidence_sum",
                "jeffrey_emotion_endpoint_confidence_count",
                "jeffrey_emotion_requests_total",
                "jeffrey_emotion_detection_duration_seconds"
            ],
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Stats generation failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate stats")


@router.get("/health")
async def emotion_health() -> Dict:
    """
    Health check pour l'endpoint emotion

    Vérifie que le service de détection d'émotion fonctionne
    """
    try:
        # Test rapide avec un texte simple
        test_request = EmotionRequest(text="test health check")
        test_response = await detect_emotion(test_request)

        return {
            "status": "healthy",
            "test_emotion": test_response.emotion,
            "test_confidence": test_response.confidence,
            "method": test_response.method,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.post("/batch")
async def detect_emotions_batch(texts: List[str]) -> List[EmotionResponse]:
    """
    Détection d'émotion en batch pour plusieurs textes

    Utile pour traiter plusieurs textes en une seule requête
    """
    if len(texts) > 50:
        raise HTTPException(status_code=400, detail="Maximum 50 texts per batch")

    results = []

    for text in texts:
        try:
            request = EmotionRequest(text=text)
            response = await detect_emotion(request)
            results.append(response)
        except Exception as e:
            # En cas d'erreur sur un texte, continuer avec les autres
            logger.warning(f"Failed to process text '{text[:50]}...': {e}")
            error_response = EmotionResponse(
                emotion="error",
                confidence=0.0,
                text=text,
                method="error",
                timestamp=datetime.now().isoformat(),
                all_scores={"error": 1.0}
            )
            results.append(error_response)

    return results