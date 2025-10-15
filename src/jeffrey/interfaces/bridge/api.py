"""
Bridge API endpoints - Clean /v1/chat implementation
Dedicated file for API routes to keep architecture clear
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from .core_client import CoreClient

logger = logging.getLogger(__name__)

app = FastAPI(title="Jeffrey Bridge API", version="2.0")

# CORS configuration for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response validation
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[Message]
    model: str | None = "jeffrey-core"
    temperature: float | None = 0.7
    max_tokens: int | None = 1000


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


# Modèles Pydantic pour l'émotion ML
class EmotionDetectRequest(BaseModel):
    """Requête de détection d'émotion"""

    text: str = Field(..., min_length=1, max_length=2000, description="Text to analyze")

    @validator('text')
    def clean_text(cls, v):
        return v.strip()


class EmotionDetectResponse(BaseModel):
    """Réponse de détection d'émotion"""

    success: bool
    emotion: str
    confidence: float
    all_scores: dict | None = None
    method: str
    latency_ms: float
    error: str | None = None


# Initialize CoreClient
core_client = CoreClient()

# Singleton pour l'adapter et rate limiting
_emotion_adapter = None
_adapter_lock = asyncio.Lock()
_request_counts = {}


async def get_emotion_adapter():
    """Dependency injection pour l'adapter"""
    global _emotion_adapter
    if _emotion_adapter is None:
        async with _adapter_lock:
            if _emotion_adapter is None:
                try:
                    from jeffrey.ml.emotion_ml_adapter import EmotionMLAdapter

                    _emotion_adapter = await EmotionMLAdapter.get_instance()
                except ImportError:
                    logger.error("EmotionMLAdapter not available")
                    raise HTTPException(status_code=503, detail="ML emotion detection not available")
    return _emotion_adapter


def check_rate_limit(client_ip: str) -> bool:
    """Simple rate limiting par IP"""
    global _request_counts
    rate_limit = int(os.getenv("EMO_RATE_LIMIT", "100"))  # per minute
    current_minute = int(time.time() / 60)
    key = f"{client_ip}:{current_minute}"

    if key not in _request_counts:
        _request_counts = {k: v for k, v in _request_counts.items() if int(k.split(':')[1]) >= current_minute - 1}
        _request_counts[key] = 0

    _request_counts[key] += 1
    return _request_counts[key] <= rate_limit


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    core_healthy = await core_client.health_check()
    return {
        "status": "healthy" if core_healthy else "degraded",
        "core_connected": core_healthy,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """
    OpenAI-compatible chat completions endpoint
    Routes requests to Jeffrey Core via Unix socket
    """
    try:
        # Prepare query for Core
        core_query = {
            "action": "chat",
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }

        # Send to Core and get response
        response = await core_client.ask_core(core_query)

        if not response:
            raise HTTPException(status_code=503, detail="Core service unavailable")

        # Format as OpenAI-compatible response
        chat_response = ChatResponse(
            id=f"chatcmpl-{datetime.utcnow().timestamp():.0f}",
            created=int(datetime.utcnow().timestamp()),
            model=request.model,
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response.get("content", "")},
                    "finish_reason": "stop",
                }
            ],
            usage={
                "prompt_tokens": response.get("prompt_tokens", 0),
                "completion_tokens": response.get("completion_tokens", 0),
                "total_tokens": response.get("total_tokens", 0),
            },
        )

        return chat_response

    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat")
async def simple_chat(messages: list[dict[str, str]]):
    """
    Simplified chat endpoint for testing
    Direct JSON input without Pydantic validation
    """
    try:
        query = {"action": "chat", "messages": messages}
        response = await core_client.ask_core(query)
        return {"response": response.get("content", "") if response else "Core unavailable"}
    except Exception as e:
        return {"error": str(e)}


@app.post("/api/v1/emotion/detect", response_model=EmotionDetectResponse)
async def detect_emotion_ml(
    request: EmotionDetectRequest, emotion_adapter=Depends(get_emotion_adapter), req: Request = None
):
    """
    Détecte l'émotion d'un texte avec le système ML.

    Features:
    - ML avec fallback automatique
    - Validation des inputs
    - Timeout configurable
    - Métriques de performance
    """
    try:
        # Rate limiting basique
        client_ip = req.client.host if req else "unknown"
        if not check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Détection avec timeout
        result = await asyncio.wait_for(
            emotion_adapter.detect_emotion(request.text),
            timeout=5.0,  # API timeout plus strict
        )

        resp = EmotionDetectResponse(**result)

        # NOUVELLE PARTIE : Publication sur NeuralBus
        try:
            from jeffrey.core.neuralbus.contracts import EMOTION_DETECTED

            # Récupérer ou créer le bus (singleton pattern)
            bus = getattr(app.state, "_neural_bus", None)
            if bus:
                # Publier l'événement
                await bus.publish(topic=EMOTION_DETECTED, data={
                    "text": request.text,
                    "emotion": resp.emotion,
                    "confidence": resp.confidence,
                    "all_scores": resp.all_scores,
                    "timestamp": datetime.utcnow().isoformat()
                })

                # Tracking pour monitoring (thread-safe best-effort)
                app.state._event_counts = getattr(app.state, "_event_counts", {})
                app.state._event_counts[EMOTION_DETECTED] = app.state._event_counts.get(EMOTION_DETECTED, 0) + 1
                logger.info(f"✅ Published emotion event #{app.state._event_counts[EMOTION_DETECTED]}")
            else:
                logger.debug("NeuralBus not available, skipping event publication")

        except Exception as e:
            logger.warning(f"Could not publish emotion event: {e}")
            # Ne pas bloquer la réponse API

        return resp

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Detection timeout")
    except Exception as e:
        logger.error(f"API emotion detection error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/emotion/stats")
async def get_emotion_stats(emotion_adapter=Depends(get_emotion_adapter)):
    """
    Retourne les statistiques du système ML.

    Inclut:
    - Nombre de prédictions
    - Latence moyenne
    - Taux de fallback
    - Taux d'erreur
    """
    try:
        stats = emotion_adapter.get_stats()
        return {
            "success": True,
            "data": stats,
            "service": "emotion_ml_adapter",
            "version": "1.0.0",
            "uptime_seconds": time.time() - getattr(app.state, '_start_time', time.time()),
        }
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/emotion/health")
async def emotion_health_check(emotion_adapter=Depends(get_emotion_adapter)):
    """
    Health check endpoint pour monitoring.
    """
    try:
        # Test rapide
        test_result = await emotion_adapter.detect_emotion("health check")

        return {
            "status": "healthy" if test_result["success"] else "degraded",
            "ml_enabled": emotion_adapter.use_ml,
            "checks": {
                "ml_model": "ok" if emotion_adapter.use_ml else "disabled",
                "fallback": "ok",
                "latency_ms": test_result.get("latency_ms", 0),
            },
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


@app.on_event("startup")
async def startup_event():
    """Initialisation au démarrage du serveur"""
    try:
        # Warmup ML (existant si disponible)
        try:
            from jeffrey.ml.emotion_ml_adapter import EmotionMLAdapter
            adapter = await EmotionMLAdapter.get_instance()
            await adapter.detect_emotion("warmup")
            logger.info("✅ ML warmup completed")
        except Exception as e:
            logger.warning(f"ML warmup failed: {e}")

        # NOUVEAU : Bootstrap du cerveau
        from jeffrey.core.brain_bootstrap import BrainBootstrap

        app.state._neural_bus = getattr(app.state, "_neural_bus", None)
        bootstrap = BrainBootstrap(app.state._neural_bus)
        app.state._neural_bus = await bootstrap.wire_minimal_loop()
        app.state._brain_bootstrap = bootstrap
        app.state._event_counts = {}

        logger.info("🧠 Brain bootstrap completed")
        logger.info(f"   Stats: {bootstrap.get_stats()}")

    except Exception as e:
        logger.error(f"❌ Startup bootstrap failed: {e}")
        # Ne pas bloquer le démarrage


@app.get("/api/v1/brain/status")
async def brain_status():
    """Endpoint pour vérifier l'état du cerveau"""
    bootstrap = getattr(app.state, "_brain_bootstrap", None)
    stats = bootstrap.get_stats() if bootstrap else {}
    events = getattr(app.state, "_event_counts", {})

    return {
        "status": "active" if stats.get("wired") else "inactive",
        "modules": {
            "memory": stats.get("memory_available", False),
            "consciousness": stats.get("consciousness_available", False),
            "memory_type": stats.get("memory_type"),
            "consciousness_type": stats.get("consciousness_type")
        },
        "activity": {
            "emotions_processed": stats.get("emotions_received", 0),
            "memories_stored": stats.get("memories_stored", 0),
            "thoughts_generated": stats.get("thoughts_generated", 0)
        },
        "events": events,
        "errors": stats.get("errors", 0)
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
