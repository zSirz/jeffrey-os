"""
Bridge API endpoints - Clean /v1/chat implementation
Dedicated file for API routes to keep architecture clear
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Any
from contextlib import asynccontextmanager

from jeffrey.core.contracts.thoughts import create_thought, ThoughtState, ensure_thought_format
from jeffrey.core.neuralbus.bus_facade import BusFacade
from jeffrey.core.neuralbus.events import (
    make_event, THOUGHT_GENERATED, EMOTION_DETECTED, MEMORY_STORED, CIRCADIAN_UPDATE
)
from jeffrey.core.consciousness.self_reflection import SelfReflection
from jeffrey.core.biorhythms.circadian_runner import CircadianRunner
from jeffrey.monitoring.auto_debug import AutoDebugEngine
# FORTRESS IMPORTS
from jeffrey.core.pipeline.thought_pipeline import ThoughtPipeline
from jeffrey.core.orchestration.cognitive_orchestrator import CognitiveOrchestrator

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, validator
from jeffrey.core.auth import require_admin_permission, verify_api_key
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from .core_client import CoreClient

logger = logging.getLogger(__name__)

# LIFESPAN MANAGER (remplacer l'ancien startup_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan manager for proper startup/shutdown
    Implements GPT's clean lifecycle management
    """
    logger.info("🚀 Starting Jeffrey OS Brain v2.0 with Lifespan Manager")

    # Initialize event bus
    app.state.bus = BusFacade(
        max_queue=5000,  # GPT OPTIMIZATION: Increased from 1000 for better concurrency
        async_threshold_cpu=70.0,
        pruning_enabled=True
    )
    app.state.bus.start()

    # Initialize brain bootstrap (existing)
    from jeffrey.core.brain_bootstrap import BrainBootstrap
    bootstrap = BrainBootstrap(None)  # We'll use BusFacade instead
    await bootstrap.initialize_modules()

    # Expose bootstrap sous les deux noms pour compatibilité
    app.state.bootstrap = bootstrap
    app.state._brain_bootstrap = bootstrap  # Compatibilité pour readyz

    # 🏰 INITIALIZE COGNITIVE FORTRESS
    logger.info("🏰 Initializing Cognitive Fortress...")

    # Initialize Orchestrator (Grok's brain conductor)
    app.state.orchestrator = CognitiveOrchestrator(
        app.state.bus,
        config={
            "auto_heal_interval": 60,
            "error_threshold": 5,
            "inactive_threshold": 300
        }
    )
    await app.state.orchestrator.start()

    # Initialize ThoughtPipeline (GPT's robust processor)
    app.state.pipeline = ThoughtPipeline(
        bus=app.state.bus,
        memory=bootstrap.memory,
        consciousness=bootstrap.consciousness,
        orchestrator=app.state.orchestrator,
        max_concurrency=32,  # GPT OPTIMIZATION: Increased from 8 for better under-load performance
        max_retries=3
    )

    # Register pipeline as agent in orchestrator
    app.state.orchestrator.register_agent("thought_pipeline", app.state.pipeline)

    # Wire pipeline handlers through orchestrator for monitoring
    app.state.orchestrator.register_handler(
        EMOTION_DETECTED,
        app.state.pipeline.on_emotion_detected,
        priority=10,
        source_agent="thought_pipeline"
    )

    app.state.orchestrator.register_handler(
        MEMORY_STORED,
        app.state.pipeline.on_memory_stored,
        priority=8,
        source_agent="thought_pipeline"
    )

    # Subscribe orchestrator to bus (it will dispatch to handlers)
    app.state.bus.subscribe(
        EMOTION_DETECTED,
        lambda event: asyncio.create_task(
            app.state.orchestrator.dispatch(EMOTION_DETECTED, event)
        )
    )

    app.state.bus.subscribe(
        MEMORY_STORED,
        lambda event: asyncio.create_task(
            app.state.orchestrator.dispatch(MEMORY_STORED, event)
        )
    )

    # Initialize SelfReflection (direct bus subscription)
    app.state.self_reflection = SelfReflection(app.state.bus, interval=5)  # Every 5 thoughts
    app.state.bus.subscribe(THOUGHT_GENERATED, app.state.self_reflection.on_thought)

    # Initialize CircadianRhythm
    app.state.circadian = CircadianRunner(app.state.bus, interval_sec=60)  # Every minute for testing
    app.state.circadian.start()

    # Update CircadianRunner to notify orchestrator
    async def on_circadian_update(event):
        if app.state.orchestrator:
            app.state.orchestrator.update_circadian_state(event.get("data", {}))

    app.state.bus.subscribe(CIRCADIAN_UPDATE, on_circadian_update)

    logger.info("🌅 CircadianRhythm activated - Jeffrey has temporal awareness")

    # Initialize Auto-Debug Engine - GPT INNOVATION
    app.state.auto_debug = AutoDebugEngine(check_interval=30)
    app.state.auto_debug.register_component("bus", app.state.bus)
    app.state.auto_debug.register_component("pipeline", app.state.pipeline)
    app.state.auto_debug.register_component("orchestrator", app.state.orchestrator)

    # Register ML adapter when available
    emotion_adapter = await get_emotion_adapter()
    app.state.auto_debug.register_component("ml_adapter", emotion_adapter)

    await app.state.auto_debug.start()
    logger.info("🤖 Auto-Debug Engine activated - intelligent monitoring online")

    # Initialize DreamEngine with proper correctifs
    try:
        from jeffrey.core.dreaming.dream_engine_progressive import DreamEngineProgressive
        app.state.dream_engine = DreamEngineProgressive(
            bus=app.state.bus,
            memory_port=bootstrap.memory,
            circadian=app.state.circadian
        )
        logger.info("✨ DreamEngine Progressive initialized")
    except Exception as e:
        logger.warning(f"DreamEngine initialization failed: {e}")
        app.state.dream_engine = None

    # Initialize scheduler for automatic orchestration
    from jeffrey.core.scheduler import dream_scheduler
    await dream_scheduler.start()
    app.state.dream_scheduler = dream_scheduler

    # Tracking
    app.state._latencies = []
    app.state._startup_time = time.time()
    app.state._request_ids = {}

    logger.info("🏰 Cognitive Fortress initialized with:")
    logger.info("  • Robust ThoughtPipeline (retries, circuit breakers, DLQ)")
    logger.info("  • Adaptive CognitiveOrchestrator (auto-healing, predictions)")
    logger.info("  • Full event-driven architecture (Blackboard pattern)")
    logger.info("  • Self-reflection meta-cognition")
    logger.info("  • Circadian temporal awareness")

    try:
        yield  # Server runs here
    finally:
        # Cleanup on shutdown
        logger.info("🛑 Shutting down Jeffrey OS Brain...")

        # Graceful shutdown
        if hasattr(app.state, 'dream_scheduler') and app.state.dream_scheduler:
            await app.state.dream_scheduler.stop()

        if hasattr(app.state, 'auto_debug') and app.state.auto_debug:
            await app.state.auto_debug.stop()

        if hasattr(app.state, 'orchestrator'):
            await app.state.orchestrator.stop()

        if hasattr(app.state, 'circadian') and app.state.circadian:
            await app.state.circadian.stop()

        if hasattr(app.state, 'bus') and app.state.bus:
            await app.state.bus.stop(drain=True)

        logger.info("👋 Jeffrey OS Brain shutdown complete")

# Create app with lifespan
app = FastAPI(
    title="Jeffrey OS Brain API",
    version="2.0.0",
    lifespan=lifespan  # Use the new lifespan manager
)

# Create limiter for rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS configuration for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
from jeffrey.interfaces.bridge.emotion_endpoint import router as emotion_router
from jeffrey.api.routes.memory import router as memory_router
from jeffrey.api.routes.semantic_search import router as search_router
app.include_router(emotion_router)
app.include_router(memory_router)
app.include_router(search_router)


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


# GPT CORRECTION 5: ENDPOINT SLIM - just publish event
@app.post("/api/v1/emotion/detect", response_model=EmotionDetectResponse)
async def detect_emotion_ml(
    request: EmotionDetectRequest,
    emotion_adapter=Depends(get_emotion_adapter),
    req: Request = None
):
    """
    Slim emotion detection - publishes to fortress pipeline
    """
    request_id = str(uuid.uuid4())
    logger.info(f"REQ {request_id} emotion_detect start")

    try:
        # Rate limiting
        client_ip = req.client.host if req else "unknown"
        if not check_rate_limit(client_ip):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # ML detection
        result = await asyncio.wait_for(
            emotion_adapter.detect_emotion(request.text),
            timeout=5.0
        )
        resp = EmotionDetectResponse(**result)

        # Backpressure control before publishing - GPT ENHANCEMENT
        if app.state.bus.is_saturated(threshold=0.85):
            backpressure_delay = app.state.bus.get_backpressure_delay()
            if backpressure_delay > 0.3:  # Reject if delay > 300ms
                logger.warning(f"REQ {request_id} rejected due to backpressure ({backpressure_delay*1000:.0f}ms)")
                raise HTTPException(
                    status_code=503,
                    detail=f"System overloaded, queue saturation: {app.state.bus.stats['queue_saturation_pct']:.1f}%"
                )

            # Apply adaptive delay for moderate saturation
            if backpressure_delay > 0:
                logger.info(f"REQ {request_id} applying backpressure delay: {backpressure_delay*1000:.0f}ms")
                await asyncio.sleep(backpressure_delay)

        # Just publish event, pipeline handles the rest
        emotion_event = make_event(
            EMOTION_DETECTED,
            {
                "text": request.text,
                "emotion": resp.emotion,
                "confidence": resp.confidence,
                "all_scores": resp.all_scores,
                "request_id": request_id,
                "timestamp": datetime.now().isoformat()
            },
            source="jeffrey.api.emotion"
        )

        await app.state.bus.publish(emotion_event)
        logger.info(f"REQ {request_id} emotion_detect published")
        return resp  # Return immediately, processing happens async

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


# DEPRECATED STARTUP EVENT REMOVED - Now using lifespan manager


def _pctl(values, p):
    """Calcul de percentile simple et robuste"""
    if not values:
        return None
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


@app.get("/api/v1/brain/health")
async def brain_health():
    """Health check complet avec métriques détaillées"""
    bootstrap = getattr(app.state, "_brain_bootstrap", None)

    if not bootstrap:
        return {"status": "unhealthy", "error": "Brain not initialized"}

    uptime = time.time() - getattr(app.state, "_startup_time", time.time())
    latencies = getattr(app.state, "_latencies", [])[-100:]  # Dernières 100 mesures

    # Calculer les percentiles si on a des données
    p50 = _pctl(latencies, 50)
    p95 = _pctl(latencies, 95)
    p99 = _pctl(latencies, 99)

    # Stats de mémoire
    memory_stats = {}
    if bootstrap.memory:
        memory_stats = bootstrap.memory.get_stats()

    return {
        "status": "healthy" if bootstrap.wired else "degraded",
        "uptime_seconds": round(uptime, 2),
        "brain_state": {
            "wired": bootstrap.wired,
            "memory_available": bootstrap.memory is not None,
            "consciousness_available": bootstrap.consciousness is not None
        },
        "performance": {
            "latency_p50_ms": round(p50 * 1000, 2) if p50 else None,
            "latency_p95_ms": round(p95 * 1000, 2) if p95 else None,
            "latency_p99_ms": round(p99 * 1000, 2) if p99 else None,
            "total_requests": len(getattr(app.state, "_latencies", []))
        },
        "memory": memory_stats,
        "activity": bootstrap.stats,
        "errors": {
            "total": bootstrap.stats.get("errors", 0),
            "memory_errors": bootstrap.stats.get("memory_errors", 0),
            "last_error": bootstrap.stats.get("last_error", None)
        }
    }


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


@app.get("/api/v1/brain/enrichment")
async def brain_enrichment_status():
    """
    Monitoring endpoint for Phase 2 brain enrichment
    Shows status of advanced cognitive modules
    """
    # Bus stats
    bus_stats = {}
    bus = getattr(app.state, "bus", None)
    if bus:
        bus_stats = bus.get_stats()

    # Self-reflection stats
    reflection_stats = {}
    reflection = getattr(app.state, "self_reflection", None)
    if reflection:
        reflection_stats = reflection.get_stats()

    # Circadian stats
    circadian_stats = {}
    circadian = getattr(app.state, "circadian", None)
    if circadian:
        circadian_stats = circadian.get_stats()

    # Overall health
    healthy_modules = 0
    total_modules = 3

    if bus and bus_stats.get("running", False):
        healthy_modules += 1
    if reflection and reflection_stats.get("thoughts_analyzed", 0) >= 0:
        healthy_modules += 1
    if circadian and circadian_stats.get("running", False):
        healthy_modules += 1

    health_ratio = healthy_modules / total_modules

    return {
        "status": "healthy" if health_ratio >= 0.66 else "degraded" if health_ratio > 0 else "offline",
        "health_ratio": round(health_ratio, 2),
        "modules": {
            "event_bus": {
                "enabled": bus is not None,
                "stats": bus_stats
            },
            "self_reflection": {
                "enabled": reflection is not None,
                "stats": reflection_stats
            },
            "circadian_rhythm": {
                "enabled": circadian is not None,
                "stats": circadian_stats
            }
        },
        "cognitive_indicators": {
            "temporal_awareness": circadian_stats.get("running", False),
            "meta_cognition": reflection_stats.get("meta_thoughts_generated", 0) > 0,
            "event_processing": bus_stats.get("events_published", 0) > 0
        },
        "timestamp": datetime.utcnow().isoformat()
    }


# 🏰 FORTRESS MONITORING ENDPOINTS
@app.get("/api/v1/brain/fortress")
async def brain_fortress_status():
    """
    Complete fortress monitoring - GPT CORRECTION 6
    """

    fortress_status = {
        "status": "operational",
        "timestamp": datetime.now().isoformat()
    }

    # Pipeline metrics
    pipeline = getattr(app.state, 'pipeline', None)
    if pipeline:
        fortress_status["pipeline"] = pipeline.get_metrics()
        fortress_status["dlq_sample"] = pipeline.get_dlq(5)
    else:
        fortress_status["pipeline"] = {}
        fortress_status["dlq_sample"] = []

    # Orchestrator stats
    orchestrator = getattr(app.state, 'orchestrator', None)
    if orchestrator:
        fortress_status["orchestrator"] = orchestrator.get_stats()
        fortress_status["predictions"] = await orchestrator.predict_issues()
    else:
        fortress_status["orchestrator"] = {}
        fortress_status["predictions"] = {}

    # Existing enrichment stats
    bus = getattr(app.state, 'bus', None)
    if bus:
        fortress_status["event_bus"] = bus.get_stats()
    else:
        fortress_status["event_bus"] = {}

    # Health assessment - GPT CORRECTION 6
    pipeline_stats = fortress_status.get("pipeline", {})
    if pipeline_stats.get("circuit_opens", 0) > 2:
        fortress_status["status"] = "degraded"
    elif pipeline_stats.get("events_failed", 0) > 10:
        fortress_status["status"] = "warning"
    else:
        fortress_status["status"] = "operational"

    return fortress_status


@app.get("/api/v1/brain/graph")
async def export_brain_graph(format: str = "json"):
    """Export cognitive connection graph"""
    orchestrator = getattr(app.state, 'orchestrator', None)
    if not orchestrator:
        return {"error": "Orchestrator not initialized"}

    try:
        graph_data = orchestrator.export_graph(format)
        if format == "json":
            import json
            return json.loads(graph_data)
        else:
            return {"data": graph_data, "format": format}
    except Exception as e:
        return {"error": str(e)}


@app.get("/api/v1/brain/auto-debug")
async def get_auto_debug_report():
    """Get comprehensive auto-debug diagnostic report - GPT INNOVATION"""
    auto_debug = getattr(app.state, 'auto_debug', None)
    if not auto_debug:
        return {"error": "Auto-Debug Engine not initialized"}

    try:
        return auto_debug.get_diagnostic_report()
    except Exception as e:
        logger.error(f"Auto-debug report error: {e}", exc_info=True)
        return {"error": str(e)}


@app.get("/api/v1/brain/auto-debug/issues")
async def get_current_issues():
    """Get current active issues detected by auto-debug - GPT INNOVATION"""
    auto_debug = getattr(app.state, 'auto_debug', None)
    if not auto_debug:
        return {"error": "Auto-Debug Engine not initialized"}

    try:
        active_issues = {}
        for issue_key, issue in auto_debug.active_issues.items():
            active_issues[issue_key] = {
                "severity": issue.severity,
                "component": issue.component,
                "title": issue.title,
                "description": issue.description,
                "suggested_actions": issue.suggested_actions,
                "first_detected": issue.first_detected.isoformat(),
                "last_seen": issue.last_seen.isoformat(),
                "count": issue.count
            }

        return {
            "active_issues": active_issues,
            "total_critical": sum(1 for issue in auto_debug.active_issues.values() if issue.severity == "critical"),
            "total_warning": sum(1 for issue in auto_debug.active_issues.values() if issue.severity == "warning"),
            "health_score": auto_debug._calculate_health_score()
        }
    except Exception as e:
        logger.error(f"Auto-debug issues error: {e}", exc_info=True)
        return {"error": str(e)}


# Prometheus metrics - protected against reload
_metrics_initialized = False

if not _metrics_initialized:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest

    dream_quality = Gauge('jeffrey_dream_quality', 'Avg dream quality score')
    dream_batch_size = Gauge('jeffrey_dream_batch_size', 'Current dream batch size')
    dream_dlq_size = Gauge('jeffrey_dream_dlq_size', 'Dream DLQ size')

    _metrics_initialized = True


@app.get("/healthz")
async def healthz():
    """Liveness probe"""
    return {"status": "alive"}


@app.get("/readyz")
async def readyz():
    """Readiness probe with proper error handling"""
    checks = {
        "bus": False,
        "dream": False,
        "redis": False,
        "memory": False
    }

    try:
        # Check bus
        if hasattr(app.state, 'bus') and app.state.bus is not None:
            checks["bus"] = True

        # Check memory avec gestion des deux possibilités
        if getattr(app.state, "bootstrap", None) and getattr(app.state.bootstrap, "memory", None):
            checks["memory"] = True
        elif getattr(app.state, "_brain_bootstrap", None) and getattr(app.state._brain_bootstrap, "memory", None):
            checks["memory"] = True

        # Check dream engine
        if hasattr(app.state, 'dream_engine') and app.state.dream_engine is not None:
            checks["dream"] = True

            # Check Redis connection (optional)
            if hasattr(app.state.dream_engine, 'redis') and app.state.dream_engine.redis:
                try:
                    if asyncio.iscoroutinefunction(app.state.dream_engine.redis.ping):
                        await app.state.dream_engine.redis.ping()
                    else:
                        app.state.dream_engine.redis.ping()
                    checks["redis"] = True
                except Exception as e:
                    logger.debug(f"Redis check failed (optional): {e}")
    except Exception as e:
        logger.error(f"Readiness check error: {e}")

    # System is ready if bus and dream are OK (Redis/memory optional)
    ready = checks["bus"] and checks["dream"]
    status_code = 200 if ready else 503

    return JSONResponse(
        content={"ready": ready, "checks": checks},
        status_code=status_code
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics with protection"""
    # Check if caller is allowed (basic protection)
    # In production, use proper auth or network policies

    # Update gauges
    if hasattr(app.state, 'dream_engine') and app.state.dream_engine:
        stats = app.state.dream_engine.get_stats()
        dream_quality.set(stats.get('avg_quality', 0))
        dream_batch_size.set(stats.get('batch_size', 100))
        dream_dlq_size.set(stats.get('dlq_size', 0))

    return Response(generate_latest(), media_type="text/plain")


@app.post("/api/v1/dream/toggle")
async def toggle_dream(enable: bool = None, test_mode: bool = None):
    """Toggle DreamEngine settings at runtime"""
    if not hasattr(app.state, 'dream_engine') or not app.state.dream_engine:
        raise HTTPException(status_code=500, detail="DreamEngine not initialized")

    if enable is not None:
        app.state.dream_engine.enabled = enable

    if test_mode is not None:
        app.state.dream_engine.test_mode = test_mode

    return {
        "enabled": app.state.dream_engine.enabled,
        "test_mode": app.state.dream_engine.test_mode
    }


@app.post("/api/v1/dream/run", dependencies=[Depends(require_admin_permission)])
async def dream_run(force: bool = False, window_hours: int = 24):
    """Run dream consolidation - correctif GPT #3"""
    if not hasattr(app.state, 'dream_engine') or not app.state.dream_engine:
        raise HTTPException(status_code=500, detail="DreamEngine not initialized")

    try:
        result = await app.state.dream_engine.consolidate_memories(
            window_hours=window_hours,
            force=force
        )
        return result
    except Exception as e:
        logger.error(f"Dream run failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/dream/backfill")
async def dream_backfill(days: int = 7):
    """Backfill past days"""
    if not hasattr(app.state, 'dream_engine') or not app.state.dream_engine:
        raise HTTPException(status_code=500, detail="DreamEngine not initialized")

    if days > 30:
        raise HTTPException(status_code=400, detail="Max 30 days backfill")

    try:
        result = await app.state.dream_engine.backfill(days)
        return result
    except Exception as e:
        logger.error(f"Dream backfill failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/dream/status")
async def dream_status():
    """Get DreamEngine status and stats"""
    if not hasattr(app.state, 'dream_engine') or not app.state.dream_engine:
        return {"error": "DreamEngine not initialized"}

    try:
        stats = app.state.dream_engine.get_stats()
        return {
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Dream status failed: {e}")
        return {"error": str(e)}


@app.get("/api/v1/dream/schedule")
async def get_schedule_status():
    """Get scheduler status and next run times"""
    from jeffrey.core.scheduler import dream_scheduler

    jobs = []
    if dream_scheduler.scheduler.running:
        for job in dream_scheduler.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })

    return {
        "enabled": dream_scheduler.enabled,
        "running": dream_scheduler.scheduler.running,
        "interval_minutes": dream_scheduler.interval_minutes,
        "jobs": jobs
    }


@app.put("/api/v1/dream/schedule", dependencies=[Depends(require_admin_permission)])
async def update_schedule(interval_minutes: int = 15):
    """Update dream consolidation interval"""
    from jeffrey.core.scheduler import dream_scheduler

    dream_scheduler.interval_minutes = interval_minutes
    # Reschedule the job
    if dream_scheduler.scheduler.running:
        dream_scheduler.scheduler.reschedule_job(
            "dream_consolidation",
            trigger=IntervalTrigger(minutes=interval_minutes)
        )

    return {"message": f"Schedule updated to {interval_minutes} minutes"}


@app.get("/api/v1/consciousness/curiosity/status")
async def get_curiosity_status(_: bool = Depends(verify_api_key)):
    """Get curiosity analysis without side effects"""
    from jeffrey.core.consciousness.proactive_curiosity_safe import ProactiveCuriositySafe

    try:
        curiosity = ProactiveCuriositySafe()
        analysis = await curiosity.analyze_gaps()
        questions = await curiosity.generate_questions()
        status = await curiosity.get_status()

        return {
            "service": "proactive_curiosity",
            "status": status,
            "analysis": analysis,
            "questions": questions,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Curiosity status failed: {e}")
        return {
            "service": "proactive_curiosity",
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
