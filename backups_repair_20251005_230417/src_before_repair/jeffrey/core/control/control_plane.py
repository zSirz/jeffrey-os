"""
Control Plane pour monitoring, health checks et métriques
VERSION CORRIGÉE : Tolérante aux failures Redis et Guardians
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any

import psutil
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

# Nouveaux imports pour Phase 1
from pydantic import BaseModel

from ..loaders.secure_module_loader import SecureModuleLoader
from ..neuralbus.bus_v2 import NeuralBusV2

logger = logging.getLogger(__name__)

# Métriques Prometheus
request_count = Counter("jeffrey_requests_total", "Total requests")
request_duration = Histogram("jeffrey_request_duration_seconds", "Request duration")
active_connections = Gauge("jeffrey_active_connections", "Active connections")
system_health = Gauge("jeffrey_system_health", "System health score")

app = FastAPI(title="Jeffrey OS Control Plane", version="2.0.0")

# Instances globales
cache_guardian: object | None = None
anti_replay: object | None = None
guardians_hub: object | None = None
memory: object | None = None

# Nouveaux composants Phase 1
bus_v2: NeuralBusV2 | None = None
loader: SecureModuleLoader | None = None
cognitive_core: Any | None = None


@app.on_event("startup")
async def startup_event():
    """Initialise les composants au démarrage - TOLÉRANT AUX FAILURES"""
    global cache_guardian, anti_replay, guardians_hub

    logger.info("🚀 Starting Jeffrey OS Control Plane...")
    mode = os.getenv("SECURITY_MODE", "dev")
    logger.info(f"Mode: {mode.upper()}")

    # Cache Guardian - toujours requis
    try:
        from ..security.cache_guardian import CacheGuardian

        cache_guardian = CacheGuardian()
        await cache_guardian.start()
        logger.info("✅ Cache Guardian initialized")
    except Exception as e:
        logger.error(f"Cache Guardian failed: {e}")
        if mode == "prod":
            raise

    # Anti-Replay - optionnel en DEV si Redis down
    try:
        from ..security.anti_replay import AntiReplaySystem

        anti_replay = AntiReplaySystem()
        await anti_replay.start()
        logger.info("✅ Anti-Replay initialized")
    except Exception as e:
        if mode == "dev":
            logger.warning(f"⚠️ Anti-Replay disabled (DEV) - Redis not available: {e}")
            anti_replay = None
        else:
            raise

    # Guardians Hub - en tâche async pour ne pas bloquer
    try:
        from ..guardians.guardians_hub import GuardiansHub

        guardians_hub = GuardiansHub()
        # Lance l'init en arrière-plan
        asyncio.create_task(guardians_hub.start())
        logger.info("✅ Guardians Hub initialization started")
    except Exception as e:
        logger.warning(f"⚠️ Guardians Hub initialization deferred: {e}")
        guardians_hub = None

    # === PHASE 1 : NeuralBus V2 + Loader + Cognitive Core ===
    global bus_v2, loader, cognitive_core

    try:
        # 1. Démarrer le NeuralBus V2
        bus_v2 = NeuralBusV2(maxsize=1000, workers=2)
        await bus_v2.start()
        logger.info("✅ NeuralBus V2 started")

        # 2. Charger la configuration des modules
        loader = SecureModuleLoader("config/modules.yaml")
        loader.load_config()
        loaded_modules = await loader.load_all_enabled()
        logger.info(f"✅ Loaded {len(loaded_modules)} modules")

        # 3. Initialiser le Cognitive Core
        cognitive_core = loaded_modules.get("cognitive_core")
        if cognitive_core:
            # Passer les guardians si disponibles
            await cognitive_core.initialize(bus_v2, loaded_modules)
            logger.info("✅ Cognitive Core initialized")
        else:
            logger.warning("⚠️ Cognitive Core not found in modules")

    except Exception as e:
        logger.error(f"Failed to initialize Phase 1 components: {e}")
        # Le système peut continuer sans ces composants (fallback)

    logger.info("✅ Control Plane ready")


@app.get("/")
async def root():
    """Page d'accueil"""
    return JSONResponse(
        {
            "name": "Jeffrey OS",
            "version": "2.0.0",
            "status": "running",
            "endpoints": ["/health", "/ready", "/metrics", "/status"],
        }
    )


@app.get("/health")
async def health():
    """Health check basique"""
    request_count.inc()
    return JSONResponse(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.0.0",
            "mode": os.getenv("SECURITY_MODE", "dev"),
        }
    )


@app.get("/ready")
async def ready():
    """Readiness check - vérifie que les composants critiques sont prêts"""
    mode = os.getenv("SECURITY_MODE", "dev")
    checks = {}

    # Cache Guardian - critique
    if cache_guardian:
        try:
            checks["cache_guardian"] = cache_guardian.get_status()
        except:
            checks["cache_guardian"] = {"error": "failed to get status"}
    else:
        checks["cache_guardian"] = {"error": "not initialized"}

    # Anti-Replay - optionnel en DEV
    if anti_replay:
        try:
            checks["anti_replay"] = anti_replay.get_status()
        except:
            checks["anti_replay"] = {"error": "failed to get status"}
    else:
        checks["anti_replay"] = {"disabled": True, "reason": "Redis not available (DEV)"}

    # Guardians Hub - optionnel
    if guardians_hub:
        try:
            checks["guardians_hub"] = guardians_hub.get_status()
        except:
            checks["guardians_hub"] = {"status": "initializing"}
    else:
        checks["guardians_hub"] = {"status": "not loaded"}

    # Calculer le score
    components_ok = sum(1 for c in checks.values() if c and not c.get("error") and not c.get("disabled"))
    total_components = len(checks)
    health_score = components_ok / total_components if total_components > 0 else 0

    system_health.set(health_score)

    # En DEV, on est moins strict
    if mode == "dev":
        is_ready = health_score >= 0.33  # Au moins 1/3 des composants
    else:
        is_ready = health_score >= 0.8  # 80% en prod

    return JSONResponse(
        {
            "ready": is_ready,
            "health_score": health_score,
            "mode": mode,
            "components": checks,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.get("/metrics")
async def metrics():
    """Endpoint Prometheus"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/status")
async def status():
    """Status détaillé du système"""
    return JSONResponse(
        {
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage("/").percent,
                "process_count": len(psutil.pids()),
            },
            "security": {
                "cache_guardian": cache_guardian.get_status() if cache_guardian else None,
                "anti_replay": anti_replay.get_status() if anti_replay else None,
            },
            # Composants Phase 1
            "bus_v2": bus_v2.get_stats() if bus_v2 else None,
            "cognitive_core": cognitive_core.get_status() if cognitive_core else None,
            "modules_loaded": list(loader.loaded.keys()) if loader else [],
            # Composants existants
            "guardians": guardians_hub.get_status() if guardians_hub else None,
            "memory": memory.get_stats() if memory and hasattr(memory, "get_stats") else None,
            "mode": os.getenv("SECURITY_MODE", "dev"),
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.post("/test/security")
async def test_security(client_id: str = "test_client"):
    """Endpoint de test pour valider la chaîne de sécurité"""
    results = {}

    # Test Anti-Replay si disponible
    if anti_replay:
        test_request = anti_replay.generate_secure_request(client_id, {"action": "test", "data": "test_data"})
        valid, error = await anti_replay.validate_request(test_request)
        results["anti_replay"] = {"valid": valid, "error": error}
    else:
        results["anti_replay"] = {"skipped": True, "reason": "not available"}

    # Test Cache Guardian
    if cache_guardian:
        cache_result = await cache_guardian.check_access(
            "test_key", "GET", {"client_id": client_id, "requests_per_minute": 10}
        )
        results["cache_guardian"] = {"allowed": cache_result[0], "reason": cache_result[1]}
    else:
        results["cache_guardian"] = {"skipped": True}

    return JSONResponse({"results": results, "timestamp": datetime.now().isoformat()})


# Modèle Pydantic pour la requête chat
class ChatRequest(BaseModel):
    user_id: str = "anonymous"
    text: str


@app.post("/chat")
async def chat_v2(request: ChatRequest):
    """
    Endpoint de chat utilisant le NeuralBus V2 avec RPC
    Fallback sur l'ancien système si non disponible
    """
    # Essayer le nouveau système
    if bus_v2 and cognitive_core and cognitive_core.initialized:
        try:
            # Utiliser ask() pour une communication synchrone
            response = await bus_v2.ask(
                request_type="user.input",
                data={"user_id": request.user_id, "text": request.text},
                response_type="response.ready",
                timeout=3.0,
            )

            if response:
                data = response.get("data", {})

                # Stocker dans la mémoire si disponible
                if memory:
                    await memory.store_message(request.user_id, "user", request.text)
                    await memory.store_message(request.user_id, "assistant", data.get("text", ""))

                return JSONResponse(
                    {
                        "reply": data.get("text", "..."),
                        "awareness_level": data.get("awareness_level", 0.5),
                        "modules_used": data.get("modules_used", []),
                        "emotion": data.get("emotion"),
                        "memory": data.get("memory"),
                        "mode": data.get("mode"),
                        "processing_ms": data.get("processing_ms"),
                        "alignment_score": data.get("alignment_score"),
                        "cognitive_core": True,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        except Exception as e:
            logger.error(f"Chat V2 error: {e}")

    # Fallback sur l'ancien système ou réponse simple
    if guardians_hub:
        try:
            # Validation éthique
            decision = await guardians_hub.validate_action("chat", {"user_id": request.user_id, "text": request.text})

            if not decision["approved"]:
                return JSONResponse(
                    {
                        "reply": "Je ne peux pas traiter cette demande.",
                        "cognitive_core": False,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
        except Exception as e:
            logger.error(f"Guardian validation error: {e}")

    # Réponse fallback simple
    return JSONResponse(
        {
            "reply": f"[Fallback] J'ai reçu : {request.text[:100]}",
            "cognitive_core": False,
            "timestamp": datetime.now().isoformat(),
        }
    )


@app.on_event("shutdown")
async def shutdown_event():
    """Arrêt propre du système"""
    global bus_v2

    if bus_v2:
        await bus_v2.stop()
        logger.info("NeuralBus V2 stopped")

    # Autres arrêts si nécessaire
    logger.info("Jeffrey OS shutdown complete")
