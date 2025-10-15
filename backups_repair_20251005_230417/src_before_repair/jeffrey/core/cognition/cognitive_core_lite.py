"""
Cognitive Core Enhanced - Version finale avec fédérations production-ready
Utilise toutes les améliorations de l'équipe
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any

from ..emotions.emotion_orchestrator_v2 import EmotionOrchestratorV2
from ..loaders.secure_module_loader import SecureModuleLoader
from ..memory.memory_federation_v2 import MemoryFederationV2
from ..utils.async_helpers import LatencyBudget

logger = logging.getLogger(__name__)


class CognitiveCore:
    """
    Core cognitif production-ready avec fédérations
    """

    def __init__(self, loader: SecureModuleLoader = None):
        self.loader = loader
        self.initialized = False
        self.bus = None
        self.guardians = None
        self.modules = {}

        # Fédérations - créer avec loader si fourni, sinon créer plus tard
        if loader:
            self.memory_federation = MemoryFederationV2(loader)
            self.emotion_orchestrator = EmotionOrchestratorV2(loader)
        else:
            self.memory_federation = None
            self.emotion_orchestrator = None

        # État cognitif
        self.state = {
            "awareness_level": 0.5,
            "last_activity": None,
            "messages_processed": 0,
            "mode": "normal",  # normal | degraded | minimal
            "active_memory_layers": 0,
            "active_emotion_categories": 0,
            "alignment_score": 0.5,
        }

        # Métriques
        self.metrics = {"total_latency_ms": [], "errors": 0, "degraded_activations": 0}

    async def initialize(self, bus: Any, modules: dict[str, Any] = None):
        """Initialise avec traçabilité"""
        trace_id = str(uuid.uuid4())

        logger.info(f"🚀 Initializing Cognitive Core (trace: {trace_id})")

        self.bus = bus
        if modules:
            self.modules = modules
            self.guardians = modules.get("guardians_hub")

            # Si pas de loader fourni au __init__, essayer de le récupérer des modules
            if not self.loader and "loader" in modules:
                self.loader = modules["loader"]
                self.memory_federation = MemoryFederationV2(self.loader)
                self.emotion_orchestrator = EmotionOrchestratorV2(self.loader)

        # Si toujours pas de fédérations, créer des versions minimales
        if not self.memory_federation:
            from ..loaders.secure_module_loader import SecureModuleLoader

            self.loader = SecureModuleLoader()
            self.memory_federation = MemoryFederationV2(self.loader)
            self.emotion_orchestrator = EmotionOrchestratorV2(self.loader)

        # Initialiser les fédérations
        try:
            await self.memory_federation.initialize(bus, trace_id)
            self.state["active_memory_layers"] = sum(1 for l in self.memory_federation.layers.values() if l.initialized)
        except Exception as e:
            logger.error(f"Memory federation init failed: {e}")

        try:
            await self.emotion_orchestrator.initialize(bus, trace_id)
            self.state["active_emotion_categories"] = sum(
                1 for c in self.emotion_orchestrator.categories.values() if c.initialized
            )
        except Exception as e:
            logger.error(f"Emotion orchestrator init failed: {e}")

        # Vérifier l'état
        if self.state["active_memory_layers"] == 0 and self.state["active_emotion_categories"] == 0:
            logger.error("No modules loaded, entering minimal mode")
            self.state["mode"] = "minimal"
        elif self.state["active_memory_layers"] < 2 or self.state["active_emotion_categories"] < 2:
            logger.warning("Limited modules loaded, entering degraded mode")
            self.state["mode"] = "degraded"

        # S'abonner aux événements
        await self.bus.subscribe("user.input", self._handle_user_input)
        await self.bus.subscribe("system.mode.change", self._handle_mode_change)

        self.initialized = True

        logger.info(
            f"""
        ✅ Cognitive Core initialized (trace: {trace_id}):
           - Mode: {self.state['mode']}
           - Memory: {self.state['active_memory_layers']} layers
           - Emotions: {self.state['active_emotion_categories']} categories
           - Total modules: {
                self.memory_federation.stats['global'].get('loaded_modules', 0)
                + sum(len(c.modules) for c in self.emotion_orchestrator.categories.values() if c.initialized)
            }
        """
        )

    async def _handle_user_input(self, envelope: dict):
        """
        Pipeline cognitif principal avec tous les modules
        """
        start_time = time.perf_counter()
        trace_id = envelope.get("trace_id") or str(uuid.uuid4())

        # Contexte
        data = envelope.get("data", {})
        text = data.get("text", "")
        user_id = data.get("user_id", "anonymous")
        correlation_id = envelope.get("correlation_id")

        logger.info(f"Processing input (trace: {trace_id}, mode: {self.state['mode']})")

        try:
            # Budget total : 1000ms
            budget = LatencyBudget(1000)

            # === PIPELINE SELON LE MODE ===

            if self.state["mode"] == "minimal":
                # Mode minimal : réponse basique
                response = await self._minimal_response(text)
                emotion = {"valence": 0, "arousal": 0, "mood": "neutral"}
                memories = []

            elif self.state["mode"] == "degraded":
                # Mode dégradé : fonctions essentielles seulement

                # Analyse émotionnelle rapide (80ms)
                emotion = {}
                if budget.has_budget(80):
                    emotion = await self.emotion_orchestrator.analyze_fast(text)

                # Mémoire rapide (50ms)
                memories = []
                if budget.has_budget(50):
                    memories = await self.memory_federation.recall_fast(user_id, 3)

                # Réponse simple
                response = await self._process_degraded(text, user_id, emotion, memories)

            else:
                # Mode normal : pipeline complet

                # 1. Validation éthique (50ms)
                if self.guardians and budget.has_budget(50):
                    validation = await self._validate_with_guardians(text, user_id, trace_id)
                    if not validation.get("approved", True):
                        await self._send_response(
                            user_id,
                            validation.get("reason", "Je ne peux pas traiter cette demande."),
                            correlation_id,
                            trace_id,
                        )
                        return

                # 2. Paralléliser analyse et rappel
                emotion_task = None
                memory_task = None

                if budget.has_budget(350):
                    emotion_task = asyncio.create_task(self.emotion_orchestrator.analyze_deep(text))

                if budget.has_budget(400):
                    memory_task = asyncio.create_task(self.memory_federation.recall_deep(user_id, 10))

                # Attendre les résultats
                emotion = await emotion_task if emotion_task else {}
                memories = await memory_task if memory_task else []

                # 3. Stocker le message (async, non-bloquant)
                asyncio.create_task(
                    self.memory_federation.store_to_relevant(
                        user_id, "user", text, {"emotion": emotion, "trace_id": trace_id}
                    )
                )

                # 4. Générer la réponse enrichie
                response = await self._process_full(text, user_id, emotion, memories)

                # 5. Mettre à jour l'état émotionnel global
                await self.emotion_orchestrator.update_global_state(emotion)

                # 6. Calculer l'alignement (symbiose)
                self.state["alignment_score"] = self.emotion_orchestrator.get_alignment_score(emotion)

            # === FIN DU PIPELINE ===

            # Stocker la réponse
            asyncio.create_task(
                self.memory_federation.store_to_relevant(
                    user_id,
                    "assistant",
                    response,
                    {"emotion": self.emotion_orchestrator.global_state, "trace_id": trace_id},
                )
            )

            # Métriques
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.metrics["total_latency_ms"].append(latency_ms)
            if len(self.metrics["total_latency_ms"]) > 100:
                self.metrics["total_latency_ms"] = self.metrics["total_latency_ms"][-100:]

            self.state["messages_processed"] += 1
            self.state["last_activity"] = datetime.now().isoformat()

            # Envoyer la réponse
            await self._send_enhanced_response(
                user_id, response, correlation_id, trace_id, emotion, memories, latency_ms
            )

            logger.info(f"Processed in {latency_ms:.1f}ms (trace: {trace_id})")

        except Exception as e:
            logger.error(f"Pipeline error (trace: {trace_id}): {e}", exc_info=True)
            self.metrics["errors"] += 1

            # Dégradation automatique
            if self.metrics["errors"] > 5:
                await self._enter_degraded_mode()

            # Réponse d'erreur
            await self._send_response(
                user_id,
                "Je rencontre une difficulté, mais je reste disponible.",
                correlation_id,
                trace_id,
            )

    async def _minimal_response(self, text: str) -> str:
        """Réponse en mode minimal"""
        if "bonjour" in text.lower():
            return "Bonjour ! Je fonctionne en mode minimal."
        elif "aide" in text.lower():
            return "Je peux vous aider de manière basique."
        else:
            return f"J'ai compris : {text[:50]}..."

    async def _process_degraded(self, text: str, user_id: str, emotion: dict, memories: list) -> str:
        """Traitement en mode dégradé"""
        mood = emotion.get("mood", "neutral")

        response_parts = []

        if mood in ["sad", "anxious"]:
            response_parts.append("Je comprends que ce n'est pas facile.")
        elif mood in ["happy", "excited"]:
            response_parts.append("Je ressens votre enthousiasme !")

        if "bonjour" in text.lower():
            response_parts.append("Bonjour ! (Mode limité actif)")
        else:
            response_parts.append(f"Message reçu : {text[:50]}...")

        return " ".join(response_parts)

    async def _process_full(self, text: str, user_id: str, emotion: dict, memories: list) -> str:
        """Traitement complet avec tous les modules"""

        # Construire le contexte enrichi
        mood = emotion.get("mood", "neutral")
        valence = emotion.get("valence", 0)
        confidence = emotion.get("confidence", 0.5)

        # Préfixe émotionnel adaptatif
        emotion_prefix = ""
        if confidence > 0.7:  # Confiant dans l'analyse
            if mood == "happy" or valence > 0.5:
                emotion_prefix = "Je partage votre joie ! "
            elif mood == "sad" or valence < -0.5:
                emotion_prefix = "Je comprends votre tristesse. "
            elif mood == "excited":
                emotion_prefix = "Quelle énergie incroyable ! "
            elif mood == "anxious":
                emotion_prefix = "Je sens votre préoccupation. "

        # Contexte mémoire enrichi
        memory_context = ""
        if memories:
            # Trouver les souvenirs pertinents
            user_memories = [m for m in memories if m.get("role") == "user"]

            if user_memories:
                # Mentionner les sources
                sources = set(m.get("_source", "") for m in user_memories if m.get("_source"))
                if len(sources) > 2:
                    memory_context = f" (J'ai consulté {len(sources)} types de mémoire)"

                # Dernier sujet
                last = user_memories[-1].get("text", "")[:50]
                if last and last not in text:
                    memory_context += f" Je me souviens que nous parlions de : {last}..."

        # Construire la réponse
        response_parts = []

        # Salutations contextuelles
        if "bonjour" in text.lower():
            greeting = f"{emotion_prefix}Bonjour !"

            # Ajouter l'état du système
            if self.state["alignment_score"] > 0.7:
                greeting += " Nous sommes vraiment en phase aujourd'hui !"

            response_parts.append(greeting)

        # État système
        elif "comment" in text.lower() and ("vas" in text.lower() or "va" in text.lower()):
            mem_stats = self.memory_federation.get_stats()
            emo_stats = self.emotion_orchestrator.get_stats()

            response_parts.append(
                f"{emotion_prefix}Je fonctionne merveilleusement ! "
                f"J'ai {self.state['active_memory_layers']} couches de mémoire actives "
                f"et {self.state['active_emotion_categories']} catégories émotionnelles. "
                f"Mon état est '{mood}' avec un score d'alignement de {self.state['alignment_score']:.2f}. "
                f"J'ai traité {self.state['messages_processed']} messages avec vous."
            )

        # Mémoire
        elif "souviens" in text.lower() or "rappelle" in text.lower():
            if memories:
                response_parts.append(f"{emotion_prefix}Oui, je me souviens !{memory_context}")

                # Détailler un souvenir
                if user_memories:
                    recent = user_memories[-1].get("text", "")
                    response_parts.append(f"Vous aviez dit : '{recent[:100]}...'")
            else:
                response_parts.append("Nous commençons tout juste notre conversation.")

        # Réponse générique enrichie
        else:
            response_parts.append(f"{emotion_prefix}J'ai bien compris votre message.{memory_context}")

            # Ajouter une touche personnelle selon l'alignement
            if self.state["alignment_score"] > 0.8:
                response_parts.append("Nous nous comprenons vraiment bien !")
            elif self.state["alignment_score"] < 0.3:
                response_parts.append("J'ajuste ma compréhension à votre état.")

        return " ".join(response_parts)

    async def _validate_with_guardians(self, text: str, user_id: str, trace_id: str) -> dict:
        """Validation éthique avec timeout"""
        if not self.guardians or not hasattr(self.guardians, "validate_action"):
            return {"approved": True}

        try:
            from ..utils.async_helpers import asyncify

            return await asyncify(
                self.guardians.validate_action,
                "chat",
                {"text": text, "user_id": user_id, "trace_id": trace_id},
                timeout=0.05,  # 50ms max
            ) or {"approved": True}
        except:
            return {"approved": True}

    async def _send_enhanced_response(
        self,
        user_id: str,
        text: str,
        correlation_id: str | None,
        trace_id: str,
        emotion: dict,
        memories: list,
        latency_ms: float,
    ):
        """Envoie une réponse enrichie avec toutes les métadonnées"""

        # Construire les modules utilisés
        modules_used = []

        # Ajouter les couches mémoire actives
        for layer in self.memory_federation.layers.values():
            if layer.initialized:
                modules_used.append(f"Memory:{layer.name}")

        # Ajouter les catégories émotions actives
        for category in self.emotion_orchestrator.categories.values():
            if category.initialized:
                modules_used.append(f"Emotion:{category.name}")

        await self.bus.publish(
            {
                "type": "response.ready",
                "data": {
                    "user_id": user_id,
                    "text": text,
                    "mode": self.state["mode"],
                    "awareness_level": self.state["awareness_level"],
                    "alignment_score": self.state["alignment_score"],
                    "emotion": {**emotion, "global_state": self.emotion_orchestrator.global_state},
                    "memory": {
                        "recalled": len(memories),
                        "layers_active": self.state["active_memory_layers"],
                    },
                    "modules_used": modules_used[:10],  # Limiter
                    "processing_ms": latency_ms,
                },
                "correlation_id": correlation_id,
                "trace_id": trace_id,
                "meta": {"timestamp": datetime.now().isoformat()},
            }
        )

    async def _send_response(self, user_id: str, text: str, correlation_id: str | None, trace_id: str):
        """Envoie une réponse simple"""
        await self.bus.publish(
            {
                "type": "response.ready",
                "data": {"user_id": user_id, "text": text, "mode": self.state["mode"]},
                "correlation_id": correlation_id,
                "trace_id": trace_id,
            }
        )

    async def _enter_degraded_mode(self):
        """Entre en mode dégradé"""
        if self.state["mode"] != "degraded":
            logger.warning("Entering degraded mode due to errors")
            self.state["mode"] = "degraded"
            self.metrics["degraded_activations"] += 1

            # Publier l'événement
            await self.bus.publish({"type": "system.mode.changed", "data": {"mode": "degraded", "reason": "errors"}})

            # Auto-recovery après 60s
            asyncio.create_task(self._auto_recover())

    async def _auto_recover(self):
        """Tente de récupérer automatiquement"""
        await asyncio.sleep(60)

        if self.metrics["errors"] < 2:  # Peu d'erreurs récentes
            logger.info("Attempting auto-recovery to normal mode")
            self.state["mode"] = "normal"
            self.metrics["errors"] = 0

            await self.bus.publish(
                {
                    "type": "system.mode.changed",
                    "data": {"mode": "normal", "reason": "auto-recovery"},
                }
            )

    async def _handle_mode_change(self, envelope):
        """Gère les changements de mode"""
        new_mode = envelope.get("data", {}).get("mode")
        if new_mode in ["normal", "degraded", "minimal"]:
            self.state["mode"] = new_mode
            logger.info(f"Mode changed to: {new_mode}")

    def get_status(self) -> dict[str, Any]:
        """Retourne le statut complet"""

        # Calculer les percentiles de latence
        latencies = self.metrics["total_latency_ms"]
        if latencies:
            latencies_sorted = sorted(latencies)
            n = len(latencies_sorted)
            p50 = latencies_sorted[n // 2]
            p95 = latencies_sorted[int(n * 0.95)] if n > 20 else latencies_sorted[-1]
            p99 = latencies_sorted[int(n * 0.99)] if n > 100 else latencies_sorted[-1]
        else:
            p50 = p95 = p99 = 0

        return {
            "initialized": self.initialized,
            "mode": self.state["mode"],
            "state": self.state,
            "modules": list(self.modules.keys()),
            "memory_federation": self.memory_federation.get_stats() if self.memory_federation else {},
            "emotion_orchestrator": self.emotion_orchestrator.get_stats() if self.emotion_orchestrator else {},
            "metrics": {**self.metrics, "latency_p50": p50, "latency_p95": p95, "latency_p99": p99},
            "total_active_modules": (
                sum(len(l.modules) for l in self.memory_federation.layers.values() if l.initialized)
                + sum(len(c.modules) for c in self.emotion_orchestrator.categories.values() if c.initialized)
            )
            if self.memory_federation and self.emotion_orchestrator
            else 0,
        }


# IMPORTANT: Add compatibility alias
CognitiveCoreLite = CognitiveCore
