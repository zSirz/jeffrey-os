"""
Orchestrator Bridge for Jeffrey
Bridges the new plugin system with the existing orchestrator
"""

import asyncio
import logging
import threading
from typing import Any, Dict, List, Optional

from .base_module import ServiceModule
from .contextual_dialogue import ContextualDialogueEngine
from .emotional_core import EmotionalCore
from .event_bus import Event, Events, event_bus
from .plugin_manager import plugin_manager

logger = logging.getLogger(__name__)


# Plugin information for auto-discovery
PLUGIN_INFO = {
    "name": "OrchestratorBridge",
    "version": "1.0.0",
    "description": "Bridge between plugin system and existing orchestrator",
    "dependencies": ["conversation_memory", "memory_enhancer"],
    "config": {
        "auto_start": True,
        "enable_memory": True,
        "enable_emotions": True
    }
}


class OrchestratorBridge(ServiceModule):
    """Bridge to connect plugin system with existing Jeffrey components"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._orchestrator = None  # Will be set by set_orchestrator
        self._emotional_core: Optional[EmotionalCore] = None
        self._conversation_memory = None
        self._memory_enhancer = None
        self._user_identity_manager = None
        self._memory_aligner = None
        self._emotion_adjuster = None

        # Initialiser le moteur de dialogue contextuel
        self._dialogue_engine = ContextualDialogueEngine()
        logger.info("Dialogue contextuel initialisé")

        # Créer un event loop dédié dans un thread séparé
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_event_loop, daemon=True)
        self._thread.start()

    def _run_event_loop(self):
        """Exécute l'event loop dans un thread dédié"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call_async_from_sync(self, coro):
        """Appelle une coroutine de manière thread-safe"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def initialize(self):
        """Initialize the bridge"""
        logger.info("Initializing OrchestratorBridge")

        try:
            # Get memory plugins if available
            self._conversation_memory = plugin_manager.get_plugin("conversation_memory")
            self._memory_enhancer = plugin_manager.get_plugin("memory_enhancer")

            # Get new naturalness plugins
            self._user_identity_manager = plugin_manager.get_plugin("user_identity_manager")
            self._memory_aligner = plugin_manager.get_plugin("memory_aligner")
            self._emotion_adjuster = plugin_manager.get_plugin("emotion_adjuster")

            # Log plugin status
            logger.info(f"UserIdentityManager: {'loaded' if self._user_identity_manager else 'not found'}")
            logger.info(f"MemoryAligner: {'loaded' if self._memory_aligner else 'not found'}")
            logger.info(f"EmotionAdjuster: {'loaded' if self._emotion_adjuster else 'not found'}")

            # Subscribe to events directly since subscribe_to_event is not async
            if event_bus:
                await event_bus.subscribe(Events.USER_MESSAGE, self._on_user_message)
                await event_bus.subscribe(Events.EMOTION_CHANGED, self._on_emotion_changed)
                await event_bus.subscribe('identity_needs_confirmation', self._on_identity_confirmation_needed)
                self._subscribed_events.extend([
                    Events.USER_MESSAGE,
                    Events.EMOTION_CHANGED,
                    'identity_needs_confirmation'
                ])

            self._running = True
            logger.info("OrchestratorBridge initialized")

        except Exception as e:
            logger.error(f"Error initializing bridge: {e}")
            raise

    async def shutdown(self):
        """Arrêter proprement le bridge"""
        logger.info("Shutting down OrchestratorBridge...")

        try:
            # Désabonner des événements de manière asynchrone
            if hasattr(self, 'unsubscribe_all'):
                if asyncio.iscoroutinefunction(self.unsubscribe_all):
                    # Si c'est une coroutine, l'exécuter dans l'event loop
                    if self._loop and self._loop.is_running():
                        # Utiliser run_coroutine_threadsafe sans await
                        future = asyncio.run_coroutine_threadsafe(self.unsubscribe_all(), self._loop)
                        try:
                            # Attendre le résultat avec un timeout
                            future.result(timeout=2.0)
                        except asyncio.TimeoutError:
                            logger.warning("Timeout lors de la désinscription des événements")
                    else:
                        # Si l'event loop n'est pas en cours, exécuter directement
                        await self.unsubscribe_all()
                else:
                    # Si c'est une fonction normale
                    self.unsubscribe_all()

            # Arrêter l'event loop s'il existe
            if hasattr(self, '_loop') and self._loop and self._loop.is_running():
                self._loop.call_soon_threadsafe(self._loop.stop)

            # Attendre que le thread se termine
            if hasattr(self, '_thread') and self._thread and self._thread.is_alive():
                self._thread.join(timeout=2.0)

            self._running = False
            logger.info("OrchestratorBridge shut down successfully")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
            raise

    async def process_message(self, message: str, session_id: str = "default") -> str:
        """Process message avec le dialogue contextuel UNIQUEMENT"""
        try:
            # Log d'entrée
            logger.info(f"📥 Message reçu: '{message}'")

            # Obtenir l'état émotionnel actuel
            emotional_context = {}
            if self._emotional_core:
                try:
                    # Utiliser get_dominant_emotion au lieu de get_current_emotion
                    emotional_state = self._emotional_core.get_dominant_emotion()
                    emotional_context["emotion"] = emotional_state
                    # Récupérer l'intensité depuis l'état émotionnel
                    emotional_states = self._emotional_core.get_emotional_state()
                    emotional_context["intensity"] = emotional_states.get(emotional_state, 0.5)
                    logger.info(f"État émotionnel actuel: {emotional_state}")
                except Exception as e:
                    logger.warning(f"Impossible d'obtenir l'état émotionnel: {e}")
                    emotional_context["emotion"] = "neutral"

            # Vérifier le dialogue engine
            if not self._dialogue_engine:
                logger.error("❌ Pas de dialogue engine!")
                return "🌿 Jeffrey : [Système non initialisé]"

            # Appel au dialogue contextuel en passant l'émotion courante (ou "neutral" par défaut)
            # generate_response est maintenant synchrone
            response = self._dialogue_engine.generate_response(
                message,
                emotional_context.get("emotion", "neutral")
            )

            # Log de sortie COMPLET
            logger.info(f"📤 Réponse générée: '{response}'")

            # Mettre à jour l'état émotionnel après la réponse
            if self._emotional_core and response:
                try:
                    # process_interaction prend seulement 2 arguments
                    self._emotional_core.process_interaction(message, response)
                    logger.info("État émotionnel mis à jour après interaction")
                except Exception as e:
                    logger.warning(f"Erreur lors de la mise à jour émotionnelle: {e}")

            return response

        except Exception as e:
            logger.error(f"❌ Erreur process_message: {e}")
            import traceback
            traceback.print_exc()
            return "🌿 Jeffrey : [Erreur de traitement]"

    async def _process_message_async(self, message: str, session_id: str = "default") -> Dict[str, Any]:
        """Version asynchrone de process_message"""
        await self.before_request({"message": message, "session_id": session_id})

        try:
            # Emit user message event
            await event_bus.emit(Event(
                Events.USER_MESSAGE,
                {
                    "content": message,
                    "session_id": session_id,
                    "timestamp": Event(Events.USER_MESSAGE, {}, "").timestamp
                },
                source="orchestrator_bridge"
            ))

            # Get current emotional state
            emotion_state = None
            if self._emotional_core:
                emotion_state = self._emotional_core.get_emotional_state()

            # Get conversation context
            context = []
            if self._conversation_memory and self.get_config("enable_memory", True):
                context = await self._conversation_memory.get_context(session_id)

            # Get memory context BEFORE processing
            memory_context = await self._get_memory_context(message, session_id)

            # Enrich the message with context
            enriched_message = self._enrich_with_context(message, memory_context)

            # Process through orchestrator with new dialogue engine
            response = {"content": "Je suis désolé, je ne suis pas encore complètement configuré."}

            if self._orchestrator:
                # Utiliser le moteur de dialogue pour générer une réponse contextuelle
                # generate_response est maintenant synchrone
                dialogue_response = self._dialogue_engine.generate_response(
                    enriched_message,
                    emotion_state.get("primary_emotion") if emotion_state else "neutral"
                )

                # Utiliser _call_async_from_sync pour appeler process de manière thread-safe
                orchestrator_response = self._call_async_from_sync(
                    asyncio.to_thread(self._orchestrator.process, dialogue_response)
                )

                response = {
                    "content": orchestrator_response,
                    "metadata": {
                        "emotion": emotion_state.get("primary_emotion", "neutral") if emotion_state else "neutral",
                        "context_used": bool(memory_context.get("recent_conversation") or memory_context.get("emotional_memories")),
                        "dialogue_context": self._dialogue_engine.get_conversation_context(),
                        "relationship_level": self._dialogue_engine.get_relationship_level()
                    },
                    "enriched_input": enriched_message if len(enriched_message) < 1000 else enriched_message[:1000] + "...",
                    "context_info": {
                        "has_history": bool(memory_context.get("recent_conversation")),
                        "has_emotions": bool(memory_context.get("emotional_memories")),
                        "memories_count": len(memory_context.get("recent_conversation", [])) + len(memory_context.get("emotional_memories", [])),
                        "user_memory": self._dialogue_engine.await memory.retrieve(MemoryQuery(memory_types=[MemoryType.CONTEXTUAL]))
                    }
                }

            # Emit assistant response event
            await event_bus.emit(Event(
                Events.ASSISTANT_RESPONSE,
                {
                    "content": response.get("content", ""),
                    "session_id": session_id,
                    "emotion": emotion_state.get("primary_emotion", "neutral") if emotion_state else "neutral",
                    "metadata": response.get("metadata", {})
                },
                source="orchestrator_bridge"
            ))

            await self.after_request(
                {"message": message, "session_id": session_id},
                response
            )

            return response

        except Exception as e:
            await self.on_error(e, {"message": message, "session_id": session_id})
            raise

    def set_orchestrator(self, orchestrator):
        """Set the orchestrator instance"""
        self._orchestrator = orchestrator
        # Pass plugin_manager to orchestrator so it can access plugins
        if hasattr(orchestrator, 'set_plugin_manager'):
            orchestrator.set_plugin_manager(plugin_manager)
        elif hasattr(orchestrator, 'plugin_manager'):
            orchestrator.plugin_manager = plugin_manager
        logger.info("Orchestrator connected to bridge")

    def set_emotional_core(self, emotional_core: EmotionalCore):
        """Set the emotional core instance"""
        self._emotional_core = emotional_core
        logger.info("EmotionalCore connected to bridge")

    async def _on_user_message(self, event: Event):
        """Handle user message events"""
        user_text = event.data.get("content", "")
        session_id = event.data.get("session_id", "default")

        # Update emotional core if available
        if self._emotional_core and self.get_config("enable_emotions", True):
            # update_state is synchronous
            self._emotional_core.update_state(user_text)

        # Process message through orchestrator with memory context
        if self._orchestrator:
            # Get memory context
            memory_context = await self._get_memory_context(user_text, session_id)

            # Enrich input with context
            enriched_input = self._enrich_with_context(user_text, memory_context)

            # Process through orchestrator
            response = await asyncio.to_thread(
                self._orchestrator.process,
                enriched_input
            )

            # IMPORTANT: Send messages to plugins for analysis
            if self._orchestrator.plugin_manager:
                # Pattern Miner
                pattern_miner = self._orchestrator.plugin_manager.get_plugin('pattern_miner')
                if pattern_miner and hasattr(pattern_miner, 'analyze_message'):
                    pattern_miner.analyze_message(user_text, 'user')
                    pattern_miner.analyze_message(response, 'assistant')

                # Q-Learner
                q_learner = self._orchestrator.plugin_manager.get_plugin('q_learner')
                if q_learner and hasattr(q_learner, 'record_interaction'):
                    q_learner.record_interaction(user_text, response)

            # Store the interaction
            await event_bus.emit(Event(
                Events.ASSISTANT_RESPONSE,
                {
                    "content": response,
                    "session_id": session_id,
                    "metadata": {"enriched": True}
                },
                source="orchestrator_bridge"
            ))

    async def _on_emotion_changed(self, event: Event):
        """Handle emotion change events"""
        # Could update orchestrator behavior based on emotion
        emotion = event.data.get("emotion")
        intensity = event.data.get("intensity", 0.5)

        logger.info(f"Emotion changed to {emotion} with intensity {intensity}")

    async def get_enhanced_context(self, session_id: str = "default") -> Dict[str, Any]:
        """Get enhanced context with memory and emotions"""
        context = {
            "conversation": [],
            "emotional_state": None,
            "emotional_context": None,
            "recent_patterns": []
        }

        # Get conversation context
        if self._conversation_memory:
            context["conversation"] = await self._conversation_memory.get_context(session_id)

        # Get emotional state
        if self._emotional_core:
            # get_emotional_state() is synchronous
            context["emotional_state"] = self._emotional_core.get_emotional_state()

        # Get emotional context from enhancer
        if self._memory_enhancer:
            context["emotional_context"] = await self._memory_enhancer.get_emotional_context(session_id)

        return context

    async def search_memories(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Search through enhanced memories"""
        results = {
            "conversations": [],
            "emotional_memories": []
        }

        # Search conversations
        if self._conversation_memory:
            results["conversations"] = await self._conversation_memory.search_conversations(
                query, session_id
            )

        # Search emotional memories
        if self._memory_enhancer:
            memories = await self._memory_enhancer.recall_memories(query)
            results["emotional_memories"] = [
                {
                    "content": m.content,
                    "emotion": m.emotion,
                    "intensity": m.intensity,
                    "timestamp": m.timestamp,
                    "associations": m.associations
                }
                for m in memories
            ]

        return results

    async def _get_memory_context(self, user_text: str, session_id: str) -> Dict[str, Any]:
        """Get relevant memory context for the user input"""
        context = {
            "recent_conversation": [],
            "emotional_memories": [],
            "patterns": [],
            "emotional_state": None,
            "q_suggestions": [],
            "user_patterns": [],
            "relevant_memories": []  # For compatibility with orchestrator
        }

        # Get recent conversation history
        if self._conversation_memory:
            context["recent_conversation"] = await self._conversation_memory.get_context(session_id)

        # Search for relevant emotional memories
        if self._memory_enhancer:
            memories = await self._memory_enhancer.recall_memories(user_text)
            context["emotional_memories"] = [
                {
                    "content": m.content,
                    "emotion": m.emotion,
                    "intensity": m.intensity,
                    "associations": m.associations[:3]  # Limit associations
                }
                for m in memories[:3]  # Top 3 relevant memories
            ]

            # Also add to relevant_memories for orchestrator compatibility
            context["relevant_memories"] = context["emotional_memories"]

            # Get emotional context
            emotional_ctx = await self._memory_enhancer.get_emotional_context(session_id)
            context["emotional_state"] = emotional_ctx

        # Get Q-Learner suggestions
        q_learner = plugin_manager.get_plugin('q_learner')
        if q_learner and hasattr(q_learner, 'get_suggested_response'):
            try:
                # Q-Learner might be async or sync, handle both
                if asyncio.iscoroutinefunction(q_learner.get_suggested_response):
                    context['q_suggestions'] = await q_learner.get_suggested_response(user_text, context)
                else:
                    context['q_suggestions'] = q_learner.get_suggested_response(user_text, context)
            except Exception as e:
                logger.warning(f"Failed to get Q-Learner suggestions: {e}")
                context['q_suggestions'] = []

        # Get Pattern Miner patterns
        pattern_miner = plugin_manager.get_plugin('pattern_miner')
        if pattern_miner and hasattr(pattern_miner, 'get_user_patterns'):
            try:
                # Pattern Miner might be async or sync, handle both
                if asyncio.iscoroutinefunction(pattern_miner.get_user_patterns):
                    context['user_patterns'] = await pattern_miner.get_user_patterns()
                else:
                    context['user_patterns'] = pattern_miner.get_user_patterns()
            except Exception as e:
                logger.warning(f"Failed to get user patterns: {e}")
                context['user_patterns'] = []

        return context

    def _enrich_with_context(self, user_text: str, memory_context: Dict[str, Any]) -> str:
        """Enrich the user input with memory context"""
        enriched = user_text

        # Add conversation context if available
        if memory_context.get("recent_conversation"):
            recent_turns = memory_context["recent_conversation"][-3:]  # Last 3 turns
            if recent_turns:
                context_str = "\n[Conversation Context:\n"
                for turn in recent_turns:
                    context_str += f"- {turn['role']}: {turn['content'][:100]}...\n"
                context_str += "]\n"
                enriched = context_str + user_text

        # Add emotional memory context if relevant
        if memory_context.get("emotional_memories"):
            memories_str = "\n[Related Memories:\n"
            for mem in memory_context["emotional_memories"]:
                memories_str += f"- {mem['content'][:80]}... (emotion: {mem['emotion']})\n"
            memories_str += "]\n"
            enriched = memories_str + enriched

        # Add current emotional state
        if memory_context.get("emotional_state"):
            state = memory_context["emotional_state"]
            # Ensure emotional_stability is a float
            stability = state.get('emotional_stability', 1.0)
            if isinstance(stability, dict):
                # If it's a dict, try to get a numeric value or default to 1.0
                stability = stability.get('value', 1.0) if isinstance(stability, dict) else 1.0
            # Ensure it's a float
            try:
                stability = float(stability)
            except (TypeError, ValueError):
                stability = 1.0
            emotion_str = f"\n[Emotional Context: Current-{state.get('current_emotion', 'neutral')}, Dominant-{state.get('dominant_emotion', 'neutral')}, Stability-{stability:.2f}]\n"
            enriched = emotion_str + enriched

        return enriched

    async def get_system_state(self) -> Dict[str, Any]:
        """Get current system state"""
        state = {
            "bridge_active": self._running,
            "orchestrator_connected": self._orchestrator is not None,
            "emotional_core_connected": self._emotional_core is not None,
            "memory_enabled": self._conversation_memory is not None,
            "enhancement_enabled": self._memory_enhancer is not None,
            "stats": await self.get_stats()
        }

        # Add plugin states
        if self._conversation_memory:
            state["conversation_memory_health"] = await self._conversation_memory.health_check()

        if self._memory_enhancer:
            state["memory_enhancer_health"] = await self._memory_enhancer.health_check()

        return state

    async def _on_identity_confirmation_needed(self, event: Event):
        """Handle identity confirmation requests"""
        extraction = event.data.get('extraction', {})
        session_id = event.data.get('session_id', 'default')

        if extraction and extraction.get('name'):
            # Store for later confirmation
            self._pending_identity = extraction
            logger.info(f"Identity confirmation needed for name: {extraction['name']}")

    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        base_health = await super().health_check()

        base_health.update({
            "orchestrator_connected": self._orchestrator is not None,
            "emotional_core_connected": self._emotional_core is not None,
            "plugins_connected": {
                "conversation_memory": self._conversation_memory is not None,
                "memory_enhancer": self._memory_enhancer is not None,
                "user_identity_manager": self._user_identity_manager is not None,
                "memory_aligner": self._memory_aligner is not None,
                "emotion_adjuster": self._emotion_adjuster is not None
            }
        })

        return base_health

    async def _update_plugins_from_dialogue_engine(self):
        """Synchronise les données du dialogue engine avec les plugins"""
        if not self._dialogue_engine:
            return

        try:
            user_memory = self._dialogue_engine.await memory.retrieve(MemoryQuery(memory_types=[MemoryType.CONTEXTUAL]))

            # Mettre à jour le UserIdentityManager
            if self._user_identity_manager and user_memory.get('name'):
                await self._user_identity_manager.confirm_identity(
                    user_memory['name']
                )

            # Synchroniser les sujets d'intérêt
            if self._user_identity_manager and user_memory.get('interests'):
                for interest in user_memory['interests']:
                    await self._user_identity_manager.add_user_fact('interest', interest)

            # Synchroniser avec le MemoryAligner si disponible
            if self._memory_aligner and user_memory:
                await self._memory_aligner.align_memories(user_memory)

            # Mettre à jour l'EmotionAdjuster si disponible
            if self._emotion_adjuster and user_memory.get('emotional_context'):
                await self._emotion_adjuster.adjust_emotions(
                    user_memory['emotional_context']
                )

        except Exception as e:
            logger.error(f"Erreur lors de la synchronisation des plugins: {e}")


async def create_orchestrator_bridge(orchestrator):
    """Crée et initialise le bridge de l'orchestrateur de manière asynchrone"""
    try:
        # Créer le bridge
        bridge = OrchestratorBridge()

        # Lier le bridge à l'orchestrateur
        bridge.set_orchestrator(orchestrator)

        # Lier le bridge au core émotionnel
        if hasattr(orchestrator, 'emotional_core'):
            bridge.set_emotional_core(orchestrator.emotional_core)

        # Initialiser le bridge de manière asynchrone
        await bridge.start()

        logger.info("✅ Bridge lié à l'orchestrateur")
        return bridge

    except Exception as e:
        logger.error(f"❌ Erreur lors de la création du bridge: {e}")
        raise
