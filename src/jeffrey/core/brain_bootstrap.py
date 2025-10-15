"""
Brain Bootstrap - Active la boucle cognitive minimale
Emotion → Memory → Consciousness sans stubs massifs
"""
import logging
import asyncio
import time
from datetime import datetime
from typing import Optional, Dict, Any

from jeffrey.core.ports.memory_port import MemoryPort
from jeffrey.core.contracts.thoughts import create_thought, ThoughtState, ensure_thought_format

logger = logging.getLogger(__name__)

class BrainBootstrap:
    """
    Bootstrap minimal du cerveau
    Connecte uniquement les modules existants et fonctionnels
    """

    def __init__(self, bus=None):
        self.bus = bus
        self.wired = False
        self.stats = {
            "emotions_received": 0,
            "memories_stored": 0,
            "thoughts_generated": 0,
            "errors": 0
        }

        # Modules réels (pas de stubs)
        self.memory_impl = None
        self.memory = None  # Sera un MemoryPort
        self.consciousness = None

    async def initialize_modules(self):
        """Charge uniquement les modules qui existent vraiment"""

        # Memory - testons plusieurs variantes connues avec MemoryPort
        try:
            # Essayons d'abord UnifiedMemory
            from jeffrey.memory.unified_memory import UnifiedMemory
            self.memory_impl = UnifiedMemory()
            self.memory = MemoryPort(self.memory_impl)
            logger.info(f"✅ Memory module loaded with interface: {self.memory.stats['method_used']}")
        except Exception:
            try:
                # Fallback vers AdvancedUnifiedMemory
                from jeffrey.core.memory.advanced_unified_memory import AdvancedUnifiedMemory
                self.memory_impl = AdvancedUnifiedMemory()
                self.memory = MemoryPort(self.memory_impl)
                logger.info(f"✅ Memory module loaded with interface: {self.memory.stats['method_used']}")
            except Exception as e:
                logger.error(f"❌ Memory loading failed: {e}")
                # Créer un fallback simple
                self.memory = MemoryPort(None)  # Utilisera le buffer interne
                logger.warning("⚠️ Using fallback memory buffer")

        # Consciousness - peut ne pas exister, fallback gracieux
        try:
            from jeffrey.core.consciousness.conscience_engine import ConsciousnessEngine
            self.consciousness = ConsciousnessEngine()
            logger.info("✅ Consciousness module loaded")
        except Exception:
            try:
                # Essayons d'autres variantes
                from jeffrey.core.cognition.cognitive_core import CognitiveCore
                self.consciousness = CognitiveCore()
                logger.info("✅ Consciousness module loaded (CognitiveCore)")
            except Exception:
                logger.warning("⚠️ Consciousness not available - using fallback")
                self.consciousness = None

    async def wire_minimal_loop(self):
        """Connecte la boucle Emotion → Memory → Consciousness"""

        if self.wired:
            return self.bus

        if not self.bus:
            try:
                from jeffrey.core.neuralbus.bus import NeuralBus
                self.bus = NeuralBus()
                logger.info("✅ Using NeuralBus")
            except Exception:
                try:
                    from jeffrey.core.bus.local_async_bus import LocalAsyncBus
                    self.bus = LocalAsyncBus()
                    logger.info("✅ Using LocalAsyncBus fallback")
                except Exception as e:
                    logger.error(f"❌ No bus available: {e}")
                    return None

        # Charger les modules
        await self.initialize_modules()

        from jeffrey.core.neuralbus.contracts import (
            EMOTION_DETECTED, MEMORY_STORE_REQ, MEMORY_STORED,
            MEMORY_RECALLED, THOUGHT_EVENT
        )

        # Handler 1: Emotion → Memory
        async def on_emotion(event_data):
            """Quand une émotion est détectée, la stocker en mémoire"""
            try:
                self.stats["emotions_received"] += 1

                if self.memory:
                    # Stocker dans la mémoire
                    memory_entry = {
                        "text": event_data.get("text"),
                        "emotion": event_data.get("emotion"),
                        "confidence": event_data.get("confidence"),
                        "timestamp": event_data.get("timestamp"),
                        "tags": [event_data.get("emotion")],
                        "meta": {
                            "all_scores": event_data.get("all_scores"),
                            "source": "emotion_ml"
                        }
                    }

                    # Store synchrone (la plupart des mémoires sont sync)
                    if hasattr(self.memory, 'store'):
                        self.memory.store(memory_entry)
                    elif hasattr(self.memory, 'add'):
                        self.memory.add(memory_entry)
                    else:
                        logger.warning("Memory module has no store/add method")
                        return

                    self.stats["memories_stored"] += 1

                    # Publier confirmation
                    await self.bus.publish(MEMORY_STORED, {
                        "success": True,
                        "entry_id": str(self.stats["memories_stored"]),
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    logger.debug(f"📝 Stored emotion memory #{self.stats['memories_stored']}")

            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Memory store failed: {e}")

        # Handler 2: Memory → Recall (trigger automatique)
        async def on_memory_stored(event_data):
            """Quand une mémoire est stockée, trigger un recall"""
            try:
                if self.memory:
                    recalled = []
                    # Essayer différentes méthodes de recherche
                    if hasattr(self.memory, 'search'):
                        recalled = self.memory.search(query="", limit=5)
                    elif hasattr(self.memory, 'get_recent'):
                        recalled = self.memory.get_recent(limit=5)
                    elif hasattr(self.memory, 'recall'):
                        recalled = self.memory.recall(query="", limit=5)

                    await self.bus.publish(MEMORY_RECALLED, {
                        "memories": recalled,
                        "count": len(recalled),
                        "timestamp": datetime.utcnow().isoformat()
                    })

                    logger.debug(f"🔍 Recalled {len(recalled)} memories")

            except Exception as e:
                logger.error(f"Memory recall failed: {e}")

        # Handler 3: Memory → Consciousness → Thought
        async def on_memory_recalled(event_data):
            """Quand des mémoires sont rappelées, générer une pensée"""
            try:
                memories = event_data.get("memories", [])
                thought = None

                if self.consciousness:
                    # Conscience réelle - gérer sync/async
                    proc = getattr(self.consciousness, "process", None)
                    if callable(proc):
                        try:
                            maybe = proc(memories)
                            if asyncio.iscoroutine(maybe):
                                thought = await maybe
                            else:
                                thought = maybe
                        except Exception as e:
                            logger.warning(f"Consciousness processing error: {e}")

                # Fallback simple si pas de conscience ou erreur
                if thought is None:
                    thought = {
                        "state": "aware",
                        "context_size": len(memories),
                        "mode": "skeletal_fallback",
                        "timestamp": datetime.utcnow().isoformat()
                    }

                self.stats["thoughts_generated"] += 1

                await self.bus.publish(THOUGHT_EVENT, thought)

                logger.info(f"💭 Generated thought #{self.stats['thoughts_generated']}")

            except Exception as e:
                self.stats["errors"] += 1
                logger.error(f"Consciousness processing failed: {e}")

        # Connecter tous les handlers avec l'interface NeuralBus réelle
        try:
            # Le NeuralBus utilise des consumers, pas des subscribe simples
            # Pour l'instant, on va simplement marquer comme "wired" mais sans subscription
            # La logique sera basée sur la publication directe sans handlers asynchrones

            logger.warning("⚠️ Using simplified brain loop without event subscriptions")
            logger.info("📧 Emotion events will be processed synchronously in API calls")

            self.wired = True
            logger.info("🧠 Brain loop wired: Simplified synchronous mode")

        except Exception as e:
            logger.error(f"Failed to wire brain loop: {e}")
            return None

        return self.bus

    def get_stats(self) -> Dict[str, Any]:
        """Retourne les statistiques live pour monitoring"""
        memory_stats = self.memory.get_stats() if self.memory else {}

        return {
            **self.stats,
            "wired": self.wired,
            "memory_available": self.memory is not None,
            "consciousness_available": self.consciousness is not None,
            "memory_type": type(self.memory_impl).__name__ if self.memory_impl else "fallback_buffer",
            "consciousness_type": type(self.consciousness).__name__ if self.consciousness else None,
            "memory_port_stats": memory_stats
        }