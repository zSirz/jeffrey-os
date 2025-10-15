#!/usr/bin/env python3
"""Jeffrey Brain V2 - Central cognitive processor"""

import asyncio

# Fix imports
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from jeffrey.core.cognitive.auto_learner import AutoLearner
from jeffrey.core.cognitive.curiosity_engine import CuriosityEngine

# Import cognitive modules
from jeffrey.core.cognitive.orchestrator import ModuleOrchestrator
from jeffrey.core.cognitive.theory_of_mind import TheoryOfMind
from jeffrey.core.envelope_helper import make_envelope
from jeffrey.core.memory.unified_memory import UnifiedMemory
from jeffrey.core.module_registry import ModuleRegistry
from jeffrey.core.neural_bus import NeuralBus
from jeffrey.utils.logger import get_logger

logger = get_logger("JeffreyBrain")


class JeffreyBrain:
    """Central brain that coordinates all cognitive functions"""

    def __init__(self):
        self.logger = logger
        self.memory = None
        self.bus = None
        self.registry = None
        self.orchestrator = None
        self.cognitive_modules = []
        self.initialized = False

    async def initialize(self):
        """Initialize all brain components"""
        if self.initialized:
            self.logger.warning("Brain already initialized")
            return

        self.logger.info("🧠 Initializing Jeffrey Brain V2...")

        try:
            # 1. Initialize memory
            self.memory = UnifiedMemory(backend="sqlite")
            await self.memory.initialize()
            self.logger.info("✅ Memory system online")

            # 2. Initialize neural bus
            self.bus = NeuralBus()
            self.logger.info("✅ Neural bus online")

            # 3. Initialize module registry
            self.registry = ModuleRegistry()
            # Check if register is async
            if asyncio.iscoroutinefunction(self.registry.register):
                await self.registry.register("memory", self.memory)
                await self.registry.register("bus", self.bus)
            else:
                self.registry.register("memory", self.memory)
                self.registry.register("bus", self.bus)
            self.logger.info("✅ Module registry online")

            # 4. Create cognitive modules - only if not already set
            if not self.cognitive_modules:
                self.cognitive_modules = [
                    AutoLearner(self.memory),
                    TheoryOfMind(self.memory),
                    CuriosityEngine(self.memory),
                ]

            # 5. Initialize orchestrator
            self.orchestrator = ModuleOrchestrator(modules=self.cognitive_modules, timeout=1.5)
            await self.orchestrator.initialize()
            self.logger.info(f"✅ Cognitive orchestrator online with {len(self.cognitive_modules)} modules")

            # 6. Subscribe to bus events (preparation for event-driven)
            await self._setup_event_handlers()

            self.initialized = True
            self.logger.info("🎉 Jeffrey Brain V2 fully initialized!")

        except Exception as e:
            self.logger.error(f"Brain initialization failed: {e}")
            raise

    async def _setup_event_handlers(self):
        """Setup event handlers for future event-driven architecture"""
        # Placeholder for event-driven architecture
        # Will be implemented when moving to full event-driven
        pass

    async def process_input(self, text: str, user_id: str = "default") -> dict[str, Any]:
        """
        Process user input through all cognitive systems
        """
        if not self.initialized:
            await self.initialize()

        start_time = time.time()
        trace_id = f"trace_{int(time.time() * 1e6)}"

        self.logger.info(f"Processing input from {user_id}: {text[:50]}...")

        # 1. Store raw input in memory
        await self.memory.store(
            {
                "type": "input",
                "message": text,
                "user_id": user_id,
                "trace_id": trace_id,
                "timestamp": time.time(),
            }
        )

        # 2. Prepare data for cognitive processing
        input_data = {
            "text": text,
            "message": text,  # Compatibility
            "user_id": user_id,
            "trace_id": trace_id,
            "timestamp": time.time(),
        }

        # 3. Process through all cognitive modules in parallel
        module_results, errors = await self.orchestrator.process(input_data)

        # 4. Extract key insights
        intention = module_results.get("TheoryOfMind", {}).get("intention", "unknown")
        emotion = module_results.get("TheoryOfMind", {}).get("emotion", "neutral")
        patterns_learned = module_results.get("AutoLearner", {}).get("unique_patterns", 0)
        curiosity = module_results.get("CuriosityEngine", {}).get("questions", [])

        # 5. Store cognitive results in memory
        await self.memory.store(
            {
                "type": "cognitive_result",
                "trace_id": trace_id,
                "user_id": user_id,
                "intention": intention,
                "emotion": emotion,
                "patterns": patterns_learned,
                "timestamp": time.time(),
            }
        )

        # 6. Publish to neural bus for other systems
        envelope = make_envelope(
            topic="cognitive.processed",
            payload={
                "original_input": text,
                "user_id": user_id,
                "trace_id": trace_id,
                "modules": module_results,
                "errors": errors,
                "summary": {
                    "intention": intention,
                    "emotion": emotion,
                    "curiosity": curiosity[:1] if curiosity else [],
                },
            },
            ns="brain",
        )

        await self.bus.publish(envelope)

        # 7. Calculate processing time
        processing_time = time.time() - start_time

        # 8. Return comprehensive result - FIXED modules_active field
        result = {
            "success": True,
            "trace_id": trace_id,
            "processing_time": processing_time,
            "intention": intention,
            "emotion": emotion,
            "patterns_learned": patterns_learned,
            "curiosity": curiosity[:1] if curiosity else None,
            "modules_active": self.orchestrator.get_stats()["active_modules"],
            "errors": errors if errors else None,
        }

        self.logger.info(f"✅ Processed in {processing_time:.3f}s - Intention: {intention}, Emotion: {emotion}")

        return result

    async def get_stats(self) -> dict[str, Any]:
        """Get comprehensive brain statistics"""
        stats = {
            "initialized": self.initialized,
            "memory": self.memory.get_stats() if self.memory else None,
            "orchestrator": self.orchestrator.get_stats() if self.orchestrator else None,
            "timestamp": time.time(),
        }
        return stats

    async def shutdown(self):
        """Graceful shutdown of all systems"""
        self.logger.info("🛑 Shutting down Jeffrey Brain...")

        try:
            # Shutdown orchestrator (which shuts down modules)
            if self.orchestrator:
                await self.orchestrator.shutdown()

            # Shutdown memory
            if self.memory:
                await self.memory.shutdown()

            self.initialized = False
            self.logger.info("✅ Brain shutdown complete")

        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
            raise


# For backward compatibility
async def boot():
    """Legacy boot function"""
    brain = JeffreyBrain()
    await brain.initialize()
    return brain


if __name__ == "__main__":
    # Quick test
    async def test():
        brain = JeffreyBrain()
        await brain.initialize()

        # Test inputs
        result = await brain.process_input("Hello Jeffrey! How are you?", "test_user")
        print(f"Result: {result}")

        result = await brain.process_input("Tell me about Python programming", "test_user")
        print(f"Result: {result}")

        stats = await brain.get_stats()
        print(f"Stats: {stats['orchestrator']}")

        await brain.shutdown()

    asyncio.run(test())
