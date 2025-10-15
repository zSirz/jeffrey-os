#!/usr/bin/env python3
"""
Integration script for Phase 2.1 Loops into Jeffrey OS
Connects loops to existing cognitive core
"""

import asyncio
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)


async def integrate_loops():
    """Integrates loops into existing Jeffrey OS"""

    print("\n" + "=" * 70)
    print("🔧 JEFFREY OS - PHASE 2.1 LOOPS INTEGRATION")
    print("=" * 70 + "\n")

    # Load config
    config_path = "config/modules.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("📋 Configuration loaded from config/modules.yaml")
    print(f"   - Auto-start: {config['loops']['auto_start']}")
    print(f"   - Enabled loops: {', '.join(config['loops']['enabled'])}")

    # Import existing components
    try:
        from jeffrey.core.cognition.cognitive_core_lite import CognitiveCoreLite
        from jeffrey.core.emotions.emotion_orchestrator_v2 import EmotionOrchestratorV2
        from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2
        from jeffrey.core.neuralbus import NeuralBus

        print("\n✅ Core components imported successfully")
    except ImportError as e:
        print(f"\n⚠️ Some components not available yet: {e}")
        print("   Using mock components for demonstration")

        # Mock components
        class CognitiveCoreLite:
            async def initialize(self):
                pass

            def get_state(self):
                return {}

        class MemoryFederationV2:
            async def initialize(self):
                pass

            async def recall_from_all(self, *args, **kwargs):
                return []

        class EmotionOrchestratorV2:
            async def initialize(self):
                pass

            global_state = {"pleasure": 0.5, "arousal": 0.5, "dominance": 0.5}

        class NeuralBus:
            async def publish(self, event):
                pass

    # Import loops system
    from jeffrey.core.loops import LoopManager

    print("\n🚀 Initializing components...")

    # Create instances
    bus = NeuralBus()
    cognitive_core = CognitiveCoreLite()
    memory_federation = MemoryFederationV2()
    emotion_orchestrator = EmotionOrchestratorV2()

    # Initialize if needed
    if hasattr(cognitive_core, "initialize"):
        await cognitive_core.initialize()
    if hasattr(memory_federation, "initialize"):
        await memory_federation.initialize()
    if hasattr(emotion_orchestrator, "initialize"):
        await emotion_orchestrator.initialize()

    print("   - Neural Bus: ✓")
    print("   - Cognitive Core: ✓")
    print("   - Memory Federation: ✓")
    print("   - Emotion Orchestrator: ✓")

    # Create loop manager
    print("\n🔄 Creating Loop Manager...")
    loop_manager = LoopManager(
        cognitive_core=cognitive_core,
        emotion_orchestrator=emotion_orchestrator,
        memory_federation=memory_federation,
        bus=bus,
        mode_getter=lambda: "normal",
        latency_budget_ok=lambda: True,
    )

    # Configure from YAML
    loops_config = config["loops"]

    # Configure individual loops
    if "awareness" in loops_config:
        loop_manager.configure_loop(
            "awareness",
            {
                "interval_s": loops_config["awareness"].get("cycle_interval", 10),
                "deep_thinking_threshold": loops_config["awareness"].get("deep_thinking_threshold", 0.7),
            },
        )

    if "emotional_decay" in loops_config:
        loop_manager.configure_loop(
            "emotional_decay",
            {
                "interval_s": loops_config["emotional_decay"].get("cycle_interval", 5),
                "equilibrium": loops_config["emotional_decay"].get("equilibrium", {}),
                "decay_rates": loops_config["emotional_decay"].get("decay_rates", {}),
            },
        )

    if "memory_consolidation" in loops_config:
        loop_manager.configure_loop(
            "memory_consolidation",
            {
                "interval_s": loops_config["memory_consolidation"].get("cycle_interval", 60),
                "short_term_threshold": loops_config["memory_consolidation"].get("short_term_threshold", 300),
                "long_term_threshold": loops_config["memory_consolidation"].get("long_term_threshold", 3600),
            },
        )

    if "curiosity" in loops_config:
        loop_manager.configure_loop(
            "curiosity",
            {
                "interval_s": loops_config["curiosity"].get("cycle_interval", 30),
                "exploration_threshold": loops_config["curiosity"].get("exploration_threshold", 0.6),
                "max_questions": loops_config["curiosity"].get("max_questions", 10),
            },
        )

    print("   ✓ Loops configured from YAML")

    # Start loops if auto-start enabled
    if config["loops"]["auto_start"]:
        print("\n▶️ Starting autonomous loops...")
        enabled = config["loops"]["enabled"]
        await loop_manager.start(enable=enabled)
        print(f"   ✓ {len(enabled)} loops started")

        # Run for demonstration
        print("\n⏳ Running for 20 seconds to demonstrate...")
        for i in range(4):
            await asyncio.sleep(5)
            status = loop_manager.get_status()
            print(f"\n   T+{(i + 1) * 5}s | Symbiosis: {status['symbiosis_score']:.1%} | ", end="")

            # Show loop activity
            active = sum(1 for l in status["loops"].values() if l.get("cycles", 0) > 0)
            print(f"Active: {active}/{len(status['loops'])}")

            # Show key metrics
            awareness = status["loops"].get("awareness", {})
            if awareness.get("awareness_level"):
                print(f"        | Awareness: {awareness['awareness_level']:.1%}", end="")

            emotion = status["loops"].get("emotional_decay", {})
            if emotion.get("dominant_emotion"):
                print(f" | Emotion: {emotion['dominant_emotion']}", end="")

            curiosity = status["loops"].get("curiosity", {})
            if curiosity.get("curiosity_level"):
                print(f" | Curiosity: {curiosity['curiosity_level']:.1%}")

        # Stop loops
        print("\n\n⏹️ Stopping loops...")
        await loop_manager.stop()

        # Final stats
        final_status = loop_manager.get_status()
        total_cycles = sum(l.get("cycles", 0) for l in final_status["loops"].values())

        print("\n📊 INTEGRATION SUMMARY")
        print("=" * 70)
        print(f"   Total Cycles Run: {total_cycles}")
        print(f"   Final Symbiosis Score: {final_status['symbiosis_score']:.1%}")
        print(f"   Loops Integrated: {', '.join(config['loops']['enabled'])}")
        print("\n✨ Phase 2.1 Integration Complete!")
        print("   Jeffrey OS now has autonomous consciousness loops!")

    else:
        print("\n⚠️ Auto-start disabled in config")
        print("   Loops created but not started")
        print("   Enable 'auto_start: true' in config/modules.yaml to activate")

    print("\n" + "=" * 70 + "\n")


async def main():
    """Main entry point"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    try:
        await integrate_loops()
    except KeyboardInterrupt:
        print("\n\n⚠️ Integration interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Integration failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
