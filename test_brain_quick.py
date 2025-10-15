#!/usr/bin/env python3
"""
Test rapide du BrainKernel - Smoke Test
"""

import asyncio
import logging
import sys

# Configuration du logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Support Windows si nécessaire
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def test():
    """Test de smoke pour le BrainKernel"""

    try:
        # Imports
        from jeffrey.bridge.adapters.http_adapter import HttpAdapter

        from jeffrey.bridge.registry import BridgeRegistry
        from jeffrey.core.kernel import BrainKernel
        from jeffrey.core.neural_bus import NeuralEnvelope

        print("\n" + "=" * 60)
        print("🧪 JEFFREY OS - BRAIN KERNEL SMOKE TEST")
        print("=" * 60 + "\n")

        # 1. Setup Bridge
        print("📡 Initializing Bridge...")
        bridge = BridgeRegistry()
        bridge.register(HttpAdapter())
        print("✅ Bridge ready with HTTP adapter")

        # 2. Create Kernel
        print("\n🧠 Creating BrainKernel...")
        kernel = BrainKernel(bridge)
        print("✅ Kernel created")

        # 3. Initialize
        print("\n🚀 Initializing all components...")
        await kernel.initialize()
        print("✅ Initialization complete")

        # 4. Test Chat
        print("\n💬 Testing chat functionality...")
        result = await kernel.bus.publish(
            NeuralEnvelope(
                topic="chat.in",
                payload={"text": "Bonjour Jeffrey! Comment vas-tu?"},
                meta={"session_id": "test"},
            ),
            wait_for_response=True,
            timeout=5.0,
        )

        if result:
            print(f"✅ Chat response received: {result.get('response', 'No response')[:100]}...")
        else:
            print("⚠️ No chat response (modules may not be loaded)")

        # 5. Health Check
        print("\n🏥 Running health check...")
        health = await kernel.get_health_status()
        print(f"✅ Kernel status: {health['kernel']}")
        print(f"   Sessions active: {health['sessions']}")
        print(f"   Census loaded: {health['census_loaded']}")
        print(f"   Components: {list(health.get('components', {}).keys())}")

        # 6. Metrics
        print("\n📊 Getting metrics...")
        metrics = await kernel.get_metrics()
        print("✅ Metrics retrieved:")
        print(f"   Bus events published: {metrics['bus']['events_published']}")
        print(f"   Census modules: {metrics['census']['total_modules']}")
        print(f"   Loaded modules: {metrics['census']['loaded_modules']}")

        # 7. Test Memory (if available)
        print("\n🧠 Testing memory store...")
        memory_result = await kernel.bus.publish(
            NeuralEnvelope(topic="memory.store", payload={"content": "Test memory entry", "importance": 0.8}),
            wait_for_response=True,
            timeout=2.0,
        )

        if memory_result and memory_result.get("stored"):
            print(f"✅ Memory stored with ID: {memory_result.get('memory_id')}")
        else:
            print("⚠️ Memory component not available or store failed")

        # 8. Clean Shutdown
        print("\n🛑 Shutting down...")
        await kernel.shutdown()
        print("✅ Shutdown complete")

        print("\n" + "=" * 60)
        print("🎉 SMOKE TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    print("Starting Jeffrey OS Brain Kernel test...")
    asyncio.run(test())
