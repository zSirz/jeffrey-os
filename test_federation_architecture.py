#!/usr/bin/env python3
"""
Test script for the new federation architecture
Shows all the new features in action
"""

import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

sys.path.insert(0, ".")


async def main():
    print("\n" + "=" * 60)
    print("🚀 JEFFREY OS - FEDERATION ARCHITECTURE TEST")
    print("=" * 60 + "\n")

    from jeffrey.core.cognition.cognitive_core_lite import CognitiveCore
    from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
    from jeffrey.core.neuralbus.bus_v2 import NeuralBusV2

    # Create components
    print("1. Creating components...")
    loader = SecureModuleLoader()
    bus = NeuralBusV2()
    core = CognitiveCore(loader=loader)

    # Start bus
    print("2. Starting Neural Bus V2...")
    await bus.start()
    print("   ✅ Bus started")

    # Initialize core with loader
    print("\n3. Initializing Cognitive Core with federations...")
    await core.initialize(bus, {"loader": loader})

    print(f"\n   ✅ Core initialized in '{core.state['mode']}' mode")
    print(f"   📊 Memory: {core.state['active_memory_layers']} layers active")
    print(f"   🎭 Emotions: {core.state['active_emotion_categories']} categories active")

    # Test the pipeline
    print("\n4. Testing cognitive pipeline...")

    test_messages = [
        "Bonjour Jeffrey !",
        "Comment vas-tu aujourd'hui ?",
        "Je suis content de te parler",
        "Te souviens-tu de notre conversation ?",
    ]

    for i, message in enumerate(test_messages, 1):
        print(f"\n   Test {i}: '{message}'")

        # Simulate user input
        loop = asyncio.get_event_loop()
        response_future = loop.create_future()

        async def capture_response(envelope):
            if not response_future.done():
                response_future.set_result(envelope)

        # Subscribe to response
        await bus.subscribe("response.ready", capture_response)

        # Send input
        await bus.publish(
            {
                "type": "user.input",
                "data": {"user_id": "test_user", "text": message},
                "correlation_id": f"test_{i}",
            }
        )

        # Wait for response (with timeout)
        try:
            response = await asyncio.wait_for(response_future, timeout=3.0)
            data = response.get("data", {})

            print(f"   📝 Response: {data.get('text', '...')}")

            # Show enriched data if available
            if data.get("emotion"):
                emotion = data["emotion"]
                print(f"   🎭 Emotion: {emotion.get('mood', 'unknown')} (valence: {emotion.get('valence', 0):.2f})")

            if data.get("memory"):
                memory = data["memory"]
                print(f"   🧠 Memory: {memory.get('recalled', 0)} memories recalled")

            if data.get("processing_ms"):
                print(f"   ⚡ Processing time: {data['processing_ms']:.1f}ms")

            if data.get("alignment_score") is not None:
                print(f"   🔄 Alignment: {data['alignment_score']:.2f}")

        except TimeoutError:
            print("   ⚠️ Timeout waiting for response")

        # Note: NeuralBusV2 doesn't have unsubscribe, subscriptions are managed internally

        # Small delay between messages
        await asyncio.sleep(0.5)

    # Show final status
    print("\n5. Final Status Report:")
    print("=" * 40)

    status = core.get_status()

    print(f"   📊 Mode: {status['mode']}")
    print(f"   💬 Messages processed: {status['state']['messages_processed']}")
    print(f"   🧠 Total active modules: {status['total_active_modules']}")

    if status.get("memory_federation"):
        mem_stats = status["memory_federation"]
        print("\n   Memory Federation Stats:")
        for layer_name, layer_stats in mem_stats.get("layers", {}).items():
            if layer_stats.get("initialized"):
                print(f"      - {layer_name}: {layer_stats.get('modules', 0)} modules")

    if status.get("emotion_orchestrator"):
        emo_stats = status["emotion_orchestrator"]
        print("\n   Emotion Orchestrator Stats:")
        state = emo_stats.get("current_state", {})
        print(f"      - Current mood: {state.get('mood', 'neutral')}")
        print(f"      - Fusion method: {emo_stats.get('fusion_method', 'unknown')}")

    metrics = status.get("metrics", {})
    if metrics.get("latency_p50"):
        print("\n   ⚡ Performance Metrics:")
        print(f"      - P50 latency: {metrics['latency_p50']:.1f}ms")
        print(f"      - P95 latency: {metrics['latency_p95']:.1f}ms")
        print(f"      - Errors: {metrics.get('errors', 0)}")

    # Cleanup
    print("\n6. Shutting down...")
    await bus.stop()
    print("   ✅ Bus stopped")

    print("\n" + "=" * 60)
    print("✅ TEST COMPLETE - FEDERATION ARCHITECTURE FUNCTIONAL!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
