#!/usr/bin/env python3
"""
Quick test to verify NeuralBus is working
"""

import asyncio

from jeffrey.core.neuralbus import CloudEvent, EventMeta, EventPriority, neural_bus


async def main():
    print("🧠 NeuralBus Quick Test")
    print("-" * 40)

    try:
        # Initialize
        print("1. Initializing NeuralBus...")
        await neural_bus.start()
        print("   ✅ Initialized")

        # Create event
        print("\n2. Creating test event...")
        event = CloudEvent(
            meta=EventMeta(type="test.quickcheck", tenant_id="test_tenant", priority=EventPriority.HIGH),
            data={"message": "NeuralBus is operational!", "timestamp": "2025-09-27"},
        )
        print(f"   ✅ Event created: {event.meta.id}")

        # Publish
        print("\n3. Publishing event...")
        event_id = await neural_bus.publish(event)
        print(f"   ✅ Published: {event_id}")

        # Health check
        print("\n4. Health check...")
        health = await neural_bus.health_check()
        print(f"   ✅ Status: {health['status']}")
        print(f"   ✅ Publisher connected: {health['publisher_connected']}")

        # Stream info
        print("\n5. Stream info...")
        info = await neural_bus.get_stream_info()
        if info:
            print(f"   ✅ Messages in stream: {info.get('messages', 0)}")
            print(f"   ✅ Stream bytes: {info.get('bytes', 0)}")

        print("\n" + "=" * 40)
        print("🎉 NeuralBus is OPERATIONAL!")
        print("=" * 40)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Shutdown
        print("\n6. Shutting down...")
        await neural_bus.shutdown()
        print("   ✅ Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
