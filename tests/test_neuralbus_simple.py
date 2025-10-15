"""
Simple tests for NeuralBus without pytest fixtures
"""

import asyncio

from jeffrey.core.neuralbus import CloudEvent, EventMeta, EventPriority, NeuralBus


async def test_basic_publish():
    """Test basic publish"""
    bus = NeuralBus()
    await bus.initialize()

    try:
        event = CloudEvent(
            meta=EventMeta(type="test.basic", tenant_id="test_tenant"),
            data={"message": "Hello NeuralBus"},
        )

        event_id = await bus.publish(event)
        assert event_id == event.meta.id
        print("✅ Basic publish test passed")

    finally:
        await bus.shutdown()


async def test_deduplication():
    """Test deduplication"""
    bus = NeuralBus()
    await bus.initialize()

    try:
        event = CloudEvent(
            meta=EventMeta(id="duplicate-test-id", type="test.dedup", tenant_id="test_tenant"),
            data={"test": "deduplication"},
        )

        # Publish same event twice
        id1 = await bus.publish(event)
        id2 = await bus.publish(event)  # Should be deduplicated

        assert id1 == id2
        print("✅ Deduplication test passed")

    finally:
        await bus.shutdown()


async def test_health_check():
    """Test health check"""
    bus = NeuralBus()
    await bus.initialize()

    try:
        health = await bus.health_check()
        assert health["status"] == "healthy"
        assert health["publisher_connected"] is True
        print("✅ Health check test passed")

    finally:
        await bus.shutdown()


async def test_priority_routing():
    """Test priority routing"""
    bus = NeuralBus()
    await bus.initialize()

    try:
        # Test different priorities
        for priority in [
            EventPriority.LOW,
            EventPriority.NORMAL,
            EventPriority.HIGH,
            EventPriority.CRITICAL,
        ]:
            event = CloudEvent(
                meta=EventMeta(type="test.priority", tenant_id="test", priority=priority),
                data={"priority": priority.value},
            )

            event_id = await bus.publish(event)
            assert event_id is not None

        print("✅ Priority routing test passed")

    finally:
        await bus.shutdown()


async def main():
    """Run all tests"""
    print("Running NeuralBus Simple Tests")
    print("=" * 40)

    tests = [test_basic_publish, test_deduplication, test_health_check, test_priority_routing]

    passed = 0
    failed = 0

    for test in tests:
        try:
            await test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            failed += 1

    print("=" * 40)
    print(f"Results: {passed} passed, {failed} failed")

    if failed == 0:
        print("🎉 All tests passed!")

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
