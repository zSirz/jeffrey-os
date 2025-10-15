"""
Comprehensive tests for NeuralBus
Tests all major features including priority, DLQ, dedup
"""

import asyncio

import pytest

from jeffrey.core.neuralbus import CloudEvent, EventMeta, EventPriority, neural_bus


@pytest.fixture
async def bus():
    """Initialize and cleanup bus for tests"""
    test_bus = neural_bus
    await test_bus.initialize()
    try:
        yield test_bus
    finally:
        await test_bus.shutdown()


@pytest.mark.asyncio
async def test_basic_publish_consume(bus):
    """Test basic publish and consume"""
    received = []

    # Create consumer
    consumer = bus.create_consumer("TEST_BASIC", "events.test.>")

    async def handler(event, headers):
        received.append(event)

    consumer.register_handler("test.basic", handler)
    await consumer.connect()

    # Start consumer
    task = asyncio.create_task(consumer.run())
    await asyncio.sleep(0.5)

    # Publish event
    event = CloudEvent(
        meta=EventMeta(type="test.basic", tenant_id="test_tenant"),
        data={"message": "Hello NeuralBus"},
    )

    event_id = await bus.publish(event)
    assert event_id == event.meta.id

    # Wait for processing
    await asyncio.sleep(1)

    # Verify
    assert len(received) == 1
    assert received[0].meta.id == event.meta.id
    assert received[0].data["message"] == "Hello NeuralBus"

    # Cleanup
    await consumer.stop()
    task.cancel()


@pytest.mark.asyncio
async def test_deduplication(bus):
    """Test that deduplication prevents duplicate processing"""
    event = CloudEvent(meta=EventMeta(type="test.dedup", tenant_id="test_tenant"), data={"test": "deduplication"})

    # Publish same event twice
    id1 = await bus.publish(event)
    id2 = await bus.publish(event)  # Should be deduplicated

    assert id1 == id2  # Same ID returned


@pytest.mark.asyncio
async def test_priority_lanes(bus):
    """Test priority event routing"""
    normal_received = []
    priority_received = []

    # Create both consumers
    normal, priority = bus.create_priority_consumers()

    async def normal_handler(event, headers):
        normal_received.append(event)

    async def priority_handler(event, headers):
        priority_received.append(event)

    # Register handlers
    normal.register_handler("*", normal_handler)
    priority.register_handler("*", priority_handler)

    # Connect and start
    await normal.connect()
    await priority.connect()

    task_normal = asyncio.create_task(normal.run())
    task_priority = asyncio.create_task(priority.run())
    await asyncio.sleep(1)

    # Publish events with different priorities
    normal_event = CloudEvent(
        meta=EventMeta(type="test.normal", tenant_id="test", priority=EventPriority.NORMAL),
        data={"type": "normal"},
    )

    critical_event = CloudEvent(
        meta=EventMeta(type="alert.critical", tenant_id="test", priority=EventPriority.CRITICAL),
        data={"type": "critical"},
    )

    # Publish
    await bus.publish(normal_event)
    await bus.publish(critical_event)

    # Wait for processing
    await asyncio.sleep(2)

    # Verify routing
    assert len(normal_received) == 1
    assert len(priority_received) == 1
    assert normal_received[0].meta.priority == EventPriority.NORMAL
    assert priority_received[0].meta.priority == EventPriority.CRITICAL

    # Cleanup
    await normal.stop()
    await priority.stop()
    task_normal.cancel()
    task_priority.cancel()


@pytest.mark.asyncio
async def test_dlq_on_failure(bus):
    """Test DLQ for failed messages"""
    dlq_received = []

    # Consumer that always fails
    consumer_fail = bus.create_consumer("TEST_FAIL", "events.fail.>")

    async def failing_handler(event, headers):
        raise Exception("Simulated failure")

    consumer_fail.register_handler("fail.test", failing_handler)
    await consumer_fail.connect()

    # DLQ consumer
    consumer_dlq = bus.create_consumer("TEST_DLQ", "dlq.>")

    async def dlq_handler(event, headers):
        if event.meta.type.startswith("dlq."):
            dlq_received.append(event)

    consumer_dlq.register_handler("*", dlq_handler)
    await consumer_dlq.connect()

    # Start consumers
    task_fail = asyncio.create_task(consumer_fail.run())
    task_dlq = asyncio.create_task(consumer_dlq.run())
    await asyncio.sleep(1)

    # Publish failing event
    event = CloudEvent(meta=EventMeta(type="fail.test", tenant_id="test_tenant"), data={"will": "fail"})

    await bus.publish(event)

    # Wait for retries and DLQ
    await asyncio.sleep(20)  # Enough time for MAX_DELIVER attempts

    # Verify DLQ
    assert len(dlq_received) >= 1
    dlq_event = dlq_received[0]
    assert dlq_event.meta.type == "dlq.fail.test"
    assert dlq_event.data["original"]["meta"]["type"] == "fail.test"
    assert dlq_event.data["error"] == "Simulated failure"

    # Cleanup
    await consumer_fail.stop()
    await consumer_dlq.stop()
    task_fail.cancel()
    task_dlq.cancel()


@pytest.mark.asyncio
async def test_circuit_breaker(bus):
    """Test circuit breaker opens after failures"""
    from jeffrey.core.neuralbus.consumer import HandlerCircuitBreaker
    from jeffrey.core.neuralbus.exceptions import CircuitOpen

    breaker = HandlerCircuitBreaker(max_failures=3, timeout=1)

    async def failing_handler(event, headers):
        raise Exception("Always fails")

    # Fail 3 times to open circuit
    for i in range(3):
        with pytest.raises(Exception):
            await breaker.call(failing_handler, None, None)

    # Circuit should be open
    with pytest.raises(CircuitOpen):
        await breaker.call(failing_handler, None, None)

    # Wait for timeout
    await asyncio.sleep(1.5)

    # Should be half-open, next failure reopens
    with pytest.raises(Exception):
        await breaker.call(failing_handler, None, None)


@pytest.mark.asyncio
async def test_health_check(bus):
    """Test health check endpoint"""
    health = await bus.health_check()

    assert health["status"] == "healthy"
    assert health["publisher_connected"] is True
    assert "stream_info" in health
    assert "consumers" in health


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
