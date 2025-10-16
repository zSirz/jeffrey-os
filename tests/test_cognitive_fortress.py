"""
Tests for the Cognitive Fortress architecture
Validates robustness, auto-healing, and predictions
"""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock

# GPT CORRECTION: marquage asyncio
pytest_plugins = ('pytest_asyncio',)

@pytest.mark.asyncio
async def test_pipeline_idempotence():
    """Test that duplicate events are handled correctly"""
    from jeffrey.core.pipeline.thought_pipeline import ThoughtPipeline

    mock_bus = AsyncMock()
    mock_memory = Mock()
    mock_memory.store.return_value = True

    pipeline = ThoughtPipeline(mock_bus, mock_memory, None)

    # Send same event twice
    event = {
        "id": "test-123",
        "type": "emotion.detected",
        "data": {"text": "test", "emotion": "joy", "timestamp": "2024-01-01T00:00:00"}
    }

    await pipeline.on_emotion_detected(event)
    await pipeline.on_emotion_detected(event)  # Duplicate

    # Should only process once
    assert mock_memory.store.call_count == 1
    assert pipeline.metrics["events_processed"] == 1
    assert pipeline.metrics["events_skipped"] == 1


@pytest.mark.asyncio
async def test_circuit_breaker():
    """Test circuit breaker opens after failures"""
    from jeffrey.core.pipeline.thought_pipeline import ThoughtPipeline

    mock_bus = AsyncMock()
    mock_memory = Mock()
    mock_memory.store.side_effect = Exception("Memory failure")

    pipeline = ThoughtPipeline(
        mock_bus, mock_memory, None,
        breaker_threshold=2,
        max_retries=1
    )

    # Trigger failures
    for i in range(3):
        event = {
            "id": f"test-{i}",
            "type": "emotion.detected",
            "data": {"emotion": "test", "timestamp": f"2024-01-0{i+1}T00:00:00"}
        }
        await pipeline.on_emotion_detected(event)

    # Circuit should be open
    assert pipeline.breakers["memory"]["state"].value == "open"
    assert pipeline.metrics["circuit_opens"] > 0
    assert len(pipeline.get_dlq()) > 0


@pytest.mark.asyncio
async def test_orchestrator_auto_healing():
    """Test orchestrator can heal failed handlers"""
    from jeffrey.core.orchestration.cognitive_orchestrator import CognitiveOrchestrator

    mock_bus = AsyncMock()
    orchestrator = CognitiveOrchestrator(
        mock_bus,
        config={"error_threshold": 2, "auto_heal_interval": 0.1}
    )

    # Register failing handler
    fail_count = 0
    async def failing_handler(event):
        nonlocal fail_count
        fail_count += 1
        if fail_count < 3:
            raise Exception("Temporary failure")
        return "Success"

    orchestrator.register_handler("test.event", failing_handler)

    # Start auto-healing
    heal_task = asyncio.create_task(orchestrator.auto_heal())

    # Dispatch events
    for i in range(5):
        await orchestrator.dispatch("test.event", {"data": "test"})
        await asyncio.sleep(0.05)

    # Should have healed
    await asyncio.sleep(0.2)
    health = orchestrator.health[failing_handler]
    assert health["errors"] == 0  # Reset after healing

    heal_task.cancel()


@pytest.mark.asyncio
async def test_predictive_monitoring():
    """Test predictive issue detection"""
    from jeffrey.core.orchestration.cognitive_orchestrator import CognitiveOrchestrator

    orchestrator = CognitiveOrchestrator(AsyncMock())

    # Simulate problematic handler
    slow_handler = lambda e: None
    orchestrator.register_handler("test", slow_handler)
    orchestrator.health[slow_handler] = {
        "errors": 4,  # Near threshold
        "avg_latency_ms": 75,  # High latency
        "last_success": 0,  # Very old
        "total_calls": 20
    }

    predictions = await orchestrator.predict_issues()

    assert len(predictions["high_error_risk"]) > 0
    assert len(predictions["performance_degradation"]) > 0
    assert len(predictions["inactive_risk"]) > 0


@pytest.mark.asyncio
async def test_fortress_integration_manual():
    """
    Test the complete fortress manually (without live server)
    GPT CORRECTION: temps d'attente réalistes
    """
    from jeffrey.core.neuralbus.bus_facade import BusFacade
    from jeffrey.core.pipeline.thought_pipeline import ThoughtPipeline
    from jeffrey.core.orchestration.cognitive_orchestrator import CognitiveOrchestrator
    from jeffrey.core.neuralbus.events import make_event, EMOTION_DETECTED, MEMORY_STORED

    # Setup components
    bus = BusFacade(max_queue=100)
    bus.start()

    orchestrator = CognitiveOrchestrator(bus)
    await orchestrator.start()

    # Mock memory and consciousness
    mock_memory = Mock()
    mock_memory.store.return_value = True
    mock_memory.search.return_value = [{"emotion": "joy"}]

    mock_consciousness = Mock()
    mock_consciousness.process.return_value = {
        "summary": "Test thought",
        "state": "aware"
    }

    pipeline = ThoughtPipeline(
        bus=bus,
        memory=mock_memory,
        consciousness=mock_consciousness,
        orchestrator=orchestrator
    )

    # Register pipeline
    orchestrator.register_agent("thought_pipeline", pipeline)
    orchestrator.register_handler(EMOTION_DETECTED, pipeline.on_emotion_detected, priority=10)
    orchestrator.register_handler(MEMORY_STORED, pipeline.on_memory_stored, priority=8)

    # Wire bus to orchestrator
    bus.subscribe(EMOTION_DETECTED, lambda e: asyncio.create_task(orchestrator.dispatch(EMOTION_DETECTED, e)))
    bus.subscribe(MEMORY_STORED, lambda e: asyncio.create_task(orchestrator.dispatch(MEMORY_STORED, e)))

    try:
        # Send test emotions
        test_emotions = [
            "I feel amazing today",
            "This makes me sad",
            "I'm surprised by this"
        ]

        for i, text in enumerate(test_emotions):
            emotion_event = make_event(
                EMOTION_DETECTED,
                {
                    "text": text,
                    "emotion": "test_emotion",
                    "confidence": 0.9,
                    "timestamp": f"2024-01-0{i+1}T00:00:00"
                },
                source="test"
            )
            await bus.publish(emotion_event)
            # GPT CORRECTION: attente réaliste
            await asyncio.sleep(0.1)

        # Wait for processing
        await asyncio.sleep(0.5)

        # Verify processing
        pipeline_metrics = pipeline.get_metrics()
        orchestrator_stats = orchestrator.get_stats()

        assert pipeline_metrics["events_processed"] >= 3
        assert pipeline_metrics["memories_stored"] >= 3
        assert orchestrator_stats["agents_registered"] >= 1
        assert orchestrator_stats["handlers_registered"] >= 2

        print("✅ Manual fortress integration test passed!")

    finally:
        # Cleanup
        await orchestrator.stop()
        await bus.stop()


# Smoke test function
async def smoke_test_fortress():
    """
    Quick smoke test for fortress components
    """
    print("🧪 TESTING COGNITIVE FORTRESS COMPONENTS")
    print("=" * 50)

    await test_pipeline_idempotence()
    print("✅ Idempotence test passed")

    await test_circuit_breaker()
    print("✅ Circuit breaker test passed")

    await test_orchestrator_auto_healing()
    print("✅ Auto-healing test passed")

    await test_predictive_monitoring()
    print("✅ Predictive monitoring test passed")

    await test_fortress_integration_manual()
    print("✅ Manual integration test passed")

    print("\n🏰 ALL FORTRESS TESTS PASSED!")


if __name__ == "__main__":
    print("🏰 RUNNING COGNITIVE FORTRESS TESTS")
    asyncio.run(smoke_test_fortress())