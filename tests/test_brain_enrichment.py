"""
Tests for Phase 2 Brain Enrichment
Validates event-driven architecture, circadian rhythms, and self-reflection
"""
import asyncio
import pytest
import time
from datetime import datetime
from unittest.mock import Mock, AsyncMock

from jeffrey.core.neuralbus.bus_facade import BusFacade
from jeffrey.core.neuralbus.events import make_event, EMOTION_DETECTED, THOUGHT_GENERATED
from jeffrey.core.consciousness.self_reflection import SelfReflection
from jeffrey.core.biorhythms.circadian_runner import CircadianRunner


class TestBusFacade:
    """Test event bus functionality"""

    @pytest.mark.asyncio
    async def test_bus_creation(self):
        """Test bus can be created and started"""
        bus = BusFacade(max_queue=100)
        assert bus is not None
        bus.start()
        assert bus.get_stats()["running"] == True
        await bus.stop()

    @pytest.mark.asyncio
    async def test_event_publishing(self):
        """Test events can be published"""
        bus = BusFacade(max_queue=100)
        bus.start()

        # Create test event
        event = make_event(
            EMOTION_DETECTED,
            {"text": "test", "emotion": "joy", "confidence": 0.9},
            source="test"
        )

        # Should not throw
        await bus.publish(event)

        stats = bus.get_stats()
        assert stats["events_published"] >= 1

        await bus.stop()

    @pytest.mark.asyncio
    async def test_subscription_system(self):
        """Test event subscription works"""
        bus = BusFacade(max_queue=100)
        bus.start()

        received_events = []

        async def handler(event):
            received_events.append(event)

        # Subscribe to events
        bus.subscribe(EMOTION_DETECTED, handler)

        # Publish event
        event = make_event(
            EMOTION_DETECTED,
            {"text": "test", "emotion": "joy"},
            source="test"
        )
        await bus.publish(event)

        # Small delay for processing
        await asyncio.sleep(0.1)

        # Should have received the event
        assert len(received_events) >= 1

        await bus.stop()


class TestSelfReflection:
    """Test meta-cognition module"""

    @pytest.mark.asyncio
    async def test_reflection_creation(self):
        """Test self-reflection can be created"""
        bus = Mock()
        reflection = SelfReflection(bus, interval=3)

        assert reflection is not None
        assert reflection.interval == 3
        assert reflection.stats["thoughts_analyzed"] == 0

    @pytest.mark.asyncio
    async def test_thought_processing(self):
        """Test thought analysis"""
        bus = AsyncMock()
        reflection = SelfReflection(bus, interval=2)  # Trigger after 2 thoughts

        # Create test thought events
        thought1 = {
            "data": {
                "summary": "I am thinking about emotions",
                "emotion_context": "joy",
                "processing_time_ms": 45
            }
        }

        thought2 = {
            "data": {
                "summary": "This is another thought",
                "emotion_context": "joy",
                "processing_time_ms": 50
            }
        }

        # Process thoughts
        await reflection.on_thought(thought1)
        assert reflection.stats["thoughts_analyzed"] == 1

        # This should trigger reflection
        await reflection.on_thought(thought2)
        assert reflection.stats["thoughts_analyzed"] == 2

        # Check if meta-thought was generated
        assert reflection.stats["meta_thoughts_generated"] >= 1
        assert bus.publish.called

    def test_emotion_analysis(self):
        """Test emotion pattern analysis"""
        bus = Mock()
        reflection = SelfReflection(bus)

        # Add some emotions
        reflection.emotion_history = ["joy", "joy", "joy", "sadness"]

        analysis = reflection._analyze_emotions()

        assert analysis["dominant"] == "joy"
        assert analysis["stability"] > 0.5  # Joy is dominant
        assert "joy" in analysis["distribution"]

    def test_cognitive_analysis(self):
        """Test cognitive pattern analysis"""
        bus = Mock()
        reflection = SelfReflection(bus)

        # Add some thoughts with metrics
        reflection.thought_buffer = [
            {"context_size": 5, "processing_time_ms": 30, "emotion_context": "joy"},
            {"context_size": 8, "processing_time_ms": 40, "emotion_context": "joy"},
            {"context_size": 3, "processing_time_ms": 25, "emotion_context": "neutral"}
        ]

        analysis = reflection._analyze_cognitive_patterns()

        assert "coherence" in analysis
        assert "avg_depth" in analysis
        assert analysis["avg_depth"] > 0


class TestCircadianRunner:
    """Test temporal awareness module"""

    def test_circadian_creation(self):
        """Test circadian runner can be created"""
        bus = Mock()
        circadian = CircadianRunner(bus, interval_sec=30)

        assert circadian is not None
        assert circadian.interval == 30
        assert circadian.current_phase is not None

    def test_phase_detection(self):
        """Test time-of-day phase detection"""
        bus = Mock()
        circadian = CircadianRunner(bus)

        phase = circadian.get_current_phase()
        assert phase in ["night", "dawn", "morning", "noon", "afternoon", "dusk", "evening"]

    def test_energy_calculation(self):
        """Test energy level calculation"""
        bus = Mock()
        circadian = CircadianRunner(bus)

        energy = circadian.get_energy_level()
        assert 0.1 <= energy <= 1.0

    @pytest.mark.asyncio
    async def test_circadian_lifecycle(self):
        """Test start/stop lifecycle"""
        bus = AsyncMock()
        circadian = CircadianRunner(bus, interval_sec=1)  # Fast for testing

        # Start
        started = circadian.start()
        assert started == True

        # Should be running
        stats = circadian.get_stats()
        assert stats["running"] == True

        # Give it a moment to publish at least one event
        await asyncio.sleep(0.1)

        # Stop
        await circadian.stop()
        stats = circadian.get_stats()
        assert stats["running"] == False


class TestEventIntegration:
    """Test integration between modules"""

    @pytest.mark.asyncio
    async def test_full_enrichment_pipeline(self):
        """Test complete enrichment pipeline"""
        # Setup bus
        bus = BusFacade(max_queue=100)
        bus.start()

        # Setup self-reflection (trigger after 2 thoughts)
        reflection = SelfReflection(bus, interval=2)
        bus.subscribe(THOUGHT_GENERATED, reflection.on_thought)

        # Setup circadian
        circadian = CircadianRunner(bus, interval_sec=1)
        circadian.start()

        try:
            # Simulate emotion detection
            emotion_event = make_event(
                EMOTION_DETECTED,
                {
                    "text": "I feel great today",
                    "emotion": "joy",
                    "confidence": 0.95,
                    "all_scores": {"joy": 0.95, "neutral": 0.05}
                },
                source="test"
            )
            await bus.publish(emotion_event)

            # Simulate thoughts
            for i in range(3):
                thought_event = make_event(
                    THOUGHT_GENERATED,
                    {
                        "summary": f"Test thought {i}",
                        "state": "aware",
                        "emotion_context": "joy",
                        "processing_time_ms": 30 + i * 5,
                        "context_size": 5
                    },
                    source="test"
                )
                await bus.publish(thought_event)
                await asyncio.sleep(0.1)  # Small delay

            # Give time for processing
            await asyncio.sleep(0.5)

            # Check stats
            bus_stats = bus.get_stats()
            reflection_stats = reflection.get_stats()
            circadian_stats = circadian.get_stats()

            assert bus_stats["events_published"] >= 4  # 1 emotion + 3 thoughts
            assert reflection_stats["thoughts_analyzed"] >= 3
            assert reflection_stats["meta_thoughts_generated"] >= 1  # Should have triggered
            assert circadian_stats["running"] == True

        finally:
            # Cleanup
            await circadian.stop()
            await bus.stop()

    @pytest.mark.asyncio
    async def test_error_resilience(self):
        """Test system resilience to errors"""
        bus = BusFacade(max_queue=10)
        bus.start()

        # Create a handler that sometimes fails
        call_count = 0

        async def flaky_handler(event):
            nonlocal call_count
            call_count += 1
            if call_count % 2 == 0:
                raise Exception("Simulated failure")

        bus.subscribe(EMOTION_DETECTED, flaky_handler)

        try:
            # Send multiple events
            for i in range(5):
                event = make_event(
                    EMOTION_DETECTED,
                    {"text": f"test {i}", "emotion": "neutral"},
                    source="test"
                )
                await bus.publish(event)

            await asyncio.sleep(0.2)

            # Bus should still be functional
            stats = bus.get_stats()
            assert stats["running"] == True
            assert stats["events_published"] >= 5

        finally:
            await bus.stop()


if __name__ == "__main__":
    # Quick smoke test
    async def smoke_test():
        print("🧪 Running enrichment smoke test...")

        bus = BusFacade(max_queue=50)
        bus.start()

        reflection = SelfReflection(bus, interval=2)
        bus.subscribe(THOUGHT_GENERATED, reflection.on_thought)

        # Send test events
        await bus.publish(make_event(
            EMOTION_DETECTED,
            {"text": "test", "emotion": "joy", "confidence": 0.8},
            source="smoke_test"
        ))

        for i in range(3):
            await bus.publish(make_event(
                THOUGHT_GENERATED,
                {"summary": f"Thought {i}", "emotion_context": "joy"},
                source="smoke_test"
            ))

        await asyncio.sleep(0.5)

        # Check results
        bus_stats = bus.get_stats()
        reflection_stats = reflection.get_stats()

        print(f"✅ Bus events: {bus_stats['events_published']}")
        print(f"✅ Thoughts analyzed: {reflection_stats['thoughts_analyzed']}")
        print(f"✅ Meta-thoughts: {reflection_stats['meta_thoughts_generated']}")

        await bus.stop()
        print("🎉 Smoke test completed successfully!")

    asyncio.run(smoke_test())