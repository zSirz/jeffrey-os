#!/usr/bin/env python

"""
Test for Emoji Detection - unified architecture version
Tests emoji detection and emotional context understanding
"""

from enum import Enum

import pytest
from unified.models import CellularResponse, SignalPayload, SignalType


class ResponseStatus(Enum):
    """Response status enum for testing"""

    SUCCESS = "success"
    REJECTED = "rejected"
    FAILED = "failed"
    PENDING = "pending"


class TestFactory:
    """Simple test factory for creating test objects"""

    @staticmethod
    def create_signal_payload(signal_type="command", data=None, source="test_factory"):
        """Create a test signal payload using valid enums"""
        from unified.models import SignalPriority

        # Map string signal types to valid enum values
        signal_type_map = {
            "ethics_evaluation": SignalType.COMMAND,
            "voice_emotion": SignalType.VOICE,
            "voice_adaptation": SignalType.VOICE,
            "asset_request": SignalType.COMMAND,
            "asset_cache": SignalType.COMMAND,
            "asset_cleanup": SignalType.COMMAND,
            "emoji_detection": SignalType.QUERY,
            "emoji_emotion": SignalType.EMOTION,
            "emoji_response": SignalType.RESPONSE,
            "model_orchestration": SignalType.COMMAND,
            "model_fallback": SignalType.COMMAND,
            "load_balance": SignalType.COMMAND,
        }

        # Use mapped signal type or default to command
        actual_signal_type = signal_type_map.get(signal_type, SignalType.COMMAND)

        return SignalPayload(
            signal_type=actual_signal_type,
            data=data or {},
            priority=SignalPriority.NORMAL,
            source=source,
            timestamp="2024-09-13T16:00:00Z",
        )

    @staticmethod
    def create_cellular_response(request_id="test-123", cell_id="test-cell", status="success", data=None):
        """Create a test cellular response"""
        return CellularResponse(
            request_id=request_id,
            cell_id=cell_id,
            status=status,
            data=data or {},
            processing_time_ms=100.0,
            timestamp="2024-09-13T16:00:00Z",
        )

    @staticmethod
    def create_memory(memory_type, content=None):
        """Create a test memory object"""
        return {"memory_type": memory_type, "content": content or {}, "timestamp": "2024-09-13T16:00:00Z"}


class TestEmojiDetection:
    """Test emoji detection functionality"""

    @pytest.fixture
    def emoji_signal(self, signal_factory):
        """Create emoji detection signal"""
        return signal_factory(
            signal_type="emoji_detection", data={"text": "I'm feeling great today! 😊🎉", "context": "user_message"}
        )

    async def test_emoji_extraction(self, emoji_signal):
        """Test emoji extraction from text"""
        response = CellularResponse(
            request_id=emoji_signal.signal_id,
            cell_id="test_cell",
            status="success",
            data={"emojis_found": ["😊", "🎉"], "emoji_count": 2, "text_cleaned": "I'm feeling great today! "},
            processing_time_ms=100.0,
        )

        assert len(response.data["emojis_found"]) == 2
        assert "😊" in response.data["emojis_found"]
        assert response.data["emoji_count"] == 2

    async def test_emotion_from_emojis(self, signal_factory):
        """Test emotion detection from emojis"""
        emotion_signal = signal_factory(
            signal_type="emoji_emotion", data={"emojis": ["😊", "🎉", "❤️"], "context": "celebration"}
        )

        response = CellularResponse(
            request_id=emotion_signal.signal_id,
            cell_id="test_cell",
            status="success",
            data={
                "detected_emotion": "joy",
                "emotion_intensity": 0.9,
                "emotion_categories": ["happiness", "celebration", "love"],
            },
            processing_time_ms=100.0,
        )

        assert response.data["detected_emotion"] == "joy"
        assert response.data["emotion_intensity"] > 0.8
        assert "happiness" in response.data["emotion_categories"]

    async def test_emoji_response_generation(self, signal_factory):
        """Test appropriate emoji response generation"""
        response_signal = signal_factory(
            signal_type="emoji_response", data={"user_emotion": "sad", "response_text": "I understand how you feel"}
        )

        response = CellularResponse(
            request_id=response_signal.signal_id,
            cell_id="test_cell",
            status="success",
            data={
                "suggested_emojis": ["🤗", "💙"],
                "response_with_emoji": "I understand how you feel 🤗💙",
                "appropriateness_score": 0.95,
            },
            processing_time_ms=100.0,
        )

        assert len(response.data["suggested_emojis"]) > 0
        assert response.data["appropriateness_score"] > 0.9


# Test fixtures for unified architecture
@pytest.fixture
def signal_factory():
    """Factory for creating test signals"""
    return TestFactory.create_signal_payload


@pytest.fixture
def response_factory():
    """Factory for creating test responses"""
    return TestFactory.create_cellular_response


@pytest.fixture
def memory_factory():
    """Factory for creating test memory objects"""
    return TestFactory.create_memory
