"""Data contracts compatible with Pydantic v1 and v2"""

import warnings
from datetime import datetime
from enum import Enum
from typing import Any

try:
    # Try Pydantic v2
    from pydantic import BaseModel, Field, field_validator

    PYDANTIC_V2 = True
except ImportError:
    # Fallback to v1
    from pydantic import BaseModel, Field, validator

    PYDANTIC_V2 = False


class MessageSource(str, Enum):
    """Source of a message"""

    HUMAN = "human"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    MODULE = "module"


class EmotionType(str, Enum):
    """Emotion types"""

    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class MemoryMoment(BaseModel):
    """Memory moment with validation"""

    message: str = Field(..., min_length=1)
    source: MessageSource = Field(default=MessageSource.HUMAN)
    timestamp: datetime = Field(default_factory=datetime.now)
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_any(cls, **data):
        """Create MemoryMoment handling legacy fields with deprecation warning"""
        if "message" not in data and "human_message" in data:
            warnings.warn(
                "MemoryMoment.human_message is deprecated, use 'message' instead",
                DeprecationWarning,
                stacklevel=2,
            )
            data["message"] = data.pop("human_message")
        return cls(**data)

    if PYDANTIC_V2:

        @field_validator("message")
        @classmethod
        def message_not_empty(cls, v):
            if not v.strip():
                raise ValueError("Message cannot be empty")
            return v

    else:

        @validator("message")
        def message_not_empty(cls, v):
            if not v.strip():
                raise ValueError("Message cannot be empty")
            return v


class EmotionState(BaseModel):
    """Emotion state with confidence"""

    primary_emotion: EmotionType
    confidence: float = Field(..., ge=0.0, le=1.0)
    secondary_emotions: dict[EmotionType, float] = Field(default_factory=dict)

    if PYDANTIC_V2:

        @field_validator("secondary_emotions")
        @classmethod
        def validate_confidences(cls, v):
            for emotion, conf in v.items():
                if not 0 <= conf <= 1:
                    raise ValueError(f"Invalid confidence for {emotion}")
            return v

    else:

        @validator("secondary_emotions")
        def validate_confidences(cls, v):
            for emotion, conf in v.items():
                if not 0 <= conf <= 1:
                    raise ValueError(f"Invalid confidence for {emotion}")
            return v
