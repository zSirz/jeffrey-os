"""
Event normalization - CloudEvent 1.0 compliant but lightweight
Combines GPT's structure with Grok's pragmatism
"""
from typing import TypedDict, Any, Dict, Optional
from datetime import datetime, timezone
from uuid import uuid4
import json

class Event(TypedDict):
    """CloudEvent 1.0 compliant event structure"""
    id: str
    specversion: str
    type: str
    source: str
    time: str
    data: Dict[str, Any]
    datacontenttype: Optional[str]
    subject: Optional[str]

def make_event(
    event_type: str,
    data: Dict[str, Any],
    source: str = "jeffrey.core",
    subject: Optional[str] = None,
    request_id: Optional[str] = None
) -> Event:
    """
    Factory to create normalized events
    Includes request_id for tracing as suggested by GPT
    """
    event = {
        "id": str(uuid4()),
        "specversion": "1.0",
        "type": event_type,
        "source": source,
        "time": datetime.now(timezone.utc).isoformat(),
        "data": data,
        "datacontenttype": "application/json"
    }

    if subject:
        event["subject"] = subject

    if request_id:
        event["data"]["request_id"] = request_id

    return event

# Standard event types (constants)
EMOTION_DETECTED = "emotion.ml.detected.v1"
MEMORY_STORED = "memory.stored.v1"
MEMORY_RECALLED = "memory.recalled.v1"
THOUGHT_GENERATED = "cognition.thought.v1"
META_THOUGHT_GENERATED = "cognition.meta_thought.v1"
CIRCADIAN_UPDATE = "state.circadian.updated.v1"
DREAM_TRIGGERED = "dream.trigger.v1"
SELF_REFLECTION = "cognition.self_reflection.v1"