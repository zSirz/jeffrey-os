"""
Event contracts with CloudEvents specification
Strict validation with Pydantic
"""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class EventPriority(Enum):
    """Event priorities for lane routing"""

    CRITICAL = "critical"  # < 10ms target
    HIGH = "high"  # < 50ms target
    NORMAL = "normal"  # < 200ms target
    LOW = "low"  # best effort


class EventMeta(BaseModel):
    """CloudEvents metadata with Jeffrey extensions"""

    # CloudEvents standard fields
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(default="jeffrey.neuralbus")
    spec_version: Literal["1.0"] = "1.0"
    type: str  # Format: domain.entity.action
    time: datetime = Field(default_factory=datetime.utcnow)

    # CloudEvents optional fields
    subject: str | None = None
    data_content_type: str = Field(default="application/json")
    data_schema: str | None = None

    # Jeffrey extensions
    correlation_id: str | None = None
    causation_id: str | None = None
    priority: EventPriority = EventPriority.NORMAL
    tenant_id: str  # Required for multi-tenant

    # Proprietary watermark
    jeffrey_copyright: Literal["proprietary-jeffrey-os"] = "proprietary-jeffrey-os"

    # Event replay support
    replay_from: str | None = None

    @field_validator("tenant_id")
    @classmethod
    def tenant_required(cls, v):
        """Tenant ID is mandatory"""
        if not v:
            raise ValueError("tenant_id is required for multi-tenant isolation")
        return v

    @field_validator("type")
    @classmethod
    def validate_type_format(cls, v):
        """Validate event type format"""
        parts = v.split(".")
        if len(parts) < 2:
            raise ValueError(f"Type must be 'domain.entity[.action]' format, got: {v}")
        return v

    model_config = {
        "use_enum_values": True,
        "json_encoders": {datetime: lambda v: v.isoformat() + "Z"},
    }


class CloudEvent(BaseModel):
    """Complete CloudEvent with data payload"""

    meta: EventMeta
    data: dict[str, Any]

    def model_dump_json(self, **kwargs) -> str:
        """Serialize to JSON (Pydantic v2)"""
        return super().model_dump_json(**kwargs)

    model_config = {"json_encoders": {datetime: lambda v: v.isoformat() + "Z"}}


# Example typed payloads for common events
class UserEventData(BaseModel):
    """User-related event data"""

    user_id: str
    action: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SystemEventData(BaseModel):
    """System-related event data"""

    component: str
    status: str
    message: str | None = None
    metrics: dict[str, float] | None = None
