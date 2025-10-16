from pydantic import BaseModel, Field
from typing import Optional, Dict
from datetime import datetime

class MemoryBase(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    emotion: Optional[str] = Field(None, max_length=50)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    metadata: Optional[Dict] = Field(default_factory=dict)

class MemoryCreate(MemoryBase):
    pass

class MemoryResponse(MemoryBase):
    id: str
    timestamp: datetime
    processed: bool = False

    class Config:
        from_attributes = True

class EmotionEventCreate(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)

class EmotionEventResponse(BaseModel):
    id: str
    text: str
    predicted_emotion: str
    confidence: float
    all_scores: Dict[str, float]
    timestamp: datetime
    processing_time_ms: Optional[float]

    class Config:
        from_attributes = True