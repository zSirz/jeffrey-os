from sqlalchemy import Column, Float, Boolean, DateTime, ForeignKey, UniqueConstraint, CheckConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
import uuid
from .base import Base

class EmotionalBond(Base):
    __tablename__ = 'emotional_bonds'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    memory_id_a = Column(UUID(as_uuid=True), ForeignKey('memories.id', ondelete='CASCADE'), nullable=False)
    memory_id_b = Column(UUID(as_uuid=True), ForeignKey('memories.id', ondelete='CASCADE'), nullable=False)
    strength = Column(Float, nullable=False, default=0.0)
    emotion_match = Column(Boolean, default=False)
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        UniqueConstraint('memory_id_a', 'memory_id_b', name='unique_bond_pair'),
        CheckConstraint('memory_id_a < memory_id_b', name='check_ordered_ids'),
        {'schema': None}
    )