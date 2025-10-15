"""
Jeffrey OS - Autonomous Loops System
Phase 2.1 Ultimate Implementation

4 boucles intelligentes avec RL adaptatif, privacy, et symbiose
"""

from .awareness import AwarenessLoop
from .base import BaseLoop
from .curiosity import CuriosityLoop
from .emotional_decay import EmotionalDecayLoop
from .gates import create_budget_gate, sanitize_event_data
from .loop_manager import LoopManager
from .memory_consolidation import MemoryConsolidationLoop

__all__ = [
    # Base
    "BaseLoop",
    # Gates & Privacy
    "create_budget_gate",
    "sanitize_event_data",
    # Loops
    "AwarenessLoop",
    "EmotionalDecayLoop",
    "MemoryConsolidationLoop",
    "CuriosityLoop",
    # Manager
    "LoopManager",
]

# Version
__version__ = "2.1.0"
