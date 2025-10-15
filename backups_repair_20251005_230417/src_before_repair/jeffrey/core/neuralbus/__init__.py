"""
NeuralBus - Event-driven message bus for Jeffrey OS
Production-ready with auto-optimization
"""

from .bus import NeuralBus, neural_bus
from .consumer import NeuralConsumer
from .contracts import CloudEvent, EventMeta, EventPriority
from .exceptions import CircuitOpen, NeuralBusError
from .publisher import NeuralPublisher

__all__ = [
    "neural_bus",
    "NeuralBus",
    "CloudEvent",
    "EventMeta",
    "EventPriority",
    "NeuralPublisher",
    "NeuralConsumer",
    "CircuitOpen",
    "NeuralBusError",
]

__version__ = "1.0.0"
