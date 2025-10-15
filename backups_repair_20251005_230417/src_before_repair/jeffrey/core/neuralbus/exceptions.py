"""
Custom exceptions for NeuralBus
"""


class NeuralBusError(Exception):
    """Base exception for NeuralBus"""

    pass


class CircuitOpen(NeuralBusError):
    """Circuit breaker is open"""

    pass


class DeduplicationError(NeuralBusError):
    """Deduplication check failed"""

    pass


class PublishError(NeuralBusError):
    """Failed to publish event"""

    pass


class ConsumerError(NeuralBusError):
    """Consumer error"""

    pass
