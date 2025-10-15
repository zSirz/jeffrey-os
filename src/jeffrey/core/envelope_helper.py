"""Helper to create compatible envelopes regardless of implementation"""

from typing import Any

from jeffrey.utils.logger import get_logger

logger = get_logger("EnvelopeHelper")

# Try to import bus version first (preferred)
try:
    from jeffrey.core.neural_bus import NeuralEnvelope as _BusEnvelope

    _ENVELOPE_TYPE = "bus"
    logger.debug("Using NeuralBus envelope (no namespace)")
except ImportError:
    _BusEnvelope = None
    _ENVELOPE_TYPE = None

# Try legacy version with namespace
try:
    from jeffrey.core.neural_envelope import NeuralEnvelope as _LegacyEnvelope

    _LEGACY_AVAILABLE = True
    logger.debug("Legacy envelope available (with namespace)")
except ImportError:
    _LegacyEnvelope = None
    _LEGACY_AVAILABLE = False

if not (_ENVELOPE_TYPE or _LEGACY_AVAILABLE):
    raise ImportError("No NeuralEnvelope class found in project")


def make_envelope(
    topic: str,
    payload: dict[str, Any],
    ns: str = "core",
    salience: float = 0.5,
    confidence: float = 0.5,
) -> Any:
    """
    Create a compatible envelope regardless of which implementation exists.
    Automatically handles namespace differences.
    """
    if _ENVELOPE_TYPE == "bus" and _BusEnvelope is not None:
        # Bus version doesn't use namespace
        return _BusEnvelope(topic=topic, payload=payload)
    elif _LegacyEnvelope is not None:
        # Legacy version requires namespace
        return _LegacyEnvelope(ns=ns, topic=topic, payload=payload, salience=salience, confidence=confidence)
    else:
        # Fallback: create a simple dict if no class available
        logger.warning("No envelope class found, using dict fallback")
        return {
            "topic": topic,
            "payload": payload,
            "ns": ns,
            "timestamp": __import__("time").time(),
        }


# Export
__all__ = ["make_envelope"]
