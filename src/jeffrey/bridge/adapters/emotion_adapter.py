"""
Adaptateur EXPLICITE pour EmotionEngine.
"""

import inspect
from typing import Any

from jeffrey.core.emotions.core.emotion_engine import EmotionEngine


class EmotionAdapter:
    """Adaptateur pour région Emotion."""

    def __init__(self) -> None:
        self.engine = EmotionEngine()

    async def process(self, data: Any) -> dict[str, Any]:
        """Interface unifiée avec gestion async."""
        # Normalisation
        payload = data if isinstance(data, dict) else {"input": str(data)}
        text = payload.get("text") or payload.get("input") or ""

        # Appel EXPLICITE - EmotionEngine.analyze() doit exister
        if not hasattr(self.engine, "analyze"):
            raise RuntimeError(
                f"EmotionEngine.analyze() manquant. Méthodes: {[m for m in dir(self.engine) if not m.startswith('_')]}"
            )

        try:
            fn = self.engine.analyze
            result = await fn(text) if inspect.iscoroutinefunction(fn) else fn(text)
            return result if isinstance(result, dict) else {"emotion": result, "status": "ok"}
        except Exception as e:
            return {"error": f"EmotionEngine.analyze() failed: {e}", "status": "error"}
