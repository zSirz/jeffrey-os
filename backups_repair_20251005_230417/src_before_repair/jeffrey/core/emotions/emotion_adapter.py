from typing import Any

from ..interfaces.protocols import EmotionModule
from ..utils.async_helpers import asyncify


class EmotionAdapter(EmotionModule):
    def __init__(self, impl: Any, name: str):
        self.impl = impl
        self.name = name

    async def analyze(self, text: str) -> dict[str, float]:
        # ordres de fallback
        if hasattr(self.impl, "analyze"):
            r = await asyncify(self.impl.analyze, text)
        elif hasattr(self.impl, "detect_emotion"):
            r = await asyncify(self.impl.detect_emotion, text)
        elif hasattr(self.impl, "detecter_emotion"):
            r = await asyncify(self.impl.detecter_emotion, text)
        elif hasattr(self.impl, "analyze_text"):
            r = await asyncify(self.impl.analyze_text, text)
        elif hasattr(self.impl, "process_text"):
            r = await asyncify(self.impl.process_text, text)
        elif hasattr(self.impl, "analyze_sentiment"):
            r = await asyncify(self.impl.analyze_sentiment, text)
        elif hasattr(self.impl, "get_emotion"):
            r = await asyncify(self.impl.get_emotion, text)
        else:
            r = None
        if not r:
            return {}
        if not isinstance(r, dict):
            r = {"value": r}
        # normalisation PAD minimale
        r.setdefault("valence", r.get("sentiment", 0.0))
        r.setdefault("arousal", abs(r.get("valence", 0.0)) * 0.5)
        r.setdefault("dominance", 0.5)
        return r

    async def update_state(self, state: dict[str, float]) -> None:
        if hasattr(self.impl, "update_state"):
            await asyncify(self.impl.update_state, state)

    def get_current_state(self) -> dict[str, float]:
        if hasattr(self.impl, "get_current_state"):
            try:
                return self.impl.get_current_state() or {}
            except Exception:
                return {}
        return {}

    def get_stats(self) -> dict[str, Any]:
        if hasattr(self.impl, "get_stats"):
            try:
                return self.impl.get_stats() or {}
            except Exception:
                return {}
        return {"adapter": self.name}
