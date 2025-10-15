"""
Adaptateur EXPLICITE pour SelfAwarenessTracker.
"""

from typing import Any

from jeffrey.core.consciousness.self_awareness_tracker import SelfAwarenessTracker


class ConscienceAdapter:
    """Adaptateur pour région Conscience."""

    def __init__(self) -> None:
        self.tracker = SelfAwarenessTracker()

    async def process(self, data: Any) -> dict[str, Any]:
        """Interface unifiée."""
        payload = data if isinstance(data, dict) else {"input": str(data)}

        # Méthodes réelles de SelfAwarenessTracker
        for method in ("record_awareness", "get_evolution_report", "snapshots"):
            if hasattr(self.tracker, method):
                try:
                    fn = getattr(self.tracker, method)
                    if method == "record_awareness":
                        # record_awareness prend des paramètres spécifiques
                        state = {"user_input": payload.get("input", ""), "context": payload}
                        result = fn(state)
                    else:
                        result = fn()
                    return result if isinstance(result, dict) else {"result": result, "status": "ok"}
                except Exception as e:
                    return {
                        "error": f"SelfAwarenessTracker.{method}() failed: {e}",
                        "status": "error",
                    }

        raise RuntimeError(
            f"SelfAwarenessTracker: aucune méthode trouvée. "
            f"Méthodes disponibles: {[m for m in dir(self.tracker) if not m.startswith('_')]}"
        )
