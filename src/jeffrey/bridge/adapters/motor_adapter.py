"""
Adaptateur EXPLICITE pour NeuralResponseOrchestrator.
"""

import inspect
from typing import Any


class MotorAdapter:
    """Adaptateur pour région Motor - VRAI NeuralResponseOrchestrator."""

    def __init__(self) -> None:
        try:
            from jeffrey.core.response.neural_response_orchestrator import NeuralResponseOrchestrator

            self.mod = NeuralResponseOrchestrator(bus=None, memory=None, apertus_client=None)
        except Exception as e:
            raise RuntimeError(f"❌ Impossible d'utiliser NeuralResponseOrchestrator : {e}")

    async def process(self, data: Any) -> dict[str, Any]:
        """Interface unifiée."""
        payload = data if isinstance(data, dict) else {"input": str(data)}

        # Chercher méthodes NeuralResponseOrchestrator dans l'ordre de priorité
        for method in ("process", "run", "generate"):
            if hasattr(self.mod, method):
                try:
                    fn = getattr(self.mod, method)
                    result = await fn(payload) if inspect.iscoroutinefunction(fn) else fn(payload)
                    return result if isinstance(result, dict) else {"response": result, "status": "ok"}
                except Exception as e:
                    return {
                        "error": f"NeuralResponseOrchestrator.{method}() failed: {e}",
                        "status": "error",
                    }

        raise RuntimeError(
            f"NeuralResponseOrchestrator sans process/run/generate utilisable. "
            f"Méthodes: {[m for m in dir(self.mod) if not m.startswith('_')]}"
        )
