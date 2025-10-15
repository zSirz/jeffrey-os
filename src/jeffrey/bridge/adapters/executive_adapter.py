"""
Adaptateur EXPLICITE pour AgiOrchestrator.
"""

import inspect
from typing import Any


class ExecutiveAdapter:
    """Adaptateur pour région Executive - VRAI AgiOrchestrator."""

    def __init__(self) -> None:
        try:
            from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator

            self.exec = AGIOrchestrator()
        except Exception as e:
            raise RuntimeError(f"❌ Impossible d'utiliser AgiOrchestrator : {e}")

    async def process(self, data: Any) -> dict[str, Any]:
        """Interface unifiée."""
        payload = data if isinstance(data, dict) else {"input": str(data)}

        # Chercher méthodes AgiOrchestrator dans l'ordre de priorité
        for method in ("decide", "plan", "execute"):
            if hasattr(self.exec, method):
                try:
                    fn = getattr(self.exec, method)
                    result = await fn(payload) if inspect.iscoroutinefunction(fn) else fn(payload)
                    return result if isinstance(result, dict) else {"result": result, "status": "ok"}
                except Exception as e:
                    return {"error": f"AgiOrchestrator.{method}() failed: {e}", "status": "error"}

        raise RuntimeError(
            f"AgiOrchestrator sans decide/plan/execute utilisable. "
            f"Méthodes: {[m for m in dir(self.exec) if not m.startswith('_')]}"
        )
