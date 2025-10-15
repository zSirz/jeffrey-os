"""
Adaptateur EXPLICITE pour TripleMemorySystem.
"""

from typing import Any

from jeffrey.core.memory.triple_memory import TripleMemorySystem


class IntegrationAdapter:
    """Adaptateur pour région Integration."""

    def __init__(self) -> None:
        self.sys = TripleMemorySystem()

    async def process(self, data: Any) -> dict[str, Any]:
        """Interface unifiée."""
        payload = data if isinstance(data, dict) else {"input": str(data)}

        # Méthodes réelles de TripleMemorySystem
        for method in ("remember", "recall_similar", "get_stats"):
            if hasattr(self.sys, method):
                try:
                    fn = getattr(self.sys, method)
                    if method == "remember":
                        # remember prend du texte
                        text = payload.get("input", "test integration")
                        result = fn(text, metadata=payload)
                    elif method == "recall_similar":
                        query = payload.get("input", "test")
                        result = fn(query, k=3)
                    else:  # get_stats
                        result = fn()
                    return result if isinstance(result, dict) else {"result": result, "status": "ok"}
                except Exception as e:
                    return {
                        "error": f"TripleMemorySystem.{method}() failed: {e}",
                        "status": "error",
                    }

        raise RuntimeError(
            f"TripleMemorySystem: aucune méthode trouvée. "
            f"Méthodes disponibles: {[m for m in dir(self.sys) if not m.startswith('_')]}"
        )
