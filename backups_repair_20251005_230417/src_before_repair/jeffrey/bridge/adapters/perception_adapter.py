"""
Adaptateur EXPLICITE pour InputParser.
Pas d'introspection, pas de try/except multiples.
"""

import inspect
from typing import Any

from jeffrey.core.input.input_parser import InputParser


class PerceptionAdapter:
    """Adaptateur pour région Perception."""

    def __init__(self) -> None:
        self.module = InputParser()

    async def process(self, data: Any) -> dict[str, Any]:
        """Interface unifiée process(dict) -> dict."""
        # Normalisation entrée
        payload = data if isinstance(data, dict) else {"input": str(data)}

        # Appel EXPLICITE : on essaye 2 méthodes documentées max
        for method in ("process", "parse"):
            if hasattr(self.module, method):
                fn = getattr(self.module, method)
                try:
                    if inspect.iscoroutinefunction(fn):
                        result = await fn(payload)
                    else:
                        result = fn(payload)

                    # Normalisation sortie
                    if isinstance(result, dict):
                        return result
                    return {"result": result, "status": "ok"}
                except Exception as e:
                    return {"error": f"InputParser.{method}() failed: {e}", "status": "error"}

        # Si aucune méthode trouvée
        raise RuntimeError(
            f"InputParser n'a ni process() ni parse(). "
            f"Méthodes disponibles: {[m for m in dir(self.module) if not m.startswith('_')]}"
        )
