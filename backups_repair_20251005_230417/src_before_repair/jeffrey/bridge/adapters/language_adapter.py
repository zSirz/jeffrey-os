"""
Adaptateur EXPLICITE pour BrocaWernickeRegion.
"""

import inspect
from typing import Any

from jeffrey.bundle2.language.broca_wernicke import BrocaWernickeRegion


class LanguageAdapter:
    """Adaptateur pour région Language."""

    def __init__(self) -> None:
        self.mod = BrocaWernickeRegion()

    async def process(self, data: Any) -> dict[str, Any]:
        """Interface unifiée."""
        payload = data if isinstance(data, dict) else {"input": str(data)}

        # Méthodes typiques pour BrocaWernickeRegion
        for method in ("process", "generate"):
            if hasattr(self.mod, method):
                try:
                    fn = getattr(self.mod, method)
                    if inspect.iscoroutinefunction(fn):
                        result = await fn(payload)
                    else:
                        result = fn(payload)
                    return result if isinstance(result, dict) else {"result": result, "status": "ok"}
                except Exception as e:
                    return {
                        "error": f"BrocaWernickeRegion.{method}() failed: {e}",
                        "status": "error",
                    }

        raise RuntimeError(
            f"BrocaWernickeRegion sans process()/generate(). "
            f"Méthodes disponibles: {[m for m in dir(self.mod) if not m.startswith('_')]}"
        )
