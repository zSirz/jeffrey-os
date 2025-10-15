"""
Adaptateur EXPLICITE pour SQLiteMemoryStore.
"""

from typing import Any

from jeffrey.bundle2.memory.sqlite_store import SQLiteMemoryStore


class MemoryAdapter:
    """Adaptateur pour région Memory."""

    def __init__(self) -> None:
        self.store = SQLiteMemoryStore()

    async def process(self, data: Any) -> dict[str, Any]:
        """Interface unifiée avec effet I/O RÉEL."""
        if not isinstance(data, dict):
            data = {"input": str(data)}

        # Paramètres pour store
        intent = data.get("intent", "note")
        text = data.get("text") or data.get("input") or ""
        meta = data.get("meta", {})

        # Appel EXPLICITE - essayer store() puis remember()
        if hasattr(self.store, "store"):
            try:
                result = self.store.store(intent=intent, text=text, meta=meta)
                return result if isinstance(result, dict) else {"result": result, "status": "ok"}
            except Exception as e:
                return {"error": f"SQLiteMemoryStore.store() failed: {e}", "status": "error"}
        elif hasattr(self.store, "remember"):
            try:
                result = self.store.remember(text=text, context=meta)
                return result if isinstance(result, dict) else {"result": result, "status": "ok"}
            except Exception as e:
                return {"error": f"SQLiteMemoryStore.remember() failed: {e}", "status": "error"}
        else:
            raise RuntimeError(
                f"SQLiteMemoryStore n'a ni store() ni remember(). "
                f"Méthodes: {[m for m in dir(self.store) if not m.startswith('_')]}"
            )
