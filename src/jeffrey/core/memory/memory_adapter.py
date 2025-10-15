from typing import Any

from ..interfaces.protocols import MemoryModule
from ..utils.async_helpers import asyncify


class MemoryAdapter(MemoryModule):
    def __init__(self, impl: Any, name: str):
        self.impl = impl
        self.name = name

    def capabilities(self) -> list[str]:
        caps = []
        for m in ["store", "store_message", "add_memory", "remember", "process_interaction"]:
            if hasattr(self.impl, m):
                caps.append("store")
        for m in [
            "recall",
            "recall_recent",
            "recall_last_k",
            "get_recent_memories",
            "retrieve",
            "get_memories",
            "search_memories",
        ]:
            if hasattr(self.impl, m):
                caps.append("recall")
        for m in ["search", "search_memories", "find_relevant", "retrieve"]:
            if hasattr(self.impl, m):
                caps.append("search")
        if any(hasattr(self.impl, m) for m in ["consolidate", "cleanup", "optimize"]):
            caps.append("consolidate")
        return sorted(set(caps))

    async def store(self, payload: dict[str, Any]) -> bool:
        # ordres de fallback
        if hasattr(self.impl, "store"):
            res = await asyncify(self.impl.store, payload)
            return bool(res)
        if hasattr(self.impl, "store_message"):
            res = await asyncify(
                self.impl.store_message,
                payload.get("user_id"),
                payload.get("role"),
                payload.get("text"),
            )
            return bool(res)
        if hasattr(self.impl, "add_memory"):
            res = await asyncify(
                self.impl.add_memory,
                content=f"{payload.get('role')}: {payload.get('text')}",
                metadata=payload,
            )
            return bool(res)
        if hasattr(self.impl, "remember"):
            res = await asyncify(
                self.impl.remember,
                payload.get("text"),
                user_id=payload.get("user_id"),
                role=payload.get("role"),
            )
            return bool(res)
        # process_interaction: besoin d'un pairing user/assistant – on ignore ici
        return False

    async def recall(self, user_id: str, limit: int = 5) -> list[dict]:
        if hasattr(self.impl, "recall_recent"):
            r = await asyncify(self.impl.recall_recent, user_id, limit)
        elif hasattr(self.impl, "recall_last_k"):
            r = await asyncify(self.impl.recall_last_k, user_id, limit)
        elif hasattr(self.impl, "get_recent_memories"):
            r = await asyncify(self.impl.get_recent_memories, limit)
        elif hasattr(self.impl, "retrieve"):
            r = await asyncify(self.impl.retrieve, query=f"user:{user_id}", limit=limit)
        elif hasattr(self.impl, "get_memories"):
            r = await asyncify(self.impl.get_memories, user_id=user_id, count=limit)
        elif hasattr(self.impl, "search_memories"):
            r = await asyncify(self.impl.search_memories, "", limit=limit)
        else:
            r = []
        return self._normalize_list(r)

    async def search(self, query: str, user_id: str | None = None) -> list[dict]:
        if hasattr(self.impl, "search"):
            r = await asyncify(self.impl.search, query, user_id)
        elif hasattr(self.impl, "search_memories"):
            r = await asyncify(self.impl.search_memories, query)
        elif hasattr(self.impl, "find_relevant"):
            r = await asyncify(self.impl.find_relevant, query)
        elif hasattr(self.impl, "retrieve"):
            r = await asyncify(self.impl.retrieve, query=query)
        else:
            r = []
        return self._normalize_list(r)

    async def consolidate(self) -> bool:
        if hasattr(self.impl, "consolidate"):
            await asyncify(self.impl.consolidate)
            return True
        if hasattr(self.impl, "cleanup"):
            await asyncify(self.impl.cleanup)
            return True
        if hasattr(self.impl, "optimize"):
            await asyncify(self.impl.optimize)
            return True
        return True

    def get_stats(self) -> dict[str, Any]:
        if hasattr(self.impl, "get_stats"):
            try:
                return self.impl.get_stats() or {}
            except Exception:
                return {}
        return {"adapter": self.name}

    def _normalize_list(self, r) -> list[dict]:
        if not r:
            return []
        if not isinstance(r, list):
            r = [r]
        out = []
        for it in r:
            if isinstance(it, dict):
                out.append(it)
            elif isinstance(it, str):
                out.append({"text": it})
            elif hasattr(it, "to_dict"):
                out.append(it.to_dict())
        return out
