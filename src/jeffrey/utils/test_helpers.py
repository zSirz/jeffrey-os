"""
Test helpers for development without full infrastructure
"""

import time
from typing import Any


class NullBus:
    """Event bus that does nothing - for testing"""

    async def publish(self, event_type: str, data: dict[str, Any] = None):
        return None

    async def subscribe(self, event_type: str, callback):
        return None


class SimpleState:
    """Simple state manager for testing"""

    def __init__(self):
        self._state = {
            "mode": "idle",
            "latency_ms": 50,
            "cpu_usage": 30,
            "memory_usage": 40,
            "error_rate": 0.01,
            "timestamp": time.time(),
        }

    def get_state(self) -> dict[str, Any]:
        self._state["timestamp"] = time.time()
        return dict(self._state)

    def update(self, **kwargs):
        self._state.update(kwargs)


class DummyMemoryFederation:
    """Dummy memory for testing consolidation"""

    def __init__(self):
        self.memories = []
        self.stored_count = 0

    async def recall_from_all(self, query: str = "", max_results: int = 100, **kwargs):
        """Return fake memories for testing"""
        # Support both old and new API
        limit = kwargs.get("limit", kwargs.get("metadata_only", max_results))

        memories = []
        for i in range(min(5, limit)):
            memory = {
                "text": f"Memory {i}: {query or 'test'}",
                "metadata": {"index": i, "timestamp": time.time() - i * 100, "type": "test"},
                "timestamp": time.time() - i * 100,
            }

            # If metadata_only, remove text
            if kwargs.get("metadata_only", False):
                memory.pop("text", None)

            memories.append(memory)

        return memories

    async def store(self, document: dict[str, Any]):
        """Store memory (track for testing)"""
        self.memories.append(document)
        self.stored_count += 1
        return True

    async def get_stats(self) -> dict[str, Any]:
        """Get memory stats for testing"""
        return {"total_memories": len(self.memories), "stored_count": self.stored_count}
