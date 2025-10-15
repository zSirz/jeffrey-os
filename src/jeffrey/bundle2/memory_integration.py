"""Memory Integration Bridge for Bundle 2"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from jeffrey.bundle2.memory.sqlite_store import SQLiteMemoryStore


class MemoryBridge:
    """Bridge entre consciousness loop et SQLite store"""

    def __init__(self):
        self.store = SQLiteMemoryStore()
        self.session_id = f"session_{int(time.time())}"
        self.session_memories = []

    async def store_memory(self, content, metadata=None):
        """Store avec session tracking"""
        metadata = metadata or {}
        if "context" not in metadata:
            metadata["context"] = {}
        metadata["context"]["session_id"] = self.session_id

        mid = await self.store.store(content, metadata)
        self.session_memories.append(mid)
        return mid

    async def get_context(self, limit=5):
        """Get session context"""
        if not self.session_memories:
            return []

        memories = []
        for mid in self.session_memories[-limit:]:
            mem = await self.store.recall(mid)
            if mem:
                memories.append(mem)
        return memories

    def health_check(self):
        return {
            "status": "healthy",
            "module": __name__,
            "session_id": self.session_id,
            "session_memories": len(self.session_memories),
            "store": self.store.health_check(),
        }


def initialize():
    """Factory function"""
    return MemoryBridge()


def health_check():
    bridge = MemoryBridge()
    return bridge.health_check()
