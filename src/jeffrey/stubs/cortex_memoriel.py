"""Stub for cortex_memoriel - replaced by UnifiedMemory"""

from jeffrey.core.memory.unified_memory import UnifiedMemory


class CortexMemoriel:
    """Legacy compatibility wrapper for UnifiedMemory"""

    def __init__(self):
        self.memory = UnifiedMemory()

    async def store(self, data):
        return await self.memory.store(data)

    async def query(self, filter_dict):
        return await self.memory.query(filter_dict)


# Default instance for compatibility
cortex = CortexMemoriel()
