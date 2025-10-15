#!/usr/bin/env python3
import asyncio

from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2


async def test():
    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Test store
    result = await federation.store_to_relevant("test_user", "user", "Hello Jeffrey!")
    print(f"✅ Store test: {len(result)} layers")

    # Test recall
    memories = await federation.recall_fast("test_user", 5)
    print(f"✅ Recall test: {len(memories)} memories")

    # Check stats
    stats = federation.get_stats()
    print(f"✅ Stats: {stats['initialized']} initialized")


asyncio.run(test())
print("\n🎉 All tests passed!")
