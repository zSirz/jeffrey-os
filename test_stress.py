#!/usr/bin/env python3
"""Stress test for UnifiedMemory"""

import asyncio
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


async def stress_test():
    """Grok's stress test: 10k concurrent writes"""
    from jeffrey.core.memory.unified_memory import UnifiedMemory

    print("⚡ STRESS TEST: 10,000 Concurrent Writes")
    print("=" * 40)

    memory = UnifiedMemory(backend="sqlite")
    await memory.initialize()

    start = time.time()

    # Generate memories
    async def write_memory(i):
        await memory.store({"text": f"Memory {i} with data {random.random()}", "type": "stress", "index": i})

    # Launch all writes
    tasks = [write_memory(i) for i in range(10000)]
    await asyncio.gather(*tasks)

    # Allow flush
    await asyncio.sleep(2)

    elapsed = time.time() - start
    rate = 10000 / elapsed

    print(f"✅ Completed: 10,000 writes in {elapsed:.2f}s")
    print(f"📊 Rate: {rate:.0f} ops/second")

    stats = memory.get_stats()
    overflows = stats.get("write_queue_overflows", 0)
    errors = stats.get("errors", 0)

    print(f"⚠️  Queue overflows: {overflows}")
    print(f"❌ Errors: {errors}")

    if rate > 1000 and errors == 0:
        print("🎉 STRESS TEST PASSED!")
    else:
        print("⚠️  Performance below expectations")

    await memory.shutdown()


if __name__ == "__main__":
    asyncio.run(stress_test())
