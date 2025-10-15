#!/usr/bin/env python3
"""
Simple test for autonomous loops without bus dependency
"""

import asyncio
import logging
import os
import sys

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configuration logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_loops_simple")


class MockBus:
    """Mock bus that doesn't require connection"""

    async def publish(self, event, dedup=False):
        logger.debug(f"Mock publish: {event}")
        return True


async def test_individual_loops():
    """Test individual loops without full system"""

    # Import loop classes
    from jeffrey.core.loops.awareness import AwarenessLoop
    from jeffrey.core.loops.curiosity import CuriosityLoop
    from jeffrey.core.loops.emotional_decay import EmotionalDecayLoop
    from jeffrey.core.loops.memory_consolidation import MemoryConsolidationLoop

    print("\n" + "=" * 60)
    print("🧪 Testing Individual Loops")
    print("=" * 60 + "\n")

    # Create mock bus
    bus = MockBus()

    # Test 1: Awareness Loop
    print("1. Testing Awareness Loop...")
    awareness = AwarenessLoop(bus=bus)
    await awareness.start()
    await asyncio.sleep(3)
    print(f"   ✅ Awareness cycles: {awareness.cycles}, errors: {awareness._err_streak}")
    await awareness.stop()

    # Test 2: Emotional Decay Loop
    print("\n2. Testing Emotional Decay Loop...")
    emotional = EmotionalDecayLoop(bus=bus)
    await emotional.start()
    await asyncio.sleep(3)
    print(f"   ✅ Emotional cycles: {emotional.cycles}, PAD state: {emotional.pad_state}")
    await emotional.stop()

    # Test 3: Memory Consolidation Loop
    print("\n3. Testing Memory Consolidation Loop...")
    memory = MemoryConsolidationLoop(bus=bus)
    await memory.start()
    await asyncio.sleep(3)
    print(f"   ✅ Memory cycles: {memory.cycles}, processed: {memory.memories_processed}")
    await memory.stop()

    # Test 4: Curiosity Loop
    print("\n4. Testing Curiosity Loop...")
    curiosity = CuriosityLoop(bus=bus)
    await curiosity.start()
    await asyncio.sleep(3)
    print(f"   ✅ Curiosity cycles: {curiosity.cycles}, questions: {len(curiosity.questions_queue)}")
    await curiosity.stop()

    print("\n" + "=" * 60)
    print("✅ All loops tested successfully!")
    print("=" * 60)


async def main():
    """Main test runner"""
    try:
        await test_individual_loops()
        print("\n🎉 Tests completed successfully!")
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
