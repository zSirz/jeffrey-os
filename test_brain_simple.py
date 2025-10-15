#!/usr/bin/env python3
"""Simple test for new brain modules"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jeffrey.core.learning.jeffrey_meta_learning_integration import MetaLearningIntegration
from jeffrey.core.memory.unified_memory import UnifiedMemory


async def test_simple():
    """Simple integration test"""
    print("🧠 Testing Jeffrey Brain Modules...")
    print("=" * 60)

    # Test UnifiedMemory
    print("\n📚 Testing UnifiedMemory...")
    memory = UnifiedMemory("data/test_memory_simple.jsonl")
    await memory.initialize()

    # Store some test data
    await memory.store({"type": "test", "content": "Hello Jeffrey"})
    await memory.store({"type": "learning", "pattern": "greeting"})

    # Query data
    results = await memory.query({"type": "test"})
    print(f"  ✅ Memory query returned {len(results)} results")

    # Test MetaLearning
    print("\n🎓 Testing MetaLearningIntegration...")
    meta_learner = MetaLearningIntegration(memory=memory)
    await meta_learner.initialize()

    # Extract patterns
    patterns = await meta_learner.extract_patterns({"text": "How does machine learning work?"})
    print(f"  ✅ Extracted {len(patterns)} patterns")
    if patterns:
        print(f"     Top pattern: {patterns[0]['value']}")

    # Learn from example
    learn_result = await meta_learner.learn(
        {"input": "What is AI?", "output": "AI is artificial intelligence", "success": True}
    )
    print(f"  ✅ Learning complete: {learn_result['total_patterns']} patterns")

    # Test memory stats
    stats = memory.get_stats()
    print("\n📊 Memory Statistics:")
    print(f"  - Total items: {stats['total_items']}")
    print(f"  - Writes: {stats['writes']}")
    print(f"  - Queries: {stats['queries']}")

    # Cleanup
    await memory.shutdown()
    print("\n✅ All tests passed!")
    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(test_simple())
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
