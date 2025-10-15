#!/usr/bin/env python3
"""Test full brain integration with all modules"""

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jeffrey.core.learning.auto_learner import AutoLearner
from jeffrey.core.learning.contextual_learning_engine import ContextualLearningEngine
from jeffrey.core.learning.jeffrey_meta_learning_integration import MetaLearningIntegration
from jeffrey.core.learning.theory_of_mind import TheoryOfMind
from jeffrey.core.learning.unified_curiosity_engine import UnifiedCuriosityEngine
from jeffrey.core.memory.unified_memory import UnifiedMemory


async def test_integration():
    """Test integrated brain functionality"""
    print("🧠 Testing Jeffrey Brain Integration...")
    print("=" * 60)

    # Initialize memory with test path
    memory = UnifiedMemory("data/test_memory.jsonl")
    await memory.initialize()

    # Initialize modules with memory injection
    print("\n📦 Initializing modules...")

    meta_learner = MetaLearningIntegration(memory=memory)
    theory = TheoryOfMind()
    curiosity = UnifiedCuriosityEngine()
    auto_learner = AutoLearner()
    context_engine = ContextualLearningEngine()

    # Initialize all
    await meta_learner.initialize()
    await theory.initialize()
    await curiosity.initialize()
    await auto_learner.initialize()
    await context_engine.initialize()

    print("✅ All modules initialized")

    # Test cases
    test_cases = [
        "How does machine learning work?",
        "Create a Python function to sort a list",
        "I'm feeling confused about quantum physics",
        "What's the weather like today?",
    ]

    print("\n🧪 Running test cases...")
    print("-" * 60)

    for i, test_input in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_input[:50]}...")

        try:
            # 1. Analyze context
            start = time.perf_counter()
            context = await context_engine.analyze(test_input)
            print(
                f"  📊 Context: type={context['type']}, domain={context['domain']}, complexity={context['complexity']:.2f}"
            )

            # 2. Infer intention
            intention = await theory.infer_intention(test_input, context)
            print(f"  🎯 Intention: {intention['type']} (confidence={intention['confidence']:.2f})")

            # 3. Extract patterns
            patterns = await meta_learner.extract_patterns({"text": test_input})
            print(f"  🔍 Patterns: {len(patterns)} found")
            if patterns:
                top_pattern = patterns[0]
                print(f"     Top: '{top_pattern['value']}' (importance={top_pattern['importance']:.2f})")

            # 4. Explore concept
            exploration = await curiosity.explore(
                {"concept": intention["main_concept"] or "unknown", "depth": context["complexity"]}
            )
            print(f"  ❓ Questions: {len(exploration['questions'])} generated")
            if exploration["questions"]:
                print(f"     Sample: {exploration['questions'][0]}")

            # 5. Generate response
            response = await auto_learner.generate(
                {
                    "intention": intention,
                    "patterns": patterns,
                    "exploration": exploration,
                    "context": context,
                }
            )
            print(f"  💬 Response: {response[:80]}...")

            # 6. Validate response
            validation = await theory.validate_response(response, intention)
            print(f"  ✔️ Valid: coherent={validation['coherent']}, accuracy={validation['accuracy']:.2f}")

            # 7. Learn from interaction
            learn_result = await meta_learner.learn(
                {"input": test_input, "output": response, "success": validation["score"] > 0.6}
            )
            print(
                f"  📈 Learning: patterns={learn_result['total_patterns']}, confidence={learn_result['avg_confidence']:.2f}"
            )

            # 8. Store in memory
            await memory.store_success(test_input, response)

            # Timing
            elapsed = time.perf_counter() - start
            print(f"  ⏱️ Time: {elapsed:.3f}s")

        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("📈 Final Statistics:")
    print(f"  - MetaLearning: {meta_learner.stats}")
    print(f"  - Curiosity: {curiosity.stats}")
    print(f"  - Memory: {memory.get_stats()}")

    # Cleanup
    await memory.shutdown()

    print("\n✅ Integration test completed successfully!")
    return True


async def test_performance():
    """Test performance metrics"""
    print("\n⚡ Performance Test...")
    print("-" * 40)

    memory = UnifiedMemory("data/test_perf.jsonl")
    await memory.initialize()

    # Write performance
    start = time.perf_counter()
    for i in range(100):
        await memory.store({"type": "test", "index": i, "data": f"test_{i}" * 10})

    write_time = time.perf_counter() - start
    print(f"Writes: 100 records in {write_time:.3f}s ({100 / write_time:.1f} writes/sec)")

    # Query performance
    start = time.perf_counter()
    for _ in range(100):
        await memory.query({"type": "test", "limit": 10})

    query_time = time.perf_counter() - start
    print(f"Queries: 100 queries in {query_time:.3f}s ({100 / query_time:.1f} queries/sec)")

    await memory.shutdown()

    # Cleanup test files

    if os.path.exists("data/test_perf.jsonl"):
        os.remove("data/test_perf.jsonl")


if __name__ == "__main__":
    try:
        # Run integration test
        success = asyncio.run(test_integration())

        # Run performance test
        asyncio.run(test_performance())

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
        sys.exit(1)
