#!/usr/bin/env python3
"""Complete test suite for Jeffrey Brain V2"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))


async def test_brain_complete():
    from jeffrey_brain import JeffreyBrain

    print("=" * 60)
    print("🧠 JEFFREY BRAIN V2 - COMPLETE INTEGRATION TEST")
    print("=" * 60)

    brain = None

    try:
        # 1. Initialize
        print("\n1️⃣ Initializing Brain...")
        brain = JeffreyBrain()
        await brain.initialize()
        print("✅ Brain initialized")

        # 2. Test various inputs
        print("\n2️⃣ Testing cognitive processing...")

        test_cases = [
            ("Hello Jeffrey!", "user1"),
            ("What is machine learning?", "user1"),
            ("I'm feeling happy today!", "user2"),
            ("Can you help me with Python?", "user1"),
            ("Thanks for your help!", "user1"),
            ("Tell me about artificial intelligence", "user3"),
        ]

        for i, (text, user_id) in enumerate(test_cases, 1):
            print(f"\n  Test {i}: '{text[:30]}...' from {user_id}")
            result = await brain.process_input(text, user_id)

            print(f"    Intention: {result['intention']}")
            print(f"    Emotion: {result.get('emotion', 'unknown')}")
            print(f"    Processing: {result['processing_time']:.3f}s")

            if result.get("curiosity"):
                print(f"    Curiosity: {result['curiosity']}")

            if result.get("errors"):
                print(f"    ⚠️ Errors: {result['errors']}")

            # Small delay between tests
            await asyncio.sleep(0.1)

        # 3. Test memory retrieval
        print("\n3️⃣ Testing memory retrieval...")
        memories = await brain.memory.retrieve("Python", limit=5)
        print(f"✅ Found {len(memories)} memories about 'Python'")

        # 4. Get statistics
        print("\n4️⃣ Getting statistics...")
        stats = await brain.get_stats()

        if stats["orchestrator"]:
            orch_stats = stats["orchestrator"]
            print("✅ Orchestrator stats:")
            print(f"   Total processes: {orch_stats['orchestrator']['total_processes']}")
            print(f"   Total errors: {orch_stats['orchestrator']['total_errors']}")
            print(f"   Active modules: {orch_stats['active_modules']}/{orch_stats['total_modules']}")

            for module in orch_stats["modules"]:
                error_rate = module["error_rate"]
                status = "✅" if error_rate < 0.1 else "⚠️"
                print(
                    f"   {status} {module['name']}: {module['process_count']} processes, {module['error_count']} errors"
                )

        # 5. Test parallel processing
        print("\n5️⃣ Testing parallel processing...")
        start = time.time()
        tasks = [brain.process_input(f"Test message {i}", f"user{i}") for i in range(5)]
        results = await asyncio.gather(*tasks)
        elapsed = time.time() - start
        print(f"✅ Processed 5 messages in parallel in {elapsed:.3f}s")

        # 6. Shutdown test
        print("\n6️⃣ Testing graceful shutdown...")
        await brain.shutdown()
        print("✅ Clean shutdown successful")

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED - BRAIN V2 FULLY OPERATIONAL!")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()

        if brain:
            await brain.shutdown()

        return False


async def test_error_handling():
    """Test that the system handles module errors gracefully"""
    print("\n🔧 Testing error handling...")

    from jeffrey.core.cognitive.base_module import BaseCognitiveModule
    from jeffrey_brain import JeffreyBrain

    # Create a faulty module
    class FaultyModule(BaseCognitiveModule):
        async def on_initialize(self):
            pass

        async def on_process(self, data):
            raise ValueError("Intentional test error")

    brain = JeffreyBrain()

    # Add the faulty module before initialization
    faulty = FaultyModule("FaultyModule")
    brain.cognitive_modules = [faulty]

    await brain.initialize()

    # Process should not crash despite faulty module
    result = await brain.process_input("Test with faulty module", "test")

    assert result["success"] == True
    assert result["errors"] is not None
    assert "FaultyModule" in str(result["errors"])

    print("✅ Error handling works - system continues despite module failure")

    await brain.shutdown()


if __name__ == "__main__":

    async def main():
        success = await test_brain_complete()

        if success:
            await test_error_handling()

        return success

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
