"""Tests d'intégration robustes pour la Consciousness Loop"""

import asyncio
import sys
import time
import unittest

sys.path.insert(0, "src")

from jeffrey.core.consciousness_loop import ConsciousnessLoop


class TestConsciousnessLoopIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loop = ConsciousnessLoop()

    def test_initialization(self):
        """Test initialization"""

        async def _test():
            await self.loop.initialize()
            self.assertTrue(self.loop.initialized)
            return True

        result = asyncio.run(_test())
        self.assertTrue(result)

    def test_basic_processing(self):
        """Test basic input processing"""

        async def _test():
            await self.loop.initialize()
            result = await self.loop.process_input("Bonjour Jeffrey!")
            self.assertTrue(result["success"])
            self.assertIsNotNone(result["response"])
            return True

        result = asyncio.run(_test())
        self.assertTrue(result)

    def test_performance(self):
        """Test performance < 250ms"""

        async def _test():
            await self.loop.initialize()
            times = []
            for i in range(3):
                start = time.time()
                result = await self.loop.process_input(f"Test {i}")
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
                self.assertTrue(result["success"])
                self.assertLess(elapsed, 250, f"Too slow: {elapsed:.0f}ms")
            return True

        result = asyncio.run(_test())
        self.assertTrue(result)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestConsciousnessLoopIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
