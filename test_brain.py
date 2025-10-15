#!/usr/bin/env python3
"""Test simple du cerveau Jeffrey"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from jeffrey_brain import JeffreyBrain


async def test():
    logging.basicConfig(level=logging.DEBUG)

    brain = JeffreyBrain()
    await brain.boot()

    print("\n" + "=" * 60)
    print("TEST: Envoi d'un 'Hello'")
    print("=" * 60)

    # Test S1 (réflexe)
    await brain.process_input("Hello Jeffrey!", "test_user")
    await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("TEST: Question complexe")
    print("=" * 60)

    # Test S2 (complexe)
    await brain.process_input("What is the meaning of life?", "test_user")
    await asyncio.sleep(2)

    print("\n✅ Test terminé")


if __name__ == "__main__":
    asyncio.run(test())
