"""Test conversation simple - validation AGI"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator


async def test():
    print("=" * 80)
    print("🧪 TEST AGI SIMPLE")
    print("=" * 80)
    print()

    orch = AGIOrchestrator()
    await orch.initialize_llm()
    print("✅ Orchestrateur OK\n")

    # Test 1
    print("TEST 1: Présentation")
    msg = "Bonjour, qui es-tu ?"
    print(f"👤 {msg}")
    resp = await orch.chat_simple(msg)
    print(f"🤖 {resp}\n")

    # Test 2
    print("TEST 2: Question simple")
    msg = "Capitale de la France ?"
    print(f"👤 {msg}")
    resp = await orch.chat_simple(msg)
    print(f"🤖 {resp}\n")

    print("=" * 80)
    print("✅ TESTS TERMINÉS")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test())
