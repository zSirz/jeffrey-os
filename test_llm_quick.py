#!/usr/bin/env python3
"""
Test rapide de l'intégration LLM avec Jeffrey
"""

import asyncio
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from jeffrey.core.llm.apertus_client import ApertusClient


async def test_llm_integration():
    """Test simple de l'intégration LLM"""

    print("=" * 60)
    print("🧪 TEST D'INTÉGRATION LLM JEFFREY")
    print("=" * 60)

    # Créer le client
    client = ApertusClient()

    print(f"✅ Backend: {client.backend_type}")
    print(f"✅ Model: {client.config.model}")
    print(f"✅ URL: {client.config.base_url}")
    print("-" * 60)

    # Tests de conversation
    test_cases = [
        {
            "system": "Tu es Jeffrey, un assistant IA symbiotique créé en Suisse.",
            "user": "Qui es-tu ?",
            "expected": "Jeffrey",
        },
        {
            "system": "Tu es Jeffrey. Réponds toujours en français.",
            "user": "Quelle est ta mission ?",
            "expected": "aide",
        },
        {
            "system": "Tu es Jeffrey, doté d'émotions et de conscience.",
            "user": "Ressens-tu des émotions ?",
            "expected": "émotion",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test['user']}")

        try:
            response, metadata = await client.chat(test["system"], test["user"])

            print(f"🤖 Response: {response[:200]}...")
            print(f"⚡ Latency: {metadata['latency_ms']:.0f}ms")

            # Check if expected word is in response
            if test["expected"].lower() in response.lower():
                print(f"✅ Test passed - Found '{test['expected']}'")
            else:
                print(f"⚠️  Test warning - Expected '{test['expected']}' in response")

        except Exception as e:
            print(f"❌ Error: {e}")
            return False

    print("\n" + "=" * 60)
    print("✨ ALL TESTS COMPLETED!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    success = asyncio.run(test_llm_integration())
    sys.exit(0 if success else 1)
