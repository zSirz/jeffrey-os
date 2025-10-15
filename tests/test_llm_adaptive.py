"""
Test adaptatif qui fonctionne avec vLLM ou Ollama
"""

import asyncio
import platform

import pytest

from jeffrey.core.llm.apertus_client import ApertusClient


async def test_adaptive():
    """Test qui s'adapte à l'environnement"""

    print(f"🔍 Platform: {platform.system()}")
    print(f"🔍 Architecture: {platform.machine()}")

    client = ApertusClient()
    print(f"📦 Using backend: {client.backend_type}")
    print(f"📦 Using model: {client.config.model}")

    # Test simple
    response, metadata = await client.chat(
        "Tu es Jeffrey, un assistant IA symbiotique créé en Suisse.",
        "Décris-toi en une phrase en français.",
    )

    print(f"\n🤖 Response: {response}")
    print(f"⚡ Latency: {metadata['latency_ms']:.2f}ms")
    print(f"📊 Model used: {metadata['model']}")

    # Test de cohérence
    assert len(response) > 10
    assert metadata["latency_ms"] < 30000  # 30 secondes max (pour la première requête Ollama)

    print("\n✅ All tests passed!")


@pytest.mark.asyncio
async def test_adaptive_pytest():
    """Version pytest du test adaptatif"""
    client = ApertusClient()

    response, metadata = await client.chat(
        "Tu es Jeffrey, un assistant IA symbiotique.", "Réponds simplement 'OK' si tu comprends."
    )

    assert response is not None
    assert len(response) > 0
    assert metadata["latency_ms"] > 0
    assert client.backend_type in ["ollama", "vllm"]


if __name__ == "__main__":
    asyncio.run(test_adaptive())
