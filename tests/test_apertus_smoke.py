"""
Test smoke pour valider l'intégration Apertus
"""

import asyncio

import pytest

from jeffrey.core.llm.apertus_client import ApertusClient
from jeffrey.core.llm.hybrid_bridge import HybridOrchestrator


@pytest.mark.asyncio
async def test_apertus_basic():
    """Test basique du client Apertus"""
    client = ApertusClient()

    response, metadata = await client.chat("Tu es Jeffrey, un assistant utile.", "Explique la symbiose en 2 phrases.")

    assert response is not None
    assert len(response) > 10
    assert metadata["latency_ms"] < 5000
    print(f"✅ Apertus response: {response[:100]}...")


@pytest.mark.asyncio
async def test_hybrid_routing():
    """Test du routing hybride"""
    apertus = ApertusClient()
    orchestrator = HybridOrchestrator(apertus)

    # Test requête simple (devrait aller vers Apertus)
    simple_query = {"content": "Quelle est la capitale de la France?", "type": "question"}

    result = await orchestrator.process(simple_query)
    assert result["success"]
    assert result["routing"] == "local"

    # Test requête complexe (devrait aller vers externe ou hybrid)
    complex_query = {
        "content": "Écris une fonction Python qui calcule la suite de Fibonacci de manière optimisée",
        "type": "code",
    }

    result = await orchestrator.process(complex_query)
    assert result["success"]
    assert result["routing"] in [
        "external",
        "hybrid_validated",
        "local",
    ]  # local possible si pas de bridge

    # Afficher les stats
    stats = orchestrator.get_stats()
    print(f"📊 Routing stats: {stats}")


if __name__ == "__main__":
    asyncio.run(test_apertus_basic())
    asyncio.run(test_hybrid_routing())
