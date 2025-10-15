"""
Tests d'intégration pour la Phase 1
"""

import asyncio

import httpx
import pytest

BASE_URL = "http://localhost:8000"


@pytest.mark.asyncio
async def test_health_check():
    """Vérifie que l'API est accessible"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        assert response.status_code == 200
        print("✅ API is healthy")


@pytest.mark.asyncio
async def test_status_with_bus_v2():
    """Vérifie que le bus V2 est dans le status"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/status")
        assert response.status_code == 200
        data = response.json()

        # Vérifier les nouveaux composants
        assert "bus_v2" in data
        assert "cognitive_core" in data
        assert "modules_loaded" in data

        if data["bus_v2"]:
            print(f"✅ Bus V2 stats: {data['bus_v2']}")

        if data["cognitive_core"]:
            print(f"✅ Cognitive Core: {data['cognitive_core']}")


@pytest.mark.asyncio
async def test_chat_v2():
    """Test le nouvel endpoint /chat"""
    async with httpx.AsyncClient() as client:
        response = await client.post(f"{BASE_URL}/chat", json={"user_id": "test_user", "text": "Bonjour Jeffrey !"})
        assert response.status_code == 200
        data = response.json()

        assert "reply" in data
        assert "timestamp" in data

        print(f"✅ Chat response: {data['reply']}")
        print(f"   Cognitive Core used: {data.get('cognitive_core', False)}")


@pytest.mark.asyncio
async def test_chat_sequence():
    """Test une séquence de messages"""
    async with httpx.AsyncClient() as client:
        messages = ["Bonjour !", "Comment vas-tu ?", "Au revoir"]

        for msg in messages:
            response = await client.post(f"{BASE_URL}/chat", json={"user_id": "test_sequence", "text": msg})
            assert response.status_code == 200
            data = response.json()
            print(f"User: {msg}")
            print(f"Jeffrey: {data['reply']}\n")


if __name__ == "__main__":
    # Lancer les tests
    asyncio.run(test_health_check())
    asyncio.run(test_status_with_bus_v2())
    asyncio.run(test_chat_v2())
    asyncio.run(test_chat_sequence())

    print("\n✅ Tous les tests Phase 1 sont passés !")
