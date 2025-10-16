import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_health_endpoints(test_client: AsyncClient):
    """Test health and ready endpoints"""
    # Health check
    response = await test_client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

    # Ready check
    response = await test_client.get("/readyz")
    assert response.status_code == 200
    assert response.json()["ready"] == True

@pytest.mark.asyncio
async def test_memory_crud_flow(test_client: AsyncClient):
    """Test complete CRUD flow for memories"""
    # Create memory
    memory_data = {
        "text": "Integration test memory",
        "emotion": "curiosity",
        "confidence": 0.75
    }

    response = await test_client.post("/api/v1/memories/", json=memory_data)
    assert response.status_code == 200
    created = response.json()
    assert created["text"] == memory_data["text"]
    assert "id" in created

    # Retrieve recent
    response = await test_client.get("/api/v1/memories/recent?hours=1&limit=10")
    assert response.status_code == 200
    memories = response.json()
    assert len(memories) > 0

    # Search
    response = await test_client.get("/api/v1/memories/search?query=Integration&limit=5")
    assert response.status_code == 200
    results = response.json()
    assert any("Integration" in m["text"] for m in results)

@pytest.mark.asyncio
async def test_emotion_detection_persistence(test_client: AsyncClient):
    """Test emotion detection saves to memory"""
    # Detect emotion
    response = await test_client.post(
        "/api/v1/emotion/detect",
        json={"text": "I am very happy about the test results!"}
    )
    assert response.status_code == 200
    emotion_result = response.json()
    assert "emotion" in emotion_result
    assert "confidence" in emotion_result

    # Verify saved in memories
    response = await test_client.get("/api/v1/memories/recent?hours=1")
    assert response.status_code == 200
    memories = response.json()
    assert any("happy" in m.get("text", "") for m in memories)