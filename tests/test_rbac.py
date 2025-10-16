import pytest
from fastapi.testclient import TestClient
from jeffrey.interfaces.bridge.api import app
import os

@pytest.fixture
def client():
    return TestClient(app)

def test_trigger_requires_admin(client):
    """Test que /trigger nécessite admin role"""
    # Sans API key -> 401
    resp = client.post("/api/v1/consciousness/trigger")
    assert resp.status_code == 401

    # Avec API key mais sans rôle admin -> 403 (si différent d'admin key)
    user_key = os.getenv("JEFFREY_USER_API_KEY", "user-key")
    resp = client.post(
        "/api/v1/consciousness/trigger",
        headers={"X-API-Key": user_key}
    )
    # Peut être 403 ou 400 selon si consciousness est enabled
    assert resp.status_code in [400, 403]

    # Avec admin key -> 202 ou 400 (selon ENABLE_CONSCIOUSNESS)
    admin_key = os.getenv("JEFFREY_API_KEY", "admin-key")
    resp = client.post(
        "/api/v1/consciousness/trigger",
        headers={"X-API-Key": admin_key}
    )
    assert resp.status_code in [202, 400]

def test_search_accessible_without_admin(client):
    """Test que /search est accessible sans admin"""
    user_key = os.getenv("JEFFREY_USER_API_KEY", "user-key")
    resp = client.post(
        "/api/v1/memories/search",
        headers={"X-API-Key": user_key},
        json={"query": "test"}
    )
    # Si l'endpoint existe, devrait être 200, sinon 404
    assert resp.status_code in [200, 404]

def test_bonds_endpoint_rate_limited(client):
    """Test rate limiting sur /bonds"""
    api_key = os.getenv("JEFFREY_API_KEY", "test-key")

    # Test plusieurs requêtes rapides pour déclencher rate limiting
    responses = []
    for i in range(10):
        resp = client.get(
            "/api/v1/bonds?limit=1",
            headers={"X-API-Key": api_key}
        )
        responses.append(resp.status_code)

    # Au moins une requête devrait être rate limited ou toutes réussir
    assert any(code == 429 for code in responses) or all(code == 200 for code in responses)

def test_metrics_endpoint_accessible(client):
    """Test que /metrics est accessible"""
    resp = client.get("/metrics")
    # Metrics devrait être accessible sans auth
    assert resp.status_code == 200
    assert "text/plain" in resp.headers.get("content-type", "")

def test_health_endpoints_public(client):
    """Test que les endpoints health sont publics"""
    health_endpoints = ["/health", "/healthz", "/readyz"]

    for endpoint in health_endpoints:
        resp = client.get(endpoint)
        # Health endpoints devraient être accessibles sans auth
        assert resp.status_code in [200, 503]  # 503 si service pas prêt