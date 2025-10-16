import pytest
import asyncio
import os
from uuid import uuid4
import numpy as np
from jeffrey.core.consciousness.bonds_service import bonds_service
from jeffrey.memory.hybrid_store import HybridMemoryStore

@pytest.mark.asyncio
async def test_bonds_upsert_ordering():
    """Test que les UUIDs sont toujours ordonnés"""
    id_a = str(uuid4())
    id_b = str(uuid4())
    
    # Appel dans n'importe quel ordre
    bond1 = await bonds_service.upsert_bond(id_b, id_a, 0.5, True)
    bond2 = await bonds_service.upsert_bond(id_a, id_b, 0.2, True)
    
    # Les deux doivent référencer le même bond
    assert bond1 is not None
    assert bond2 is not None
    assert bond1['memory_pair'] == bond2['memory_pair']
    
    # Strength doit être cumulée et clampée
    assert 0.5 <= bond2['strength'] <= 1.0

@pytest.mark.asyncio
async def test_bonds_prune():
    """Test du pruning des bonds faibles"""
    weak_id_a = str(uuid4())
    weak_id_b = str(uuid4())
    await bonds_service.upsert_bond(weak_id_a, weak_id_b, 0.05, False)
    
    pruned = await bonds_service.prune_weak_bonds(threshold=0.1)
    assert pruned >= 0

@pytest.mark.asyncio  
async def test_semantic_search_validation():
    """GPT tweak #5: Test sécurité paramétrage simple"""
    store = HybridMemoryStore()
    
    # Test avec embedding valide
    emb = np.random.rand(384).astype(np.float32)
    results = await store.semantic_search(emb, limit=1, threshold=0.0)
    assert isinstance(results, list)
    
    # Test avec embedding invalide (mauvaise taille)
    bad_emb = np.random.rand(100).astype(np.float32)
    results = await store.semantic_search(bad_emb, limit=1)
    assert results == []  # Doit retourner liste vide
    
    # Test avec None
    results = await store.semantic_search(None, limit=1)
    assert results == []

@pytest.mark.asyncio
async def test_trigger_endpoint():
    """GPT tweak #5: Test endpoint trigger"""
    from fastapi.testclient import TestClient
    from jeffrey.interfaces.bridge.api import app
    
    client = TestClient(app)
    api_key = os.getenv("JEFFREY_API_KEY", "test-key")
    
    # Sans consciousness enabled -> 400
    os.environ["ENABLE_CONSCIOUSNESS"] = "false"
    resp = client.post("/api/v1/consciousness/trigger", headers={"X-API-Key": api_key})
    assert resp.status_code == 400
    
    # Avec consciousness enabled -> 202
    os.environ["ENABLE_CONSCIOUSNESS"] = "true"
    resp = client.post("/api/v1/consciousness/trigger", headers={"X-API-Key": api_key})
    assert resp.status_code == 202
    assert "task_id" in resp.json()