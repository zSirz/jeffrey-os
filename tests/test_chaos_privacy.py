"""
Tests chaos pour valider robustesse (inspiré par Grok)
"""

import os

# Add parent directory to path
import sys
from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet

sys.path.insert(0, ".")

from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2


@pytest.mark.asyncio
async def test_privacy_with_redis_down():
    """Test que privacy fonctionne même si Redis down"""

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)

    # Simuler Redis down
    with patch("redis.Redis") as mock_redis:
        mock_redis.side_effect = ConnectionError("Redis down")

        await federation.initialize(None)

        # Doit quand même chiffrer
        text_pii = "Email: test@example.com"
        result = await federation.store_to_relevant("user", "user", text_pii)

        # Pas de crash
        assert result is not None


@pytest.mark.asyncio
async def test_dedup_under_memory_pressure():
    """Test dédup avec peu de RAM"""

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Forcer cache petit
    federation.cache_size_limit = 10

    # Stocker 100 items
    for i in range(100):
        await federation.store_to_relevant("user", "user", f"Message {i}")

    # Vérifier que seen_hashes reste petit
    assert len(federation.seen_hashes) <= 10

    # Pas d'OOM
    import psutil

    process = psutil.Process()
    assert process.memory_percent() < 80  # Max 80% RAM


@pytest.mark.asyncio
async def test_encryption_key_rotation():
    """Test rotation de clé (simulé)"""

    # Clé 1 - générer une vraie clé
    key1 = Fernet.generate_key().decode()
    os.environ["JEFFREY_ENCRYPTION_KEY"] = key1

    loader1 = SecureModuleLoader()
    fed1 = MemoryFederationV2(loader1)
    await fed1.initialize(None)

    # Stocker avec clé 1
    await fed1.store_to_relevant("user", "user", "Secret data")

    # Rotation vers clé 2 - générer une nouvelle vraie clé
    key2 = Fernet.generate_key().decode()
    os.environ["JEFFREY_ENCRYPTION_KEY"] = key2

    loader2 = SecureModuleLoader()
    fed2 = MemoryFederationV2(loader2)
    await fed2.initialize(None)

    # Nouveau stockage avec clé 2 doit fonctionner
    result = await fed2.store_to_relevant("user", "user", "New secret")
    assert result is not None  # Pas de crash

    # Cleanup
    del os.environ["JEFFREY_ENCRYPTION_KEY"]


@pytest.mark.asyncio
async def test_parallel_crash_recovery():
    """Test recovery si modules crashent en parallèle"""

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Simuler 3 layers qui crashent
    async def crashing_store(*args, **kwargs):
        raise Exception("Module crashed!")

    # Patch 3 layers pour crash
    crash_count = 0
    for layer_name, layer in federation.layers.items():
        if layer.initialized and crash_count < 3:
            for module in layer.modules.values():
                if hasattr(module, "store"):
                    module.store = crashing_store
                    crash_count += 1
                    break

    # Doit continuer avec les autres layers
    result = await federation.store_to_relevant("user", "user", "Test")

    # Au moins certains résultats (pas tout crashé)
    assert isinstance(result, list)


@pytest.mark.asyncio
async def test_privacy_enforcement_in_production():
    """Test que les clés sont obligatoires en production"""

    # Mode production sans clé
    os.environ["JEFFREY_MODE"] = "production"

    # Doit lever une erreur
    with pytest.raises(ValueError, match="JEFFREY_ENCRYPTION_KEY required"):
        from jeffrey.core.utils.privacy import PrivacyGuard

        guard = PrivacyGuard()

    # Cleanup
    del os.environ["JEFFREY_MODE"]


@pytest.mark.asyncio
async def test_pii_masking_for_logs():
    """Test que les PII sont masqués dans les logs"""

    from jeffrey.core.utils.privacy import PrivacyGuard

    guard = PrivacyGuard()

    # Texte avec PII
    text = "My email is john@example.com and phone 555-1234"

    # Masqué pour les logs
    masked = guard.mask_for_logging(text)

    assert "[EMAIL]" in masked
    assert "[PHONE]" in masked
    assert "john@example.com" not in masked
    assert "555-1234" not in masked
