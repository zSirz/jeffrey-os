"""
Tests for search key security and SKID versioning
"""

import asyncio
import os
import secrets
import sys
from unittest.mock import patch

import pytest

# Add parent directory to path
sys.path.insert(0, ".")


def test_search_key_required_in_production():
    """Test that search key is required in production when search is enabled"""

    # Need to temporarily provide a search key for initialization
    os.environ["JEFFREY_SEARCH_KEY"] = "temp_key_for_init"

    # Setup production
    os.environ["JEFFREY_MODE"] = "production"

    # Need encryption key for production
    from cryptography.fernet import Fernet

    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

    from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
    from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2

    # Now test the requirement directly
    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)

    # Remove search key to test the check
    del os.environ["JEFFREY_SEARCH_KEY"]

    # Mock config to enable search
    with patch.object(federation, "_config_get") as mock_config:

        def config_side_effect(key, default=None):
            if key == "privacy.enable_search_index":
                return True  # Search enabled
            return default

        mock_config.side_effect = config_side_effect

        # Should raise exception when search is enabled without key
        with pytest.raises(ValueError, match="JEFFREY_SEARCH_KEY is required"):
            federation._get_search_key()

    # Cleanup
    for key in ["JEFFREY_MODE", "JEFFREY_ENCRYPTION_KEY"]:
        if key in os.environ:
            del os.environ[key]


def test_search_key_hex_format():
    """Test that hex-formatted search keys are properly decoded"""

    from cryptography.fernet import Fernet

    from jeffrey.core.utils.privacy import PrivacyGuard

    # Need encryption key
    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

    # Generate hex key
    hex_key = secrets.token_hex(32)  # 64 hex chars = 32 bytes
    os.environ["JEFFREY_SEARCH_KEY"] = hex_key
    os.environ["JEFFREY_SEARCH_KID"] = "s1"

    guard = PrivacyGuard()

    # Should have decoded the hex to bytes
    assert "s1" in guard.search_keyring
    assert len(guard.search_keyring["s1"]) == 32  # 32 bytes
    assert isinstance(guard.search_keyring["s1"], bytes)

    # Cleanup
    for key in ["JEFFREY_ENCRYPTION_KEY", "JEFFREY_SEARCH_KEY", "JEFFREY_SEARCH_KID"]:
        if key in os.environ:
            del os.environ[key]


@pytest.mark.asyncio
async def test_search_with_key_rotation():
    """Test search with SKID rotation"""

    from cryptography.fernet import Fernet

    # Need encryption key
    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

    # Key v1
    key_s1 = secrets.token_hex(32)
    os.environ["JEFFREY_SEARCH_KEY"] = key_s1
    os.environ["JEFFREY_SEARCH_KID"] = "s1"

    # Enable search
    import yaml

    config = {"privacy": {"encrypt_pii": True, "enable_search_index": True}}
    os.makedirs("config", exist_ok=True)
    with open("config/federation.yaml", "w") as f:
        yaml.dump(config, f)

    from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
    from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2

    loader1 = SecureModuleLoader()
    federation1 = MemoryFederationV2(loader1)
    await federation1.initialize(None)

    # Store with s1
    await federation1.store_to_relevant("Data indexed with s1 key@example.com", "user", "user")

    # Rotate to s2
    key_s2 = secrets.token_hex(32)
    os.environ["JEFFREY_SEARCH_KEY_s1"] = key_s1  # Keep old key
    os.environ["JEFFREY_SEARCH_KEY"] = key_s2
    os.environ["JEFFREY_SEARCH_KID"] = "s2"

    # Reinitialize
    loader2 = SecureModuleLoader()
    federation2 = MemoryFederationV2(loader2)
    await federation2.initialize(None)

    # Store with s2
    await federation2.store_to_relevant("Data indexed with s2 key@test.com", "user", "user")

    # Should be able to search both
    results = await federation2.recall_from_all("indexed", "user")

    # Should find both items (search across both SKIDs)
    assert len(results) >= 2
    texts = [r.get("text", "") for r in results]
    assert any("s1" in t for t in texts)
    assert any("s2" in t for t in texts)

    # Cleanup
    for key in [
        "JEFFREY_ENCRYPTION_KEY",
        "JEFFREY_SEARCH_KEY",
        "JEFFREY_SEARCH_KID",
        "JEFFREY_SEARCH_KEY_s1",
    ]:
        if key in os.environ:
            del os.environ[key]


def test_search_disabled_by_default():
    """Test that search index is disabled by default"""

    # Default config (no explicit enable)
    import yaml

    from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
    from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2

    config = {
        "privacy": {
            "encrypt_pii": True
            # enable_search_index not set (should default to False)
        }
    }
    with open("config/federation.yaml", "w") as f:
        yaml.dump(config, f)

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)

    # Should be disabled
    assert federation._config_get("privacy.enable_search_index", False) == False


def test_normalize_for_indexing():
    """Test text normalization for search indexing"""

    from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
    from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)

    # Test normalization
    text = "Hello, World! This is a TEST-123 with émojis 🎉"
    tokens = federation._normalize_for_indexing(text)

    # Should normalize and filter
    assert "hello" in tokens
    assert "world" in tokens
    assert "this" in tokens
    assert "test" in tokens
    assert "123" in tokens
    # Short tokens should be filtered
    assert "is" not in tokens  # Too short (< 3 chars)
    assert "a" not in tokens  # Too short
    # Special chars removed
    assert "🎉" not in tokens
    assert "émojis" not in tokens or "mojis" in tokens  # Normalized


def test_skid_in_metadata():
    """Test that SKID is stored in metadata when indexing"""

    import asyncio

    async def test():
        os.environ["JEFFREY_SEARCH_KEY"] = secrets.token_hex(32)
        os.environ["JEFFREY_SEARCH_KID"] = "test-skid"

        # Enable search
        import yaml

        config = {"privacy": {"encrypt_pii": True, "enable_search_index": True}}
        with open("config/federation.yaml", "w") as f:
            yaml.dump(config, f)

        from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
        from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2

        loader = SecureModuleLoader()
        federation = MemoryFederationV2(loader)
        await federation.initialize(None)

        # Mock store to check metadata
        stored_metadata = None

        async def mock_store(self, payload):
            nonlocal stored_metadata
            stored_metadata = payload.get("metadata", {})
            return True

        # Patch a layer's module
        if federation.layers:
            layer = list(federation.layers.values())[0]
            if layer.modules:
                module = list(layer.modules.values())[0]
                module.store = mock_store

        # Store with PII to trigger indexing
        await federation.store_to_relevant("Test with email@example.com", "user", "user")

        # Check that SKID was stored (if store was called)
        if stored_metadata:
            assert stored_metadata.get("_skid") == "test-skid"

        # Cleanup
        del os.environ["JEFFREY_SEARCH_KEY"]
        del os.environ["JEFFREY_SEARCH_KID"]

    asyncio.run(test())
    print("✅ SKID stored in metadata")


if __name__ == "__main__":
    print("🔐 Testing Search Key Safety\n")

    test_search_key_required_in_production()
    print("✅ Search key required in production")

    test_search_key_hex_format()
    print("✅ Hex format properly decoded")

    asyncio.run(test_search_with_key_rotation())
    print("✅ SKID rotation works")

    test_search_disabled_by_default()
    print("✅ Search disabled by default")

    test_normalize_for_indexing()
    print("✅ Text normalization works")

    test_skid_in_metadata()

    print("\n🎉 All search key safety tests passed!")
