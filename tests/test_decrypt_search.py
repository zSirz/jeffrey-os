"""
Tests for decryption and encrypted text search
"""

import asyncio
import os
import sys
import time

import pytest
from cryptography.fernet import Fernet

# Add parent directory to path
sys.path.insert(0, ".")

from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2


@pytest.mark.asyncio
async def test_encrypt_decrypt_cycle():
    """Test complete cycle: encryption → storage → recall → decryption"""

    # Setup with encryption enabled
    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
    os.environ["JEFFREY_MODE"] = "test"

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Store text with PII
    original_text = "Contact me at john@example.com or 555-1234"
    await federation.store_to_relevant(text=original_text, user_id="test_user", role="user")

    # Recall text
    results = await federation.recall_from_all(query="", user_id="test_user")

    # Verify text is properly decrypted
    assert len(results) > 0
    recalled_text = results[0].get("text", "")

    # Recalled text should match original
    assert recalled_text == original_text

    # Metadata should not contain technical flags
    metadata = results[0].get("metadata", {})
    assert "_original_text_encrypted" not in metadata
    assert "contains_pii" in metadata  # User-friendly flag

    # Cleanup
    del os.environ["JEFFREY_ENCRYPTION_KEY"]
    del os.environ["JEFFREY_MODE"]


@pytest.mark.asyncio
async def test_search_encrypted_text():
    """Test search on encrypted texts via HMAC index"""

    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
    os.environ["JEFFREY_SEARCH_KEY"] = Fernet.generate_key().decode()

    # Enable search index
    import yaml

    config = {
        "privacy": {"encrypt_pii": True, "enable_search_index": True, "enable_ngram_search": False},
        "memory_federation": {"enabled": True, "budget_ms": 400},
    }

    # Write temp config
    os.makedirs("config", exist_ok=True)
    with open("config/federation.yaml", "w") as f:
        yaml.dump(config, f)

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Store multiple texts with PII (will be encrypted)
    texts = [
        "My email is alice@example.com",
        "Call me at 555-9876",
        "Contact bob@example.com for details",
    ]

    for text in texts:
        await federation.store_to_relevant(text, "user", "user")

    # Search "email" (should find items with email)
    results = await federation.recall_from_all(query="email", user_id="user")

    # Should find at least 2 results (alice and bob)
    assert len(results) >= 2

    # Verify texts are decrypted
    for result in results:
        text = result.get("text", "")
        assert "@example.com" in text  # Email decrypted and visible

    # Cleanup
    del os.environ["JEFFREY_ENCRYPTION_KEY"]
    del os.environ["JEFFREY_SEARCH_KEY"]


@pytest.mark.asyncio
async def test_key_rotation_support():
    """Test that KID is stored properly for future rotation"""

    # Key v1
    key1 = Fernet.generate_key().decode()
    os.environ["JEFFREY_ENCRYPTION_KEY"] = key1
    os.environ["JEFFREY_KID"] = "v1"

    loader1 = SecureModuleLoader()
    federation1 = MemoryFederationV2(loader1)
    await federation1.initialize(None)

    # Store with KID v1
    await federation1.store_to_relevant("Secret with john@example.com", "user", "user")

    # Simulate rotation: new key v2
    key2 = Fernet.generate_key().decode()
    os.environ["JEFFREY_ENCRYPTION_KEY"] = key2
    os.environ["JEFFREY_KID"] = "v2"
    # Keep old key for decryption
    os.environ["JEFFREY_KEY_v1"] = key1

    # Reinitialize with new config
    loader2 = SecureModuleLoader()
    federation2 = MemoryFederationV2(loader2)
    await federation2.initialize(None)

    # Store with KID v2
    await federation2.store_to_relevant("New secret with jane@example.com", "user", "user")

    # Recall all (v1 and v2)
    results = await federation2.recall_from_all("", "user")

    # Should decrypt both (with different keys)
    assert len(results) == 2
    texts = [r["text"] for r in results]
    assert any("john" in t for t in texts)  # v1 decrypted
    assert any("jane" in t for t in texts)  # v2 decrypted

    # Cleanup
    del os.environ["JEFFREY_ENCRYPTION_KEY"]
    del os.environ["JEFFREY_KID"]
    del os.environ["JEFFREY_KEY_v1"]


@pytest.mark.asyncio
async def test_search_performance_with_encryption():
    """Test that search remains performant with encryption"""

    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
    os.environ["JEFFREY_SEARCH_KEY"] = Fernet.generate_key().decode()

    # Enable search index
    import yaml

    config = {
        "privacy": {"encrypt_pii": True, "enable_search_index": True},
        "memory_federation": {"enabled": True, "cache_size": 10000},
    }

    with open("config/federation.yaml", "w") as f:
        yaml.dump(config, f)

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Store 100 encrypted texts
    for i in range(100):
        await federation.store_to_relevant(f"User {i} email is user{i}@test.com", "user", "user")

    # Measure search time
    start = time.perf_counter()
    results = await federation.recall_from_all(query="user50", user_id="user")
    search_time = (time.perf_counter() - start) * 1000

    # Should find the right user
    assert len(results) > 0
    assert "user50" in results[0]["text"]

    # Search must stay fast (< 100ms even with 100 encrypted items)
    assert search_time < 100, f"Search took {search_time}ms"

    # Cleanup
    del os.environ["JEFFREY_ENCRYPTION_KEY"]
    del os.environ["JEFFREY_SEARCH_KEY"]


@pytest.mark.asyncio
async def test_privacy_without_search_index():
    """Test that encryption works even without search index"""

    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

    # Disable search index
    import yaml

    config = {"privacy": {"encrypt_pii": True, "enable_search_index": False}}  # Disabled

    with open("config/federation.yaml", "w") as f:
        yaml.dump(config, f)

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Store encrypted text
    await federation.store_to_relevant("Private: john@example.com", "user", "user")

    # Recall without search
    results = await federation.recall_from_all("", "user")

    # Should decrypt properly
    assert len(results) > 0
    assert "john@example.com" in results[0]["text"]

    # Search with query should return nothing (index disabled)
    search_results = await federation.recall_from_all("john", "user")
    assert len(search_results) == 0  # No results when searching encrypted without index

    # Cleanup
    del os.environ["JEFFREY_ENCRYPTION_KEY"]


@pytest.mark.asyncio
async def test_metadata_not_mutated():
    """Test that original metadata is not mutated during recall"""

    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Store with PII
    await federation.store_to_relevant("Contact: test@example.com", "user", "user")

    # First recall
    results1 = await federation.recall_from_all("", "user")
    metadata1 = results1[0].get("metadata", {})

    # Second recall
    results2 = await federation.recall_from_all("", "user")
    metadata2 = results2[0].get("metadata", {})

    # Metadata should be consistent between recalls
    assert metadata1 == metadata2
    assert "contains_pii" in metadata1
    assert "_original_text_encrypted" not in metadata1

    # Cleanup
    del os.environ["JEFFREY_ENCRYPTION_KEY"]


if __name__ == "__main__":
    # Run specific test
    asyncio.run(test_encrypt_decrypt_cycle())
    print("✅ Encrypt/decrypt cycle test passed")

    asyncio.run(test_search_encrypted_text())
    print("✅ Encrypted search test passed")

    asyncio.run(test_key_rotation_support())
    print("✅ Key rotation test passed")

    asyncio.run(test_search_performance_with_encryption())
    print("✅ Performance test passed")

    asyncio.run(test_privacy_without_search_index())
    print("✅ Privacy without index test passed")

    asyncio.run(test_metadata_not_mutated())
    print("✅ Metadata immutability test passed")

    print("\n🎉 All decryption and search tests passed!")
