#!/usr/bin/env python3
"""
Simple test for encryption/decryption functionality
"""

import asyncio
import os

from cryptography.fernet import Fernet

# Set encryption key
os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()

from jeffrey.core.utils.privacy import PrivacyGuard


def test_privacy_guard():
    """Test PrivacyGuard basic functionality"""

    guard = PrivacyGuard()

    # Test PII detection
    text_with_pii = "My email is test@example.com"
    assert guard.detect_pii(text_with_pii) == True
    print("✅ PII detection works")

    # Test masking
    masked = guard.mask_for_logging(text_with_pii)
    assert "[EMAIL]" in masked
    assert "test@example.com" not in masked
    print("✅ PII masking works")

    # Test encryption/decryption
    encrypted = guard.cipher.encrypt(text_with_pii.encode())
    decrypted = guard.cipher.decrypt(encrypted).decode()
    assert decrypted == text_with_pii
    print("✅ Encryption/decryption works")

    # Test key rotation
    os.environ["JEFFREY_KEY_v1"] = os.environ["JEFFREY_ENCRYPTION_KEY"]
    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
    os.environ["JEFFREY_KID"] = "v2"

    guard2 = PrivacyGuard()

    # Should be able to decrypt with old key
    decrypted2 = guard2.decrypt_with_kid(encrypted, "v1").decode()
    assert decrypted2 == text_with_pii
    print("✅ Key rotation works")

    return True


def test_hmac_search():
    """Test HMAC-based search indexing"""
    import hashlib
    import hmac

    search_key = b"test_search_key"

    def token_tag(token: str) -> str:
        return hmac.new(search_key, token.encode(), hashlib.sha256).hexdigest()[:16]

    # Generate tags for tokens
    text = "john email test"
    tokens = text.split()
    tags = [token_tag(t) for t in tokens]

    # Search for "email"
    query_tag = token_tag("email")
    assert query_tag in tags
    print("✅ HMAC search indexing works")

    # Different key produces different tags
    search_key2 = b"different_key"

    def token_tag2(token: str) -> str:
        return hmac.new(search_key2, token.encode(), hashlib.sha256).hexdigest()[:16]

    query_tag2 = token_tag2("email")
    assert query_tag != query_tag2  # Different keys = different tags
    print("✅ HMAC provides security via key")

    return True


async def test_memory_federation_simple():
    """Test memory federation with minimal setup"""
    from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
    from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2

    # Enable features
    os.environ["JEFFREY_SEARCH_KEY"] = Fernet.generate_key().decode()

    # Create minimal config
    import yaml

    config = {
        "privacy": {"encrypt_pii": True, "enable_search_index": True},
        "memory_federation": {"enabled": True, "cache_size": 100},
    }

    os.makedirs("config", exist_ok=True)
    with open("config/federation.yaml", "w") as f:
        yaml.dump(config, f)

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)

    # Test without full initialization
    assert federation.privacy_guard is not None
    print("✅ Federation created with privacy guard")

    # Test search key setup
    assert federation.search_key is not None
    print("✅ Search key configured")

    # Test token tag generation
    tag = federation._token_tag("test")
    assert len(tag) == 16
    print("✅ Token tag generation works")

    # Test decrypt method exists
    metadata = {"_original_text_encrypted": False}
    text = federation._maybe_decrypt_text("plain text", metadata)
    assert text == "plain text"
    print("✅ Decrypt method works for plain text")

    return True


if __name__ == "__main__":
    print("🔐 Testing Decryption & Search Features\n")

    # Test PrivacyGuard
    test_privacy_guard()
    print()

    # Test HMAC search
    test_hmac_search()
    print()

    # Test federation setup
    asyncio.run(test_memory_federation_simple())
    print()

    print("🎉 All simple tests passed!")
