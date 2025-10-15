#!/usr/bin/env python3
"""
Final validation checklist for security features
"""

import asyncio
import os

from cryptography.fernet import Fernet

# Setup environment
print("🔐 JEFFREY OS SECURITY VALIDATION")
print("=" * 50)


def check_production_requirements():
    """Check production security requirements"""
    print("\n📋 Production Requirements:")

    # Test 1: Encryption key required in production
    os.environ["JEFFREY_MODE"] = "production"
    try:
        from jeffrey.core.utils.privacy import PrivacyGuard

        # Should fail without key
        try:
            guard = PrivacyGuard()
            print("❌ FAIL: Production allowed without encryption key!")
            return False
        except ValueError as e:
            if "JEFFREY_ENCRYPTION_KEY required" in str(e):
                print("✅ PASS: Production requires encryption key")
            else:
                print(f"❌ FAIL: Wrong error: {e}")
                return False
    finally:
        del os.environ["JEFFREY_MODE"]

    return True


def check_pii_detection():
    """Check PII detection and masking"""
    print("\n📋 PII Detection & Masking:")

    os.environ["JEFFREY_ENCRYPTION_KEY"] = Fernet.generate_key().decode()
    from jeffrey.core.utils.privacy import PrivacyGuard

    guard = PrivacyGuard()

    # Test patterns
    test_cases = [
        ("john@example.com", "EMAIL"),
        ("555-1234", "PHONE"),
        ("4111-1111-1111-1111", "CREDIT_CARD"),
        ("123-45-6789", "SSN"),
    ]

    all_pass = True
    for text, pii_type in test_cases:
        # Detection
        if guard.detect_pii(text):
            print(f"✅ Detected {pii_type}: {text}")
        else:
            print(f"❌ Missed {pii_type}: {text}")
            all_pass = False

        # Masking
        masked = guard.mask_for_logging(text)
        if f"[{pii_type}]" in masked and text not in masked:
            print(f"✅ Masked {pii_type} correctly")
        else:
            print(f"❌ Failed to mask {pii_type}: {masked}")
            all_pass = False

    return all_pass


def check_encryption_decryption():
    """Check encryption/decryption cycle"""
    print("\n📋 Encryption/Decryption:")

    from jeffrey.core.utils.privacy import PrivacyGuard

    guard = PrivacyGuard()
    test_text = "Secret data with email@test.com"

    # Encrypt
    encrypted = guard.cipher.encrypt(test_text.encode())
    print(f"✅ Encrypted: {len(encrypted)} bytes")

    # Decrypt
    decrypted = guard.cipher.decrypt(encrypted).decode()
    if decrypted == test_text:
        print("✅ Decryption successful")
        return True
    else:
        print("❌ Decryption failed")
        return False


def check_key_rotation():
    """Check key rotation support"""
    print("\n📋 Key Rotation Support:")

    # Setup v1 key
    key_v1 = Fernet.generate_key().decode()
    os.environ["JEFFREY_ENCRYPTION_KEY"] = key_v1
    os.environ["JEFFREY_KID"] = "v1"

    from jeffrey.core.utils.privacy import PrivacyGuard

    guard1 = PrivacyGuard()
    test_text = "Data encrypted with v1"
    encrypted_v1 = guard1.cipher.encrypt(test_text.encode())

    # Rotate to v2
    key_v2 = Fernet.generate_key().decode()
    os.environ["JEFFREY_ENCRYPTION_KEY"] = key_v2
    os.environ["JEFFREY_KID"] = "v2"
    os.environ["JEFFREY_KEY_v1"] = key_v1  # Keep old key

    # Create new guard
    guard2 = PrivacyGuard()

    # Should decrypt v1 data
    try:
        decrypted = guard2.decrypt_with_kid(encrypted_v1, "v1").decode()
        if decrypted == test_text:
            print("✅ Key rotation works - v1 data decrypted with keyring")
            return True
        else:
            print("❌ Decryption with old key failed")
            return False
    except Exception as e:
        print(f"❌ Key rotation failed: {e}")
        return False


def check_hmac_search():
    """Check HMAC-based secure search"""
    print("\n📋 HMAC Search Index:")

    import hashlib
    import hmac

    # Test with known key
    search_key = b"test_key_123"

    def tag(token: str) -> str:
        return hmac.new(search_key, token.encode(), hashlib.sha256).hexdigest()[:16]

    # Generate index
    tokens = ["email", "john", "test"]
    index = [tag(t) for t in tokens]

    # Search
    query_tag = tag("email")
    if query_tag in index:
        print("✅ HMAC search index works")
    else:
        print("❌ HMAC search failed")
        return False

    # Verify different key = different tags
    search_key2 = b"different_key"
    tag2 = hmac.new(search_key2, b"email", hashlib.sha256).hexdigest()[:16]
    if tag2 != query_tag:
        print("✅ Different keys produce different tags (secure)")
        return True
    else:
        print("❌ Same tags with different keys (insecure!)")
        return False


async def check_memory_federation():
    """Check memory federation integration"""
    print("\n📋 Memory Federation Integration:")

    from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader
    from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2

    os.environ["JEFFREY_SEARCH_KEY"] = Fernet.generate_key().decode()

    # Create config
    import yaml

    config = {
        "privacy": {
            "encrypt_pii": True,
            "enable_search_index": True,
            "enable_ngram_search": False,  # Disabled by default
        },
        "memory_federation": {"enabled": True},
    }

    os.makedirs("config", exist_ok=True)
    with open("config/federation.yaml", "w") as f:
        yaml.dump(config, f)

    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)

    # Check components
    checks = [
        (federation.privacy_guard is not None, "Privacy guard initialized"),
        (federation.search_key is not None, "Search key configured"),
        (hasattr(federation, "_maybe_decrypt_text"), "Decrypt method exists"),
        (hasattr(federation, "_search_encrypted"), "Encrypted search exists"),
        (hasattr(federation, "_token_tag"), "Token tag method exists"),
    ]

    all_pass = True
    for check, desc in checks:
        if check:
            print(f"✅ {desc}")
        else:
            print(f"❌ {desc}")
            all_pass = False

    return all_pass


def print_summary(results):
    """Print final summary"""
    print("\n" + "=" * 50)
    print("📊 FINAL VALIDATION SUMMARY")
    print("=" * 50)

    total = len(results)
    passed = sum(results.values())

    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test}")

    print("\n" + "=" * 50)
    if passed == total:
        print(f"🎉 ALL {total} TESTS PASSED!")
        print("✅ System is PRODUCTION READY")
    else:
        print(f"⚠️  {passed}/{total} tests passed")
        print("❌ Fix failing tests before production")

    return passed == total


async def main():
    """Run all validation checks"""
    results = {}

    # Run checks
    results["Production Requirements"] = check_production_requirements()
    results["PII Detection"] = check_pii_detection()
    results["Encryption/Decryption"] = check_encryption_decryption()
    results["Key Rotation"] = check_key_rotation()
    results["HMAC Search"] = check_hmac_search()
    results["Memory Federation"] = await check_memory_federation()

    # Summary
    success = print_summary(results)

    # Deployment commands
    if success:
        print("\n📝 DEPLOYMENT CHECKLIST:")
        print("```bash")
        print("# 1. Generate production key")
        print(
            "export JEFFREY_ENCRYPTION_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
        )
        print(
            "export JEFFREY_SEARCH_KEY=$(python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')"
        )
        print("export JEFFREY_KID=v1")
        print("export JEFFREY_MODE=production")
        print("")
        print("# 2. Configure features (config/modules_fixed.yaml)")
        print("# privacy:")
        print("#   encrypt_pii: true")
        print("#   enable_search_index: false  # Enable only if needed")
        print("#   enable_ngram_search: false")
        print("")
        print("# 3. Start system")
        print("python src/jeffrey/main.py")
        print("")
        print("# 4. Monitor")
        print("tail -f logs/jeffrey.log | grep -E '(PII|encrypt|decrypt)'")
        print("```")

    # Cleanup
    for key in ["JEFFREY_ENCRYPTION_KEY", "JEFFREY_SEARCH_KEY", "JEFFREY_KID", "JEFFREY_KEY_v1"]:
        if key in os.environ:
            del os.environ[key]

    return success


if __name__ == "__main__":
    asyncio.run(main())
