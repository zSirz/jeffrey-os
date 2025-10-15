#!/usr/bin/env python3
"""
Test PII redaction in monitoring logs
"""

import json
import re
import sys

sys.path.insert(0, "src")

import logging

from jeffrey.utils.monitoring import _redact_pii, get_monitor

logger = logging.getLogger(__name__)


def test_pii_redaction():
    """Test that PII is correctly redacted."""

    print("🧪 Testing PII redaction...")

    # Test cases with PII
    test_cases = [
        # Original cases
        ("Contact me at john.doe@example.com", "Contact me at [EMAIL]"),
        ("Visit https://example.com/sensitive", "Visit [URL]/sensitive"),
        ("Email: test@test.com and URL: www.test.com", "Email: [EMAIL] and URL: [URL]"),
        ("My email is david@jeffrey-ai.com", "My email is [EMAIL]"),
        # Phone numbers (extended)
        ("Call me at +33 6 12 34 56 78", "[PHONE]"),
        ("Phone: 0612345678", "[PHONE]"),
        ("My number is 06.12.34.56.78", "[PHONE]"),
        ("Contact: +1-555-123-4567", "[PHONE]"),
        # IP addresses
        ("Server IP: 192.168.1.1", "[IP]"),
        ("From 10.0.0.1 to db", "[IP]"),
        ("Host: 127.0.0.1", "[IP]"),
        # Mixed
        ("Email test@example.com, IP 192.168.1.1, Phone +33612345678", "[EMAIL], [IP], [PHONE]"),
    ]

    passed = 0
    for original, expected_pattern in test_cases:
        redacted = _redact_pii(original)

        # Check if appropriate redactions occurred
        email_ok = ("[EMAIL]" in redacted) if "@" in original else True
        url_ok = ("[URL]" in redacted) if ("http" in original or "www" in original) else True

        # Check for phone patterns (loose check for test purposes)
        phone_ok = True
        if any(c in original for c in ["+", "06", "01"]) and sum(c.isdigit() for c in original) >= 9:
            phone_ok = "[PHONE]" in redacted

        # Check for IP patterns
        ip_ok = ("[IP]" in redacted) if re.search(r'\d+\.\d+\.\d+\.\d+', original) else True

        is_correct = email_ok and url_ok and phone_ok and ip_ok

        status = "✅" if is_correct else "❌"
        passed += int(is_correct)

        print(f"{status} Original: '{original}'")
        print(f"   Redacted: '{redacted}'")

    accuracy = (passed / len(test_cases)) * 100
    print(f"\n🎯 PII Redaction: {passed}/{len(test_cases)} correct ({accuracy:.1f}%)")

    if accuracy >= 100.0:
        print("✅ PASS: PII redaction working correctly!")
        return True
    else:
        print("❌ FAIL: PII redaction has issues")
        return False


def test_monitoring_integration():
    """Test that monitoring correctly logs with PII protection."""

    print("\n🧪 Testing monitoring integration with PII...")

    monitor = get_monitor()

    # Log a prediction with PII
    monitor.log_prediction(
        text="Contact john@example.com or visit https://secret-site.com for more info",
        primary_emotion="neutral",
        confidence=0.75,
        all_scores={"neutral": 0.75, "joy": 0.15, "fear": 0.10},
        route="linear_head",
        low_confidence=False,
        rule_applied=None,
    )

    # Read the log and verify PII was redacted
    log_file = monitor._get_log_file()

    with open(log_file) as f:
        last_line = list(f)[-1]  # Get last logged entry
        entry = json.loads(last_line)

    text_preview = entry["text_preview"]

    # Check redaction
    has_email_redacted = "[EMAIL]" in text_preview
    has_url_redacted = "[URL]" in text_preview
    no_real_email = "@" not in text_preview
    no_real_url = "http" not in text_preview and "www" not in text_preview

    passed = has_email_redacted and has_url_redacted and no_real_email and no_real_url

    status = "✅" if passed else "❌"
    print(f"{status} Logged text preview: '{text_preview}'")
    print(f"   Email redacted: {has_email_redacted}")
    print(f"   URL redacted: {has_url_redacted}")
    print(f"   No real PII: {no_real_email and no_real_url}")

    if passed:
        print("✅ PASS: Monitoring PII protection working!")
        return True
    else:
        print("❌ FAIL: Monitoring PII protection has issues")
        return False


if __name__ == "__main__":
    test1 = test_pii_redaction()
    test2 = test_monitoring_integration()

    success = test1 and test2
    sys.exit(0 if success else 1)
