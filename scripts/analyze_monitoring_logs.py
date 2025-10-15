#!/usr/bin/env python3
"""
Analyze monitoring logs for production readiness
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)


def analyze_logs():
    """Analyze monitoring logs for key metrics."""

    print("📊 Analyzing monitoring logs...")

    logs_dir = Path("logs/predictions")
    log_files = sorted(logs_dir.glob("predictions_*.jsonl"))

    if not log_files:
        print("⚠️ No log files found")
        return

    # Use latest log file
    latest_log = log_files[-1]
    print(f"📁 Reading: {latest_log}")

    # Collect metrics
    total_predictions = 0
    route_counts = Counter()
    emotion_counts = Counter()
    low_confidence_count = 0
    rule_applied_counts = Counter()
    confidence_values = []

    with open(latest_log) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)
                total_predictions += 1

                # Route distribution
                route = entry.get("route", "unknown")
                route_counts[route] += 1

                # Emotion distribution
                emotion = entry["prediction"]["primary"]
                emotion_counts[emotion] += 1

                # Confidence stats
                confidence = entry["prediction"]["confidence"]
                confidence_values.append(confidence)

                if entry["prediction"]["low_confidence"]:
                    low_confidence_count += 1

                # Rule application tracking
                rule = entry["prediction"].get("rule_applied")
                if rule:
                    rule_applied_counts[rule] += 1

            except Exception as e:
                print(f"Error parsing line: {e}")

    # Calculate statistics
    print("\n🎯 MONITORING STATISTICS")
    print("=" * 60)
    print(f"Total predictions: {total_predictions}")

    print("\n📊 Route Distribution:")
    for route, count in route_counts.most_common():
        pct = (count / total_predictions) * 100
        print(f"   {route}: {count} ({pct:.1f}%)")

    print("\n🎭 Emotion Distribution:")
    for emotion, count in emotion_counts.most_common(10):
        pct = (count / total_predictions) * 100
        print(f"   {emotion}: {count} ({pct:.1f}%)")

    if confidence_values:
        import numpy as np

        print("\n💯 Confidence Statistics:")
        print(f"   Mean: {np.mean(confidence_values):.3f}")
        print(f"   Median: {np.median(confidence_values):.3f}")
        print(f"   P10: {np.percentile(confidence_values, 10):.3f}")
        print(f"   P90: {np.percentile(confidence_values, 90):.3f}")

        low_conf_pct = (low_confidence_count / total_predictions) * 100
        print(f"   Low confidence (<0.4): {low_confidence_count} ({low_conf_pct:.1f}%)")

    if rule_applied_counts:
        print("\n🎯 Rule Applications:")
        for rule, count in rule_applied_counts.items():
            pct = (count / total_predictions) * 100
            print(f"   {rule}: {count} ({pct:.1f}%)")
    else:
        print("\n🎯 Rule Applications: None")

    # Production readiness checks
    print("\n✅ PRODUCTION READINESS CHECKS")
    print("=" * 60)

    checks_passed = []
    checks_failed = []

    # Check 1: Primary route is linear_head
    linear_pct = (route_counts.get("linear_head", 0) / total_predictions) * 100
    if linear_pct >= 90.0:
        checks_passed.append(f"✅ Linear head usage: {linear_pct:.1f}% (target ≥90%)")
    else:
        checks_failed.append(f"❌ Linear head usage: {linear_pct:.1f}% (target ≥90%)")

    # Check 2: Regex fallback minimal
    regex_pct = (route_counts.get("regex", 0) / total_predictions) * 100
    if regex_pct <= 1.0:
        checks_passed.append(f"✅ Regex fallback: {regex_pct:.1f}% (target ≤1%)")
    else:
        checks_failed.append(f"❌ Regex fallback: {regex_pct:.1f}% (target ≤1%)")

    # Check 3: Low confidence rate acceptable
    low_conf_pct = (low_confidence_count / total_predictions) * 100
    if low_conf_pct <= 15.0:
        checks_passed.append(f"✅ Low confidence rate: {low_conf_pct:.1f}% (target ≤15%)")
    else:
        checks_failed.append(f"⚠️ Low confidence rate: {low_conf_pct:.1f}% (target ≤15%)")

    # Check 4: PII protection active
    # Read a sample and verify redaction
    with open(latest_log) as f:
        sample_lines = [json.loads(line) for line in list(f)[:10] if line.strip()]

    has_pii_protection = all(
        "@" not in entry["text_preview"] and "http" not in entry["text_preview"] for entry in sample_lines
    )

    if has_pii_protection:
        checks_passed.append("✅ PII protection active")
    else:
        checks_failed.append("❌ PII protection not working")

    # Display results
    for check in checks_passed:
        print(check)
    for check in checks_failed:
        print(check)

    # Final verdict
    if len(checks_failed) == 0:
        print("\n🎉 ALL CHECKS PASSED - PRODUCTION READY!")
        return True
    elif len(checks_failed) <= 1:
        print("\n⚠️ MOSTLY READY - Minor issues to address")
        return True
    else:
        print("\n❌ NOT READY - Multiple issues detected")
        return False


if __name__ == "__main__":
    success = analyze_logs()
    sys.exit(0 if success else 1)
