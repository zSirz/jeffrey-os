#!/usr/bin/env python3
"""
Chaos Test: Validate complete fallback chain
Tests: linear_head → prototypes → regex

GPT Improvement: Guaranteed restoration with try/finally
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")

import logging

from jeffrey.core.emotion_backend import _METRICS, ProtoEmotionDetector

logger = logging.getLogger(__name__)


class ChaosTestResults:
    """Track chaos test metrics."""

    def __init__(self):
        self.phase1_success = False
        self.phase2_success = False
        self.phase3_success = False
        self.consistency_pct = 0.0
        self.linear_count = 0
        self.proto_count = 0
        self.regex_count = 0


def test_fallback_chain() -> ChaosTestResults:
    """Test the complete fallback chain under chaos conditions."""

    results = ChaosTestResults()

    logger.info("🧪 CHAOS TEST: Complete Fallback Chain Validation")
    logger.info("=" * 70)

    # Test cases (multilingual)
    test_cases = [
        "I'm so happy today!",
        "This is very frustrating",
        "I feel angry about this",
        "Je suis content",
        "C'est vraiment frustrant",
        "Quelle surprise !",
        "I'm disgusted",
        "This makes me sad",
    ]

    # ========== PHASE 1: Normal operation ==========
    logger.info("\n📊 PHASE 1: Normal Operation (linear head active)")
    logger.info("-" * 70)

    # Reset metrics
    _METRICS.clear()
    _METRICS.update(
        {
            "linear_head_success": 0,
            "linear_head_failures": 0,
            "proto_success": 0,
            "proto_failures": 0,
            "regex_fallback": 0,
            "total_predictions": 0,
            "fallback_triggered": 0,
            "proto_dimension_mismatch": 0,
            "encoder_mismatch_warnings": 0,
        }
    )

    detector = ProtoEmotionDetector()

    linear_predictions = []
    for i, text in enumerate(test_cases, 1):
        result = detector.predict_proba(text)
        linear_predictions.append(
            {
                "text": text,
                "emotion": result[0] if isinstance(result, tuple) else result.emotion,
                "confidence": max(result[0].values()) if isinstance(result, tuple) else result.confidence,
            }
        )
        emotion = linear_predictions[-1]["emotion"]
        confidence = linear_predictions[-1]["confidence"]
        logger.info(f"   {i}. '{text[:40]}...' → {emotion} ({confidence:.3f})")

    linear_success = _METRICS.get("linear_head_success", 0)
    logger.info("\n✅ Phase 1 Metrics:")
    logger.info(f"   Linear head successes: {linear_success}/{len(test_cases)}")
    logger.info(f"   Proto fallback: {_METRICS.get('proto_success', 0)}")
    logger.info(f"   Regex fallback: {_METRICS.get('regex_fallback', 0)}")

    results.phase1_success = linear_success >= len(test_cases) * 0.9  # Allow 90% success
    results.linear_count = linear_success

    if not results.phase1_success:
        logger.error("❌ Phase 1 FAILED: Linear head should handle ≥90% of requests")

    # ========== PHASE 2: Simulate linear head failure ==========
    logger.info("\n📊 PHASE 2: Chaos Injection (linear head removed)")
    logger.info("-" * 70)

    linear_head_path = Path("data/linear_head.joblib")
    backup_path = Path("data/linear_head.joblib.chaos_backup")

    # GPT Improvement: Guaranteed restoration with try/finally
    try:
        # Backup and remove linear head
        if linear_head_path.exists():
            logger.info(f"   💾 Backing up: {linear_head_path}")
            shutil.copy(linear_head_path, backup_path)
            linear_head_path.unlink()
            logger.info(f"   ❌ Removed: {linear_head_path} (simulating failure)")
        else:
            logger.warning(f"   ⚠️ Linear head not found: {linear_head_path}")

        # Reset metrics
        _METRICS.clear()
        _METRICS.update(
            {
                "linear_head_success": 0,
                "linear_head_failures": 0,
                "proto_success": 0,
                "proto_failures": 0,
                "regex_fallback": 0,
                "total_predictions": 0,
                "fallback_triggered": 0,
                "proto_dimension_mismatch": 0,
                "encoder_mismatch_warnings": 0,
            }
        )

        # Create new detector (will use prototypes)
        detector_chaos = ProtoEmotionDetector()

        proto_predictions = []
        for i, text in enumerate(test_cases, 1):
            result = detector_chaos.predict_proba(text)
            proto_predictions.append(
                {
                    "text": text,
                    "emotion": result[0] if isinstance(result, tuple) else result.emotion,
                    "confidence": max(result[0].values()) if isinstance(result, tuple) else result.confidence,
                }
            )
            emotion = proto_predictions[-1]["emotion"]
            confidence = proto_predictions[-1]["confidence"]
            logger.info(f"   {i}. '{text[:40]}...' → {emotion} ({confidence:.3f})")

        proto_success = _METRICS.get("proto_success", 0)
        regex_used = _METRICS.get("regex_fallback", 0)

        logger.info("\n✅ Phase 2 Metrics:")
        logger.info(f"   Linear head attempts: {_METRICS.get('linear_head_failures', 0)}")
        logger.info(f"   Prototypes successes: {proto_success}/{len(test_cases)}")
        logger.info(f"   Regex fallback: {regex_used}")

        results.phase2_success = (proto_success > 0) and (proto_success >= len(test_cases) * 0.8)
        results.proto_count = proto_success
        results.regex_count = regex_used

        if not results.phase2_success:
            logger.error("❌ Phase 2 FAILED: Prototypes should handle ≥80% of requests")

        # ========== PHASE 4: Consistency Analysis ==========
        logger.info("\n📊 PHASE 4: Consistency Analysis")
        logger.info("-" * 70)

        consistency_score = 0
        for i, (lin, proto) in enumerate(zip(linear_predictions, proto_predictions), 1):
            match = lin["emotion"] == proto["emotion"]
            consistency_score += int(match)

            status = "✅" if match else "⚠️"
            logger.info(f"{status} Case {i}: Linear={lin['emotion']}, Proto={proto['emotion']}")

        results.consistency_pct = (consistency_score / len(test_cases)) * 100

    finally:
        # GPT Improvement: GUARANTEED restoration even if exception occurs
        logger.info("\n📊 PHASE 3: Restoration (GUARANTEED)")
        logger.info("-" * 70)

        if backup_path.exists():
            logger.info(f"   ♻️ Restoring: {backup_path} → {linear_head_path}")
            shutil.copy(backup_path, linear_head_path)
            backup_path.unlink()
            logger.info(f"   ✅ Restored: {linear_head_path}")
            results.phase3_success = True
        else:
            logger.error("   ❌ Backup not found, cannot restore")
            results.phase3_success = False

    # ========== FINAL VERDICT ==========
    logger.info("\n🎯 CHAOS TEST FINAL RESULTS")
    logger.info("=" * 70)
    logger.info(f"Phase 1 (Linear Head):  {'✅ PASS' if results.phase1_success else '❌ FAIL'}")
    logger.info(f"Phase 2 (Prototypes):   {'✅ PASS' if results.phase2_success else '❌ FAIL'}")
    logger.info(f"Phase 3 (Restoration):  {'✅ PASS' if results.phase3_success else '❌ FAIL'}")
    logger.info("\nMetrics:")
    logger.info(f"  Linear successes:    {results.linear_count}/{len(test_cases)}")
    logger.info(f"  Proto successes:     {results.proto_count}/{len(test_cases)}")
    logger.info(f"  Regex fallback:      {results.regex_count} (target: ≤2)")
    logger.info(f"  Consistency:         {results.consistency_pct:.1f}%")

    all_passed = (
        results.phase1_success
        and results.phase2_success
        and results.phase3_success
        and results.regex_count <= 2  # Allow minimal regex usage
    )

    if all_passed:
        logger.info("\n✅ CHAOS TEST PASSED - Fallback chain is ROBUST!")
        logger.info("   ✓ Linear head works when available")
        logger.info("   ✓ Prototypes activate when linear head fails")
        logger.info("   ✓ Minimal regex fallback (ML resilience proven)")
        logger.info("   ✓ System restored successfully")
    else:
        logger.error("\n❌ CHAOS TEST FAILED - Issues detected")
        if not results.phase1_success:
            logger.error("   ✗ Linear head not working properly")
        if not results.phase2_success:
            logger.error("   ✗ Prototypes fallback insufficient")
        if results.regex_count > 2:
            logger.error(f"   ✗ Excessive regex fallback: {results.regex_count} times")
        if not results.phase3_success:
            logger.error("   ✗ System restoration failed")

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    try:
        results = test_fallback_chain()

        # Exit code based on results
        success = (
            results.phase1_success and results.phase2_success and results.phase3_success and results.regex_count <= 2
        )
        sys.exit(0 if success else 1)

    except Exception as e:
        logger.error(f"❌ Chaos test crashed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
