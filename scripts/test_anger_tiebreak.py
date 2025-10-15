#!/usr/bin/env python3
"""
Test anger↔frustration tie-break rule
"""

import sys

sys.path.insert(0, "src")

import logging

from jeffrey.core.emotion_backend import ProtoEmotionDetector

logger = logging.getLogger(__name__)


def test_anger_tiebreak():
    """Test tie-break rule with explicit anger expressions."""

    print("🧪 Testing anger↔frustration tie-break...")

    detector = ProtoEmotionDetector()

    # Test cases where anger should win
    anger_cases = [
        "I'm so angry about this!",
        "I'm furious!",
        "This makes me rage!",
        "Je suis en rage!",
        "Je suis furax!",
        "C'est inadmissible, je suis hors de moi!",
    ]

    # Test cases where frustration is legitimate
    frustration_cases = [
        "This is so frustrating...",
        "I'm frustrated with the situation",
        "C'est frustrant",
    ]

    # Test anger cases
    print("\n📊 Testing ANGER cases:")
    anger_correct = 0
    for text in anger_cases:
        result_tuple = detector.predict_proba(text)
        probs_dict = result_tuple[0]  # Extract probabilities dict
        predicted_emotion = max(probs_dict, key=probs_dict.get)
        confidence = probs_dict[predicted_emotion]

        is_correct = predicted_emotion == "anger"
        anger_correct += int(is_correct)

        status = "✅" if is_correct else "❌"
        print(f"{status} '{text[:40]}...' → {predicted_emotion} (conf: {confidence:.3f})")

    # Test frustration cases
    print("\n📊 Testing FRUSTRATION cases:")
    frustration_correct = 0
    for text in frustration_cases:
        result_tuple = detector.predict_proba(text)
        probs_dict = result_tuple[0]  # Extract probabilities dict
        predicted_emotion = max(probs_dict, key=probs_dict.get)
        confidence = probs_dict[predicted_emotion]

        is_correct = predicted_emotion == "frustration"
        frustration_correct += int(is_correct)

        status = "✅" if is_correct else "❌"
        print(f"{status} '{text[:40]}...' → {predicted_emotion} (conf: {confidence:.3f})")

    # Summary
    total = len(anger_cases) + len(frustration_cases)
    correct = anger_correct + frustration_correct
    accuracy = (correct / total) * 100

    print("\n🎯 Tie-break Rule Performance:")
    print(f"   Anger cases: {anger_correct}/{len(anger_cases)} correct")
    print(f"   Frustration cases: {frustration_correct}/{len(frustration_cases)} correct")
    print(f"   Overall accuracy: {accuracy:.1f}%")

    # Pass/fail
    if accuracy >= 80.0:
        print("✅ PASS: Tie-break rule working correctly!")
        return True
    else:
        print("❌ FAIL: Tie-break rule needs adjustment")
        return False


if __name__ == "__main__":
    success = test_anger_tiebreak()
    sys.exit(0 if success else 1)
