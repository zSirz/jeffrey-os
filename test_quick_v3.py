#!/usr/bin/env python3
"""Test rapide pour EmotionDetectorV3"""

import sys
from pathlib import Path

# Setup path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from jeffrey.nlp.emotion_detector_v3 import EmotionDetectorV3


def test_quick():
    detector = EmotionDetectorV3()

    # Tests simples
    tests = [
        ("Je suis content", "joy"),
        ("Je suis triste", "sadness"),
        ("Je galère", "frustration"),
        ("Ça fait du bien", "relief"),
        ("Je panique", "panic"),
    ]

    print("🧪 Test rapide EmotionDetectorV3")
    print("-" * 40)

    for text, expected in tests:
        result = detector.detect(text)
        print(f"Text: '{text}'")
        print(f"  → {result.primary} (intensity: {result.intensity:.2f})")
        print(f"  → Cues: {result.cues[:3]}")
        print()


if __name__ == "__main__":
    test_quick()
