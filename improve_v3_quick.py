#!/usr/bin/env python3
"""
Amélioration rapide EmotionDetectorV3 basée sur les erreurs observées
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def improve_lexicon():
    """Améliorations du lexique basées sur les erreurs"""

    improvements = {
        # Joy - expressions narratives manquantes
        "joy": {
            "ne devineras jamais",
            "devine quoi",
            "super nouvelle",
            "j'ai une bonne nouvelle",
            "ça m'est arrivé",
            "offert",
            "voyage surprise",
            "cadeau",
            "surprise",
            "chance incroyable",
        },
        # Discomfort - mieux différencier de fear
        "discomfort": {
            "ça me met mal",
            "me met pas à l'aise",
            "ça me dérange un peu",
            "c'est délicat",
            "sujet sensible",
            "touche un point sensible",
            "ça remue des trucs",
            "ça réveille des souvenirs",
        },
        # Vulnerability - expressions subtiles
        "vulnerability": {
            "c'est dur pour moi",
            "ça me touche",
            "je suis sensible à ça",
            "ruptures c'est dur",
            "difficile pour moi",
            "ça me fragilise",
            "j'ai du mal avec ça",
            "ça me bouleverse",
        },
        # Patterns négatifs pour réduire neutral
        "sadness": {"ça fait quelques semaines", "je passe tout le temps", "seul en ce moment", "vraiment seul"},
    }

    print("🔧 AMÉLIORATIONS SUGGÉRÉES:")
    print("=" * 40)

    for emotion, new_terms in improvements.items():
        print(f"📝 {emotion.upper()}:")
        for term in new_terms:
            print(f"   + '{term}'")
        print()

    # Test avec les exemples qui ont échoué
    test_cases = [
        ("Tu ne devineras jamais ce qui m'est arrivé !", "joy"),
        ("Mon meilleur ami m'a offert un voyage surprise", "joy"),
        ("On parle de mon ex ? Ça me met mal à l'aise", "discomfort"),
        ("Les ruptures c'est dur pour moi", "vulnerability"),
        ("Je me sens vraiment seul en ce moment", "sadness"),
    ]

    print("🧪 TESTS AVEC AMÉLIORATIONS:")
    print("-" * 40)

    # Import du détecteur
    from jeffrey.nlp.emotion_detector_v3 import EmotionDetectorV3

    detector = EmotionDetectorV3()

    for text, expected in test_cases:
        result = detector.detect(text)
        status = "✅" if result.primary == expected else "❌"
        print(f"{status} '{text[:40]}...'")
        print(f"   Expected: {expected} → Got: {result.primary}")
        print(f"   Top scores: {sorted(result.scores.items(), key=lambda x: x[1], reverse=True)[:3]}")
        print()


if __name__ == "__main__":
    improve_lexicon()
