"""
Smoke Test FR/EN pour Jeffrey OS v2.4.2 - ÉTAPE 1.3
Test de validation avec 50 phrases réalistes couvrant tous les cas.

Gates de validation:
- Accuracy ≥ 60%
- Fallback rate ≤ 2%
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path

from jeffrey.core.emotion_backend import ProtoEmotionDetector

# Dataset de test avec 50 phrases réalistes FR/EN
SMOKE_TEST_CASES = [
    # JOY (6 cases)
    {"text": "Je suis tellement heureux aujourd'hui !", "expected": "joy", "lang": "fr"},
    {"text": "I'm so excited about this news!", "expected": "joy", "lang": "en"},
    {"text": "C'est fantastique, j'adore ça !", "expected": "joy", "lang": "fr"},
    {"text": "This is absolutely wonderful!", "expected": "joy", "lang": "en"},
    {"text": "Quelle joie de vous revoir !", "expected": "joy", "lang": "fr"},
    {"text": "I'm thrilled with the results!", "expected": "joy", "lang": "en"},
    # SADNESS (6 cases)
    {"text": "Je me sens vraiment triste.", "expected": "sadness", "lang": "fr"},
    {"text": "I'm feeling so down today.", "expected": "sadness", "lang": "en"},
    {"text": "C'est déprimant cette situation.", "expected": "sadness", "lang": "fr"},
    {"text": "This makes me feel really sad.", "expected": "sadness", "lang": "en"},
    {"text": "Je suis découragé par les événements.", "expected": "sadness", "lang": "fr"},
    {"text": "I'm heartbroken by this news.", "expected": "sadness", "lang": "en"},
    # ANGER (6 cases)
    {"text": "Ça m'énerve vraiment !", "expected": "anger", "lang": "fr"},
    {"text": "I'm so angry about this!", "expected": "anger", "lang": "en"},
    {"text": "Je suis furieux contre toi !", "expected": "anger", "lang": "fr"},
    {"text": "This is absolutely infuriating!", "expected": "anger", "lang": "en"},
    {"text": "C'est inacceptable, je suis en colère !", "expected": "anger", "lang": "fr"},
    {"text": "I can't stand this anymore!", "expected": "anger", "lang": "en"},
    # FEAR (6 cases)
    {"text": "J'ai peur de ce qui va arriver.", "expected": "fear", "lang": "fr"},
    {"text": "I'm terrified of the consequences.", "expected": "fear", "lang": "en"},
    {"text": "Ça m'inquiète énormément.", "expected": "fear", "lang": "fr"},
    {"text": "This really scares me.", "expected": "fear", "lang": "en"},
    {"text": "Je suis anxieux à propos de ça.", "expected": "fear", "lang": "fr"},
    {"text": "I'm worried about what might happen.", "expected": "fear", "lang": "en"},
    # SURPRISE (6 cases)
    {"text": "Wow, je ne m'y attendais pas !", "expected": "surprise", "lang": "fr"},
    {"text": "I can't believe this happened!", "expected": "surprise", "lang": "en"},
    {"text": "C'est incroyable, quelle surprise !", "expected": "surprise", "lang": "fr"},
    {"text": "This is absolutely amazing!", "expected": "surprise", "lang": "en"},
    {"text": "Je suis stupéfait par ce résultat.", "expected": "surprise", "lang": "fr"},
    {"text": "What an unexpected turn of events!", "expected": "surprise", "lang": "en"},
    # DISGUST (6 cases)
    {"text": "Beurk, c'est dégoûtant !", "expected": "disgust", "lang": "fr"},
    {"text": "That's absolutely disgusting!", "expected": "disgust", "lang": "en"},
    {"text": "C'est répugnant, je ne supporte pas.", "expected": "disgust", "lang": "fr"},
    {"text": "This makes me feel sick.", "expected": "disgust", "lang": "en"},
    {"text": "Ça me dégoûte profondément.", "expected": "disgust", "lang": "fr"},
    {"text": "I find this revolting.", "expected": "disgust", "lang": "en"},
    # NEUTRAL (6 cases)
    {"text": "OK, j'ai bien noté.", "expected": "neutral", "lang": "fr"},
    {"text": "Alright, I understand.", "expected": "neutral", "lang": "en"},
    {"text": "Merci pour l'information.", "expected": "neutral", "lang": "fr"},
    {"text": "Thanks for letting me know.", "expected": "neutral", "lang": "en"},
    {"text": "Je vois, c'est noté.", "expected": "neutral", "lang": "fr"},
    {"text": "Got it, thanks.", "expected": "neutral", "lang": "en"},
    # FRUSTRATION (6 cases)
    {"text": "C'est vraiment frustrant à la longue.", "expected": "frustration", "lang": "fr"},
    {"text": "This is getting really frustrating.", "expected": "frustration", "lang": "en"},
    {"text": "Je suis exaspéré par cette situation.", "expected": "frustration", "lang": "fr"},
    {"text": "I'm getting tired of this.", "expected": "frustration", "lang": "en"},
    {"text": "Ça devient pénible.", "expected": "frustration", "lang": "fr"},
    {"text": "This is becoming exhausting.", "expected": "frustration", "lang": "en"},
    # EDGE CASES (8 cases) - négations, ironie, mixte
    {"text": "Je ne suis pas heureux du tout.", "expected": "sadness", "lang": "fr", "note": "négation"},
    {"text": "I'm not angry at all.", "expected": "neutral", "lang": "en", "note": "négation forte"},
    {"text": "Super, encore un problème... génial.", "expected": "frustration", "lang": "fr", "note": "ironie"},
    {"text": "Great, another issue... fantastic.", "expected": "frustration", "lang": "en", "note": "ironie"},
    {"text": "Je suis content mais un peu inquiet.", "expected": "joy", "lang": "fr", "note": "émotion mixte"},
    {"text": "I'm happy but also worried.", "expected": "joy", "lang": "en", "note": "émotion mixte"},
    {"text": "Ça ne me dérange pas.", "expected": "neutral", "lang": "fr", "note": "négation neutre"},
    {"text": "I don't really care about this.", "expected": "neutral", "lang": "en", "note": "négation neutre"},
]


class SmokeTestRunner:
    def __init__(self):
        self.detector = None
        self.results = []

    def setup_classifier(self):
        """Initialize classifier with trained prototypes."""
        prototypes_path = Path("data/prototypes.npz")
        if not prototypes_path.exists():
            raise FileNotFoundError("❌ Prototypes not found! Run train_prototypes_optimized.py first")

        print("🔧 Initializing ProtoEmotionDetector...")
        self.detector = ProtoEmotionDetector()
        print("✅ Detector loaded and ready")

    def run_test(self, test_case):
        """Run single test case and return result."""
        text = test_case["text"]
        expected = test_case["expected"]
        lang = test_case["lang"]
        note = test_case.get("note", "")

        try:
            # Get prediction and probabilities
            prediction = self.detector.predict_label(text)
            probabilities, used_fallback = self.detector.predict_proba(text)
            confidence = probabilities.get(prediction, 0.0)

            # Check if prediction matches expected
            correct = prediction == expected

            # Determine if it's a fallback (low confidence or explicit fallback)
            is_fallback = used_fallback or confidence < 0.3

            result = {
                "text": text,
                "expected": expected,
                "predicted": prediction,
                "confidence": confidence,
                "correct": correct,
                "is_fallback": is_fallback,
                "lang": lang,
                "note": note,
            }

            return result

        except Exception as e:
            return {
                "text": text,
                "expected": expected,
                "predicted": "ERROR",
                "confidence": 0.0,
                "correct": False,
                "is_fallback": True,
                "lang": lang,
                "note": f"ERROR: {str(e)}",
            }

    def run_all_tests(self):
        """Run all smoke tests and calculate metrics."""
        print("🚀 Starting Jeffrey OS v2.4.2 Smoke Test FR/EN...")
        print(f"📊 Testing {len(SMOKE_TEST_CASES)} cases\n")

        self.setup_classifier()

        # Run all tests
        for i, test_case in enumerate(SMOKE_TEST_CASES, 1):
            result = self.run_test(test_case)
            self.results.append(result)

            # Print progress
            status = "✅" if result["correct"] else "❌"
            conf_str = f"{result['confidence']:.3f}"
            note_str = f" ({result['note']})" if result['note'] else ""

            print(
                f"{status} {i:2d}/50: {result['predicted']:12s} | {conf_str} | {result['lang']} | {result['text'][:50]}...{note_str}"
            )

        # Calculate final metrics
        return self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate and display final metrics."""
        total_cases = len(self.results)
        correct_cases = sum(1 for r in self.results if r["correct"])
        fallback_cases = sum(1 for r in self.results if r["is_fallback"])
        error_cases = sum(1 for r in self.results if r["predicted"] == "ERROR")

        accuracy = correct_cases / total_cases * 100
        fallback_rate = fallback_cases / total_cases * 100
        error_rate = error_cases / total_cases * 100

        print("\n" + "=" * 60)
        print("📊 RÉSULTATS DU SMOKE TEST FR/EN")
        print("=" * 60)
        print(f"Total cases      : {total_cases}")
        print(f"Correct          : {correct_cases}")
        print(f"Accuracy         : {accuracy:.1f}%")
        print(f"Fallback cases   : {fallback_cases}")
        print(f"Fallback rate    : {fallback_rate:.1f}%")
        print(f"Error cases      : {error_cases}")
        print(f"Error rate       : {error_rate:.1f}%")
        print("=" * 60)

        # Check gates
        accuracy_gate = accuracy >= 60.0
        fallback_gate = fallback_rate <= 2.0

        print("\n🎯 VALIDATION DES GATES:")
        print(
            f"{'✅' if accuracy_gate else '❌'} Accuracy ≥ 60%    : {accuracy:.1f}% {'PASS' if accuracy_gate else 'FAIL'}"
        )
        print(
            f"{'✅' if fallback_gate else '❌'} Fallback ≤ 2%     : {fallback_rate:.1f}% {'PASS' if fallback_gate else 'FAIL'}"
        )

        overall_pass = accuracy_gate and fallback_gate
        print(f"\n{'🎉 SMOKE TEST RÉUSSI !' if overall_pass else '🚨 SMOKE TEST ÉCHOUÉ !'}")

        # Show emotion breakdown
        self.show_emotion_breakdown()

        # Show challenging cases
        self.show_challenging_cases()

        return overall_pass

    def show_emotion_breakdown(self):
        """Show per-emotion accuracy breakdown."""
        from collections import defaultdict

        emotion_stats = defaultdict(lambda: {"total": 0, "correct": 0})

        for result in self.results:
            expected = result["expected"]
            emotion_stats[expected]["total"] += 1
            if result["correct"]:
                emotion_stats[expected]["correct"] += 1

        print("\n📈 PRÉCISION PAR ÉMOTION:")
        print("-" * 40)
        for emotion in sorted(emotion_stats.keys()):
            stats = emotion_stats[emotion]
            accuracy = stats["correct"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"{emotion:12s}: {stats['correct']:2d}/{stats['total']:2d} ({accuracy:5.1f}%)")

    def show_challenging_cases(self):
        """Show the most challenging cases (errors and low confidence)."""
        errors = [r for r in self.results if not r["correct"]]

        if errors:
            print(f"\n🔍 CAS DIFFICILES ({len(errors)} erreurs):")
            print("-" * 80)
            for err in errors[:5]:  # Show top 5 errors
                print(
                    f"❌ Expected: {err['expected']:12s} | Got: {err['predicted']:12s} | Conf: {err['confidence']:.3f}"
                )
                print(f"   Text: {err['text']}")
                if err['note']:
                    print(f"   Note: {err['note']}")
                print()


def main():
    """Run smoke test."""
    runner = SmokeTestRunner()
    try:
        success = runner.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"💥 ERREUR CRITIQUE: {e}")
        return 2


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
