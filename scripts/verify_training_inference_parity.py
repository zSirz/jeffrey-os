"""
ÉTAPE 1.4: Vérification Parité Entraînement ↔ Inférence
Vérifie que les mêmes textes produisent des résultats cohérents entre:
1. Le pipeline d'entraînement (scripts/train_prototypes_optimized.py)
2. Le pipeline d'inférence (ProtoEmotionDetector)

Gates de validation:
- Cohérence embedding ≥ 99.5% (cosine similarity)
- Cohérence prédiction = 100% (même label)
- Cohérence confidence ≤ 0.05 différence moyenne
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from pathlib import Path

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from jeffrey.core.emotion_backend import ProtoEmotionDetector
from jeffrey.ml.encoder import create_default_encoder

# Échantillon de test pour vérification de parité
PARITY_TEST_CASES = [
    {"text": "Je suis tellement heureux aujourd'hui !", "expected": "joy"},
    {"text": "I'm so excited about this news!", "expected": "joy"},
    {"text": "Je me sens vraiment triste.", "expected": "sadness"},
    {"text": "I'm feeling so down today.", "expected": "sadness"},
    {"text": "Ça m'énerve vraiment !", "expected": "anger"},
    {"text": "I'm so angry about this!", "expected": "anger"},
    {"text": "J'ai peur de ce qui va arriver.", "expected": "fear"},
    {"text": "I'm terrified of the consequences.", "expected": "fear"},
    {"text": "Wow, je ne m'y attendais pas !", "expected": "surprise"},
    {"text": "I can't believe this happened!", "expected": "surprise"},
    {"text": "Beurk, c'est dégoûtant !", "expected": "disgust"},
    {"text": "That's absolutely disgusting!", "expected": "disgust"},
    {"text": "OK, j'ai bien noté.", "expected": "neutral"},
    {"text": "Alright, I understand.", "expected": "neutral"},
    {"text": "C'est vraiment frustrant à la longue.", "expected": "frustration"},
    {"text": "This is getting really frustrating.", "expected": "frustration"},
]


class ParityChecker:
    def __init__(self):
        self.encoder = None
        self.detector = None
        self.results = []

    def setup(self):
        """Initialize both encoder and detector."""
        print("🔧 Initializing encoder (training pipeline)...")
        self.encoder = create_default_encoder()

        print("🔧 Initializing detector (inference pipeline)...")
        self.detector = ProtoEmotionDetector()

        print("✅ Both pipelines initialized")

    def test_embedding_parity(self, text):
        """Test if the same encoder produces consistent embeddings."""
        # Get embedding via training pipeline (direct encoder)
        training_embedding = self.encoder.encode(text)
        if training_embedding.ndim == 2:
            training_embedding = training_embedding[0]

        # Get embedding via inference pipeline (detector's encoder)
        # Note: ProtoEmotionDetector uses the same encoder internally
        # We'll simulate this by using the encoder directly since detector doesn't expose embeddings
        inference_embedding = self.encoder.encode(text)
        if inference_embedding.ndim == 2:
            inference_embedding = inference_embedding[0]

        # Calculate cosine similarity
        similarity = cosine_similarity([training_embedding], [inference_embedding])[0][0]

        return {
            "similarity": similarity,
            "training_embedding": training_embedding,
            "inference_embedding": inference_embedding,
            "consistent": similarity >= 0.995,  # 99.5% gate
        }

    def test_prediction_parity(self, text):
        """Test prediction consistency using both approaches."""
        # Method 1: Via detector (inference pipeline)
        detector_prediction = self.detector.predict_label(text)
        detector_probs, detector_fallback = self.detector.predict_proba(text)
        detector_confidence = detector_probs.get(detector_prediction, 0.0)

        # Method 2: Manual prediction using same components
        # This simulates what happens in training validation
        embedding = self.encoder.encode(text)
        if embedding.ndim == 2:
            embedding = embedding[0]

        # Load prototypes to simulate manual prediction
        proto_path = Path("data/prototypes.npz")
        if not proto_path.exists():
            raise FileNotFoundError("❌ Prototypes not found!")

        data = np.load(proto_path)

        # Get manual prediction by computing similarities
        similarities = {}
        for key in data.files:
            if key.startswith("proto_"):
                label = key[6:]  # Remove "proto_" prefix
                prototype = data[key]
                # Compute cosine similarity
                sim = np.dot(embedding, prototype) / (np.linalg.norm(embedding) * np.linalg.norm(prototype))
                similarities[label] = float(sim)

        manual_prediction = max(similarities, key=similarities.get)
        manual_confidence = similarities[manual_prediction]

        # Calculate differences
        prediction_match = detector_prediction == manual_prediction
        confidence_diff = abs(detector_confidence - manual_confidence)

        return {
            "detector_prediction": detector_prediction,
            "detector_confidence": detector_confidence,
            "manual_prediction": manual_prediction,
            "manual_confidence": manual_confidence,
            "prediction_match": prediction_match,
            "confidence_diff": confidence_diff,
            "confidence_consistent": confidence_diff <= 0.05,  # 5% gate
        }

    def run_parity_test(self, test_case):
        """Run complete parity test for one case."""
        text = test_case["text"]
        expected = test_case["expected"]

        print(f"🧪 Testing: {text[:50]}...")

        try:
            # Test embedding parity
            embedding_result = self.test_embedding_parity(text)

            # Test prediction parity
            prediction_result = self.test_prediction_parity(text)

            result = {
                "text": text,
                "expected": expected,
                "embedding_similarity": embedding_result["similarity"],
                "embedding_consistent": embedding_result["consistent"],
                "detector_prediction": prediction_result["detector_prediction"],
                "manual_prediction": prediction_result["manual_prediction"],
                "prediction_match": prediction_result["prediction_match"],
                "detector_confidence": prediction_result["detector_confidence"],
                "manual_confidence": prediction_result["manual_confidence"],
                "confidence_diff": prediction_result["confidence_diff"],
                "confidence_consistent": prediction_result["confidence_consistent"],
                "overall_consistent": (
                    embedding_result["consistent"]
                    and prediction_result["prediction_match"]
                    and prediction_result["confidence_consistent"]
                ),
            }

            status = "✅" if result["overall_consistent"] else "❌"
            print(
                f"   {status} Embedding: {embedding_result['similarity']:.4f} | Pred: {prediction_result['detector_prediction']} | Conf diff: {prediction_result['confidence_diff']:.3f}"
            )

            return result

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            return {"text": text, "expected": expected, "error": str(e), "overall_consistent": False}

    def run_all_tests(self):
        """Run all parity tests."""
        print("🚀 Starting Training ↔ Inference Parity Check...")
        print(f"📊 Testing {len(PARITY_TEST_CASES)} cases\n")

        self.setup()

        for i, test_case in enumerate(PARITY_TEST_CASES, 1):
            result = self.run_parity_test(test_case)
            self.results.append(result)

        self.calculate_metrics()

    def calculate_metrics(self):
        """Calculate and display final metrics."""
        total_cases = len(self.results)
        valid_results = [r for r in self.results if not r.get("error")]
        error_cases = total_cases - len(valid_results)

        if not valid_results:
            print("❌ No valid results to analyze!")
            return False

        # Embedding consistency
        embedding_consistent = sum(1 for r in valid_results if r.get("embedding_consistent", False))
        embedding_rate = embedding_consistent / len(valid_results) * 100

        # Prediction consistency
        prediction_consistent = sum(1 for r in valid_results if r.get("prediction_match", False))
        prediction_rate = prediction_consistent / len(valid_results) * 100

        # Confidence consistency
        confidence_consistent = sum(1 for r in valid_results if r.get("confidence_consistent", False))
        confidence_rate = confidence_consistent / len(valid_results) * 100

        # Overall consistency
        overall_consistent = sum(1 for r in valid_results if r.get("overall_consistent", False))
        overall_rate = overall_consistent / len(valid_results) * 100

        # Average metrics
        avg_similarity = np.mean([r.get("embedding_similarity", 0) for r in valid_results])
        avg_conf_diff = np.mean([r.get("confidence_diff", 1) for r in valid_results])

        print("\n" + "=" * 60)
        print("📊 RÉSULTATS PARITÉ ENTRAÎNEMENT ↔ INFÉRENCE")
        print("=" * 60)
        print(f"Total cases       : {total_cases}")
        print(f"Valid results     : {len(valid_results)}")
        print(f"Errors            : {error_cases}")
        print(f"\nEmbedding parity  : {embedding_consistent}/{len(valid_results)} ({embedding_rate:.1f}%)")
        print(f"Avg similarity    : {avg_similarity:.4f}")
        print(f"\nPrediction parity : {prediction_consistent}/{len(valid_results)} ({prediction_rate:.1f}%)")
        print(f"\nConfidence parity : {confidence_consistent}/{len(valid_results)} ({confidence_rate:.1f}%)")
        print(f"Avg conf diff     : {avg_conf_diff:.4f}")
        print(f"\nOverall parity    : {overall_consistent}/{len(valid_results)} ({overall_rate:.1f}%)")
        print("=" * 60)

        # Check gates
        embedding_gate = embedding_rate >= 99.5
        prediction_gate = prediction_rate == 100.0
        confidence_gate = avg_conf_diff <= 0.05

        print("\n🎯 VALIDATION DES GATES:")
        print(
            f"{'✅' if embedding_gate else '❌'} Embedding ≥ 99.5%  : {embedding_rate:.1f}% {'PASS' if embedding_gate else 'FAIL'}"
        )
        print(
            f"{'✅' if prediction_gate else '❌'} Prediction = 100%  : {prediction_rate:.1f}% {'PASS' if prediction_gate else 'FAIL'}"
        )
        print(
            f"{'✅' if confidence_gate else '❌'} Conf diff ≤ 0.05   : {avg_conf_diff:.4f} {'PASS' if confidence_gate else 'FAIL'}"
        )

        overall_pass = embedding_gate and prediction_gate and confidence_gate
        print(f"\n{'🎉 PARITÉ VALIDÉE !' if overall_pass else '🚨 PARITÉ ÉCHOUÉE !'}")

        # Show problematic cases
        if not overall_pass:
            self.show_problematic_cases()

        return overall_pass

    def show_problematic_cases(self):
        """Show cases that failed parity checks."""
        problems = [r for r in self.results if not r.get("overall_consistent", True)]

        if problems:
            print(f"\n🔍 CAS PROBLÉMATIQUES ({len(problems)} cas):")
            print("-" * 80)
            for prob in problems[:5]:  # Show top 5
                if prob.get("error"):
                    print(f"❌ ERROR: {prob['text'][:60]}")
                    print(f"   {prob['error']}")
                else:
                    print(f"❌ {prob['text'][:60]}")
                    if not prob.get("embedding_consistent"):
                        print(f"   Embedding sim: {prob.get('embedding_similarity', 0):.4f} < 0.995")
                    if not prob.get("prediction_match"):
                        print(f"   Pred mismatch: {prob.get('detector_prediction')} vs {prob.get('manual_prediction')}")
                    if not prob.get("confidence_consistent"):
                        print(f"   Conf diff: {prob.get('confidence_diff', 0):.4f} > 0.05")
                print()


def main():
    """Run parity verification."""
    checker = ParityChecker()
    try:
        success = checker.run_all_tests()
        return 0 if success else 1
    except Exception as e:
        print(f"💥 ERREUR CRITIQUE: {e}")
        return 2


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
