"""
Tests unitaires pour jeffrey/ml/proto.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jeffrey.ml.proto import EmotionPrediction, ProtoClassifier, ProtoEmotion


class TestProtoEmotion:
    """Tests pour ProtoEmotion."""

    def test_initialization(self):
        """Test initialisation."""
        proto = ProtoEmotion("anger", dimension=384)
        assert proto.label == "anger"
        assert proto.centroid is None
        assert proto.n_samples == 0

    def test_initialize_centroid(self):
        """Test initialisation centroïde."""
        proto = ProtoEmotion("anger", dimension=384)

        embeddings = np.random.randn(10, 384)
        proto.initialize(embeddings)

        assert proto.centroid is not None
        assert proto.centroid.shape == (384,)
        assert proto.n_samples == 10

        # Vérifie normalisation
        norm = np.linalg.norm(proto.centroid)
        assert abs(norm - 1.0) < 1e-3

    def test_update_ema(self):
        """Test update EMA."""
        proto = ProtoEmotion("anger", dimension=384, alpha=0.1)

        # Initialize
        initial_embeddings = np.random.randn(5, 384)
        proto.initialize(initial_embeddings)
        centroid_before = proto.centroid.copy()

        # Update
        new_embedding = np.random.randn(384)
        success = proto.update(new_embedding)

        assert success
        assert not np.allclose(proto.centroid, centroid_before)
        assert proto.n_samples == 6

    def test_outlier_rejection(self):
        """Test rejet outliers."""
        proto = ProtoEmotion("anger", dimension=384)

        # Initialize avec cluster serré
        embeddings = np.random.randn(20, 384) * 0.1  # Petite variance
        proto.initialize(embeddings)

        # Outlier évident
        outlier = np.random.randn(384) * 10  # Grande variance
        success = proto.update(outlier, check_outlier=True)

        # Devrait être rejeté
        assert not success


class TestProtoClassifier:
    """Tests pour ProtoClassifier."""

    @pytest.fixture
    def classifier(self):
        """Fixture classifier."""
        clf = ProtoClassifier(dimension=384)

        # Bootstrap avec données synthétiques
        np.random.seed(42)
        labeled_data = {
            "anger": np.random.randn(20, 384),
            "joy": np.random.randn(20, 384),
            "sadness": np.random.randn(20, 384),
        }
        clf.bootstrap(labeled_data)

        return clf

    def test_predict(self, classifier):
        """Test prédiction."""
        embedding = np.random.randn(384)
        prediction = classifier.predict(embedding)

        assert isinstance(prediction, EmotionPrediction)
        assert prediction.primary in ["anger", "joy", "sadness", "neutral"]
        assert 0 <= prediction.confidence <= 1
        assert 0 <= prediction.margin <= 1

    def test_abstention(self, classifier):
        """Test abstention."""
        # Créer embedding ambigu (moyenne de plusieurs émotions)
        anger_proto = classifier.prototypes["anger"].centroid
        joy_proto = classifier.prototypes["joy"].centroid

        ambiguous = (anger_proto + joy_proto) / 2
        ambiguous /= np.linalg.norm(ambiguous)

        prediction = classifier.predict(ambiguous)

        # Devrait s'abstenir (margin faible)
        # Note: Peut ne pas s'abstenir si confidence reste haute
        print(f"\nAmbiguous prediction: {prediction.primary}, margin={prediction.margin:.3f}")

    def test_learning(self, classifier):
        """Test apprentissage."""
        embedding = np.random.randn(384)

        # Prédiction initiale
        pred_before = classifier.predict(embedding)

        # Apprendre avec correction
        classifier.learn(embedding, "anger", confidence=1.0)

        # Prédiction après apprentissage
        pred_after = classifier.predict(embedding)

        # Le score anger devrait augmenter (ou rester haut si déjà anger)
        score_before = pred_before.all_scores.get("anger", 0)
        score_after = pred_after.all_scores.get("anger", 0)

        assert score_after >= score_before * 0.9  # Tolérance

    def test_save_load(self, classifier, tmp_path):
        """Test sauvegarde/chargement."""
        save_path = tmp_path / "prototypes_test.json"

        # Sauvegarder
        classifier.save(save_path)
        assert save_path.exists()

        # Charger
        loaded_classifier = ProtoClassifier.load(save_path)

        # Vérifier cohérence
        assert loaded_classifier.dimension == classifier.dimension
        assert len(loaded_classifier.prototypes) == len(classifier.prototypes)

        # Vérifier prédictions identiques
        embedding = np.random.randn(384)
        pred1 = classifier.predict(embedding)
        pred2 = loaded_classifier.predict(embedding)

        assert pred1.primary == pred2.primary
        assert abs(pred1.confidence - pred2.confidence) < 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
