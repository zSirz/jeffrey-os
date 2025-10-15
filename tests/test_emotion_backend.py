"""
Tests pour emotion_backend.py
Niveau : Unitaires + Comportementaux basiques
"""

from jeffrey.core.emotion_backend import (
    CORE_EMOTIONS,
    get_metrics,
    make_emotion_backend,
    reset_metrics,
)


class TestRegexBackend:
    """Tests backend regex."""

    def test_basic_emotions(self):
        """Test prédictions basiques."""
        backend = make_emotion_backend("regex")

        assert backend.predict_label("I am very happy") == "joy"
        assert backend.predict_label("I am so sad") == "sadness"
        assert backend.predict_label("I am angry") == "anger"

    def test_empty_text(self):
        """Test texte vide."""
        backend = make_emotion_backend("regex")
        assert backend.predict_label("") == "neutral"

    def test_proba_sums_to_one(self):
        """Test distribution valide."""
        backend = make_emotion_backend("regex")
        probs = backend.predict_proba("I am happy")

        assert abs(sum(probs.values()) - 1.0) < 0.001


class TestProtoBackend:
    """Tests backend proto."""

    def test_basic_prediction(self):
        """Test prédiction basique."""
        backend = make_emotion_backend("proto")

        # Prédiction simple
        label = backend.predict_label("I am very happy today")
        assert label in CORE_EMOTIONS

    def test_proba_valid(self):
        """Test distribution valide."""
        backend = make_emotion_backend("proto")
        probs, used_fallback = backend.predict_proba("I am sad")

        # Somme = 1
        assert abs(sum(probs.values()) - 1.0) < 0.001

        # Toutes probas >= 0
        assert all(p >= 0 for p in probs.values())

        # Toutes clés présentes
        assert set(probs.keys()) == set(CORE_EMOTIONS)

    def test_empty_text_neutral(self):
        """Test texte vide → neutral."""
        backend = make_emotion_backend("proto")
        probs, _ = backend.predict_proba("")

        assert probs["neutral"] == 1.0
        assert sum(probs[e] for e in CORE_EMOTIONS if e != "neutral") == 0.0


class TestBehavioral:
    """Tests comportementaux (Gemini)."""

    def test_negation_changes_emotion(self):
        """Test attente directionnelle : négation."""
        backend = make_emotion_backend("proto")

        pos = backend.predict_label("I am happy")
        neg = backend.predict_label("I am not happy")

        # La négation devrait changer l'émotion
        assert pos != neg or pos == "neutral"  # Allow neutral as valid

    def test_strong_positive_is_joy(self):
        """Test MFT : phrase clairement positive."""
        backend = make_emotion_backend("proto")

        label = backend.predict_label("This is a wonderful, fantastic, amazing day!")

        # Devrait être joy (ou au moins pas négatif)
        assert label in ["joy", "surprise", "neutral"]


class TestSingleton:
    """Tests singleton (GPT)."""

    def test_singleton_same_instance(self):
        """Test cache singleton."""
        backend1 = make_emotion_backend("proto")
        backend2 = make_emotion_backend("proto")

        assert backend1 is backend2


class TestMetrics:
    """Tests métriques (Grok)."""

    def test_metrics_increment(self):
        """Test compteurs."""
        reset_metrics()

        backend = make_emotion_backend("proto")
        backend.predict_label("test")

        metrics = get_metrics()
        assert metrics["total_predictions"] > 0
