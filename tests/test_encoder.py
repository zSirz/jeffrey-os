"""
Tests unitaires pour jeffrey/ml/encoder.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jeffrey.ml.encoder import create_default_encoder


class TestSentenceEncoder:
    """Tests pour SentenceEncoder."""

    @pytest.fixture
    def encoder(self):
        """Fixture encoder."""
        return create_default_encoder()

    def test_initialization(self, encoder):
        """Test initialisation."""
        assert encoder is not None
        assert encoder.get_dimension() == 384

    def test_encode_single_text(self, encoder):
        """Test encodage texte unique."""
        text = "Je suis content"
        embedding = encoder.encode(text)

        assert embedding.shape == (1, 384)
        assert np.isfinite(embedding).all()

        # Vérifie normalisation L2
        norm = np.linalg.norm(embedding)
        assert abs(norm - 1.0) < 1e-3

    def test_encode_batch(self, encoder):
        """Test encodage batch."""
        texts = ["Je suis content", "Je suis triste", "Je suis en colère"]
        embeddings = encoder.encode(texts)

        assert embeddings.shape == (3, 384)
        assert np.isfinite(embeddings).all()

    def test_cache_hit(self, encoder):
        """Test cache LRU."""
        text = "Test cache"

        # Premier appel
        _ = encoder.encode(text)
        initial_calls = encoder.stats["encode_calls"]

        # Deuxième appel (devrait être caché)
        _ = encoder.encode(text)

        stats = encoder.get_stats()
        assert stats["cache_hits"] > 0
        assert stats["cache_hit_rate_pct"] > 0

    def test_performance(self, encoder):
        """Test latence."""
        texts = ["Test performance"] * 16

        import time

        start = time.time()
        _ = encoder.encode(texts, batch_size=16)
        latency_ms = (time.time() - start) * 1000

        # Devrait être < 100ms pour 16 textes
        assert latency_ms < 100

        avg_per_text = latency_ms / len(texts)
        print(f"\nLatency: {latency_ms:.2f}ms total, {avg_per_text:.2f}ms/text")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
