"""
Tests d'intégration complets pour le système ML d'émotion.
Couvre : fonctionnalité, performance, fallback, edge cases, comportement du modèle.
"""
import pytest
import asyncio
import time
from unittest.mock import patch, MagicMock
import numpy as np

from jeffrey.ml.emotion_ml_adapter import EmotionMLAdapter
from jeffrey.core.orchestration.jeffrey_orchestrator import JeffreyOrchestrator
from jeffrey.core.neuralbus.bus import NeuralBus

# Fixtures
@pytest.fixture
async def emotion_adapter():
    """Fixture pour l'adapter ML"""
    adapter = await EmotionMLAdapter.get_instance()
    yield adapter
    # Cleanup si nécessaire

@pytest.fixture
async def orchestrator():
    """Fixture pour l'orchestrateur avec ML"""
    neural_bus = NeuralBus()
    emotion_adapter = await EmotionMLAdapter.get_instance()
    orch = JeffreyOrchestrator(
        neural_bus=neural_bus,
        emotion_detector=emotion_adapter
    )
    yield orch

# Tests Fonctionnels
@pytest.mark.asyncio
async def test_ml_adapter_basic_emotions(emotion_adapter):
    """Test détection des émotions de base"""
    test_cases = [
        ("Je suis très content aujourd'hui!", "joy", 0.7),
        ("Je suis triste et déprimé", "sadness", 0.6),
        ("J'ai peur du noir", "fear", 0.5),
        ("C'est vraiment surprenant!", "surprise", 0.5),
        ("Je suis en colère", "anger", 0.5),
    ]

    for text, expected_emotion, min_confidence in test_cases:
        result = await emotion_adapter.detect_emotion(text)

        assert result["success"] is True
        assert result["emotion"] == expected_emotion
        assert result["confidence"] >= min_confidence
        assert "latency_ms" in result
        assert result["latency_ms"] < 200  # Max 200ms

        print(f"✅ '{text[:30]}...' → {result['emotion']} ({result['confidence']:.2%})")

@pytest.mark.asyncio
async def test_ml_adapter_invalid_inputs(emotion_adapter):
    """Test gestion des inputs invalides"""

    # Texte vide
    result = await emotion_adapter.detect_emotion("")
    assert result["success"] is False
    assert result["error"] == "Empty text"

    # Texte trop long (sera tronqué)
    long_text = "a" * 3000
    result = await emotion_adapter.detect_emotion(long_text)
    assert result["success"] is True
    assert result["text_length"] <= 2000

    # Type invalide géré avant l'appel
    # (L'adapter attend un str, donc on ne teste pas les types invalides ici)

@pytest.mark.asyncio
async def test_ml_adapter_fallback_on_timeout(emotion_adapter, monkeypatch):
    """Test fallback vers regex en cas de timeout"""

    # Simuler un timeout du modèle ML
    async def slow_predict(*args):
        await asyncio.sleep(5)  # Simuler une opération lente
        return "joy", 0.9, {"joy": 0.9}

    with patch.object(emotion_adapter, '_predict_sync', side_effect=slow_predict):
        # Réduire le timeout pour le test
        emotion_adapter.timeout_ms = 100

        result = await emotion_adapter.detect_emotion("Test timeout")

        assert result["success"] is True
        assert "fallback" in result["method"]
        assert result["latency_ms"] < 200

@pytest.mark.asyncio
async def test_ml_adapter_fallback_on_error(emotion_adapter):
    """Test fallback vers regex en cas d'erreur ML"""

    # Simuler une erreur du modèle ML
    with patch.object(emotion_adapter._encoder, 'encode', side_effect=Exception("Model error")):
        result = await emotion_adapter.detect_emotion("Test error")

        assert result["success"] is True
        assert "fallback" in result["method"]
        assert result["emotion"] in ["joy", "sadness", "anger", "fear", "surprise", "neutral"]

@pytest.mark.asyncio
async def test_ml_adapter_thread_safety(emotion_adapter):
    """Test thread-safety avec requêtes concurrentes"""

    async def detect_concurrent(text, index):
        result = await emotion_adapter.detect_emotion(f"{text} {index}")
        return result

    # Lancer 50 requêtes concurrentes
    tasks = [
        detect_concurrent("Concurrent test", i)
        for i in range(50)
    ]

    results = await asyncio.gather(*tasks)

    # Vérifier que toutes ont réussi
    assert all(r["success"] for r in results)

    # Vérifier les stats
    stats = emotion_adapter.get_stats()
    assert stats["total_predictions"] >= 50

# Tests de Performance
@pytest.mark.asyncio
async def test_ml_adapter_performance(emotion_adapter):
    """Test performance avec benchmarks"""

    # Warmup
    await emotion_adapter.detect_emotion("warmup")

    latencies = []

    # 100 prédictions séquentielles
    for i in range(100):
        start = time.perf_counter()
        result = await emotion_adapter.detect_emotion(f"Performance test message {i}")
        latencies.append((time.perf_counter() - start) * 1000)

        assert result["success"] is True

    # Calculs statistiques
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)

    print(f"\n📊 Performance Stats:")
    print(f"  P50: {p50:.1f}ms")
    print(f"  P95: {p95:.1f}ms")
    print(f"  P99: {p99:.1f}ms")

    # Assertions
    assert p50 < 50   # P50 sous 50ms
    assert p95 < 150  # P95 sous 150ms
    assert p99 < 250  # P99 sous 250ms

# Tests de Comportement du Modèle
@pytest.mark.asyncio
async def test_model_invariance(emotion_adapter):
    """Test invariance du modèle (résultats cohérents)"""

    # Test changement de nom ne doit pas changer l'émotion
    pairs = [
        ("Je suis heureux", "Marie est heureuse"),
        ("Je suis triste", "Paul est triste"),
        ("J'ai peur", "Elle a peur")
    ]

    for text1, text2 in pairs:
        result1 = await emotion_adapter.detect_emotion(text1)
        result2 = await emotion_adapter.detect_emotion(text2)

        # Même émotion détectée
        assert result1["emotion"] == result2["emotion"], \
            f"Invariance failed: '{text1}' vs '{text2}'"

        # Confidence similaire (tolérance 20%)
        assert abs(result1["confidence"] - result2["confidence"]) < 0.2

@pytest.mark.asyncio
async def test_model_directionality(emotion_adapter):
    """Test directionnalité (intensificateurs)"""

    pairs = [
        ("Je suis content", "Je suis très content"),
        ("Je suis triste", "Je suis extrêmement triste"),
    ]

    for base_text, intense_text in pairs:
        base_result = await emotion_adapter.detect_emotion(base_text)
        intense_result = await emotion_adapter.detect_emotion(intense_text)

        # Même émotion
        assert base_result["emotion"] == intense_result["emotion"]

        # Confidence plus élevée pour version intense (ou égale)
        assert intense_result["confidence"] >= base_result["confidence"] - 0.05

@pytest.mark.asyncio
async def test_model_edge_cases(emotion_adapter):
    """Test cas limites du modèle"""

    edge_cases = [
        ("", "neutral"),  # Vide -> erreur ou neutral
        ("...", "neutral"),  # Ponctuation seule
        ("😀😃😄", "joy"),  # Emojis positifs
        ("😢😭", "sadness"),  # Emojis tristes
        ("AAAAAHHHHH!!!", "surprise"),  # Cri
        ("12345", "neutral"),  # Nombres
    ]

    for text, expected_emotion in edge_cases:
        result = await emotion_adapter.detect_emotion(text)
        # Pour texte vide, on attend une erreur
        if text == "":
            assert result["success"] is False
        else:
            # Accepter soit l'émotion attendue, soit neutral pour cas ambigus
            assert result["emotion"] in [expected_emotion, "neutral"]

# Tests d'Intégration avec Orchestrateur
@pytest.mark.asyncio
async def test_orchestrator_integration(orchestrator):
    """Test intégration complète avec l'orchestrateur"""

    # Capturer les events publiés
    published_events = []

    async def capture_event(topic, payload):
        published_events.append((topic, payload))

    # Monkey-patch le publish
    with patch.object(orchestrator.neural_bus, 'publish', side_effect=capture_event):
        with patch.object(orchestrator.neural_bus, 'publish_emotion_ml_detection', side_effect=capture_event):

            await orchestrator.process_user_input("Je suis très heureux aujourd'hui!")

            # Vérifier qu'un event a été publié
            assert len(published_events) > 0

            # Vérifier le contenu
            for topic, payload in published_events:
                if "emotion" in topic:
                    if isinstance(payload, dict):
                        if "version" in payload:
                            # Event versionné
                            assert payload["version"] == "1.0"
                            assert payload["data"]["primary"] == "joy"

@pytest.mark.asyncio
async def test_orchestrator_fallback(orchestrator):
    """Test fallback de l'orchestrateur en cas d'échec ML"""

    # Forcer une erreur dans l'adapter
    with patch.object(orchestrator.emotion_detector, 'detect_emotion',
                     side_effect=Exception("ML failed")):

        # Doit utiliser le fallback sans planter
        try:
            await orchestrator.process_user_input("Test fallback")
            # Si on arrive ici, le fallback a fonctionné
            assert True
        except Exception as e:
            pytest.fail(f"Orchestrator should not crash: {e}")

# Test de Monitoring
@pytest.mark.asyncio
async def test_monitoring_logs(emotion_adapter, caplog):
    """Test que les logs de monitoring sont créés"""

    import logging
    caplog.set_level(logging.DEBUG)

    await emotion_adapter.detect_emotion("Test monitoring")

    # Vérifier les logs structurés
    assert any("Emotion detected" in record.message for record in caplog.records)

    # Vérifier les stats
    stats = emotion_adapter.get_stats()
    assert stats["total_predictions"] > 0
    assert stats["avg_latency_ms"] > 0

# Test d'API (si FastAPI configuré)
@pytest.mark.asyncio
async def test_api_endpoints():
    """Test endpoints API (nécessite FastAPI test client)"""

    try:
        from fastapi.testclient import TestClient
        from jeffrey.interfaces.bridge.api import app

        client = TestClient(app)

        # Test detect endpoint
        response = client.post(
            "/api/v1/emotion/detect",
            json={"text": "Test API emotion"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "emotion" in data

        # Test stats endpoint
        response = client.get("/api/v1/emotion/stats")
        assert response.status_code == 200

        # Test health endpoint
        response = client.get("/api/v1/emotion/health")
        assert response.status_code == 200
        assert response.json()["status"] in ["healthy", "degraded"]

    except ImportError:
        pytest.skip("FastAPI not available for API tests")

# Benchmark final
@pytest.mark.asyncio
async def test_final_benchmark(emotion_adapter):
    """Benchmark final avec rapport détaillé"""

    print("\n" + "="*60)
    print("🏁 BENCHMARK FINAL - EMOTION ML ADAPTER")
    print("="*60)

    # Warmup
    for _ in range(10):
        await emotion_adapter.detect_emotion("warmup")

    # Test différentes longueurs de texte
    text_lengths = [10, 50, 100, 500, 1000]

    for length in text_lengths:
        text = "Test " * (length // 5)
        latencies = []

        for _ in range(50):
            start = time.perf_counter()
            result = await emotion_adapter.detect_emotion(text)
            latencies.append((time.perf_counter() - start) * 1000)

        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)

        print(f"\nText length ~{length} chars:")
        print(f"  P50: {p50:.1f}ms")
        print(f"  P95: {p95:.1f}ms")

    # Stats finales
    stats = emotion_adapter.get_stats()
    print(f"\n📊 Final Stats:")
    print(f"  Total predictions: {stats['total_predictions']}")
    print(f"  ML predictions: {stats['ml_percentage']:.1f}%")
    print(f"  Fallback predictions: {stats['fallback_percentage']:.1f}%")
    print(f"  Failure rate: {stats['failure_rate']:.1f}%")
    print(f"  Avg latency: {stats['avg_latency_ms']:.1f}ms")
    print("="*60)

if __name__ == "__main__":
    # Pour exécution directe
    asyncio.run(test_ml_adapter_basic_emotions(EmotionMLAdapter()))
    asyncio.run(test_final_benchmark(EmotionMLAdapter()))
    print("\n✅ All tests completed!")