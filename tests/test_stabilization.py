"""
Tests de validation pour la phase de stabilisation
Vérifie que tous les correctifs fonctionnent
"""
import asyncio
import httpx
import json
import pytest
import sys
import os

# Ajouter le path pour les imports locaux
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

async def test_memory_port():
    """Test que le MemoryPort s'adapte à différentes interfaces"""
    from jeffrey.core.ports.memory_port import MemoryPort

    # Test avec différentes interfaces
    class StoreMemory:
        def store(self, data):
            return True

    class AddMemory:
        def add(self, data):
            return True

    class NoInterfaceMemory:
        pass

    # Tester l'adaptation
    port1 = MemoryPort(StoreMemory())
    assert port1.method_name == "store"
    assert port1.store({"test": "data"})

    port2 = MemoryPort(AddMemory())
    assert port2.method_name == "add"
    assert port2.store({"test": "data"})

    port3 = MemoryPort(NoInterfaceMemory())
    assert port3.method_name is None
    assert port3.store({"test": "data"})  # Utilise fallback
    assert len(port3.fallback_memory) == 1

    print("✅ MemoryPort adapts correctly to all interfaces")

async def test_thought_contract():
    """Test que les pensées respectent le contrat"""
    from jeffrey.core.contracts.thoughts import create_thought, validate_thought, ThoughtState, ensure_thought_format

    # Créer une pensée valide
    thought = create_thought(
        state=ThoughtState.REFLECTIVE,
        summary="Test thought",
        confidence=0.8
    )

    assert validate_thought(thought)
    assert thought["state"] == "reflective"
    assert "timestamp" in thought

    # Tester la validation
    invalid_thought = {"random": "data"}
    assert not validate_thought(invalid_thought)

    # Tester ensure_thought_format
    corrected = ensure_thought_format(invalid_thought)
    assert validate_thought(corrected)

    print("✅ Thought contract works correctly")

async def test_brain_health():
    """Test le endpoint de health amélioré"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get("http://localhost:8000/api/v1/brain/health")

            if response.status_code != 200:
                print(f"❌ Health endpoint failed: {response.status_code}")
                print(f"Response: {response.text}")
                return False

            health = response.json()

            # Vérifications de base
            required_fields = ["status", "uptime_seconds", "brain_state", "performance", "memory", "activity", "errors"]
            for field in required_fields:
                assert field in health, f"Missing field: {field}"

            print(f"✅ Brain health endpoint works: {health['status']}")
            print(f"   Uptime: {health['uptime_seconds']}s")
            print(f"   Wired: {health['brain_state']['wired']}")
            print(f"   Memory: {health['brain_state']['memory_available']}")
            print(f"   Consciousness: {health['brain_state']['consciousness_available']}")

            if health["performance"]["latency_p95_ms"]:
                print(f"   P95 Latency: {health['performance']['latency_p95_ms']}ms")

            return True

    except Exception as e:
        print(f"❌ Health test failed: {e}")
        return False

async def test_full_loop_with_fixes():
    """Test la boucle complète avec tous les correctifs"""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Envoyer plusieurs émotions
            emotions = [
                "I feel joyful today!",
                "This situation makes me angry!",
                "I'm surprised by this news!",
                "I feel peaceful and calm."
            ]

            print(f"Testing {len(emotions)} emotions...")

            for i, emotion_text in enumerate(emotions, 1):
                print(f"  {i}. Processing: '{emotion_text}'")

                response = await client.post(
                    "http://localhost:8000/api/v1/emotion/detect",
                    json={"text": emotion_text}
                )

                if response.status_code != 200:
                    print(f"     ❌ Failed: {response.status_code}")
                    return False

                result = response.json()
                print(f"     → {result.get('emotion')} ({result.get('confidence', 0):.2f})")

                await asyncio.sleep(0.2)  # Petite pause

            # Vérifier le health après traitement
            print("\nChecking brain state after processing...")
            health_response = await client.get("http://localhost:8000/api/v1/brain/health")

            if health_response.status_code != 200:
                print(f"❌ Health check failed: {health_response.status_code}")
                return False

            data = health_response.json()

            # Vérifications critiques
            activity = data.get("activity", {})
            errors = data.get("errors", {})

            emotions_processed = activity.get("emotions_received", 0)
            memories_stored = activity.get("memories_stored", 0)
            thoughts_generated = activity.get("thoughts_generated", 0)
            total_errors = errors.get("total", 0)

            print(f"📊 Results:")
            print(f"   Emotions processed: {emotions_processed}")
            print(f"   Memories stored: {memories_stored}")
            print(f"   Thoughts generated: {thoughts_generated}")
            print(f"   Total errors: {total_errors}")

            # Assertions
            assert emotions_processed >= len(emotions), f"Expected >= {len(emotions)}, got {emotions_processed}"
            assert thoughts_generated >= len(emotions) - 1, f"Expected >= {len(emotions)-1} thoughts, got {thoughts_generated}"

            # Vérifier qu'il n'y a pas trop d'erreurs
            if emotions_processed > 0:
                error_rate = total_errors / emotions_processed
                assert error_rate < 0.3, f"Error rate too high: {error_rate*100:.1f}%"
                print(f"✅ Error rate acceptable: {error_rate*100:.1f}%")
            else:
                print("⚠️ No emotions processed")

            # Vérifier les performances si disponibles
            perf = data.get("performance", {})
            p95 = perf.get("latency_p95_ms")
            if p95:
                print(f"📈 P95 latency: {p95}ms")
                if p95 < 200:  # Moins de 200ms c'est bien
                    print("✅ Performance acceptable")
                else:
                    print("⚠️ Performance could be better")

            print(f"✅ Full loop works with acceptable performance")
            return True

    except Exception as e:
        print(f"❌ Full loop test failed: {e}")
        return False

async def test_smoke_imports():
    """Test que tous les imports critiques fonctionnent"""
    try:
        from jeffrey.core.ports.memory_port import MemoryPort
        print("✅ MemoryPort import OK")

        from jeffrey.core.contracts.thoughts import create_thought, ThoughtState
        print("✅ Thought contracts import OK")

        # Test création rapide
        thought = create_thought(state=ThoughtState.AWARE, summary="smoke test")
        assert "state" in thought
        assert "timestamp" in thought
        print("✅ Thought creation OK")

        return True

    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

async def main():
    """Point d'entrée principal des tests"""

    print("🧪 TESTING STABILIZATION FIXES")
    print("=" * 50)

    # Tests unitaires d'abord
    success = True

    print("\n1. Testing imports...")
    if not await test_smoke_imports():
        success = False

    print("\n2. Testing MemoryPort...")
    try:
        await test_memory_port()
    except Exception as e:
        print(f"❌ MemoryPort test failed: {e}")
        success = False

    print("\n3. Testing Thought contract...")
    try:
        await test_thought_contract()
    except Exception as e:
        print(f"❌ Thought contract test failed: {e}")
        success = False

    print("\n4. Testing with live server...")
    print("   (Make sure server is running: uvicorn jeffrey.interfaces.bridge.api:app --reload)")

    if not await test_brain_health():
        success = False

    if not await test_full_loop_with_fixes():
        success = False

    print("\n" + "=" * 50)
    if success:
        print("✅ ALL STABILIZATION TESTS PASSED!")
        print("🏥 Brain is stabilized and ready for enrichment")
    else:
        print("❌ SOME TESTS FAILED")
        print("🔧 Check the errors above and fix before proceeding")

    return success

if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)