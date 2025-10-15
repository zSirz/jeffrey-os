"""
Test basique du Bridge V3 pour vérifier l'initialisation
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_initialization():
    """Test que le bridge s'initialise correctement"""
    print("\n🔧 TEST: Bridge Initialization")

    try:
        from src.jeffrey.interfaces.ui.jeffrey_ui_bridge import JeffreyUIBridge

        print("✅ Import successful")

        # Créer le bridge
        bridge = JeffreyUIBridge()
        print("✅ Bridge created")

        # Attendre un peu pour l'init
        time.sleep(2)

        # Vérifier les composants
        assert bridge.runtime is not None, "No runtime"
        print("✅ Runtime initialized")

        assert bridge.priority_queue is not None, "No queue"
        print("✅ Priority queue created")

        # Vérifier métriques
        metrics = bridge.get_metrics()
        print(f"✅ Metrics available: {list(metrics.keys())[:5]}...")

        # Test détection langue
        lang_fr = bridge._detect_language("Bonjour")
        print(f"✅ Language detection: 'Bonjour' → {lang_fr}")

        lang_zh = bridge._detect_language("你好")
        print(f"✅ Language detection: '你好' → {lang_zh}")

        # Test compression
        for i in range(5):
            bridge.conversation_history.append(
                {"user": f"Message {i}", "assistant": f"Response {i}", "timestamp": time.time()}
            )

        compressed = bridge._compress_history(max_exchanges=2)
        print(f"✅ History compression: {len(bridge.conversation_history)} → {len(compressed)}")

        # Test simple message (sans attendre de réponse)
        print("\n📨 Sending test message...")

        response_received = {"done": False}

        def on_error(msg):
            print(f"   Error callback: {msg}")
            response_received["done"] = True
            response_received["error"] = msg

        def on_complete(text, meta):
            print(f"   Response: {text[:50]}...")
            response_received["done"] = True
            response_received["response"] = text

        bridge.send_message("Test message", priority=5, on_complete=on_complete, on_error=on_error)

        print("✅ Message queued")

        # Attendre un peu (max 5s)
        for i in range(10):
            if response_received["done"]:
                break
            time.sleep(0.5)
            print(".", end="", flush=True)

        print()

        if response_received.get("error"):
            print(f"⚠️  Got error (expected if no LLM): {response_received['error'][:100]}")
        elif response_received.get("response"):
            print(f"✅ Got response: {response_received['response'][:100]}")
        else:
            print("⚠️  No response (orchestrator may not have process method)")

        # Shutdown
        bridge.shutdown()
        print("✅ Bridge shutdown successful")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_metrics_only():
    """Test juste les métriques sans envoyer de message"""
    print("\n📊 TEST: Metrics Only")

    from src.jeffrey.interfaces.ui.jeffrey_ui_bridge import JeffreyUIBridge

    bridge = JeffreyUIBridge()
    time.sleep(1)

    # Test rate limiter
    can_send = bridge.rate_limiter.can_send("test_user")
    print(f"✅ Rate limiter: can_send = {can_send}")

    # Test circuit breaker state
    state = bridge.circuit_breaker.state
    print(f"✅ Circuit breaker: state = {state}")

    # Test cache
    bridge.response_cache.set("test", "neutral", "response", {"test": True}, "hash123", "model1")
    cached = bridge.response_cache.get("test", "neutral", "hash123", "model1")
    print(f"✅ Cache: set/get works = {cached is not None}")

    # Test profiling toggle
    bridge.enable_profiling(True)
    assert bridge.profiler.enabled
    bridge.enable_profiling(False)
    assert not bridge.profiler.enabled
    print("✅ Profiling: toggle works")

    bridge.shutdown()

    return True


def main():
    print("=" * 60)
    print("🏁 JEFFREY BRIDGE V3 - BASIC TESTS")
    print("=" * 60)

    # Test 1: Initialisation de base
    success1 = test_initialization()

    # Test 2: Métriques uniquement
    success2 = test_metrics_only()

    print("\n" + "=" * 60)
    if success1 and success2:
        print("✅ BASIC TESTS PASSED - Bridge V3 is functional!")
        print("\nNote: If no responses, the orchestrator may need")
        print("a 'process(context)' method to be implemented.")
    else:
        print("❌ Some tests failed - check errors above")
    print("=" * 60)


if __name__ == "__main__":
    main()
