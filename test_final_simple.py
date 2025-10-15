#!/usr/bin/env python3
"""
Test simplifié du système Jeffrey Bridge V3
"""

import asyncio
import sys

sys.path.insert(0, ".")


async def main():
    print("🚀 JEFFREY BRIDGE V3 - SYSTEM TEST")
    print("=" * 50)

    # 1. Import et initialisation
    from src.jeffrey.interfaces.ui.jeffrey_ui_bridge import JeffreyUIBridge
    from src.jeffrey.runtime import get_runtime

    runtime = get_runtime()
    bridge = JeffreyUIBridge()

    await asyncio.sleep(2)  # Laisser le système s'initialiser

    # 2. Test conversation réelle
    print("\nTesting real conversation...")

    result = {"done": False, "response": None}

    def on_complete(text, metadata):
        result["done"] = True
        result["response"] = text
        result["latency"] = metadata.get("latency_ms", 0)

    def on_error(error):
        result["done"] = True
        result["error"] = error

    # Envoyer message
    bridge.send_message(
        text="Quelle est la capitale de la France?",
        emotion="curious",
        on_complete=on_complete,
        on_error=on_error,
    )

    # Attendre réponse
    for i in range(60):
        if result["done"]:
            break
        await asyncio.sleep(0.5)
        if i % 4 == 0:
            print(".", end="", flush=True)

    print()

    if result.get("response"):
        print(f"✅ Got response: {result['response'][:100]}...")
        print(f"   Latency: {result.get('latency', 0):.0f}ms")
    elif result.get("error"):
        print(f"❌ Error: {result['error']}")
    else:
        print("⚠️ No response received")

    # 3. Vérifier métriques
    metrics = bridge.get_metrics()
    print("\n📊 Metrics:")
    print(f"   Total requests: {metrics['total_requests']}")
    print(f"   Cache hits: {metrics['cache_hits']}")
    print(f"   Circuit state: {metrics['circuit_state']}")
    print(f"   Healthy: {metrics['connection_healthy']}")

    # 4. Test cache (même message)
    print("\nTesting cache...")

    result2 = {"done": False}

    def on_complete2(text, metadata):
        result2["done"] = True
        result2["from_cache"] = metadata.get("from_cache", False)

    bridge.send_message(text="Quelle est la capitale de la France?", emotion="curious", on_complete=on_complete2)

    for i in range(10):
        if result2["done"]:
            break
        await asyncio.sleep(0.1)

    if result2.get("from_cache"):
        print("✅ Cache hit!")
    else:
        print("⚠️ Cache miss (expected on first run)")

    # Cleanup
    bridge.shutdown()

    print("\n" + "=" * 50)
    print("✅ System test complete!")


if __name__ == "__main__":
    asyncio.run(main())
