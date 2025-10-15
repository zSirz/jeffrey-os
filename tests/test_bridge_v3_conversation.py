#!/usr/bin/env python3
"""
Test de conversation réelle avec le Bridge V3 et Ollama
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.jeffrey.interfaces.ui.jeffrey_ui_bridge import JeffreyUIBridge


def test_real_conversation():
    """Test une vraie conversation avec Ollama"""
    print("\n🗣️ TEST DE CONVERSATION RÉELLE AVEC JEFFREY")
    print("=" * 60)

    bridge = JeffreyUIBridge()
    time.sleep(2)  # Laisser le warmup se faire

    # Questions de test
    test_messages = [
        ("Bonjour Jeffrey, comment vas-tu ?", "friendly"),
        ("Quelle est la capitale de la France ?", "curious"),
        ("Raconte-moi une blague courte", "playful"),
        ("Merci Jeffrey, au revoir !", "grateful"),
    ]

    for i, (message, emotion) in enumerate(test_messages, 1):
        print(f"\n--- Message {i} ---")
        print(f"👤 User: {message}")
        print(f"   Émotion: {emotion}")

        result = {"done": False, "response": None, "error": None}
        start_time = time.time()

        def on_start():
            print("   ⏳ Jeffrey réfléchit...")

        def on_complete(text, metadata):
            result["done"] = True
            result["response"] = text
            result["metadata"] = metadata

        def on_error(error):
            result["done"] = True
            result["error"] = error

        # Envoyer le message
        bridge.send_message(
            message,
            emotion=emotion,
            priority=5,
            on_start=on_start,
            on_complete=on_complete,
            on_error=on_error,
        )

        # Attendre la réponse (max 30s)
        timeout = 30
        while not result["done"] and time.time() - start_time < timeout:
            time.sleep(0.5)
            print(".", end="", flush=True)

        print()  # Nouvelle ligne

        if result["error"]:
            print(f"   ❌ Erreur: {result['error']}")
        elif result["response"]:
            # Afficher la réponse
            response_text = result["response"][:300]  # Limiter l'affichage
            if len(result["response"]) > 300:
                response_text += "..."

            print(f"🤖 Jeffrey: {response_text}")

            # Afficher les métriques
            metadata = result.get("metadata", {})
            latency = metadata.get("latency_ms", 0)
            from_cache = metadata.get("from_cache", False)

            if from_cache:
                print(f"   ⚡ Réponse du cache en {latency:.1f}ms")
            else:
                print(f"   ⏱️ Réponse générée en {latency:.1f}ms")

            # Vérifier que c'est une vraie réponse (pas un stub)
            if len(result["response"]) > 50 and "Jeffrey" not in result["response"][:20]:
                print("   ✅ Vraie réponse générée")
            elif from_cache:
                print("   ✅ Réponse du cache")
            else:
                print("   ⚠️ Réponse courte ou stub")
        else:
            print("   ⚠️ Pas de réponse reçue")

        # Pause entre les messages
        time.sleep(1)

    # Afficher les métriques finales
    print("\n" + "=" * 60)
    print("📊 MÉTRIQUES FINALES")
    print("=" * 60)

    metrics = bridge.get_metrics()

    print(
        f"""
    Total requêtes:      {metrics['total_requests']}
    Cache hits:          {metrics['cache_hits']} ({metrics['cache_hit_rate']:.1f}%)
    Streaming:           {metrics['streaming_requests']} ({metrics['streaming_rate']:.1f}%)
    Circuit breaks:      {metrics['circuit_breaks']} ({metrics['circuit_break_rate']:.1f}%)
    Rate limits:         {metrics['rate_limits']} ({metrics['rate_limit_rate']:.1f}%)
    Latence moyenne:     {metrics['avg_latency_ms']:.1f}ms
    Queue size:          {metrics['queue_size']}
    Circuit state:       {metrics['circuit_state']}
    Connection healthy:  {metrics['connection_healthy']}
    """
    )

    # Test du cache : renvoyer le même message
    print("\n🔄 TEST DU CACHE")
    print("=" * 60)
    print("Renvoi du premier message pour tester le cache...")

    result2 = {"done": False}

    def on_complete2(text, metadata):
        result2["done"] = True
        result2["response"] = text
        result2["from_cache"] = metadata.get("from_cache", False)
        result2["latency"] = metadata.get("latency_ms", 0)

    bridge.send_message("Bonjour Jeffrey, comment vas-tu ?", emotion="friendly", on_complete=on_complete2)

    # Attendre
    for _ in range(10):
        if result2["done"]:
            break
        time.sleep(0.1)

    if result2.get("from_cache"):
        print(f"✅ Cache hit! Réponse instantanée en {result2['latency']:.1f}ms")
    else:
        print(f"⚠️ Cache miss, nouvelle génération en {result2.get('latency', 0):.1f}ms")

    # Shutdown propre
    bridge.shutdown()
    print("\n✅ Test terminé avec succès!")


if __name__ == "__main__":
    test_real_conversation()
