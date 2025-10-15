#!/usr/bin/env python3
"""Test 1 : Import et initialisation complète"""

import sys

sys.path.insert(0, 'src')

print("🧪 TEST 1 : IMPORT & INITIALISATION COMPLÈTE")
print("=" * 70)

try:
    from jeffrey.core.orchestration.emotion_engine_bridge import get_emotion_bridge

    print("✅ Import réussi")

    bridge = get_emotion_bridge()
    print("✅ Bridge créé")
    print(f"   Mode : {bridge.integration_mode.value}")
    print(f"   Initialisé : {bridge.initialized}")

    health = bridge.health_check()
    print(f"✅ Health check : {health['status']}")
    print(f"   Core : {health['engines']['core']}")
    print(f"   Advanced : {health['engines']['advanced']}")

    if health['circuit_breakers']:
        print(f"   Circuit breakers : {health['circuit_breakers']}")

    metrics = bridge.get_emotional_metrics()
    print("✅ Métriques disponibles")
    print(f"   Moteurs actifs : {metrics['engines_active']}")
    print(f"   Config : {metrics['config']}")

    print("\n✅ TEST 1 RÉUSSI ! Bridge opérationnel.")

except Exception as e:
    print(f"❌ ERREUR : {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
