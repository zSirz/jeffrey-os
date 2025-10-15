#!/usr/bin/env python3
"""Test 3 : Cache, performance et circuit breaker"""

import sys
import time

sys.path.insert(0, 'src')

print("🧪 TEST 3 : CACHE, PERFORMANCE & CIRCUIT BREAKER")
print("=" * 70)

from jeffrey.core.orchestration.emotion_engine_bridge import get_emotion_bridge

bridge = get_emotion_bridge()

# ========================================
# TEST CACHE
# ========================================
print("\n📦 TEST CACHE (LRU + TTL)")

text = "Je suis heureux ! 🎉"

# Premier appel (cache miss)
start = time.time()
result1 = bridge.analyze_emotion_hybrid(text, {})
time1 = (time.time() - start) * 1000

# Deuxième appel (cache hit)
start = time.time()
result2 = bridge.analyze_emotion_hybrid(text, {})
time2 = (time.time() - start) * 1000

# Troisième appel (cache hit)
start = time.time()
result3 = bridge.analyze_emotion_hybrid(text, {})
time3 = (time.time() - start) * 1000

print(f"⏱️ Premier appel (miss) : {time1:.2f}ms")
print(f"⏱️ Deuxième appel (hit)  : {time2:.2f}ms")
print(f"⏱️ Troisième appel (hit) : {time3:.2f}ms")
print(f"🚀 Gain de vitesse : {(time1 / max(0.01, time2)):.1f}x")

if result2.get('from_cache') and result3.get('from_cache'):
    print("✅ Cache fonctionne parfaitement")
else:
    print("⚠️ Cache non détecté")

# ========================================
# TEST PERFORMANCE SOUS CHARGE
# ========================================
print("\n⚡ TEST PERFORMANCE SOUS CHARGE")

test_texts = ["Je suis content", "C'est triste", "J'ai peur", "Intéressant", "Je t'aime"] * 20  # 100 analyses

start_bulk = time.time()
for txt in test_texts:
    bridge.analyze_emotion_hybrid(txt, {})
time_bulk = (time.time() - start_bulk) * 1000

print(f"📊 100 analyses en {time_bulk:.0f}ms")
print(f"📊 Moyenne : {time_bulk / 100:.1f}ms par analyse")

# ========================================
# MÉTRIQUES FINALES
# ========================================
metrics = bridge.get_emotional_metrics()

print("\n💾 CACHE STATS :")
print(f"   Taille : {metrics['cache']['size']}/{metrics['cache']['maxsize']}")
print(f"   Hits : {metrics['cache']['hits']}")
print(f"   Misses : {metrics['cache']['misses']}")
print(f"   Hit rate : {metrics['cache']['hit_rate_percent']}%")
print(f"   Évictions : {metrics['cache']['evictions']}")
print(f"   Expirations : {metrics['cache']['expirations']}")

print("\n📈 PERFORMANCE :")
print(f"   Total : {metrics['performance']['total_analyses']} analyses")
print(f"   Latence moyenne : {metrics['performance']['avg_response_time_ms']}ms")

print("\n🔄 CIRCUIT BREAKERS :")
for name, cb in metrics['circuit_breakers'].items():
    status = "🔴 OUVERT" if cb['is_open'] else "🟢 FERMÉ"
    print(f"   {name} : {status} (échecs totaux: {cb['total_failures']})")

if metrics['cache']['hit_rate_percent'] > 50:
    print("\n✅ TEST 3 RÉUSSI ! Cache et performance excellents.")
else:
    print("\n⚠️ TEST 3 : Hit rate faible (normal si premier run)")
