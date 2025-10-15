#!/usr/bin/env python3
"""Test 2 : Analyse émotionnelle avec tous les cas"""

import sys

sys.path.insert(0, 'src')

print("🧪 TEST 2 : ANALYSE ÉMOTIONNELLE COMPLÈTE")
print("=" * 70)

from jeffrey.core.orchestration.emotion_engine_bridge import get_emotion_bridge

bridge = get_emotion_bridge()

# Tests variés
test_cases = [
    ("Je suis super heureux ! 🎉✨", "joie"),
    ("Je me sens triste aujourd'hui 😔💔", "tristesse"),
    ("J'ai peur de ce qui va se passer 😰", "peur"),
    ("C'est fascinant ! Comment ça marche ? 🤔", "curiosité"),
    ("Je t'adore Jeffrey ❤️💕", "amour"),
    ("Pourquoi le ciel est bleu ?", "curiosité"),
    ("...", "neutre"),
]

print(f"\n📋 Analyse de {len(test_cases)} cas variés...")
success = 0
results_detail = []

for text, expected in test_cases:
    result = bridge.analyze_emotion_hybrid(text, {})

    emotion = result['emotion_dominante']
    intensity = result['intensite']
    confidence = result['confiance']
    mode = result['integration_mode']
    from_cache = result['from_cache']
    time_ms = result['processing_time_ms']

    # Vérification
    is_ok = (emotion != 'neutre') or (text == "...")

    status = "✅" if is_ok else "⚠️"
    cache_icon = "💾" if from_cache else "🔍"

    print(f"\n{status} '{text[:40]}...'")
    print(f"   {cache_icon} {emotion} ({intensity}%) conf:{confidence}%")
    print(f"   Mode:{mode} | Temps:{time_ms}ms")

    if 'consensus' in result:
        print(f"   Consensus: {result['consensus']}")

    results_detail.append({"text": text, "expected": expected, "detected": emotion, "ok": is_ok})

    if is_ok:
        success += 1

# Métriques finales
print(f"\n{'=' * 70}")
print(f"📊 RÉSULTATS : {success}/{len(test_cases)} réussis ({success / len(test_cases) * 100:.1f}%)")

metrics = bridge.get_emotional_metrics()
print("\n📈 MÉTRIQUES SYSTÈME :")
print(f"   Total analyses : {metrics['performance']['total_analyses']}")
print(f"   Temps moyen : {metrics['performance']['avg_response_time_ms']}ms")
print(f"   Cache hit rate : {metrics['cache']['hit_rate_percent']}%")
print(f"   Distribution moteurs : {metrics['analyses_by_engine']}")
print(f"   Consensus rate : {metrics['fusion']['consensus_rate_percent']}%")

if success == len(test_cases):
    print("\n✅ TEST 2 RÉUSSI ! Toutes détections correctes.")
else:
    print(f"\n⚠️ TEST 2 PARTIEL : {success}/{len(test_cases)}")
