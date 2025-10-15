#!/usr/bin/env python3
"""Évaluation rapide EmotionDetectorV3"""

import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml

# Setup
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from jeffrey.nlp.emotion_detector_v3 import EmotionDetectorV3


def load_sample_scenarios(max_files=5):
    """Charge un échantillon de scénarios"""
    conv_dir = ROOT / "tests" / "convos"
    scenarios = []

    yaml_files = list(conv_dir.glob("*.yaml"))[:max_files]

    for yaml_file in yaml_files:
        with open(yaml_file, encoding='utf-8') as f:
            data = yaml.safe_load(f)
            scenarios.append(data)

    return scenarios


def extract_examples(scenarios):
    """Extrait les exemples pour évaluation"""
    examples = []

    for scenario in scenarios:
        if 'conversation' not in scenario:
            continue

        for i, turn in enumerate(scenario['conversation']):
            # Prendre les messages utilisateur uniquement avec expected_emotion
            if turn.get('role') == 'user':
                text = turn.get('content', '').strip()
                expected_emotion = turn.get('expected_emotion', None)

                if text and expected_emotion:
                    examples.append(
                        {'text': text, 'expected': expected_emotion, 'scenario': scenario['metadata']['scenario_id']}
                    )

    return examples


def evaluate():
    print("🧪 ÉVALUATION RAPIDE - EmotionDetectorV3")
    print("=" * 50)

    # Charger échantillon
    scenarios = load_sample_scenarios(5)
    examples = extract_examples(scenarios)

    print(f"📊 {len(examples)} exemples dans {len(scenarios)} scénarios")
    print()

    # Initialiser détecteur
    detector = EmotionDetectorV3()

    # Évaluer
    correct = 0
    total = len(examples)
    latencies = []

    print("🔍 PRÉDICTIONS:")
    for i, ex in enumerate(examples[:10]):  # Limite aux 10 premiers
        start = time.time()
        result = detector.detect(ex['text'])
        latency = (time.time() - start) * 1000
        latencies.append(latency)

        is_correct = result.primary == ex['expected']
        if is_correct:
            correct += 1

        status = "✅" if is_correct else "❌"
        print(f"{status} {ex['expected']:12s} → {result.primary:12s} | {ex['text'][:40]}...")

    # Métriques
    accuracy = correct / min(10, total) * 100
    avg_latency = sum(latencies) / len(latencies)

    print()
    print("📈 RÉSULTATS:")
    print(f"   Accuracy : {accuracy:.1f}% ({correct}/{min(10, total)})")
    print(f"   Latence  : {avg_latency:.1f}ms moyenne")
    print()

    # Bonus : Distribution des émotions prédites
    predicted_emotions = defaultdict(int)
    for ex in examples[:20]:
        result = detector.detect(ex['text'])
        predicted_emotions[result.primary] += 1

    print("🎯 DISTRIBUTION PRÉDICTIONS:")
    for emotion, count in sorted(predicted_emotions.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {emotion:12s} : {count} fois")


if __name__ == "__main__":
    evaluate()
