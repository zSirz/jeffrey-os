#!/usr/bin/env python3
"""
JEFFREY OS - Analyse des Erreurs Sprint 1
==========================================

Script pour analyser les erreurs de détection émotionnelle
et identifier les patterns à améliorer.

PATCH GPT Point 9 : Boucle d'amélioration pilotée par erreurs
"""

import json
from collections import Counter
from pathlib import Path

# Charger le rapport
report_file = Path("test_results/sprint1_emotion_eval.json")

if not report_file.exists():
    print("❌ Rapport non trouvé. Lance d'abord les tests.")
    exit(1)

with open(report_file, encoding='utf-8') as f:
    report = json.load(f)

# Extraire les erreurs
samples = report.get("samples", [])
errors = [s for s in samples if s["gold"] != s["pred"]]

print("=" * 80)
print("📊 ANALYSE DES ERREURS - SPRINT 1")
print("=" * 80)
print()
print(f"Total erreurs : {len(errors)} / {len(samples)}")
print()

# Confusions les plus fréquentes
confusions = Counter()
for err in errors:
    pair = (err["gold"], err["pred"])
    confusions[pair] += 1

print("🔀 TOP 10 CONFUSIONS :")
for (gold, pred), count in confusions.most_common(10):
    print(f"   {gold:15s} → {pred:15s} : {count} fois")

print()

# Mots-clés des textes mal classés
print("🔍 TEXTES MAL CLASSÉS (échantillon) :")
for err in errors[:10]:
    print(f"\n   Gold: {err['gold']:15s} → Pred: {err['pred']:15s}")
    print(f"   Texte: {err['text'][:100]}...")

print()
print("=" * 80)
print("💡 RECOMMANDATIONS :")
print("   1. Ajouter des mots-clés pour les émotions confondues")
print("   2. Améliorer les patterns pour les expressions fréquentes")
print("   3. Relancer les tests après modifications")
print("=" * 80)
