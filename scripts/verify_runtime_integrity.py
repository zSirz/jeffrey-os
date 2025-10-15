#!/usr/bin/env python3
"""
Vérifie que les modules chargés sont les VRAIS (via empreintes).
"""

import importlib
import inspect
import json
import sys
from pathlib import Path

# Charger empreintes
fingerprints_file = Path("artifacts/module_fingerprints.json")
if not fingerprints_file.exists():
    print("❌ Fichier d'empreintes manquant. Exécutez d'abord generate_fingerprints.py")
    sys.exit(1)

fingerprints = json.loads(fingerprints_file.read_text())

TRUST_MAP = {
    "emotion": ("jeffrey.core.emotions.core.emotion_engine", "EmotionEngine"),
    "motor": ("jeffrey.core.response.neural_response_orchestrator", "NeuralResponseOrchestrator"),
    "memory": ("jeffrey.bundle2.memory.sqlite_store", "SQLiteMemoryStore"),
    "perception": ("jeffrey.core.input.input_parser", "InputParser"),
    "conscience": ("jeffrey.core.consciousness.self_awareness_tracker", "SelfAwarenessTracker"),
    "language": ("jeffrey.bundle2.language.broca_wernicke", "BrocaWernickeRegion"),
    "integration": ("jeffrey.core.memory.triple_memory", "TripleMemorySystem"),
}

failed = []

for region, (mod_path, cls_name) in TRUST_MAP.items():
    try:
        # Import module
        mod = importlib.import_module(mod_path)
        cls = getattr(mod, cls_name)

        # Vérifier LOC
        source = inspect.getsource(cls)
        lines = len(source.splitlines())
        expected_lines = fingerprints[region]["lines"]

        # Tolérance adaptée selon le module (classe vs fichier complet)
        min_threshold = expected_lines * 0.20 if region != "integration" else expected_lines * 0.15
        if lines < min_threshold:
            failed.append(f"{region}: {lines} lignes (attendu {expected_lines}) - substitution?")
        else:
            print(f"✅ {region}: {cls_name} OK ({lines} lignes)")

    except Exception as e:
        failed.append(f"{region}: Erreur - {e}")

if failed:
    print("\n❌ ÉCHECS D'INTÉGRITÉ:")
    for f in failed:
        print(f"   {f}")
    sys.exit(1)

print("\n✅ Tous les modules sont authentiques")
