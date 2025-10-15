#!/usr/bin/env python3
"""
Génère empreintes SHA256 + LOC des 8 vrais modules.
"""

import hashlib
import json
from pathlib import Path

REAL_MODULES = {
    "perception": "src/jeffrey/core/input/input_parser.py",
    "memory": "src/jeffrey/bundle2/memory/sqlite_store.py",
    "emotion": "src/jeffrey/core/emotions/core/emotion_engine.py",
    "conscience": "src/jeffrey/core/consciousness/self_awareness_tracker.py",
    "language": "src/jeffrey/bundle2/language/broca_wernicke.py",
    "executive": "src/jeffrey/core/orchestration/agi_orchestrator.py",
    "motor": "src/jeffrey/core/response/neural_response_orchestrator.py",
    "integration": "src/jeffrey/core/memory/triple_memory.py",
}

fingerprints = {}

for region, path in REAL_MODULES.items():
    file_path = Path(path)
    if not file_path.exists():
        print(f"⚠️ {region}: {path} introuvable")
        continue

    content = file_path.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    lines = len(file_path.read_text().splitlines())

    fingerprints[region] = {"path": path, "sha256": sha256, "lines": lines}
    print(f"✅ {region}: {lines} lignes, SHA256={sha256[:8]}...")

# Sauvegarder
output = Path("artifacts/module_fingerprints.json")
output.parent.mkdir(exist_ok=True)
output.write_text(json.dumps(fingerprints, indent=2))
print(f"\n✅ Empreintes sauvegardées: {output}")
