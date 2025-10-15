#!/usr/bin/env python3
"""
Correction finale : remplace les modules problématiques par Bundle2.
"""

import json
import sys
from pathlib import Path

# Meilleurs modules trouvés pour chaque région
FINAL_MODULES = {
    "emotion": "src/jeffrey/core/orchestration/emotion_engine_bridge.py",
    "conscience": "src/jeffrey/core/consciousness/self_awareness_tracker.py",
    "perception": "src/jeffrey/core/input/input_parser.py",
    "memory": "src/jeffrey/bundle2/memory/sqlite_store.py",
    "language": "src/jeffrey/bundle2/language/broca_wernicke.py",
    "executive": "src/jeffrey/core/orchestration/emotion_engine_bridge.py",
    "motor": "src/jeffrey/core/generation/response_generator.py",
    "integration": "src/jeffrey/core/memory/triple_memory.py",
}


def fix_final_regions():
    """Remplace avec les meilleurs modules pour chaque région"""
    inv_path = Path("artifacts/inventory_ultimate.json")

    if not inv_path.exists():
        print("❌ Inventaire introuvable")
        sys.exit(1)

    print("🔧 Correction finale des 8 régions...")

    # Créer nouvel inventaire propre
    new_modules = []
    for region, path in FINAL_MODULES.items():
        if not Path(path).exists():
            print(f"⚠️  {region}: Fichier introuvable - {path}")
            continue

        module_name = Path(path).stem
        new_modules.append(
            {
                "name": module_name,
                "path": path,
                "brain_region": region,
                "gfc": region,
                "type": "real",
                "priority": 1,
            }
        )
        print(f"✅ {region}: {module_name}")

    # Créer nouvel inventaire
    inventory = {
        "bundle1_recommendations": {"modules": new_modules},
        "configuration": {
            "parallel_processing": True,
            "cache_enabled": True,
            "max_latency_ms": 50,
            "profiling_enabled": True,
            "target_regions": 8,
        },
    }

    inv_path.write_text(json.dumps(inventory, indent=2))
    print(f"\n✅ Inventaire final créé: {len(new_modules)}/8 régions")


if __name__ == "__main__":
    fix_final_regions()
