#!/usr/bin/env python3
"""
Patch chirurgical de l'inventaire.
"""

import json
import sys
from pathlib import Path

INV = Path("artifacts/inventory_ultimate.json")
BEST = Path("best_modules.json")
DISCOVERED = Path("discovered_regions.json")

if not BEST.exists():
    print("❌ best_modules.json introuvable")
    sys.exit(1)

if not DISCOVERED.exists():
    print("❌ discovered_regions.json introuvable")
    sys.exit(1)

best = json.loads(BEST.read_text())
discovered = json.loads(DISCOVERED.read_text())

if INV.exists():
    inventory = json.loads(INV.read_text())
else:
    inventory = {"bundle1_recommendations": {"modules": []}}

all_modules = {**best, **discovered}

new_modules = []
for region, path in all_modules.items():
    if not path:
        print(f"⚠️  {region}: Aucun module trouvé", file=sys.stderr)
        continue

    if "/simple_modules/" in path or "/stubs/" in path:
        print(f"❌ {region}: Module interdit ignoré - {path}", file=sys.stderr)
        continue

    if not Path(path).exists():
        print(f"⚠️  {region}: Fichier introuvable - {path}", file=sys.stderr)
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

    print(f"✅ {region}: {module_name}", file=sys.stderr)

# ÉCHEC SI < 8
if len(new_modules) < 8:
    print(f"\n❌ ÉCHEC : Seulement {len(new_modules)}/8 régions trouvées", file=sys.stderr)
    print("\nRégions manquantes:")
    all_regions = {
        "perception",
        "memory",
        "emotion",
        "conscience",
        "executive",
        "motor",
        "language",
        "integration",
    }
    found_regions = {m["brain_region"] for m in new_modules}
    missing = all_regions - found_regions
    for r in missing:
        print(f"   • {r}")
    print("\n⛔ TU DOIS DÉVELOPPER LES RÉGIONS MANQUANTES.")
    print("\nOptions:")
    print("1. Refactor les dépendances cassées")
    print("2. Développer de vrais modules (pas de simples!)")
    print("3. Chercher dans d'autres emplacements")
    sys.exit(1)

inventory["bundle1_recommendations"]["modules"] = new_modules

INV.parent.mkdir(parents=True, exist_ok=True)
INV.write_text(json.dumps(inventory, indent=2))

print(f"\n✅ Inventaire mis à jour: {len(new_modules)}/8 régions", file=sys.stderr)
