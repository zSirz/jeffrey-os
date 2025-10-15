#!/usr/bin/env python3
"""
Corrige les 3 régions qui échouent en trouvant de meilleurs modules.
"""

import json
import sys
from pathlib import Path

# Modules de remplacement pour les régions qui échouent
REPLACEMENTS = {
    "conscience": "src/jeffrey/core/consciousness/conscience_engine.py",
    "memory": "src/jeffrey/core/memory/unified_memory.py",
    "language": "src/jeffrey/core/llm/autonomous_language_system.py",
}


def fix_failed_regions():
    """Remplace les modules problématiques par des modules fonctionnels"""
    inv_path = Path("artifacts/inventory_ultimate.json")

    if not inv_path.exists():
        print("❌ Inventaire introuvable")
        sys.exit(1)

    inventory = json.loads(inv_path.read_text())
    modules = inventory["bundle1_recommendations"]["modules"]

    print("🔧 Correction des régions problématiques...")

    # Créer un mapping des régions vers les modules
    region_to_module = {}
    for module in modules:
        region = module["brain_region"]
        region_to_module[region] = module

    # Remplacer les modules problématiques
    for region, new_path in REPLACEMENTS.items():
        if region in region_to_module:
            old_path = region_to_module[region]["path"]
            print(f"   {region}: {old_path} → {new_path}")

            # Mettre à jour le module
            region_to_module[region]["path"] = new_path
            region_to_module[region]["name"] = Path(new_path).stem

    # Sauvegarder l'inventaire corrigé
    inv_path.write_text(json.dumps(inventory, indent=2))
    print("\n✅ Inventaire corrigé et sauvegardé")


if __name__ == "__main__":
    fix_failed_regions()
