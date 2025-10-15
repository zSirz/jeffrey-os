#!/usr/bin/env python3
"""
Patch l'inventaire avec les modules manquants détectés
"""

import json
import os

INV = "artifacts/inventory_ultimate.json"
MISSING = "missing_regions.json"

# Charger les données
if not os.path.exists(MISSING):
    print("❌ Lancez d'abord find_missing_regions.py")
    exit(1)

with open(MISSING) as f:
    missing = json.load(f)

with open(INV) as f:
    inventory = json.load(f)

# Ajouter les modules manquants
modules = inventory["bundle1_recommendations"]["modules"]
existing_regions = {m["brain_region"] for m in modules}

print("📋 INVENTORY CURRENT STATE:")
for m in modules:
    print(f"   {m['brain_region']:15} → {os.path.basename(m['path'])}")

print("\n🔍 MISSING REGIONS TO ADD:")
added_count = 0

for region, info in missing.items():
    print(f"\n🎯 Processing {region}:")
    print(f"   Best module: {info['path']}")
    print(f"   Score: {info['score']}")
    print(f"   Classes: {', '.join(info['classes'][:3])}")

    if region not in existing_regions:
        # Extraire le nom du module
        path = info["path"]
        module_name = os.path.basename(path).replace(".py", "")

        # Ajouter avec plus d'info
        new_module = {
            "name": module_name,
            "path": path,
            "brain_region": region,
            "type": "real",
            "priority": 1,
            "score": info["score"],
            "classes": info["classes"][:3],
            "key_methods": [m for m in info["methods"] if "✅" in str(m)][:3],
        }

        modules.append(new_module)
        added_count += 1
        print(f"   ✅ Added {region}: {module_name}")
    else:
        # Mettre à jour le module existant avec un meilleur si le score est plus élevé
        existing_module = next((m for m in modules if m["brain_region"] == region), None)
        if existing_module and info["score"] > existing_module.get("score", 0):
            existing_module.update(
                {
                    "name": os.path.basename(info["path"]).replace(".py", ""),
                    "path": info["path"],
                    "score": info["score"],
                    "classes": info["classes"][:3],
                    "key_methods": [m for m in info["methods"] if "✅" in str(m)][:3],
                }
            )
            print(f"   🔄 Updated {region} with better module: {info['path']}")
        else:
            print(f"   ⏭️  Kept existing {region} module")

# Vérifier qu'on a exactement les 8 régions requises
required_regions = [
    "perception",
    "memory",
    "emotion",
    "conscience",
    "executive",
    "motor",
    "language",
    "integration",
]
current_regions = {m["brain_region"] for m in modules}

print("\n📊 FINAL STATUS:")
print(f"   Required regions: {len(required_regions)}")
print(f"   Current regions: {len(current_regions)}")
print(f"   Added modules: {added_count}")

missing_still = set(required_regions) - current_regions
if missing_still:
    print(f"   ⚠️  Still missing: {missing_still}")
else:
    print("   ✅ All required regions present!")

# Afficher le résumé final
print("\n📋 FINAL MODULES LIST:")
for m in sorted(modules, key=lambda x: x["brain_region"]):
    score = m.get("score", "N/A")
    classes = m.get("classes", ["Unknown"])
    print(f"   {m['brain_region']:15} → {os.path.basename(m['path']):30} (score: {score})")

# Sauvegarder
with open(INV, "w") as f:
    json.dump(inventory, f, indent=2)

print(f"\n💾 Inventaire mis à jour: {len(modules)} modules totaux")

# Créer un backup de validation
backup_file = "inventory_backup_before_validation.json"
with open(backup_file, "w") as f:
    json.dump(inventory, f, indent=2)
print(f"📀 Backup created: {backup_file}")

print("\n✅ Ready for validation!")
