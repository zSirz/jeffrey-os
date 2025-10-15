#!/usr/bin/env python3
import json
import os
import shutil

INV = "artifacts/inventory_ultimate.json"
BM = "best_modules.json"

# Créer inventaire minimal si absent
if not os.path.exists(INV):
    os.makedirs("artifacts", exist_ok=True)
    with open(INV, "w") as f:
        json.dump({"bundle1_recommendations": {"modules": []}}, f)

# Lire les meilleurs modules
best = json.load(open(BM))
data = json.load(open(INV))
modules = data.setdefault("bundle1_recommendations", {}).setdefault("modules", [])


# Fonction pour mettre à jour ou ajouter
def upsert_module(region, name, path):
    found = False
    for m in modules:
        if m.get("brain_region") == region:
            m["name"] = name
            m["path"] = path
            m["type"] = "real"
            m["priority"] = 1
            found = True
            break
    if not found and path:
        modules.append({"name": name, "path": path, "brain_region": region, "type": "real", "priority": 1})


# Mettre à jour emotion et conscience
upsert_module("emotion", "emotion_engine", best.get("emotion", ""))
upsert_module("conscience", "conscience_engine", best.get("conscience", ""))

# Sauvegarder
shutil.copyfile(INV, INV + ".bak")
with open(INV, "w") as f:
    json.dump(data, f, indent=2)

print("✅ Emotion et conscience patchés")
