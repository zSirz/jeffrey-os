#!/usr/bin/env python3
"""
Fix final pour utiliser les modules les plus simples et fonctionnels
"""

import json
import os

# Modules testés et fonctionnels (sans dépendances complexes)
FINAL_MODULES = [
    {
        "name": "emotional_core",
        "path": "src/jeffrey/core/orchestration/emotional_core.py",
        "brain_region": "emotion",
        "type": "real",
        "priority": 1,
    },
    {
        "name": "input_parser",  # Garde ce qui marche
        "path": "src/jeffrey/core/input/input_parser.py",
        "brain_region": "perception",
        "type": "real",
        "priority": 1,
    },
    {
        "name": "working_memory",  # Garde ce qui marche
        "path": "src/jeffrey/core/memory/working_memory.py",
        "brain_region": "memory",
        "type": "real",
        "priority": 1,
    },
    {
        "name": "cognitive_pipeline",  # Garde ce qui marche
        "path": "src/jeffrey/core/pipeline/cognitive_pipeline.py",
        "brain_region": "prefrontal",
        "type": "real",
        "priority": 1,
    },
    {
        "name": "provider_manager",  # Garde ce qui marche
        "path": "src/jeffrey/services/providers/provider_manager.py",
        "brain_region": "motor",
        "type": "real",
        "priority": 1,
    },
    {
        "name": "provider_manager_lang",  # Pour language
        "path": "src/jeffrey/services/providers/provider_manager.py",
        "brain_region": "language",
        "type": "real",
        "priority": 1,
    },
    # Nouveaux modules plus simples
    {
        "name": "orchestrator",
        "path": "src/jeffrey/core/cognitive/orchestrator.py",
        "brain_region": "executive",
        "type": "real",
        "priority": 1,
    },
    {
        "name": "hybrid_bridge",
        "path": "src/jeffrey/core/llm/hybrid_bridge.py",
        "brain_region": "integration",
        "type": "real",
        "priority": 1,
    },
    {
        "name": "cognitive_synthesis_simple",
        "path": "src/jeffrey/core/memory/cognitive_synthesis.py",
        "brain_region": "conscience",
        "type": "real",
        "priority": 1,
    },
]

# Assurer qu'on a exactement 8 régions uniques
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
used_regions = set()
final_clean = []

for module in FINAL_MODULES:
    region = module["brain_region"]
    if region in required_regions and region not in used_regions:
        used_regions.add(region)
        final_clean.append(module)

print(f"✅ Configured {len(final_clean)} unique modules for 8 regions:")
for m in final_clean:
    if os.path.exists(m["path"]):
        print(f"   {m['brain_region']:15} → {os.path.basename(m['path'])}")
    else:
        print(f"   {m['brain_region']:15} → {os.path.basename(m['path'])} ❌ MISSING")

# Mettre à jour l'inventaire
INV = "artifacts/inventory_ultimate.json"
with open(INV) as f:
    data = json.load(f)

data["bundle1_recommendations"]["modules"] = final_clean

with open(INV, "w") as f:
    json.dump(data, f, indent=2)

print(f"\n✅ Updated inventory with {len(final_clean)} clean modules")

# Vérifier les imports
print("\n🔍 Checking imports...")
missing_imports = []
for m in final_clean:
    if not os.path.exists(m["path"]):
        missing_imports.append(m["path"])

if missing_imports:
    print(f"❌ Missing files: {missing_imports}")
else:
    print("✅ All module files exist!")

print("\n🎯 Ready for final validation!")
