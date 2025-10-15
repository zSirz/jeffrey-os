#!/usr/bin/env python3
"""
Garde-fou : Vérifie qu'aucun module simple/stub n'est dans l'inventaire.
"""

import json
import sys
from pathlib import Path

INV = Path("artifacts/inventory_ultimate.json")

if not INV.exists():
    print("⚠️  Inventaire introuvable")
    sys.exit(0)

data = json.loads(INV.read_text())
modules = data.get("bundle1_recommendations", {}).get("modules", [])

bad = [
    m
    for m in modules
    if "/simple_modules/" in m.get("path", "") or "/stubs/" in m.get("path", "") or "Stub" in m.get("name", "")
]

if bad:
    print("❌ MODULES INTERDITS DÉTECTÉS:")
    for m in bad:
        print(f"   • {m['brain_region']} → {m['path']}")
    sys.exit(1)

print(f"✅ Aucun placeholder ({len(modules)} modules vérifiés)")
