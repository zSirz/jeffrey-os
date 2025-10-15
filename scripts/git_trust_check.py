#!/usr/bin/env python3
"""
Vérifie que tous les modules sont trackés par Git et propres.
"""

import json
import subprocess
import sys
from pathlib import Path

# Check compatible CI
ROOT = Path(".")
if not (ROOT / ".git").exists():
    print("ℹ️  Pas de dépôt Git détecté (CI/shallow?) — check ignoré.")
    sys.exit(0)

INV = Path("artifacts/inventory_ultimate.json")

if not INV.exists():
    print("⚠️  Inventaire introuvable")
    sys.exit(0)

mods = json.loads(INV.read_text())["bundle1_recommendations"]["modules"]
bad = False

print("🔍 Vérification Git des modules...\n")

for m in mods:
    p = m["path"]
    try:
        # Vérifie que le fichier est tracké
        subprocess.run(
            ["git", "ls-files", "--error-unmatch", p],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Vérifie qu'il n'y a pas de modifications non commitées
        changed = subprocess.run(["git", "status", "--porcelain", "--", p], capture_output=True, text=True, check=True)

        if changed.stdout.strip():
            print(f"❌ Fichier modifié non commité: {p}")
            bad = True
        else:
            print(f"✅ {m['brain_region']}: {p}")

    except subprocess.CalledProcessError:
        print(f"❌ Fichier non tracké par Git: {p}")
        bad = True

if bad:
    print("\n⛔ Tous les modules doivent être trackés et propres")
    sys.exit(1)

print("\n✅ Tous les fichiers sont trackés et propres")
