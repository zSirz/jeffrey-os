#!/usr/bin/env python3
"""
Vérifie noyau intact via Git diff.
"""

import subprocess
import sys

PROTECTED = ("src/jeffrey/core/", "src/jeffrey/bundle2/")

result = subprocess.run(["git", "diff", "--name-only"], capture_output=True, text=True, check=False)

changed = result.stdout.strip().splitlines()
violations = [f for f in changed if any(f.startswith(d) for d in PROTECTED)]

if violations:
    print("❌ NOYAU MODIFIÉ (interdit):")
    for v in violations:
        print(f"   {v}")

    # Afficher diff
    for v in violations:
        print(f"\n--- Diff de {v} ---")
        diff = subprocess.run(["git", "diff", v], capture_output=True, text=True)
        print(diff.stdout)

    sys.exit(1)

print("✅ Noyau intact")
