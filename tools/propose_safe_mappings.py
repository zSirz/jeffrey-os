#!/usr/bin/env python3
"""Propose des mappings sûrs core.* → jeffrey.core.*"""

import importlib.util
import sys
from pathlib import Path

# Ajouter src au path
sys.path.insert(0, str(Path("src").resolve()))


def check_module_exists(module_name):
    """Vérifie si un module peut être importé"""
    try:
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def main():
    # Lire les imports core.* détectés
    core_modules = []
    with open("reports/core_imports_found.txt") as f:
        core_modules = [line.strip() for line in f if line.strip()]

    print("Analyse des mappings possibles...")
    print()

    safe_mappings = {}
    unsafe_mappings = {}

    for core_module in core_modules:
        # Proposer jeffrey.core.* en remplacement
        if core_module == "core":
            jeffrey_module = "jeffrey.core"
        else:
            jeffrey_module = core_module.replace("core.", "jeffrey.core.", 1)

        # Vérifier si la cible existe
        if check_module_exists(jeffrey_module):
            safe_mappings[core_module] = jeffrey_module
            print(f"✅ {core_module} → {jeffrey_module}")
        else:
            unsafe_mappings[core_module] = jeffrey_module
            print(f"❌ {core_module} → {jeffrey_module} (cible n'existe pas)")

    # Sauvegarder les mappings sûrs
    with open("reports/safe_mappings.txt", "w") as f:
        for old, new in safe_mappings.items():
            f.write(f"{old} => {new}\n")

    # Sauvegarder les mappings à investiguer
    with open("reports/unsafe_mappings.txt", "w") as f:
        for old, new in unsafe_mappings.items():
            f.write(f"{old} => {new} (À RESTAURER)\n")

    print()
    print(f"✅ Mappings sûrs : {len(safe_mappings)}")
    print(f"❌ Mappings à investiguer : {len(unsafe_mappings)}")
    print()
    print("📝 Fichiers générés :")
    print("  - reports/safe_mappings.txt (à appliquer)")
    print("  - reports/unsafe_mappings.txt (à restaurer d'abord)")

    return len(unsafe_mappings)


if __name__ == "__main__":
    sys.exit(main())
