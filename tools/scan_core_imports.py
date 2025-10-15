#!/usr/bin/env python3
"""Détecte tous les imports core.* dans le projet"""

import re
from collections import defaultdict
from pathlib import Path


def scan_imports(root):
    core_imports = defaultdict(set)

    for py_file in Path(root).rglob("*.py"):
        if "__pycache__" in str(py_file):
            continue

        try:
            content = py_file.read_text(encoding="utf-8")

            # from core.xxx import
            for match in re.finditer(r"from\s+(core(?:\.[\w]+)*)\s+import", content):
                module = match.group(1)
                core_imports[module].add(str(py_file))

            # import core.xxx
            for match in re.finditer(r"import\s+(core(?:\.[\w]+)*)", content):
                module = match.group(1)
                core_imports[module].add(str(py_file))

        except Exception:
            pass

    return core_imports


if __name__ == "__main__":
    imports = scan_imports("src")

    print(f"Imports core.* détectés : {len(imports)}")
    print()

    for module in sorted(imports.keys()):
        print(f"• {module}")
        for file in sorted(imports[module]):
            print(f"  - {file}")
        print()

    # Sauvegarder pour usage ultérieur
    with open("reports/core_imports_found.txt", "w") as f:
        for module in sorted(imports.keys()):
            f.write(f"{module}\n")

    print("📝 Liste sauvegardée dans reports/core_imports_found.txt")
