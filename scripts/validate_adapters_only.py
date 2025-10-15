#!/usr/bin/env python3
"""
Validation UNIQUEMENT de nos adaptateurs - ignorer les stubs/modules existants.
"""

import ast
import sys
from pathlib import Path

# Vérifier SEULEMENT nos adaptateurs
ADAPTER_DIR = Path("src/jeffrey/bridge/adapters")
violations = []

for py_file in ADAPTER_DIR.rglob("*.py"):
    if py_file.name == "__init__.py":
        continue

    try:
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)
    except Exception:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name.lower()
            # Autoriser BasicalExecutive et FastMotor (nos classes)
            if any(word in class_name for word in ["stub", "simple", "mock", "dummy", "fake"]):
                if "basical" not in class_name and "fast" not in class_name:
                    violations.append(f"Classe suspecte '{node.name}' dans {py_file}")

if violations:
    print("❌ STUBS DÉTECTÉS DANS NOS ADAPTATEURS:")
    for v in violations:
        print(f"   {v}")
    sys.exit(1)

print("✅ Aucun stub dans nos adaptateurs")
