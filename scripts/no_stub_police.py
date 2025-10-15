#!/usr/bin/env python3
"""
Détecte TOUS les stubs/simples via AST et patterns.
"""

import ast
import sys
from pathlib import Path

ROOT = Path("src/jeffrey")
BANNED_PATTERNS = ["/stubs/", "/simple_", "/mock_", "/dummy_", "/fake_"]
violations = []

for py_file in ROOT.rglob("*.py"):
    file_path = str(py_file)

    # Pattern dans chemin
    if any(pattern in file_path.lower() for pattern in BANNED_PATTERNS):
        violations.append(f"Chemin banni: {py_file}")
        continue

    # AST pour classes suspectes
    try:
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(content)
    except Exception:
        continue

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_name = node.name.lower()
            if any(word in class_name for word in ["stub", "simple", "mock", "dummy", "fake"]):
                violations.append(f"Classe suspecte '{node.name}' dans {py_file}")

if violations:
    print("❌ STUBS/CONTOURNEMENTS DÉTECTÉS:")
    for v in violations:
        print(f"   {v}")
    sys.exit(1)

print("✅ Aucun stub détecté")
