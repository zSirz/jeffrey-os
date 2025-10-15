#!/usr/bin/env python3
"""Minimal smoke test for critical modules"""

import ast
import sys
from pathlib import Path


def test_critical_syntax():
    """Test que les modules critiques n'ont pas d'erreur de syntaxe"""
    critical_paths = [
        "src/jeffrey/core/bus",
        "src/jeffrey/core/security/guardian",
        "src/jeffrey/core/orchestration",
        "src/jeffrey/core/memory",
        "src/jeffrey/core/consciousness",
    ]

    errors = []

    for path_str in critical_paths:
        path = Path(path_str)
        if not path.exists():
            continue

        for py_file in path.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file) as f:
                    ast.parse(f.read())
            except SyntaxError as e:
                errors.append(f"{py_file}: {e}")

    if errors:
        print(f"❌ {len(errors)} syntax errors found in critical modules:")
        for error in errors[:5]:  # Show first 5
            print(f"  {error}")
        sys.exit(1)

    print("✅ All critical modules have valid syntax")
    return True


if __name__ == "__main__":
    test_critical_syntax()
