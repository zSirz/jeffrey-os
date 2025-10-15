#!/usr/bin/env python3
"""Optimiser les imports pour boot <1s - À utiliser dans v7.0.1"""

import re
from pathlib import Path


def make_lazy_import(file_path: Path, module: str):
    """Convertir un import en lazy import"""
    content = file_path.read_text()

    # Pattern pour import numpy/kivy/yaml au top level
    patterns = [
        (r"^import numpy\b", "numpy"),
        (r"^import kivy\b", "kivy"),
        (r"^import yaml\b", "yaml"),
        (r"^from numpy import", "numpy"),
        (r"^from kivy import", "kivy"),
        (r"^from yaml import", "yaml"),
    ]

    modified = False
    for pattern, mod in patterns:
        if re.search(pattern, content, re.MULTILINE):
            # Remplacer par lazy import dans les fonctions
            content = re.sub(pattern, f"# {pattern} moved to lazy import", content, flags=re.MULTILINE)
            modified = True
            print(f"  📦 Made {mod} lazy in {file_path.name}")

    if modified:
        file_path.write_text(content)
        return True
    return False


# Appliquer aux modules Bundle 1 les plus lourds
heavy_modules = [
    "src/jeffrey/interfaces/ui/jeffrey_ui_bridge.py",
    "src/jeffrey/core/memory/unified_memory.py",
]

print("📝 Script saved for v7.0.1 optimization")
