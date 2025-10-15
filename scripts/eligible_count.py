#!/usr/bin/env python3
"""Count eligible modules with exact same rules as inventory"""

import fnmatch
import os
from pathlib import Path

MIN_LINES = 60

# Mêmes patterns d'exclusion que inventory_ultimate.py
EXCLUSION_PATTERNS = [
    "*/tests/*",
    "*/test/*",
    "*_test.py",
    "*test_*.py",
    "*/__pycache__/*",
    "*/.venv/*",
    "*/venv/*",
    "*/archive/*",
    "*/deprecated/*",
    "*/old/*",
    "*/backup/*",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
]

# Whitelist complète (même que inventory_ultimate.py)
WHITELIST_PATTERNS = [
    "response_generator",
    "ollama_interface",
    "adapter_*",
    "bridge_*",
    "config",
    "constants",
    "simple_*",
]


def is_excluded(path: Path) -> bool:
    """Check if path matches exclusion patterns"""
    path_str = os.path.normpath(str(path))
    for pattern in EXCLUSION_PATTERNS:
        if fnmatch.fnmatch(path_str, pattern):
            return True
    return False


def is_whitelisted(stem: str) -> bool:
    """Check if module name matches whitelist patterns"""
    for pattern in WHITELIST_PATTERNS:
        if fnmatch.fnmatch(stem, pattern):
            return True
    return False


def count_eligible():
    """Count eligible modules with exact inventory rules"""
    root = Path("src/jeffrey")
    if not root.exists():
        print("0")
        return 0

    files = [p for p in root.rglob("*.py") if not is_excluded(p)]

    eligible = 0
    for path in files:
        try:
            # Count lines efficiently
            line_count = 0
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_count, _ in enumerate(f, 1):
                    if line_count > MIN_LINES:
                        break

            stem = path.stem

            # Apply same rules as inventory: >=60 lines OR whitelisted
            if line_count >= MIN_LINES or is_whitelisted(stem):
                eligible += 1

        except Exception:
            # Skip unreadable files
            pass

    return eligible


if __name__ == "__main__":
    count = count_eligible()
    print(count)
