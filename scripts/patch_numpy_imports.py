#!/usr/bin/env python3
"""Patch numpy imports in P0 modules to add protection"""

import ast
import re
from pathlib import Path


def patch_numpy_import(filepath):
    """Replace unprotected numpy imports with try/except"""

    numpy_protection = '''try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # DEV-ONLY MOCK - Ne pas utiliser en production
    class MockNumpy:
        """Minimal numpy mock for testing only"""
        def array(self, data, *args, **kwargs):
            return list(data) if hasattr(data, '__iter__') else [data]

        def zeros(self, shape, *args, **kwargs):
            if isinstance(shape, tuple):
                if len(shape) == 2:
                    return [[0] * shape[1] for _ in range(shape[0])]
                elif len(shape) == 1:
                    return [0] * shape[0]
                else:
                    return [0] * shape[0] if shape else [0]
            return [0] * shape

        def ones(self, shape, *args, **kwargs):
            if isinstance(shape, tuple):
                if len(shape) == 2:
                    return [[1] * shape[1] for _ in range(shape[0])]
                elif len(shape) == 1:
                    return [1] * shape[0]
                else:
                    return [1] * shape[0] if shape else [1]
            return [1] * shape

        def mean(self, data, *args, **kwargs):
            flat = [x for row in data for x in row] if (isinstance(data, list) and data and isinstance(data[0], list)) else data
            return sum(flat) / len(flat) if flat else 0

        def dot(self, a, b):
            if isinstance(a, list) and isinstance(b, list):
                return sum(x * y for x, y in zip(a, b))
            return 0

        class random:
            @staticmethod
            def rand(*shape):
                import random
                if len(shape) == 2:
                    return [[random.random() for _ in range(shape[1])] for _ in range(shape[0])]
                elif len(shape) == 1:
                    return [random.random() for _ in range(shape[0])]
                return random.random()

            @staticmethod
            def choice(arr):
                import random
                return random.choice(list(arr)) if arr else None

            @staticmethod
            def randint(low, high=None, size=None):
                import random
                if high is None:
                    high = low
                    low = 0
                if size is None:
                    return random.randint(low, high - 1)
                elif isinstance(size, int):
                    return [random.randint(low, high - 1) for _ in range(size)]
                else:
                    return [[random.randint(low, high - 1) for _ in range(size[1])] for _ in range(size[0])]

    np = MockNumpy()
    np.random = np.random()'''

    if not filepath.exists():
        print(f"⚠️ {filepath.name} not found")
        return False

    content = filepath.read_text()

    # Vérifier si déjà protégé
    if "try:" in content and "import numpy" in content:
        lines = content.splitlines()
        for i, line in enumerate(lines):
            if "import numpy" in line:
                # Vérifier si dans un bloc try
                if i > 0 and "try:" in lines[i - 1]:
                    print(f"✅ {filepath.name} already protected")
                    return False

    # Pattern pour trouver import numpy non protégé
    patterns = [
        (r"^import numpy as np\s*$", numpy_protection),
        (r"^from numpy import \*\s*$", numpy_protection),
        (r"^import numpy\s*$", numpy_protection.replace("as np", "")),
    ]

    modified = False
    for pattern, replacement in patterns:
        if re.search(pattern, content, re.MULTILINE):
            content = re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE)
            modified = True
            break

    if modified:
        # Vérifier que le code est toujours valide
        try:
            ast.parse(content)
            filepath.write_text(content)
            print(f"✅ Patched {filepath.name}")
            return True
        except SyntaxError as e:
            print(f"❌ Syntax error after patching {filepath.name}: {e}")
            return False
    else:
        print(f"ℹ️ {filepath.name} - no unprotected numpy import found")
        return False


def main():
    """Main patching function"""
    base = Path.cwd()

    # Vérifier qu'on est dans le bon répertoire
    if not (base / ".git").exists():
        print(f"❌ Not in a git repository: {base}")
        return

    print("🚀 Patching numpy imports in P0 modules")
    print(f"   Base directory: {base}")
    print("=" * 50)

    # Modules P0 avec les bons chemins
    modules = [
        base / "src/jeffrey/core/dreaming/dream_engine.py",
        base / "src/jeffrey/core/consciousness/self_awareness_tracker.py",
        base / "src/jeffrey/core/memory/cognitive_synthesis.py",
        base / "src/jeffrey/core/memory/cortex_memoriel.py",
    ]

    patched_count = 0
    for module_path in modules:
        if module_path.exists():
            if patch_numpy_import(module_path):
                patched_count += 1
        else:
            print(f"⚠️ {module_path.relative_to(base)} not found")

    print("=" * 50)
    if patched_count > 0:
        print(f"✅ Numpy patching complete! Patched {patched_count} file(s)")
    else:
        print("ℹ️ No files needed patching")


if __name__ == "__main__":
    main()
