#!/usr/bin/env python3
"""
Patch numpy imports de manière sécurisée avec flag environnement
"""

import os
import re
import sys
from pathlib import Path

# GARDE-FOU GPT : Flag obligatoire pour éviter le patch en prod
if os.getenv("JEFFREY_OFFLINE_MODE") != "1":
    print("❌ ERROR: Numpy patching requires JEFFREY_OFFLINE_MODE=1")
    print("   This prevents dev mocks from leaking to production")
    print("   Run with: JEFFREY_OFFLINE_MODE=1 python3 scripts/patch_numpy_secure.py")
    sys.exit(1)


def patch_numpy_in_file(filepath):
    """Patch numpy import with idempotence check"""

    numpy_protection = '''# [JEFFREY_NP_PATCH] - Dev-only mock for offline mode
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # DEV-ONLY MOCK - DO NOT USE IN PRODUCTION
    import os
    if os.getenv("JEFFREY_OFFLINE_MODE") != "1":
        raise ImportError("Numpy not available and mock disabled in production")

    class MockNumpy:
        """Minimal numpy mock for offline testing only"""
        def array(self, data, *args, **kwargs):
            return list(data) if hasattr(data, '__iter__') else [data]

        def zeros(self, shape, *args, **kwargs):
            if isinstance(shape, tuple):
                if len(shape) == 2:
                    return [[0] * shape[1] for _ in range(shape[0])]
                elif len(shape) == 1:
                    return [0] * shape[0]
            return [0] * (shape if isinstance(shape, int) else 1)

        def ones(self, shape, *args, **kwargs):
            if isinstance(shape, tuple):
                if len(shape) == 2:
                    return [[1] * shape[1] for _ in range(shape[0])]
                elif len(shape) == 1:
                    return [1] * shape[0]
            return [1] * (shape if isinstance(shape, int) else 1)

        def mean(self, data, *args, **kwargs):
            if not data: return 0
            if isinstance(data[0] if data else None, list):
                flat = [x for row in data for x in row]
            else:
                flat = list(data)
            return sum(flat) / len(flat) if flat else 0

        def std(self, data, *args, **kwargs):
            return 0.1  # Mock value

        def sum(self, data, *args, **kwargs):
            if isinstance(data[0] if data else None, list):
                return sum(sum(row) for row in data)
            return sum(data) if data else 0

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
                    return random.randint(low, high-1)
                return [random.randint(low, high-1) for _ in range(size)]

    np = MockNumpy()
    np.random = np.random()'''

    if not filepath.exists():
        return False, "File not found"

    content = filepath.read_text()

    # IDEMPOTENCE : Check if already patched with our tag
    if "[JEFFREY_NP_PATCH]" in content:
        return False, "Already patched (tag found)"

    # Check if numpy import exists
    if "import numpy as np" not in content:
        return False, "No numpy import found"

    # Replace unprotected numpy import
    new_content = re.sub(r"^import numpy as np\s*$", numpy_protection, content, count=1, flags=re.MULTILINE)

    if new_content != content:
        filepath.write_text(new_content)
        return True, "Patched successfully"
    else:
        return False, "Pattern not found"


def sanity_check():
    """Quick import test before patching"""
    print("\n🔍 Sanity check before patching:")

    targets = [
        ("src.jeffrey.core.consciousness.dream_engine", "DreamEngine"),
        ("src.jeffrey.core.consciousness.cognitive_synthesis", "CognitiveSynthesis"),
        ("src.jeffrey.core.consciousness.self_awareness_tracker", "SelfAwarenessTracker"),
        ("src.jeffrey.core.memory.cortex_memoriel", "CortexMemoriel"),
    ]

    import sys

    sys.path.insert(0, ".")

    for mod_path, class_name in targets:
        try:
            mod = __import__(mod_path, fromlist=[class_name])
            getattr(mod, class_name)
            print(f"   ✅ {mod_path} importable")
        except Exception as e:
            error_msg = str(e)
            if "numpy" in error_msg.lower():
                print(f"   ⚠️ {mod_path} needs numpy (expected)")
            else:
                print(f"   ❌ {mod_path} has other error: {error_msg[:50]}")


def main():
    print("🔧 Numpy Patching - SECURE MODE")
    print(f"   JEFFREY_OFFLINE_MODE = {os.getenv('JEFFREY_OFFLINE_MODE')}")
    print("=" * 50)

    # Sanity check first
    sanity_check()

    print("\n📝 Patching numpy imports:")
    print("=" * 50)

    base_dir = Path.cwd()

    # Modules P0 aux emplacements RÉELS
    modules_to_patch = [
        base_dir / "src/jeffrey/core/consciousness/dream_engine.py",
        base_dir / "src/jeffrey/core/consciousness/cognitive_synthesis.py",
        base_dir / "src/jeffrey/core/consciousness/self_awareness_tracker.py",
        base_dir / "src/jeffrey/core/memory/cortex_memoriel.py",
    ]

    patched_count = 0
    for module_path in modules_to_patch:
        success, message = patch_numpy_in_file(module_path)
        status = "✅" if success else "⏭️"
        print(f"{status} {module_path.name}: {message}")
        if success:
            patched_count += 1

    print("=" * 50)
    print(f"✅ Patching complete! {patched_count} file(s) modified")

    # Verification
    print("\n📊 Final verification:")
    for module_path in modules_to_patch:
        if module_path.exists():
            content = module_path.read_text()
            has_patch = "[JEFFREY_NP_PATCH]" in content
            status = "✅" if has_patch else "❌"
            print(f"  {status} {module_path.name}")

    print("\n⚠️ Remember: This is a DEV-ONLY patch!")
    print("   Never deploy with JEFFREY_OFFLINE_MODE=1 in production")


if __name__ == "__main__":
    main()
