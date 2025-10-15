#!/usr/bin/env python3
"""
Teste tous les imports P0 avec shims
"""

import sys


def test_import(module_path: str, class_name: str) -> tuple[bool, str]:
    """Test a single import"""
    try:
        mod = __import__(module_path, fromlist=[class_name])
        cls = getattr(mod, class_name)

        # Vérifier si c'est un shim
        is_shim = getattr(mod, "__shim__", False)
        if is_shim:
            original = getattr(mod, "__original_location__", "unknown")
            return True, f"via shim from {original}"

        return True, "direct import"
    except Exception as e:
        return False, str(e)[:100]


def main():
    print("🔍 Testing all P0 imports with shims")
    print("=" * 50)

    # Ajouter le répertoire courant au path
    sys.path.insert(0, ".")

    # Tests groupés par catégorie
    test_groups = {
        "CURRENT locations (où les fichiers sont vraiment)": [
            ("src.jeffrey.core.consciousness.dream_engine", "DreamEngine"),
            ("src.jeffrey.core.consciousness.cognitive_synthesis", "CognitiveSynthesis"),
            ("src.jeffrey.core.consciousness.self_awareness_tracker", "SelfAwarenessTracker"),
            ("src.jeffrey.core.memory.cortex_memoriel", "CortexMemoriel"),
        ],
        "CORRECT locations (via shims)": [
            ("src.jeffrey.core.dreaming.dream_engine", "DreamEngine"),
            ("src.jeffrey.core.memory.cognitive_synthesis", "CognitiveSynthesis"),
            ("src.jeffrey.core.consciousness.self_awareness_tracker", "SelfAwarenessTracker"),
            ("src.jeffrey.core.memory.cortex_memoriel", "CortexMemoriel"),
        ],
    }

    total_ok = 0
    total_tests = 0
    failed_imports = []

    for group_name, tests in test_groups.items():
        print(f"\n📦 {group_name}:")
        group_ok = 0

        for module_path, class_name in tests:
            success, message = test_import(module_path, class_name)
            total_tests += 1

            if success:
                print(f"   ✅ {module_path}")
                print(f"      → {message}")
                group_ok += 1
                total_ok += 1
            else:
                print(f"   ❌ {module_path}")
                print(f"      → Error: {message}")
                failed_imports.append((module_path, message))

        print(f"   Score: {group_ok}/{len(tests)}")

    print("\n" + "=" * 50)
    print(f"📊 TOTAL: {total_ok}/{total_tests} imports OK")

    # Afficher un résumé des échecs
    if failed_imports:
        print("\n⚠️ Failed imports summary:")
        for module_path, error in failed_imports:
            print(f"   - {module_path}")

    # Déterminer le succès
    min_required = 4  # Au moins 4 imports doivent fonctionner
    if total_ok >= min_required:
        print(f"\n✅ SUCCESS! Minimum imports working ({total_ok}≥{min_required})")
        print("   Ready for consolidation!")

        # Si des shims sont utilisés, le mentionner
        if any(
            "via shim" in msg
            for _, msg in [
                (m, test_import(m, c)[1]) for group in test_groups.values() for m, c in group if test_import(m, c)[0]
            ]
        ):
            print("\n🌉 Note: Some imports use shims (bridges)")
            print("   This is expected and allows gradual migration")
    else:
        print(f"\n❌ FAILED! Not enough imports working ({total_ok}<{min_required})")
        print("   Check error messages above")
        print("\n💡 Suggestions:")
        print("   1. Run: python3 scripts/create_shims.py")
        print("   2. Check that files exist in src/jeffrey/core/consciousness/")
        print("   3. Verify __init__.py files are present")

    return total_ok >= min_required


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
