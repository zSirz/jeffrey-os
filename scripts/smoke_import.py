"""
Smoke test complet qui ignore vendors/icloud pour se concentrer sur le core.
Distingue SyntaxError (bloquant) vs ImportError (dépendances manquantes).
"""

import importlib
import os
import pkgutil
import sys


def should_skip(module_name: str) -> bool:
    """Skip vendors/icloud et backups (backups corrompues)."""
    SKIP_PREFIXES = (
        "jeffrey.vendors.icloud",
        "jeffrey.vendors.backups",
    )
    return module_name.startswith(SKIP_PREFIXES)


def smoke_test():
    root_pkg = "jeffrey"
    sys.path.insert(0, os.path.abspath("src"))

    syntax_errors = []
    import_errors = []
    success = []
    others = []

    print("🔍 Scanning modules (ignoring vendors/icloud)...\n")

    for mod_info in pkgutil.walk_packages([f"src/{root_pkg}"], prefix=f"{root_pkg}."):
        name = mod_info.name

        # Skip vendors/icloud
        if should_skip(name):
            continue

        try:
            importlib.import_module(name)
            success.append(name)
        except (SyntaxError, IndentationError) as e:
            syntax_errors.append((name, f"{e.__class__.__name__}: {e}"))
        except ImportError as e:
            import_errors.append((name, f"ImportError: {str(e)[:100]}"))
        except Exception as e:
            others.append((name, f"{e.__class__.__name__}: {str(e)[:100]}"))

    total = len(success) + len(syntax_errors) + len(import_errors) + len(others)

    print("=" * 70)
    print(f"Total modules scannés    : {total}")
    print(f"✅ Imports réussis       : {len(success)} ({len(success) / total * 100:.1f}%)")
    print(f"🔴 Erreurs syntaxe       : {len(syntax_errors)}")
    print(f"⚠️  Erreurs imports      : {len(import_errors)}")
    print(f"⚠️  Autres exceptions    : {len(others)}")
    print("=" * 70 + "\n")

    def dump(title, items, limit=12):
        if not items:
            return
        print(title)
        for name, err in items[:limit]:
            print(f"  ❌ {name}")
            print(f"     → {err}\n")

    dump("🔴 À CORRIGER EN PRIORITÉ (Syntax/Indent) :", syntax_errors)
    dump("⚠️  Import manquant (dépendances / chemins) :", import_errors)
    dump("⚠️  Autres exceptions (à inspecter) :", others)

    # Exit code policy
    if syntax_errors:
        sys.exit(1)  # Bloquant
    elif len(import_errors) > 10:
        sys.exit(1)  # Trop d'imports manquants
    else:
        print("✅ SMOKE TEST PASSED")
        sys.exit(0)


if __name__ == "__main__":
    smoke_test()
