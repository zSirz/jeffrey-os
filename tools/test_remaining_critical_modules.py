#!/usr/bin/env python3
"""
Test des 3 Modules Critiques Restants
======================================

Modules à tester:
1. jeffrey.core.emotions.core.emotion_ml_enhancer
2. jeffrey.core.jeffrey_emotional_core
3. jeffrey.core.orchestration.orchestrator_manager

Pour chaque module:
- Test d'import
- Test d'initialisation basique (si possible)
- Identification des dépendances manquantes
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Modules à tester
REMAINING_CRITICAL = [
    "jeffrey.core.emotions.core.emotion_ml_enhancer",
    "jeffrey.core.jeffrey_emotional_core",
    "jeffrey.core.orchestration.orchestrator_manager",
]


def test_module(module_name: str) -> dict:
    """
    Teste un module de manière approfondie.

    Returns:
        dict avec status, message, details
    """
    result = {
        "module": module_name,
        "import_success": False,
        "spec_found": False,
        "classes_found": [],
        "error": None,
        "missing_deps": [],
    }

    # Test 1: find_spec
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            result["spec_found"] = True
            result["file_path"] = spec.origin if spec.origin else "N/A"
        else:
            result["error"] = "Module spec is None (module not found)"
            return result
    except (ImportError, ModuleNotFoundError, ValueError) as e:
        result["error"] = f"find_spec failed: {e}"
        return result

    # Test 2: Import réel
    try:
        module = importlib.import_module(module_name)
        result["import_success"] = True

        # Lister les classes/fonctions publiques
        public_items = [name for name in dir(module) if not name.startswith('_') and not name.startswith('__')]
        result["classes_found"] = public_items[:10]  # Top 10

    except ImportError as e:
        result["error"] = f"Import failed: {e}"

        # Tenter d'identifier dépendances manquantes
        error_msg = str(e).lower()
        if "no module named" in error_msg:
            # Extraire le nom du module manquant
            parts = str(e).split("'")
            if len(parts) >= 2:
                missing = parts[1]
                result["missing_deps"].append(missing)

    except Exception as e:
        result["error"] = f"Unexpected error: {type(e).__name__}: {e}"

    return result


def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  TEST DES 3 MODULES CRITIQUES RESTANTS                   ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    results = []

    for i, module_name in enumerate(REMAINING_CRITICAL, 1):
        print(f"━━━ TEST {i}/3 : {module_name} ━━━")
        print()

        result = test_module(module_name)
        results.append(result)

        # Affichage résultat
        if result["import_success"]:
            print("  ✅ SUCCÈS")
            print(f"     Fichier: {result.get('file_path', 'N/A')}")
            if result["classes_found"]:
                print(f"     Classes/Fonctions: {', '.join(result['classes_found'][:5])}...")
        else:
            print("  ❌ ÉCHEC")
            if result["error"]:
                print(f"     Erreur: {result['error']}")
            if result["missing_deps"]:
                print(f"     Dépendances manquantes: {', '.join(result['missing_deps'])}")

        print()

    # Résumé
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("📊 RÉSUMÉ")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    success_count = sum(1 for r in results if r["import_success"])
    total = len(results)

    print(f"\nModules testés: {total}")
    print(f"Succès: {success_count}")
    print(f"Échecs: {total - success_count}")
    print(f"Taux de réussite: {success_count / total * 100:.0f}%\n")

    if success_count == total:
        print("✅ TOUS LES MODULES CRITIQUES RESTANTS SONT FONCTIONNELS\n")
        return 0
    else:
        print("⚠️  CERTAINS MODULES CRITIQUES ONT DES PROBLÈMES\n")
        print("Modules en échec:")
        for r in results:
            if not r["import_success"]:
                print(f"  - {r['module']}")
                if r["missing_deps"]:
                    print(f"    → Dépendances: {', '.join(r['missing_deps'])}")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
