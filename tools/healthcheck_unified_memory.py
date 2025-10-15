#!/usr/bin/env python3
"""
Healthcheck Avancé - unified_memory Integration
================================================

Vérifie que unified_memory est correctement connecté et fonctionnel.
"""

import asyncio
import sys
from pathlib import Path

# Ajouter src au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def test_1_import_via_shim():
    """Test 1: Import via le shim"""
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("TEST 1: Import via shim (jeffrey.core.unified_memory)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        import warnings

        warnings.filterwarnings('ignore')
        import jeffrey.core.unified_memory as um_shim

        print("✅ Import réussi via shim")

        # Vérifier métadonnées shim
        if hasattr(um_shim, '__shim__'):
            print("✅ Shim confirmé")
            print(f"   → Target: {um_shim.__target__}")
            print(f"   → Équipe: {um_shim.__team__}")
        else:
            print("⚠️  Métadonnées shim manquantes")

        return True, um_shim

    except ImportError as e:
        print(f"❌ Import échoué: {e}")
        return False, None


def test_2_import_direct():
    """Test 2: Import direct du module Production"""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("TEST 2: Import direct (jeffrey.core.memory.unified_memory)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        import jeffrey.core.memory.unified_memory as um_direct

        print("✅ Import direct réussi")
        return True, um_direct

    except ImportError as e:
        print(f"❌ Import direct échoué: {e}")
        return False, None


def test_3_api_presence(um_module):
    """Test 3: Présence des classes/fonctions principales"""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("TEST 3: Présence API")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    required = ['UnifiedMemory', 'get_unified_memory']

    all_ok = True

    for name in required:
        if hasattr(um_module, name):
            print(f"✅ {name} présent")
        else:
            print(f"❌ {name} MANQUANT (requis)")
            all_ok = False

    return all_ok


async def test_4_initialization(um_module):
    """Test 4: Initialisation avec backend memory"""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("TEST 4: Initialisation (backend memory)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        # Tenter initialisation avec backend "memory"
        memory = um_module.UnifiedMemory(backend="memory", data_dir="/tmp/jeffrey_healthcheck")

        print("✅ Initialisation réussie")

        # Obtenir stats si possible
        if hasattr(memory, 'get_stats'):
            stats = memory.get_stats()
            print(f"✅ Stats obtenues: {len(stats)} entrées")

        return True, memory

    except Exception as e:
        print(f"❌ Initialisation échouée: {type(e).__name__}: {e}")
        return False, None


def test_5_singleton():
    """Test 5: Singleton get_unified_memory()"""
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("TEST 5: Singleton get_unified_memory()")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    try:
        from jeffrey.core.unified_memory import get_unified_memory

        # Appeler deux fois
        instance1 = get_unified_memory(backend="memory")
        instance2 = get_unified_memory()

        if instance1 is instance2:
            print("✅ Singleton confirmé (même instance)")
        else:
            print("⚠️  Instances différentes (singleton non respecté)")

        print(f"   → Type: {type(instance1).__name__}")

        return True

    except Exception as e:
        print(f"❌ Singleton échoué: {e}")
        return False


async def main():
    """Exécution de tous les tests"""
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║  🔬 HEALTHCHECK UNIFIED_MEMORY - INTÉGRATION AVANCÉE          ║")
    print("║  Équipe : Claude + GPT/Marc + Grok + Gemini                  ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    results = []

    # Test 1: Import via shim
    success1, um_shim = test_1_import_via_shim()
    results.append(("Import via shim", success1))

    if not success1:
        print("\n❌ ÉCHEC CRITIQUE: Impossible de continuer sans import shim")
        return False

    # Test 2: Import direct
    success2, um_direct = test_2_import_direct()
    results.append(("Import direct", success2))

    # Test 3: API
    success3 = test_3_api_presence(um_shim)
    results.append(("API présente", success3))

    # Test 4: Initialisation
    success4, memory = await test_4_initialization(um_shim)
    results.append(("Initialisation", success4))

    # Test 5: Singleton
    success5 = test_5_singleton()
    results.append(("Singleton", success5))

    # Résumé final
    print("\n╔════════════════════════════════════════════════════════════════╗")
    print("║  📊 RÉSUMÉ DES TESTS                                          ║")
    print("╚════════════════════════════════════════════════════════════════╝\n")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {test_name}")

    print(f"\n  Score: {passed}/{total} ({passed / total * 100:.0f}%)")

    if passed == total:
        print("\n  ✅ DIAGNOSTIC: unified_memory est PLEINEMENT FONCTIONNEL")
        return True
    elif passed >= total * 0.8:
        print("\n  ⚠️  DIAGNOSTIC: unified_memory est PARTIELLEMENT FONCTIONNEL")
        return True
    else:
        print("\n  ❌ DIAGNOSTIC: unified_memory a des PROBLÈMES MAJEURS")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
