#!/usr/bin/env python3
"""
Health Check Global Jeffrey OS
===============================

Teste les 7 modules critiques identifiés par l'équipe:
1. jeffrey.core.consciousness_loop (déjà OK)
2. jeffrey.core.emotions.core.emotion_engine (déjà OK)
3. jeffrey.core.memory.memory_manager (déjà OK)
4. jeffrey.core.unified_memory (déjà OK - shim créé)
5. jeffrey.core.emotions.core.emotion_ml_enhancer (à tester)
6. jeffrey.core.jeffrey_emotional_core (à tester)
7. jeffrey.core.orchestration.orchestrator_manager (à tester)
"""

import importlib.util
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

CRITICAL_MODULES = [
    ("consciousness_loop", "jeffrey.core.consciousness_loop"),
    ("emotion_engine", "jeffrey.core.emotions.core.emotion_engine"),
    ("memory_manager", "jeffrey.core.memory.memory_manager"),
    ("unified_memory", "jeffrey.core.unified_memory"),
    ("emotion_ml_enhancer", "jeffrey.core.emotions.core.emotion_ml_enhancer"),
    ("jeffrey_emotional_core", "jeffrey.core.jeffrey_emotional_core"),
    ("orchestrator_manager", "jeffrey.core.orchestration.orchestrator_manager"),
]


def check_module(module_name: str) -> tuple[bool, str]:
    """Vérifie si un module peut être importé."""
    try:
        # Ignorer warnings pour clarté
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            spec = importlib.util.find_spec(module_name)
            if spec is None:
                return False, "Module spec is None"

            module = importlib.import_module(module_name)

            if not hasattr(module, '__name__'):
                return False, "Module structure invalid"

            return True, "OK"

    except ImportError as e:
        return False, f"ImportError: {e}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main():
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  🏥 HEALTH CHECK GLOBAL - JEFFREY OS                      ║")
    print("║  7 Modules Critiques                                      ║")
    print("╚════════════════════════════════════════════════════════════╝\n")

    results = []
    success_count = 0

    for short_name, module_name in CRITICAL_MODULES:
        success, message = check_module(module_name)
        results.append((short_name, module_name, success, message))

        if success:
            success_count += 1
            print(f"✅ {short_name:25s} OK")
        else:
            print(f"❌ {short_name:25s} FAIL")
            print(f"   → {message}")

    print("\n" + "=" * 60)

    total = len(CRITICAL_MODULES)
    health_score = (success_count / total) * 100

    print("\n📊 RÉSULTATS:")
    print(f"   Modules OK: {success_count}/{total}")
    print(f"   Health Score: {health_score:.1f}%\n")

    # Diagnostic
    if health_score == 100.0:
        print("✅ SYSTÈME PARFAITEMENT SAIN\n")
        return 0
    elif health_score >= 80.0:
        print("✅ SYSTÈME SAIN (≥80%)\n")
        return 0
    elif health_score >= 60.0:
        print("⚠️  SYSTÈME DÉGRADÉ (60-80%)\n")
        return 1
    else:
        print("❌ SYSTÈME CRITIQUE (<60%)\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
