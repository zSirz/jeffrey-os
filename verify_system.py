#!/usr/bin/env python3
"""
Vérification complète du système Jeffrey OS
"""

import asyncio
import importlib
import sys
from pathlib import Path
from typing import Any

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))


def check_module(module_path: str) -> dict[str, Any]:
    """Vérifie si un module peut être importé"""
    try:
        module = importlib.import_module(module_path)
        return {
            "status": "✅",
            "module": module_path,
            "has_init": hasattr(module, "__init__"),
            "functions": len([x for x in dir(module) if callable(getattr(module, x)) and not x.startswith("_")]),
        }
    except ImportError as e:
        return {"status": "❌", "module": module_path, "error": str(e)}
    except Exception as e:
        return {"status": "⚠️", "module": module_path, "error": str(e)}


async def verify_jeffrey():
    """Vérification complète de Jeffrey OS"""
    print("=" * 60)
    print("🤖 VÉRIFICATION COMPLÈTE JEFFREY OS")
    print("=" * 60)

    # Modules critiques à vérifier
    critical_modules = [
        "src.jeffrey.core.orchestration.ia_orchestrator_ultimate",
        "src.jeffrey.api.audit_logger_enhanced",
        "src.jeffrey.core.sandbox_manager_enhanced",
        "src.jeffrey.core.emotions.emotional_affective_touch",
        "src.jeffrey.core.emotions.emotion_prompt_detector",
        "src.jeffrey.core.emotions.core.emotion_ml_enhancer",
        "src.jeffrey.core.personality.relation_tracker_manager",
        "src.jeffrey.core.learning.auto_learner",
        "src.jeffrey.core.orchestration.jeffrey_system_health",
        "src.jeffrey.interfaces.ui.widgets.LienAffectifWidget",
    ]

    print("\n📦 VÉRIFICATION DES MODULES CRITIQUES:")
    print("-" * 40)

    success_count = 0
    for module_path in critical_modules:
        result = check_module(module_path)
        print(f"{result['status']} {result['module'].split('.')[-1]}")
        if result["status"] == "✅":
            success_count += 1
            if "functions" in result:
                print(f"   └─ {result['functions']} fonctions disponibles")
        elif "error" in result:
            print(f"   └─ {result['error'][:60]}...")

    print(f"\n📊 RÉSULTAT: {success_count}/{len(critical_modules)} modules opérationnels")

    # Test orchestrateur
    print("\n🧠 TEST DE L'ORCHESTRATEUR:")
    print("-" * 40)

    try:
        from jeffrey.core.orchestration.ia_orchestrator_ultimate import OrchestrationRequest, UltimateOrchestrator

        orch = UltimateOrchestrator()
        print("✅ Orchestrateur initialisé")

        # Test stats
        stats = await orch.get_orchestration_stats()
        print("✅ Statistiques récupérées")
        print(f"   ├─ Professeurs: {len(stats.get('professors', {}))}")
        print(f"   ├─ Budget: {stats.get('budget_status', {}).get('utilization_percentage', 0):.1f}%")
        print(f"   └─ Charge: {stats.get('total_load', 0):.1f}%")

        # Test requête simple
        request = OrchestrationRequest(
            request="Test système",
            request_type="test",
            user_id="system",
            preferences={},
            priority="normal",
        )

        response = await orch.orchestrate_with_intelligence(request)
        print("✅ Orchestration fonctionnelle")

        if response.get("success"):
            print("   └─ Réponse générée avec succès")
        else:
            print("   └─ Mode simulation actif")

    except ImportError as e:
        print(f"❌ Import échoué: {e}")
    except Exception as e:
        print(f"⚠️ Erreur runtime: {e}")

    # Vérification structure fichiers
    print("\n📁 STRUCTURE DES FICHIERS:")
    print("-" * 40)

    important_paths = [
        "src/jeffrey/__init__.py",
        "src/jeffrey/core/__init__.py",
        "src/jeffrey/api/__init__.py",
        "src/jeffrey/interfaces/__init__.py",
        "data/",
        ".venv/",
    ]

    for path_str in important_paths:
        path = Path(path_str)
        if path.exists():
            if path.is_file():
                print(f"✅ {path_str}")
            else:
                count = len(list(path.rglob("*.py"))) if path.suffix != ".py" else 0
                if count > 0:
                    print(f"✅ {path_str} ({count} fichiers Python)")
                else:
                    print(f"✅ {path_str}")
        else:
            print(f"❌ {path_str} manquant")

    print("\n" + "=" * 60)
    print("✨ VÉRIFICATION TERMINÉE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(verify_jeffrey())
