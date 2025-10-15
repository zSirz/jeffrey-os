#!/usr/bin/env python3
"""
Carte complète de tous les modules Jeffrey OS
Analyse et catégorise tout ce qui est disponible
"""

import ast
import json
from pathlib import Path


def analyze_module(file_path):
    """Analyse un module Python pour extraire les classes et méthodes importantes"""
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)

        classes = []
        functions = []
        has_async = False
        imports_bus = "NeuroBus" in content or "neural_bus" in content

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Récupère les méthodes principales
                methods = []
                for item in node.body:
                    if isinstance(item, ast.AsyncFunctionDef):
                        has_async = True
                        methods.append(f"async {item.name}")
                    elif isinstance(item, ast.FunctionDef):
                        methods.append(item.name)

                classes.append(
                    {"name": node.name, "methods": methods[:5], "async": has_async}  # Top 5 methods
                )
            elif isinstance(node, ast.FunctionDef):
                functions.append(node.name)

        return {
            "classes": classes,
            "functions": functions[:3],
            "imports_bus": imports_bus,
            "has_async": has_async,
            "size": len(content.splitlines()),
        }
    except Exception as e:
        return {"error": str(e)}


def scan_jeffrey():
    """Scan complet de Jeffrey avec analyse détaillée"""

    categories = {
        "memory": {"path": "memory", "modules": {}},
        "learning": {"path": "learning", "modules": {}},
        "consciousness": {"path": "consciousness", "modules": {}},
        "emotions": {"path": "emotions", "modules": {}},
        "orchestration": {"path": "orchestration", "modules": {}},
        "personality": {"path": "personality", "modules": {}},
        "cognition": {"path": "cognition", "modules": {}},
        "dreaming": {"path": "dreaming", "modules": {}},
        "dreams": {"path": "dreams", "modules": {}},
        "loops": {"path": "loops", "modules": {}},
        "ml": {"path": "ml", "modules": {}},
        "monitoring": {"path": "monitoring", "modules": {}},
        "symbiosis": {"path": "symbiosis", "modules": {}},
        "guardians": {"path": "guardians", "modules": {}},
        "neuralbus": {"path": "neuralbus", "modules": {}},
    }

    base_path = Path("src/jeffrey/core")

    # Scan par catégorie
    for cat_name, cat_info in categories.items():
        cat_path = base_path / cat_info["path"]
        if cat_path.exists():
            for py_file in cat_path.glob("**/*.py"):
                if py_file.name not in ["__init__.py", "__pycache__"]:
                    analysis = analyze_module(py_file)
                    if not analysis.get("error") and (analysis.get("classes") or analysis.get("functions")):
                        cat_info["modules"][py_file.stem] = analysis

    # Affichage détaillé
    print("=" * 80)
    print("🧠 JEFFREY OS - COMPLETE MODULE INVENTORY")
    print("=" * 80)

    total_classes = 0
    total_files = 0
    bus_ready = []
    priority_modules = []

    for cat_name, cat_info in categories.items():
        modules = cat_info["modules"]
        if modules:
            # Compter les classes
            class_count = sum(len(m.get("classes", [])) for m in modules.values())
            total_classes += class_count
            total_files += len(modules)

            print(f"\n📂 {cat_name.upper()} ({len(modules)} files, {class_count} classes)")
            print("-" * 60)

            # Afficher les modules principaux
            for name, info in list(modules.items())[:5]:
                classes = info.get("classes", [])
                if classes:
                    main_class = classes[0]
                    status = []
                    if info.get("imports_bus"):
                        status.append("🚌 Bus-ready")
                        bus_ready.append(f"{cat_name}/{name}")
                    if info.get("has_async"):
                        status.append("⚡ Async")

                    print(f"  📄 {name}.py ({info['size']} lines) {' '.join(status)}")
                    print(f"     └─ {main_class['name']}")

                    # Méthodes importantes
                    if main_class.get("methods"):
                        for method in main_class["methods"][:3]:
                            print(f"        • {method}()")

                    # Module prioritaire ?
                    if any(keyword in name.lower() for keyword in ["unified", "main", "core", "manager"]):
                        priority_modules.append((cat_name, name, main_class["name"]))

            if len(modules) > 5:
                print(f"\n  ... et {len(modules) - 5} autres modules")

    # Résumé et recommandations
    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ")
    print("=" * 80)
    print(f"Total: {total_files} fichiers, {total_classes} classes")
    print(f"Modules NeuroBus-ready: {len(bus_ready)}")

    print("\n🎯 MODULES PRIORITAIRES DÉTECTÉS:")
    for cat, file, cls in priority_modules[:10]:
        print(f"  • {cat}/{file}.py -> {cls}")

    print("\n🚀 PLAN DE CONNEXION RECOMMANDÉ:")
    print("\n  Phase 1 - Mémoire de base:")
    print("    1. memory/unified_memory.py -> UnifiedMemory")
    print("    2. memory/memory_manager.py -> MemoryManager")

    print("\n  Phase 2 - Apprentissage:")
    print("    3. learning/auto_learner.py -> AutoLearner")
    print("    4. learning/adaptive_integrator.py -> AdaptiveIntegrator")

    print("\n  Phase 3 - Conscience:")
    print("    5. consciousness/jeffrey_living_consciousness.py")
    print("    6. emotions/emotional_consciousness.py")

    print("\n  Phase 4 - Loops autonomes:")
    print("    7. loops/awareness.py -> AwarenessLoop")
    print("    8. loops/memory_consolidation.py -> MemoryConsolidationLoop")

    # Sauvegarder l'inventaire
    inventory = {
        "categories": {
            name: {"count": len(info["modules"]), "modules": list(info["modules"].keys())}
            for name, info in categories.items()
            if info["modules"]
        },
        "stats": {
            "total_files": total_files,
            "total_classes": total_classes,
            "bus_ready": bus_ready,
        },
    }

    with open("jeffrey_module_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

    print("\n💾 Inventaire sauvé dans jeffrey_module_inventory.json")

    return inventory


if __name__ == "__main__":
    scan_jeffrey()
