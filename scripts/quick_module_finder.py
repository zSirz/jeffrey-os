#!/usr/bin/env python3
"""
Quick module finder for Jeffrey OS
Finds all real modules (>100 lines) and categorizes them
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def find_all_modules():
    """Find all Python modules > 100 lines"""

    modules = {"local": defaultdict(list), "icloud": defaultdict(list)}

    # Scan local src/
    local_path = Path("src/jeffrey")
    if local_path.exists():
        print("📂 Scanning local src/jeffrey...")
        for py_file in local_path.rglob("*.py"):
            # Skip tests and pycache
            if any(skip in str(py_file) for skip in ["__pycache__", "test_", "_test.py"]):
                continue

            try:
                lines = len(py_file.read_text(errors="ignore").splitlines())
                if lines >= 100:
                    rel_path = py_file.relative_to(Path("src/jeffrey"))
                    category = str(rel_path.parent) if rel_path.parent != Path(".") else "root"
                    modules["local"][category].append({"name": py_file.stem, "path": str(py_file), "lines": lines})
            except:
                pass

    # Scan iCloud
    icloud_path = Path("/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS")
    if icloud_path.exists():
        print("☁️ Scanning iCloud modules...")
        for py_file in icloud_path.rglob("*.py"):
            # Skip tests, backups, archives
            if any(
                skip in str(py_file)
                for skip in [
                    "__pycache__",
                    "test_",
                    "_test.py",
                    "backup",
                    "archive",
                    "Archive",
                    "Backup",
                    "Old",
                    "deprecated",
                ]
            ):
                continue

            try:
                lines = len(py_file.read_text(errors="ignore").splitlines())
                if lines >= 100:
                    # Get relative path from iCloud base
                    rel_path = py_file.relative_to(icloud_path)
                    category = str(rel_path.parent) if rel_path.parent != Path(".") else "root"
                    modules["icloud"][category].append({"name": py_file.stem, "path": str(py_file), "lines": lines})
            except:
                pass

    return modules


def categorize_modules(modules):
    """Categorize modules by brain region based on path/name"""

    brain_categories = {
        "CORTEX_FRONTAL": ["orchestrator", "executive", "decision", "control", "agi", "meta"],
        "CORTEX_TEMPORAL": ["memory", "recall", "storage", "cortex", "unified_memory"],
        "SYSTEME_LIMBIQUE": ["emotion", "mood", "empathy", "limbic", "jeffrey_emotional"],
        "CORTEX_OCCIPITAL": ["input", "parser", "perception", "detector", "thalamic"],
        "AIRE_BROCA_WERNICKE": ["response", "generator", "llm", "apertus", "ollama"],
        "HIPPOCAMPE": ["learning", "curiosity", "loop", "auto_learner", "theory_of_mind"],
        "TRONC_CEREBRAL": ["bus", "kernel", "pipeline", "runtime", "neural_bus"],
        "CORPS_CALLEUX": ["bridge", "adapter", "interface", "glue", "connector"],
    }

    categorized = defaultdict(list)

    for source, categories in modules.items():
        for category, module_list in categories.items():
            for module in module_list:
                # Try to detect brain region
                module_lower = module["name"].lower()
                path_lower = module["path"].lower()

                found_region = "UNKNOWN"
                for region, keywords in brain_categories.items():
                    if any(kw in module_lower or kw in path_lower for kw in keywords):
                        found_region = region
                        break

                categorized[found_region].append({**module, "source": source, "category": category})

    return categorized


def print_report(modules, categorized):
    """Print comprehensive report"""

    print("\n" + "=" * 80)
    print("📊 JEFFREY OS MODULE INVENTORY REPORT")
    print("=" * 80)

    # Count totals
    total_local = sum(len(mods) for mods in modules["local"].values())
    total_icloud = sum(len(mods) for mods in modules["icloud"].values())

    print("\n📈 TOTALS:")
    print(f"   Local modules: {total_local}")
    print(f"   iCloud modules: {total_icloud}")
    print(f"   Total modules: {total_local + total_icloud}")

    # By category (local)
    if modules["local"]:
        print("\n📂 LOCAL MODULES BY CATEGORY:")
        for category in sorted(modules["local"].keys()):
            module_list = modules["local"][category]
            print(f"\n   {category}/: {len(module_list)} modules")
            for mod in sorted(module_list, key=lambda x: x["lines"], reverse=True)[:5]:
                print(f"      - {mod['name']}: {mod['lines']} lines")

    # By brain region
    print("\n🧠 MODULES BY BRAIN REGION:")

    brain_names = {
        "CORTEX_FRONTAL": "🎭 Cortex Frontal (Executive)",
        "CORTEX_TEMPORAL": "🧩 Cortex Temporal (Memory)",
        "SYSTEME_LIMBIQUE": "💭 Système Limbique (Emotions)",
        "CORTEX_OCCIPITAL": "👁️ Cortex Occipital (Perception)",
        "AIRE_BROCA_WERNICKE": "🗣️ Broca/Wernicke (Expression)",
        "HIPPOCAMPE": "🔄 Hippocampe (Learning)",
        "TRONC_CEREBRAL": "⚡ Tronc Cérébral (Infrastructure)",
        "CORPS_CALLEUX": "🌟 Corps Calleux (Integration)",
        "UNKNOWN": "❓ Unknown Region",
    }

    for region, name in brain_names.items():
        if region in categorized:
            mods = categorized[region]
            print(f"\n   {name}: {len(mods)} modules")
            # Show top 3 by size
            for mod in sorted(mods, key=lambda x: x["lines"], reverse=True)[:3]:
                src = "📁" if mod["source"] == "local" else "☁️"
                print(f"      {src} {mod['name']}: {mod['lines']} lines")

    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_local": total_local,
            "total_icloud": total_icloud,
            "total": total_local + total_icloud,
        },
        "by_source": modules,
        "by_brain_region": {
            k: [{"name": m["name"], "path": m["path"], "lines": m["lines"]} for m in v] for k, v in categorized.items()
        },
    }

    Path("artifacts").mkdir(exist_ok=True)
    with open("artifacts/quick_module_inventory.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n📝 Detailed report saved to: artifacts/quick_module_inventory.json")

    # List of key modules for Bundle 1
    print("\n⭐ KEY MODULES FOR BUNDLE 1:")
    key_modules = [
        "orchestrator",
        "neural_bus",
        "unified_memory",
        "emotion",
        "input_parser",
        "response_generator",
        "auto_learner",
        "cognitive_pipeline",
        "local_async_bus",
    ]

    all_modules = []
    for category_modules in categorized.values():
        all_modules.extend(category_modules)

    for key_name in key_modules:
        found = [m for m in all_modules if key_name in m["name"].lower()]
        if found:
            mod = found[0]
            src = "📁" if mod["source"] == "local" else "☁️"
            print(f"   ✅ {src} {mod['name']}: {mod['path']}")
        else:
            print(f"   ❌ {key_name}: NOT FOUND")


if __name__ == "__main__":
    print("🔍 Starting quick module inventory...")
    modules = find_all_modules()
    categorized = categorize_modules(modules)
    print_report(modules, categorized)
