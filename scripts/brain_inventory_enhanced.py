#!/usr/bin/env python3
"""
Inventaire intelligent des modules Jeffrey avec auto-refactor
Version améliorée avec toutes les optimisations de l'équipe
"""

import ast
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Configuration
SEARCH_PATHS = [
    Path("src/jeffrey/core"),
    Path("src/jeffrey/modules"),
    Path("/Users/davidproz/Library/Mobile Documents/com~apple~CloudDocs/Jeffrey/Jeffrey_Apps/Jeffrey_OS"),
]

# Mapping amélioré keywords -> GFC (Groupes de Fonctions Cognitives)
KEYWORDS_TO_GFC = {
    "input|parser|intent|detector|normaliz|preprocess": "perception_integration",
    "memory|cortex_memoriel|working|episodic|semantic|vector|storage|recall": "memory_associative",
    "emotion|mood|empathy|affect|valence|feeling|sentiment": "valence_emotional",
    "executive|decision|sovereign|verdict|goal|plan|strategy|judge": "executive_function",
    "response|generat|personality|style|express|conversation|output|voice": "expression_generation",
    "conscience|awareness|self|introspect|meta|healing|monitor": "metacognition",
    "orchestrat|kernel|runtime|scheduler|blackboard|bus|pipeline": "infrastructure",
    "loop|consolidat|curiosity|decay|evolution|learning|adapt": "autonomous_loops",
}


class ModuleAnalyzer:
    """Analyseur avancé de modules Python"""

    def __init__(self):
        self.stats = {
            "total_files": 0,
            "total_lines": 0,
            "real_modules": 0,
            "stubs_detected": 0,
            "errors": 0,
        }

    def analyze_file(self, filepath: Path) -> dict:
        """Analyse approfondie d'un fichier Python"""
        self.stats["total_files"] += 1

        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            lines = content.splitlines()
            self.stats["total_lines"] += len(lines)

            # Parse AST pour analyse sémantique
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                print(f"⚠️  Syntax error in {filepath.name}: {e}")
                tree = None

            # Extraction des éléments du code
            classes = []
            functions = []
            imports = []
            has_main = False

            if tree:
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        classes.append(
                            {
                                "name": node.name,
                                "methods": [m.name for m in node.body if isinstance(m, ast.FunctionDef)],
                                "lines": node.end_lineno - node.lineno if hasattr(node, "end_lineno") else 0,
                            }
                        )
                    elif isinstance(node, ast.FunctionDef):
                        if node.name not in [m["name"] for c in classes for m in c.get("methods", [])]:
                            functions.append(
                                {
                                    "name": node.name,
                                    "args": [arg.arg for arg in node.args.args],
                                    "async": isinstance(node, ast.AsyncFunctionDef),
                                }
                            )
                    elif isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.append(node.module)

                    # Détection du if __name__ == "__main__"
                    if isinstance(node, ast.If):
                        if isinstance(node.test, ast.Compare):
                            if isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__":
                                has_main = True

            # Heuristique améliorée pour détecter les stubs (merci Grok!)
            loc = len(lines)
            empty_lines = sum(1 for line in lines if not line.strip())
            code_lines = loc - empty_lines

            # Détection de stub plus précise
            is_stub = any(
                [
                    code_lines < 30,  # Très peu de code réel
                    "NotImplementedError" in content,
                    len(re.findall(r"^\s*pass\s*$", content, re.MULTILINE)) >= 3,
                    content.count("TODO") > 5,
                    content.count("FIXME") > 3,
                    (code_lines < 50 and "..." in content),
                    (len(classes) == 0 and len(functions) < 2 and code_lines < 40),
                ]
            )

            if is_stub:
                self.stats["stubs_detected"] += 1

            # Détection du GFC avec priorité
            filename_lower = filepath.name.lower()
            content_lower = content.lower()[:500]  # Check first 500 chars too

            gfc = "unknown"
            priority = 1  # Par défaut: priorité basse

            for pattern, group in KEYWORDS_TO_GFC.items():
                if re.search(pattern, filename_lower) or re.search(pattern, content_lower):
                    gfc = group
                    # Priorités ajustées (merci équipe!)
                    if any(
                        keyword in filename_lower
                        for keyword in ["conscience", "cortex", "memory_manager", "emotion_core"]
                    ):
                        priority = 5  # Cœur critique
                    elif gfc != "unknown":
                        priority = 3  # GFC connu
                    break

            # Calcul de la qualité du module
            quality_score = self._calculate_quality_score(code_lines, len(classes), len(functions), is_stub, has_main)

            # Module est une "pépite" si qualité élevée
            is_jewel = quality_score >= 70 and not is_stub
            if is_jewel:
                self.stats["real_modules"] += 1

            return {
                "path": str(filepath),
                "filename": filepath.name,
                "lines": loc,
                "code_lines": code_lines,
                "classes": classes,
                "functions": functions[:10],  # Top 10 functions
                "imports": list(set(imports))[:15],  # Top 15 imports
                "is_stub": is_stub,
                "is_jewel": is_jewel,
                "gfc": gfc,
                "priority": priority,
                "quality_score": quality_score,
                "has_main": has_main,
                "last_modified": datetime.fromtimestamp(filepath.stat().st_mtime).isoformat(),
            }

        except Exception as e:
            self.stats["errors"] += 1
            return {
                "path": str(filepath),
                "filename": filepath.name,
                "error": str(e),
                "lines": 0,
                "is_stub": True,
                "is_jewel": False,
                "gfc": "error",
                "priority": 0,
            }

    def _calculate_quality_score(
        self, code_lines: int, classes: int, functions: int, is_stub: bool, has_main: bool
    ) -> int:
        """Calcule un score de qualité pour le module"""
        if is_stub:
            return 0

        score = 0

        # Lignes de code (max 40 points)
        if code_lines >= 200:
            score += 40
        elif code_lines >= 100:
            score += 30
        elif code_lines >= 50:
            score += 20
        else:
            score += code_lines * 0.3

        # Classes (max 30 points)
        score += min(classes * 10, 30)

        # Functions (max 20 points)
        score += min(functions * 2, 20)

        # Bonus pour main (10 points)
        if has_main:
            score += 10

        return int(min(score, 100))


def find_duplicates(modules: list[dict]) -> dict[str, list[dict]]:
    """Identifie et analyse les modules dupliqués"""
    duplicates = defaultdict(list)

    for module in modules:
        if not module.get("error"):
            duplicates[module["filename"]].append(module)

    # Garder seulement les vrais duplicatas
    real_duplicates = {name: modules_list for name, modules_list in duplicates.items() if len(modules_list) > 1}

    # Analyser les duplicatas pour recommander le meilleur
    for name, modules_list in real_duplicates.items():
        # Trier par qualité et date de modification
        modules_list.sort(key=lambda m: (-m.get("quality_score", 0), m.get("last_modified", "")))

        # Marquer le meilleur
        if modules_list:
            modules_list[0]["recommended"] = True

    return real_duplicates


def auto_refactor_with_tools(modules: list[dict]):
    """Auto-refactor avec ruff et black (merci Gemini!)"""
    print("\n🔧 AUTO-REFACTOR DES MODULES")
    print("=" * 60)

    # Vérifier disponibilité des outils
    has_ruff = subprocess.run(["which", "ruff"], capture_output=True).returncode == 0
    has_black = subprocess.run(["which", "black"], capture_output=True).returncode == 0

    if not has_ruff and not has_black:
        print("⚠️  Ruff et Black non installés. Installation recommandée:")
        print("   pip install ruff black")
        return

    jewels = [m for m in modules if m.get("is_jewel")]

    for module in jewels[:10]:  # Limiter aux 10 premiers pour test
        filepath = module["path"]

        try:
            if has_ruff:
                print(f"   🔍 Ruff check: {module['filename']}")
                subprocess.run(["ruff", "check", "--fix", filepath], capture_output=True, timeout=5)

            if has_black:
                print(f"   ✨ Black format: {module['filename']}")
                subprocess.run(["black", "--quiet", filepath], capture_output=True, timeout=5)

        except subprocess.TimeoutExpired:
            print(f"   ⚠️  Timeout sur {module['filename']}")
        except Exception as e:
            print(f"   ❌ Erreur refactor {module['filename']}: {e}")


def check_gpu_availability():
    """Vérifie disponibilité GPU (merci Gemini!)"""
    try:
        import torch

        if torch.cuda.is_available():
            print(f"🎮 GPU CUDA disponible: {torch.cuda.get_device_name(0)}")
            print(f"   Mémoire: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return True
    except ImportError:
        pass

    # Check Metal Performance Shaders pour Mac M1/M2
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🎮 GPU Metal (Apple Silicon) disponible")
            return True
    except:
        pass

    print("💻 Mode CPU uniquement")
    return False


def generate_visual_report(by_gfc: dict, output_path: str = "artifacts/gfc_visualization.png"):
    """Génère un graphique de l'activité par GFC (merci Gemini!)"""
    try:
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt

        # Préparer les données
        gfc_names = list(by_gfc.keys())
        module_counts = [len(modules) for modules in by_gfc.values()]
        total_lines = [sum(m.get("code_lines", 0) for m in modules) for modules in by_gfc.values()]

        # Créer le graphique
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Graphique 1: Nombre de modules par GFC
        colors = plt.cm.viridis(range(len(gfc_names)))
        ax1.bar(range(len(gfc_names)), module_counts, color=colors)
        ax1.set_xticks(range(len(gfc_names)))
        ax1.set_xticklabels(gfc_names, rotation=45, ha="right")
        ax1.set_ylabel("Nombre de modules")
        ax1.set_title("Distribution des Modules par GFC")
        ax1.grid(axis="y", alpha=0.3)

        # Graphique 2: Lignes de code par GFC
        ax2.pie(total_lines, labels=gfc_names, autopct="%1.1f%%", colors=colors)
        ax2.set_title("Répartition du Code par GFC")

        plt.suptitle("Jeffrey OS - Architecture Neuro-Pragmatique", fontsize=16, fontweight="bold")
        plt.tight_layout()

        # Sauvegarder
        Path("artifacts").mkdir(exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\n📊 Visualisation sauvée: {output_path}")

    except ImportError:
        print("\n⚠️  matplotlib non installé pour visualisation")
        print("   pip install matplotlib")


def main():
    print("=" * 80)
    print("🧠 INVENTAIRE INTELLIGENT JEFFREY OS - VERSION ENHANCED")
    print("=" * 80)
    print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Check GPU
    print("\n🔍 DÉTECTION HARDWARE")
    print("-" * 40)
    gpu_available = check_gpu_availability()

    # Analyzer
    analyzer = ModuleAnalyzer()
    all_modules = []

    # Scanner tous les paths
    print("\n📁 SCAN DES RÉPERTOIRES")
    print("-" * 40)

    for base_path in SEARCH_PATHS:
        if not base_path.exists():
            print(f"⚠️  Path inexistant: {base_path}")
            continue

        print(f"\n🔍 Scan: {base_path}")
        py_files = list(base_path.rglob("*.py"))
        print(f"   → {len(py_files)} fichiers Python trouvés")

        # Analyser avec barre de progression simple
        for i, filepath in enumerate(py_files):
            # Ignorer les tests et caches
            if any(skip in str(filepath) for skip in ["__pycache__", "test_", "_test.py", ".pytest"]):
                continue

            # Progress indicator
            if i % 10 == 0:
                print(f"   Analyse... {i}/{len(py_files)}", end="\r")

            module_info = analyzer.analyze_file(filepath)
            all_modules.append(module_info)

    print(f"\n   ✅ Analyse terminée: {len(all_modules)} modules")

    # Trier par priorité et qualité
    all_modules.sort(key=lambda m: (-m.get("priority", 0), -m.get("quality_score", 0), -m.get("code_lines", 0)))

    # Identifier duplicatas
    duplicates = find_duplicates(all_modules)

    # Statistiques par GFC
    jewels = [m for m in all_modules if m.get("is_jewel")]
    by_gfc = defaultdict(list)
    for module in jewels:
        by_gfc[module["gfc"]].append(module)

    # Créer répertoire artifacts
    Path("artifacts").mkdir(exist_ok=True)

    # Sauvegarder JSON complet
    report_data = {
        "scan_date": datetime.now().isoformat(),
        "gpu_available": gpu_available,
        "stats": analyzer.stats,
        "total_modules": len(all_modules),
        "real_modules": len(jewels),
        "stubs_detected": analyzer.stats["stubs_detected"],
        "duplicates_count": len(duplicates),
        "by_gfc": {gfc: len(modules) for gfc, modules in by_gfc.items()},
        "modules": all_modules,
        "duplicates": {
            name: [
                {
                    "path": m["path"],
                    "quality": m.get("quality_score", 0),
                    "recommended": m.get("recommended", False),
                }
                for m in mods
            ]
            for name, mods in duplicates.items()
        },
    }

    with open("artifacts/brain_inventory.json", "w") as f:
        json.dump(report_data, f, indent=2)

    # Sauvegarder CSV des pépites pour Bundle 1
    with open("artifacts/brain_jewels_bundle1.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Filename",
                "GFC",
                "Lines",
                "Classes",
                "Functions",
                "Priority",
                "Quality",
                "Path",
                "Recommended_Order",
            ]
        )

        # Top 10 modules pour Bundle 1
        for i, module in enumerate(jewels[:10], 1):
            classes_str = "|".join([c["name"] for c in module.get("classes", [])])
            functions_str = "|".join([f["name"] for f in module.get("functions", [])[:3]])

            writer.writerow(
                [
                    module["filename"],
                    module["gfc"],
                    module["code_lines"],
                    classes_str,
                    functions_str,
                    module.get("priority", 0),
                    module.get("quality_score", 0),
                    module["path"],
                    i,  # Ordre recommandé pour Bundle 1
                ]
            )

    # Rapport console détaillé
    print("\n" + "=" * 80)
    print("📊 RÉSULTATS DE L'INVENTAIRE")
    print("=" * 80)

    print("\n📈 STATISTIQUES GLOBALES:")
    print(f"   Total fichiers analysés : {analyzer.stats['total_files']}")
    print(f"   Total lignes de code    : {analyzer.stats['total_lines']:,}")
    print(f"   Modules réels (pépites) : {len(jewels)}")
    print(f"   Stubs/Mocks détectés    : {analyzer.stats['stubs_detected']}")
    print(f"   Duplicatas trouvés      : {len(duplicates)}")
    print(f"   Erreurs d'analyse       : {analyzer.stats['errors']}")

    print("\n💎 RÉPARTITION PAR GFC (Groupes de Fonctions Cognitives):")
    print(f"   {'GFC':<30} | {'Modules':<8} | {'Lignes':<10} | {'Top Module'}")
    print("   " + "-" * 75)

    for gfc, modules in sorted(by_gfc.items(), key=lambda x: -len(x[1])):
        total_lines = sum(m.get("code_lines", 0) for m in modules)
        top_module = modules[0]["filename"] if modules else "N/A"
        print(f"   {gfc:<30} | {len(modules):<8} | {total_lines:<10,} | {top_module}")

    print("\n🎯 TOP 10 MODULES POUR BUNDLE 1:")
    print(f"   {'#':<3} | {'Module':<35} | {'GFC':<25} | {'Qualité':<8} | {'Lignes'}")
    print("   " + "-" * 85)

    for i, module in enumerate(jewels[:10], 1):
        print(
            f"   {i:<3} | {module['filename']:<35} | {module['gfc']:<25} | "
            f"{module.get('quality_score', 0):<8} | {module.get('code_lines', 0):,}"
        )

    if duplicates:
        print("\n⚠️  DUPLICATAS À RÉSOUDRE:")
        for name, modules_list in list(duplicates.items())[:5]:
            print(f"\n   📄 {name}:")
            for module in modules_list[:3]:
                recommended = "⭐ RECOMMANDÉ" if module.get("recommended") else ""
                print(f"      - {module['path']}")
                print(f"        Qualité: {module.get('quality_score', 0)} {recommended}")

    # Auto-refactor optionnel
    if "--refactor" in sys.argv:
        auto_refactor_with_tools(jewels[:10])

    # Génération visualisation
    if by_gfc:
        generate_visual_report(by_gfc)

    print("\n" + "=" * 80)
    print("✅ INVENTAIRE TERMINÉ")
    print("   → JSON complet : artifacts/brain_inventory.json")
    print("   → Bundle 1 CSV : artifacts/brain_jewels_bundle1.csv")
    print("   → Visualisation : artifacts/gfc_visualization.png")
    print("=" * 80)

    return jewels, by_gfc, duplicates


if __name__ == "__main__":
    jewels, by_gfc, duplicates = main()
