#!/usr/bin/env python3
"""
Scanner de santé du noyau Jeffrey OS
Analyse statique pour détecter tous les problèmes d'imports
"""

import ast
import sys
from pathlib import Path


class ImportAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.imports: set[str] = set()
        self.from_imports: dict[str, list[str]] = {}

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            items = [alias.name for alias in node.names]
            self.from_imports[node.module] = items
        self.generic_visit(node)


def analyze_file(filepath: Path) -> tuple[bool, str, set[str], dict[str, list[str]]]:
    """Analyse un fichier Python et retourne son état"""
    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(filepath))
        analyzer = ImportAnalyzer()
        analyzer.visit(tree)
        return True, "OK", analyzer.imports, analyzer.from_imports
    except SyntaxError as e:
        return False, f"SYNTAX_ERROR: {e}", set(), {}
    except Exception as e:
        return False, f"PARSE_ERROR: {e}", set(), {}


def scan_directory(root: Path) -> dict[str, any]:
    """Scan complet d'un répertoire"""
    results = {
        "total_files": 0,
        "syntax_errors": [],
        "parse_errors": [],
        "valid_files": [],
        "all_imports": set(),
        "all_from_imports": {},
    }

    for py_file in root.rglob("*.py"):
        if "__pycache__" in str(py_file) or "/tests/" in str(py_file):
            continue

        results["total_files"] += 1
        valid, status, imports, from_imports = analyze_file(py_file)

        rel_path = py_file.relative_to(root)

        if "SYNTAX_ERROR" in status:
            results["syntax_errors"].append((str(rel_path), status))
        elif "PARSE_ERROR" in status:
            results["parse_errors"].append((str(rel_path), status))
        else:
            results["valid_files"].append(str(rel_path))
            results["all_imports"].update(imports)
            for module, items in from_imports.items():
                results["all_from_imports"].setdefault(module, set()).update(items)

    return results


def check_module_exists(module_name: str, root: Path) -> bool:
    """Vérifie la présence sans exécuter le module"""
    try:
        parts = module_name.split(".")
        mod_path = root / "/".join(parts)
        if (mod_path.with_suffix(".py")).exists():  # module.py
            return True
        if (mod_path / "__init__.py").exists():  # package/
            return True
        import importlib.util

        sys.path.insert(0, str(root))
        spec = importlib.util.find_spec(module_name)
        return spec is not None
    except Exception:
        return False


def main():
    root = Path("src")

    print("=" * 80)
    print("JEFFREY OS - RAPPORT DE SANTÉ DU NOYAU")
    print("=" * 80)
    print()

    # Scan
    print("📊 Analyse en cours...")
    results = scan_directory(root)

    # Rapport des erreurs de syntaxe
    if results["syntax_errors"]:
        print(f"\n{'=' * 80}")
        print(f"❌ ERREURS DE SYNTAXE ({len(results['syntax_errors'])} fichiers)")
        print(f"{'=' * 80}")
        for filepath, error in results["syntax_errors"]:
            print(f"\n📄 {filepath}")
            print(f"   {error}")

    # Rapport des erreurs de parsing
    if results["parse_errors"]:
        print(f"\n{'=' * 80}")
        print(f"❌ ERREURS DE PARSING ({len(results['parse_errors'])} fichiers)")
        print(f"{'=' * 80}")
        for filepath, error in results["parse_errors"]:
            print(f"\n📄 {filepath}")
            print(f"   {error}")

    # Analyse des imports
    print(f"\n{'=' * 80}")
    print("🔍 ANALYSE DES IMPORTS")
    print(f"{'=' * 80}")

    print(f"\nFichiers analysés : {results['total_files']}")
    print(f"Fichiers valides : {len(results['valid_files'])}")
    health_score = (len(results["valid_files"]) / results["total_files"] * 100) if results["total_files"] else 0.0
    print(f"Taux de validité : {health_score:.1f}%")

    # Vérifier les imports core.*
    core_imports = {m for m in results["all_from_imports"].keys() if m.startswith("core.")}
    if core_imports:
        print(f"\n⚠️  IMPORTS 'core.*' DÉTECTÉS ({len(core_imports)})")
        for module in sorted(core_imports):
            exists = check_module_exists(module, root)
            status = "✅" if exists else "❌"
            print(f"   {status} {module}")
            if not exists:
                # Chercher un équivalent jeffrey.core.*
                jeffrey_equiv = "jeffrey." + module
                if check_module_exists(jeffrey_equiv, root):
                    print(f"      → Équivalent trouvé : {jeffrey_equiv}")

    # Vérifier les imports jeffrey.*
    jeffrey_imports = {m for m in results["all_from_imports"].keys() if m.startswith("jeffrey.")}
    missing_jeffrey = []
    for module in jeffrey_imports:
        if not check_module_exists(module, root):
            missing_jeffrey.append(module)

    if missing_jeffrey:
        print(f"\n❌ MODULES JEFFREY.* MANQUANTS ({len(missing_jeffrey)})")
        for module in sorted(missing_jeffrey):
            print(f"   • {module}")

    # Résumé final
    print(f"\n{'=' * 80}")
    print("📋 RÉSUMÉ")
    print(f"{'=' * 80}")
    print(f"✅ Fichiers valides : {len(results['valid_files'])}")
    print(f"❌ Erreurs syntaxe : {len(results['syntax_errors'])}")
    print(f"❌ Erreurs parsing : {len(results['parse_errors'])}")
    print(f"⚠️  Imports core.* : {len(core_imports)}")
    print(f"❌ Modules jeffrey.* manquants : {len(missing_jeffrey)}")

    # Score de santé
    if health_score >= 90:
        print(f"\n🟢 Score de santé : {health_score:.1f}% - EXCELLENT")
    elif health_score >= 70:
        print(f"\n🟡 Score de santé : {health_score:.1f}% - BON")
    elif health_score >= 50:
        print(f"\n🟠 Score de santé : {health_score:.1f}% - MOYEN")
    else:
        print(f"\n🔴 Score de santé : {health_score:.1f}% - CRITIQUE")

    return 0 if health_score >= 80 else 1


if __name__ == "__main__":
    sys.exit(main())
