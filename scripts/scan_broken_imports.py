#!/usr/bin/env python3
"""
Scanner de diagnostic pour imports cassés dans Jeffrey OS
Mission: Identifier les erreurs d'imports sans les réparer
"""

import ast
import importlib.util
import os
from collections import Counter


def scan_python_files(root_dir):
    """Scan récursif des fichiers Python"""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        # Skip __pycache__ et .git
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]

        for file in files:
            if file.endswith(".py"):
                python_files.append(os.path.join(root, file))
    return python_files


def parse_imports(file_path):
    """Parse les imports d'un fichier Python"""
    imports = []
    try:
        with open(file_path, encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content, filename=file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({"type": "import", "module": alias.name, "line": node.lineno})
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(
                        {
                            "type": "from_import",
                            "module": module,
                            "name": alias.name,
                            "line": node.lineno,
                            "level": node.level,
                        }
                    )

    except SyntaxError as e:
        return None, f"SyntaxError: {e}"
    except Exception as e:
        return None, f"ParseError: {e}"

    return imports, None


def test_import(module_name, from_name=None):
    """Test si un import fonctionne"""
    try:
        if from_name:
            module = importlib.import_module(module_name)
            if from_name == "*":
                return True  # On peut pas vraiment tester *
            getattr(module, from_name)
        else:
            importlib.import_module(module_name)
        return True
    except:
        return False


def is_external_dependency(module_name):
    """Détermine si c'est une dépendance externe (non-Jeffrey)"""
    external_deps = {
        "numpy",
        "pandas",
        "torch",
        "sklearn",
        "requests",
        "flask",
        "sqlite3",
        "json",
        "os",
        "sys",
        "pathlib",
        "datetime",
        "time",
        "typing",
        "abc",
        "dataclasses",
        "asyncio",
        "threading",
        "logging",
        "unittest",
        "pytest",
        "click",
        "yaml",
        "pydantic",
        "fastapi",
        "transformers",
        "openai",
        "anthropic",
    }

    # Module stdlib ou externe connu
    if module_name.split(".")[0] in external_deps:
        return True

    # Pas un module Jeffrey
    if not (module_name.startswith("jeffrey") or module_name.startswith("src")):
        return True

    return False


def main():
    print("=== DIAGNOSTIC IMPORTS CASSÉS JEFFREY OS ===\n")

    # Stats globales
    total_files = 0
    syntax_errors = 0
    broken_imports = 0
    missing_modules = Counter()
    broken_details = []

    # Scan du répertoire src/
    src_dir = "src"
    if not os.path.exists(src_dir):
        print(f"❌ Répertoire {src_dir} introuvable")
        return

    print(f"📁 Scan de {src_dir}/...")
    python_files = scan_python_files(src_dir)
    print(f"📊 {len(python_files)} fichiers Python trouvés\n")

    for file_path in python_files:
        total_files += 1
        rel_path = os.path.relpath(file_path)

        # Parse les imports
        imports, error = parse_imports(file_path)

        if error:
            print(f"💥 SYNTAX ERROR: {rel_path}")
            print(f"   └─ {error}\n")
            syntax_errors += 1
            continue

        # Test chaque import
        file_has_errors = False
        for imp in imports:
            module = imp["module"]
            line = imp["line"]

            # Skip imports relatifs vides ou externes
            if not module or is_external_dependency(module):
                continue

            # Test l'import
            if imp["type"] == "import":
                success = test_import(module)
                if not success:
                    if not file_has_errors:
                        print(f"🔴 BROKEN IMPORTS: {rel_path}")
                        file_has_errors = True
                    print(f"   └─ L{line}: import {module}")
                    broken_imports += 1
                    missing_modules[module] += 1
                    broken_details.append({"file": rel_path, "line": line, "type": "import", "module": module})

            elif imp["type"] == "from_import":
                from_name = imp["name"]
                success = test_import(module, from_name)
                if not success:
                    if not file_has_errors:
                        print(f"🔴 BROKEN IMPORTS: {rel_path}")
                        file_has_errors = True
                    print(f"   └─ L{line}: from {module} import {from_name}")
                    broken_imports += 1
                    missing_modules[f"{module}.{from_name}"] += 1
                    broken_details.append(
                        {
                            "file": rel_path,
                            "line": line,
                            "type": "from_import",
                            "module": module,
                            "name": from_name,
                        }
                    )

        if file_has_errors:
            print()

    # Rapport final
    print("=" * 60)
    print("📊 RAPPORT DE DIAGNOSTIC")
    print("=" * 60)
    print(f"📁 Fichiers scannés: {total_files}")
    print(f"💥 Erreurs de syntaxe: {syntax_errors}")
    print(f"🔴 Imports cassés (hors deps externes): {broken_imports}")
    print()

    if missing_modules:
        print("🏆 TOP 5 des modules manquants les plus utilisés:")
        for i, (module, count) in enumerate(missing_modules.most_common(5), 1):
            print(f"   {i}. {module} ({count} références)")

    print("\n⚠️  DIAGNOSTIC TERMINÉ - AUCUNE RÉPARATION EFFECTUÉE")


if __name__ == "__main__":
    main()
