import ast
import json
import os
from collections import defaultdict
from pathlib import Path

# Configuration
PROJECT_ROOT = Path.cwd()
EXCLUDE_DIRS = {".git", "venv", ".venv", "__pycache__", "build", "dist", ".mypy_cache", ".pytest_cache", "node_modules"}

RESULTS = {
    "stubs_detected": [],
    "missing_imports": [],
    "partial_implementations": [],
    "real_implementations": [],
    "metrics": defaultdict(int),
}


def analyze_file(filepath):
    """Analyse un fichier Python pour détecter stubs/implémentations réelles"""
    try:
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            content = f.read()
            tree = ast.parse(content)

        file_info = {
            "path": str(filepath.relative_to(PROJECT_ROOT)),
            "total_lines": len(content.split('\n')),
            "functions": [],
            "classes": [],
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_info = analyze_function(node, content)
                file_info["functions"].append(func_info)
            elif isinstance(node, ast.ClassDef):
                class_info = analyze_class(node, content)
                file_info["classes"].append(class_info)

        # Classification
        if is_stub_file(file_info):
            RESULTS["stubs_detected"].append(file_info)
        elif is_partial(file_info):
            RESULTS["partial_implementations"].append(file_info)
        else:
            RESULTS["real_implementations"].append(file_info)

    except Exception as e:
        RESULTS["metrics"]["errors"] += 1
        print(f"❌ Erreur analyse {filepath}: {e}")


def analyze_function(node, content):
    """Analyse une fonction pour déterminer si c'est un stub"""
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    func_lines = content.split('\n')[start:end]
    body_src = '\n'.join(func_lines).strip()

    # Détection docstring seule
    only_docstring = (
        len(node.body) == 1
        and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, (ast.Str, ast.Constant))
    )

    # Patterns stub améliorés
    patterns_stub = (
        "pass" in body_src
        and len(func_lines) <= 3
        or "return None" in body_src
        and len(func_lines) <= 2
        or "raise NotImplementedError" in body_src
        or "TODO" in body_src
        or "FIXME" in body_src
        or body_src in {"", '"""', "'''"}
        or only_docstring
    )

    return {
        "name": node.name,
        "line": node.lineno,
        "is_stub": patterns_stub,
        "lines": len(func_lines),
        "body_preview": body_src[:200],
    }


def analyze_class(node, content):
    """Analyse une classe"""
    methods = [m for m in node.body if isinstance(m, ast.FunctionDef)]
    stub_methods = sum(1 for m in methods if analyze_function(m, content)["is_stub"])

    return {
        "name": node.name,
        "line": node.lineno,
        "total_methods": len(methods),
        "stub_methods": stub_methods,
        "completion": ((len(methods) - stub_methods) / len(methods) * 100) if methods else 100,
    }


def is_stub_file(file_info):
    """Détermine si le fichier est majoritairement un stub"""
    total_funcs = len(file_info["functions"])
    if total_funcs < 3:  # Pas assez de fonctions pour être considéré comme stub
        return False

    stub_funcs = sum(1 for f in file_info["functions"] if f["is_stub"])
    return (stub_funcs / total_funcs) >= 0.8  # Ratio plus exigeant


def is_partial(file_info):
    """Détermine si le fichier est partiellement implémenté"""
    total_funcs = len(file_info["functions"])
    if total_funcs == 0:
        return False

    stub_funcs = sum(1 for f in file_info["functions"] if f["is_stub"])
    ratio = stub_funcs / total_funcs
    return 0.3 < ratio < 0.8


# Scan du projet avec exclusions
print("🔍 Scanning Jeffrey OS project...")
print(f"📁 Racine: {PROJECT_ROOT}")
print(f"🚫 Exclusions: {EXCLUDE_DIRS}")

for root, dirs, files in os.walk(PROJECT_ROOT):
    # Exclure les dossiers indésirables
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

    for name in files:
        if name.endswith(".py"):
            analyze_file(Path(root) / name)

# Génération du rapport
print("\n📊 RÉSULTATS DU DIAGNOSTIC\n")
print(f"✅ Implémentations réelles : {len(RESULTS['real_implementations'])}")
print(f"⚠️  Implémentations partielles : {len(RESULTS['partial_implementations'])}")
print(f"🚨 Stubs détectés : {len(RESULTS['stubs_detected'])}")
print(f"❌ Erreurs d'analyse : {RESULTS['metrics']['errors']}")

print("\n🚨 FICHIERS STUBS À REMPLACER :")
for stub in RESULTS["stubs_detected"]:
    print(f"  - {stub['path']} ({stub['total_lines']} lignes)")

print("\n⚠️  FICHIERS PARTIELS À COMPLÉTER :")
for partial in RESULTS["partial_implementations"]:
    print(f"  - {partial['path']} ({partial['total_lines']} lignes)")

# Top stubs par taille
top_stubs = sorted(RESULTS["stubs_detected"], key=lambda x: x["total_lines"], reverse=True)[:10]
print("\n🏁 Top stubs (par taille) :")
for s in top_stubs:
    print(f"  - {s['path']} ({s['total_lines']} lignes)")

# Sauvegarde JSON
with open("diagnostic_jeffrey_results.json", "w", encoding='utf-8') as f:
    json.dump(RESULTS, f, indent=2, default=str, ensure_ascii=False)

print("\n💾 Rapport détaillé sauvegardé : diagnostic_jeffrey_results.json")
