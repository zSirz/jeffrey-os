import ast
import json
import os
from collections import defaultdict
from pathlib import Path

# Configuration - focus uniquement sur src/jeffrey
PROJECT_ROOT = Path.cwd()
JEFFREY_SRC = PROJECT_ROOT / "src" / "jeffrey"
EXCLUDE_DIRS = {".git", "venv", ".venv", "__pycache__", "build", "dist", ".mypy_cache", ".pytest_cache", "node_modules"}

RESULTS = {
    "stubs_detected": [],
    "missing_imports": [],
    "partial_implementations": [],
    "real_implementations": [],
    "broken_files": [],
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
            "imports": [],
        }

        # Analyser les imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    file_info["imports"].append(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        file_info["imports"].append(alias.name)

        # Analyser fonctions et classes
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

        RESULTS["metrics"]["analyzed"] += 1

    except SyntaxError as e:
        RESULTS["broken_files"].append(
            {
                "path": str(filepath.relative_to(PROJECT_ROOT)),
                "error": f"SyntaxError: {e}",
                "line": getattr(e, 'lineno', 'unknown'),
            }
        )
        RESULTS["metrics"]["syntax_errors"] += 1
    except Exception:
        RESULTS["metrics"]["other_errors"] += 1


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
        and len(func_lines) <= 4
        or "return None" in body_src
        and len(func_lines) <= 3
        or "raise NotImplementedError" in body_src
        or "TODO" in body_src
        or "FIXME" in body_src
        or only_docstring
        or body_src.count('\n') <= 2
        and 'pass' in body_src
    )

    return {
        "name": node.name,
        "line": node.lineno,
        "is_stub": patterns_stub,
        "lines": len(func_lines),
        "body_preview": body_src[:100],
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
    if total_funcs < 2:  # Fichiers très petits
        return False

    stub_funcs = sum(1 for f in file_info["functions"] if f["is_stub"])

    # Si plus de 80% de stubs ET au moins 3 fonctions
    if total_funcs >= 3:
        return (stub_funcs / total_funcs) >= 0.8

    return False


def is_partial(file_info):
    """Détermine si le fichier est partiellement implémenté"""
    total_funcs = len(file_info["functions"])
    if total_funcs == 0:
        return False

    stub_funcs = sum(1 for f in file_info["functions"] if f["is_stub"])
    ratio = stub_funcs / total_funcs
    return 0.3 <= ratio < 0.8


# Scan focalisé sur Jeffrey src uniquement
print("🎯 DIAGNOSTIC FOCALISÉ - CODE SOURCE JEFFREY")
print(f"📁 Analyse: {JEFFREY_SRC}")
print("=" * 60)

if not JEFFREY_SRC.exists():
    print(f"❌ Dossier Jeffrey non trouvé: {JEFFREY_SRC}")
    exit(1)

for root, dirs, files in os.walk(JEFFREY_SRC):
    # Exclure les dossiers indésirables
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

    for name in files:
        if name.endswith(".py"):
            analyze_file(Path(root) / name)

# Génération du rapport
print(f"\n📊 RÉSULTATS JEFFREY OS (Analysé: {RESULTS['metrics']['analyzed']} fichiers)")
print("=" * 60)
print(f"✅ Implémentations réelles : {len(RESULTS['real_implementations'])}")
print(f"⚠️  Implémentations partielles : {len(RESULTS['partial_implementations'])}")
print(f"🚨 Stubs détectés : {len(RESULTS['stubs_detected'])}")
print(f"💥 Fichiers cassés (syntaxe) : {len(RESULTS['broken_files'])}")

if RESULTS["broken_files"]:
    print("\n💥 FICHIERS AVEC ERREURS SYNTAXE:")
    for broken in RESULTS["broken_files"][:10]:  # Top 10
        print(f"  - {broken['path']} (ligne {broken['line']}): {broken['error']}")

print("\n🚨 STUBS JEFFREY À REMPLACER:")
jeffrey_stubs = [s for s in RESULTS["stubs_detected"] if "src/jeffrey" in s["path"]]
for stub in jeffrey_stubs:
    print(f"  - {stub['path']} ({stub['total_lines']} lignes)")

print("\n⚠️  PARTIELS JEFFREY À COMPLÉTER:")
jeffrey_partials = [p for p in RESULTS["partial_implementations"] if "src/jeffrey" in p["path"]]
for partial in jeffrey_partials:
    print(f"  - {partial['path']} ({partial['total_lines']} lignes)")

print("\n✅ TOP IMPLÉMENTATIONS RÉELLES:")
jeffrey_real = [r for r in RESULTS["real_implementations"] if "src/jeffrey" in r["path"]]
top_real = sorted(jeffrey_real, key=lambda x: x["total_lines"], reverse=True)[:10]
for real in top_real:
    classes = ", ".join([c["name"] for c in real["classes"]])
    print(f"  - {real['path']} ({real['total_lines']} lignes) - Classes: {classes}")

# Analyse des modules critiques
print("\n🎯 MODULES CRITIQUES JEFFREY:")
critical_paths = [
    "core/emotions/core/jeffrey_emotional_core.py",
    "core/orchestration/agi_orchestrator.py",
    "core/memory_systems.py",
    "core/self_learning.py",
    "core/orchestration/dialogue_engine.py",
]

for critical in critical_paths:
    found = False
    for impl in RESULTS["real_implementations"]:
        if critical in impl["path"]:
            status = "✅ RÉEL"
            found = True
            break
    for impl in RESULTS["partial_implementations"]:
        if critical in impl["path"]:
            status = "⚠️ PARTIEL"
            found = True
            break
    for impl in RESULTS["stubs_detected"]:
        if critical in impl["path"]:
            status = "🚨 STUB"
            found = True
            break
    for broken in RESULTS["broken_files"]:
        if critical in broken["path"]:
            status = "💥 CASSÉ"
            found = True
            break

    if not found:
        status = "❌ MANQUANT"

    print(f"  {status} {critical}")

# Sauvegarde JSON
with open("diagnostic_jeffrey_focused.json", "w", encoding='utf-8') as f:
    json.dump(RESULTS, f, indent=2, default=str, ensure_ascii=False)

print("\n💾 Rapport détaillé sauvegardé : diagnostic_jeffrey_focused.json")
print("🧪 Import test: python verify_imports.py")
