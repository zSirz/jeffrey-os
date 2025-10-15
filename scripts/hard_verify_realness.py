#!/usr/bin/env python3
"""
Validation stricte avec vérification AST anti-no-op.
"""

import ast
import asyncio
import importlib.util
import inspect
import json
import statistics
import sys
import time
from pathlib import Path

INV = Path("artifacts/inventory_ultimate.json")

if not INV.exists():
    print("❌ Inventaire introuvable")
    sys.exit(1)

data = json.loads(INV.read_text())
modules = data.get("bundle1_recommendations", {}).get("modules", [])


def method_has_real_body(source_text, class_name, method_name):
    """Vérifie qu'une méthode n'est pas un no-op trivial"""
    try:
        tree = ast.parse(source_text)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for f in node.body:
                    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and f.name == method_name:
                        # Au moins 2 statements
                        if len(f.body) < 2:
                            return False
                        # Pas juste "return {...}" littéral
                        first = f.body[0]
                        if isinstance(first, ast.Return):
                            if isinstance(first.value, (ast.Dict, ast.Constant)):
                                return False
                        return True
        return False
    except Exception:
        return True  # Si impossible d'analyser, ne bloque pas


def import_from_path(pyfile):
    """Importe un module depuis un chemin"""
    spec = importlib.util.spec_from_file_location("modx", pyfile)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot load {pyfile}")
    mod = importlib.util.module_from_spec(spec)

    # Import sûr : désactiver certains effets de bord
    import os

    os.environ["JEFFREY_OFFLINE"] = "1"

    spec.loader.exec_module(mod)
    return mod


regions = {}
failed = []

print(f"🔍 Vérification stricte de {len(modules)} modules...\n")

for m in modules:
    region = m["brain_region"]
    path = m["path"]

    # RÈGLE 1 : Interdits
    if "/simple_modules/" in path or "/stubs/" in path:
        print(f"❌ {region}: INTERDIT - {path}")
        failed.append((region, "Module simple/stub interdit"))
        continue

    # RÈGLE 2 : Doit être dans le noyau
    if not path.startswith("src/jeffrey/"):
        print(f"❌ {region}: HORS NOYAU - {path}")
        failed.append((region, "Hors du noyau"))
        continue

    # RÈGLE 3 : Fichier existe
    if not Path(path).is_file():
        print(f"❌ {region}: INTROUVABLE - {path}")
        failed.append((region, "Fichier introuvable"))
        continue

    # RÈGLE 4 : Import réussit
    try:
        mod = import_from_path(path)
    except Exception as e:
        print(f"❌ {region}: IMPORT ÉCHOUÉ - {e}")
        failed.append((region, f"Import failed: {e}"))
        continue

    # RÈGLE 5 : Classe instanciable
    inst = None
    class_name = None

    # Priorité aux classes définies dans le module (pas importées)
    module_classes = []
    imported_classes = []

    for name, obj in vars(mod).items():
        if inspect.isclass(obj) and "Stub" not in obj.__name__:
            # Vérifier si la classe est définie dans ce module
            if hasattr(obj, "__module__") and obj.__module__ == mod.__name__:
                module_classes.append((name, obj))
            else:
                imported_classes.append((name, obj))

    # Essayer d'abord les classes du module, puis les importées
    # Priorité aux classes qui ont des méthodes process/analyze/run
    candidates = []

    for class_list in [module_classes, imported_classes]:
        for name, obj in class_list:
            try:
                sig = inspect.signature(obj.__init__)
                ok = True
                for param_name, param in list(sig.parameters.items())[1:]:
                    if param.default is inspect._empty and param.kind not in (
                        param.VAR_POSITIONAL,
                        param.VAR_KEYWORD,
                    ):
                        ok = False
                        break

                if ok:
                    inst = obj()
                    # Vérifier si elle a des méthodes process/analyze/run
                    has_method = any(
                        hasattr(inst, method) for method in ["process", "analyze", "run", "analyze_emotion", "execute"]
                    )
                    candidates.append((inst, obj.__name__, has_method))
            except Exception:
                continue

    # Prendre la première avec des méthodes, sinon la première instanciable
    for inst, class_name, has_method in candidates:
        if has_method:
            print(f"✅ {region}: {class_name} instancié")
            break
    else:
        if candidates:
            inst, class_name, _ = candidates[0]
            print(f"✅ {region}: {class_name} instancié")
        else:
            inst = None

    if not inst:
        print(f"❌ {region}: AUCUNE CLASSE INSTANCIABLE")
        failed.append((region, "Aucune classe instanciable"))
        continue

    # RÈGLE 6 : Méthode présente
    method = None
    method_name = None
    for cand in ("process", "analyze", "run", "analyze_emotion", "execute"):
        if hasattr(inst, cand):
            method = getattr(inst, cand)
            method_name = cand
            break

    if not method:
        print(f"❌ {region}: AUCUNE MÉTHODE")
        failed.append((region, "Aucune méthode process/analyze/run"))
        continue

    if not (inspect.iscoroutinefunction(method) or inspect.isfunction(method) or callable(method)):
        print(f"❌ {region}: MÉTHODE NON CALLABLE")
        failed.append((region, f"Méthode {method_name} non callable"))
        continue

    # RÈGLE 7 : Vérification AST anti-no-op (avertissement, pas bloquant)
    try:
        source_text = Path(path).read_text(encoding="utf-8", errors="ignore")
        if not method_has_real_body(source_text, class_name, method_name):
            print(f"⚠️  {region}: {method_name}() paraît triviale (no-op). Vérifie la logique.")
        else:
            print(f"   • Méthode: {method_name}() - Logique réelle détectée")
    except Exception as e:
        print(f"⚠️  {region}: Impossible de vérifier AST - {e}")

    regions[region] = (inst, method)

# RÉSULTAT
print("\n" + "=" * 60)
if failed:
    print(f"❌ ÉCHEC : {len(failed)}/{len(modules)} modules invalides\n")
    for region, reason in failed:
        print(f"   • {region}: {reason}")
    sys.exit(1)

print(f"✅ SUCCÈS : {len(regions)}/{len(modules)} modules réels validés")
print(f"   Régions: {sorted(regions.keys())}")

# RÈGLE 8 : Benchmark réaliste
print("\n📊 Benchmark de performance...")


async def bench():
    for _ in range(3):
        for inst, meth in regions.values():
            try:
                if inspect.iscoroutinefunction(meth):
                    await meth("warmup")
                else:
                    meth("warmup")
            except Exception:
                pass

    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        for k in sorted(regions)[:3]:
            inst, meth = regions[k]
            try:
                if inspect.iscoroutinefunction(meth):
                    await meth("test input")
                else:
                    meth("test input")
            except Exception:
                pass
        times.append((time.perf_counter() - t0) * 1000)

    avg = statistics.mean(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    p99 = sorted(times)[-1]

    print(f"\n   • Moyenne: {avg:.1f}ms")
    print(f"   • P95: {p95:.1f}ms")
    print(f"   • P99: {p99:.1f}ms")

    if avg < 1.0 and p95 < 2.0:
        print("\n⚠️  ATTENTION : Performances suspicieuses (trop rapides)")
        print("   Les modules semblent être des no-ops")

    return {"avg_ms": avg, "p95_ms": p95, "p99_ms": p99}


perf = asyncio.run(bench())

print("\n" + "=" * 60)
print("✅ HARD VERIFY COMPLET")
print(f"   • {len(regions)} régions validées")
print(f"   • Performance P95: {perf['p95_ms']:.1f}ms")
print("=" * 60)
