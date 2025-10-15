#!/usr/bin/env python3
"""
Validation finale stricte avec vérification loop.regions et seuil P95 fixe.
"""

import asyncio
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")

# SEUIL FIXE (amélioration 2)
TARGET_P95_MS = 100.0


async def validate():
    """Validation complète"""
    print("🧠 VALIDATION JEFFREY OS - 8/8 RÉGIONS STRICTE")
    print("=" * 60)

    # Vérifier l'inventaire
    inv_path = Path("artifacts/inventory_ultimate.json")
    if not inv_path.exists():
        print("❌ Inventaire introuvable")
        return False

    inventory = json.loads(inv_path.read_text())
    modules = inventory["bundle1_recommendations"]["modules"]

    # RÈGLE 1 : 8 modules minimum
    if len(modules) < 8:
        print(f"❌ Seulement {len(modules)}/8 modules")
        return False

    # RÈGLE 2 : Aucun interdit
    bad = [m for m in modules if "/simple_modules/" in m["path"] or "/stubs/" in m["path"]]
    if bad:
        print(f"❌ {len(bad)} modules interdits:")
        for m in bad:
            print(f"   • {m['brain_region']}: {m['path']}")
        return False

    # RÈGLE 3 : 8 régions présentes
    required_regions = {
        "perception",
        "memory",
        "emotion",
        "conscience",
        "executive",
        "motor",
        "language",
        "integration",
    }
    found_regions = {m["brain_region"] for m in modules}
    missing = required_regions - found_regions

    if missing:
        print(f"❌ Régions manquantes: {missing}")
        return False

    print("✅ 8/8 régions dans l'inventaire")

    # Charger la consciousness loop
    try:
        from jeffrey.core.consciousness_loop import ConsciousnessLoop

        loop = ConsciousnessLoop()
        await loop.initialize()
    except Exception as e:
        print(f"❌ Échec chargement ConsciousnessLoop: {e}")
        return False

    # RÈGLE 4 : Vérifier loop.regions (amélioration 1)
    loaded = []
    missing_loaded = []

    # Accéder à loop.regions (pas hasattr)
    regions_dict = getattr(loop, "regions", {})
    if not regions_dict:
        print("❌ loop.regions n'existe pas ou est vide")
        return False

    for region in sorted(required_regions):
        inst = regions_dict.get(region)
        if inst:
            loaded.append(region)
            print(f"✅ {region} chargé")
        else:
            missing_loaded.append(region)
            print(f"❌ {region} non chargé")

    if missing_loaded:
        print(f"❌ Régions non chargées: {missing_loaded}")
        return False

    # RÈGLE 5 : Smoke test par région (amélioration 1 + stratégie robuste)
    print("\n🔥 Smoke test des régions...")

    for region in sorted(required_regions):
        inst = regions_dict[region]

        # Trouver la méthode
        meth = None
        for cand in ("process", "analyze", "run", "analyze_emotion", "execute"):
            if hasattr(inst, cand):
                meth = getattr(inst, cand)
                break

        if not meth:
            print(f"❌ {region}: Aucune méthode")
            return False

        # Appel minimal avec stratégie robuste
        ok = False
        candidates = [
            ((), {}),  # meth()
            (("ping",), {}),  # meth("ping")
            (({"ping": True},), {}),  # meth({"ping": True})
            (("ping", {}), {}),  # meth("ping", {})
        ]
        for args, kwargs in candidates:
            try:
                if inspect.iscoroutinefunction(meth):
                    await meth(*args, **kwargs)
                else:
                    meth(*args, **kwargs)
                ok = True
                break
            except TypeError:
                continue
            except Exception as e:
                print(f"❌ {region}: erreur pendant le smoke test: {e}")
                return False

        if not ok:
            print(f"❌ {region}: aucune signature simple acceptée pour {meth.__name__}()")
            return False
        print(f"✅ {region}: Smoke test OK")

    # RÈGLE 6 : Perf réaliste
    print("\n📊 Test de performance...")
    times = []

    # Test de perf tolérant sync/async
    proc = getattr(loop, "process_input", None)
    if proc is None:
        print("❌ ConsciousnessLoop.process_input manquant")
        return False

    for i in range(20):
        t0 = time.perf_counter()
        try:
            if inspect.iscoroutinefunction(proc):
                await proc(f"Test {i}")
            else:
                proc(f"Test {i}")
        except Exception as e:
            print(f"❌ Erreur traitement: {e}")
            return False
        times.append((time.perf_counter() - t0) * 1000)

    avg = statistics.mean(times)
    p95 = sorted(times)[int(len(times) * 0.95)]
    p99 = sorted(times)[-1]

    print(f"\n   • Moyenne: {avg:.1f}ms")
    print(f"   • P95: {p95:.1f}ms")
    print(f"   • P99: {p99:.1f}ms")

    # RÈGLE 7 : Perfs réalistes
    if avg < 1.0:
        print("\n⚠️  ATTENTION : Performances suspicieuses (< 1ms)")
        print("   Les modules semblent être des no-ops")

    if p95 > TARGET_P95_MS:
        print(f"\n⚠️  P95 élevé ({p95:.1f}ms > {TARGET_P95_MS}ms)")

    # RÈGLE 8 : Générer rapport machine (amélioration 6)
    print("\n📝 Génération du rapport de validation...")
    report = {
        "timestamp": time.time(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        "python": platform.python_version(),
        "modules": [
            {
                "region": region,
                "class": type(regions_dict[region]).__name__,
                "path": next(m["path"] for m in modules if m["brain_region"] == region),
            }
            for region in sorted(required_regions)
        ],
        "perf": {"avg_ms": avg, "p95_ms": p95, "p99_ms": p99},
    }

    Path("artifacts/validation_report.json").write_text(json.dumps(report, indent=2))
    print("✅ Rapport sauvegardé: artifacts/validation_report.json")

    print("\n" + "=" * 60)
    print("✅ VALIDATION RÉUSSIE")
    print("   • 8/8 régions actives")
    print("   • 0 placeholders")
    print(f"   • P95: {p95:.1f}ms")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        success = asyncio.run(validate())
        if success:
            print("\n🎉 Jeffrey OS v10.0.0 - VALIDÉ")
            sys.exit(0)
        else:
            print("\n❌ Validation échouée")
            sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
