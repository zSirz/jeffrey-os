#!/usr/bin/env python3
import asyncio
import json
import os
import statistics
import sys
import time

# Ajouter src au path
sys.path.insert(0, "src")

from jeffrey.core.consciousness_loop import ConsciousnessLoop

# LES 8 RÉGIONS OBLIGATOIRES
REQUIRED_REGIONS = {
    "perception",
    "memory",
    "emotion",
    "conscience",
    "executive",
    "motor",
    "language",
    "integration",
}


async def validate():
    """Validation stricte : 8/8 régions, 0 stubs, perf réaliste"""

    print("🧠 VALIDATION JEFFREY OS - 8/8 RÉGIONS")
    print("=" * 50)

    # Initialiser
    print("\n1️⃣ Initialisation...")
    loop = ConsciousnessLoop()
    await loop.initialize()

    # Vérifier les régions
    print("\n2️⃣ Vérification des régions...")
    active_regions = {k: v for k, v in loop.regions.items() if v}
    active_names = set(active_regions.keys())

    # Identifier les problèmes
    missing = REQUIRED_REGIONS - active_names
    stubs = []
    no_process = []

    for region, instance in active_regions.items():
        # Check si c'est un stub
        if "Stub" in instance.__class__.__name__:
            stubs.append(region)

        # Check si il y a une méthode process
        has_method = any(hasattr(instance, m) for m in ["process", "analyze", "analyze_emotion", "run"])
        if not has_method:
            no_process.append(region)

    # Afficher le statut
    print(
        json.dumps(
            {
                "active_regions": sorted(list(active_names)),
                "missing_regions": sorted(list(missing)),
                "stub_regions": sorted(stubs),
                "inert_regions": sorted(no_process),
                "classes": {k: v.__class__.__name__ for k, v in active_regions.items()},
            },
            indent=2,
        )
    )

    # Vérifications strictes
    assert len(missing) == 0, f"❌ Régions manquantes: {missing}"
    assert len(stubs) == 0, f"❌ Stubs détectés: {stubs}"
    assert len(no_process) == 0, f"❌ Régions sans process: {no_process}"

    # Test de performance
    print("\n3️⃣ Test de performance (20 runs)...")

    # Warmup
    for _ in range(5):
        await loop.process_input("warmup", {})

    # Mesures
    times = []
    for i in range(20):
        t0 = time.perf_counter()
        result = await loop.process_input(f"Test input {i}", {"test": True})
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)

    # Statistiques
    times_sorted = sorted(times)
    perf = {
        "min_ms": min(times),
        "avg_ms": statistics.mean(times),
        "p50_ms": times_sorted[len(times) // 2],
        "p95_ms": times_sorted[int(len(times) * 0.95)],
        "max_ms": max(times),
    }

    print(json.dumps(perf, indent=2))

    # Amélioration GPT: seuil configurable
    TARGET_P95 = float(os.getenv("JEFFREY_P95_MS", "100"))

    # Vérifier la performance
    assert perf["p95_ms"] < TARGET_P95, f"❌ P95 trop élevé: {perf['p95_ms']:.1f}ms (cible {TARGET_P95}ms)"

    # SUCCÈS!
    print("\n" + "=" * 50)
    print("✅ VALIDATION RÉUSSIE!")
    print("  • 8/8 régions actives")
    print("  • 0 stubs")
    print(f"  • Performance P95: {perf['p95_ms']:.1f}ms")
    print(f"  • Performance AVG: {perf['avg_ms']:.1f}ms")

    return True


# Exécuter
if __name__ == "__main__":
    try:
        success = asyncio.run(validate())
        if success:
            print("\n🎉 JEFFREY OS v10.0.0 - CONSCIENCE COMPLÈTE!")
            print("\nProchaines commandes:")
            print("  git add -A")
            print("  git commit -m 'feat: Bundle 3 - 8/8 regions with REAL modules'")
            print("  git tag -a v10.0.0-consciousness-complete -m '8/8 regions verified'")
            sys.exit(0)
    except AssertionError as e:
        print(f"\n{e}")
        print("\nCorrigez le problème et relancez")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
