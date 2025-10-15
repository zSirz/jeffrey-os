#!/usr/bin/env python3
"""
Intègre toutes les améliorations Phase 2.3
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main():
    print("🚀 JEFFREY OS - Phase 2.3 Integration")
    print("=" * 50)

    # 1. Import avec vérification dépendances
    try:
        from jeffrey.core.loops.loop_manager import LoopManager
        from jeffrey.core.ml.memory_clusterer import AdaptiveMemoryClusterer
        from jeffrey.core.monitoring.entropy_guardian import EntropyGuardian
        from jeffrey.utils.test_helpers import DummyMemoryFederation, NullBus, SimpleState
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Installing optional dependencies...")
        import subprocess

        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements-optional.txt"])
        print("Please restart the script")
        return

    # 2. Initialisation avec monitoring
    manager = LoopManager(
        cognitive_core=SimpleState(),
        emotion_orchestrator=None,
        memory_federation=DummyMemoryFederation(),
        bus=NullBus(),
        mode_getter=lambda: "normal",
        latency_budget_ok=lambda: True,
    )
    clusterer = AdaptiveMemoryClusterer()
    guardian = EntropyGuardian()

    # 3. Démarrer manager
    print("\n📊 Starting LoopManager with monitoring...")
    await manager.start()

    # 4. Monitoring loop (60 secondes)
    start_time = time.time()
    monitoring_duration = 60  # seconds

    while time.time() - start_time < monitoring_duration:
        # Métriques
        metrics = manager.get_metrics()

        # Check entropie sur Q-tables
        for name, loop in manager.loops.items():
            if hasattr(loop, "q_table"):
                entropy = guardian.check_entropy(f"{name}_qtable", loop.q_table)

        # Affichage
        print(f"\n⏱️  Time: {time.time() - start_time:.0f}s")
        print(f"📊 Symbiosis: {metrics['system']['symbiosis_score']:.3f}")
        print(f"🔄 Cycles: {metrics['system']['total_cycles']}")
        print(f"❌ Dropped: {metrics['system']['bus_dropped']}")

        # Afficher alertes biais
        if guardian.bias_alerts:
            print(f"⚠️  Bias alerts: {len(guardian.bias_alerts)}")

        # Recommandations
        recs = guardian.get_recommendations()
        if recs:
            print("💡 Recommendations:")
            for rec in recs[:3]:  # Top 3
                print(f"   - {rec}")

        await asyncio.sleep(5)

    # 5. Arrêt propre
    print("\n🛑 Stopping...")
    await manager.stop()

    # 6. Rapport final
    print("\n" + "=" * 50)
    print("✅ PHASE 2.3 VALIDATION COMPLETE")
    print("=" * 50)

    # Vérifications
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    checks = [
        ("Replay buffer saved", (data_dir / "replay_buffer.pkl").exists()),
        ("Symbiosis > 0.5", metrics["system"]["symbiosis_score"] > 0.5),
        (
            "Bus drops < 1%",
            metrics["system"]["bus_dropped"] < metrics["system"]["total_cycles"] * 0.01,
        ),
        ("No critical bias", not any(a["risk"] == "high" for a in guardian.bias_alerts)),
    ]

    all_passed = True
    for check_name, passed in checks:
        status = "✅" if passed else "❌"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\n🎉 READY FOR PRODUCTION!")
        print("Next: git tag -a v2.4.2 -m 'Production ready with ML'")
    else:
        print("\n⚠️  Some checks failed, review before tagging")


if __name__ == "__main__":
    asyncio.run(main())
