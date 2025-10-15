"""
Script de monitoring basique (Grok inspiration).
À lancer en parallèle de Jeffrey pour surveiller.
"""

import time

from jeffrey.core.emotion_backend import get_metrics


def monitor_loop(interval_seconds: int = 5):
    """Affiche métriques en continu."""
    print("🔍 Monitoring Emotion Backend...")
    print("=" * 50)

    prev_metrics = get_metrics()

    while True:
        time.sleep(interval_seconds)

        current = get_metrics()

        # Calculs
        total = current["total_predictions"]
        success = current["proto_success"]
        failures = current["proto_failures"]
        fallbacks = current["fallback_triggered"]

        # Taux
        success_rate = (success / total * 100) if total > 0 else 0
        fallback_rate = (fallbacks / total * 100) if total > 0 else 0

        # Delta depuis dernier check
        delta_total = total - prev_metrics["total_predictions"]

        # Affichage
        print(f"\n[{time.strftime('%H:%M:%S')}]")
        print(f"  Total: {total} (+{delta_total})")
        print(f"  Success: {success} ({success_rate:.1f}%)")
        print(f"  Failures: {failures}")
        print(f"  Fallbacks: {fallbacks} ({fallback_rate:.1f}%)")

        # Alertes
        if fallback_rate > 10:
            print("  ⚠️ HIGH FALLBACK RATE!")

        prev_metrics = current.copy()


if __name__ == "__main__":
    try:
        monitor_loop()
    except KeyboardInterrupt:
        print("\n✅ Monitoring stopped")
