#!/usr/bin/env python3
"""
Analyze prediction logs from monitoring.
Provides statistics on route usage, latency, confidence distribution.
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def analyze_logs(log_file):
    """Analyze a JSONL log file."""
    log_path = Path(log_file)

    # Try to find the latest log file if pattern is provided
    if not log_path.exists():
        # Try to find latest predictions file
        log_dir = Path("logs/predictions")
        if log_dir.exists():
            log_files = list(log_dir.glob("predictions_*.jsonl"))
            if log_files:
                log_path = max(log_files, key=lambda p: p.stat().st_mtime)
                print(f"📁 Using latest log file: {log_path}")
            else:
                print(f"❌ No prediction log files found in {log_dir}")
                return
        else:
            print(f"❌ Log file not found: {log_file}")
            return

    predictions = []
    with open(log_path, encoding='utf-8') as f:
        for line in f:
            try:
                predictions.append(json.loads(line))
            except Exception:
                continue

    if not predictions:
        print("❌ No predictions found in log file")
        return

    print(f"📊 PREDICTION ANALYSIS - {len(predictions)} predictions")
    print("=" * 80)
    print()

    # Route distribution
    routes = Counter(p["route"] for p in predictions)
    print("🛣️  Route Distribution:")
    for route, count in routes.most_common():
        pct = 100 * count / len(predictions)
        print(f"   {route:15s}: {count:5d} ({pct:5.1f}%)")
    print()

    # Latency stats
    latencies = [p["latency_ms"] for p in predictions]
    if latencies:
        print("⚡ Latency Statistics (ms):")
        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)
        print(f"   P50: {sorted_latencies[n // 2]:.1f}")
        print(f"   P95: {sorted_latencies[int(n * 0.95)]:.1f}")
        print(f"   P99: {sorted_latencies[int(n * 0.99)]:.1f}")
        print(f"   Max: {max(latencies):.1f}")
        print(f"   Mean: {sum(latencies) / len(latencies):.1f}")
        print()

    # Confidence distribution
    confidences = [p["prediction"]["confidence"] for p in predictions]
    low_conf_count = sum(1 for c in confidences if c < 0.4)
    print("🎯 Confidence Distribution:")
    print(f"   Mean: {sum(confidences) / len(confidences):.3f}")
    print(f"   Low confidence (<0.4): {low_conf_count} ({100 * low_conf_count / len(predictions):.1f}%)")
    print()

    # Emotion distribution
    emotions = Counter(p["prediction"]["primary"] for p in predictions)
    print("😊 Emotion Distribution:")
    for emotion, count in emotions.most_common():
        pct = 100 * count / len(predictions)
        print(f"   {emotion:12s}: {count:5d} ({pct:5.1f}%)")
    print()

    # Encoder distribution
    encoders = Counter(p.get("encoder", "unknown") for p in predictions)
    print("🤖 Encoder Distribution:")
    for encoder, count in encoders.most_common():
        pct = 100 * count / len(predictions)
        print(f"   {encoder:30s}: {count:5d} ({pct:5.1f}%)")
    print()

    # Low confidence cases by route
    print("⚠️  Low Confidence by Route:")
    for route in routes:
        route_preds = [p for p in predictions if p["route"] == route]
        if route_preds:
            low_conf_route = sum(1 for p in route_preds if p["prediction"]["confidence"] < 0.4)
            pct = 100 * low_conf_route / len(route_preds)
            print(f"   {route:15s}: {low_conf_route:3d}/{len(route_preds):3d} ({pct:5.1f}%)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", default="logs/predictions/predictions_latest.jsonl", help="Path to log file")
    args = parser.parse_args()
    analyze_logs(args.log_file)
