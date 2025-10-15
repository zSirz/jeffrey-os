#!/usr/bin/env python3
"""
Real-time SLO monitoring for canary deployment
Checks critical metrics and alerts if thresholds exceeded

GPT Improvement: Handles timestamp 'Z' format
"""

import json
import logging
import os
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

# ========== SLO THRESHOLDS (Configurable via env vars) ==========
# Seuils configurables via env (valeurs par défaut pour trafic normal)
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("SLO_LOWCONF_MAX", "15.0"))  # 15% par défaut
TIEBREAK_THRESHOLD = float(os.getenv("SLO_TIEBREAK_MAX", "2.0"))  # 2% par défaut

# Note : Pour le générateur canary local avec 20% borderline + 5% edge,
#        utiliser SLO_LOWCONF_MAX=0.30 et SLO_TIEBREAK_MAX=0.05

SLO_THRESHOLDS = {
    "linear_head_min_pct": float(os.getenv("SLO_LINEAR_MIN", "90.0")),  # Linear head should handle ≥90%
    "regex_max_pct": float(os.getenv("SLO_REGEX_MAX", "1.0")),  # Regex fallback should be ≤1%
    "low_confidence_max_pct": LOW_CONFIDENCE_THRESHOLD,  # Low confidence rate configurable
    "tiebreak_max_pct": TIEBREAK_THRESHOLD,  # Tie-break application rate configurable
    "p95_latency_max_ms": float(os.getenv("SLO_P95_MAX_MS", "200.0")),  # P95 latency ≤200ms
    "dimension_mismatch_max": int(os.getenv("SLO_DIMENSION_MAX", "0")),  # Zero dimension mismatches
}


class SLOViolation:
    """Track SLO violation details."""

    def __init__(self, metric: str, current: float, threshold: float, severity: str):
        self.metric = metric
        self.current = current
        self.threshold = threshold
        self.severity = severity  # "warning" or "critical"

    def __str__(self):
        icon = "⚠️" if self.severity == "warning" else "🚨"
        return f"{icon} {self.metric}: {self.current:.2f} (threshold: {self.threshold:.2f})"


def _parse_timestamp(ts: str) -> datetime:
    """Parse timestamp with Z format compatibility (GPT improvement)."""
    # Handle 'Z' suffix (UTC timezone)
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def check_canary_slo(time_window_minutes: int = 15) -> tuple[list[SLOViolation], bool]:
    """
    Check SLO compliance for recent predictions.

    Args:
        time_window_minutes: Check logs from last N minutes

    Returns:
        (violations, should_rollback)
    """
    logger.info(f"🔍 Checking SLO compliance (last {time_window_minutes} min)")

    # Find latest log file
    logs_dir = Path("logs/predictions")
    log_files = sorted(logs_dir.glob("predictions_*.jsonl"))

    if not log_files:
        logger.error("❌ No log files found")
        return [], False

    latest_log = log_files[-1]

    # Time threshold
    cutoff_time = datetime.now() - timedelta(minutes=time_window_minutes)

    # Collect metrics
    total_predictions = 0
    route_counts = Counter()
    low_confidence_count = 0
    tiebreak_count = 0
    latencies = []

    with open(latest_log) as f:
        for line in f:
            if not line.strip():
                continue

            try:
                entry = json.loads(line)

                # Check if within time window (GPT improvement: handle Z timestamps)
                entry_time = _parse_timestamp(entry["timestamp"])
                if entry_time < cutoff_time:
                    continue

                total_predictions += 1

                # Route
                route = entry.get("route", "unknown")
                route_counts[route] += 1

                # Low confidence
                if entry["prediction"]["low_confidence"]:
                    low_confidence_count += 1

                # Tie-break
                if entry["prediction"].get("rule_applied"):
                    tiebreak_count += 1

                # Latency
                latency = entry.get("latency_ms", 0)
                if latency > 0:
                    latencies.append(latency)

            except Exception as e:
                logger.warning(f"Error parsing entry: {e}")

    if total_predictions == 0:
        logger.warning(f"⚠️ No predictions in last {time_window_minutes} minutes")
        return [], False

    # Calculate metrics
    linear_pct = (route_counts.get("linear_head", 0) / total_predictions) * 100
    proto_pct = (route_counts.get("prototypes", 0) / total_predictions) * 100
    regex_pct = (route_counts.get("regex", 0) / total_predictions) * 100
    low_conf_pct = (low_confidence_count / total_predictions) * 100
    tiebreak_pct = (tiebreak_count / total_predictions) * 100

    # Latency P95
    if latencies:
        latencies_sorted = sorted(latencies)
        p95_idx = int(len(latencies) * 0.95)
        p95_latency = latencies_sorted[p95_idx] if p95_idx < len(latencies) else latencies_sorted[-1]
    else:
        p95_latency = 0

    # Display current metrics
    logger.info(f"\n📊 Current Metrics (last {time_window_minutes} min, n={total_predictions}):")
    logger.info(f"   Linear head:      {linear_pct:.1f}% (threshold: ≥{SLO_THRESHOLDS['linear_head_min_pct']}%)")
    logger.info(f"   Prototypes:       {proto_pct:.1f}%")
    logger.info(f"   Regex fallback:   {regex_pct:.1f}% (threshold: ≤{SLO_THRESHOLDS['regex_max_pct']}%)")
    logger.info(f"   Low confidence:   {low_conf_pct:.1f}% (threshold: ≤{SLO_THRESHOLDS['low_confidence_max_pct']}%)")
    logger.info(f"   Tie-break rate:   {tiebreak_pct:.1f}% (threshold: ≤{SLO_THRESHOLDS['tiebreak_max_pct']}%)")
    logger.info(f"   P95 latency:      {p95_latency:.1f}ms (threshold: ≤{SLO_THRESHOLDS['p95_latency_max_ms']}ms)")

    # Check violations
    # Pour trafic canary local (générateur avec borderline/edge), lancer avec :
    # SLO_LOWCONF_MAX=0.30 SLO_TIEBREAK_MAX=0.05 python scripts/monitor_canary_slo.py
    violations = []

    if linear_pct < SLO_THRESHOLDS["linear_head_min_pct"]:
        violations.append(
            SLOViolation("linear_head_usage", linear_pct, SLO_THRESHOLDS["linear_head_min_pct"], "critical")
        )

    if regex_pct > SLO_THRESHOLDS["regex_max_pct"]:
        violations.append(SLOViolation("regex_fallback", regex_pct, SLO_THRESHOLDS["regex_max_pct"], "critical"))

    if low_conf_pct > SLO_THRESHOLDS["low_confidence_max_pct"]:
        violations.append(
            SLOViolation("low_confidence_rate", low_conf_pct, SLO_THRESHOLDS["low_confidence_max_pct"], "warning")
        )

    if tiebreak_pct > SLO_THRESHOLDS["tiebreak_max_pct"]:
        violations.append(SLOViolation("tiebreak_rate", tiebreak_pct, SLO_THRESHOLDS["tiebreak_max_pct"], "warning"))

    if p95_latency > SLO_THRESHOLDS["p95_latency_max_ms"]:
        violations.append(SLOViolation("p95_latency", p95_latency, SLO_THRESHOLDS["p95_latency_max_ms"], "warning"))

    # Determine if rollback needed
    critical_violations = [v for v in violations if v.severity == "critical"]
    should_rollback = len(critical_violations) > 0

    # Display results
    if violations:
        logger.warning(f"\n⚠️ SLO VIOLATIONS DETECTED ({len(violations)}):")
        for v in violations:
            logger.warning(f"   {v}")

        if should_rollback:
            logger.error("\n🚨 CRITICAL: Automatic rollback recommended!")
    else:
        logger.info("\n✅ All SLOs within acceptable range")

    return violations, should_rollback


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    violations, should_rollback = check_canary_slo(time_window_minutes=15)

    # Exit code: 0 = OK, 1 = warnings, 2 = critical (rollback)
    if should_rollback:
        sys.exit(2)
    elif violations:
        sys.exit(1)
    else:
        sys.exit(0)
