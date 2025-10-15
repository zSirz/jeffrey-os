#!/bin/bash
# Continuous canary monitoring
# Checks SLO every 5 minutes and alerts if issues

INTERVAL_SECONDS=300  # 5 minutes
TIME_WINDOW_MINUTES=15

echo "🔄 Starting continuous canary monitoring..."
echo "   Checking every $((INTERVAL_SECONDS / 60)) minutes"
echo "   Time window: ${TIME_WINDOW_MINUTES} minutes"
echo "   Press Ctrl+C to stop"
echo ""

while true; do
    echo "=========================================="
    echo "⏰ $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="

    # Run SLO check
    PYTHONPATH=src python scripts/monitor_canary_slo.py
    EXIT_CODE=$?

    echo ""

    if [ $EXIT_CODE -eq 2 ]; then
        echo "🚨🚨🚨 CRITICAL VIOLATION - ROLLBACK RECOMMENDED 🚨🚨🚨"
        # Log to file for alerting
        echo "$(date): CRITICAL SLO violation detected - rollback recommended" >> logs/canary_alerts.log
        # Could trigger automatic rollback here
        # For now, just alert loudly
        echo -e "\a"  # System beep
    elif [ $EXIT_CODE -eq 1 ]; then
        echo "⚠️ Warnings detected - monitor closely"
        echo "$(date): SLO warnings detected" >> logs/canary_alerts.log
    else
        echo "✅ System healthy"
    fi

    echo ""
    echo "Next check in $((INTERVAL_SECONDS / 60)) minutes..."
    sleep $INTERVAL_SECONDS
done
