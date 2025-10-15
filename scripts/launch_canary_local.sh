#!/bin/bash
# Launch local canary test with traffic generation

echo "🚀 JEFFREY OS - LOCAL CANARY TEST"
echo "=================================="
echo ""

# Configuration
export PYTHONPATH=src
export JEFFREY_TIEBREAK_ENABLED=true
export JEFFREY_TIEBREAK_DELTA=0.05
export JEFFREY_TIEBREAK_EXTENDED_DELTA=0.15

# Traffic parameters - SAFE DEFAULTS for initial run
export TRAFFIC_RATE_PER_MIN=20      # 20 events/min = ~28k/day
export BURST_EVERY=120              # Burst every 2 min
export BURST_SIZE=5                 # 5 extra events per burst
export JITTER_MS=300                # 300ms jitter
export PII_MAX_PER_MIN=2            # Max 2 PII samples per minute
export MAX_EVENTS=200               # 200 events for smoke run (~10 min)

echo "📊 Configuration:"
echo "   Rate: ${TRAFFIC_RATE_PER_MIN} events/min"
echo "   Burst: ${BURST_SIZE} events every ${BURST_EVERY}s"
echo "   PII limit: ${PII_MAX_PER_MIN}/min"
echo "   Max events: ${MAX_EVENTS} (smoke run)"
echo "   Tie-break: ENABLED (delta=${JEFFREY_TIEBREAK_DELTA}, extended=${JEFFREY_TIEBREAK_EXTENDED_DELTA})"
echo ""
echo "💡 This is a SMOKE RUN (200 events, ~10 min)"
echo "   After verifying SLO, remove MAX_EVENTS for 24h run:"
echo "   unset MAX_EVENTS"
echo ""
echo "📊 Monitoring commands (run in another terminal):"
echo "   PYTHONPATH=src python scripts/monitor_canary_slo.py"
echo "   python scripts/analyze_monitoring_logs.py | head -50"
echo ""
echo "🛑 Stop with Ctrl+C"
echo ""

# Launch traffic generator
python scripts/generate_canary_traffic.py
