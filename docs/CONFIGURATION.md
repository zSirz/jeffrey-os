# JEFFREY OS - Configuration Guide

## Environment Variables

### Tie-Break Rules

Control anger/frustration disambiguation behavior:

**JEFFREY_TIEBREAK_ENABLED** (default: `true`)
- Enable or disable tie-break rule entirely
- Supports: `1/0`, `true/false`, `yes/no`, `on/off`
- Set to `false` to use only ML predictions
- Use as kill switch if tie-break causes issues

**JEFFREY_TIEBREAK_DELTA** (default: `0.05`)
- Threshold for "very close" anger/frustration scores
- Lower = more conservative (fewer tie-breaks)
- Recommended: `0.03` (strict) to `0.05` (balanced)

**JEFFREY_TIEBREAK_EXTENDED_DELTA** (default: `0.15`)
- Threshold for extended tie-break when emotion in top-2
- Only applies if anger OR frustration is top-2
- Recommended: `0.10` to `0.15`

### Examples

```bash
# Conservative (strict tie-break)
export JEFFREY_TIEBREAK_ENABLED=true
export JEFFREY_TIEBREAK_DELTA=0.03
export JEFFREY_TIEBREAK_EXTENDED_DELTA=0.10

# Balanced (recommended for production)
export JEFFREY_TIEBREAK_ENABLED=true
export JEFFREY_TIEBREAK_DELTA=0.05
export JEFFREY_TIEBREAK_EXTENDED_DELTA=0.15

# Disabled (ML only)
export JEFFREY_TIEBREAK_ENABLED=false

# Kill switch for emergency rollback
export JEFFREY_TIEBREAK_ENABLED=false  # Instant disable without code change
```

### Monitoring

Check tie-break usage:

```bash
PYTHONPATH=src python scripts/analyze_monitoring_logs.py | grep "tiebreak"
```

**Target rate:** 1-2% of predictions
**Alert if:** >3% (may indicate over-application)

## Schema Versions

### Log Schema V2

Current monitoring logs use schema version 2 with these fields:

```json
{
  "schema_version": 2,
  "timestamp": "2025-10-13T14:30:00.123456",
  "version": "2.4.2",
  "route": "linear_head",
  "encoder": "intfloat/multilingual-e5-large",
  "text_preview": "This is [EMAIL] with [URL]",
  "prediction": {
    "primary": "anger",
    "confidence": 0.543,
    "top2": [["anger", 0.543], ["frustration", 0.298]],
    "low_confidence": false,
    "rule_applied": "tiebreak_anger_keyword_standard"
  },
  "latency_ms": 45.2,
  "all_scores_top5": {
    "anger": 0.543,
    "frustration": 0.298,
    "sadness": 0.089,
    "neutral": 0.045,
    "joy": 0.025
  }
}
```

### Rule Types

- `tiebreak_anger_keyword_standard`: Standard tie-break (gap < 0.05)
- `tiebreak_anger_keyword_extended`: Extended tie-break (gap < 0.15, emotion in top-2)
- `null`: No rule applied

## Privacy & GDPR

### PII Redaction

Automatic redaction of:
- **Emails:** `test@example.com` → `[EMAIL]`
- **URLs:** `https://example.com` → `[URL]`
- **Phones:** `+33612345678` → `[PHONE]`
- **IPs:** `192.168.1.1` → `[IP]`

### Log Retention

- **Active logs:** `.jsonl` format (current day)
- **Archived logs:** `.jsonl.gz` format (1+ days old)
- **Retention:** 14 days total
- **Rotation:** Daily at 2 AM (via cron)

```bash
# Manual rotation
PYTHONPATH=src python scripts/rotate_logs.py
```

## Production Monitoring

### Key Metrics

- **Linear head usage:** ≥90%
- **Regex fallback:** ≤1%
- **Low confidence rate:** ≤15%
- **Tie-break rate:** 1-2% (alert if >3%)
- **P95 latency:** ≤200ms

### Real-time Monitoring

```bash
# One-shot analysis
PYTHONPATH=src python scripts/analyze_monitoring_logs.py

# Continuous monitoring
./scripts/watch_canary.sh
```

### SLO Monitoring

```bash
# Check SLO compliance (last 15 minutes)
PYTHONPATH=src python scripts/monitor_canary_slo.py

# Exit codes:
# 0 = All OK
# 1 = Warnings
# 2 = Critical (rollback recommended)
```
