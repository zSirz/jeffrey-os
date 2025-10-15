"""
Runtime monitoring for Jeffrey OS emotion detection.
Logs predictions in structured JSON format for observability.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def _redact_pii(text: str) -> str:
    """
    Redact personally identifiable information from text.

    Complies with GDPR requirements by removing:
    - URLs (http/https/www)
    - Email addresses
    - Phone numbers (FR/international formats)
    - IP addresses (IPv4)

    Args:
        text: Original text

    Returns:
        Text with PII redacted
    """
    # Redact URLs (http/https/www)
    text = re.sub(r'https?://\S+', '[URL]', text)
    text = re.sub(r'www\.\S+', '[URL]', text)

    # Redact email addresses
    text = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[EMAIL]', text)

    # Redact phone numbers (conservative pattern to avoid dates/IDs)
    # GPT improvement: more conservative regex supporting dots and extended patterns
    text = re.sub(r'(?<!\w)(?:\+?\d{1,3}[\s()./-]?)?(?:\d[\s()./-]?){8,}\d(?!\w)', '[PHONE]', text)

    # Redact IPv4 addresses
    text = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '[IP]', text)

    # Redact IPv6 addresses (compressed & full forms)
    text = re.sub(r'\b(?:[A-Fa-f0-9]{1,4}:){2,7}[A-Fa-f0-9]{0,4}\b', '[IPV6]', text)

    return text


class PredictionMonitor:
    """Monitor and log prediction metrics in JSON format."""

    def __init__(self, log_dir="logs/predictions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create daily log file
        today = datetime.now().strftime("%Y-%m-%d")
        self.log_file = self.log_dir / f"predictions_{today}.jsonl"

        logger.info(f"📊 Prediction monitoring enabled: {self.log_file}")

    def log_prediction(
        self,
        text: str,
        route: str,  # "linear_head" | "prototypes" | "regex"
        primary_emotion: str,
        confidence: float,
        all_scores: dict,
        latency_ms: float | None = None,
        encoder_name: str | None = None,
        version: str = "2.4.2",
        low_confidence: bool = False,  # nouveau
        rule_applied: str | None = None,  # nouveau
    ):
        """
        Log a prediction event in JSON format.

        Args:
            text: Input text (truncated for privacy)
            route: Which prediction route was used
            primary_emotion: Predicted primary emotion
            confidence: Confidence score [0-1]
            all_scores: Dict of all emotion scores
            latency_ms: Prediction latency in milliseconds
            encoder_name: Name of encoder used
            version: System version
            low_confidence: Whether prediction has low confidence
            rule_applied: Name of any rule that was applied
        """
        # Redact PII before truncating
        redacted_text = _redact_pii(text)
        preview = redacted_text[:50]  # Truncate after redaction

        # Get top-2 emotions
        sorted_emotions = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)
        top2 = sorted_emotions[:2] if len(sorted_emotions) >= 2 else sorted_emotions

        # Get top-5 emotions for detailed analysis
        all_scores_top5 = dict(sorted_emotions[:5])

        # GPT improvement: simplified low_confidence calculation
        low_confidence = bool(low_confidence) or (confidence < 0.4)

        # Build log entry
        log_entry = {
            "schema_version": 2,  # Version du schéma
            "timestamp": datetime.now().isoformat(),
            "version": version,
            "route": route,
            "encoder": encoder_name or "unknown",
            "text_preview": preview,  # Now with PII redacted
            "prediction": {
                "primary": primary_emotion,
                "confidence": round(confidence, 3),
                "top2": [(e, round(s, 3)) for e, s in top2],
                "low_confidence": low_confidence,
                "rule_applied": rule_applied,  # New field
            },
            "latency_ms": round(latency_ms, 2) if latency_ms else 0.0,
            "all_scores_top5": {k: round(v, 3) for k, v in all_scores_top5.items()},
        }

        # Write to JSONL file (one JSON object per line)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"Failed to write monitoring log: {e}")

    def _get_log_file(self):
        """Get current log file path (for testing)."""
        return self.log_file


# Global monitor instance
_monitor = None


def get_monitor():
    """Get or create global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor()
    return _monitor
