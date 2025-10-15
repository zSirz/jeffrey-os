"""Global privacy filters with international PII patterns"""

import logging
import os
import re
import time  # GPT fix: missing import
from typing import Any

logger = logging.getLogger(__name__)


class GlobalPrivacyFilter:
    """Privacy filter with international coverage"""

    # International PII patterns
    PII_PATTERNS = {
        # Credit cards (global)
        "credit_card": re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        # US Social Security
        "ssn_us": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        # French Social Security
        "ssn_fr": re.compile(r"\b[12]\d{2}[01]\d[0-9]{2}\d{3}\d{3}\d{2}\b"),
        # China ID Card
        "id_cn": re.compile(r"\b\d{17}[\dXx]\b"),
        # India Aadhaar
        "aadhaar_in": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
        # UK National Insurance
        "ni_uk": re.compile(r"\b[A-Z]{2}\d{6}[A-Z]\b"),
        # Email (global)
        "email": re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
        # Phone numbers (flexible)
        "phone": re.compile(r"\b(?:\+\d{1,3}[\s-]?)?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,4}\b"),
        # IP addresses
        "ip_address": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
        # IBAN (international bank)
        "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
        # Passport (generic)
        "passport": re.compile(r"\b[A-Z0-9]{6,9}\b"),
    }

    REDACTION_TOKEN = "[REDACTED]"

    def __init__(self):
        self.enabled = os.getenv("JEFFREY_REDACT_PII", "false").lower() == "true"
        self.mode = os.getenv("JEFFREY_REDACT_MODE", "replace")  # replace or report
        self.redaction_log = []

    def redact(self, text: str, context: str = "unknown") -> str:
        """Redact PII from text"""
        if not self.enabled or not text:
            return text

        redacted = text
        redaction_count = {}

        for pattern_name, pattern in self.PII_PATTERNS.items():
            matches = pattern.findall(redacted)
            if matches:
                redaction_count[pattern_name] = len(matches)

                if self.mode == "replace":
                    redacted = pattern.sub(self.REDACTION_TOKEN, redacted)

                # Log for compliance
                self.redaction_log.append(
                    {
                        "timestamp": time.time(),
                        "context": context,
                        "pattern": pattern_name,
                        "count": len(matches),
                    }
                )

        if redaction_count:
            total = sum(redaction_count.values())
            logger.info(f"Redacted {total} PII items: {redaction_count}")

        return redacted

    def get_compliance_report(self) -> dict[str, Any]:
        """Get GDPR compliance report"""
        return {
            "total_redactions": len(self.redaction_log),
            "patterns_triggered": list(set(log["pattern"] for log in self.redaction_log)),
            "redaction_summary": self.redaction_log[-10:],  # Last 10
        }

    def clear_log(self):
        """Clear redaction log (for GDPR)"""
        self.redaction_log.clear()


# Global instance
privacy_filter = GlobalPrivacyFilter()
redact = privacy_filter.redact
