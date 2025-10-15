"""
PII Redactor - Protection des données personnelles
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


class PIIRedactor:
    """Redactor pour masquer les informations sensibles dans les logs"""

    # Patterns PII
    PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
        "ip": r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b",
        "cc": r"\b(?:\d[ -]?){13,16}\b",
        "api_key": r"[A-Za-z0-9]{32,}",
        "password": r'(?i)(password|pwd|passwd|pass)["\']?\s*[:=]\s*["\']?[^"\s]+',
    }

    @classmethod
    def redact_text(cls, text: str) -> str:
        """Masque les PII dans du texte"""
        if not text:
            return text

        result = text
        for pattern_name, pattern in cls.PATTERNS.items():
            result = re.sub(pattern, f"[{pattern_name.upper()}_REDACTED]", result)

        return result

    @classmethod
    def redact_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Masque les PII dans un dictionnaire"""
        if not data:
            return data

        result = {}
        sensitive_keys = ["password", "secret", "token", "key", "auth", "credential"]

        for key, value in data.items():
            # Masquer les clés sensibles
            if any(sk in key.lower() for sk in sensitive_keys):
                result[key] = "[REDACTED]"
            elif isinstance(value, str):
                result[key] = cls.redact_text(value)
            elif isinstance(value, dict):
                result[key] = cls.redact_dict(value)
            elif isinstance(value, list):
                result[key] = cls.redact_list(value)
            else:
                result[key] = value

        return result

    @classmethod
    def redact_list(cls, data: list[Any]) -> list[Any]:
        """Masque les PII dans une liste"""
        result = []
        for item in data:
            if isinstance(item, str):
                result.append(cls.redact_text(item))
            elif isinstance(item, dict):
                result.append(cls.redact_dict(item))
            elif isinstance(item, list):
                result.append(cls.redact_list(item))
            else:
                result.append(item)
        return result
