"""
Gates de ressources et sanitizer pour privacy
"""

import copy
import re
from collections.abc import Callable
from typing import Any

# Import psutil optionnel (évite crash si non installé)
try:
    import psutil
except Exception:
    psutil = None


def create_budget_gate(mode_getter: Callable[[], str], latency_budget_ok: Callable[[], bool]) -> Callable[[], bool]:
    """
    Crée une gate qui vérifie si on peut exécuter
    """

    def _gate() -> bool:
        # Vérifier le mode
        mode = mode_getter()
        if mode != "normal":
            return False

        # Vérifier le budget latence
        if not latency_budget_ok():
            return False

        # Vérifier la charge système (Mac spécifique)
        if psutil is not None:
            try:
                # Non-bloquant avec interval=0.0
                cpu_percent = psutil.cpu_percent(interval=0.0)
                memory_percent = psutil.virtual_memory().percent

                # Si CPU > 80% ou RAM > 85%, bloquer
                if cpu_percent > 80 or memory_percent > 85:
                    return False
            except Exception:
                pass

        return True

    return _gate


def sanitize_event_data(data: dict[str, Any]) -> dict[str, Any]:
    """
    Sanitize les données avant publication (anti-PII)
    """
    sanitized = copy.deepcopy(data)

    def _sanitize_recursive(obj):
        if isinstance(obj, str):
            # Masquer emails
            obj = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}", "[EMAIL]", obj)
            # Masquer téléphones
            obj = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", obj)
            # Masquer SSN
            obj = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[SSN]", obj)
            # Masquer clés API (format générique)
            obj = re.sub(r"\b[A-Za-z0-9]{32,}\b", "[API_KEY]", obj)
            # Masquer IPs
            obj = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]", obj)
            return obj
        elif isinstance(obj, dict):
            # Nettoyer récursivement
            cleaned = {}
            for k, v in obj.items():
                # Masquer les clés sensibles
                if any(sensitive in k.lower() for sensitive in ["password", "token", "secret", "api_key", "private"]):
                    cleaned[k] = "[REDACTED]"
                else:
                    cleaned[k] = _sanitize_recursive(v)
            return cleaned
        elif isinstance(obj, list):
            return [_sanitize_recursive(item) for item in obj]
        else:
            return obj

    return _sanitize_recursive(sanitized)
