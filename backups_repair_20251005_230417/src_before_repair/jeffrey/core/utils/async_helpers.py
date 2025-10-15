"""
Helpers pour gérer l'async/sync et les timeouts adaptatifs
"""

import asyncio
import inspect
import logging
import time
from collections.abc import Callable
from typing import Any

import psutil

logger = logging.getLogger(__name__)


async def asyncify(
    func: Callable,
    *args,
    timeout: float = 2.0,
    adaptive: bool = True,
    to_thread: bool = True,
    **kwargs,
) -> Any:
    """
    Convertit n'importe quelle fonction en async avec timeout
    Gère automatiquement sync/async et adapte le timeout selon la charge
    """
    # Adapter le timeout selon la charge système
    if adaptive:
        timeout = get_adaptive_timeout(timeout)

    async def _call():
        if inspect.iscoroutinefunction(func):
            # Déjà async
            return await func(*args, **kwargs)
        elif to_thread:
            # Sync -> thread pour ne pas bloquer
            return await asyncio.to_thread(func, *args, **kwargs)
        else:
            # Sync direct (rapide)
            return func(*args, **kwargs)

    try:
        return await asyncio.wait_for(_call(), timeout=timeout)
    except TimeoutError:
        logger.warning(f"Timeout after {timeout}s for {func.__name__}")
        return None


def get_adaptive_timeout(base_timeout: float) -> float:
    """
    Calcule un timeout adaptatif basé sur la charge système
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_percent = psutil.virtual_memory().percent

        # Si système chargé, augmenter le timeout
        if cpu_percent > 80 or memory_percent > 80:
            return base_timeout * 2.0
        elif cpu_percent > 60 or memory_percent > 70:
            return base_timeout * 1.5
        else:
            return base_timeout
    except:
        return base_timeout


class LatencyBudget:
    """Budget de latence basé sur perf_counter (plus robuste)"""

    def __init__(self, total_ms: float):
        self.total_ms = total_ms
        self.start_time = time.perf_counter()  # Plus robuste
        self.consumed_ms = 0

    def remaining_ms(self) -> float:
        """Retourne le budget restant en ms"""
        elapsed = (time.perf_counter() - self.start_time) * 1000
        return max(0.0, self.total_ms - elapsed)

    def has_budget(self, min_ms: float = 10) -> bool:
        """Vérifie s'il reste assez de budget"""
        return self.remaining_ms() > min_ms
