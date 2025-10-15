"""
Orchestrator adapté pour unified models
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from unified.compat import to_signalpayload
from unified.models import SignalPayload, SignalType
from unified.models.common import CellularResponse, ErrorPayload

Handler = Callable[[SignalPayload], Any] | Callable[[SignalPayload], Awaitable[Any]]


class Orchestrator:
    """Orchestrateur principal adapté unified."""

    def __init__(self) -> None:
        self.handlers: dict[SignalType, Handler] = {}
        self.active_signals: list[SignalPayload] = []

    def register_handler(self, signal_type: SignalType, handler: Handler) -> None:
        """Enregistre un handler pour un type de signal."""
        self.handlers[signal_type] = handler

    async def process_signal(self, signal: SignalPayload | dict) -> CellularResponse:
        """Traite un signal avec le handler approprié."""
        # Compat dict legacy
        if isinstance(signal, dict):
            signal = to_signalpayload(signal)

        if not isinstance(signal, SignalPayload):
            return CellularResponse(
                request_id=getattr(signal, 'signal_id', 'unknown'),
                cell_id="orchestrator",
                status="error",
                error=ErrorPayload(error_code="INVALID_SIGNAL", message="Invalid signal format"),
                processing_time_ms=0.0,
            )

        handler = self.handlers.get(signal.signal_type)
        if not handler:
            return CellularResponse(
                request_id=signal.signal_id,
                cell_id="orchestrator",
                status="error",
                error=ErrorPayload(error_code="NO_HANDLER", message=f"No handler for type {signal.signal_type}"),
                processing_time_ms=0.0,
            )

        try:
            import time

            start = time.time()
            result = await handler(signal) if asyncio.iscoroutinefunction(handler) else handler(signal)
            elapsed = (time.time() - start) * 1000

            # Normalise en dict
            data: dict[str, Any] = result if isinstance(result, dict) else {"result": result}
            # Optionnel: message d'info dans data
            data.setdefault("message", "Signal processed")
            return CellularResponse(
                request_id=signal.signal_id,
                cell_id="orchestrator",
                status="success",
                data=data,
                processing_time_ms=elapsed,
            )
        except Exception as e:
            return CellularResponse(
                request_id=signal.signal_id,
                cell_id="orchestrator",
                status="error",
                error=ErrorPayload(error_code="HANDLER_ERROR", message=f"Handler error: {e}"),
                data={"exception": type(e).__name__},
                processing_time_ms=0.0,
            )

    def get_active_signals(self) -> list[SignalPayload]:
        return self.active_signals

    def clear_signals(self) -> None:
        self.active_signals.clear()


# Compat
DefaultOrchestrator = Orchestrator
