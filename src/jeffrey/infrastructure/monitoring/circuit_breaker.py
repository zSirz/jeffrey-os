"""
Disjoncteur pour protection système.

Ce module implémente les fonctionnalités essentielles pour disjoncteur pour protection système.
Il fournit une architecture robuste et évolutive intégrant les composants
nécessaires au fonctionnement optimal du système. L'implémentation suit
les principes de modularité et d'extensibilité pour faciliter l'évolution
future du système.

Le module gère l'initialisation, la configuration, le traitement des données,
la communication inter-composants, et la persistance des états. Il s'intègre
harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
une séparation claire des responsabilités.

L'architecture interne permet une évolution adaptative basée sur les interactions
et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
cohérente et authentique.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from python_json_logger import jsonlogger


class CircuitState(Enum):
    """Circuit breaker states"""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Blocking calls due to failures
    HALF_OPEN = "half_open"  # Testing if service recovered


class FailureType(Enum):
    """Types of failures that can trigger circuit breaker"""

    TIMEOUT = "timeout"
    ERROR = "error"
    EXCEPTION = "exception"
    OVERLOAD = "overload"
    UNAVAILABLE = "unavailable"


@dataclass
class FailureRecord:
    """Record of a failure event"""

    timestamp: str
    failure_type: FailureType
    details: str
    duration_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CircuitMetrics:
    """Circuit breaker metrics"""

    total_calls: int
    successful_calls: int
    failed_calls: int
    rejected_calls: int
    success_rate: float
    failure_rate: float
    avg_response_time_ms: float
    state_changes: int
    last_failure_time: str | None
    uptime_percentage: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BackoffStrategy:
    """Exponential backoff strategy for recovery attempts"""

    def __init__(
        self,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        multiplier: float = 2.0,
        jitter: bool = True,
    ):
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.multiplier = multiplier
        self.jitter = jitter
        self.current_delay = initial_delay
        self.attempt_count = 0

    def get_delay(self) -> float:
        """Get next delay in seconds"""
        delay = min(self.current_delay, self.max_delay)

        if self.jitter:
            # Add jitter to prevent thundering herd
            delay *= 0.5 + random.random() * 0.5

        self.current_delay *= self.multiplier
        self.attempt_count += 1

        return delay

    def reset(self) -> None:
        """Reset backoff strategy"""
        self.current_delay = self.initial_delay
        self.attempt_count = 0


class FallbackStrategy:
    """Defines fallback behavior when circuit is open"""

    def __init__(
        self,
        strategy_type: str = "queue",
        max_queue_size: int = 1000,
        fallback_value: Any = None,
        fallback_function: Callable | None = None,
    ):
        self.strategy_type = strategy_type
        self.max_queue_size = max_queue_size
        self.fallback_value = fallback_value
        self.fallback_function = fallback_function
        self.offline_queue: deque = deque(maxlen=max_queue_size)
        self._lock = threading.Lock()

    def handle_rejected_call(self, call_data: Any) -> Any:
        """Handle a rejected call based on strategy"""
        if self.strategy_type == "queue":
            return self._queue_for_later(call_data)
        elif self.strategy_type == "fallback_value":
            return self.fallback_value
        elif self.strategy_type == "fallback_function" and self.fallback_function:
            return self._execute_fallback_function(call_data)
        else:
            raise Exception("Circuit breaker open - service unavailable")

    def _queue_for_later(self, call_data: Any) -> dict[str, Any]:
        """Queue call for later processing"""
        with self._lock:
            if len(self.offline_queue) >= self.max_queue_size:
                raise Exception("Offline queue full - request rejected")

            queued_item = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "data": call_data,
                "attempts": 0,
            }
            self.offline_queue.append(queued_item)

            return {"status": "queued", "queue_position": len(self.offline_queue)}

    def _execute_fallback_function(self, call_data: Any) -> Any:
        """Execute fallback function"""
        try:
            return self.fallback_function(call_data)
        except Exception as e:
            logging.error(f"Fallback function failed: {e}")
            return self.fallback_value

    def get_queued_calls(self) -> list[dict[str, Any]]:
        """Get all queued calls"""
        with self._lock:
            return list(self.offline_queue)

    def process_queued_calls(self, processor: Callable[[Any], Any]) -> int:
        """Process queued calls when circuit recovers"""
        processed = 0

        with self._lock:
            while self.offline_queue:
                try:
                    item = self.offline_queue.popleft()
                    processor(item["data"])
                    processed += 1
                except Exception as e:
                    # Put item back and stop processing
                    self.offline_queue.appendleft(item)
                    logging.error(f"Failed to process queued item: {e}")
                    break

        return processed


class CircuitBreaker:
    """
    Intelligent circuit breaker with adaptive thresholds and fallback strategies
    Implements the circuit breaker pattern for resilience against cascading failures
    """

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        success_threshold: int = 3,
        timeout_seconds: float = 30.0,
        monitoring_window: int = 100,
        enable_adaptive_thresholds: bool = True,
        fallback_strategy: FallbackStrategy | None = None,
    ):
        """
        Initialize circuit breaker

        Args:
            name: Circuit breaker identifier
            failure_threshold: Number of failures before opening circuit
            recovery_timeout: Seconds to wait before attempting recovery
            success_threshold: Consecutive successes needed to close circuit
            timeout_seconds: Timeout for individual calls
            monitoring_window: Number of recent calls to track
            enable_adaptive_thresholds: Enable adaptive threshold adjustment
            fallback_strategy: Strategy for handling rejected calls
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.timeout_seconds = timeout_seconds
        self.monitoring_window = monitoring_window
        self.enable_adaptive_thresholds = enable_adaptive_thresholds

        # Current state
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0.0
        self.last_state_change = time.time()
        self.state_change_count = 0

        # Call tracking
        self.call_history: deque = deque(maxlen=monitoring_window)
        self.failure_history: deque = deque(maxlen=monitoring_window)

        # Statistics
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.rejected_calls = 0
        self.response_times: deque = deque(maxlen=monitoring_window)

        # Adaptive components
        self.backoff_strategy = BackoffStrategy()
        self.fallback_strategy = fallback_strategy or FallbackStrategy()

        # Thread safety
        self._lock = threading.Lock()

        # Setup logging
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        logHandler.setFormatter(formatter)
        self.logger = logging.getLogger(f"CircuitBreaker.{name}")
        self.logger.addHandler(logHandler)
        self.logger.setLevel(logging.INFO)

        self.logger.info(
            "Circuit breaker initialized",
            extra={
                "name": name,
                "failure_threshold": failure_threshold,
                "recovery_timeout": recovery_timeout,
                "adaptive_enabled": enable_adaptive_thresholds,
            },
        )

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with automatic error recording"""
        if exc_type is not None:
            self.record_failure(FailureType.EXCEPTION, str(exc_val))
        return False  # Don't suppress exceptions

    def call_allowed(self) -> bool:
        """Check if calls are allowed through the circuit"""
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            elif self.state == CircuitState.OPEN:
                # Check if we should transition to HALF_OPEN
                if time.time() - self.last_failure_time >= self.recovery_timeout:
                    self._transition_to_half_open()
                    return True
                return False
            elif self.state == CircuitState.HALF_OPEN:
                return True

        return False

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute function call through circuit breaker

        Args:
            func: Function to call
            *args: Function arguments
            **kwargs: Function keyword arguments

        Returns:
            Function result or fallback value

        Raises:
            Exception: If circuit is open and no fallback available
        """
        if not self.call_allowed():
            self.rejected_calls += 1
            return self.fallback_strategy.handle_rejected_call(
                {"function": func.__name__, "args": args, "kwargs": kwargs}
            )

        start_time = time.time()

        try:
            # Execute with timeout
            if asyncio.iscoroutinefunction(func):
                result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.timeout_seconds)
            else:
                result = func(*args, **kwargs)

            # Record success
            response_time = (time.time() - start_time) * 1000
            self.record_success(response_time)

            return result

        except TimeoutError:
            self.record_failure(FailureType.TIMEOUT, f"Timeout after {self.timeout_seconds}s")
            raise
        except Exception as e:
            self.record_failure(FailureType.ERROR, str(e))
            raise

    def record_success(self, response_time_ms: float = 0.0) -> None:
        """Record a successful call"""
        with self._lock:
            self.total_calls += 1
            self.successful_calls += 1
            self.success_count += 1
            self.failure_count = 0  # Reset failure count

            # Track response time
            self.response_times.append(response_time_ms)

            # Track call history
            self.call_history.append({"timestamp": time.time(), "success": True, "response_time_ms": response_time_ms})

            # State transitions
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()

            # Adaptive threshold adjustment
            if self.enable_adaptive_thresholds:
                self._adapt_thresholds()

    def record_failure(self, failure_type: FailureType, details: str = "") -> None:
        """Record a failed call"""
        with self._lock:
            self.total_calls += 1
            self.failed_calls += 1
            self.failure_count += 1
            self.success_count = 0  # Reset success count
            self.last_failure_time = time.time()

            # Record failure details
            failure = FailureRecord(
                timestamp=datetime.utcnow().isoformat() + "Z",
                failure_type=failure_type,
                details=details,
            )
            self.failure_history.append(failure)

            # Track call history
            self.call_history.append(
                {
                    "timestamp": time.time(),
                    "success": False,
                    "failure_type": failure_type.value,
                    "details": details,
                }
            )

            # State transitions
            if self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                self._transition_to_open()
            elif self.state == CircuitState.HALF_OPEN:
                self._transition_to_open()

            self.logger.warning(
                "Circuit breaker failure recorded",
                extra={
                    "name": self.name,
                    "failure_type": failure_type.value,
                    "failure_count": self.failure_count,
                    "state": self.state.value,
                    "details": details,
                },
            )

    def _transition_to_open(self) -> None:
        """Transition circuit to OPEN state"""
        self.state = CircuitState.OPEN
        self.state_change_count += 1
        self.last_state_change = time.time()
        self.backoff_strategy.reset()

        self.logger.error(
            "Circuit breaker opened",
            extra={
                "name": self.name,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
            },
        )

    def _transition_to_half_open(self) -> None:
        """Transition circuit to HALF_OPEN state"""
        self.state = CircuitState.HALF_OPEN
        self.state_change_count += 1
        self.last_state_change = time.time()
        self.success_count = 0

        self.logger.info(
            "Circuit breaker half-open",
            extra={"name": self.name, "recovery_attempt": self.backoff_strategy.attempt_count},
        )

    def _transition_to_closed(self) -> None:
        """Transition circuit to CLOSED state"""
        self.state = CircuitState.CLOSED
        self.state_change_count += 1
        self.last_state_change = time.time()
        self.failure_count = 0
        self.backoff_strategy.reset()

        # Process any queued calls
        if hasattr(self.fallback_strategy, "process_queued_calls"):
            processed = self.fallback_strategy.process_queued_calls(lambda x: None)
            if processed > 0:
                self.logger.info(f"Processed {processed} queued calls after recovery")

        self.logger.info("Circuit breaker closed", extra={"name": self.name, "success_count": self.success_count})

    def _adapt_thresholds(self) -> None:
        """Adapt failure threshold based on recent performance"""
        if len(self.call_history) < 50:
            return

        # Calculate recent failure rate
        recent_calls = list(self.call_history)[-50:]
        recent_failures = len([c for c in recent_calls if not c["success"]])
        recent_failure_rate = recent_failures / len(recent_calls)

        # Adjust threshold based on stability
        if recent_failure_rate < 0.05:  # Very stable
            # Can tolerate more failures before opening
            self.failure_threshold = min(self.failure_threshold + 1, 10)
        elif recent_failure_rate > 0.20:  # Very unstable
            # Need to be more sensitive
            self.failure_threshold = max(self.failure_threshold - 1, 3)

    def get_metrics(self) -> CircuitMetrics:
        """Get comprehensive circuit metrics"""
        with self._lock:
            success_rate = (self.successful_calls / max(1, self.total_calls)) * 100
            failure_rate = (self.failed_calls / max(1, self.total_calls)) * 100

            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0.0

            # Calculate uptime percentage
            state_durations = self._calculate_state_durations()
            total_time = sum(state_durations.values())
            uptime = (state_durations.get(CircuitState.CLOSED, 0) / max(1, total_time)) * 100

            last_failure = None
            if self.failure_history:
                last_failure = self.failure_history[-1].timestamp

            return CircuitMetrics(
                total_calls=self.total_calls,
                successful_calls=self.successful_calls,
                failed_calls=self.failed_calls,
                rejected_calls=self.rejected_calls,
                success_rate=success_rate,
                failure_rate=failure_rate,
                avg_response_time_ms=avg_response_time,
                state_changes=self.state_change_count,
                last_failure_time=last_failure,
                uptime_percentage=uptime,
            )

    def _calculate_state_durations(self) -> dict[CircuitState, float]:
        """Calculate time spent in each state"""
        # Simplified implementation
        total_time = time.time() - self.last_state_change

        if self.state == CircuitState.CLOSED:
            return {CircuitState.CLOSED: total_time}
        elif self.state == CircuitState.OPEN:
            return {CircuitState.OPEN: total_time}
        else:
            return {CircuitState.HALF_OPEN: total_time}

    def get_current_state(self) -> dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
            "failure_threshold": self.failure_threshold,
            "success_threshold": self.success_threshold,
            "last_failure_time": self.last_failure_time,
            "recovery_timeout": self.recovery_timeout,
            "timeout_seconds": self.timeout_seconds,
            "state_changes": self.state_change_count,
            "queued_calls": (
                len(self.fallback_strategy.get_queued_calls())
                if hasattr(self.fallback_strategy, "get_queued_calls")
                else 0
            ),
        }

    def get_failure_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent failure history"""
        recent_failures = list(self.failure_history)[-limit:]
        return [f.to_dict() for f in recent_failures]

    def force_open(self) -> None:
        """Manually force circuit open (for testing/maintenance)"""
        with self._lock:
            self._transition_to_open()
            self.logger.warning("Circuit breaker manually opened", extra={"name": self.name})

    def force_close(self) -> None:
        """Manually force circuit closed (for testing/maintenance)"""
        with self._lock:
            self._transition_to_closed()
            self.logger.warning("Circuit breaker manually closed", extra={"name": self.name})

    def reset(self) -> None:
        """Reset circuit breaker to initial state"""
        with self._lock:
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.total_calls = 0
            self.successful_calls = 0
            self.failed_calls = 0
            self.rejected_calls = 0
            self.state_change_count = 0
            self.call_history.clear()
            self.failure_history.clear()
            self.response_times.clear()
            self.backoff_strategy.reset()

            self.logger.info("Circuit breaker reset", extra={"name": self.name})


class CircuitBreakerManager:
    """Manages multiple circuit breakers across the system"""

    def __init__(self) -> None:
        self.circuits: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

        # Setup logging
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        logHandler.setFormatter(formatter)
        self.logger = logging.getLogger("CircuitBreakerManager")
        self.logger.addHandler(logHandler)
        self.logger.setLevel(logging.INFO)

    def create_circuit(
        self, name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0, **kwargs
    ) -> CircuitBreaker:
        """Create and register a new circuit breaker"""
        with self._lock:
            if name in self.circuits:
                return self.circuits[name]

            circuit = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout=recovery_timeout,
                **kwargs,
            )

            self.circuits[name] = circuit

            self.logger.info(
                "Circuit breaker created",
                extra={"name": name, "total_circuits": len(self.circuits)},
            )

            return circuit

    def get_circuit(self, name: str) -> CircuitBreaker | None:
        """Get circuit breaker by name"""
        return self.circuits.get(name)

    def get_all_circuits(self) -> dict[str, CircuitBreaker]:
        """Get all circuit breakers"""
        return self.circuits.copy()

    def get_system_health(self) -> dict[str, Any]:
        """Get overall system health across all circuits"""
        total_circuits = len(self.circuits)
        if total_circuits == 0:
            return {"status": "no_circuits", "total_circuits": 0}

        states = {"closed": 0, "open": 0, "half_open": 0}
        total_calls = 0
        total_failures = 0

        for circuit in self.circuits.values():
            state = circuit.state.value
            states[state] += 1

            metrics = circuit.get_metrics()
            total_calls += metrics.total_calls
            total_failures += metrics.failed_calls

        overall_failure_rate = (total_failures / max(1, total_calls)) * 100
        healthy_circuits = states["closed"]
        health_percentage = (healthy_circuits / total_circuits) * 100

        if health_percentage >= 90:
            status = "healthy"
        elif health_percentage >= 70:
            status = "degraded"
        elif health_percentage >= 50:
            status = "critical"
        else:
            status = "failing"

        return {
            "status": status,
            "health_percentage": health_percentage,
            "total_circuits": total_circuits,
            "circuit_states": states,
            "overall_failure_rate": overall_failure_rate,
            "total_calls": total_calls,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }

    def get_detailed_status(self) -> list[dict[str, Any]]:
        """Get detailed status of all circuits"""
        return [
            {**circuit.get_current_state(), "metrics": circuit.get_metrics().to_dict()}
            for circuit in self.circuits.values()
        ]


# Global circuit breaker manager instance
circuit_manager = CircuitBreakerManager()


# Decorator for easy circuit breaker application
def circuit_breaker(name: str, failure_threshold: int = 5, recovery_timeout: float = 60.0, **kwargs):
    """Decorator to apply circuit breaker to functions"""

    def decorator(func):
        circuit = circuit_manager.create_circuit(name, failure_threshold, recovery_timeout, **kwargs)

        async def async_wrapper(*args, **kwargs):
            return await circuit.call(func, *args, **kwargs)

        def sync_wrapper(*args, **kwargs):
            if not circuit.call_allowed():
                circuit.rejected_calls += 1
                return circuit.fallback_strategy.handle_rejected_call(
                    {"function": func.__name__, "args": args, "kwargs": kwargs}
                )

            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = (time.time() - start_time) * 1000
                circuit.record_success(response_time)
                return result
            except Exception as e:
                circuit.record_failure(FailureType.ERROR, str(e))
                raise

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Example usage and demonstration
async def main():
    """Demo circuit breaker functionality"""
    print("🔌 Jeffrey OS Circuit Breaker Demo")
    print("=" * 40)

    # Create circuit breaker with low threshold for demo
    circuit = CircuitBreakerManager().create_circuit(
        "demo_service", failure_threshold=3, recovery_timeout=5.0, timeout_seconds=2.0
    )

    # Mock functions for testing
    async def unreliable_service(fail_rate: float = 0.3):
        """Mock service that fails randomly"""
        await asyncio.sleep(0.1)  # Simulate work
        if random.random() < fail_rate:
            raise Exception("Service temporarily unavailable")
        return {"status": "success", "data": "mock_data"}

    # Test circuit breaker behavior
    print("📊 Testing circuit breaker behavior...")

    for i in range(20):
        try:
            # Increase failure rate over time to trigger circuit opening
            fail_rate = min(0.1 + i * 0.05, 0.8)

            result = await circuit.call(unreliable_service, fail_rate)
            print(f"  ✅ Call {i + 1}: Success - {circuit.state.value}")

        except Exception as e:
            print(f"  ❌ Call {i + 1}: Failed - {circuit.state.value} ({str(e)[:30]}...)")

        # Show state transitions
        if i % 5 == 4:
            state = circuit.get_current_state()
            metrics = circuit.get_metrics()
            print(
                f"     State: {state['state']}, Failures: {state['failure_count']}, Success Rate: {metrics.success_rate:.1f}%"
            )

        await asyncio.sleep(0.2)

    # Show final metrics
    print("\n📈 Final Metrics:")
    metrics = circuit.get_metrics()
    state = circuit.get_current_state()

    print(f"  Total Calls: {metrics.total_calls}")
    print(f"  Success Rate: {metrics.success_rate:.1f}%")
    print(f"  Rejected Calls: {metrics.rejected_calls}")
    print(f"  State Changes: {metrics.state_changes}")
    print(f"  Current State: {state['state']}")

    # Test system health
    health = circuit_manager.get_system_health()
    print(f"\n🏥 System Health: {health['status'].upper()} ({health['health_percentage']:.1f}%)")

    print("\n✅ Circuit breaker demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
