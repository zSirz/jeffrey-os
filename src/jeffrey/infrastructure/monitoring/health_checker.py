"""
Vérificateur de santé des composants.

Ce module implémente les fonctionnalités essentielles pour vérificateur de santé des composants.
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
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any

import psutil

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from python_json_logger import jsonlogger


class HealthStatus(Enum):
    """System health status levels"""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class AlertSeverity(Enum):
    """Alert severity levels"""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """System health metrics snapshot"""

    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_available_gb: float
    disk_percent: float
    disk_available_gb: float
    event_processing_rate: float
    queue_size: int
    response_time_p50: float
    response_time_p95: float
    response_time_p99: float
    active_connections: int
    error_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Alert:
    """Health alert with context"""

    id: str
    timestamp: str
    severity: AlertSeverity
    component: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    context: dict[str, Any]
    resolved: bool = False
    resolution_time: str | None = None


class MetricsCollector:
    """Collects system and application metrics"""

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size
        self.response_times: deque = deque(maxlen=window_size)
        self.event_timestamps: deque = deque(maxlen=window_size)
        self.error_count = 0
        self.total_requests = 0
        self._lock = threading.Lock()

        # Prometheus metrics (if available)
        if PROMETHEUS_AVAILABLE:
            self.registry = CollectorRegistry()
            self._setup_prometheus_metrics()

    def _setup_prometheus_metrics(self):
        """Setup Prometheus metrics"""
        # Counters
        self.prom_events_total = Counter(
            "jeffrey_events_total",
            "Total number of events processed",
            ["event_type"],
            registry=self.registry,
        )

        self.prom_errors_total = Counter(
            "jeffrey_errors_total", "Total number of errors", ["component"], registry=self.registry
        )

        # Histograms
        self.prom_response_time = Histogram(
            "jeffrey_response_time_seconds", "Response time in seconds", registry=self.registry
        )

        # Gauges
        self.prom_cpu_usage = Gauge("jeffrey_cpu_usage_percent", "CPU usage percentage", registry=self.registry)

        self.prom_memory_usage = Gauge(
            "jeffrey_memory_usage_percent", "Memory usage percentage", registry=self.registry
        )

        self.prom_queue_size = Gauge("jeffrey_queue_size", "Current queue size", registry=self.registry)

    def record_response_time(self, time_ms: float) -> None:
        """Record response time"""
        with self._lock:
            self.response_times.append(time_ms)
            self.total_requests += 1

        if PROMETHEUS_AVAILABLE:
            self.prom_response_time.observe(time_ms / 1000.0)

    def record_event(self, event_type: str = "generic") -> None:
        """Record event processing"""
        with self._lock:
            self.event_timestamps.append(time.time())

        if PROMETHEUS_AVAILABLE:
            self.prom_events_total.labels(event_type=event_type).inc()

    def record_error(self, component: str = "unknown") -> None:
        """Record error occurrence"""
        with self._lock:
            self.error_count += 1

        if PROMETHEUS_AVAILABLE:
            self.prom_errors_total.labels(component=component).inc()

    def get_event_processing_rate(self) -> float:
        """Calculate events per second"""
        with self._lock:
            if len(self.event_timestamps) < 2:
                return 0.0

            recent_events = [ts for ts in self.event_timestamps if time.time() - ts <= 60]
        return len(recent_events) / 60.0

    def get_response_time_percentiles(self) -> dict[str, float]:
        """Calculate response time percentiles"""
        with self._lock:
            if not self.response_times:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

            sorted_times = sorted(self.response_times)
            n = len(sorted_times)

        return {
            "p50": sorted_times[int(n * 0.5)] if n > 0 else 0.0,
            "p95": sorted_times[int(n * 0.95)] if n > 0 else 0.0,
            "p99": sorted_times[int(n * 0.99)] if n > 0 else 0.0,
        }

    def get_error_rate(self) -> float:
        """Calculate error rate"""
        with self._lock:
            if self.total_requests == 0:
                return 0.0
        return (self.error_count / self.total_requests) * 100.0

    def update_prometheus_gauges(self, metrics: HealthMetrics) -> None:
        """Update Prometheus gauge metrics"""
        if not PROMETHEUS_AVAILABLE:
            return

        self.prom_cpu_usage.set(metrics.cpu_percent)
        self.prom_memory_usage.set(metrics.memory_percent)
        self.prom_queue_size.set(metrics.queue_size)


class ThresholdManager:
    """Manages dynamic thresholds with adaptive adjustment"""

    def __init__(self) -> None:
        self.thresholds = {
            "cpu_percent": {"warning": 70.0, "critical": 90.0},
            "memory_percent": {"warning": 80.0, "critical": 95.0},
            "disk_percent": {"warning": 85.0, "critical": 95.0},
            "response_time_p95": {"warning": 1000.0, "critical": 5000.0},  # ms
            "event_processing_rate": {"warning": 10.0, "critical": 1.0},  # min rate
            "queue_size": {"warning": 1000, "critical": 5000},
            "error_rate": {"warning": 5.0, "critical": 15.0},  # percentage
        }

        self.adaptive_enabled = True
        self.adjustment_history: dict[str, list[float]] = {}

    def get_threshold(self, metric: str, severity: str) -> float:
        """Get threshold for metric and severity"""
        return self.thresholds.get(metric, {}).get(severity, float("inf"))

    def update_threshold(self, metric: str, severity: str, value: float) -> None:
        """Update threshold value"""
        if metric in self.thresholds:
            self.thresholds[metric][severity] = value

            # Track adjustment history
            if metric not in self.adjustment_history:
                self.adjustment_history[metric] = []
            self.adjustment_history[metric].append(value)

    def adapt_thresholds(self, recent_metrics: list[HealthMetrics]) -> None:
        """Adapt thresholds based on historical data"""
        if not self.adaptive_enabled or len(recent_metrics) < 10:
            return

        # Analyze patterns and adjust thresholds
        # This is a simplified implementation
        for metric_name in ["cpu_percent", "memory_percent"]:
            values = [getattr(m, metric_name) for m in recent_metrics]
            avg_value = sum(values) / len(values)

            # If average is consistently below warning threshold, we can be more sensitive
            warning_threshold = self.get_threshold(metric_name, "warning")
        if avg_value < warning_threshold * 0.5:
            new_warning = warning_threshold * 0.9
            self.update_threshold(metric_name, "warning", new_warning)


class AlertManager:
    """Manages health alerts with deduplication and escalation"""

    def __init__(self, max_alerts: int = 1000) -> None:
        self.max_alerts = max_alerts
        self.active_alerts: dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=max_alerts)
        self.alert_callbacks: list[Callable[[Alert], None]] = []
        self._lock = asyncio.Lock()

    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)

    async def create_alert(
        self,
        component: str,
        metric_name: str,
        current_value: float,
        threshold: float,
        severity: AlertSeverity,
        message: str,
        context: dict[str, Any] | None = None,
    ) -> Alert:
        """Create new alert with deduplication"""
        alert_key = f"{component}_{metric_name}_{severity.value}"

        async with self._lock:
            # Check if similar alert already exists
            if alert_key in self.active_alerts:
                existing = self.active_alerts[alert_key]
                existing.current_value = current_value
                existing.timestamp = datetime.utcnow().isoformat() + "Z"
                return existing

            # Create new alert
            alert = Alert(
                id=f"alert_{int(time.time() * 1000)}_{hash(alert_key) % 10000}",
                timestamp=datetime.utcnow().isoformat() + "Z",
                severity=severity,
                component=component,
                message=message,
                metric_name=metric_name,
                current_value=current_value,
                threshold=threshold,
                context=context or {},
                resolved=False,
            )

            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)

            # Notify callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logging.error(f"Alert callback failed: {e}")

        return alert

    async def resolve_alert(self, component: str, metric_name: str, severity: AlertSeverity) -> bool:
        """Resolve an alert"""
        alert_key = f"{component}_{metric_name}_{severity.value}"

        async with self._lock:
            if alert_key in self.active_alerts:
                alert = self.active_alerts[alert_key]
                alert.resolved = True
                alert.resolution_time = datetime.utcnow().isoformat() + "Z"

                del self.active_alerts[alert_key]
        return True

        return False

    def get_active_alerts(self) -> list[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())

    def get_alert_summary(self) -> dict[str, Any]:
        """Get alert summary statistics"""
        active = list(self.active_alerts.values())
        history = list(self.alert_history)

        severity_counts = {}
        for severity in AlertSeverity:
            severity_counts[severity.value] = len([a for a in active if a.severity == severity])

        return {
            "active_count": len(active),
            "total_alerts_today": len([a for a in history if self._is_today(a.timestamp)]),
            "severity_breakdown": severity_counts,
            "most_frequent_component": self._get_most_frequent_component(history),
            "avg_resolution_time_minutes": self._calculate_avg_resolution_time(history),
        }

    def _is_today(self, timestamp: str) -> bool:
        """Check if timestamp is from today"""
        try:
            ts = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            return ts.date() == datetime.utcnow().date()
        except Exception:
            return False

    def _get_most_frequent_component(self, alerts: list[Alert]) -> str:
        """Get most frequently alerted component"""
        if not alerts:
            return "none"

        component_counts = {}
        for alert in alerts:
            component_counts[alert.component] = component_counts.get(alert.component, 0) + 1

        return max(component_counts, key=component_counts.get)

    def _calculate_avg_resolution_time(self, alerts: list[Alert]) -> float:
        """Calculate average resolution time in minutes"""
        resolved_alerts = [a for a in alerts if a.resolved and a.resolution_time]
        if not resolved_alerts:
            return 0.0

        total_minutes = 0.0
        for alert in resolved_alerts:
            try:
                created = datetime.fromisoformat(alert.timestamp.replace("Z", "+00:00"))
                resolved = datetime.fromisoformat(alert.resolution_time.replace("Z", "+00:00"))
                total_minutes += (resolved - created).total_seconds() / 60.0
            except Exception:
                continue

        return total_minutes / len(resolved_alerts) if resolved_alerts else 0.0


class HealthChecker:
    """
    Main health checker with real-time monitoring and intelligent alerting
    Monitors system resources, application metrics, and provides Prometheus export
    """

    def __init__(
        self,
        check_interval: float = 5.0,
        metrics_window_size: int = 1000,
        enable_prometheus: bool = True,
    ):
        """
        Initialize health checker

        Args:
            check_interval: Seconds between health checks
            metrics_window_size: Size of metrics history window
            enable_prometheus: Enable Prometheus metrics export
        """
        self.check_interval = check_interval
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE

        # Core components
        self.metrics_collector = MetricsCollector(metrics_window_size)
        self.threshold_manager = ThresholdManager()
        self.alert_manager = AlertManager()

        # Health tracking
        self.current_status = HealthStatus.HEALTHY
        self.metrics_history: deque = deque(maxlen=100)  # Keep 100 recent snapshots
        self.last_check_time = 0.0

        # Monitoring state
        self.running = False
        self.monitor_task: asyncio.Task | None = None

        # External queue monitoring (injected from GuardianBus)
        self.external_queue_size_callback: Callable[[], int] | None = None

        # Setup logging
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        logHandler.setFormatter(formatter)
        self.logger = logging.getLogger("HealthChecker")
        self.logger.addHandler(logHandler)
        self.logger.setLevel(logging.INFO)

        # Setup alert callback
        self.alert_manager.add_alert_callback(self._log_alert)

        self.logger.info(
            "HealthChecker initialized",
            extra={
                "check_interval": check_interval,
                "prometheus_enabled": self.enable_prometheus,
                "metrics_window_size": metrics_window_size,
            },
        )

    def set_queue_monitor(self, callback: Callable[[], int]) -> None:
        """Set callback to monitor external queue size"""
        self.external_queue_size_callback = callback

    async def start_monitoring(self) -> None:
        """Start continuous health monitoring"""
        if self.running:
            return

        self.running = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop())

        self.logger.info("Health monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.running = False

        if self.monitor_task:
            self.monitor_task.cancel()
        try:
            await self.monitor_task
        except asyncio.CancelledError:
            pass

        self.logger.info("Health monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect metrics
                # metrics = await self._collect_metrics()

                # Store in history
                self.metrics_history.append(metrics)

                # Update Prometheus gauges
                if self.enable_prometheus:
                    self.metrics_collector.update_prometheus_gauges(metrics)

                # Check thresholds and generate alerts
                await self._check_thresholds(metrics)

                # Update overall health status
                self._update_health_status(metrics)

                # Adapt thresholds if enabled
                if len(self.metrics_history) >= 10:
                    self.threshold_manager.adapt_thresholds(list(self.metrics_history)[-10:])

                self.last_check_time = time.time()

            except Exception as e:
                self.logger.error("Health check failed", extra={"error": str(e)})
                self.metrics_collector.record_error("health_checker")

            await asyncio.sleep(self.check_interval)

    async def _collect_metrics(self) -> HealthMetrics:
        """Collect comprehensive system metrics"""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        # Application metrics
        response_times = self.metrics_collector.get_response_time_percentiles()
        event_rate = self.metrics_collector.get_event_processing_rate()
        error_rate = self.metrics_collector.get_error_rate()

        # Queue size from external callback
        queue_size = 0
        if self.external_queue_size_callback:
            try:
                queue_size = self.external_queue_size_callback()
            except Exception:
                pass

        # Network connections (simplified)
        try:
            connections = len(psutil.net_connections())
        except (psutil.AccessDenied, Exception):
            connections = 0

        return HealthMetrics(
            timestamp=datetime.utcnow().isoformat() + "Z",
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_available_gb=memory.available / (1024**3),
            disk_percent=disk.percent,
            disk_available_gb=disk.free / (1024**3),
            event_processing_rate=event_rate,
            queue_size=queue_size,
            response_time_p50=response_times["p50"],
            response_time_p95=response_times["p95"],
            response_time_p99=response_times["p99"],
            active_connections=connections,
            error_rate=error_rate,
        )

    async def _check_thresholds(self, metrics: HealthMetrics) -> None:
        """Check metrics against thresholds and create alerts"""
        threshold_checks = [
            ("system", "cpu_percent", metrics.cpu_percent),
            ("system", "memory_percent", metrics.memory_percent),
            ("system", "disk_percent", metrics.disk_percent),
            ("application", "response_time_p95", metrics.response_time_p95),
            ("application", "queue_size", metrics.queue_size),
            ("application", "error_rate", metrics.error_rate),
        ]

        for component, metric_name, current_value in threshold_checks:
            # Check warning threshold
            warning_threshold = self.threshold_manager.get_threshold(metric_name, "warning")
            critical_threshold = self.threshold_manager.get_threshold(metric_name, "critical")

            # Handle inverted metrics (lower is worse)
            if metric_name == "event_processing_rate":
                if current_value < critical_threshold:
                    await self.alert_manager.create_alert(
                        component,
                        metric_name,
                        current_value,
                        critical_threshold,
                        AlertSeverity.CRITICAL,
                        f"Event processing rate critically low: {current_value:.2f} events/sec",
                        {"expected_minimum": critical_threshold},
                    )
                elif current_value < warning_threshold:
                    await self.alert_manager.create_alert(
                        component,
                        metric_name,
                        current_value,
                        warning_threshold,
                        AlertSeverity.WARNING,
                        f"Event processing rate low: {current_value:.2f} events/sec",
                        {"expected_minimum": warning_threshold},
                    )
                else:
                    # Resolve alerts if value is back to normal
                    await self.alert_manager.resolve_alert(component, metric_name, AlertSeverity.WARNING)
                    await self.alert_manager.resolve_alert(component, metric_name, AlertSeverity.CRITICAL)
            else:
                # Normal metrics (higher is worse)
                if current_value > critical_threshold:
                    await self.alert_manager.create_alert(
                        component,
                        metric_name,
                        current_value,
                        critical_threshold,
                        AlertSeverity.CRITICAL,
                        f"{metric_name} critically high: {current_value:.2f}%",
                        {"threshold": critical_threshold},
                    )
                elif current_value > warning_threshold:
                    await self.alert_manager.create_alert(
                        component,
                        metric_name,
                        current_value,
                        warning_threshold,
                        AlertSeverity.WARNING,
                        f"{metric_name} high: {current_value:.2f}%",
                        {"threshold": warning_threshold},
                    )
                else:
                    # Resolve alerts if value is back to normal
                    await self.alert_manager.resolve_alert(component, metric_name, AlertSeverity.WARNING)
                    await self.alert_manager.resolve_alert(component, metric_name, AlertSeverity.CRITICAL)

    def _update_health_status(self, metrics: HealthMetrics) -> None:
        """Update overall health status based on active alerts"""
        active_alerts = self.alert_manager.get_active_alerts()

        if not active_alerts:
            self.current_status = HealthStatus.HEALTHY
        elif any(a.severity == AlertSeverity.CRITICAL for a in active_alerts):
            self.current_status = HealthStatus.CRITICAL
        elif any(a.severity == AlertSeverity.ERROR for a in active_alerts):
            self.current_status = HealthStatus.UNHEALTHY
        elif any(a.severity == AlertSeverity.WARNING for a in active_alerts):
            self.current_status = HealthStatus.DEGRADED
        else:
            self.current_status = HealthStatus.HEALTHY

    def _log_alert(self, alert: Alert) -> None:
        """Log alert callback"""
        self.logger.warning(
            "Health alert",
            extra={
                "alert_id": alert.id,
                "severity": alert.severity.value,
                "component": alert.component,
                "metric": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
                "message": alert.message,
            },
        )

    def get_current_health(self) -> dict[str, Any]:
        """Get current health status and latest metrics"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        active_alerts = self.alert_manager.get_active_alerts()
        alert_summary = self.alert_manager.get_alert_summary()

        return {
            "status": self.current_status.value,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metrics": latest_metrics.to_dict() if latest_metrics else None,
            "active_alerts": [alert.id for alert in active_alerts],
            "alert_summary": alert_summary,
            "monitoring_uptime_seconds": time.time() - self.last_check_time if self.running else 0,
            "prometheus_enabled": self.enable_prometheus,
        }

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics in standard format"""
        if not self.enable_prometheus:
            return "# Prometheus not enabled\n"

        return generate_latest(self.metrics_collector.registry).decode("utf-8")

    def get_metrics_history(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent metrics history"""
        recent = list(self.metrics_history)[-limit:]
        return [m.to_dict() for m in recent]

    # External API for metrics recording
    def record_response_time(self, time_ms: float) -> None:
        """External API to record response time"""
        self.metrics_collector.record_response_time(time_ms)

    def record_event(self, event_type: str = "generic") -> None:
        """External API to record event"""
        self.metrics_collector.record_event(event_type)

    def record_error(self, component: str = "unknown") -> None:
        """External API to record error"""
        self.metrics_collector.record_error(component)


# Example usage and demonstration
async def main():
    """Demo health checker functionality"""
    checker = HealthChecker(check_interval=2.0)

    print("🏥 Jeffrey OS Health Checker Demo")
    print("=" * 40)

    # Start monitoring
    await checker.start_monitoring()

    # Simulate some activity
    for i in range(10):
        # Simulate event processing
        checker.record_event("test_event")
        checker.record_response_time(50 + i * 10)  # Increasing response times

        if i % 3 == 0:
            checker.record_error("test_component")

        await asyncio.sleep(1)

    # Get health status
    health = checker.get_current_health()
    print(f"\n📊 Current Health: {health['status'].upper()}")

    if health["metrics"]:
        metrics = health["metrics"]
        print(f"  CPU: {metrics['cpu_percent']:.1f}%")
        print(f"  Memory: {metrics['memory_percent']:.1f}%")
        print(f"  Event Rate: {metrics['event_processing_rate']:.1f}/sec")
        print(f"  Error Rate: {metrics['error_rate']:.1f}%")

    print(f"\n🚨 Active Alerts: {len(health['active_alerts'])}")

    # Show Prometheus metrics (first few lines)
    if checker.enable_prometheus:
        prom_metrics = checker.get_prometheus_metrics()
        print("\n📈 Prometheus Metrics (sample):")
        print("\n".join(prom_metrics.split("\n")[:10]))

    # Stop monitoring
    await checker.stop_monitoring()
    print("\n✅ Health monitoring demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
