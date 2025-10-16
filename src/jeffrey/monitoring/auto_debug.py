"""
Auto-Debug Engine - Intelligent monitoring and self-diagnostic system
GPT INNOVATION: Proactive issue detection with automated remediation suggestions
"""
import asyncio
import logging
import time
import json
import os
from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class DiagnosticIssue:
    """Represents a detected issue with severity and context"""
    severity: str  # critical, warning, info
    component: str
    title: str
    description: str
    metrics: Dict[str, Any]
    suggested_actions: List[str]
    first_detected: datetime
    last_seen: datetime
    count: int = 1

class AutoDebugEngine:
    """
    Intelligent monitoring system that:
    - Continuously analyzes system metrics
    - Detects anomalies and performance issues
    - Provides automated remediation suggestions
    - Generates actionable insights
    """

    def __init__(self, check_interval: int = 30):
        self.check_interval = check_interval
        self.running = False
        self._task: Optional[asyncio.Task] = None

        # Issue tracking
        self.active_issues: Dict[str, DiagnosticIssue] = {}
        self.resolved_issues: deque = deque(maxlen=100)

        # Metrics history for trend analysis
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))

        # Thresholds for anomaly detection
        self.thresholds = {
            "queue_saturation_critical": 90.0,
            "queue_saturation_warning": 75.0,
            "latency_p95_critical": 500.0,
            "latency_p95_warning": 200.0,
            "error_rate_critical": 10.0,
            "error_rate_warning": 5.0,
            "cache_hit_rate_warning": 50.0,
            "memory_usage_critical": 90.0,
            "cpu_usage_critical": 85.0,
            "concurrent_requests_warning": 80.0
        }

        # System components to monitor
        self.components = {}

    def register_component(self, name: str, component: Any) -> None:
        """Register a component for monitoring"""
        self.components[name] = component
        logger.info(f"🔍 Registered component for monitoring: {name}")

    async def start(self) -> None:
        """Start the auto-debug monitoring loop"""
        if self.running:
            return

        self.running = True
        self._task = asyncio.create_task(self._monitoring_loop())
        logger.info("🤖 Auto-Debug Engine started - intelligent monitoring active")

    async def stop(self) -> None:
        """Stop the monitoring"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🛑 Auto-Debug Engine stopped")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop with intelligent analysis"""
        while self.running:
            try:
                await self._perform_health_check()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Auto-debug monitoring error: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)

    async def _perform_health_check(self) -> None:
        """Comprehensive health check with anomaly detection"""
        current_metrics = await self._collect_metrics()

        # Store metrics for trend analysis
        for metric_name, value in current_metrics.items():
            self.metrics_history[metric_name].append({
                "timestamp": time.time(),
                "value": value
            })

        # Analyze for issues
        await self._analyze_bus_performance(current_metrics)
        await self._analyze_pipeline_health(current_metrics)
        await self._analyze_ml_performance(current_metrics)
        await self._analyze_orchestrator_health(current_metrics)
        await self._analyze_system_resources(current_metrics)

        # Update issue tracking
        self._update_issue_tracking()

        # Log summary if issues found
        if self.active_issues:
            await self._log_issues_summary()

    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from all registered components"""
        metrics = {
            "timestamp": time.time(),
            "check_interval": self.check_interval
        }

        # Collect from BusFacade
        if "bus" in self.components:
            bus_stats = self.components["bus"].get_stats()
            metrics.update({
                f"bus_{key}": value for key, value in bus_stats.items()
            })

        # Collect from ThoughtPipeline
        if "pipeline" in self.components:
            pipeline_stats = self.components["pipeline"].get_metrics()
            metrics.update({
                f"pipeline_{key}": value for key, value in pipeline_stats.items()
            })

        # Collect from ML Adapter
        if "ml_adapter" in self.components:
            ml_stats = self.components["ml_adapter"].get_stats()
            metrics.update({
                f"ml_{key}": value for key, value in ml_stats.items()
            })

        # Collect from Orchestrator
        if "orchestrator" in self.components:
            orch_stats = self.components["orchestrator"].get_stats()
            metrics.update({
                f"orch_{key}": value for key, value in orch_stats.items()
            })

        return metrics

    async def _analyze_bus_performance(self, metrics: Dict[str, Any]) -> None:
        """Analyze event bus performance and detect issues"""

        # Queue saturation analysis
        saturation = metrics.get("bus_queue_saturation_pct", 0)
        if saturation >= self.thresholds["queue_saturation_critical"]:
            await self._report_issue(
                "critical", "event_bus", "Queue Critical Saturation",
                f"Event queue is {saturation:.1f}% full, system overloaded",
                {"saturation": saturation, "queue_size": metrics.get("bus_queue_size", 0)},
                [
                    "Reduce incoming request rate immediately",
                    "Increase queue size if memory allows",
                    "Scale horizontally if possible",
                    "Check for stuck event handlers"
                ]
            )
        elif saturation >= self.thresholds["queue_saturation_warning"]:
            await self._report_issue(
                "warning", "event_bus", "Queue High Saturation",
                f"Event queue is {saturation:.1f}% full, approaching capacity",
                {"saturation": saturation},
                [
                    "Monitor closely for continued growth",
                    "Consider scaling up before reaching critical threshold",
                    "Review event processing efficiency"
                ]
            )

        # Backpressure activation analysis
        backpressure_activations = metrics.get("bus_backpressure_activations", 0)
        if backpressure_activations > 0:
            await self._report_issue(
                "warning", "event_bus", "Backpressure Activated",
                f"Backpressure triggered {backpressure_activations} times",
                {"activations": backpressure_activations},
                [
                    "System is rejecting requests due to overload",
                    "Consider increasing processing capacity",
                    "Review rate limiting configuration"
                ]
            )

    async def _analyze_pipeline_health(self, metrics: Dict[str, Any]) -> None:
        """Analyze cognitive pipeline health"""

        # Error rate analysis
        total_events = metrics.get("pipeline_events_processed", 0) + metrics.get("pipeline_events_failed", 0)
        if total_events > 0:
            error_rate = (metrics.get("pipeline_events_failed", 0) / total_events) * 100
            if error_rate >= self.thresholds["error_rate_critical"]:
                await self._report_issue(
                    "critical", "pipeline", "High Error Rate",
                    f"Pipeline error rate is {error_rate:.1f}%",
                    {"error_rate": error_rate, "total_events": total_events},
                    [
                        "Investigate pipeline exception logs immediately",
                        "Check memory and consciousness component health",
                        "Review circuit breaker states"
                    ]
                )

        # Circuit breaker analysis
        breakers = metrics.get("pipeline_breakers", {})
        for breaker_name, breaker_info in breakers.items():
            if breaker_info.get("state") == "open":
                await self._report_issue(
                    "critical", "pipeline", f"Circuit Breaker Open: {breaker_name}",
                    f"Circuit breaker '{breaker_name}' is open due to failures",
                    {"breaker": breaker_name, "failures": breaker_info.get("failure_count")},
                    [
                        f"Investigate {breaker_name} component failures",
                        "Check dependency health",
                        "Wait for auto-recovery or manual intervention"
                    ]
                )

        # Latency analysis
        p95_latency = metrics.get("pipeline_p95_latency_ms", 0)
        if p95_latency >= self.thresholds["latency_p95_critical"]:
            await self._report_issue(
                "critical", "pipeline", "High P95 Latency",
                f"P95 latency is {p95_latency:.1f}ms",
                {"p95_latency": p95_latency},
                [
                    "Investigate slow operations in pipeline",
                    "Check memory store performance",
                    "Review consciousness processing speed"
                ]
            )

    async def _analyze_ml_performance(self, metrics: Dict[str, Any]) -> None:
        """Analyze ML adapter performance"""

        # Cache hit rate analysis
        cache_hit_rate = metrics.get("ml_cache_hit_rate", 0)
        if cache_hit_rate < self.thresholds["cache_hit_rate_warning"]:
            await self._report_issue(
                "warning", "ml_adapter", "Low Cache Hit Rate",
                f"ML cache hit rate is only {cache_hit_rate:.1f}%",
                {"cache_hit_rate": cache_hit_rate},
                [
                    "Review cache TTL configuration",
                    "Check if requests are too diverse for caching",
                    "Consider increasing cache size"
                ]
            )

        # Timeout analysis
        timeout_rate = metrics.get("ml_timeout_rate", 0)
        if timeout_rate > 5.0:  # More than 5% timeouts
            await self._report_issue(
                "warning", "ml_adapter", "High Timeout Rate",
                f"ML timeout rate is {timeout_rate:.1f}%",
                {"timeout_rate": timeout_rate},
                [
                    "Consider increasing ML timeout threshold",
                    "Check model inference performance",
                    "Review system load and resource availability"
                ]
            )

        # Concurrency utilization
        concurrency_util = metrics.get("ml_concurrency_utilization", 0)
        if concurrency_util >= self.thresholds["concurrent_requests_warning"]:
            await self._report_issue(
                "warning", "ml_adapter", "High Concurrency Utilization",
                f"ML concurrency utilization is {concurrency_util:.1f}%",
                {"concurrency_utilization": concurrency_util},
                [
                    "Consider increasing EMO_MAX_CONCURRENT limit",
                    "Review request patterns for batching opportunities",
                    "Monitor for potential bottlenecks"
                ]
            )

    async def _analyze_orchestrator_health(self, metrics: Dict[str, Any]) -> None:
        """Analyze orchestrator health"""

        # Handler health analysis
        health_summary = metrics.get("orch_health_summary", {})
        critical_handlers = health_summary.get("critical", 0)
        if critical_handlers > 0:
            await self._report_issue(
                "critical", "orchestrator", "Critical Handlers Detected",
                f"{critical_handlers} handlers in critical state",
                {"critical_handlers": critical_handlers},
                [
                    "Review handler error logs",
                    "Check auto-healing effectiveness",
                    "Consider manual handler reset"
                ]
            )

    async def _analyze_system_resources(self, metrics: Dict[str, Any]) -> None:
        """Analyze system resource usage"""

        # System load analysis (if psutil available)
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent

            if cpu_percent >= self.thresholds["cpu_usage_critical"]:
                await self._report_issue(
                    "critical", "system", "High CPU Usage",
                    f"CPU usage is {cpu_percent:.1f}%",
                    {"cpu_percent": cpu_percent},
                    [
                        "Investigate high CPU processes",
                        "Consider scaling resources",
                        "Review processing efficiency"
                    ]
                )

            if memory_percent >= self.thresholds["memory_usage_critical"]:
                await self._report_issue(
                    "critical", "system", "High Memory Usage",
                    f"Memory usage is {memory_percent:.1f}%",
                    {"memory_percent": memory_percent},
                    [
                        "Check for memory leaks",
                        "Review cache sizes",
                        "Consider memory optimization"
                    ]
                )
        except ImportError:
            # psutil not available, skip system resource monitoring
            pass

    async def _report_issue(
        self,
        severity: str,
        component: str,
        title: str,
        description: str,
        metrics: Dict[str, Any],
        suggested_actions: List[str]
    ) -> None:
        """Report or update an issue"""
        issue_key = f"{component}:{title}"

        if issue_key in self.active_issues:
            # Update existing issue
            issue = self.active_issues[issue_key]
            issue.last_seen = datetime.now()
            issue.count += 1
            issue.metrics = metrics  # Update with latest metrics
        else:
            # Create new issue
            issue = DiagnosticIssue(
                severity=severity,
                component=component,
                title=title,
                description=description,
                metrics=metrics,
                suggested_actions=suggested_actions,
                first_detected=datetime.now(),
                last_seen=datetime.now()
            )
            self.active_issues[issue_key] = issue
            logger.warning(f"🚨 New {severity} issue detected: {component} - {title}")

    def _update_issue_tracking(self) -> None:
        """Update issue tracking and resolve old issues"""
        current_time = datetime.now()
        resolved = []

        for issue_key, issue in list(self.active_issues.items()):
            # Resolve issues not seen for 5 minutes
            if (current_time - issue.last_seen).total_seconds() > 300:
                resolved.append(issue_key)
                self.resolved_issues.append(issue)
                del self.active_issues[issue_key]
                logger.info(f"✅ Issue resolved: {issue.component} - {issue.title}")

    async def _log_issues_summary(self) -> None:
        """Log a summary of current issues"""
        critical_count = sum(1 for issue in self.active_issues.values() if issue.severity == "critical")
        warning_count = sum(1 for issue in self.active_issues.values() if issue.severity == "warning")

        if critical_count > 0 or warning_count > 0:
            logger.warning(
                f"🔍 Auto-Debug Summary: {critical_count} critical, {warning_count} warning issues active"
            )

    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        return {
            "timestamp": datetime.now().isoformat(),
            "active_issues": {
                issue_key: {
                    "severity": issue.severity,
                    "component": issue.component,
                    "title": issue.title,
                    "description": issue.description,
                    "metrics": issue.metrics,
                    "suggested_actions": issue.suggested_actions,
                    "first_detected": issue.first_detected.isoformat(),
                    "last_seen": issue.last_seen.isoformat(),
                    "count": issue.count
                }
                for issue_key, issue in self.active_issues.items()
            },
            "resolved_issues_count": len(self.resolved_issues),
            "monitoring_stats": {
                "check_interval": self.check_interval,
                "components_monitored": list(self.components.keys()),
                "metrics_tracked": len(self.metrics_history),
                "uptime_checks": len(next(iter(self.metrics_history.values()), []))
            },
            "health_score": self._calculate_health_score()
        }

    def _calculate_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        if not self.active_issues:
            return 100.0

        critical_penalty = sum(20 for issue in self.active_issues.values() if issue.severity == "critical")
        warning_penalty = sum(5 for issue in self.active_issues.values() if issue.severity == "warning")

        score = max(0.0, 100.0 - critical_penalty - warning_penalty)
        return score