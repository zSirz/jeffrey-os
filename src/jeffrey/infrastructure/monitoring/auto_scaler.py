"""
Module de infrastructure système de base pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de infrastructure système de base pour jeffrey os.
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
from python_json_logger import jsonlogger


class ScalingProfile(Enum):
    """Scaling profiles for different environments"""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    CLOUD = "cloud"
    EDGE = "edge"
    CUSTOM = "custom"


class ScalingAction(Enum):
    """Types of scaling actions"""

    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    OPTIMIZE = "optimize"
    MAINTAIN = "maintain"


@dataclass
class ScalingDecision:
    """Scaling decision with context"""

    timestamp: str
    action: ScalingAction
    component: str
    previous_value: Any
    new_value: Any
    reason: str
    confidence: float
    estimated_impact: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ResourceLimits:
    """Resource limits and constraints"""

    min_deque_size: int
    max_deque_size: int
    min_worker_threads: int
    max_worker_threads: int
    min_batch_size: int
    max_batch_size: int
    min_buffer_size: int
    max_buffer_size: int
    memory_limit_gb: float
    cpu_limit_percent: float


class ProfileManager:
    """Manages scaling profiles and configurations"""

    def __init__(self) -> None:
        self.profiles = {
            ScalingProfile.DEVELOPMENT: ResourceLimits(
                min_deque_size=100,
                max_deque_size=1000,
                min_worker_threads=1,
                max_worker_threads=4,
                min_batch_size=10,
                max_batch_size=50,
                min_buffer_size=50,
                max_buffer_size=200,
                memory_limit_gb=2.0,
                cpu_limit_percent=50.0,
            ),
            ScalingProfile.PRODUCTION: ResourceLimits(
                min_deque_size=500,
                max_deque_size=5000,
                min_worker_threads=2,
                max_worker_threads=16,
                min_batch_size=50,
                max_batch_size=500,
                min_buffer_size=200,
                max_buffer_size=1000,
                memory_limit_gb=8.0,
                cpu_limit_percent=80.0,
            ),
            ScalingProfile.CLOUD: ResourceLimits(
                min_deque_size=1000,
                max_deque_size=10000,
                min_worker_threads=4,
                max_worker_threads=32,
                min_batch_size=100,
                max_batch_size=1000,
                min_buffer_size=500,
                max_buffer_size=2000,
                memory_limit_gb=16.0,
                cpu_limit_percent=90.0,
            ),
            ScalingProfile.EDGE: ResourceLimits(
                min_deque_size=50,
                max_deque_size=500,
                min_worker_threads=1,
                max_worker_threads=2,
                min_batch_size=5,
                max_batch_size=25,
                min_buffer_size=25,
                max_buffer_size=100,
                memory_limit_gb=1.0,
                cpu_limit_percent=30.0,
            ),
        }

        self.current_profile = ScalingProfile.PRODUCTION
        self.custom_limits: ResourceLimits | None = None

    def set_profile(self, profile: ScalingProfile) -> None:
        """Set current scaling profile"""
        self.current_profile = profile

    def set_custom_limits(self, limits: ResourceLimits) -> None:
        """Set custom resource limits"""
        self.custom_limits = limits
        self.current_profile = ScalingProfile.CUSTOM

    def get_current_limits(self) -> ResourceLimits:
        """Get current resource limits"""
        if self.current_profile == ScalingProfile.CUSTOM and self.custom_limits:
            return self.custom_limits
        return self.profiles[self.current_profile]

    def get_profile_info(self) -> dict[str, Any]:
        """Get current profile information"""
        limits = self.get_current_limits()
        return {
            "profile": self.current_profile.value,
            "limits": asdict(limits),
            "auto_detected": self._detect_optimal_profile().value,
        }

    def _detect_optimal_profile(self) -> ScalingProfile:
        """Auto-detect optimal profile based on system resources"""
        try:
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()

            if memory_gb >= 16 and cpu_count >= 8:
                return ScalingProfile.CLOUD
            elif memory_gb >= 8 and cpu_count >= 4:
                return ScalingProfile.PRODUCTION
            elif memory_gb >= 2 and cpu_count >= 2:
                return ScalingProfile.DEVELOPMENT
            else:
                return ScalingProfile.EDGE
        except Exception:
            return ScalingProfile.DEVELOPMENT


class MetricsAnalyzer:
    """Analyzes metrics to make scaling decisions"""

    def __init__(self, history_window: int = 100) -> None:
        self.history_window = history_window
        self.cpu_history: deque = deque(maxlen=history_window)
        self.memory_history: deque = deque(maxlen=history_window)
        self.throughput_history: deque = deque(maxlen=history_window)
        self.response_time_history: deque = deque(maxlen=history_window)
        self.queue_size_history: deque = deque(maxlen=history_window)

        self._lock = threading.Lock()

    def add_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        throughput: float,
        response_time_ms: float,
        queue_size: int,
    ) -> None:
        """Add metrics sample to history"""
        with self._lock:
            timestamp = time.time()
            self.cpu_history.append((timestamp, cpu_percent))
            self.memory_history.append((timestamp, memory_percent))
            self.throughput_history.append((timestamp, throughput))
            self.response_time_history.append((timestamp, response_time_ms))
            self.queue_size_history.append((timestamp, queue_size))

    def analyze_trends(self) -> dict[str, Any]:
        """Analyze trends in metrics"""
        with self._lock:
            if len(self.cpu_history) < 10:
                return {"insufficient_data": True}

            # Analyze recent trends (last 10 samples vs previous 10)
            recent_window = 10

            cpu_values = [v for _, v in self.cpu_history]
            memory_values = [v for _, v in self.memory_history]
            throughput_values = [v for _, v in self.throughput_history]
            response_time_values = [v for _, v in self.response_time_history]
            queue_size_values = [v for _, v in self.queue_size_history]

            recent_cpu = sum(cpu_values[-recent_window:]) / recent_window
            previous_cpu = (
                sum(cpu_values[-recent_window * 2 : -recent_window]) / recent_window
                if len(cpu_values) >= recent_window * 2
                else recent_cpu
            )

            recent_memory = sum(memory_values[-recent_window:]) / recent_window
            previous_memory = (
                sum(memory_values[-recent_window * 2 : -recent_window]) / recent_window
                if len(memory_values) >= recent_window * 2
                else recent_memory
            )

            recent_throughput = sum(throughput_values[-recent_window:]) / recent_window
            previous_throughput = (
                sum(throughput_values[-recent_window * 2 : -recent_window]) / recent_window
                if len(throughput_values) >= recent_window * 2
                else recent_throughput
            )

            recent_response_time = sum(response_time_values[-recent_window:]) / recent_window
            previous_response_time = (
                sum(response_time_values[-recent_window * 2 : -recent_window]) / recent_window
                if len(response_time_values) >= recent_window * 2
                else recent_response_time
            )

            recent_queue_size = sum(queue_size_values[-recent_window:]) / recent_window
            previous_queue_size = (
                sum(queue_size_values[-recent_window * 2 : -recent_window]) / recent_window
                if len(queue_size_values) >= recent_window * 2
                else recent_queue_size
            )

        return {
            "cpu": {
                "current": recent_cpu,
                "trend": (
                    "increasing"
                    if recent_cpu > previous_cpu * 1.1
                    else "decreasing"
                    if recent_cpu < previous_cpu * 0.9
                    else "stable"
                ),
                "change_percent": (((recent_cpu - previous_cpu) / previous_cpu * 100) if previous_cpu > 0 else 0),
            },
            "memory": {
                "current": recent_memory,
                "trend": (
                    "increasing"
                    if recent_memory > previous_memory * 1.1
                    else "decreasing"
                    if recent_memory < previous_memory * 0.9
                    else "stable"
                ),
                "change_percent": (
                    ((recent_memory - previous_memory) / previous_memory * 100) if previous_memory > 0 else 0
                ),
            },
            "throughput": {
                "current": recent_throughput,
                "trend": (
                    "increasing"
                    if recent_throughput > previous_throughput * 1.1
                    else "decreasing"
                    if recent_throughput < previous_throughput * 0.9
                    else "stable"
                ),
                "change_percent": (
                    ((recent_throughput - previous_throughput) / previous_throughput * 100)
                    if previous_throughput > 0
                    else 0
                ),
            },
            "response_time": {
                "current": recent_response_time,
                "trend": (
                    "increasing"
                    if recent_response_time > previous_response_time * 1.1
                    else ("decreasing" if recent_response_time < previous_response_time * 0.9 else "stable")
                ),
                "change_percent": (
                    ((recent_response_time - previous_response_time) / previous_response_time * 100)
                    if previous_response_time > 0
                    else 0
                ),
            },
            "queue_size": {
                "current": recent_queue_size,
                "trend": (
                    "increasing"
                    if recent_queue_size > previous_queue_size * 1.1
                    else "decreasing"
                    if recent_queue_size < previous_queue_size * 0.9
                    else "stable"
                ),
                "change_percent": (
                    ((recent_queue_size - previous_queue_size) / previous_queue_size * 100)
                    if previous_queue_size > 0
                    else 0
                ),
            },
        }

    def predict_resource_needs(self) -> dict[str, str]:
        """Predict future resource needs based on trends"""
        trends = self.analyze_trends()
        if trends.get("insufficient_data"):
            return {"prediction": "insufficient_data"}

        predictions = {}

        # CPU prediction
        cpu_trend = trends["cpu"]["trend"]
        cpu_change = trends["cpu"]["change_percent"]
        if cpu_trend == "increasing" and cpu_change > 20:
            predictions["cpu"] = "scale_up_soon"
        elif cpu_trend == "decreasing" and cpu_change < -20:
            predictions["cpu"] = "scale_down_possible"
        else:
            predictions["cpu"] = "stable"

        # Memory prediction
        memory_trend = trends["memory"]["trend"]
        memory_change = trends["memory"]["change_percent"]
        if memory_trend == "increasing" and memory_change > 15:
            predictions["memory"] = "scale_up_soon"
        elif memory_trend == "decreasing" and memory_change < -15:
            predictions["memory"] = "scale_down_possible"
        else:
            predictions["memory"] = "stable"

        # Throughput prediction
        throughput_trend = trends["throughput"]["trend"]
        if throughput_trend == "decreasing":
            predictions["throughput"] = "performance_degrading"
        elif throughput_trend == "increasing":
            predictions["throughput"] = "performance_improving"
        else:
            predictions["throughput"] = "stable"

        return predictions


class ComponentScaler:
    """Handles scaling of individual components"""

    def __init__(self, limits: ResourceLimits) -> None:
        self.limits = limits
        self.current_settings = {
            "deque_size": limits.min_deque_size * 2,  # Start in middle
            "worker_threads": max(2, limits.min_worker_threads),
            "batch_size": limits.min_batch_size * 2,
            "buffer_size": limits.min_buffer_size * 2,
        }
        self.scaling_history: deque = deque(maxlen=50)
        self.last_scaling_time = 0.0
        self.scaling_cooldown = 30.0  # 30 seconds between scaling actions

    def can_scale(self) -> bool:
        """Check if enough time has passed since last scaling"""
        return time.time() - self.last_scaling_time > self.scaling_cooldown

    def scale_deque_size(self, target_factor: float, reason: str, confidence: float = 0.8) -> ScalingDecision | None:
        """Scale deque size based on memory pressure and queue utilization"""
        if not self.can_scale():
            return None

        current_size = self.current_settings["deque_size"]
        new_size = int(current_size * target_factor)

        # Apply limits
        new_size = max(self.limits.min_deque_size, min(new_size, self.limits.max_deque_size))

        if new_size == current_size:
            return None

        action = ScalingAction.SCALE_UP if new_size > current_size else ScalingAction.SCALE_DOWN

        decision = ScalingDecision(
            timestamp=datetime.utcnow().isoformat() + "Z",
            action=action,
            component="deque_size",
            previous_value=current_size,
            new_value=new_size,
            reason=reason,
            confidence=confidence,
            estimated_impact=f"Memory usage {'increase' if action == ScalingAction.SCALE_UP else 'decrease'}",
        )

        self.current_settings["deque_size"] = new_size
        self.scaling_history.append(decision)
        self.last_scaling_time = time.time()

        return decision

    def scale_worker_threads(
        self, target_factor: float, reason: str, confidence: float = 0.8
    ) -> ScalingDecision | None:
        """Scale worker thread count based on CPU and throughput"""
        if not self.can_scale():
            return None

        current_threads = self.current_settings["worker_threads"]
        new_threads = int(current_threads * target_factor)

        # Apply limits
        new_threads = max(self.limits.min_worker_threads, min(new_threads, self.limits.max_worker_threads))

        if new_threads == current_threads:
            return None

        action = ScalingAction.SCALE_UP if new_threads > current_threads else ScalingAction.SCALE_DOWN

        decision = ScalingDecision(
            timestamp=datetime.utcnow().isoformat() + "Z",
            action=action,
            component="worker_threads",
            previous_value=current_threads,
            new_value=new_threads,
            reason=reason,
            confidence=confidence,
            estimated_impact=f"Parallel processing {'increase' if action == ScalingAction.SCALE_UP else 'decrease'}",
        )

        self.current_settings["worker_threads"] = new_threads
        self.scaling_history.append(decision)
        self.last_scaling_time = time.time()

        return decision

    def scale_batch_size(self, target_factor: float, reason: str, confidence: float = 0.8) -> ScalingDecision | None:
        """Scale batch processing size based on throughput needs"""
        if not self.can_scale():
            return None

        current_batch = self.current_settings["batch_size"]
        new_batch = int(current_batch * target_factor)

        # Apply limits
        new_batch = max(self.limits.min_batch_size, min(new_batch, self.limits.max_batch_size))

        if new_batch == current_batch:
            return None

        action = ScalingAction.SCALE_UP if new_batch > current_batch else ScalingAction.SCALE_DOWN

        decision = ScalingDecision(
            timestamp=datetime.utcnow().isoformat() + "Z",
            action=action,
            component="batch_size",
            previous_value=current_batch,
            new_value=new_batch,
            reason=reason,
            confidence=confidence,
            estimated_impact=f"Batch efficiency {'increase' if action == ScalingAction.SCALE_UP else 'decrease'}",
        )

        self.current_settings["batch_size"] = new_batch
        self.scaling_history.append(decision)
        self.last_scaling_time = time.time()

        return decision

    def apply_aggressive_compression(self, reason: str) -> ScalingDecision | None:
        """Apply aggressive compression when disk usage is high"""
        decision = ScalingDecision(
            timestamp=datetime.utcnow().isoformat() + "Z",
            action=ScalingAction.OPTIMIZE,
            component="compression",
            previous_value="normal",
            new_value="aggressive",
            reason=reason,
            confidence=0.9,
            estimated_impact="Reduced disk usage, slightly higher CPU usage",
        )

        self.scaling_history.append(decision)
        return decision

    def get_current_settings(self) -> dict[str, Any]:
        """Get current component settings"""
        return self.current_settings.copy()

    def get_scaling_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent scaling history"""
        recent = list(self.scaling_history)[-limit:]
        return [decision.to_dict() for decision in recent]


class AutoScaler:
    """
    Intelligent auto-scaler with adaptive resource management
    Automatically adjusts system resources based on load and performance
    """

    def __init__(
        self,
        profile: ScalingProfile = ScalingProfile.PRODUCTION,
        scaling_interval: float = 30.0,
        enable_predictive: bool = True,
    ):
        """
        Initialize auto-scaler

        Args:
            profile: Scaling profile to use
            scaling_interval: Seconds between scaling evaluations
            enable_predictive: Enable predictive scaling
        """
        self.scaling_interval = scaling_interval
        self.enable_predictive = enable_predictive

        # Core components
        self.profile_manager = ProfileManager()
        self.profile_manager.set_profile(profile)
        self.metrics_analyzer = MetricsAnalyzer()
        self.component_scaler = ComponentScaler(self.profile_manager.get_current_limits())

        # Scaling state
        self.running = False
        self.scaling_task: asyncio.Task | None = None
        self.last_scaling_check = 0.0

        # External callbacks for applying scaling decisions
        self.scaling_callbacks: dict[str, Callable[[Any], None]] = {}

        # Statistics
        self.stats = {
            "total_scaling_actions": 0,
            "successful_scalings": 0,
            "scaling_by_component": {},
            "avg_scaling_interval": scaling_interval,
        }

        # Setup logging
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        logHandler.setFormatter(formatter)
        self.logger = logging.getLogger("AutoScaler")
        self.logger.addHandler(logHandler)
        self.logger.setLevel(logging.INFO)

        self.logger.info(
            "AutoScaler initialized",
            extra={
                "profile": profile.value,
                "scaling_interval": scaling_interval,
                "predictive_enabled": enable_predictive,
            },
        )

    def register_scaling_callback(self, component: str, callback: Callable[[Any], None]) -> None:
        """Register callback for applying scaling decisions"""
        self.scaling_callbacks[component] = callback

    async def start_scaling(self) -> None:
        """Start automatic scaling"""
        if self.running:
            return

        self.running = True
        self.scaling_task = asyncio.create_task(self._scaling_loop())

        self.logger.info("Auto-scaling started")

    async def stop_scaling(self) -> None:
        """Stop automatic scaling"""
        self.running = False

        if self.scaling_task:
            self.scaling_task.cancel()
        try:
            await self.scaling_task
        except asyncio.CancelledError:
            pass

        self.logger.info("Auto-scaling stopped")

    async def _scaling_loop(self) -> None:
        """Main scaling evaluation loop"""
        while self.running:
            try:
                # Collect current metrics
                await self._collect_system_metrics()

                # Analyze and make scaling decisions
                # scaling_decisions = await self._evaluate_scaling_needs()

                # Apply scaling decisions
                # for decision in scaling_decisions:
                #     await self._apply_scaling_decision(decision)

                self.last_scaling_check = time.time()

            except Exception as e:
                self.logger.error("Scaling evaluation failed", extra={"error": str(e)})

            await asyncio.sleep(self.scaling_interval)

    async def _collect_system_metrics(self) -> None:
        """Collect current system metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()

            # Mock throughput and response time (would come from health checker)
            throughput = 100.0  # events/sec
            response_time = 50.0  # ms
            queue_size = 100  # items

            self.metrics_analyzer.add_metrics(cpu_percent, memory.percent, throughput, response_time, queue_size)

        except Exception as e:
            self.logger.error("Metrics collection failed", extra={"error": str(e)})

    async def _evaluate_scaling_needs(self) -> list[ScalingDecision]:
        """Evaluate scaling needs and create decisions"""
        decisions = []

        # Get current system state
        trends = self.metrics_analyzer.analyze_trends()
        if trends.get("insufficient_data"):
            return decisions

        # Current resource usage
        cpu_current = trends["cpu"]["current"]
        memory_current = trends["memory"]["current"]
        response_time_current = trends["response_time"]["current"]
        queue_size_current = trends["queue_size"]["current"]

        limits = self.profile_manager.get_current_limits()

        # Memory-based deque scaling
        if memory_current > 85:
            decision = self.component_scaler.scale_deque_size(
                0.7,
                f"High memory usage: {memory_current:.1f}%",
                confidence=0.8,  # Reduce by 30%
            )
        if decision:
            decisions.append(decision)
        elif memory_current < 40 and queue_size_current > self.component_scaler.current_settings["deque_size"] * 0.8:
            decision = self.component_scaler.scale_deque_size(
                1.3,  # Increase by 30%
                "Low memory usage with high queue utilization",
                confidence=0.7,
            )
        if decision:
            decisions.append(decision)

        # CPU-based worker thread scaling
        if cpu_current > 80 and trends["cpu"]["trend"] == "increasing":
            decision = self.component_scaler.scale_worker_threads(
                1.5,  # Increase by 50%
                f"High CPU usage: {cpu_current:.1f}% with increasing trend",
                confidence=0.8,
            )
        if decision:
            decisions.append(decision)
        elif cpu_current < 30 and trends["cpu"]["trend"] == "stable":
            decision = self.component_scaler.scale_worker_threads(
                0.8,  # Reduce by 20%
                f"Low CPU usage: {cpu_current:.1f}% with stable trend",
                confidence=0.6,
            )
        if decision:
            decisions.append(decision)

        # Response time-based batch scaling
        if response_time_current > 1000:  # > 1 second
            decision = self.component_scaler.scale_batch_size(
                0.7,  # Reduce batch size for lower latency
                f"High response time: {response_time_current:.1f}ms",
                confidence=0.7,
            )
        if decision:
            decisions.append(decision)
        elif response_time_current < 100 and trends["throughput"]["trend"] == "increasing":
            decision = self.component_scaler.scale_batch_size(
                1.3,  # Increase batch size for better throughput
                "Low response time with increasing throughput",
                confidence=0.6,
            )
        if decision:
            decisions.append(decision)

        # Disk-based compression scaling
        try:
            disk = psutil.disk_usage("/")
            if disk.percent > 80:
                decision = self.component_scaler.apply_aggressive_compression(f"High disk usage: {disk.percent:.1f}%")
                if decision:
                    decisions.append(decision)
        except Exception:
            pass

        # Predictive scaling
        if self.enable_predictive:
            predictions = self.metrics_analyzer.predict_resource_needs()

        if predictions.get("cpu") == "scale_up_soon":
            decision = self.component_scaler.scale_worker_threads(
                1.2,  # Preemptive increase
                "Predictive scaling: CPU increase predicted",
                confidence=0.5,
            )
        if decision:
            decisions.append(decision)

        return decisions

    async def _apply_scaling_decision(self, decision: ScalingDecision) -> None:
        """Apply a scaling decision"""
        try:
            component = decision.component

            # Apply through registered callback
            if component in self.scaling_callbacks:
                self.scaling_callbacks[component](decision.new_value)
                self.stats["successful_scalings"] += 1

                self.logger.info(
                    "Scaling applied",
                    extra={
                        "component": component,
                        "action": decision.action.value,
                        "previous_value": decision.previous_value,
                        "new_value": decision.new_value,
                        "reason": decision.reason,
                        "confidence": decision.confidence,
                    },
                )
            else:
                self.logger.warning(
                    "No callback for component",
                    extra={"component": component, "decision": decision.to_dict()},
                )

            # Update statistics
            self.stats["total_scaling_actions"] += 1
            if component not in self.stats["scaling_by_component"]:
                self.stats["scaling_by_component"][component] = 0
            self.stats["scaling_by_component"][component] += 1

        except Exception as e:
            self.logger.error(
                "Failed to apply scaling decision",
                extra={"decision": decision.to_dict(), "error": str(e)},
            )

    def update_metrics(
        self,
        cpu_percent: float,
        memory_percent: float,
        throughput: float,
        response_time_ms: float,
        queue_size: int,
    ) -> None:
        """External API to update metrics"""
        self.metrics_analyzer.add_metrics(cpu_percent, memory_percent, throughput, response_time_ms, queue_size)

    def get_current_settings(self) -> dict[str, Any]:
        """Get current scaling settings and state"""
        return {
            "profile": self.profile_manager.get_profile_info(),
            "component_settings": self.component_scaler.get_current_settings(),
            "scaling_enabled": self.running,
            "last_check": self.last_scaling_check,
            "statistics": self.stats,
            "predictive_enabled": self.enable_predictive,
        }

    def get_scaling_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Get recent scaling history"""
        return self.component_scaler.get_scaling_history(limit)

    def get_trends_analysis(self) -> dict[str, Any]:
        """Get current trends analysis"""
        return self.metrics_analyzer.analyze_trends()

    def set_profile(self, profile: ScalingProfile) -> None:
        """Change scaling profile"""
        self.profile_manager.set_profile(profile)
        limits = self.profile_manager.get_current_limits()
        self.component_scaler = ComponentScaler(limits)

        self.logger.info(
            "Scaling profile changed",
            extra={"new_profile": profile.value, "limits": asdict(limits)},
        )


# Example usage and demonstration
async def main():
    """Demo auto-scaler functionality"""
    scaler = AutoScaler(profile=ScalingProfile.DEVELOPMENT, scaling_interval=5.0, enable_predictive=True)

    print("⚡ Jeffrey OS Auto-Scaler Demo")
    print("=" * 35)

    # Register mock callbacks
    def mock_deque_callback(new_size):
        print(f"  🔧 Deque size adjusted to: {new_size}")

    def mock_worker_callback(new_threads):
        print(f"  🔧 Worker threads adjusted to: {new_threads}")

    def mock_batch_callback(new_batch):
        print(f"  🔧 Batch size adjusted to: {new_batch}")

    scaler.register_scaling_callback("deque_size", mock_deque_callback)
    scaler.register_scaling_callback("worker_threads", mock_worker_callback)
    scaler.register_scaling_callback("batch_size", mock_batch_callback)

    # Start scaling
    await scaler.start_scaling()

    print("📊 Simulating load patterns...")

    # Simulate increasing load
    for i in range(15):
        cpu = 30 + i * 5  # Gradually increasing CPU
        memory = 40 + i * 3  # Gradually increasing memory
        throughput = 100 - i * 2  # Decreasing throughput
        response_time = 50 + i * 10  # Increasing response time
        queue_size = 100 + i * 20  # Growing queue

        scaler.update_metrics(cpu, memory, throughput, response_time, queue_size)

        if i % 3 == 0:
            print(f"  📈 Load step {i // 3 + 1}: CPU={cpu}%, Memory={memory}%, RT={response_time}ms")

        await asyncio.sleep(1)

    # Get final state
    settings = scaler.get_current_settings()
    history = scaler.get_scaling_history()

    print("\n📋 Final Settings:")
    print(f"  Profile: {settings['profile']['profile']}")
    print(f"  Deque Size: {settings['component_settings']['deque_size']}")
    print(f"  Worker Threads: {settings['component_settings']['worker_threads']}")
    print(f"  Batch Size: {settings['component_settings']['batch_size']}")

    print(f"\n📊 Scaling Actions: {len(history)}")
    for action in history[-3:]:  # Show last 3 actions
        print(
            f"  • {action['component']}: {action['previous_value']} → {action['new_value']} ({action['reason'][:50]}...)"
        )

    await scaler.stop_scaling()
    print("\n✅ Auto-scaling demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
