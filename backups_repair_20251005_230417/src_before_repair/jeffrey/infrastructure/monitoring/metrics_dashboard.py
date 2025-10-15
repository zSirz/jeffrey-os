"""
Tableau de bord des métriques système.

Ce module implémente les fonctionnalités essentielles pour tableau de bord des métriques système.
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
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from rich.align import Align
    from rich.columns import Columns
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (BarColumn, Progress, SpinnerColumn,
                               TaskProgress, TextColumn)
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

import logging

from auto_scaler import AutoScaler, ScalingProfile
from circuit_breaker import CircuitBreakerManager, CircuitState
# Import monitoring components
from health_checker import HealthChecker, HealthStatus
from python_json_logger import jsonlogger


class DashboardMode(Enum):
    """Dashboard display modes"""
    OVERVIEW = "overview"
    HEALTH = "health"
    SCALING = "scaling"
    CIRCUITS = "circuits"
    LOGS = "logs"


@dataclass
class DashboardConfig:
    """Dashboard configuration"""
    refresh_rate: float = 1.0
    max_log_lines: int = 50
    chart_width: int = 60
    chart_height: int = 10
    enable_colors: bool = True
    compact_mode: bool = False


class ASCIIChart:
    """Simple ASCII chart generator for terminal display"""

    @staticmethod
    def line_chart(data: List[float], width: int = 60, height: int = 10, title: str = "") -> str:
        """Generate ASCII line chart"""
        if not data or len(data) < 2:
            return f"{title}\n" + "No data available"

        # Normalize data to chart height
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1

        normalized = [(val - min_val) / range_val * (height - 1) for val in data]

        # Create chart grid
        chart_lines = []
        for y in range(height):
            line = ""
            for x in range(min(width, len(data))):
                data_index = int(x * len(data) / width)
                if data_index < len(normalized):
                    if int(normalized[data_index]) == (height - 1 - y):
                        line += "●"
                    elif y == height - 1:
                        line += "─"
                    else:
                        line += " "
                else:
                    line += " "
            chart_lines.append(line)

        # Add title and labels
        result = f"{title}\n"
        result += f"Max: {max_val:.2f}\n"
        result += "\n".join(chart_lines)
        result += f"\nMin: {min_val:.2f}"

        return result

    @staticmethod
    def bar_chart(data: Dict[str, float], width: int = 40, title: str = "") -> str:
        """Generate ASCII bar chart"""
        if not data:
            return f"{title}\nNo data available"

        max_val = max(data.values()) if data.values() else 1
        max_label_len = max(len(k) for k in data.keys()) if data else 0

        result = f"{title}\n"
        for label, value in data.items():
            bar_len = int((value / max_val) * width) if max_val > 0 else 0
            bar = "█" * bar_len + "░" * (width - bar_len)
            result += f"{label:<{max_label_len}} │{bar}│ {value:.1f}\n"

        return result.rstrip()

    @staticmethod
    def gauge(value: float, max_value: float = 100, width: int = 30, label: str = "") -> str:
        """Generate ASCII gauge"""
        percentage = (value / max_value) * 100 if max_value > 0 else 0
        filled = int((percentage / 100) * width)

        # Color coding for gauge
        if percentage < 50:
            gauge_char = "█"
        elif percentage < 80:
            gauge_char = "▓"
        else:
            gauge_char = "▒"

        bar = gauge_char * filled + "░" * (width - filled)
        return f"{label}: [{bar}] {percentage:.1f}%"


class TerminalDashboard:
    """
    Terminal-based dashboard with ASCII charts and real-time updates
    Falls back to simple text if Rich is not available
    """

    def __init__(
        self,
        config: DashboardConfig = None,
        health_checker: Optional[HealthChecker] = None,
        auto_scaler: Optional[AutoScaler] = None,
        circuit_manager: Optional[CircuitBreakerManager] = None
    ):
        """
        Initialize terminal dashboard

        Args:
            config: Dashboard configuration
            health_checker: Health monitoring component
            auto_scaler: Auto-scaling component
            circuit_manager: Circuit breaker management
        """
        self.config = config or DashboardConfig()
        self.health_checker = health_checker
        self.auto_scaler = auto_scaler
        self.circuit_manager = circuit_manager

        # Dashboard state
        self.current_mode = DashboardMode.OVERVIEW
        self.running = False
        self.console = Console() if RICH_AVAILABLE else None

        # Data storage for charts
        self.cpu_history: List[float] = []
        self.memory_history: List[float] = []
        self.throughput_history: List[float] = []
        self.response_time_history: List[float] = []
        self.log_buffer: List[str] = []

        # Threading
        self._data_lock = threading.Lock()

        # Setup logging capture
        self._setup_log_capture()

        # Terminal size
        self.terminal_width, self.terminal_height = self._get_terminal_size()

        logging.info("Metrics Dashboard initialized", extra={
            "rich_available": RICH_AVAILABLE,
            "terminal_size": f"{self.terminal_width}x{self.terminal_height}",
            "mode": self.current_mode.value
        })

    def _setup_log_capture(self) -> None:
        """Setup log capture for dashboard display"""
        class DashboardLogHandler(logging.Handler):
    """
    Classe DashboardLogHandler pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """
            def __init__(self, dashboard) -> None:
                super().__init__()
                self.dashboard = dashboard

            def emit(self, record):
                log_entry = self.format(record)
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted = f"[{timestamp}] {log_entry}"

                with self.dashboard._data_lock:
                    self.dashboard.log_buffer.append(formatted)
                    if len(self.dashboard.log_buffer) > self.dashboard.config.max_log_lines:
                        self.dashboard.log_buffer.pop(0)

        # Add handler to root logger
        handler = DashboardLogHandler(self)
        handler.setFormatter(logging.Formatter('%(name)s: %(message)s'))
        logging.getLogger().addHandler(handler)

    def _get_terminal_size(self) -> Tuple[int, int]:
        """Get terminal dimensions"""
        try:
            size = shutil.get_terminal_size()
            return size.columns, size.lines
        except Exception:
            return 80, 24  # Default fallback

    async def start(self) -> None:
        """Start the dashboard"""
        if self.running:
            return

        self.running = True

        if RICH_AVAILABLE:
            await self._run_rich_dashboard()
        else:
            await self._run_simple_dashboard()

    async def stop(self) -> None:
        """Stop the dashboard"""
        self.running = False

    async def _run_rich_dashboard(self) -> None:
        """Run Rich-based dashboard with live updates"""
        try:
            with Live(self._create_rich_layout(), refresh_per_second=1/self.config.refresh_rate, console=self.console) as live:
                while self.running:
                    # Update data
                    await self._update_data()

                    # Update layout
                    live.update(self._create_rich_layout())

                    await asyncio.sleep(self.config.refresh_rate)

        except KeyboardInterrupt:
            self.running = False
        except Exception as e:
            logging.error(f"Dashboard error: {e}")

    async def _run_simple_dashboard(self) -> None:
        """Run simple text-based dashboard"""
        try:
            while self.running:
                # Clear screen (simple method)
                os.system('clear' if os.name == 'posix' else 'cls')

                # Update data
                await self._update_data()

                # Display dashboard
                print(self._create_simple_layout())

                await asyncio.sleep(self.config.refresh_rate)

        except KeyboardInterrupt:
            self.running = False
        except Exception as e:
            logging.error(f"Dashboard error: {e}")

    async def _update_data(self) -> None:
        """Update dashboard data from monitoring components"""
        try:
            # Update from health checker
            if self.health_checker:
                health = self.health_checker.get_current_health()
                if health.get('metrics'):
                    metrics = health['metrics']

                    with self._data_lock:
                        self.cpu_history.append(metrics.get('cpu_percent', 0))
                        self.memory_history.append(metrics.get('memory_percent', 0))
                        self.throughput_history.append(metrics.get('event_processing_rate', 0))
                        self.response_time_history.append(metrics.get('response_time_p95', 0))

                        # Keep only recent data
                        max_points = self.config.chart_width
                        self.cpu_history = self.cpu_history[-max_points:]
                        self.memory_history = self.memory_history[-max_points:]
                        self.throughput_history = self.throughput_history[-max_points:]
                        self.response_time_history = self.response_time_history[-max_points:]

        except Exception as e:
            logging.error(f"Data update failed: {e}")

    def _create_rich_layout(self) -> Layout:
        """Create Rich layout for dashboard"""
        if not RICH_AVAILABLE:
            return None

        layout = Layout()

        if self.current_mode == DashboardMode.OVERVIEW:
            layout.split_column(
                Layout(self._create_header_panel(), size=3),
                Layout().split_row(
                    Layout(self._create_system_metrics_panel()),
                    Layout(self._create_performance_panel())
                ),
                Layout().split_row(
                    Layout(self._create_health_status_panel()),
                    Layout(self._create_alerts_panel())
                ),
                Layout(self._create_footer_panel(), size=3)
            )
        elif self.current_mode == DashboardMode.HEALTH:
            layout.split_column(
                Layout(self._create_header_panel(), size=3),
                Layout(self._create_detailed_health_panel()),
                Layout(self._create_footer_panel(), size=3)
            )
        # Add other modes as needed

        return layout

    def _create_simple_layout(self) -> str:
        """Create simple text layout for dashboard"""
        lines = []
        lines.append("=" * self.terminal_width)
        lines.append("JEFFREY OS - MONITORING DASHBOARD")
        lines.append(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Mode: {self.current_mode.value.upper()}")
        lines.append("=" * self.terminal_width)

        if self.current_mode == DashboardMode.OVERVIEW:
            lines.extend(self._create_simple_overview())
        elif self.current_mode == DashboardMode.HEALTH:
            lines.extend(self._create_simple_health())

        lines.append("=" * self.terminal_width)
        lines.append("Controls: [O]verview [H]ealth [S]caling [C]ircuits [L]ogs [Q]uit")

        return "\n".join(lines)

    def _create_simple_overview(self) -> List[str]:
        """Create simple overview display"""
        lines = []

        # System metrics
        lines.append("\nSYSTEM METRICS:")
        if self.cpu_history:
            lines.append(ASCIIChart.gauge(self.cpu_history[-1], 100, 30, "CPU Usage"))
        if self.memory_history:
            lines.append(ASCIIChart.gauge(self.memory_history[-1], 100, 30, "Memory Usage"))

        # Performance chart
        if len(self.cpu_history) > 1:
            lines.append("\nCPU HISTORY:")
            lines.append(ASCIIChart.line_chart(self.cpu_history[-20:], 60, 8))

        # Health status
        if self.health_checker:
            health = self.health_checker.get_current_health()
            lines.append(f"\nHEALTH STATUS: {health.get('status', 'unknown').upper()}")
            if health.get('active_alerts'):
                lines.append(f"Active Alerts: {len(health['active_alerts'])}")

        # Circuit breakers
        if self.circuit_manager:
            health = self.circuit_manager.get_system_health()
            lines.append(f"\nCIRCUIT BREAKERS: {health.get('status', 'unknown').upper()}")
            lines.append(f"Total Circuits: {health.get('total_circuits', 0)}")

        return lines

    def _create_simple_health(self) -> List[str]:
        """Create simple health display"""
        lines = []

        if self.health_checker:
            health = self.health_checker.get_current_health()
            lines.append(f"\nHEALTH STATUS: {health.get('status', 'unknown').upper()}")

            if health.get('metrics'):
                metrics = health['metrics']
                lines.append(f"CPU: {metrics.get('cpu_percent', 0):.1f}%")
                lines.append(f"Memory: {metrics.get('memory_percent', 0):.1f}%")
                lines.append(f"Disk: {metrics.get('disk_percent', 0):.1f}%")
                lines.append(f"Event Rate: {metrics.get('event_processing_rate', 0):.1f}/sec")
                lines.append(f"Response Time (P95): {metrics.get('response_time_p95', 0):.1f}ms")

        return lines

    def _create_header_panel(self) -> Panel:
        """Create header panel for Rich display"""
        if not RICH_AVAILABLE:
            return None

        header_text = Text("JEFFREY OS - MONITORING DASHBOARD", style="bold blue")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        mode_text = f"Mode: {self.current_mode.value.upper()} | Time: {timestamp}"

        return Panel(
            Align.center(header_text + "\n" + mode_text),
            title="🚀 Jeffrey OS v0.6.1",
            border_style="blue"
        )

    def _create_system_metrics_panel(self) -> Panel:
        """Create system metrics panel"""
        if not RICH_AVAILABLE:
            return None

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Current", justify="right")
        table.add_column("Status", justify="center")

        # Add system metrics
        if self.cpu_history:
            cpu = self.cpu_history[-1]
            status = "🟢" if cpu < 70 else "🟡" if cpu < 90 else "🔴"
            table.add_row("CPU Usage", f"{cpu:.1f}%", status)

        if self.memory_history:
            memory = self.memory_history[-1]
            status = "🟢" if memory < 80 else "🟡" if memory < 95 else "🔴"
            table.add_row("Memory Usage", f"{memory:.1f}%", status)

        if self.throughput_history:
            throughput = self.throughput_history[-1]
            table.add_row("Event Rate", f"{throughput:.1f}/sec", "🟢")

        return Panel(table, title="📊 System Metrics", border_style="green")

    def _create_performance_panel(self) -> Panel:
        """Create performance panel with charts"""
        if not RICH_AVAILABLE:
            return None

        content = "Performance trends:\n\n"

        if len(self.cpu_history) > 1:
            content += ASCIIChart.line_chart(
                self.cpu_history[-20:],
                width=40,
                height=6,
                title="CPU Usage (20 samples)"
            )
            content += "\n\n"

        if len(self.response_time_history) > 1:
            content += ASCIIChart.line_chart(
                self.response_time_history[-20:],
                width=40,
                height=6,
                title="Response Time (ms)"
            )

        return Panel(content, title="📈 Performance", border_style="yellow")

    def _create_health_status_panel(self) -> Panel:
        """Create health status panel"""
        if not RICH_AVAILABLE:
            return None

        content = ""

        if self.health_checker:
            health = self.health_checker.get_current_health()
            status = health.get('status', 'unknown')

            status_emoji = {
                'healthy': '🟢',
                'degraded': '🟡',
                'unhealthy': '🟠',
                'critical': '🔴'
            }.get(status, '⚪')

            content += f"Status: {status_emoji} {status.upper()}\n\n"

            if health.get('alert_summary'):
                summary = health['alert_summary']
                content += f"Active Alerts: {summary.get('active_count', 0)}\n"
                content += f"Today's Alerts: {summary.get('total_alerts_today', 0)}\n"
        else:
            content = "Health checker not available"

        return Panel(content, title="🏥 Health Status", border_style="blue")

    def _create_alerts_panel(self) -> Panel:
        """Create alerts panel"""
        if not RICH_AVAILABLE:
            return None

        content = ""

        # Show recent log entries
        with self._data_lock:
            recent_logs = self.log_buffer[-10:] if self.log_buffer else []

        if recent_logs:
            content = "\n".join(recent_logs[-5:])  # Show last 5 logs
        else:
            content = "No recent alerts"

        return Panel(content, title="🚨 Recent Alerts", border_style="red")

    def _create_detailed_health_panel(self) -> Panel:
        """Create detailed health panel"""
        if not RICH_AVAILABLE:
            return None

        content = "Detailed health information:\n\n"

        if self.health_checker:
            health = self.health_checker.get_current_health()

            if health.get('metrics'):
                metrics = health['metrics']
                content += f"CPU: {metrics.get('cpu_percent', 0):.1f}%\n"
                content += f"Memory: {metrics.get('memory_percent', 0):.1f}%\n"
                content += f"Disk: {metrics.get('disk_percent', 0):.1f}%\n"
                content += f"Event Processing Rate: {metrics.get('event_processing_rate', 0):.1f}/sec\n"
                content += f"Response Time P95: {metrics.get('response_time_p95', 0):.1f}ms\n"
                content += f"Response Time P99: {metrics.get('response_time_p99', 0):.1f}ms\n"
                content += f"Queue Size: {metrics.get('queue_size', 0)}\n"
                content += f"Error Rate: {metrics.get('error_rate', 0):.1f}%\n"

        return Panel(content, title="🔍 Detailed Health", border_style="cyan")

    def _create_footer_panel(self) -> Panel:
        """Create footer panel"""
        if not RICH_AVAILABLE:
            return None

        footer_text = "Controls: [O]verview [H]ealth [S]caling [C]ircuits [L]ogs [Q]uit"
        return Panel(
            Align.center(footer_text),
            border_style="white"
        )

    def switch_mode(self, mode: DashboardMode) -> None:
        """Switch dashboard mode"""
        self.current_mode = mode
        logging.info(f"Dashboard mode changed to: {mode.value}")

    def export_snapshot(self, filename: Optional[str] = None) -> str:
        """Export current dashboard state to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = filename or f"dashboard_snapshot_{timestamp}.json"

        snapshot = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mode": self.current_mode.value,
            "system_metrics": {
                "cpu_history": self.cpu_history[-50:],  # Last 50 points
                "memory_history": self.memory_history[-50:],
                "throughput_history": self.throughput_history[-50:],
                "response_time_history": self.response_time_history[-50:]
            },
            "health_status": self.health_checker.get_current_health() if self.health_checker else None,
            "scaling_status": self.auto_scaler.get_current_settings() if self.auto_scaler else None,
            "circuit_status": self.circuit_manager.get_system_health() if self.circuit_manager else None,
            "recent_logs": self.log_buffer[-20:] if self.log_buffer else []
        }

        try:
            with open(filename, 'w') as f:
                json.dump(snapshot, f, indent=2, default=str)

            logging.info(f"Dashboard snapshot exported to: {filename}")
            return filename

        except Exception as e:
            logging.error(f"Failed to export snapshot: {e}")
            return ""


# Example usage and demonstration
async def main():
    """Demo dashboard functionality"""
    print("📺 Jeffrey OS Metrics Dashboard Demo")
    print("=" * 40)

    # Create mock components for demo
    from auto_scaler import AutoScaler
    from circuit_breaker import CircuitBreakerManager
    from health_checker import HealthChecker

    # Initialize components
    health_checker = HealthChecker(check_interval=2.0)
    auto_scaler = AutoScaler(scaling_interval=10.0)
    circuit_manager = CircuitBreakerManager()

    # Create dashboard
    dashboard = TerminalDashboard(
        config=DashboardConfig(refresh_rate=1.0),
        health_checker=health_checker,
        auto_scaler=auto_scaler,
        circuit_manager=circuit_manager
    )

    # Start monitoring components
    await health_checker.start_monitoring()
    await auto_scaler.start_scaling()

    # Create some test circuit breakers
    circuit_manager.create_circuit("test_service_1", failure_threshold=3)
    circuit_manager.create_circuit("test_service_2", failure_threshold=5)

    print(f"🚀 Starting dashboard...")
    print(f"Rich UI available: {RICH_AVAILABLE}")
    print(f"Press Ctrl+C to stop")

    try:
        # Start dashboard (will run until interrupted)
        await dashboard.start()

    except KeyboardInterrupt:
        print("\n⏹️  Stopping dashboard...")

    finally:
        # Cleanup
        await dashboard.stop()
        await health_checker.stop_monitoring()
        await auto_scaler.stop_scaling()

        # Export final snapshot
        snapshot_file = dashboard.export_snapshot()
        if snapshot_file:
            print(f"📄 Snapshot exported: {snapshot_file}")

    print("✅ Dashboard demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
