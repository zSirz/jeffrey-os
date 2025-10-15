#!/usr/bin/env python3

"""
Système de monitoring pour Jeffrey
Surveille les performances, les ressources et la santé du système
"""

import json
import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import psutil

# Configuration du logging
logging.basicConfig(
    filename=f'logs/monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


class StructuredLogger:
    """
    Logger structuré pour Jeffrey OS.
    Compatible avec le système de monitoring.
    """

    def __init__(self, name: str):
        """Initialise le logger structuré."""
        self.name = name
        self.logger = logging.getLogger(name)
        self.events = []

    async def log(self, level: str, event: str, data: dict = None):
        """
        Log un événement structuré.

        Args:
            level: Niveau de log (info, warning, error, debug)
            event: Nom de l'événement
            data: Données associées à l'événement
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'name': self.name,
            'level': level,
            'event': event,
            'data': data or {},
        }

        # Sauvegarder l'événement
        self.events.append(log_entry)

        # Logger selon le niveau
        message = f"[{event}] {json.dumps(data) if data else ''}"

        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
        elif level == "debug":
            self.logger.debug(message)
        else:
            self.logger.info(message)

    def get_events(self, event_type: str = None) -> list:
        """Récupère les événements loggés."""
        if event_type:
            return [e for e in self.events if e['event'] == event_type]
        return self.events

    def clear_events(self):
        """Efface l'historique des événements."""
        self.events = []


@dataclass
class SystemMetrics:
    """Métriques système de base."""

    cpu_percent: float
    memory_used_mb: float
    disk_usage_percent: float
    network_io: dict[str, float]
    timestamp: datetime


class MonitoringSystem(ABC):
    """Interface de base pour le monitoring."""

    @abstractmethod
    def collect_metrics(self) -> SystemMetrics:
        """Collecte les métriques du système."""
        pass

    @abstractmethod
    def check_health(self) -> dict[str, Any]:
        """Vérifie la santé du système."""
        pass

    @abstractmethod
    def generate_report(self) -> dict[str, Any]:
        """Génère un rapport de monitoring."""
        pass


class JeffreyMonitor(MonitoringSystem):
    """Système de monitoring spécifique à Jeffrey."""

    def __init__(self, process_name: str = "jeffrey.py"):
        """Initialise le moniteur."""
        self.process_name = process_name
        self.process = None
        self.metrics_history: list[SystemMetrics] = []
        self.alert_thresholds = {"cpu_percent": 80.0, "memory_percent": 75.0, "disk_percent": 90.0}
        self._find_process()

    def _find_process(self) -> None:
        """Trouve le processus Jeffrey."""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            if self.process_name in ' '.join(proc.info['cmdline'] or []):
                self.process = psutil.Process(proc.info['pid'])
                break

    def collect_metrics(self) -> SystemMetrics:
        """Collecte les métriques système."""
        if not self.process:
            self._find_process()
            if not self.process:
                raise RuntimeError("Processus Jeffrey non trouvé")

        # Collecter les métriques
        metrics = SystemMetrics(
            cpu_percent=self.process.cpu_percent(),
            memory_used_mb=self.process.memory_info().rss / 1024 / 1024,
            disk_usage_percent=psutil.disk_usage('/').percent,
            network_io=dict(psutil.net_io_counters()._asdict()),
            timestamp=datetime.now(),
        )

        # Sauvegarder l'historique
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > 1000:  # Garder les 1000 dernières métriques
            self.metrics_history.pop(0)

        return metrics

    def check_health(self) -> dict[str, Any]:
        """Vérifie la santé du système."""
        metrics = self.collect_metrics()
        health_status = {"status": "healthy", "alerts": [], "timestamp": datetime.now().isoformat()}

        # Vérifier les seuils
        if metrics.cpu_percent > self.alert_thresholds["cpu_percent"]:
            health_status["alerts"].append({"type": "cpu", "message": f"CPU usage high: {metrics.cpu_percent}%"})

        if metrics.memory_used_mb / psutil.virtual_memory().total * 100 > self.alert_thresholds["memory_percent"]:
            health_status["alerts"].append(
                {"type": "memory", "message": f"Memory usage high: {metrics.memory_used_mb:.1f}MB"}
            )

        if metrics.disk_usage_percent > self.alert_thresholds["disk_percent"]:
            health_status["alerts"].append(
                {"type": "disk", "message": f"Disk usage high: {metrics.disk_usage_percent}%"}
            )

        # Mettre à jour le statut
        if health_status["alerts"]:
            health_status["status"] = "warning"

        return health_status

    def generate_report(self) -> dict[str, Any]:
        """Génère un rapport de monitoring."""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}

        # Calculer les statistiques
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_used_mb for m in self.metrics_history]

        report = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "cpu": {
                    "current": cpu_values[-1],
                    "average": sum(cpu_values) / len(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values),
                },
                "memory": {
                    "current_mb": memory_values[-1],
                    "average_mb": sum(memory_values) / len(memory_values),
                    "max_mb": max(memory_values),
                    "min_mb": min(memory_values),
                },
                "disk": {"usage_percent": self.metrics_history[-1].disk_usage_percent},
                "network": self.metrics_history[-1].network_io,
            },
            "health": self.check_health(),
            "process_info": {
                "pid": self.process.pid if self.process else None,
                "status": self.process.status() if self.process else "not found",
                "uptime": (datetime.now() - datetime.fromtimestamp(self.process.create_time())).total_seconds()
                if self.process
                else 0,
            },
        }

        return report


class MonitoringDaemon:
    """Daemon de monitoring en arrière-plan."""

    def __init__(self, interval: int = 60):
        """Initialise le daemon."""
        self.monitor = JeffreyMonitor()
        self.interval = interval
        self.running = False
        self.thread: threading.Thread | None = None

    def start(self):
        """Démarre le daemon de monitoring."""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._monitoring_loop)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        """Arrête le daemon de monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitoring_loop(self):
        """Boucle principale de monitoring."""
        while self.running:
            try:
                # Collecter les métriques
                metrics = self.monitor.collect_metrics()

                # Vérifier la santé
                health = self.monitor.check_health()

                # Logguer les alertes
                if health["alerts"]:
                    for alert in health["alerts"]:
                        logging.warning(f"Alert: {alert['message']}")

                # Générer et sauvegarder le rapport
                report = self.monitor.generate_report()
                with open(f'reports/monitoring_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                    json.dump(report, f, indent=2)

            except Exception as e:
                logging.error(f"Error in monitoring loop: {str(e)}")

            time.sleep(self.interval)


def main():
    """Point d'entrée principal."""
    # Créer les dossiers nécessaires
    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)

    # Démarrer le daemon
    daemon = MonitoringDaemon(interval=60)  # Collecte toutes les minutes
    try:
        daemon.start()
        logging.info("Monitoring daemon started")

        # Garder le programme en vie
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        logging.info("Stopping monitoring daemon...")
        daemon.stop()
        logging.info("Monitoring daemon stopped")


if __name__ == '__main__':
    main()
