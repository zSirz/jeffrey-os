#!/usr/bin/env python3
"""
Générateur de charge production-grade pour Jeffrey OS
Inclut chaos testing avancé, monitoring système, adaptation ML, et corruption de données
"""

import argparse
import asyncio
import json
import os
import random
import signal
import statistics

# Import du manager NATS
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import psutil

sys.path.insert(0, str(Path(__file__).parent))
from nats_manager import NATSManager

# Prometheus metrics
from prometheus_client import Counter, Gauge, start_http_server

# Import avec le nouveau packaging (jeffrey.* pas src.jeffrey.*)
from jeffrey.core.loops.loop_manager import LoopManager
from jeffrey.core.neuralbus.connection_config import get_subject_with_namespace


# Configuration enrichie
@dataclass
class LoadConfig:
    """Configuration avancée du générateur de charge"""

    duration_hours: float = 2.0
    target_rate: int = 500

    # Chaos
    chaos_enabled: bool = False
    chaos_probability: float = 0.01
    chaos_adaptive: bool = True  # Augmente si système stable
    corruption_enabled: bool = False
    corruption_rate: float = 0.05
    malicious_enabled: bool = False
    malicious_rate: float = 0.05

    # Monitoring
    monitor_resources: bool = True
    monitor_interval: int = 30
    prometheus_port: int = 8000

    # ML Adaptive
    adaptive_ml: bool = True
    compression_threshold: int = 4096
    batch_size: int = 10

    # Limites
    max_pending: int = 5000
    min_pending: int = 100
    rate_adjust_factor: float = 0.05

    # Seuils par phase (adaptatifs)
    thresholds: dict[str, dict] = field(
        default_factory=lambda: {
            "quick": {"p99": 100, "drops": 1.0, "duration": 0.05},
            "stress": {"p99": 70, "drops": 0.7, "duration": 0.1},
            "chaos": {"p99": 100, "drops": 1.5, "duration": 0.1},
            "soak": {"p99": 50, "drops": 0.5, "duration": 2.0},
        }
    )

    # System limits
    memory_limit_percent: float = 85.0  # Plus safe sur macOS
    cpu_limit_percent: float = 85.0

    # Mode
    non_interactive: bool = False  # Pour CI/CD


# Métriques Prometheus
prom_messages_sent = Counter("jeffrey_messages_sent_total", "Total messages sent")
prom_messages_failed = Counter("jeffrey_messages_failed_total", "Total messages failed")
prom_chaos_events = Counter("jeffrey_chaos_events_total", "Total chaos events", ["type"])
prom_current_rate = Gauge("jeffrey_current_rate", "Current message rate")
prom_p99_latency = Gauge("jeffrey_p99_latency_ms", "P99 latency in ms")
prom_symbiosis = Gauge("jeffrey_symbiosis_score", "Symbiosis score")
prom_memory_mb = Gauge("jeffrey_memory_mb", "Memory usage in MB")
prom_cpu_percent = Gauge("jeffrey_cpu_percent", "CPU usage percent")


class SystemMonitor:
    """Moniteur système avancé avec détection de patterns"""

    def __init__(self, config: LoadConfig):
        self.config = config
        self.process = psutil.Process()
        self.start_memory = self.process.memory_info().rss
        self.memory_samples = deque(maxlen=120)
        self.cpu_samples = deque(maxlen=120)
        self.leak_threshold = 50  # MB

        # Prime CPU monitoring to avoid initial 0% readings
        self.process.cpu_percent(interval=None)  # Prime the CPU counter
        time.sleep(0.1)  # Wait a bit
        initial_cpu = self.process.cpu_percent(interval=None)
        if initial_cpu > 0:
            self.cpu_samples.append(initial_cpu)

    def get_metrics(self) -> dict[str, Any]:
        """Récupère métriques système avec analyse de tendance"""
        memory = self.process.memory_info()
        # Use interval=None for non-blocking, but more accurate CPU reading
        cpu_percent = self.process.cpu_percent(interval=None)
        # Fallback if still 0: use short interval
        if cpu_percent == 0:
            cpu_percent = self.process.cpu_percent(interval=0.1)

        # Échantillonnage
        self.memory_samples.append(memory.rss)
        self.cpu_samples.append(cpu_percent)

        # Analyse de tendance mémoire
        memory_trend = self._analyze_trend(self.memory_samples)

        # Métriques
        metrics = {
            "memory_mb": memory.rss / 1024 / 1024,
            "memory_percent": memory.rss / psutil.virtual_memory().total * 100,
            "memory_increase_mb": (memory.rss - self.start_memory) / 1024 / 1024,
            "memory_trend": memory_trend,
            "cpu_percent": cpu_percent,
            "cpu_avg": statistics.mean(self.cpu_samples) if self.cpu_samples else 0,
            "open_files": len(self.process.open_files()),
            "num_threads": self.process.num_threads(),
            "timestamp": datetime.now().isoformat(),
        }

        # Update Prometheus
        prom_memory_mb.set(metrics["memory_mb"])
        prom_cpu_percent.set(metrics["cpu_percent"])

        return metrics

    def _analyze_trend(self, samples: deque) -> str:
        """Analyse la tendance des échantillons"""
        if len(samples) < 20:
            return "insufficient_data"

        recent = list(samples)[-10:]
        older = list(samples)[-20:-10]

        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)

        if recent_avg > older_avg * 1.10:  # +10%
            return "increasing"
        elif recent_avg < older_avg * 0.90:  # -10%
            return "decreasing"
        else:
            return "stable"

    def check_limits(self) -> dict[str, bool]:
        """Vérifie les limites système"""
        metrics = self.get_metrics()

        return {
            "memory_ok": metrics["memory_percent"] < self.config.memory_limit_percent,
            "cpu_ok": metrics["cpu_percent"] < self.config.cpu_limit_percent,
            "leak_suspected": (
                metrics["memory_trend"] == "increasing" and metrics["memory_increase_mb"] > self.leak_threshold
            ),
        }


def generate_malicious_event():
    """Génère un événement malveillant pour tester la résilience"""
    attacks = [
        # Payload oversized
        {"type": "oversized", "data": "X" * 10000},
        # SQL injection attempt
        {"type": "user.query", "data": "'; DROP TABLE users; --"},
        # Invalid JSON structure
        {"type": None, "data": {"corrupted": True}},
        # Deeply nested object (stack overflow attempt)
        {"type": "nested", "data": {"level": {}} * 1000},
        # XSS attempt
        {"type": "display", "data": "<script>alert('XSS')</script>"},
        # Command injection
        {"type": "exec", "data": "; rm -rf /"},
        # Buffer overflow attempt
        {"type": "buffer", "data": "A" * 65536},
    ]
    return random.choice(attacks)


class ChaosMonkey:
    """Gestionnaire de chaos avancé avec corruption de données"""

    def __init__(self, config: LoadConfig, nats_manager: NATSManager):
        self.config = config
        self.nats_manager = nats_manager
        self.chaos_events = []
        self.base_probability = config.chaos_probability

    async def maybe_cause_chaos(self, manager: Any, symbiosis: float = 0.5) -> str | None:
        """Chaos avec probabilité adaptative"""
        if not self.config.chaos_enabled:
            return None

        # Adaptation : plus agressif si système stable
        probability = self.base_probability
        if self.config.chaos_adaptive and symbiosis > 0.6:
            probability *= 1.5  # 50% plus agressif si stable

        if random.random() > probability:
            return None

        # Types de chaos avec poids
        chaos_types = [
            ("nats_restart", 0.15),  # 15% - Plus critique
            ("high_burst", 0.25),  # 25%
            ("slow_consumer", 0.20),  # 20%
            ("memory_pressure", 0.15),  # 15%
            ("network_delay", 0.15),  # 15%
            ("consumer_crash", 0.10),  # 10% - Nouveau
        ]

        # Sélection pondérée
        chaos_type = random.choices([t[0] for t in chaos_types], weights=[t[1] for t in chaos_types])[0]

        event = {
            "type": chaos_type,
            "timestamp": datetime.now().isoformat(),
            "symbiosis_before": symbiosis,
            "recovered": False,
        }

        print(f"\n🔥 CHAOS EVENT: {chaos_type} (symbiosis: {symbiosis:.2f})")
        prom_chaos_events.labels(type=chaos_type).inc()

        try:
            if chaos_type == "nats_restart":
                # Utilise le manager pour restart safe
                success = self.nats_manager.chaos_restart()
                await asyncio.sleep(3)
                event["recovered"] = success

            elif chaos_type == "high_burst":
                # Burst adaptatif selon charge actuelle
                burst_size = random.randint(500, 2000)
                print(f"  → Sending burst of {burst_size} messages")

                tasks = []
                for i in range(burst_size):
                    if manager.event_bus:
                        tasks.append(
                            manager.event_bus.publish("chaos.burst", {"id": i, "data": "x" * random.randint(10, 100)})
                        )

                # Envoyer par batches pour éviter overwhelming
                for i in range(0, len(tasks), 100):
                    batch = tasks[i : i + 100]
                    await asyncio.gather(*batch, return_exceptions=True)
                    await asyncio.sleep(0.1)

                event["recovered"] = True

            elif chaos_type == "slow_consumer":
                delay = random.uniform(3, 8)
                print(f"  → Simulating slow consumer ({delay:.1f}s delay)")
                await asyncio.sleep(delay)
                event["recovered"] = True

            elif chaos_type == "memory_pressure":
                # Allocation contrôlée
                size_mb = random.randint(50, 150)
                print(f"  → Creating {size_mb}MB memory pressure")

                try:
                    # Utilise bytearray pour contrôle précis
                    blob = bytearray(size_mb * 1024 * 1024)
                    await asyncio.sleep(2)
                    del blob
                    event["recovered"] = True
                except MemoryError:
                    print("  ⚠️  Memory allocation failed (system protected)")
                    event["recovered"] = False

            elif chaos_type == "network_delay":
                delay = random.uniform(2, 5)
                print(f"  → Simulating network delay ({delay:.1f}s)")
                await asyncio.sleep(delay)
                event["recovered"] = True

            elif chaos_type == "consumer_crash":
                print("  → Simulating consumer crash (restart loops)")
                # Arrêt temporaire des loops
                if hasattr(manager, "stop"):
                    await manager.stop()
                    await asyncio.sleep(5)
                    await manager.start()
                    event["recovered"] = True
                else:
                    event["recovered"] = False

            if event["recovered"]:
                print(f"  ✅ Recovered from {chaos_type}")
            else:
                print(f"  ❌ Failed to recover from {chaos_type}")

        except Exception as e:
            print(f"  ❌ Chaos error: {e}")
            event["error"] = str(e)

        self.chaos_events.append(event)
        return chaos_type

    def corrupt_data(self, data: Any) -> tuple[Any, bool]:
        """Corrompt les données pour tester la résilience"""
        if not self.config.corruption_enabled:
            return data, False

        if random.random() > self.config.corruption_rate:
            return data, False

        # Types de corruption
        corruption_type = random.choice(["missing_field", "wrong_type", "invalid_json", "oversized", "null_data"])

        corrupted = data.copy() if isinstance(data, dict) else data

        try:
            if corruption_type == "missing_field" and isinstance(corrupted, dict):
                # Supprime un champ requis
                if corrupted:
                    key = random.choice(list(corrupted.keys()))
                    del corrupted[key]

            elif corruption_type == "wrong_type" and isinstance(corrupted, dict):
                # Change le type d'un champ
                if corrupted:
                    key = random.choice(list(corrupted.keys()))
                    corrupted[key] = "CORRUPTED_TYPE"

            elif corruption_type == "invalid_json":
                # Retourne une string invalide
                return "{{INVALID JSON{", True

            elif corruption_type == "oversized":
                # Ajoute un champ énorme
                if isinstance(corrupted, dict):
                    corrupted["corrupted_payload"] = "X" * (1024 * 1024)  # 1MB

            elif corruption_type == "null_data":
                return None, True

            return corrupted, True

        except:
            return data, False


class AdaptiveController:
    """Contrôleur ML adaptatif avec historique persistant"""

    def __init__(self, config: LoadConfig):
        self.config = config
        self.current_rate = config.target_rate
        self.rate_history = deque(maxlen=100)
        self.performance_history = deque(maxlen=100)
        self.adjustment_count = 0

    def adapt_rate(self, metrics: dict[str, Any], phase: str = "soak") -> float:
        """Adapte le rate selon métriques et phase"""
        # Récupérer seuils selon la phase
        thresholds = self.config.thresholds.get(phase, self.config.thresholds["soak"])
        p99_target = thresholds["p99"]

        # Métriques clés
        p99 = metrics.get("p99_latency_ms", 0)
        pending = metrics.get("pending_messages", 0)
        symbiosis = metrics.get("symbiosis_score", 0.5)
        dropped = metrics.get("dropped", 0)

        # Historique
        self.rate_history.append(self.current_rate)
        self.performance_history.append(
            {
                "p99": p99,
                "pending": pending,
                "symbiosis": symbiosis,
                "dropped": dropped,
                "timestamp": time.time(),
            }
        )

        # Logique d'adaptation
        adjustment = 1.0
        reasons = []

        # Latence
        if p99 > p99_target * 1.5:
            adjustment *= 0.7
            reasons.append(f"P99 critical ({p99:.1f}ms)")
        elif p99 > p99_target:
            adjustment *= 0.9
            reasons.append(f"P99 high ({p99:.1f}ms)")
        elif p99 < p99_target * 0.5:
            adjustment *= 1.1
            reasons.append(f"P99 low ({p99:.1f}ms), can increase")

        # Backpressure
        if pending > self.config.max_pending:
            adjustment *= 0.85
            reasons.append(f"Backpressure ({pending} pending)")
        elif pending < self.config.min_pending and p99 < p99_target * 0.7:
            adjustment *= 1.05
            reasons.append("System underutilized")

        # Symbiosis
        if symbiosis < 0.3:
            adjustment *= 0.6
            reasons.append(f"ML degradation (symbiosis={symbiosis:.2f})")
        elif symbiosis > 0.7 and p99 < p99_target:
            adjustment *= 1.05
            reasons.append(f"System healthy (symbiosis={symbiosis:.2f})")

        # Messages dropped
        if dropped > 100:
            adjustment *= 0.8
            reasons.append(f"High drop count ({dropped})")

        # Appliquer avec limites
        new_rate = self.current_rate * adjustment
        new_rate = max(10, min(new_rate, self.config.target_rate * 2))

        # Smooth pour éviter oscillations
        if len(self.rate_history) > 10:
            avg_recent = statistics.mean(list(self.rate_history)[-10:])
            new_rate = 0.7 * new_rate + 0.3 * avg_recent

        # Log si changement significatif
        if abs(new_rate - self.current_rate) > self.current_rate * 0.1:
            self.adjustment_count += 1
            print(f"  📊 Rate adjusted: {self.current_rate:.0f} → {new_rate:.0f} msg/s")
            if reasons:
                print(f"     Reasons: {', '.join(reasons)}")

        self.current_rate = new_rate
        prom_current_rate.set(new_rate)

        return new_rate


class LoadGenerator:
    """Générateur de charge principal production-grade"""

    def __init__(self, config: LoadConfig):
        self.config = config
        self.nats_manager = NATSManager()
        self.monitor = SystemMonitor(config)
        self.chaos = ChaosMonkey(config, self.nats_manager)
        self.controller = AdaptiveController(config)
        self.stats = {
            "messages_sent": 0,
            "errors": 0,
            "compressed": 0,
            "corrupted": 0,
            "chaos_events": 0,
            "rate_adjustments": 0,
            "dlq_messages": 0,
        }

        # Start Prometheus metrics server only if port specified
        if config.prometheus_port > 0:
            try:
                start_http_server(config.prometheus_port)
                print(f"📊 Prometheus metrics available at http://localhost:{config.prometheus_port}/metrics")
            except Exception as e:
                print(f"⚠️  Could not start Prometheus server on port {config.prometheus_port}: {e}")

    def get_event_templates(self) -> list[tuple]:
        """Templates d'événements avec distribution réaliste"""
        return [
            # 40% - Petits messages haute fréquence
            (
                "awareness.update",
                lambda i: {
                    "level": random.random(),
                    "timestamp": time.time(),
                    "cycle": i,
                    "state": random.choice(["active", "passive", "deep"]),
                    "metadata": {
                        "source": "load_test",
                        "session": os.getenv("NB_NS", "dev"),
                        "sequence": i,
                    },
                },
                0.40,
            ),
            # 25% - Messages moyens ML
            (
                "ml.inference",
                lambda i: {
                    "model": random.choice(["gpt", "claude", "custom"]),
                    "input_tokens": random.randint(10, 500),
                    "output_tokens": random.randint(10, 200),
                    "latency_ms": random.uniform(10, 100),
                    "confidence": random.random(),
                    "embeddings": [random.random() for _ in range(128)],
                    "timestamp": time.time(),
                },
                0.25,
            ),
            # 20% - Messages émotionnels moyens
            (
                "emotional.state",
                lambda i: {
                    "pleasure": random.random(),
                    "arousal": random.random(),
                    "dominance": random.random(),
                    "triggers": [f"trigger_{j}" for j in range(random.randint(5, 20))],
                    "context": "x" * random.randint(500, 2000),
                    "analysis": {
                        "confidence": random.random(),
                        "stability": random.random(),
                        "trend": random.choice(["increasing", "stable", "decreasing"]),
                    },
                    "timestamp": time.time(),
                },
                0.20,
            ),
            # 10% - Messages larges (test compression)
            (
                "memory.store",
                lambda i: {
                    "id": f"mem_{i}",
                    "content": f"Memory content {i}",
                    "embeddings": [random.random() for _ in range(768)],
                    "metadata": {
                        "source": "test",
                        "importance": random.random(),
                        "timestamp": time.time(),
                        "relations": [f"rel_{j}" for j in range(50)],
                        "tags": [f"tag_{j}" for j in range(100)],
                    },
                    "clusters": [random.random() for _ in range(128)],
                },
                0.10,
            ),
            # 5% - Messages système/alertes
            (
                "system.metric",
                lambda i: {
                    "type": random.choice(["health", "performance", "error", "warning"]),
                    "severity": random.choice(["info", "warning", "error", "critical"]),
                    "component": random.choice(["bus", "loops", "memory", "ml"]),
                    "value": random.random() * 100,
                    "threshold": random.random() * 100,
                    "message": f"Metric {i}",
                    "timestamp": time.time(),
                    "metadata": {"test_id": i, "batch": i // 100},
                },
                0.05,
            ),
        ]

    async def run(self, phase: str = "soak"):
        """Lance le générateur de charge pour une phase spécifique"""
        print("\n" + "=" * 70)
        print(f"🚀 JEFFREY OS - PRODUCTION LOAD GENERATOR - PHASE: {phase.upper()}")
        print("=" * 70)
        print(f"Duration: {self.config.duration_hours} hours")
        print(f"Target Rate: {self.config.target_rate} msg/sec")
        print(f"Chaos: {'ENABLED (Adaptive)' if self.config.chaos_enabled else 'Disabled'}")
        print(f"Corruption: {'ENABLED' if self.config.corruption_enabled else 'Disabled'}")
        print(f"ML Adaptive: {'ENABLED' if self.config.adaptive_ml else 'Disabled'}")
        print(f"Monitoring: {'ENABLED' if self.config.monitor_resources else 'Disabled'}")
        print(f"Namespace: {os.getenv('NB_NS', 'dev')}")
        print("=" * 70 + "\n")

        # Vérifier/Démarrer NATS avec gestion du cas externe
        if not self.nats_manager.is_running():
            print("📦 Starting NATS...")
            if not self.nats_manager.start():
                # Fallback: accepter un NATS externe déjà up
                if not self.nats_manager.health_check():
                    print("❌ No NATS available (not running and cannot start)")
                    return
                else:
                    print("ℹ️  External NATS detected, continuing with existing instance")

        # Initialiser LoopManager
        print("📦 Initializing LoopManager...")
        manager = LoopManager()

        try:
            await manager.start()
            print("✅ LoopManager started successfully\n")
        except Exception as e:
            print(f"❌ Failed to start LoopManager: {e}")
            return

        # Event templates avec poids
        templates = self.get_event_templates()
        event_types = [t[0] for t in templates]
        event_generators = [t[1] for t in templates]
        weights = [t[2] for t in templates]

        # Tracking
        start_time = time.time()
        last_monitor_time = start_time
        last_chaos_check = start_time
        last_stats_display = start_time

        # Durée en secondes
        duration_seconds = self.config.duration_hours * 3600

        # Signal handler pour arrêt propre
        stop_signal = False

        def signal_handler(signum, frame):
            nonlocal stop_signal
            print("\n🛑 Stopping gracefully...")
            stop_signal = True

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            print("📊 Starting load generation...\n")

            while not stop_signal and time.time() - start_time < duration_seconds:
                batch_start = time.time()

                # Check chaos périodiquement
                if self.config.chaos_enabled and time.time() - last_chaos_check > 60:
                    try:
                        loop_metrics = manager.get_metrics()
                        symbiosis = loop_metrics.get("system", {}).get("symbiosis_score", 0.5)
                        chaos_event = await self.chaos.maybe_cause_chaos(manager, symbiosis)
                        if chaos_event:
                            self.stats["chaos_events"] += 1
                    except:
                        pass
                    last_chaos_check = time.time()

                # Monitoring système périodique
                if self.config.monitor_resources and time.time() - last_monitor_time > self.config.monitor_interval:
                    await self.monitor_and_report(manager)
                    last_monitor_time = time.time()

                # Adaptation ML du rate
                if self.config.adaptive_ml and self.stats["messages_sent"] % 100 == 0:
                    try:
                        metrics = await self.get_combined_metrics(manager)
                        new_rate = self.controller.adapt_rate(metrics, phase)
                        if abs(new_rate - self.controller.current_rate) > 10:
                            self.stats["rate_adjustments"] += 1
                    except:
                        pass

                # Calcul du délai batch
                current_rate = self.controller.current_rate if self.config.adaptive_ml else self.config.target_rate
                batch_delay = self.config.batch_size / current_rate if current_rate > 0 else 1

                # Envoyer un batch de messages
                batch_tasks = []

                for _ in range(self.config.batch_size):
                    # Sélection pondérée du type d'event
                    idx = random.choices(range(len(event_types)), weights=weights)[0]
                    event_type = event_types[idx]
                    data_generator = event_generators[idx]

                    data = data_generator(self.stats["messages_sent"])

                    # Corruption potentielle
                    original_data = data
                    if self.config.corruption_enabled:
                        data, is_corrupted = self.chaos.corrupt_data(data)
                        if is_corrupted:
                            self.stats["corrupted"] += 1

                    # Estimation taille pour compression
                    try:
                        data_size = len(json.dumps(data).encode()) if data else 0
                        if data_size > self.config.compression_threshold:
                            self.stats["compressed"] += 1
                    except:
                        data_size = 0

                    # Publier avec namespace
                    try:
                        if manager.event_bus:
                            subject = get_subject_with_namespace(event_type)
                            task = manager.event_bus.publish(subject, data)
                            batch_tasks.append(task)
                            self.stats["messages_sent"] += 1
                            prom_messages_sent.inc()
                    except Exception as e:
                        self.stats["errors"] += 1
                        prom_messages_failed.inc()
                        if self.stats["errors"] <= 5:
                            print(f"⚠️  Publish error: {e}")

                # Attendre le batch
                if batch_tasks:
                    results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    for r in results:
                        if isinstance(r, Exception):
                            self.stats["errors"] += 1
                            prom_messages_failed.inc()

                # Affichage stats périodique
                if time.time() - last_stats_display >= 10:  # Toutes les 10s
                    await self.display_progress(manager, start_time, phase)
                    last_stats_display = time.time()

                # Sleep adaptatif
                batch_duration = time.time() - batch_start
                if batch_duration < batch_delay:
                    await asyncio.sleep(batch_delay - batch_duration)

        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Rapport final
            await self.final_report(manager, start_time, phase)

            # Arrêt propre
            print("\n🔄 Shutting down LoopManager...")
            await manager.stop()
            print("✅ Clean shutdown complete")

    async def get_combined_metrics(self, manager) -> dict[str, Any]:
        """Combine métriques du bus et des loops"""
        metrics = {}

        try:
            # Métriques du bus
            if hasattr(manager, "event_bus") and manager.event_bus:
                bus_metrics = manager.event_bus.get_metrics()
                metrics.update(bus_metrics)

            # Métriques des loops
            loop_metrics = manager.get_metrics()
            if "system" in loop_metrics:
                metrics["symbiosis_score"] = loop_metrics["system"].get("symbiosis_score", 0.5)
                metrics["active_loops"] = loop_metrics["system"].get("active_loops", 0)

            # Métriques DLQ - Get from bus metrics first, fallback to stats
            if "dlq_count" not in metrics or metrics["dlq_count"] == 0:
                metrics["dlq_count"] = self.stats.get("dlq_messages", 0)

            # Also get corrupted count from bus
            if "corrupted_count" not in metrics:
                metrics["corrupted_count"] = self.stats.get("corrupted_sent", 0)

        except Exception as e:
            print(f"⚠️  Error getting metrics: {e}")

        # Update Prometheus
        prom_p99_latency.set(metrics.get("p99_latency_ms", 0))
        prom_symbiosis.set(metrics.get("symbiosis_score", 0.5))

        return metrics

    async def monitor_and_report(self, manager):
        """Monitoring et alertes système"""
        metrics = self.monitor.get_metrics()
        limits = self.monitor.check_limits()

        # Alertes
        alerts = []

        if not limits["memory_ok"]:
            alert = f"⚠️  MEMORY WARNING: {metrics['memory_percent']:.1f}% used"
            alerts.append(alert)
            print(alert)

        if not limits["cpu_ok"]:
            alert = f"⚠️  CPU WARNING: {metrics['cpu_percent']:.1f}% used"
            alerts.append(alert)
            print(alert)

        if limits["leak_suspected"]:
            alert = (
                f"🔴 MEMORY LEAK SUSPECTED: +{metrics['memory_increase_mb']:.1f}MB, trend: {metrics['memory_trend']}"
            )
            alerts.append(alert)
            print(alert)

            # Réduire automatiquement le rate si leak
            if self.config.adaptive_ml:
                self.controller.current_rate *= 0.8
                print(f"   → Reducing rate to {self.controller.current_rate:.0f} msg/s")

        return alerts

    async def display_progress(self, manager, start_time: float, phase: str):
        """Affichage progression avec métriques clés"""
        elapsed = time.time() - start_time
        current_rate = self.stats["messages_sent"] / elapsed if elapsed > 0 else 0
        error_rate = (self.stats["errors"] / max(1, self.stats["messages_sent"])) * 100

        # Métriques combinées
        metrics = await self.get_combined_metrics(manager)
        p99 = metrics.get("p99_latency_ms", 0)
        dropped = metrics.get("dropped", 0)
        pending = metrics.get("pending_messages", 0)
        symbiosis = metrics.get("symbiosis_score", 0.5)

        # Métriques système
        sys_metrics = self.monitor.get_metrics()

        # Validation selon phase
        thresholds = self.config.thresholds.get(phase, self.config.thresholds["soak"])
        p99_ok = "✅" if p99 <= thresholds["p99"] else "❌"

        # Format compact mais informatif
        print(
            f"[{datetime.now().strftime('%H:%M:%S')}] Phase: {phase.upper()} | "
            f"Msgs: {self.stats['messages_sent']:,} | "
            f"Rate: {current_rate:.0f}/s | "
            f"Err: {error_rate:.2f}% | "
            f"P99: {p99:.1f}ms {p99_ok} | "
            f"Symb: {symbiosis:.2f} | "
            f"Mem: {sys_metrics['memory_mb']:.0f}MB | "
            f"CPU: {sys_metrics['cpu_percent']:.1f}%"
        )

        # Ligne supplémentaire si événements spéciaux
        special = []
        if self.config.chaos_enabled and self.stats["chaos_events"] > 0:
            special.append(f"Chaos: {self.stats['chaos_events']}")
        if self.config.corruption_enabled and self.stats["corrupted"] > 0:
            special.append(f"Corrupted: {self.stats['corrupted']}")
        if self.config.adaptive_ml and self.stats["rate_adjustments"] > 0:
            special.append(f"Adjusted: {self.stats['rate_adjustments']}x")
        if metrics.get("dlq_count", 0) > 0:
            special.append(f"DLQ: {metrics['dlq_count']}")

        if special:
            print(f"     → {' | '.join(special)}")

    async def final_report(self, manager, start_time: float, phase: str):
        """Rapport final détaillé avec validation"""
        elapsed = time.time() - start_time

        print("\n" + "=" * 70)
        print(f"📊 LOAD GENERATION COMPLETE - PHASE: {phase.upper()}")
        print("=" * 70)

        # Statistiques générales
        print("\n📈 GENERAL STATISTICS:")
        print(f"  Duration: {elapsed / 60:.1f} minutes ({elapsed / 3600:.2f} hours)")
        print(f"  Messages sent: {self.stats['messages_sent']:,}")
        print(f"  Average rate: {self.stats['messages_sent'] / elapsed:.0f} msg/s")
        print(f"  Target rate: {self.config.target_rate} msg/s")
        print(
            f"  Errors: {self.stats['errors']:,} ({self.stats['errors'] / max(1, self.stats['messages_sent']) * 100:.3f}%)"
        )

        if self.config.corruption_enabled:
            print(f"  Corrupted sent: {self.stats['corrupted']:,}")
        if self.stats["compressed"] > 0:
            print(
                f"  Compressed: ~{self.stats['compressed']:,} ({self.stats['compressed'] / max(1, self.stats['messages_sent']) * 100:.0f}%)"
            )

        # Métriques système
        sys_metrics = self.monitor.get_metrics()
        print("\n💻 SYSTEM METRICS:")
        print(f"  Final Memory: {sys_metrics['memory_mb']:.1f}MB")
        print(f"  Memory Increase: {sys_metrics['memory_increase_mb']:.1f}MB")
        print(f"  Memory Trend: {sys_metrics['memory_trend']}")
        print(f"  Average CPU: {sys_metrics['cpu_avg']:.1f}%")
        print(f"  Peak CPU: {max(self.monitor.cpu_samples) if self.monitor.cpu_samples else 0:.1f}%")
        print(f"  Open Files: {sys_metrics['open_files']}")
        print(f"  Threads: {sys_metrics['num_threads']}")

        # Métriques du bus et loops
        try:
            metrics = await self.get_combined_metrics(manager)

            print("\n🧠 NEURALBUS METRICS:")
            print(f"  Published: {metrics.get('published', 0):,}")
            print(f"  Consumed: {metrics.get('consumed', 0):,}")
            print(f"  Dropped: {metrics.get('dropped', 0)}")
            print(f"  DLQ Count: {metrics.get('dlq_count', 0)}")
            print(f"  P50 Latency: {metrics.get('p50_latency_ms', 0):.1f}ms")
            print(f"  P95 Latency: {metrics.get('p95_latency_ms', 0):.1f}ms")
            print(f"  P99 Latency: {metrics.get('p99_latency_ms', 0):.1f}ms")

            print("\n🔮 ML METRICS:")
            print(f"  Symbiosis Score: {metrics.get('symbiosis_score', 0.5):.3f}")
            print(f"  Active Loops: {metrics.get('active_loops', 0)}")
            print(f"  Total Cycles: {metrics.get('total_cycles', 0)}")

        except Exception as e:
            print(f"\n⚠️  Could not get final metrics: {e}")
            metrics = {}

        # Chaos report
        if self.config.chaos_enabled and self.chaos.chaos_events:
            print("\n🔥 CHAOS TESTING REPORT:")
            print(f"  Total Events: {len(self.chaos.chaos_events)}")

            recovered = sum(1 for e in self.chaos.chaos_events if e.get("recovered", False))
            print(
                f"  Recovered: {recovered}/{len(self.chaos.chaos_events)} ({recovered / len(self.chaos.chaos_events) * 100:.0f}%)"
            )

            # Par type
            by_type = {}
            for event in self.chaos.chaos_events:
                event_type = event["type"]
                by_type[event_type] = by_type.get(event_type, 0) + 1

            print("  Events by type:")
            for chaos_type, count in sorted(by_type.items()):
                print(f"    - {chaos_type}: {count}x")

        # ML Adaptive report
        if self.config.adaptive_ml:
            print("\n🤖 ADAPTIVE CONTROL REPORT:")
            print(f"  Rate Adjustments: {self.stats['rate_adjustments']}")
            print(f"  Final Rate: {self.controller.current_rate:.0f} msg/s")
            print(f"  Target Rate: {self.config.target_rate} msg/s")

            if self.controller.performance_history:
                recent = list(self.controller.performance_history)[-min(10, len(self.controller.performance_history)) :]
                if recent:
                    avg_p99 = statistics.mean([p["p99"] for p in recent if "p99" in p])
                    avg_symbiosis = statistics.mean([p["symbiosis"] for p in recent if "symbiosis" in p])
                    print(f"  Recent Avg P99: {avg_p99:.1f}ms")
                    print(f"  Recent Avg Symbiosis: {avg_symbiosis:.3f}")

        # VALIDATION FINALE
        print("\n" + "=" * 70)
        print(f"🏁 VALIDATION RESULTS - PHASE: {phase.upper()}")

        # Critères selon la phase
        thresholds = self.config.thresholds.get(phase, self.config.thresholds["soak"])

        success_criteria = {
            f'P99 < {thresholds["p99"]}ms': metrics.get("p99_latency_ms", 999) < thresholds["p99"],
            f'Drop Rate < {thresholds["drops"]}%': (
                metrics.get("dropped", 0) / max(1, self.stats["messages_sent"]) * 100
            )
            < thresholds["drops"],
            "Error Rate < 1%": (self.stats["errors"] / max(1, self.stats["messages_sent"]) * 100) < 1,
            "Symbiosis > 0.3": metrics.get("symbiosis_score", 0) > 0.3,
            "No Memory Leak": sys_metrics["memory_trend"] != "increasing" or sys_metrics["memory_increase_mb"] < 50,
        }

        # Si corruption activée, vérifier DLQ
        if self.config.corruption_enabled:
            # Get DLQ count from bus metrics
            bus_dlq_count = 0
            if hasattr(manager, "event_bus") and manager.event_bus:
                bus_metrics = manager.event_bus.get_metrics()
                bus_dlq_count = bus_metrics.get("dlq_count", 0)
                # Also update metrics for display
                metrics["dlq_count"] = bus_dlq_count
                metrics["corrupted_count"] = bus_metrics.get("corrupted_count", 0)

            success_criteria["DLQ handles corruption"] = bus_dlq_count >= self.stats["corrupted"] * 0.8

        # Si chaos activé, vérifier recovery
        if self.config.chaos_enabled and self.chaos.chaos_events:
            recovery_rate = sum(1 for e in self.chaos.chaos_events if e.get("recovered", False)) / len(
                self.chaos.chaos_events
            )
            success_criteria["Chaos Recovery > 80%"] = recovery_rate > 0.8

        all_pass = True
        for criterion, passed in success_criteria.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {criterion}")
            if not passed:
                all_pass = False

        print("\n" + "=" * 70)
        if all_pass:
            print(f"🎉 PHASE {phase.upper()} PASSED - SYSTEM VALIDATED!")
        else:
            print(f"⚠️  PHASE {phase.upper()} - SOME CRITERIA NOT MET")
        print("=" * 70)

        return all_pass


async def main():
    """Point d'entrée principal avec CLI améliorée"""
    parser = argparse.ArgumentParser(
        description="Jeffrey OS Production Load Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phases disponibles et leurs paramètres par défaut :
  quick  : 3 minutes, 100 msg/s, P99<100ms
  stress : 6 minutes, 500 msg/s, P99<70ms
  chaos  : 6 minutes, 500 msg/s, P99<100ms, chaos activé
  soak   : 2 heures, 1000 msg/s, P99<50ms

Examples:
  # Test rapide de validation
  %(prog)s --phase quick

  # Test de stress avec chaos
  %(prog)s --phase stress --chaos --corruption

  # Soak test production 2h
  %(prog)s --phase soak --ml --monitor

  # Test personnalisé
  %(prog)s --hours 1 --rate 800 --chaos --ml --corruption

  # Mode CI/CD non-interactif
  %(prog)s --phase quick --non-interactive
        """,
    )

    # Arguments
    parser.add_argument(
        "--phase",
        choices=["quick", "stress", "chaos", "soak", "custom"],
        default="custom",
        help="Test phase with predefined settings",
    )

    parser.add_argument("--hours", type=float, help="Test duration in hours (overrides phase default)")
    parser.add_argument("--rate", type=int, help="Target messages per second (overrides phase default)")

    parser.add_argument("--chaos", action="store_true", help="Enable chaos testing")
    parser.add_argument("--corruption", action="store_true", help="Enable data corruption testing")
    parser.add_argument("--malicious", action="store_true", help="Enable malicious event testing (5% rate)")
    parser.add_argument("--ml", action="store_true", help="Enable ML adaptive control")
    parser.add_argument("--monitor", action="store_true", help="Enable resource monitoring")

    parser.add_argument(
        "--prometheus-port",
        type=int,
        default=8000,
        help="Prometheus metrics port (default: 8000, 0 to disable)",
    )
    parser.add_argument("--namespace", help="NATS subject namespace (default: auto-generated)")

    parser.add_argument("--non-interactive", action="store_true", help="Non-interactive mode for CI/CD")

    args = parser.parse_args()

    # Configuration selon la phase
    phase_configs = {
        "quick": {
            "duration_hours": 0.05,  # 3 minutes
            "target_rate": 100,
            "chaos_enabled": False,
            "corruption_enabled": False,
            "adaptive_ml": False,
        },
        "stress": {
            "duration_hours": 0.1,  # 6 minutes
            "target_rate": 500,
            "chaos_enabled": False,
            "corruption_enabled": False,
            "adaptive_ml": True,
        },
        "chaos": {
            "duration_hours": 0.1,  # 6 minutes
            "target_rate": 500,
            "chaos_enabled": True,
            "corruption_enabled": True,
            "adaptive_ml": True,
        },
        "soak": {
            "duration_hours": 2.0,  # 2 heures
            "target_rate": 1000,
            "chaos_enabled": False,
            "corruption_enabled": False,
            "adaptive_ml": True,
            "monitor_resources": True,
        },
    }

    # Appliquer la config de phase
    if args.phase in phase_configs:
        base_config = phase_configs[args.phase]
    else:
        base_config = {}

    # Override avec les arguments CLI
    if args.hours is not None:
        base_config["duration_hours"] = args.hours
    if args.rate is not None:
        base_config["target_rate"] = args.rate
    if args.chaos:
        base_config["chaos_enabled"] = True
        base_config["chaos_adaptive"] = True
    if args.corruption:
        base_config["corruption_enabled"] = True
    if args.malicious:
        base_config["malicious_enabled"] = True
    if args.ml:
        base_config["adaptive_ml"] = True
    if args.monitor:
        base_config["monitor_resources"] = True

    base_config["prometheus_port"] = args.prometheus_port
    base_config["non_interactive"] = args.non_interactive

    # Set namespace
    if args.namespace:
        os.environ["NB_NS"] = args.namespace
    else:
        os.environ["NB_NS"] = f"test_{args.phase}_{int(time.time())}"

    # Créer la configuration
    config = LoadConfig(**base_config)

    # Lancer le générateur
    generator = LoadGenerator(config)
    test_passed = await generator.run(phase=args.phase if args.phase != "custom" else "soak")

    # Exit with proper code
    import sys

    if test_passed:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Failure


if __name__ == "__main__":
    asyncio.run(main())
