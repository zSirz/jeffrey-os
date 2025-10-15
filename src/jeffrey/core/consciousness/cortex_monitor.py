#!/usr/bin/env python3
"""
Cortex Monitor - Système de surveillance temps réel pour l'architecture cognitive de Jeffrey OS

Ce module implémente un système de monitoring complet et professionnel pour surveiller
en temps réel l'état de santé, les performances, et l'évolution du cortex neuronal.
Il fournit des tableaux de bord dynamiques, des métriques détaillées, des alertes
automatiques, et des recommandations d'optimisation intelligentes.

Le moniteur suit les composants critiques incluant l'utilisation système, les performances
de mémoire épisodique/sémantique, la santé des connexions Redis, les métriques de fairness,
les temps de réponse, et l'évolution de la conscience. Il génère des rapports exportables
et maintient un historique des métriques pour l'analyse de tendances.

Fonctionnalités principales:
- Surveillance temps réel multi-composants
- Alertes automatiques basées sur seuils configurables
- Tableaux de bord interactifs avec actualisation continue
- Export de métriques au format JSON pour analyse externe
- Recommandations d'optimisation contextuelles
- Monitoring de l'évolution cognitive et des insights de rêve

Utilisation:
    monitor = CortexMonitor(cortex_instance)
    monitor.display_live_dashboard()  # Interface temps réel
    status = monitor.get_comprehensive_status()  # Rapport complet
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from collections import Counter, deque
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil non disponible - métriques système limitées")

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("⚠️ Redis non disponible - pas de monitoring Redis")

# Import du système
sys.path.append(str(Path(__file__).parent.parent))
from memory_bridge import get_memory_bridge


class CortexMonitor:
    """
    Système de monitoring professionnel pour surveillance cognitive temps réel.

    Orchestre la surveillance complète du cortex neuronal avec collecte de métriques
    multi-dimensionnelles, détection d'anomalies, génération d'alertes intelligentes,
    et visualisation dynamique des performances. Maintient l'historique des données
    pour analyse de tendances et optimisation prédictive.
    """

    def __init__(self, cortex: Any | None = None) -> None:
        """
        Initialise le système de monitoring avec configuration des seuils et métriques.

        Args:
            cortex: Instance du cortex à surveiller (récupérée automatiquement si None)
        """
        self.cortex = cortex or get_memory_bridge().cortex
        self.bridge = get_memory_bridge()

        # Métriques temps réel
        self.start_time = datetime.now()
        self.metrics_history = deque(maxlen=100)  # 100 dernières mesures
        self.performance_alerts = []
        self.health_alerts = []

        # Configuration monitoring
        self.refresh_interval = 5  # secondes
        self.alert_thresholds = {
            'memory_usage_mb': 1000,
            'cpu_percent': 80,
            'response_time_ms': 2000,
            'error_rate': 0.1,
            'fairness_score_min': 0.7,
        }

        # État du monitoring
        self.monitoring_active = False
        self.monitor_thread = None

        print("📊 CortexMonitor initialisé")

    def get_comprehensive_status(self) -> dict[str, Any]:
        """
        Génère un rapport d'état complet et structuré du système cognitif.

        Collecte et agrège toutes les métriques disponibles incluant système,
        cortex, mémoire, performance, santé, évolution, alertes actives,
        et recommandations d'optimisation contextuelles.

        Returns:
            Dict[str, Any]: Rapport complet avec timestamp, uptime, métriques
                           multi-dimensionnelles, alertes, et recommandations
        """

        status = {
            "timestamp": datetime.now().isoformat(),
            "uptime": str(datetime.now() - self.start_time),
            "system": self._get_system_metrics(),
            "cortex": self._get_cortex_metrics(),
            "memory": self._get_memory_metrics(),
            "performance": self._get_performance_metrics(),
            "health": self._get_health_metrics(),
            "evolution": self._get_evolution_metrics(),
            "alerts": self._get_current_alerts(),
            "recommendations": self._generate_recommendations(),
        }

        return status

    def _get_system_metrics(self) -> dict[str, Any]:
        """
        Collecte les métriques système fondamentales via psutil.

        Récupère l'utilisation CPU, consommation mémoire processus et système,
        nombre de threads, fichiers ouverts, espace disque disponible.

        Returns:
            Dict[str, Any]: Métriques système avec platform, versions, utilisation
                           ressources, ou messages d'erreur si psutil indisponible
        """

        metrics = {"platform": sys.platform, "python_version": sys.version.split()[0]}

        if PSUTIL_AVAILABLE:
            try:
                process = psutil.Process(os.getpid())

                metrics.update(
                    {
                        "cpu_percent": psutil.cpu_percent(interval=0.1),
                        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
                        "memory_percent": process.memory_percent(),
                        "thread_count": process.num_threads(),
                        "open_files": len(process.open_files()),
                        "system_memory_total_gb": psutil.virtual_memory().total / 1024**3,
                        "system_memory_available_gb": psutil.virtual_memory().available / 1024**3,
                        "disk_usage_percent": psutil.disk_usage('/').percent
                        if os.name != 'nt'
                        else psutil.disk_usage('C:\\').percent,
                    }
                )
            except Exception as e:
                metrics["error"] = f"Erreur psutil: {e}"
        else:
            metrics["warning"] = "psutil non disponible"

        return metrics

    def _get_cortex_metrics(self) -> dict[str, Any]:
        """
        Extraction des métriques spécifiques à l'architecture cognitive du cortex.

        Analyse l'état du bridge mémoire, niveaux de conscience, compteurs de mémoire
        épisodique/sémantique/relationnelle, performances de recherche/stockage,
        et scores de fairness moyens.

        Returns:
            Dict[str, Any]: Métriques cognitives incluant santé bridge, interactions,
                           erreurs, contenus mémoriels, et performances opérationnelles
        """

        metrics = {
            "bridge_healthy": self.bridge.is_healthy,
            "fallback_mode": self.bridge.fallback_mode,
            "total_interactions": self.bridge.total_interactions,
            "error_count": self.bridge.error_count,
            "error_rate": self.bridge.error_count / max(1, self.bridge.total_interactions),
        }

        # Métriques du cortex lui-même
        if hasattr(self.cortex, 'cortex'):
            cortex_core = self.cortex.cortex

            if hasattr(cortex_core, 'consciousness_level'):
                metrics["consciousness_level"] = cortex_core.consciousness_level

            if hasattr(cortex_core, 'episodic_memory'):
                metrics["episodic_memory_count"] = len(cortex_core.episodic_memory)

            if hasattr(cortex_core, 'semantic_memory'):
                metrics["semantic_memory_count"] = len(cortex_core.semantic_memory)

            if hasattr(cortex_core, 'relational_memory'):
                metrics["relational_memory_count"] = len(cortex_core.relational_memory)

            if hasattr(cortex_core, 'performance_metrics'):
                perf = cortex_core.performance_metrics
                if perf.get('search_times'):
                    metrics["avg_search_time_ms"] = sum(perf['search_times']) / len(perf['search_times']) * 1000
                if perf.get('storage_times'):
                    metrics["avg_storage_time_ms"] = sum(perf['storage_times']) / len(perf['storage_times']) * 1000
                if perf.get('fairness_scores'):
                    metrics["avg_fairness_score"] = sum(perf['fairness_scores']) / len(perf['fairness_scores'])

        return metrics

    def _get_memory_metrics(self) -> dict[str, Any]:
        """
        Analyse approfondie des systèmes de mémoire épisodique et distributions.

        Examine la mémoire épisodique pour extraire statistiques temporelles,
        distributions émotionnelles des souvenirs récents, niveaux de sensibilité,
        et scores de fairness des interactions passées.

        Returns:
            Dict[str, Any]: Métriques mémorielles avec compteurs temporels,
                           distributions émotionnelles/sensibilité, fairness récente
        """

        metrics = {}

        if hasattr(self.cortex, 'cortex') and hasattr(self.cortex.cortex, 'episodic_memory'):
            episodic = self.cortex.cortex.episodic_memory

            # Statistiques temporelles
            if episodic:
                now = datetime.now()
                recent_count = sum(1 for m in episodic if (now - getattr(m, 'timestamp', now)).total_seconds() < 3600)

                metrics.update(
                    {
                        "recent_memories_1h": recent_count,
                        "oldest_memory": min(getattr(m, 'timestamp', now) for m in episodic).isoformat()
                        if episodic
                        else None,
                        "newest_memory": max(getattr(m, 'timestamp', now) for m in episodic).isoformat()
                        if episodic
                        else None,
                    }
                )

                # Distribution émotionnelle
                emotions = [getattr(m, 'emotion', 'unknown') for m in episodic[-20:]]  # 20 derniers
                emotion_dist = dict(Counter(emotions))
                metrics["emotion_distribution"] = emotion_dist

                # Niveaux de sensibilité
                sensitivity_levels = [getattr(m, 'sensitivity_level', 'normal') for m in episodic[-20:]]
                sensitivity_dist = dict(Counter(sensitivity_levels))
                metrics["sensitivity_distribution"] = sensitivity_dist

                # Scores de fairness récents
                fairness_scores = [getattr(m, 'fairness_score', 1.0) for m in episodic[-10:]]
                if fairness_scores:
                    metrics["recent_fairness_avg"] = sum(fairness_scores) / len(fairness_scores)
                    metrics["recent_fairness_min"] = min(fairness_scores)

        return metrics

    def _get_performance_metrics(self) -> dict[str, Any]:
        """
        Collecte et analyse des métriques de performance opérationnelles détaillées.

        Calcule statistiques de temps de réponse pour recherche/stockage,
        métriques de fairness avec détection de violations, throughput
        opérationnel, et ratios performance/temps.

        Returns:
            Dict[str, Any]: Métriques performance avec temps moyens/min/max,
                           violations fairness, compteurs opérationnels, throughput
        """

        metrics = {}

        if hasattr(self.cortex, 'cortex') and hasattr(self.cortex.cortex, 'performance_metrics'):
            perf = self.cortex.cortex.performance_metrics

            # Temps de réponse
            if perf.get('search_times'):
                search_times = list(perf['search_times'])
                metrics.update(
                    {
                        "search_time_avg_ms": sum(search_times) / len(search_times) * 1000,
                        "search_time_max_ms": max(search_times) * 1000,
                        "search_time_min_ms": min(search_times) * 1000,
                        "search_operations_count": len(search_times),
                    }
                )

            if perf.get('storage_times'):
                storage_times = list(perf['storage_times'])
                metrics.update(
                    {
                        "storage_time_avg_ms": sum(storage_times) / len(storage_times) * 1000,
                        "storage_time_max_ms": max(storage_times) * 1000,
                        "storage_operations_count": len(storage_times),
                    }
                )

            # Métriques de fairness
            if perf.get('fairness_scores'):
                fairness_scores = list(perf['fairness_scores'])
                metrics.update(
                    {
                        "fairness_score_avg": sum(fairness_scores) / len(fairness_scores),
                        "fairness_score_min": min(fairness_scores),
                        "fairness_checks_count": len(fairness_scores),
                        "fairness_violations": sum(1 for score in fairness_scores if score < 0.8),
                    }
                )

            # Throughput
            metrics["total_memory_operations"] = perf.get('total_memories', 0)

            uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
            if uptime_hours > 0:
                metrics["operations_per_hour"] = perf.get('total_memories', 0) / uptime_hours

        return metrics

    def _get_health_metrics(self) -> dict[str, Any]:
        """
        Évaluation complète de la santé des composants système critiques.

        Vérifie l'état du bridge mémoire, cortex, connexions Redis,
        système de fichiers, et détermine l'état de santé global
        basé sur l'agrégation des composants individuels.

        Returns:
            Dict[str, Any]: État santé global et détails par composant
                           (healthy/degraded/unhealthy/error)
        """

        health = {"overall_status": "unknown", "components": {}}

        # Santé du bridge
        bridge_health = self.bridge.health_check()
        health["components"]["memory_bridge"] = "healthy" if bridge_health else "unhealthy"

        # Santé du cortex
        if hasattr(self.cortex, 'cortex') and hasattr(self.cortex.cortex, 'health_check'):
            try:
                cortex_health = self.cortex.cortex.health_check()
                if isinstance(cortex_health, dict):
                    health["components"]["cortex"] = cortex_health.get('overall_health', 'unknown')
                    health["cortex_details"] = cortex_health.get('components', {})
                else:
                    health["components"]["cortex"] = "healthy" if cortex_health else "unhealthy"
            except Exception as e:
                health["components"]["cortex"] = f"error: {e}"
        else:
            health["components"]["cortex"] = "no_health_check"

        # Santé Redis
        health["components"]["redis"] = self._check_redis_health()

        # Santé du système de fichiers
        health["components"]["filesystem"] = self._check_filesystem_health()

        # Déterminer l'état global
        component_states = list(health["components"].values())
        if any("error" in state or state == "unhealthy" for state in component_states):
            health["overall_status"] = "unhealthy"
        elif any(state in ["degraded", "warning"] for state in component_states):
            health["overall_status"] = "degraded"
        elif all(state == "healthy" for state in component_states):
            health["overall_status"] = "healthy"
        else:
            health["overall_status"] = "partial"

        return health

    def _get_evolution_metrics(self) -> dict[str, Any]:
        """
        Analyse des métriques d'évolution cognitive et d'apprentissage adaptatif.

        Trace l'évolution du niveau de conscience, entités reconnues,
        relations établies, insights générés par le système de rêve,
        et taux de croissance cognitive.

        Returns:
            Dict[str, Any]: Métriques évolutives avec conscience actuelle/delta,
                           entités/relations, cycles de rêve, insights second-ordre
        """

        evolution = {}

        if hasattr(self.cortex, 'cortex'):
            cortex_core = self.cortex.cortex

            # Évolution de conscience
            if hasattr(cortex_core, 'consciousness_level') and hasattr(cortex_core, 'consciousness_trajectory'):
                evolution["current_consciousness"] = cortex_core.consciousness_level

                trajectory = cortex_core.consciousness_trajectory
                if len(trajectory) >= 2:
                    first_level = trajectory[0].get('level', 0)
                    last_level = trajectory[-1].get('level', 0)
                    evolution["consciousness_delta"] = last_level - first_level
                    evolution["consciousness_growth_rate"] = evolution["consciousness_delta"] / len(trajectory)

            # Entités reconnues
            if hasattr(cortex_core, 'relational_memory'):
                evolution["recognized_entities"] = list(cortex_core.relational_memory.keys())
                evolution["relationship_count"] = len(cortex_core.relational_memory)

            # Insights de rêve
            if hasattr(self.cortex, 'dream_engine'):
                dream_engine = self.cortex.dream_engine
                if hasattr(dream_engine, 'insights_second_order'):
                    evolution["total_insights"] = len(dream_engine.insights_second_order)
                if hasattr(dream_engine, 'dream_cycles'):
                    evolution["dream_cycles"] = dream_engine.dream_cycles

        return evolution

    def _check_redis_health(self) -> str:
        """
        Diagnostic de connectivité et réactivité du serveur Redis.

        Teste la connexion, latence ping, et détecte les timeouts
        pour évaluer la santé de l'infrastructure de communication.

        Returns:
            str: État Redis ('healthy', 'disconnected', 'timeout', 'unavailable', 'error')
        """
        if not REDIS_AVAILABLE:
            return "unavailable"

        try:
            r = redis.Redis(host='localhost', port=6380, decode_responses=True, socket_timeout=1)
            r.ping()
            return "healthy"
        except redis.ConnectionError:
            return "disconnected"
        except redis.TimeoutError:
            return "timeout"
        except Exception as e:
            return f"error: {str(e)[:50]}"

    def _check_filesystem_health(self) -> str:
        """
        Diagnostic de l'intégrité et disponibilité du système de fichiers.

        Vérifie l'espace disque disponible, permissions d'écriture,
        et accès aux répertoires critiques pour la persistance.

        Returns:
            str: État filesystem ('healthy', 'low_space', 'critical_space',
                 'permission_error', 'error')
        """
        try:
            # Vérifier l'espace disque
            if PSUTIL_AVAILABLE:
                disk_usage = psutil.disk_usage('/' if os.name != 'nt' else 'C:\\')
                if disk_usage.percent > 95:
                    return "critical_space"
                elif disk_usage.percent > 85:
                    return "low_space"

            # Vérifier l'accès en écriture
            test_file = Path("core/memory/.health_check")
            test_file.touch()
            test_file.unlink()

            return "healthy"

        except PermissionError:
            return "permission_error"
        except OSError as e:
            return f"error: {str(e)[:50]}"

    def _get_current_alerts(self) -> list[dict[str, Any]]:
        """
        Génération d'alertes basées sur l'évaluation des seuils configurés.

        Compare les métriques courantes aux seuils d'alerte pour détecter
        usage mémoire/CPU excessif, taux d'erreur élevé, performances
        dégradées, et violations de fairness.

        Returns:
            List[Dict[str, Any]]: Liste d'alertes avec type, sévérité,
                                 message descriptif, et timestamp
        """

        alerts = []

        # Récupérer les métriques système
        system_metrics = self._get_system_metrics()
        cortex_metrics = self._get_cortex_metrics()
        performance_metrics = self._get_performance_metrics()

        # Vérifier les seuils d'alerte
        if system_metrics.get('memory_usage_mb', 0) > self.alert_thresholds['memory_usage_mb']:
            alerts.append(
                {
                    "type": "memory_usage",
                    "severity": "warning",
                    "message": f"Utilisation mémoire élevée: {system_metrics['memory_usage_mb']:.1f}MB",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        if system_metrics.get('cpu_percent', 0) > self.alert_thresholds['cpu_percent']:
            alerts.append(
                {
                    "type": "cpu_usage",
                    "severity": "warning",
                    "message": f"Utilisation CPU élevée: {system_metrics['cpu_percent']:.1f}%",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        if cortex_metrics.get('error_rate', 0) > self.alert_thresholds['error_rate']:
            alerts.append(
                {
                    "type": "error_rate",
                    "severity": "critical",
                    "message": f"Taux d'erreur élevé: {cortex_metrics['error_rate']:.2%}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        if performance_metrics.get('search_time_avg_ms', 0) > self.alert_thresholds['response_time_ms']:
            alerts.append(
                {
                    "type": "performance",
                    "severity": "warning",
                    "message": f"Temps de recherche élevé: {performance_metrics['search_time_avg_ms']:.1f}ms",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        if performance_metrics.get('fairness_score_avg', 1.0) < self.alert_thresholds['fairness_score_min']:
            alerts.append(
                {
                    "type": "fairness",
                    "severity": "critical",
                    "message": f"Score de fairness bas: {performance_metrics['fairness_score_avg']:.3f}",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return alerts

    def _generate_recommendations(self) -> list[str]:
        """
        Analyse intelligente pour générer recommandations d'optimisation contextuelles.

        Examine les métriques de performance, santé, et utilisation
        pour suggérer actions correctives spécifiques et améliorations
        d'efficacité du système cognitif.

        Returns:
            List[str]: Recommandations prioritaires pour optimisation
                      mémoire, performance, sécurité, infrastructure
        """

        recommendations = []

        # Analyser les métriques
        system_metrics = self._get_system_metrics()
        cortex_metrics = self._get_cortex_metrics()
        performance_metrics = self._get_performance_metrics()
        health_metrics = self._get_health_metrics()

        # Recommandations mémoire
        if system_metrics.get('memory_usage_mb', 0) > 500:
            recommendations.append("Considérer l'optimisation de l'utilisation mémoire")

        if cortex_metrics.get('episodic_memory_count', 0) > 50000:
            recommendations.append("Envisager la compression des anciens souvenirs")

        # Recommandations performance
        if performance_metrics.get('search_time_avg_ms', 0) > 1000:
            recommendations.append("Optimiser l'index FAISS ou activer la mise en cache")

        # Recommandations sécurité
        if performance_metrics.get('fairness_violations', 0) > 0:
            recommendations.append("Réviser les réponses pour améliorer la fairness")

        # Recommandations infrastructure
        if health_metrics["components"].get("redis") != "healthy":
            recommendations.append("Vérifier la configuration Redis")

        if cortex_metrics.get('error_rate', 0) > 0.05:
            recommendations.append("Analyser les logs d'erreur pour identifier les causes")

        return recommendations

    def display_live_dashboard(self) -> None:
        """
        Interface de monitoring temps réel avec actualisation continue.

        Affiche tableau de bord dynamique avec métriques système,
        état cortex, performance mémoire, évolution cognitive,
        santé composants, alertes actives, et recommandations.
        Actualisation automatique configurable.

        Raises:
            KeyboardInterrupt: Arrêt gracieux via Ctrl+C
        """

        try:
            while True:
                # Nettoyer l'écran
                os.system('clear' if os.name == 'posix' else 'cls')

                # Récupérer le statut complet
                status = self.get_comprehensive_status()

                # En-tête
                print("🧠 JEFFREY CORTEX MONITOR - TABLEAU DE BORD TEMPS RÉEL")
                print("=" * 80)
                print(f"⏰ {status['timestamp']}")
                print(f"⏱️  Uptime: {status['uptime']}")

                # État global
                health_icon = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌", "partial": "🔶"}.get(
                    status['health']['overall_status'], "❓"
                )
                print(f"{health_icon} État global: {status['health']['overall_status'].upper()}")

                # Métriques système
                print("\n💻 SYSTÈME")
                sys_metrics = status['system']
                print(f"   CPU: {sys_metrics.get('cpu_percent', 'N/A')}%")
                print(
                    f"   RAM: {sys_metrics.get('memory_usage_mb', 'N/A'):.1f} MB ({sys_metrics.get('memory_percent', 'N/A'):.1f}%)"
                )
                if 'disk_usage_percent' in sys_metrics:
                    print(f"   Disque: {sys_metrics['disk_usage_percent']:.1f}%")

                # Métriques cortex
                print("\n🧠 CORTEX")
                cortex_metrics = status['cortex']
                print(f"   Conscience: {cortex_metrics.get('consciousness_level', 'N/A'):.4f}")
                print(f"   Interactions: {cortex_metrics.get('total_interactions', 0)}")
                print(f"   Erreurs: {cortex_metrics.get('error_count', 0)} ({cortex_metrics.get('error_rate', 0):.2%})")
                print(f"   Mode: {'🔶 Dégradé' if cortex_metrics.get('fallback_mode') else '✅ Normal'}")

                # Mémoire
                print("\n📚 MÉMOIRE")
                memory_metrics = status['memory']
                print(f"   Épisodique: {cortex_metrics.get('episodic_memory_count', 0)} souvenirs")
                print(f"   Sémantique: {cortex_metrics.get('semantic_memory_count', 0)} concepts")
                print(f"   Relationnelle: {cortex_metrics.get('relational_memory_count', 0)} entités")
                print(f"   Récents (1h): {memory_metrics.get('recent_memories_1h', 0)}")

                # Performance
                print("\n⚡ PERFORMANCE")
                perf_metrics = status['performance']
                if 'search_time_avg_ms' in perf_metrics:
                    print(f"   Recherche: {perf_metrics['search_time_avg_ms']:.1f}ms moyen")
                if 'storage_time_avg_ms' in perf_metrics:
                    print(f"   Stockage: {perf_metrics['storage_time_avg_ms']:.1f}ms moyen")
                if 'fairness_score_avg' in perf_metrics:
                    print(f"   Fairness: {perf_metrics['fairness_score_avg']:.3f}")

                # Évolution
                print("\n📈 ÉVOLUTION")
                evolution_metrics = status['evolution']
                if 'consciousness_delta' in evolution_metrics:
                    delta = evolution_metrics['consciousness_delta']
                    print(f"   Croissance conscience: {delta:+.4f}")
                if 'recognized_entities' in evolution_metrics:
                    entities = evolution_metrics['recognized_entities']
                    print(f"   Entités reconnues: {', '.join(entities) if entities else 'Aucune'}")
                if 'total_insights' in evolution_metrics:
                    print(f"   Insights générés: {evolution_metrics['total_insights']}")

                # Santé des composants
                print("\n🏥 SANTÉ COMPOSANTS")
                components = status['health']['components']
                for component, state in components.items():
                    icon = {"healthy": "✅", "degraded": "⚠️", "unhealthy": "❌", "unavailable": "➖"}.get(
                        state.split(':')[0], "❓"
                    )
                    print(f"   {icon} {component}: {state}")

                # Alertes
                alerts = status['alerts']
                if alerts:
                    print("\n🚨 ALERTES")
                    for alert in alerts[:5]:  # Afficher max 5 alertes
                        severity_icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(alert['severity'], "⚪")
                        print(f"   {severity_icon} {alert['message']}")

                # Recommandations
                recommendations = status['recommendations']
                if recommendations:
                    print("\n💡 RECOMMANDATIONS")
                    for i, rec in enumerate(recommendations[:3], 1):  # Top 3
                        print(f"   {i}. {rec}")

                print("\n" + "=" * 80)
                print("Ctrl+C pour quitter | Actualisation automatique toutes les 5s")

                # Attendre avant la prochaine actualisation
                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring arrêté")

    def export_metrics_json(self, filename: str | None = None) -> str:
        """
        Export complet des métriques au format JSON pour analyse externe.

        Génère fichier JSON avec rapport de statut complet horodaté
        pour archivage, analyse de tendances, ou intégration externe.

        Args:
            filename: Nom fichier personnalisé (généré automatiquement si None)

        Returns:
            str: Chemin du fichier généré
        """

        if filename is None:
            filename = f"cortex_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        status = self.get_comprehensive_status()

        with open(filename, 'w') as f:
            json.dump(status, f, indent=2, default=str)

        print(f"📊 Métriques exportées: {filename}")
        return filename

    def start_monitoring(self) -> None:
        """
        Démarrage du monitoring continu en thread dédié d'arrière-plan.

        Lance collecte périodique de métriques, accumulation historique,
        détection d'alertes, avec gestion d'erreurs robuste.
        Thread daemon pour arrêt automatique avec processus principal.
        """

        if self.monitoring_active:
            print("⚠️ Monitoring déjà actif")
            return

        self.monitoring_active = True

        def monitor_loop():
            while self.monitoring_active:
                try:
                    # Collecter les métriques
                    status = self.get_comprehensive_status()

                    # Stocker dans l'historique
                    self.metrics_history.append({'timestamp': datetime.now(), 'status': status})

                    # Vérifier les alertes
                    current_alerts = status['alerts']
                    for alert in current_alerts:
                        if alert not in self.performance_alerts[-10:]:  # Éviter les doublons récents
                            self.performance_alerts.append(alert)

                    time.sleep(self.refresh_interval)

                except Exception as e:
                    print(f"⚠️ Erreur monitoring: {e}")
                    time.sleep(self.refresh_interval)

        self.monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()

        print("📊 Monitoring démarré en arrière-plan")

    def stop_monitoring(self) -> None:
        """
        Arrêt gracieux du système de monitoring d'arrière-plan.

        Signal d'arrêt au thread de monitoring avec timeout
        de sécurité pour éviter blocages.
        """

        self.monitoring_active = False

        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

        print("🛑 Monitoring arrêté")


def main() -> int:
    """
    Point d'entrée principal pour interface interactive de monitoring.

    Propose choix entre tableau de bord temps réel, export JSON,
    ou rapport unique avec gestion d'erreurs complète.

    Returns:
        int: Code de sortie (0=succès, 1=erreur)
    """
    """Point d'entrée principal pour le monitoring"""

    print("📊 CORTEX MONITOR - DÉMARRAGE")
    print("=" * 50)

    try:
        monitor = CortexMonitor()

        # Choix du mode
        print("\nModes disponibles:")
        print("1. Tableau de bord temps réel")
        print("2. Export métriques JSON")
        print("3. Rapport unique")

        choice = input("\nChoisissez un mode (1-3): ").strip()

        if choice == "1":
            print("\n🚀 Lancement du tableau de bord temps réel...")
            monitor.display_live_dashboard()

        elif choice == "2":
            filename = monitor.export_metrics_json()
            print(f"✅ Métriques exportées dans {filename}")

        elif choice == "3":
            status = monitor.get_comprehensive_status()
            print("\n📊 RAPPORT DE STATUT COMPLET")
            print("=" * 50)
            print(json.dumps(status, indent=2, default=str))

        else:
            print("❌ Choix invalide")

    except Exception as e:
        print(f"❌ Erreur: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
