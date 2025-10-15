"""
Module système pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module système pour jeffrey os.
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
import logging
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .guardian_communication import EventBus, EventType, GuardianEvent, guardian_bus

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Correlation:
    """Structure représentant une corrélation entre événements"""

    id: str
    events: list[GuardianEvent]
    pattern_type: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Insight:
    """Structure représentant un insight généré"""

    id: str
    title: str
    description: str
    severity: float
    action_required: bool
    suggested_actions: list[str]
    source_correlations: list[Correlation]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ImprovementProposal:
    """Structure représentant une proposition d'amélioration"""

    id: str
    title: str
    description: str
    impact_analysis: dict[str, Any]
    implementation_steps: list[str]
    estimated_benefits: dict[str, float]
    priority: float
    timestamp: datetime = field(default_factory=datetime.now)


class GuardianSymphony:
    """
    Orchestrateur central qui coordonne tous les gardiens
    Implémente l'intelligence collective et l'auto-régulation
    """

    def __init__(self, config_path: str = "config/symphony.json", event_bus: EventBus | None = None):
        self.config = self._load_config(config_path)
        self.event_bus = event_bus or guardian_bus
        self.running = False

        # Stockage
        self.correlations: list[Correlation] = []
        self.insights: list[Insight] = []
        self.proposals: list[ImprovementProposal] = []

        # Patterns de corrélation
        self.correlation_patterns = self._init_correlation_patterns()

        # Gardiens
        self.guardians = {}
        self._init_guardians()

        # Statistiques
        self.stats = {
            "correlations_found": 0,
            "insights_generated": 0,
            "proposals_created": 0,
            "events_processed": 0,
        }

        logger.info("Guardian Symphony initialized")

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Charge la configuration de la symphonie"""
        default_config = {
            "correlation_rules": {
                "complexity_cost": {
                    "description": "Corrèle complexité élevée avec coûts API",
                    "events": ["COMPLEXITY_ALERT", "COST_THRESHOLD"],
                    "time_window_seconds": 3600,
                    "min_severity": 0.6,
                },
                "bias_documentation": {
                    "description": "Corrèle biais avec manque de documentation",
                    "events": ["BIAS_DETECTED", "DOC_MISSING"],
                    "time_window_seconds": 7200,
                    "min_severity": 0.5,
                },
                "quality_cost_bias": {
                    "description": "Triple corrélation qualité-coût-biais",
                    "events": ["QUALITY_DROP", "COST_THRESHOLD", "BIAS_DETECTED"],
                    "time_window_seconds": 3600,
                    "min_severity": 0.7,
                },
            },
            "insight_thresholds": {
                "min_correlations": 2,
                "confidence_threshold": 0.7,
                "action_threshold": 0.8,
            },
            "proposal_settings": {
                "min_insights": 3,
                "auto_generate": True,
                "priority_threshold": 0.75,
            },
            "monitoring": {
                "health_check_interval": 60,
                "correlation_window": 3600,
                "max_events_buffer": 1000,
            },
        }

        config_file = Path(config_path)
        if config_file.exists():
            try:
                with open(config_file) as f:
                    loaded_config = json.load(f)
                    # Fusionner avec les défauts
                    for key, value in loaded_config.items():
                        if isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
        else:
            # Créer le fichier avec les défauts
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w") as f:
                json.dump(default_config, f, indent=2)

        return default_config

    def _init_guardians(self):
        """Initialise les références aux gardiens"""
        try:
            from .doc_zen import DocZen
            from .ethics_guardian import EthicsGuardian
            from .jeffrey_auditor import JeffreyAuditor
            from .resource_zen import ResourceZen

            self.guardians = {
                "ethics": EthicsGuardian(),
                "resources": ResourceZen(),
                "auditor": JeffreyAuditor(),
                "docs": DocZen(),
            }

            logger.info("All guardians initialized")
        except ImportError as e:
            logger.error(f"Failed to import guardians: {e}")
            self.guardians = {}

    def _init_correlation_patterns(self) -> dict[str, Any]:
        """Initialise les patterns de corrélation"""
        return {
            "complexity_cost": self._correlate_complexity_cost,
            "bias_documentation": self._correlate_bias_documentation,
            "quality_cost_bias": self._correlate_quality_cost_bias,
            "error_spike": self._correlate_error_spike,
            "performance_degradation": self._correlate_performance_degradation,
        }

    async def start(self):
        """Démarre l'orchestration"""
        self.running = True
        logger.info("Starting Guardian Symphony...")

        # S'abonner aux événements
        self._subscribe_to_events()

        # Démarrer les tâches de fond
        tasks = [
            await asyncio.create_task(self._event_processor()),
            await asyncio.create_task(self._correlation_engine()),
            await asyncio.create_task(self._insight_generator()),
            await asyncio.create_task(self._proposal_engine()),
            await asyncio.create_task(self._health_monitor()),
        ]

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.info("Symphony tasks cancelled")
        except Exception as e:
            logger.error(f"Symphony error: {e}")
        finally:
            self.running = False

    def stop(self):
        """Arrête l'orchestration"""
        logger.info("Stopping Guardian Symphony...")
        self.running = False

    def _subscribe_to_events(self):
        """S'abonne à tous les événements pertinents"""
        # S'abonner à tous les types d'événements
        for event_type in EventType:
            if event_type != EventType.CROSS_CORRELATION:  # Éviter la boucle
                self.event_bus.subscribe(event_type, self._handle_event)

    def _handle_event(self, event: GuardianEvent):
        """Gère un événement reçu"""
        self.stats["events_processed"] += 1

        # Logger l'événement important
        if event.severity > 0.7:
            logger.warning(f"High severity event: {event.event_type.value} from {event.source}")

        # L'événement sera traité par les tâches asynchrones

    async def _event_processor(self):
        """Traite les événements en continu"""
        while self.running:
            try:
                # Récupérer les événements récents
                recent_events = self.event_bus.get_recent_events(limit=100)

                # Analyser les patterns
                for pattern_name, pattern_func in self.correlation_patterns.items():
                    correlations = await pattern_func(recent_events)

                    for correlation in correlations:
                        if correlation.confidence > self.config["insight_thresholds"]["confidence_threshold"]:
                            self.correlations.append(correlation)
                            self.stats["correlations_found"] += 1

                            # Publier un événement de corrélation
                            correlation_event = GuardianEvent(
                                id=str(uuid.uuid4()),
                                source="symphony",
                                event_type=EventType.CROSS_CORRELATION,
                                severity=max(e.severity for e in correlation.events),
                                data={
                                    "correlation_id": correlation.id,
                                    "pattern_type": correlation.pattern_type,
                                    "confidence": correlation.confidence,
                                    "events_count": len(correlation.events),
                                },
                            )

                            self.event_bus.publish(correlation_event)

                await asyncio.sleep(5)  # Traiter toutes les 5 secondes

            except Exception as e:
                logger.error(f"Error in event processor: {e}")
                await asyncio.sleep(5)

    async def _correlation_engine(self):
        """Moteur de corrélation principal"""
        while self.running:
            try:
                # Nettoyer les vieilles corrélations
                cutoff = datetime.now() - timedelta(hours=24)
                self.correlations = [c for c in self.correlations if c.timestamp > cutoff]

                # Analyser les corrélations complexes
                await self._analyze_complex_patterns()

                await asyncio.sleep(30)  # Analyser toutes les 30 secondes

            except Exception as e:
                logger.error(f"Error in correlation engine: {e}")
                await asyncio.sleep(30)

    async def _insight_generator(self):
        """Génère des insights basés sur les corrélations"""
        while self.running:
            try:
                # Analyser les corrélations récentes
                recent_correlations = [
                    c for c in self.correlations if c.timestamp > datetime.now() - timedelta(hours=1)
                ]

                if len(recent_correlations) >= self.config["insight_thresholds"]["min_correlations"]:
                    insights = await self._generate_insights(recent_correlations)

                    for insight in insights:
                        self.insights.append(insight)
                        self.stats["insights_generated"] += 1

                        # Publier l'insight
                        insight_event = GuardianEvent(
                            id=str(uuid.uuid4()),
                            source="symphony",
                            event_type=EventType.INSIGHT_GENERATED,
                            severity=insight.severity,
                            data={
                                "insight_id": insight.id,
                                "title": insight.title,
                                "action_required": insight.action_required,
                            },
                        )

                        await self.event_bus.publish_async(insight_event)

                        # Logger les insights importants
                        if insight.severity > 0.8:
                            logger.warning(f"Critical insight: {insight.title}")

                await asyncio.sleep(60)  # Générer toutes les minutes

            except Exception as e:
                logger.error(f"Error in insight generator: {e}")
                await asyncio.sleep(60)

    async def _proposal_engine(self):
        """Génère des propositions d'amélioration"""
        while self.running:
            try:
                # Analyser les insights récents
                recent_insights = [i for i in self.insights if i.timestamp > datetime.now() - timedelta(hours=6)]

                if len(recent_insights) >= self.config["proposal_settings"]["min_insights"]:
                    proposals = await self._generate_proposals(recent_insights)

                    for proposal in proposals:
                        if proposal.priority > self.config["proposal_settings"]["priority_threshold"]:
                            self.proposals.append(proposal)
                            self.stats["proposals_created"] += 1

                            # Créer le fichier de proposition
                            await self._save_proposal(proposal)

                            # Publier l'événement
                            proposal_event = GuardianEvent(
                                id=str(uuid.uuid4()),
                                source="symphony",
                                event_type=EventType.IMPROVEMENT_PROPOSED,
                                severity=proposal.priority,
                                data={
                                    "proposal_id": proposal.id,
                                    "title": proposal.title,
                                    "priority": proposal.priority,
                                    "estimated_benefits": proposal.estimated_benefits,
                                },
                            )

                            await self.event_bus.publish_async(proposal_event)

                            logger.info(f"New proposal: {proposal.title} (priority: {proposal.priority:.2f})")

                await asyncio.sleep(300)  # Toutes les 5 minutes

            except Exception as e:
                logger.error(f"Error in proposal engine: {e}")
                await asyncio.sleep(300)

    async def _health_monitor(self):
        """Monitore la santé du système"""
        while self.running:
            try:
                health = await self._calculate_system_health()

                # Logger la santé
                logger.info(f"System health: {health['score']:.2f}/1.0")

                # Alerter si santé faible
                if health["score"] < 0.5:
                    logger.warning(f"Low system health: {health['issues']}")

                await asyncio.sleep(self.config["monitoring"]["health_check_interval"])

            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)

    # Méthodes de corrélation spécifiques

    async def _correlate_complexity_cost(self, events: list[GuardianEvent]) -> list[Correlation]:
        """Corrèle la complexité du code avec les coûts API"""
        correlations = []

        # Chercher les événements de complexité
        complexity_events = [e for e in events if e.event_type == EventType.COMPLEXITY_ALERT]
        cost_events = [e for e in events if e.event_type == EventType.COST_THRESHOLD]

        time_window = timedelta(seconds=self.config["correlation_rules"]["complexity_cost"]["time_window_seconds"])

        for comp_event in complexity_events:
            # Chercher les événements de coût proches
            related_costs = [cost for cost in cost_events if abs(comp_event.timestamp - cost.timestamp) < time_window]

            if related_costs:
                # Vérifier si le module complexe est mentionné dans les coûts
                module_path = comp_event.data.get("module_path", "")

                for cost_event in related_costs:
                    if module_path in str(cost_event.data):
                        correlation = Correlation(
                            id=str(uuid.uuid4()),
                            events=[comp_event, cost_event],
                            pattern_type="complexity_cost",
                            confidence=0.85,
                        )
                        correlations.append(correlation)

        return correlations

    async def _correlate_bias_documentation(self, events: list[GuardianEvent]) -> list[Correlation]:
        """Corrèle les biais détectés avec le manque de documentation"""
        correlations = []

        bias_events = [e for e in events if e.event_type == EventType.BIAS_DETECTED]
        doc_events = [e for e in events if e.event_type == EventType.DOC_MISSING]

        time_window = timedelta(seconds=self.config["correlation_rules"]["bias_documentation"]["time_window_seconds"])

        for bias_event in bias_events:
            # Chercher les événements de doc manquante proches
            related_docs = [doc for doc in doc_events if abs(bias_event.timestamp - doc.timestamp) < time_window]

            if related_docs:
                # Calculer la confiance basée sur la proximité temporelle
                for doc_event in related_docs:
                    time_diff = abs(bias_event.timestamp - doc_event.timestamp).total_seconds()
                    confidence = 1.0 - (time_diff / time_window.total_seconds()) * 0.3

                    correlation = Correlation(
                        id=str(uuid.uuid4()),
                        events=[bias_event, doc_event],
                        pattern_type="bias_documentation",
                        confidence=confidence,
                    )
                    correlations.append(correlation)

        return correlations

    async def _correlate_quality_cost_bias(self, events: list[GuardianEvent]) -> list[Correlation]:
        """Triple corrélation entre qualité, coût et biais"""
        correlations = []

        quality_events = [e for e in events if e.event_type == EventType.QUALITY_DROP]
        cost_events = [e for e in events if e.event_type == EventType.COST_THRESHOLD]
        bias_events = [e for e in events if e.event_type == EventType.BIAS_DETECTED]

        time_window = timedelta(seconds=self.config["correlation_rules"]["quality_cost_bias"]["time_window_seconds"])

        for quality_event in quality_events:
            # Chercher les événements liés
            related_costs = [c for c in cost_events if abs(quality_event.timestamp - c.timestamp) < time_window]
            related_biases = [b for b in bias_events if abs(quality_event.timestamp - b.timestamp) < time_window]

            if related_costs and related_biases:
                # Créer une corrélation triple
                events_list = [quality_event] + related_costs[:1] + related_biases[:1]

                correlation = Correlation(
                    id=str(uuid.uuid4()),
                    events=events_list,
                    pattern_type="quality_cost_bias",
                    confidence=0.9,
                )
                correlations.append(correlation)

        return correlations

    async def _correlate_error_spike(self, events: list[GuardianEvent]) -> list[Correlation]:
        """Détecte les pics d'erreurs corrélés"""
        correlations = []

        # Grouper les événements par fenêtre temporelle
        time_buckets = defaultdict(list)
        bucket_size = 300  # 5 minutes

        for event in events:
            if event.severity > 0.6:  # Seulement les événements significatifs
                bucket = int(event.timestamp.timestamp() / bucket_size)
                time_buckets[bucket].append(event)

        # Chercher les buckets avec beaucoup d'événements
        for bucket, bucket_events in time_buckets.items():
            if len(bucket_events) > 5:  # Pic détecté
                correlation = Correlation(
                    id=str(uuid.uuid4()),
                    events=bucket_events,
                    pattern_type="error_spike",
                    confidence=min(0.95, len(bucket_events) / 10),
                )
                correlations.append(correlation)

        return correlations

    async def _correlate_performance_degradation(self, events: list[GuardianEvent]) -> list[Correlation]:
        """Détecte les dégradations de performance corrélées"""
        correlations = []

        # Types d'événements indiquant une dégradation
        degradation_types = [
            EventType.QUALITY_DROP,
            EventType.COMPLEXITY_ALERT,
            EventType.TEST_COVERAGE_LOW,
            EventType.USAGE_SPIKE,
        ]

        degradation_events = [e for e in events if e.event_type in degradation_types]

        # Grouper par source
        by_source = defaultdict(list)
        for event in degradation_events:
            by_source[event.source].append(event)

        # Chercher les sources avec plusieurs problèmes
        for source, source_events in by_source.items():
            if len(source_events) > 3:
                correlation = Correlation(
                    id=str(uuid.uuid4()),
                    events=source_events[:5],  # Limiter
                    pattern_type="performance_degradation",
                    confidence=0.8,
                )
                correlations.append(correlation)

        return correlations

    async def _analyze_complex_patterns(self):
        """Analyse des patterns complexes multi-événements"""
        # Implémenter des analyses ML plus sophistiquées ici
        # Pour l'instant, on garde simple
        pass

    async def _generate_insights(self, correlations: list[Correlation]) -> list[Insight]:
        """Génère des insights à partir des corrélations"""
        insights = []

        # Grouper les corrélations par type
        by_pattern = defaultdict(list)
        for corr in correlations:
            by_pattern[corr.pattern_type].append(corr)

        # Générer des insights par pattern
        for pattern_type, pattern_correlations in by_pattern.items():
            if pattern_type == "complexity_cost":
                insight = await self._insight_complexity_cost(pattern_correlations)
                if insight:
                    insights.append(insight)

            elif pattern_type == "bias_documentation":
                insight = await self._insight_bias_documentation(pattern_correlations)
                if insight:
                    insights.append(insight)

            elif pattern_type == "quality_cost_bias":
                insight = await self._insight_quality_cost_bias(pattern_correlations)
                if insight:
                    insights.append(insight)

            elif pattern_type == "error_spike":
                insight = await self._insight_error_spike(pattern_correlations)
                if insight:
                    insights.append(insight)

        return insights

    async def _insight_complexity_cost(self, correlations: list[Correlation]) -> Insight | None:
        """Génère un insight sur la corrélation complexité-coût"""
        if not correlations:
            return None

        # Analyser les modules problématiques
        problematic_modules = set()
        total_cost_impact = 0

        for corr in correlations:
            for event in corr.events:
                if event.event_type == EventType.COMPLEXITY_ALERT:
                    module = event.data.get("module_path", "unknown")
                    problematic_modules.add(module)
                elif event.event_type == EventType.COST_THRESHOLD:
                    total_cost_impact += event.data.get("cost", 0)

        if problematic_modules:
            return Insight(
                id=str(uuid.uuid4()),
                title="Modules complexes générant des coûts élevés",
                description=f"{len(problematic_modules)} modules avec complexité élevée génèrent des coûts API importants (${total_cost_impact:.2f})",
                severity=0.8,
                action_required=True,
                suggested_actions=[f"Refactoriser le module {module}" for module in list(problematic_modules)[:3]]
                + ["Implémenter du caching pour réduire les appels API"],
                source_correlations=correlations,
            )

        return None

    async def _insight_bias_documentation(self, correlations: list[Correlation]) -> Insight | None:
        """Génère un insight sur la corrélation biais-documentation"""
        if not correlations:
            return None

        undocumented_biased_count = len(correlations)

        if undocumented_biased_count > 2:
            return Insight(
                id=str(uuid.uuid4()),
                title="Code biaisé sans documentation adéquate",
                description=f"{undocumented_biased_count} cas de code avec biais détectés dans des modules non documentés",
                severity=0.7,
                action_required=True,
                suggested_actions=[
                    "Documenter les modules concernés",
                    "Ajouter des guidelines sur l'écriture inclusive",
                    "Réviser le code pour neutralité",
                ],
                source_correlations=correlations,
            )

        return None

    async def _insight_quality_cost_bias(self, correlations: list[Correlation]) -> Insight | None:
        """Génère un insight sur la triple corrélation"""
        if not correlations:
            return None

        return Insight(
            id=str(uuid.uuid4()),
            title="Problème systémique détecté",
            description="Corrélation entre baisse de qualité, augmentation des coûts et présence de biais",
            severity=0.9,
            action_required=True,
            suggested_actions=[
                "Audit complet du module concerné",
                "Revue de code approfondie",
                "Formation de l'équipe sur les bonnes pratiques",
                "Mise en place de gates de qualité automatiques",
            ],
            source_correlations=correlations,
        )

    async def _insight_error_spike(self, correlations: list[Correlation]) -> Insight | None:
        """Génère un insight sur les pics d'erreurs"""
        if not correlations:
            return None

        # Compter les événements par source
        source_counts = Counter()
        for corr in correlations:
            for event in corr.events:
                source_counts[event.source] += 1

        most_common_source = source_counts.most_common(1)[0] if source_counts else ("unknown", 0)

        return Insight(
            id=str(uuid.uuid4()),
            title="Pic d'erreurs détecté",
            description=f"Augmentation anormale des erreurs, principalement depuis {most_common_source[0]}",
            severity=0.85,
            action_required=True,
            suggested_actions=[
                "Investiguer la cause racine immédiatement",
                "Vérifier les logs détaillés",
                "Activer le mode debug si nécessaire",
                "Préparer un rollback si nécessaire",
            ],
            source_correlations=correlations,
        )

    async def _generate_proposals(self, insights: list[Insight]) -> list[ImprovementProposal]:
        """Génère des propositions d'amélioration à partir des insights"""
        proposals = []

        # Analyser les thèmes récurrents
        themes = defaultdict(list)
        for insight in insights:
            if "complex" in insight.title.lower() or "cost" in insight.title.lower():
                themes["optimization"].append(insight)
            if "bias" in insight.title.lower() or "documentation" in insight.title.lower():
                themes["quality"].append(insight)
            if "error" in insight.title.lower() or "spike" in insight.title.lower():
                themes["stability"].append(insight)

        # Générer des propositions par thème
        for theme, theme_insights in themes.items():
            if len(theme_insights) >= 2:
                if theme == "optimization":
                    proposal = await self._proposal_optimization(theme_insights)
                elif theme == "quality":
                    proposal = await self._proposal_quality(theme_insights)
                elif theme == "stability":
                    proposal = await self._proposal_stability(theme_insights)
                else:
                    continue

                if proposal:
                    proposals.append(proposal)

        return proposals

    async def _proposal_optimization(self, insights: list[Insight]) -> ImprovementProposal | None:
        """Génère une proposition d'optimisation"""
        # Calculer l'impact potentiel
        total_cost_reduction = sum(i.severity * 100 for i in insights)  # Estimation

        return ImprovementProposal(
            id=str(uuid.uuid4()),
            title="Plan d'optimisation des coûts et performances",
            description="Refactoring ciblé des modules complexes pour réduire les coûts API",
            impact_analysis={
                "modules_affected": len(insights) * 2,  # Estimation
                "cost_reduction_monthly": total_cost_reduction,
                "complexity_reduction": "30-40%",
                "timeline_days": 14,
            },
            implementation_steps=[
                "1. Identifier les 5 modules les plus coûteux",
                "2. Analyser les patterns d'utilisation API",
                "3. Implémenter un système de cache intelligent",
                "4. Refactoriser pour réduire la complexité",
                "5. Optimiser les prompts pour moins de tokens",
                "6. Mettre en place des tests de performance",
            ],
            estimated_benefits={
                "cost_reduction": 0.35,
                "performance_improvement": 0.25,
                "maintainability": 0.40,
            },
            priority=0.85,
        )

    async def _proposal_quality(self, insights: list[Insight]) -> ImprovementProposal | None:
        """Génère une proposition d'amélioration qualité"""
        return ImprovementProposal(
            id=str(uuid.uuid4()),
            title="Initiative qualité et documentation",
            description="Programme d'amélioration de la qualité du code et de la documentation",
            impact_analysis={
                "code_coverage_increase": "+25%",
                "documentation_coverage": "+40%",
                "bias_reduction": "80%",
                "timeline_days": 21,
            },
            implementation_steps=[
                "1. Audit complet de la documentation existante",
                "2. Génération automatique de docstrings manquantes",
                "3. Formation sur l'écriture inclusive",
                "4. Mise en place de linters pour détecter les biais",
                "5. Revue systématique du code existant",
                "6. Documentation des bonnes pratiques",
            ],
            estimated_benefits={
                "code_quality": 0.45,
                "team_productivity": 0.30,
                "bias_reduction": 0.80,
            },
            priority=0.75,
        )

    async def _proposal_stability(self, insights: list[Insight]) -> ImprovementProposal | None:
        """Génère une proposition d'amélioration de stabilité"""
        return ImprovementProposal(
            id=str(uuid.uuid4()),
            title="Plan de stabilisation système",
            description="Amélioration de la résilience et réduction des erreurs",
            impact_analysis={
                "error_rate_reduction": "70%",
                "uptime_improvement": "99.9%",
                "response_time": "-30%",
                "timeline_days": 30,
            },
            implementation_steps=[
                "1. Implémenter un système de monitoring avancé",
                "2. Ajouter des circuit breakers sur les APIs",
                "3. Mettre en place des retry avec backoff",
                "4. Créer des health checks automatiques",
                "5. Implémenter des fallbacks gracieux",
                "6. Automatiser les tests de charge",
            ],
            estimated_benefits={
                "stability": 0.70,
                "user_satisfaction": 0.60,
                "operational_cost": 0.25,
            },
            priority=0.90,
        )

    async def _save_proposal(self, proposal: ImprovementProposal):
        """Sauvegarde une proposition d'amélioration"""
        proposals_dir = Path("proposals")
        proposals_dir.mkdir(exist_ok=True)

        proposal_file = proposals_dir / f"proposal_{proposal.id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        proposal_dict = {
            "id": proposal.id,
            "title": proposal.title,
            "description": proposal.description,
            "impact_analysis": proposal.impact_analysis,
            "implementation_steps": proposal.implementation_steps,
            "estimated_benefits": proposal.estimated_benefits,
            "priority": proposal.priority,
            "timestamp": proposal.timestamp.isoformat(),
            "status": "pending",
        }

        with open(proposal_file, "w") as f:
            json.dump(proposal_dict, f, indent=2)

        # Créer aussi un fichier Markdown pour lisibilité
        md_file = proposal_file.with_suffix(".md")
        with open(md_file, "w") as f:
            f.write(f"# {proposal.title}\n\n")
            f.write(f"**Priority:** {proposal.priority:.2f}\n")
            f.write(f"**Date:** {proposal.timestamp.strftime('%Y-%m-%d %H:%M')}\n\n")
            f.write(f"## Description\n{proposal.description}\n\n")
            f.write("## Impact Analysis\n")
            for key, value in proposal.impact_analysis.items():
                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
            f.write("\n## Implementation Steps\n")
            for step in proposal.implementation_steps:
                f.write(f"{step}\n")
            f.write("\n## Estimated Benefits\n")
            for benefit, value in proposal.estimated_benefits.items():
                f.write(f"- **{benefit.replace('_', ' ').title()}:** {value * 100:.0f}%\n")

    async def _calculate_system_health(self) -> dict[str, Any]:
        """Calcule la santé globale du système"""
        health_factors = {
            "no_critical_events": 0.25,
            "low_error_rate": 0.25,
            "good_performance": 0.20,
            "cost_under_control": 0.15,
            "documentation_ok": 0.15,
        }

        score = 0.0
        issues = []

        # Analyser les événements récents
        recent_events = self.event_bus.get_recent_events(limit=100)

        # Facteur 1: Pas d'événements critiques
        critical_count = sum(1 for e in recent_events if e.severity > 0.8)
        if critical_count == 0:
            score += health_factors["no_critical_events"]
        else:
            issues.append(f"{critical_count} critical events detected")

        # Facteur 2: Taux d'erreur bas
        error_events = [e for e in recent_events if "error" in e.event_type.value.lower() or e.severity > 0.7]
        error_rate = len(error_events) / max(len(recent_events), 1)
        if error_rate < 0.1:
            score += health_factors["low_error_rate"]
        else:
            issues.append(f"High error rate: {error_rate * 100:.1f}%")

        # Facteur 3: Bonne performance
        complexity_events = [e for e in recent_events if e.event_type == EventType.COMPLEXITY_ALERT]
        if len(complexity_events) < 5:
            score += health_factors["good_performance"]
        else:
            issues.append("Performance issues detected")

        # Facteur 4: Coûts sous contrôle
        cost_events = [e for e in recent_events if e.event_type in [EventType.COST_THRESHOLD, EventType.LIMIT_EXCEEDED]]
        if len(cost_events) == 0:
            score += health_factors["cost_under_control"]
        else:
            issues.append("Cost control issues")

        # Facteur 5: Documentation OK
        doc_events = [e for e in recent_events if e.event_type == EventType.DOC_MISSING]
        if len(doc_events) < 3:
            score += health_factors["documentation_ok"]
        else:
            issues.append("Documentation gaps detected")

        return {
            "score": score,
            "status": "healthy" if score > 0.7 else "degraded" if score > 0.4 else "critical",
            "issues": issues,
            "factors": {
                "critical_events": critical_count,
                "error_rate": error_rate,
                "complexity_issues": len(complexity_events),
                "cost_issues": len(cost_events),
                "doc_issues": len(doc_events),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def get_dashboard_data(self) -> dict[str, Any]:
        """Retourne les données pour un dashboard de monitoring"""
        recent_events = self.event_bus.get_recent_events(limit=200)
        recent_correlations = self.correlations[-20:] if self.correlations else []
        recent_insights = self.insights[-10:] if self.insights else []
        recent_proposals = self.proposals[-5:] if self.proposals else []

        # Calculer les métriques
        events_by_type = defaultdict(int)
        events_by_source = defaultdict(int)
        severity_distribution = {"low": 0, "medium": 0, "high": 0, "critical": 0}

        for event in recent_events:
            events_by_type[event.event_type.value] += 1
            events_by_source[event.source] += 1

            if event.severity < 0.3:
                severity_distribution["low"] += 1
            elif event.severity < 0.6:
                severity_distribution["medium"] += 1
            elif event.severity < 0.8:
                severity_distribution["high"] += 1
            else:
                severity_distribution["critical"] += 1

        # Calculer la santé système de manière synchrone pour le dashboard
        health = {"score": 0.5, "status": "unknown", "issues": []}
        try:
            # Utiliser une version simplifiée synchrone
            health = self._calculate_system_health_sync()
        except Exception as e:
            logger.error(f"Error calculating health: {e}")

        return {
            "summary": {
                "total_events": len(recent_events),
                "active_correlations": len(recent_correlations),
                "insights_generated": len(recent_insights),
                "pending_proposals": len(recent_proposals),
                "system_health": health,
            },
            "events": {
                "by_type": dict(events_by_type),
                "by_source": dict(events_by_source),
                "severity_distribution": severity_distribution,
            },
            "recent_insights": [
                {
                    "id": i.id,
                    "title": i.title,
                    "severity": i.severity,
                    "timestamp": i.timestamp.isoformat(),
                }
                for i in recent_insights
            ],
            "recent_proposals": [
                {
                    "id": p.id,
                    "title": p.title,
                    "priority": p.priority,
                    "timestamp": p.timestamp.isoformat(),
                }
                for p in recent_proposals
            ],
            "statistics": self.stats,
            "timestamp": datetime.now().isoformat(),
        }

    def _calculate_system_health_sync(self) -> dict[str, Any]:
        """Version synchrone du calcul de santé système"""
        score = 0.5  # Score par défaut
        issues = []

        # Analyser les événements récents
        recent_events = self.event_bus.get_recent_events(limit=50)

        if recent_events:
            # Compter les événements critiques
            critical_count = sum(1 for e in recent_events if e.severity > 0.8)
            if critical_count == 0:
                score += 0.2
            else:
                issues.append(f"{critical_count} critical events")

            # Taux d'erreur
            error_rate = len([e for e in recent_events if e.severity > 0.7]) / len(recent_events)
            if error_rate < 0.1:
                score += 0.3
            else:
                issues.append(f"High error rate: {error_rate * 100:.1f}%")

        return {
            "score": min(1.0, score),
            "status": "healthy" if score > 0.7 else "degraded" if score > 0.4 else "critical",
            "issues": issues,
        }
