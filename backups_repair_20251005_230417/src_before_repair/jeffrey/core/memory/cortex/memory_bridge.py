#!/usr/bin/env python3
"""
🌉 Jeffrey V2.0 UX Memory Bridge - Pont entre Living Memory et AGI Orchestrator
Intègre harmonieusement le système UX Living Memory avec l'architecture AGI existante

Enrichit chaque interaction avec l'intelligence émotionnelle relationnelle
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .emotional_timeline import EmotionalTimelineEngine, TimelineType, VisualizationStyle

# Import des systèmes UX Living Memory
from .living_memory import LivingMemoryCore, MemoryMoment, RelationshipStage
from .memory_rituals import MemoryRitualsEngine, RitualContext, RitualMood, RitualType
from .memory_templates import TemplateCoordinationContext, TemplateCoordinationStrategy, TemplateCoordinator
from .micro_interactions import InteractionTiming, MicroInteractionsEngine

logger = logging.getLogger(__name__)


@dataclass
class UXEnrichmentResult:
    """Résultat d'enrichissement UX pour une interaction"""

    # Analyse du moment
    memory_moment: MemoryMoment | None = None
    relationship_evolution: bool = False

    # Adaptations stylystiques
    communication_style: dict[str, Any] = None
    personalized_greeting: str | None = None
    personalized_closure: str | None = None

    # Micro-interactions
    micro_interaction: str | None = None
    surprise_opportunity: dict[str, Any] | None = None

    # Contexte relationnel
    relationship_stage: RelationshipStage = RelationshipStage.DISCOVERY
    intimacy_indicators: list[str] = None

    # Références enrichissantes
    contextual_references: list[str] = None
    emotional_bridges: list[str] = None

    # Métriques UX
    interaction_quality_score: float = 0.5
    emotional_resonance_score: float = 0.5
    personalization_level: float = 0.5


class UXMemoryBridge:
    """🌉 Pont principal entre UX Living Memory et AGI Orchestrator"""

    def __init__(self, storage_path: str = "data/ux_memory") -> None:
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)

        # Systèmes UX Living Memory
        self.living_memory: LivingMemoryCore | None = None
        self.rituals_engine: MemoryRitualsEngine | None = None
        self.micro_interactions: MicroInteractionsEngine | None = None
        self.timeline_engine: EmotionalTimelineEngine | None = None
        self.template_coordinator: TemplateCoordinator | None = None

        # Integration hooks
        self.capture_hooks: list[Callable] = []
        self.enrichment_hooks: list[Callable] = []

        # Performance metrics
        self.interaction_count = 0
        self.enrichment_success_rate = 0.0
        self.avg_processing_time = 0.0

        # États d'initialisation
        self.is_initialized = False
        self.initialization_error = None

    async def initialize(self):
        """🚀 Initialise tous les systèmes UX Living Memory"""
        try:
            start_time = time.time()

            # Création des répertoires de stockage
            Path(self.storage_path).mkdir(parents=True, exist_ok=True)
            Path(f"{self.storage_path}/living_memory").mkdir(exist_ok=True)
            Path(f"{self.storage_path}/rituals").mkdir(exist_ok=True)
            Path(f"{self.storage_path}/micro_interactions").mkdir(exist_ok=True)
            Path(f"{self.storage_path}/timelines").mkdir(exist_ok=True)
            Path(f"{self.storage_path}/templates").mkdir(exist_ok=True)

            # Initialisation en parallèle pour performance
            initialization_tasks = [
                self._initialize_living_memory(),
                self._initialize_rituals_engine(),
                self._initialize_micro_interactions(),
                self._initialize_timeline_engine(),
                self._initialize_template_coordinator(),
            ]

            results = await asyncio.gather(*initialization_tasks, return_exceptions=True)

            # Vérification des résultats
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.error(f"Erreur initialisation système UX {i}: {result}")
                    self.initialization_error = result
                    return False

            self.is_initialized = True
            init_time = time.time() - start_time

            self.logger.info(f"🌉 UX Memory Bridge initialisé en {init_time:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"Erreur critique lors de l'initialisation UX Memory Bridge: {e}")
            self.initialization_error = e
            return False

    async def _initialize_living_memory(self):
        """🧠 Initialise le cœur Living Memory"""
        self.living_memory = LivingMemoryCore(f"{self.storage_path}/living_memory")
        await self.living_memory.initialize()

    async def _initialize_rituals_engine(self):
        """🎭 Initialise le moteur de rituels"""
        self.rituals_engine = MemoryRitualsEngine(f"{self.storage_path}/rituals")
        await self.rituals_engine.initialize()

    async def _initialize_micro_interactions(self):
        """💝 Initialise le moteur de micro-interactions"""
        self.micro_interactions = MicroInteractionsEngine(f"{self.storage_path}/micro_interactions")
        await self.micro_interactions.initialize()

    async def _initialize_timeline_engine(self):
        """📊 Initialise le moteur de timeline émotionnelle"""
        self.timeline_engine = EmotionalTimelineEngine(f"{self.storage_path}/timelines")
        await self.timeline_engine.initialize()

    async def _initialize_template_coordinator(self):
        """🎨 Initialise le coordinateur de templates"""
        from .memory_templates.template_coordinator import create_template_coordinator

        self.template_coordinator = await create_template_coordinator(f"{self.storage_path}/templates")

    async def capture_and_enrich_interaction(
        self,
        user_input: str,
        ai_response: str,
        emotional_analysis: dict[str, Any],
        user_id: str,
        conversation_context: dict[str, Any] = None,
    ) -> UXEnrichmentResult:
        """🎭 Capture et enrichit une interaction avec l'intelligence UX"""

        if not self.is_initialized:
            self.logger.warning("UX Memory Bridge non initialisé, enrichissement basique")
            return UXEnrichmentResult()

        start_time = time.time()

        try:
            if conversation_context is None:
                conversation_context = {}

            # 1. Capture du moment mémorable
            memory_moment = await self._capture_memory_moment(
                user_input, ai_response, emotional_analysis, user_id, conversation_context
            )

            # 2. Analyse de l'évolution relationnelle
            relationship_evolution = await self._analyze_relationship_evolution(user_id, memory_moment)

            # 3. Génération des adaptations stylistiques
            style_adaptations = await self._generate_style_adaptations(user_id, user_input, conversation_context)

            # 4. Détection et génération de micro-interactions
            micro_interaction_result = await self._process_micro_interactions(
                user_id, user_input, ai_response, emotional_analysis, conversation_context
            )

            # 5. Génération de références contextuelles
            contextual_enrichments = await self._generate_contextual_enrichments(user_id, user_input, memory_moment)

            # 6. Calcul des métriques UX
            ux_metrics = await self._calculate_ux_metrics(
                memory_moment, relationship_evolution, style_adaptations, micro_interaction_result
            )

            # Assemblage du résultat
            enrichment_result = UXEnrichmentResult(
                memory_moment=memory_moment,
                relationship_evolution=relationship_evolution,
                communication_style=style_adaptations.get("style", {}),
                personalized_greeting=style_adaptations.get("greeting"),
                personalized_closure=style_adaptations.get("closure"),
                micro_interaction=micro_interaction_result.get("interaction"),
                surprise_opportunity=micro_interaction_result.get("surprise"),
                relationship_stage=await self._get_current_relationship_stage(user_id),
                intimacy_indicators=contextual_enrichments.get("intimacy_indicators", []),
                contextual_references=contextual_enrichments.get("references", []),
                emotional_bridges=contextual_enrichments.get("bridges", []),
                interaction_quality_score=ux_metrics.get("quality", 0.5),
                emotional_resonance_score=ux_metrics.get("resonance", 0.5),
                personalization_level=ux_metrics.get("personalization", 0.5),
            )

            # Mise à jour des métriques de performance
            processing_time = time.time() - start_time
            await self._update_performance_metrics(processing_time, True)

            # Trigger des hooks d'enrichissement
            await self._trigger_enrichment_hooks(enrichment_result)

            return enrichment_result

        except Exception as e:
            self.logger.error(f"Erreur lors de l'enrichissement UX: {e}")
            processing_time = time.time() - start_time
            await self._update_performance_metrics(processing_time, False)
            return UXEnrichmentResult()

    async def _capture_memory_moment(
        self,
        user_input: str,
        ai_response: str,
        emotional_analysis: dict[str, Any],
        user_id: str,
        context: dict[str, Any],
    ) -> MemoryMoment | None:
        """🧠 Capture un moment mémorable via Living Memory"""

        if not self.living_memory:
            return None

        try:
            memory_moment = await self.living_memory.capture_moment(
                user_input=user_input,
                jeffrey_response=ai_response,
                user_id=user_id,
                emotional_context=emotional_analysis.get("primary_emotion", "neutral"),
                intensity=emotional_analysis.get("intensity", 0.5),
                conversation_context=context,
            )

            # Trigger des hooks de capture
            if memory_moment:
                await self._trigger_capture_hooks(memory_moment)

            return memory_moment

        except Exception as e:
            self.logger.error(f"Erreur capture moment mémorable: {e}")
            return None

    async def _analyze_relationship_evolution(self, user_id: str, memory_moment: MemoryMoment | None) -> bool:
        """📈 Analyse l'évolution de la relation"""

        if not self.living_memory or not memory_moment:
            return False

        try:
            # Récupération du profil relationnel
            profile = await self.living_memory.get_relationship_profile(user_id)
            if not profile:
                return False

            # Détection d'évolution significative
            previous_stage = profile.relationship_stage
            current_stage = await self.living_memory.calculate_relationship_stage(user_id)

            # Mise à jour si évolution
            if current_stage != previous_stage:
                await self.living_memory.update_relationship_stage(user_id, current_stage)
                self.logger.info(
                    f"🎊 Évolution relationnelle user {user_id}: {previous_stage.value} → {current_stage.value}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Erreur analyse évolution relationnelle: {e}")
            return False

    async def _generate_style_adaptations(
        self, user_id: str, user_input: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        """🎨 Génère les adaptations de style via templates"""

        if not self.template_coordinator:
            return {}

        try:
            # Contexte de coordination
            coordination_context = TemplateCoordinationContext(
                user_id=user_id,
                base_content=user_input,
                conversation_type=context.get("conversation_type", "general"),
                emotional_context=context.get("emotional_context", "neutral"),
                time_context=context.get("time_context", "any_time"),
                recent_interactions=context.get("recent_interactions", []),
                user_input=user_input,
                coordination_strategy=TemplateCoordinationStrategy.BALANCED_FUSION,
            )

            # Génération d'adaptations
            style_response = await self.template_coordinator.generate_coordinated_response(coordination_context)

            # Génération de greeting/closure personnalisés
            personalized_greeting = None
            personalized_closure = None

            if self.rituals_engine:
                # Détection du mood
                mood = await self._detect_ritual_mood(context.get("emotional_context", "neutral"))

                # Contexte de rituel
                ritual_context = RitualContext(
                    user_id=user_id,
                    current_mood=mood,
                    time_context=context.get("time_context", "any_time"),
                    relationship_stage=await self._get_current_relationship_stage(user_id),
                )

                # Génération des rituels
                if context.get("is_conversation_start", False):
                    personalized_greeting = await self.rituals_engine.generate_ritual(
                        RitualType.GREETING, user_id, asdict(ritual_context)
                    )

                if context.get("is_conversation_end", False):
                    personalized_closure = await self.rituals_engine.generate_ritual(
                        RitualType.FAREWELL, user_id, asdict(ritual_context)
                    )

            return {
                "style": {"adapted_response": style_response},
                "greeting": personalized_greeting,
                "closure": personalized_closure,
            }

        except Exception as e:
            self.logger.error(f"Erreur génération adaptations style: {e}")
            return {}

    async def _process_micro_interactions(
        self,
        user_id: str,
        user_input: str,
        ai_response: str,
        emotional_analysis: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """💝 Traite les micro-interactions émotionnelles"""

        if not self.micro_interactions:
            return {}

        try:
            # Détection du timing optimal
            timing = self._determine_interaction_timing(context)

            # Traitement du contexte conversationnel
            micro_interaction_text = await self.micro_interactions.process_conversation_context(
                user_id=user_id,
                user_input=user_input,
                jeffrey_response=ai_response,
                timing=timing,
                emotional_context=emotional_analysis.get("primary_emotion", "neutral"),
                conversation_context=context,
            )

            # Détection d'opportunité de surprise
            surprise_opportunity = await self.micro_interactions.detect_surprise_opportunity(
                user_id=user_id,
                current_context=context,
                emotional_intensity=emotional_analysis.get("intensity", 0.5),
            )

            return {"interaction": micro_interaction_text, "surprise": surprise_opportunity}

        except Exception as e:
            self.logger.error(f"Erreur traitement micro-interactions: {e}")
            return {}

    async def _generate_contextual_enrichments(
        self, user_id: str, user_input: str, memory_moment: MemoryMoment | None
    ) -> dict[str, Any]:
        """🔗 Génère les enrichissements contextuels"""

        if not self.living_memory:
            return {}

        try:
            # Recherche de moments similaires
            similar_moments = await self.living_memory.find_similar_moments(user_id, user_input, limit=3)

            # Génération de références contextuelles
            references = []
            for moment in similar_moments:
                if moment.content and len(moment.content) > 10:
                    references.append(f"Comme la fois où {moment.content[:50]}...")

            # Détection d'indicateurs d'intimité
            intimacy_indicators = []
            if memory_moment:
                if memory_moment.emotional_intensity > 0.7:
                    intimacy_indicators.append("moment émotionnellement intense")
                if memory_moment.moment_type.value in ["personal_share", "bonding", "support"]:
                    intimacy_indicators.append("partage personnel")

            # Génération de ponts émotionnels
            bridges = []
            if similar_moments:
                bridges.append("Cette expérience résonne avec nos échanges précédents")

            return {
                "references": references,
                "intimacy_indicators": intimacy_indicators,
                "bridges": bridges,
            }

        except Exception as e:
            self.logger.error(f"Erreur génération enrichissements contextuels: {e}")
            return {}

    async def _calculate_ux_metrics(
        self,
        memory_moment: MemoryMoment | None,
        relationship_evolution: bool,
        style_adaptations: dict[str, Any],
        micro_interaction_result: dict[str, Any],
    ) -> dict[str, float]:
        """📊 Calcule les métriques UX de l'interaction"""

        quality_score = 0.5
        resonance_score = 0.5
        personalization_score = 0.5

        # Score de qualité basé sur la capture de moment
        if memory_moment:
            quality_score += memory_moment.importance_score * 0.3
            if memory_moment.emotional_intensity > 0.6:
                quality_score += 0.2

        # Score de résonance basé sur l'évolution relationnelle
        if relationship_evolution:
            resonance_score += 0.3

        # Score de personnalisation basé sur les adaptations
        if style_adaptations.get("greeting") or style_adaptations.get("closure"):
            personalization_score += 0.2

        if micro_interaction_result.get("interaction"):
            personalization_score += 0.2

        if micro_interaction_result.get("surprise"):
            personalization_score += 0.1
            resonance_score += 0.1

        # Normalisation des scores
        quality_score = min(1.0, max(0.0, quality_score))
        resonance_score = min(1.0, max(0.0, resonance_score))
        personalization_score = min(1.0, max(0.0, personalization_score))

        return {
            "quality": quality_score,
            "resonance": resonance_score,
            "personalization": personalization_score,
        }

    def _determine_interaction_timing(self, context: dict[str, Any]) -> InteractionTiming:
        """⏰ Détermine le timing optimal pour micro-interactions"""

        if context.get("is_conversation_start"):
            return InteractionTiming.CONVERSATION_START
        elif context.get("is_conversation_end"):
            return InteractionTiming.CONVERSATION_END
        elif context.get("emotional_peak"):
            return InteractionTiming.EMOTIONAL_PEAK
        elif context.get("after_personal_sharing"):
            return InteractionTiming.AFTER_SHARING
        elif context.get("natural_pause"):
            return InteractionTiming.NATURAL_PAUSE
        else:
            return InteractionTiming.CONVERSATION_MID

    async def _detect_ritual_mood(self, emotional_context: str) -> RitualMood:
        """🎭 Détecte l'humeur pour les rituels"""

        mood_mapping = {
            "happy": RitualMood.JOYFUL,
            "excited": RitualMood.ENTHUSIASTIC,
            "calm": RitualMood.CALM,
            "contemplative": RitualMood.CONTEMPLATIVE,
            "creative": RitualMood.CREATIVE,
            "grateful": RitualMood.GRATEFUL,
            "curious": RitualMood.CURIOUS,
            "nostalgic": RitualMood.NOSTALGIC,
            "tired": RitualMood.TIRED,
            "energetic": RitualMood.ENERGETIC,
        }

        return mood_mapping.get(emotional_context, RitualMood.CALM)

    async def _get_current_relationship_stage(self, user_id: str) -> RelationshipStage:
        """📊 Récupère le stade relationnel actuel"""

        if not self.living_memory:
            return RelationshipStage.DISCOVERY

        try:
            profile = await self.living_memory.get_relationship_profile(user_id)
            return profile.relationship_stage if profile else RelationshipStage.DISCOVERY
        except:
            return RelationshipStage.DISCOVERY

    async def _update_performance_metrics(self, processing_time: float, success: bool):
        """📈 Met à jour les métriques de performance"""

        self.interaction_count += 1

        # Mise à jour du temps de traitement moyen
        if self.interaction_count == 1:
            self.avg_processing_time = processing_time
        else:
            self.avg_processing_time = (
                self.avg_processing_time * (self.interaction_count - 1) + processing_time
            ) / self.interaction_count

        # Mise à jour du taux de succès
        if success:
            current_successes = self.enrichment_success_rate * (self.interaction_count - 1)
            self.enrichment_success_rate = (current_successes + 1) / self.interaction_count
        else:
            current_successes = self.enrichment_success_rate * (self.interaction_count - 1)
            self.enrichment_success_rate = current_successes / self.interaction_count

    async def _trigger_capture_hooks(self, memory_moment: MemoryMoment):
        """🪝 Déclenche les hooks de capture"""
        for hook in self.capture_hooks:
            try:
                await hook(memory_moment)
            except Exception as e:
                self.logger.error(f"Erreur hook capture: {e}")

    async def _trigger_enrichment_hooks(self, enrichment_result: UXEnrichmentResult):
        """🪝 Déclenche les hooks d'enrichissement"""
        for hook in self.enrichment_hooks:
            try:
                await hook(enrichment_result)
            except Exception as e:
                self.logger.error(f"Erreur hook enrichissement: {e}")

    def register_capture_hook(self, hook_func: Callable):
        """📌 Enregistre un hook de capture"""
        self.capture_hooks.append(hook_func)

    def register_enrichment_hook(self, hook_func: Callable):
        """📌 Enregistre un hook d'enrichissement"""
        self.enrichment_hooks.append(hook_func)

    async def get_ux_insights(self, user_id: str) -> dict[str, Any]:
        """📊 Récupère les insights UX pour un utilisateur"""

        insights = {
            "user_id": user_id,
            "bridge_status": {
                "initialized": self.is_initialized,
                "initialization_error": (str(self.initialization_error) if self.initialization_error else None),
                "systems_available": {
                    "living_memory": self.living_memory is not None,
                    "rituals_engine": self.rituals_engine is not None,
                    "micro_interactions": self.micro_interactions is not None,
                    "timeline_engine": self.timeline_engine is not None,
                    "template_coordinator": self.template_coordinator is not None,
                },
            },
            "performance_metrics": {
                "total_interactions": self.interaction_count,
                "avg_processing_time_ms": round(self.avg_processing_time * 1000, 2),
                "enrichment_success_rate": round(self.enrichment_success_rate, 3),
            },
        }

        # Insights des systèmes individuels
        if self.living_memory:
            try:
                living_insights = await self.living_memory.get_user_insights(user_id)
                insights["living_memory_insights"] = living_insights
            except Exception as e:
                insights["living_memory_error"] = str(e)

        if self.template_coordinator:
            try:
                template_insights = await self.template_coordinator.get_coordination_insights(user_id)
                insights["template_coordination_insights"] = template_insights
            except Exception as e:
                insights["template_coordination_error"] = str(e)

        return insights

    async def generate_emotional_timeline(
        self, user_id: str, timeline_type: TimelineType = TimelineType.RELATIONSHIP_EVOLUTION
    ) -> str | None:
        """📊 Génère une timeline émotionnelle pour l'utilisateur"""

        if not self.timeline_engine:
            return None

        try:
            timeline = await self.timeline_engine.generate_timeline(
                user_id=user_id, timeline_type=timeline_type, style=VisualizationStyle.STORYTELLING
            )

            if timeline:
                # Export en HTML pour visualisation
                return await self.timeline_engine.export_timeline(
                    timeline, "html", f"timeline_{user_id}_{timeline_type.value}.html"
                )

            return None

        except Exception as e:
            self.logger.error(f"Erreur génération timeline émotionnelle: {e}")
            return None


# Instance globale pour intégration facile
_ux_memory_bridge_instance = None


async def get_ux_memory_bridge() -> UXMemoryBridge:
    """🏭 Factory pour obtenir l'instance UX Memory Bridge"""
    global _ux_memory_bridge_instance

    if _ux_memory_bridge_instance is None:
        _ux_memory_bridge_instance = UXMemoryBridge()
        await _ux_memory_bridge_instance.initialize()

    return _ux_memory_bridge_instance


async def create_ux_memory_bridge(storage_path: str = "data/ux_memory") -> UXMemoryBridge:
    """🏭 Factory pour créer une nouvelle instance UX Memory Bridge"""
    bridge = UXMemoryBridge(storage_path)
    await bridge.initialize()
    return bridge
