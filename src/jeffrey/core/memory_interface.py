#!/usr/bin/env python3
"""
Jeffrey V2.0 Memory Interface - API claire et intuitive
Interface haut niveau pour interaction avec le système de mémoire

Features:
- API simplifiée pour AGI Orchestrator et DialogueEngine
- Hooks automatiques pour capture contextuelle
- Gestion asynchrone optimisée
- Integration seamless avec Emotion Engine
"""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime, timedelta
from typing import Any

from jeffrey.core.memory_systems import MemoryCore, MemoryEntry, MemoryType, get_memory_core

logger = logging.getLogger(__name__)


class MemoryInterface:
    """
    Interface haut niveau pour le système de mémoire Jeffrey V2.0

    Simplifie l'interaction avec MemoryCore et fournit des hooks
    automatiques pour l'intégration avec l'architecture existante.
    """

    def __init__(self, memory_core: MemoryCore | None = None):
        """
        Initialise l'interface mémoire

        Args:
            memory_core: Instance MemoryCore (utilise l'instance globale si None)
        """
        self.memory_core = memory_core or get_memory_core()
        self.auto_capture_enabled = True
        self.capture_hooks: list[Callable] = []
        self.recall_hooks: list[Callable] = []

        # Cache de session pour performance
        self._session_cache: dict[str, list[MemoryEntry]] = {}
        self._last_interaction_time = datetime.now()

        logger.info("🔗 MemoryInterface initialized - Ready for seamless integration")

    # ============================================================================
    # API PRINCIPALE - STOCKAGE
    # ============================================================================

    async def remember(
        self,
        content: str,
        context: dict[str, Any] | None = None,
        memory_type: str = "dialogue",
        importance: float | None = None,
    ) -> str:
        """
        API simplifiée pour mémoriser du contenu

        Args:
            content: Contenu à mémoriser
            context: Contexte additionnel (user_id, conversation_id, etc.)
            memory_type: Type de mémoire ("dialogue", "emotional", "creative", etc.)
            importance: Score d'importance forcé (0.0-1.0)

        Returns:
            str: ID de la mémoire créée
        """
        try:
            # Conversion du type string vers enum
            mem_type = self._parse_memory_type(memory_type)

            # Extraction du contexte
            ctx = context or {}
            user_id = ctx.get("user_id", "default")
            conversation_id = ctx.get("conversation_id", "")

            # Stockage via MemoryCore
            memory_id = await self.memory_core.store_memory(
                content=content, memory_type=mem_type, user_id=user_id, conversation_id=conversation_id, metadata=ctx
            )

            # Forcer l'importance si spécifiée
            if importance is not None:
                await self.memory_core.update_memory(memory_id, {"importance_score": max(0.0, min(1.0, importance))})

            # Déclencher les hooks de capture
            await self._trigger_capture_hooks(memory_id, content, ctx)

            return memory_id

        except Exception as e:
            logger.error(f"Erreur remember: {e}")
            raise

    async def remember_dialogue(
        self, user_input: str, ai_response: str, user_id: str = "default", emotional_context: dict | None = None
    ) -> str:
        """
        Mémorise un échange de dialogue complet

        Args:
            user_input: Entrée utilisateur
            ai_response: Réponse de Jeffrey
            user_id: ID utilisateur
            emotional_context: Contexte émotionnel

        Returns:
            str: ID de la mémoire dialogue
        """
        try:
            # Construire le contenu dialogue
            content = f"User: {user_input} | Jeffrey: {ai_response}"

            context = {
                "user_id": user_id,
                "conversation_id": f"conv_{datetime.now().strftime('%Y%m%d_%H')}",
                "emotional_context": emotional_context or {},
                "dialogue_components": {"user_input": user_input, "ai_response": ai_response},
            }

            return await self.remember(
                content=content,
                context=context,
                memory_type="dialogue",
                importance=0.6,  # Dialogues ont importance moyenne par défaut
            )

        except Exception as e:
            logger.error(f"Erreur remember_dialogue: {e}")
            raise

    async def remember_emotional_moment(
        self, emotion: str, intensity: float, context: str, user_id: str = "default"
    ) -> str:
        """
        Mémorise un moment émotionnel significatif

        Args:
            emotion: Émotion détectée
            intensity: Intensité (0.0-1.0)
            context: Contexte de l'émotion
            user_id: ID utilisateur

        Returns:
            str: ID de la mémoire émotionnelle
        """
        try:
            content = f"Moment émotionnel: {emotion} (intensité: {intensity:.1f}) - {context}"

            ctx = {"user_id": user_id, "emotion_data": {"emotion": emotion, "intensity": intensity, "context": context}}

            return await self.remember(
                content=content,
                context=ctx,
                memory_type="emotional",
                importance=0.7 + intensity * 0.3,  # Plus intense = plus important
            )

        except Exception as e:
            logger.error(f"Erreur remember_emotional_moment: {e}")
            raise

    # ============================================================================
    # API PRINCIPALE - RAPPEL
    # ============================================================================

    async def recall(
        self, query: str, user_id: str = "default", limit: int = 5, filters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """
        API simplifiée pour rappel contextuel

        Args:
            query: Requête de recherche
            user_id: ID utilisateur
            limit: Nombre max de résultats
            filters: Filtres additionnels (emotion, time_range, type)

        Returns:
            List[Dict]: Mémoires formatées pour utilisation
        """
        try:
            # Préparation des filtres
            filters = filters or {}
            emotional_filter = filters.get("emotion")
            time_range = self._parse_time_range(filters.get("time_range"))

            # Rappel via MemoryCore
            memories = await self.memory_core.recall_contextual(
                query=query, user_id=user_id, limit=limit, emotional_filter=emotional_filter, time_range=time_range
            )

            # Formatage pour l'API
            formatted_memories = []
            for memory in memories:
                formatted = self._format_memory_for_api(memory)
                formatted_memories.append(formatted)

            # Déclencher les hooks de rappel
            await self._trigger_recall_hooks(query, formatted_memories)

            return formatted_memories

        except Exception as e:
            logger.error(f"Erreur recall: {e}")
            return []

    async def recall_recent_conversation(self, user_id: str = "default", hours: int = 24) -> list[dict[str, Any]]:
        """
        Rappelle la conversation récente d'un utilisateur

        Args:
            user_id: ID utilisateur
            hours: Nombre d'heures dans le passé

        Returns:
            List[Dict]: Messages de la conversation récente
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)

            filters = {"time_range": (start_time, end_time), "type": "dialogue"}

            memories = await self.recall(
                query="",  # Requête vide pour récupérer tout
                user_id=user_id,
                limit=50,
                filters=filters,
            )

            # Trier par chronologie
            memories.sort(key=lambda m: m["timestamp"])

            return memories

        except Exception as e:
            logger.error(f"Erreur recall_recent_conversation: {e}")
            return []

    async def recall_by_emotion(self, emotion: str, user_id: str = "default", limit: int = 10) -> list[dict[str, Any]]:
        """
        Rappelle les mémoires associées à une émotion

        Args:
            emotion: Émotion à rechercher
            user_id: ID utilisateur
            limit: Nombre max de résultats

        Returns:
            List[Dict]: Mémoires émotionnelles
        """
        try:
            filters = {"emotion": emotion}

            return await self.recall(query=f"émotion {emotion}", user_id=user_id, limit=limit, filters=filters)

        except Exception as e:
            logger.error(f"Erreur recall_by_emotion: {e}")
            return []

    # ============================================================================
    # HOOKS ET INTÉGRATION
    # ============================================================================

    def register_capture_hook(self, hook_func: Callable):
        """
        Enregistre un hook appelé lors de la capture de mémoire

        Args:
            hook_func: Fonction appelée avec (memory_id, content, context)
        """
        self.capture_hooks.append(hook_func)
        logger.debug(f"🪝 Capture hook registered: {hook_func.__name__}")

    def register_recall_hook(self, hook_func: Callable):
        """
        Enregistre un hook appelé lors du rappel de mémoire

        Args:
            hook_func: Fonction appelée avec (query, memories)
        """
        self.recall_hooks.append(hook_func)
        logger.debug(f"🪝 Recall hook registered: {hook_func.__name__}")

    async def auto_capture_interaction(
        self, user_input: str, ai_response: str, emotional_analysis: dict | None = None, user_id: str = "default"
    ) -> str:
        """
        Capture automatique d'une interaction (hook pour AGI Orchestrator)

        Args:
            user_input: Entrée utilisateur
            ai_response: Réponse AI
            emotional_analysis: Analyse émotionnelle
            user_id: ID utilisateur

        Returns:
            str: ID de la mémoire créée
        """
        if not self.auto_capture_enabled:
            return ""

        try:
            # Mémoriser le dialogue
            memory_id = await self.remember_dialogue(
                user_input=user_input, ai_response=ai_response, user_id=user_id, emotional_context=emotional_analysis
            )

            # Si analyse émotionnelle forte, créer aussi une mémoire émotionnelle
            if emotional_analysis:
                emotion = emotional_analysis.get("emotion_dominante", "")
                intensity = emotional_analysis.get("intensite", 0) / 100.0

                if intensity > 0.6:  # Seuil pour mémoire émotionnelle
                    await self.remember_emotional_moment(
                        emotion=emotion,
                        intensity=intensity,
                        context=f"Dialogue: {user_input[:100]}...",
                        user_id=user_id,
                    )

            return memory_id

        except Exception as e:
            logger.error(f"Erreur auto_capture_interaction: {e}")
            return ""

    def enable_auto_capture(self):
        """Active la capture automatique"""
        self.auto_capture_enabled = True
        logger.info("✅ Auto-capture enabled")

    def disable_auto_capture(self):
        """Désactive la capture automatique"""
        self.auto_capture_enabled = False
        logger.info("❌ Auto-capture disabled")

    # ============================================================================
    # GESTION ET MAINTENANCE
    # ============================================================================

    async def cleanup_memories(self) -> dict[str, int]:
        """Lance le nettoyage des mémoires"""
        return await self.memory_core.process_memory_decay()

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques du système mémoire"""
        return self.memory_core.get_memory_stats()

    async def export_user_memories(self, user_id: str, format: str = "json") -> str:
        """Exporte les mémoires d'un utilisateur"""
        return self.memory_core.export_memories(user_id=user_id, format=format)

    async def get_memory_by_id(self, memory_id: str) -> dict[str, Any] | None:
        """Récupère une mémoire par son ID"""
        memory = self.memory_core.get_memory_by_id(memory_id)
        if memory:
            return self._format_memory_for_api(memory)
        return None

    async def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """Met à jour l'importance d'une mémoire"""
        return await self.memory_core.update_memory(memory_id, {"importance_score": max(0.0, min(1.0, importance))})

    async def delete_memory(self, memory_id: str) -> bool:
        """Supprime une mémoire"""
        return await self.memory_core.delete_memory(memory_id)

    # ============================================================================
    # MÉTHODES UTILITAIRES PRIVÉES
    # ============================================================================

    def _parse_memory_type(self, type_str: str) -> MemoryType:
        """Convertit une string en MemoryType"""
        type_mapping = {
            "dialogue": MemoryType.DIALOGUE,
            "emotional": MemoryType.EMOTIONAL,
            "factual": MemoryType.FACTUAL,
            "creative": MemoryType.CREATIVE,
            "relationship": MemoryType.RELATIONSHIP,
            "learning": MemoryType.LEARNING,
        }
        return type_mapping.get(type_str.lower(), MemoryType.DIALOGUE)

    def _parse_time_range(self, time_range_spec: Any) -> tuple | None:
        """Parse une spécification de plage temporelle"""
        if not time_range_spec:
            return None

        if isinstance(time_range_spec, (list, tuple)) and len(time_range_spec) == 2:
            return tuple(time_range_spec)

        if isinstance(time_range_spec, str):
            # Format "last_24h", "last_7d", etc.
            if time_range_spec.startswith("last_"):
                duration_str = time_range_spec[5:]
                if duration_str.endswith("h"):
                    hours = int(duration_str[:-1])
                    end_time = datetime.now()
                    start_time = end_time - timedelta(hours=hours)
                    return (start_time, end_time)
                elif duration_str.endswith("d"):
                    days = int(duration_str[:-1])
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=days)
                    return (start_time, end_time)

        return None

    def _format_memory_for_api(self, memory: MemoryEntry) -> dict[str, Any]:
        """Formate une MemoryEntry pour l'API"""
        try:
            # Conversion en dict de base
            formatted = asdict(memory)

            # Ajouts pour faciliter l'utilisation
            formatted["age_days"] = (datetime.now() - datetime.fromisoformat(memory.timestamp)).days

            formatted["freshness"] = memory.decay_factor
            formatted["relevance_boost"] = memory.importance_score * memory.decay_factor

            # Simplification des tags émotionnels
            if memory.emotional_tags:
                formatted["emotions"] = [
                    {"emotion": tag.emotion, "intensity": tag.intensity, "confidence": tag.confidence}
                    for tag in memory.emotional_tags
                ]
            else:
                formatted["emotions"] = []

            # Simplification du contexte
            formatted["keywords"] = memory.context_vector.keywords[:5]  # Top 5
            formatted["topics"] = memory.context_vector.topics

            return formatted

        except Exception as e:
            logger.warning(f"Erreur formatage mémoire: {e}")
            return {"error": "Format error", "id": memory.id}

    async def _trigger_capture_hooks(self, memory_id: str, content: str, context: dict):
        """Déclenche les hooks de capture"""
        for hook in self.capture_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(memory_id, content, context)
                else:
                    hook(memory_id, content, context)
            except Exception as e:
                logger.warning(f"Erreur hook capture: {e}")

    async def _trigger_recall_hooks(self, query: str, memories: list[dict]):
        """Déclenche les hooks de rappel"""
        for hook in self.recall_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook(query, memories)
                else:
                    hook(query, memories)
            except Exception as e:
                logger.warning(f"Erreur hook recall: {e}")


# ============================================================================
# INTÉGRATION AVEC AGI ORCHESTRATOR
# ============================================================================


class AGIMemoryIntegration:
    """
    Classe d'intégration spécialisée pour AGI Orchestrator
    Fournit des méthodes optimisées pour l'architecture Jeffrey
    """

    def __init__(self, memory_interface: MemoryInterface | None = None):
        self.memory = memory_interface or MemoryInterface()
        self._orchestrator = None
        self._emotion_engine = None

    def integrate_with_orchestrator(self, orchestrator):
        """Intègre avec AGI Orchestrator"""
        self._orchestrator = orchestrator

        # Hook automatique pour capture
        async def capture_hook(memory_id, content, context):
            if self._orchestrator:
                # Notifier l'orchestrator de la nouvelle mémoire
                pass

        self.memory.register_capture_hook(capture_hook)
        logger.info("🎯 AGI Orchestrator integration completed")

    def integrate_with_emotion_engine(self, emotion_engine):
        """Intègre avec Emotion Engine"""
        self._emotion_engine = emotion_engine
        self.memory.memory_core.integrate_emotion_engine(emotion_engine)
        logger.info("🎭 Emotion Engine integration completed")

    async def process_interaction_with_memory(
        self, user_input: str, ai_response: str, emotional_analysis: dict, user_id: str = "default"
    ) -> dict[str, Any]:
        """
        Traite une interaction complète avec capture mémoire

        Returns:
            Dict contenant memory_id et contexte enrichi
        """
        try:
            # Capture automatique
            memory_id = await self.memory.auto_capture_interaction(
                user_input=user_input, ai_response=ai_response, emotional_analysis=emotional_analysis, user_id=user_id
            )

            # Récupération du contexte récent
            recent_context = await self.memory.recall_recent_conversation(
                user_id=user_id,
                hours=2,  # Contexte des 2 dernières heures
            )

            return {
                "memory_id": memory_id,
                "recent_context": recent_context[-5:],  # 5 derniers échanges
                "emotional_context": emotional_analysis,
                "context_summary": self._generate_context_summary(recent_context),
            }

        except Exception as e:
            logger.error(f"Erreur process_interaction_with_memory: {e}")
            return {"memory_id": "", "recent_context": [], "context_summary": ""}

    def _generate_context_summary(self, recent_memories: list[dict]) -> str:
        """Génère un résumé du contexte récent"""
        if not recent_memories:
            return "Nouvelle conversation"

        # Extraire les thèmes principaux
        all_topics = []
        emotions = []

        for memory in recent_memories[-3:]:  # 3 dernières mémoires
            all_topics.extend(memory.get("topics", []))
            emotions.extend([e["emotion"] for e in memory.get("emotions", [])])

        # Résumé basique
        main_topics = list(set(all_topics))[:3]
        main_emotions = list(set(emotions))[:2]

        summary_parts = []
        if main_topics:
            summary_parts.append(f"Sujets: {', '.join(main_topics)}")
        if main_emotions:
            summary_parts.append(f"Émotions: {', '.join(main_emotions)}")

        return " | ".join(summary_parts) or "Conversation générale"


# Instance globale pour intégration facile
_memory_interface_instance: MemoryInterface | None = None
_agi_memory_integration_instance: AGIMemoryIntegration | None = None


def get_memory_interface() -> MemoryInterface:
    """Récupère l'instance globale de MemoryInterface"""
    global _memory_interface_instance
    if _memory_interface_instance is None:
        _memory_interface_instance = MemoryInterface()
    return _memory_interface_instance


def get_agi_memory_integration() -> AGIMemoryIntegration:
    """Récupère l'instance globale d'AGIMemoryIntegration"""
    global _agi_memory_integration_instance
    if _agi_memory_integration_instance is None:
        _agi_memory_integration_instance = AGIMemoryIntegration()
    return _agi_memory_integration_instance
