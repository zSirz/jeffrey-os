"""
Système linguistique autonome utilisant les modules d'apprentissage existants
Version V2 avec Executive Cortex, Triple Memory et Quality Critic
"""

import hashlib
import time
from typing import Any

from jeffrey.core.brain.executive_cortex import ExecutiveCortex
from jeffrey.core.brain.quality_critic import QualityCritic
from jeffrey.core.learning.auto_learner import AutoLearner
from jeffrey.core.learning.contextual_learning_engine import ContextualLearningEngine
from jeffrey.core.learning.jeffrey_meta_learning_integration import MetaLearningIntegration
from jeffrey.core.learning.theory_of_mind import TheoryOfMind
from jeffrey.core.learning.unified_curiosity_engine import UnifiedCuriosityEngine
from jeffrey.core.memory.triple_memory import LocalEmbedder, TripleMemorySystem
from jeffrey.core.memory.unified_memory import UnifiedMemory
from jeffrey.utils.logger import get_logger

logger = get_logger("AutonomousLanguage")


class AutonomousLanguageSystem:
    """
    Système linguistique autonome avec apprentissage profond
    """

    def __init__(self, fallback_llm=None):
        # Modules d'apprentissage existants
        self.meta_learner = MetaLearningIntegration()
        self.theory_of_mind = TheoryOfMind()
        self.curiosity = UnifiedCuriosityEngine()
        self.auto_learner = AutoLearner()
        self.context_engine = ContextualLearningEngine()

        # Mémoire unifiée existante
        self.memory = UnifiedMemory()

        # Nouveaux composants cerveau V2
        self.executive_cortex = ExecutiveCortex()
        self.triple_memory = TripleMemorySystem()
        self.quality_critic = QualityCritic(self.theory_of_mind)

        # Embedder local pour offline
        self.embedder = LocalEmbedder(dim=384)

        # LLM externe optionnel
        self.fallback_llm = fallback_llm

        # Phase d'apprentissage (0: dépendant, 1: autonome)
        self.autonomy_level = 0.0

        # Cache simple
        self.response_cache = {}
        self.cache_max_size = 1000

        logger.info("🧠 Autonomous Language System V2 initialized")

    async def initialize(self):
        """Initialise le système et démarre les loops"""
        await self.triple_memory.start_consolidation_loop()
        logger.info("✨ Memory consolidation loop started")

    async def shutdown(self):
        """Arrête proprement le système"""
        await self.triple_memory.shutdown()

    async def process(
        self, query: dict[str, Any], force_autonomous: bool = False, force_external: bool = False
    ) -> dict[str, Any]:
        """
        Version V2 avec Executive Cortex et Triple Memory
        """
        start_time = time.time()

        # 1. Préparer le contexte enrichi
        context = await self._prepare_context(query)

        # 2. Décision par Executive Cortex
        arm, decision_metadata = await self.executive_cortex.decide(context)

        logger.info(f"🎯 Executive decision: {arm} (context: {decision_metadata['context']})")

        # 3. Exécuter selon le bras choisi
        if arm == "cache":
            result = await self._try_cache(query, context)
        elif arm == "autonomous":
            result = await self._generate_autonomous_v2(query, context)
        else:  # llm
            result = await self._use_external_llm(query, context)

        # 4. Évaluation qualité
        if result["success"]:
            validation = await self.quality_critic.evaluate(result["response"], context)

            result["quality_score"] = validation.overall_quality
            result["quality_report"] = validation.to_dict()

            # 5. Reward au bandit
            latency_ms = (time.time() - start_time) * 1000
            await self.executive_cortex.reward(arm, context, validation.overall_quality, latency_ms, result["success"])

            # 6. Mémorisation si bonne qualité
            if validation.overall_quality > 0.6:
                await self.triple_memory.remember(
                    text_in=query["content"],
                    text_out=result["response"],
                    intent=context.get("intent_type", "general"),
                    quality_score=validation.overall_quality,
                    embedding_in=context.get("embedding"),
                    metadata={
                        "arm": arm,
                        "routing": result.get("routing", arm),
                        "decision": decision_metadata,
                    },
                )

                # Cache pour accès rapide
                self._add_to_cache(query["content"], result)

        return result

    async def _prepare_context(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        Prépare un contexte enrichi pour la décision
        """
        # Analyse contextuelle de base
        base = await self.context_engine.analyze({"text": query["content"], "type": query.get("type", "general")})

        # Inférence d'intention
        intention = await self.theory_of_mind.infer_intention(query["content"], base)

        # Évaluation de familiarité
        familiarity = await self.auto_learner.assess_familiarity(base)

        # Embedding pour recherche vectorielle
        embedding = self.embedder.encode(query["content"])

        return {
            "input": query["content"],
            "intent_type": intention.get("type", "general"),
            "main_concept": intention.get("main_concept", ""),
            "complexity": float(intention.get("complexity", 0.5)),
            "familiarity": float(familiarity),
            "domain": base.get("domain", "general"),
            "emotional_state": query.get("emotional_state", {}),
            "intention": intention,
            "embedding": embedding,
        }

    async def _try_cache(self, query: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Tente de récupérer depuis le cache
        """
        cache_key = self._get_cache_key(query["content"])

        if cache_key in self.response_cache:
            cached = self.response_cache[cache_key]
            logger.info("💾 Cache hit!")
            return {
                "success": True,
                "response": cached["response"],
                "model": "cache",
                "routing": "cache_hit",
                "metadata": {
                    "cached_quality": cached.get("quality_score", 0),
                    "cache_age": time.time() - cached.get("timestamp", 0),
                },
            }

        # Cache miss - fallback
        logger.debug("Cache miss, generating new response")
        return await self._generate_autonomous_v2(query, context)

    async def _generate_autonomous_v2(self, query: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Pipeline RAG → Sketch → Compose → Revise
        """
        try:
            # 1. RAG : Récupérer épisodes et patterns similaires
            similar_episodes = await self.triple_memory.recall_similar(
                query["content"], embedding=context.get("embedding"), k=5
            )

            patterns = await self.triple_memory.get_patterns(context.get("intent_type"))

            # 2. Sketch : Plan de réponse
            sketch = await self._create_sketch(query, similar_episodes, patterns, context)

            # 3. Compose : Générer la réponse
            response = await self._compose_response(sketch, context, query.get("emotional_state", {}))

            # 4. Revise : Améliorer si nécessaire
            revision_count = 0
            for revision in range(2):  # Max 2 révisions
                validation = await self.quality_critic.evaluate(response, context)

                if validation.overall_quality >= 0.7:
                    break

                response = await self._revise_response(response, validation, context)
                revision_count += 1

            return {
                "success": True,
                "response": response,
                "model": "jeffrey-autonomous-v2",
                "routing": "rag_pipeline",
                "metadata": {
                    "episodes_used": len(similar_episodes),
                    "patterns_used": len(patterns),
                    "revisions": revision_count,
                },
            }

        except Exception as e:
            logger.error(f"Autonomous V2 generation error: {e}")
            return {"success": False, "reason": str(e)}

    async def _use_external_llm(self, query: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Utilise le LLM externe si disponible
        """
        if not self.fallback_llm:
            # Pas de LLM externe, utiliser fallback simple
            return self._generate_fallback(query["content"], context)

        try:
            result = await self.fallback_llm.process(query, force_local=True)  # Mode offline

            if result.get("success"):
                # Apprendre de la réponse externe
                await self.learn_from_external(query["content"], result["response"], context)

            return result

        except Exception as e:
            logger.error(f"External LLM error: {e}")
            return self._generate_fallback(query["content"], context)

    async def _create_sketch(
        self,
        query: dict[str, Any],
        episodes: list[Any],
        patterns: list[dict],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Crée un plan de réponse basé sur les données récupérées
        """
        # Extraire les éléments clés des épisodes
        key_elements = []
        for episode in episodes[:3]:  # Top 3 plus pertinents
            if hasattr(episode, "text_out"):
                key_elements.append({"concept": episode.text_out[:50], "quality": episode.quality_score})

        # Identifier le pattern dominant
        best_pattern = None
        if patterns:
            best_pattern = max(patterns, key=lambda p: p.get("quality_mean", 0))

        return {
            "opening": self._select_opening(context["intent_type"]),
            "key_elements": key_elements,
            "pattern": best_pattern,
            "emotional_tone": context.get("emotional_state", {}),
            "structure": self._determine_structure(context),
        }

    def _select_opening(self, intent_type: str) -> str:
        """Sélectionne une ouverture appropriée"""
        openings = {
            "question": "I find myself contemplating",
            "exploration": "My curiosity leads me to observe",
            "reflection": "In my experience",
            "general": "I notice",
        }
        return openings.get(intent_type, "I observe")

    def _determine_structure(self, context: dict[str, Any]) -> str:
        """Détermine la structure de la réponse"""
        complexity = context.get("complexity", 0.5)
        if complexity < 0.3:
            return "simple"
        elif complexity < 0.7:
            return "compound"
        else:
            return "complex"

    async def _compose_response(
        self, sketch: dict[str, Any], context: dict[str, Any], emotional_state: dict[str, float]
    ) -> str:
        """
        Compose une réponse à partir du sketch
        """
        # Début
        response_parts = [sketch["opening"]]

        # Corps basé sur les éléments clés
        if sketch["key_elements"]:
            for element in sketch["key_elements"][:2]:
                response_parts.append(element["concept"])

        # Pattern si disponible
        if sketch["pattern"] and sketch["pattern"].get("examples"):
            example = sketch["pattern"]["examples"][0]
            response_parts.append(example.get("output", "")[:100])

        # Ajout émotionnel
        if emotional_state:
            dominant = max(emotional_state.items(), key=lambda x: x[1])[0] if emotional_state else None
            endings = {
                "curiosity": "and find myself wondering about the deeper implications.",
                "joy": "which brings a sense of satisfaction.",
                "concern": "though I remain thoughtful about the implications.",
                "empathy": "and I appreciate the complexity involved.",
            }
            response_parts.append(endings.get(dominant, "with interest."))

        # Assembler
        response = " ".join(filter(None, response_parts))

        # Nettoyer et ajuster
        response = self._clean_response(response)

        return response

    async def _revise_response(self, response: str, validation: Any, context: dict[str, Any]) -> str:
        """
        Révise une réponse basée sur le feedback du critic
        """
        # Appliquer les suggestions
        revised = response

        for issue in validation.issues:
            if "too short" in issue.lower():
                revised += " Let me elaborate on this fascinating topic."
            elif "missing jeffrey" in issue.lower():
                revised = "I wonder, " + revised.lower()
            elif "too formal" in issue.lower():
                revised = revised.replace("therefore", "so")
                revised = revised.replace("furthermore", "also")

        return revised

    def _clean_response(self, response: str) -> str:
        """Nettoie et formate la réponse"""
        # Supprimer espaces multiples
        response = " ".join(response.split())

        # Capitaliser première lettre
        if response:
            response = response[0].upper() + response[1:]

        # S'assurer que ça finit par ponctuation
        if response and response[-1] not in ".!?":
            response += "."

        return response

    def _generate_fallback(self, input_text: str, context: dict[str, Any]) -> dict[str, Any]:
        """Génère une réponse de fallback basique"""
        intent_type = context.get("intent_type", "general")

        fallbacks = {
            "question": "That's an interesting question. Based on my understanding, the patterns suggest multiple perspectives worth exploring.",
            "exploration": "My exploration of this concept reveals fascinating connections I hadn't previously considered.",
            "reflection": "Reflecting on this, I observe patterns that resonate with deeper systemic principles.",
            "general": "I find this topic intriguing and notice several interconnected aspects worth considering.",
        }

        return {
            "success": True,
            "response": fallbacks.get(intent_type, fallbacks["general"]),
            "model": "fallback",
            "routing": "fallback",
            "metadata": {"reason": "no_external_llm"},
        }

    def _get_cache_key(self, text: str) -> str:
        """Génère une clé de cache pour un texte"""
        return hashlib.md5(text.lower().strip().encode()).hexdigest()

    def _add_to_cache(self, query: str, result: dict[str, Any]):
        """Ajoute au cache avec limite de taille"""
        if len(self.response_cache) >= self.cache_max_size:
            # Supprimer les plus anciens (FIFO simple)
            oldest = min(self.response_cache.items(), key=lambda x: x[1].get("timestamp", 0))
            del self.response_cache[oldest[0]]

        cache_key = self._get_cache_key(query)
        self.response_cache[cache_key] = {
            "response": result["response"],
            "quality_score": result.get("quality_score", 0),
            "timestamp": time.time(),
        }

    async def learn_from_external(self, input_text: str, response: str, context: dict[str, Any]):
        """
        Apprend d'une interaction avec un LLM externe
        """
        # Extraire des patterns
        patterns = await self.meta_learner.extract_patterns(input_text)

        # Mettre à jour l'autonomie
        self.autonomy_level = min(1.0, self.autonomy_level + 0.001)

        logger.debug(f"Learning from external. Autonomy: {self.autonomy_level:.3f}")

    async def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques du système"""
        return {
            "autonomy_level": self.autonomy_level,
            "executive_stats": self.executive_cortex.get_stats(),
            "memory_stats": self.triple_memory.get_stats(),
            "cache_size": len(self.response_cache),
        }
