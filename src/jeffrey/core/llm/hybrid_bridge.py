"""
Bridge Orchestrateur Hybride
Décide entre Apertus local et IA externes
"""

import asyncio
import os
from enum import Enum
from typing import Any

from jeffrey.core.llm.apertus_client import ApertusClient
from jeffrey.utils.logger import get_logger

logger = get_logger("HybridBridge")


class ComplexityLevel(Enum):
    """Niveaux de complexité pour le routing"""

    SIMPLE = 1  # Apertus local
    MODERATE = 2  # Apertus + validation
    COMPLEX = 3  # Bridge externe
    CRITICAL = 4  # Multi-model consensus


class HybridOrchestrator:
    """
    Orchestrateur intelligent entre Apertus et IA externes
    Implémente l'architecture System 1 / System 2
    """

    def __init__(self, apertus_client: ApertusClient, ai_bridge: Any | None = None):
        self.apertus = apertus_client  # System 1 - Rapide, local
        self.ai_bridge = ai_bridge  # System 2 - Lent, puissant

        self.complexity_threshold = float(os.getenv("BRIDGE_COMPLEXITY_THRESHOLD", "0.3"))
        self.fallback_enabled = os.getenv("BRIDGE_ENABLED", "true").lower() == "true"

        # Statistiques de routing
        self.routing_stats = {"local": 0, "external": 0, "hybrid": 0, "failures": 0}

        logger.info(f"✅ Hybrid Orchestrator initialized (threshold: {self.complexity_threshold})")

    async def process(self, query: dict[str, Any], force_local: bool = False) -> dict[str, Any]:
        """
        Route intelligemment les requêtes selon leur complexité
        """
        # Analyser la complexité
        complexity = await self._analyze_complexity(query)
        query["complexity_score"] = complexity.value / 4.0

        logger.info(f"📊 Query complexity: {complexity.name} ({query['complexity_score']:.2f})")

        # Forcer local si demandé ou si Bridge désactivé
        if force_local or not self.fallback_enabled or not self.ai_bridge:
            return await self._process_local(query)

        # Routing selon complexité
        if complexity == ComplexityLevel.SIMPLE:
            # Simple -> Apertus seul
            self.routing_stats["local"] += 1
            return await self._process_local(query)

        elif complexity == ComplexityLevel.MODERATE:
            # Modéré -> Apertus avec validation possible
            self.routing_stats["hybrid"] += 1
            return await self._process_hybrid(query)

        elif complexity == ComplexityLevel.COMPLEX:
            # Complexe -> Bridge externe
            self.routing_stats["external"] += 1
            return await self._process_external(query)

        else:  # CRITICAL
            # Critique -> Consensus multi-model
            self.routing_stats["external"] += 1
            return await self._process_consensus(query)

    async def _analyze_complexity(self, query: dict[str, Any]) -> ComplexityLevel:
        """
        Analyse la complexité d'une requête
        """
        text = query.get("content", "")
        task_type = query.get("type", "general")

        # Indicateurs de complexité
        indicators = {
            "length": len(text.split()),
            "has_code": "def " in text or "class " in text or "function" in text,
            "has_math": any(op in text for op in ["equation", "calculate", "solve", "∫", "∑"]),
            "multi_step": any(word in text.lower() for word in ["étapes", "steps", "plan", "stratégie"]),
            "creative": any(word in text.lower() for word in ["histoire", "poème", "créatif", "imagine"]),
            "technical": any(word in text.lower() for word in ["debug", "optimize", "architecture", "algorithm"]),
            "requires_current": any(word in text.lower() for word in ["aujourd'hui", "actuel", "récent", "news"]),
        }

        # Calculer le score de complexité
        complexity_score = 0.0

        if indicators["length"] > 100:
            complexity_score += 0.2
        if indicators["has_code"]:
            complexity_score += 0.4
        if indicators["has_math"]:
            complexity_score += 0.3
        if indicators["multi_step"]:
            complexity_score += 0.3
        if indicators["creative"] and indicators["length"] > 50:
            complexity_score += 0.3
        if indicators["technical"]:
            complexity_score += 0.4
        if indicators["requires_current"]:
            complexity_score += 0.5  # Nécessite infos externes

        # Mapper vers niveau
        if complexity_score < 0.3:
            return ComplexityLevel.SIMPLE
        elif complexity_score < 0.6:
            return ComplexityLevel.MODERATE
        elif complexity_score < 1.0:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.CRITICAL

    async def _process_local(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        Traite localement avec Apertus
        """
        try:
            system = "Tu es Jeffrey, assistant IA symbiotique. Réponds de manière claire et utile."
            user_message = query.get("content", "")
            emotional_state = query.get("emotional_state")

            response, metadata = await self.apertus.chat(system, user_message, emotional_state=emotional_state)

            return {
                "success": True,
                "response": response,
                "model": "apertus",
                "routing": "local",
                "metadata": metadata,
            }

        except Exception as e:
            logger.error(f"❌ Local processing failed: {e}")
            self.routing_stats["failures"] += 1

            # Fallback vers externe si possible
            if self.ai_bridge and self.fallback_enabled:
                logger.info("🔄 Falling back to external model")
                return await self._process_external(query)
            else:
                return {"success": False, "error": str(e), "routing": "local_failed"}

    async def _process_hybrid(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        Traite avec Apertus puis valide si nécessaire
        """
        # D'abord essayer local
        local_result = await self._process_local(query)

        if not local_result["success"]:
            return await self._process_external(query)

        # Vérifier si validation nécessaire
        coherence = local_result["metadata"].get("coherence_score", 1.0)

        if coherence < 0.7:  # Cohérence faible
            logger.info(f"⚠️  Low coherence ({coherence:.2f}), validating with external model")

            # Valider avec modèle externe
            validation_query = {
                "content": f"Améliore cette réponse si nécessaire: {local_result['response']}",
                "original_query": query["content"],
            }

            external_result = await self._process_external(validation_query)

            if external_result["success"]:
                return {
                    "success": True,
                    "response": external_result["response"],
                    "model": f"apertus+{external_result.get('model', 'external')}",
                    "routing": "hybrid_validated",
                    "metadata": {**local_result["metadata"], "validated": True},
                }

        return local_result

    async def _process_external(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        Traite via Bridge externe
        """
        if not self.ai_bridge:
            logger.error("❌ No external bridge available")
            return {
                "success": False,
                "error": "External bridge not configured",
                "routing": "external_unavailable",
            }

        try:
            # Déterminer le meilleur modèle selon le type
            model_selection = self._select_best_model(query)

            # Appeler le Bridge
            result = await self.ai_bridge.process(
                query["content"], model=model_selection, context=query.get("context", {})
            )

            return {
                "success": True,
                "response": result.get("response", ""),
                "model": model_selection,
                "routing": "external",
                "metadata": result.get("metadata", {}),
            }

        except Exception as e:
            logger.error(f"❌ External processing failed: {e}")
            self.routing_stats["failures"] += 1
            return {"success": False, "error": str(e), "routing": "external_failed"}

    async def _process_consensus(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        Obtient un consensus de plusieurs modèles pour requêtes critiques
        """
        logger.info("🎯 Seeking multi-model consensus")

        models = ["apertus"]
        # Only add external models if bridge is available
        if self.ai_bridge:
            models.extend(["claude", "gpt"])

        responses = []

        # Collecter les réponses en parallèle
        tasks = []
        for model in models:
            if model == "apertus":
                tasks.append(self._process_local(query))
            else:
                model_query = {**query, "preferred_model": model}
                tasks.append(self._process_external(model_query))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Analyser les réponses
        valid_responses = []
        for i, result in enumerate(results):
            if isinstance(result, dict) and result.get("success"):
                valid_responses.append({"model": models[i], "response": result["response"]})

        if not valid_responses:
            return {
                "success": False,
                "error": "No valid responses from any model",
                "routing": "consensus_failed",
            }

        # Synthétiser le consensus
        consensus = await self._synthesize_consensus(valid_responses)

        return {
            "success": True,
            "response": consensus,
            "model": "consensus",
            "routing": "multi_model",
            "metadata": {
                "models_consulted": [r["model"] for r in valid_responses],
                "response_count": len(valid_responses),
            },
        }

    def _select_best_model(self, query: dict[str, Any]) -> str:
        """
        Sélectionne le meilleur modèle externe selon le type de requête
        """
        content = query.get("content", "").lower()

        # Heuristiques simples
        if any(word in content for word in ["code", "debug", "programme", "function"]):
            return "gpt"  # GPT pour code
        elif any(word in content for word in ["écris", "histoire", "créatif", "poème"]):
            return "claude"  # Claude pour créativité
        elif any(word in content for word in ["actualité", "récent", "news"]):
            return "grok"  # Grok pour actualités
        else:
            return "claude"  # Default

    async def _synthesize_consensus(self, responses: list) -> str:
        """
        Synthétise un consensus à partir de plusieurs réponses
        """
        # Pour l'instant, simple agrégation
        # TODO: Implémenter une vraie synthèse intelligente

        synthesis = "Synthèse basée sur plusieurs perspectives:\n\n"

        for resp in responses:
            synthesis += f"**{resp['model'].upper()}**: {resp['response'][:200]}...\n\n"

        return synthesis

    def get_stats(self) -> dict[str, Any]:
        """
        Retourne les statistiques de routing
        """
        total = sum(self.routing_stats.values())

        if total == 0:
            return self.routing_stats

        return {
            **self.routing_stats,
            "percentages": {
                "local": self.routing_stats["local"] / total * 100,
                "external": self.routing_stats["external"] / total * 100,
                "hybrid": self.routing_stats["hybrid"] / total * 100,
                "failures": self.routing_stats["failures"] / total * 100,
            },
        }
