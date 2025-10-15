# TODO: Précompiler les regex utilisées dans les boucles
# TODO: Précompiler les regex utilisées dans les boucles
# TODO: Précompiler les regex utilisées dans les boucles
"""
Orchestrateur principal du système cognitif.

Ce module implémente les fonctionnalités essentielles pour orchestrateur principal du système cognitif.
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
import re
from enum import Enum

from jeffrey.core.emotions.core.emotion_ml_enhancer import EmotionMLEnhancer
from jeffrey.core.orchestration.multi_model_orchestrator import MultiModelOrchestrator


class AICapability(Enum):
    """Capacités spécifiques des différentes IA"""

    CREATIVE = "creative"  # Génération créative, imagination (Grok)
    ANALYTICAL = "analytical"  # Analyse, explication, technique (ChatGPT)
    EMPATHETIC = "empathetic"  # Empathie, soutien émotionnel (Claude)
    TECHNICAL = "technical"  # Code, debug, solutions techniques
    EMOTIONAL = "emotional"  # Réponses émotionnelles (Jeffrey Core)
    NARRATIVE = "narrative"  # Storytelling, récits
    PHILOSOPHICAL = "philosophical"  # Questions existentielles
    HUMOROUS = "humorous"  # Humour, légèreté


class EnhancedOrchestrator(MultiModelOrchestrator):
    """
    Orchestrateur amélioré avec sélection dynamique basée sur les capacités.
    """

    # Mapping des providers vers leurs capacités
    PROVIDER_CAPABILITIES = {
        "grok": [
            AICapability.CREATIVE,
            AICapability.EMOTIONAL,
            AICapability.NARRATIVE,
            AICapability.HUMOROUS,
        ],
        "gpt": [AICapability.ANALYTICAL, AICapability.TECHNICAL, AICapability.PHILOSOPHICAL],
        "claude": [AICapability.EMPATHETIC, AICapability.ANALYTICAL, AICapability.EMOTIONAL],
        "chatgpt": [AICapability.ANALYTICAL, AICapability.TECHNICAL],  # Alias
        "gpt-4": [AICapability.ANALYTICAL, AICapability.TECHNICAL],  # Alias
    }

    # Patterns pour détecter le besoin de capacités
    CAPABILITY_PATTERNS = {
        AICapability.CREATIVE: [
            r"\bimagine\b",
            r"\binvente\b",
            r"\bcrée\b",
            r"\bstory\b",
            r"\bidée\b",
        ],
        AICapability.ANALYTICAL: [
            r"\bexplique\b",
            r"\banalyse\b",
            r"\bcomment\b",
            r"\bpourquoi\b",
            r"\braison\b",
        ],
        AICapability.EMPATHETIC: [
            r"\btriste\b",
            r"\bpeur\b",
            r"\binquiet\b",
            r"\bsoutien\b",
            r"\baide\b",
            r"\bmal\b",
        ],
        AICapability.TECHNICAL: [
            r"\bcode\b",
            r"\bbug\b",
            r"\berreur\b",
            r"\bprogramme\b",
            r"\balgorithme\b",
        ],
        AICapability.EMOTIONAL: [
            r"\bressens\b",
            r"\bémotion\b",
            r"\bsentiment\b",
            r"\bamour\b",
            r"\bcoeur\b",
        ],
        AICapability.NARRATIVE: [
            r"\braconte\b",
            r"\bhistoire\b",
            r"\bconte\b",
            r"\brécit\b",
            r"\baventure\b",
        ],
        AICapability.PHILOSOPHICAL: [
            r"\bsens.*vie\b",
            r"\bexistence\b",
            r"\bphilosoph\b",
            r"\bconscience\b",
            r"\bréalité\b",
        ],
        AICapability.HUMOROUS: [
            r"\bblague\b",
            r"\bdrôle\b",
            r"\brire\b",
            r"\bhumour\b",
            r"\bamusant\b",
        ],
    }

    def __init__(self, ia_clients: dict[str, object] = None) -> None:
        """Initialise l'orchestrateur amélioré."""
        super().__init__(ia_clients)
        self.emotion_enhancer = EmotionMLEnhancer()
        self.capability_scores = {}  # Cache des scores de capacités

    def detect_required_capabilities(self, prompt: str) -> set[AICapability]:
        """
        Détecte les capacités nécessaires pour répondre au prompt.

        Args:
            prompt: Le message utilisateur

        Returns:
            Set des capacités détectées
        """
        required = set()
        prompt_lower = prompt.lower()

        # Détection par patterns
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        for capability, patterns in self.CAPABILITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, prompt_lower):
                    required.add(capability)
                    break

        # Détection émotionnelle
        emotion_result = self.emotion_enhancer.detect_emotion_enhanced(prompt)
        if emotion_result["confidence"] > 0.5:
            if emotion_result["emotion"] in ["tristesse", "peur", "inquietude"]:
                required.add(AICapability.EMPATHETIC)
            elif emotion_result["emotion"] in ["joie", "amour"]:
                required.add(AICapability.EMOTIONAL)

        # Si aucune capacité détectée, utiliser ANALYTICAL par défaut
        if not required:
            required.add(AICapability.ANALYTICAL)

        return required

    def score_providers_by_capability(self, required_capabilities: set[AICapability]) -> dict[str, float]:
        """
        Score chaque provider selon sa correspondance avec les capacités requises.

        Args:
            required_capabilities: Capacités nécessaires

        Returns:
            Dict {provider: score}
        """
        scores = {}

        for provider, capabilities in self.PROVIDER_CAPABILITIES.items():
            score = 0.0
            provider_caps = set(capabilities)

            # Score basé sur le nombre de capacités correspondantes
            matching = provider_caps.intersection(required_capabilities)
            score = len(matching) / len(required_capabilities) if required_capabilities else 0

            # Bonus si toutes les capacités requises sont couvertes
            if matching == required_capabilities:
                score += 0.5

            # Normaliser entre 0 et 1
            scores[provider] = min(1.0, score)

        return scores

    async def send_to_best_models(self, prompt: str, max_models: int = 2) -> list[dict[str, str]]:
        """
        Envoie le prompt aux meilleurs modèles selon les capacités requises.

        Args:
            prompt: Le message utilisateur
            max_models: Nombre maximum de modèles à utiliser

        Returns:
            Liste des réponses
        """
        # Détecter les capacités requises
        required_caps = self.detect_required_capabilities(prompt)

        # Scorer les providers
        provider_scores = self.score_providers_by_capability(required_caps)

        # Sélectionner les meilleurs providers
        sorted_providers = sorted(provider_scores.items(), key=lambda x: x[1], reverse=True)
        selected_providers = [p[0] for p in sorted_providers[:max_models] if p[1] > 0]

        # Si aucun provider sélectionné, utiliser les défauts
        if not selected_providers:
            selected_providers = list(self.ia_clients.keys())[:max_models]

        # Log la sélection
        print(f"🎯 Capacités requises: {[c.value for c in required_caps]}")
        print(f"🤖 Providers sélectionnés: {selected_providers}")

        # Envoyer aux modèles sélectionnés
        tasks = []
        for provider in selected_providers:
            if provider in self.ia_clients or self._find_provider_alias(provider):
                tasks.append(self.send_to_model(self._find_provider_alias(provider) or provider, prompt))

        results = await asyncio.gather(*tasks)
        return results

    def _find_provider_alias(self, provider: str) -> str | None:
        """Trouve le nom réel du provider dans ia_clients."""
        # Mapping des alias
        aliases = {
            "gpt": ["gpt-4", "chatgpt", "openai"],
            "claude": ["claude-3", "anthropic"],
            "grok": ["grok", "x-ai"],
        }

        for alias_group, names in aliases.items():
            if provider in names:
                # Chercher dans ia_clients
                for client_name in self.ia_clients:
                    if any(name in client_name.lower() for name in names):
                        return client_name

        return None

    def ask_smart(self, prompt: str) -> str:
        """
        Version intelligente qui sélectionne dynamiquement les meilleurs modèles.

        Args:
            prompt: Le message utilisateur

        Returns:
            Réponse fusionnée des meilleurs modèles
        """
        # Obtenir les réponses des meilleurs modèles
        raw_responses = asyncio.run(self.send_to_best_models(prompt))

        # Fusionner les réponses
        from jeffrey.core.orchestration.fusion_engine import fuse_responses

        fused_response = fuse_responses(raw_responses)

        # Sauvegarder dans la mémoire
        self.memory.add_interaction(prompt, raw_responses, fused_response)

        return fused_response

    def get_capability_analysis(self, prompt: str) -> dict[str, any]:
        """
        Analyse détaillée des capacités pour un prompt.

        Args:
            prompt: Le message à analyser

        Returns:
            Dict avec l'analyse complète
        """
        required_caps = self.detect_required_capabilities(prompt)
        provider_scores = self.score_providers_by_capability(required_caps)
        emotion_result = self.emotion_enhancer.detect_emotion_enhanced(prompt)

        return {
            "required_capabilities": [c.value for c in required_caps],
            "provider_scores": provider_scores,
            "emotional_context": {
                "emotion": emotion_result["emotion"],
                "confidence": emotion_result["confidence"],
                "intensity": emotion_result.get("intensity", {}),
            },
            "recommended_providers": sorted(provider_scores.items(), key=lambda x: x[1], reverse=True)[:2],
        }
