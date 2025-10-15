"""
Interface simple pour Ollama/Mistral
Module réel pour Bundle 1
"""

import logging
from typing import Any

from jeffrey.core.llm.apertus_client import ApertusClient

logger = logging.getLogger(__name__)


class OllamaInterface:
    """Interface pour Ollama avec Mistral"""

    def __init__(self):
        self.client = None
        self.initialized = False

    async def initialize(self, config: dict[str, Any]):
        """Initialise le client Ollama"""
        try:
            self.client = ApertusClient()
            self.initialized = True
            logger.info("✅ Ollama interface initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            self.initialized = False

    async def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        """Génère une réponse avec Ollama"""
        if not self.initialized:
            await self.initialize({})

        input_text = context.get("input", "")
        emotion = context.get("emotion", "neutral")
        memories = context.get("memories", [])

        # Construire le prompt système
        system_prompt = "Tu es Jeffrey, un assistant IA symbiotique créé en Suisse. "

        # Ajouter contexte émotionnel
        if emotion != "neutral":
            system_prompt += f"Tu ressens actuellement: {emotion}. "

        # Ajouter mémoires
        if memories:
            # Extraire le texte des mémoires si ce sont des dicts
            memory_texts = []
            for m in memories[:3]:
                if isinstance(m, dict):
                    memory_texts.append(m.get("text", str(m)))
                else:
                    memory_texts.append(str(m))
            if memory_texts:
                system_prompt += f"Tu te souviens: {', '.join(memory_texts)}. "

        try:
            # Appeler Ollama via ApertusClient
            response, metadata = await self.client.chat(system_prompt, input_text)

            context["response"] = response
            context["llm_metadata"] = metadata

            logger.info(f"Generated response in {metadata['latency_ms']:.0f}ms")

        except Exception as e:
            logger.error(f"Generation error: {e}")
            context["response"] = "Je réfléchis à ta question..."

        return context

    async def shutdown(self):
        """Arrêt propre"""
        self.initialized = False
        logger.info("Ollama interface shutdown")
