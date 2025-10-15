"""
Adaptateur LLM Unifié - Utilise les clients existants
Connexion simple et robuste pour Jeffrey AGI
"""

import os

from jeffrey.core.llm.apertus_client import ApertusClient
from jeffrey.core.llm.ollama_interface import OllamaInterface


class LLMError(Exception):
    """Erreur LLM"""

    pass


class LLMProvider:
    """
    Provider unifié qui utilise les clients existants.

    Config via env :
    - JEFFREY_LLM_PROVIDER: "ollama" ou "apertus" (défaut: apertus)
    - LLM_BASE_URL: URL du serveur LLM
    - LLM_MODEL: Modèle à utiliser
    """

    def __init__(self):
        self.provider = os.getenv("JEFFREY_LLM_PROVIDER", "apertus").lower()
        self.client = None
        self.initialized = False

    async def initialize(self):
        """Initialise le client LLM"""
        try:
            if self.provider == "ollama":
                self.client = OllamaInterface()
                await self.client.initialize({})
                self.initialized = self.client.initialized
            elif self.provider == "apertus":
                self.client = ApertusClient()
                # Le client Apertus s'initialise tout seul
                self.initialized = True
            else:
                raise ValueError(f"Provider inconnu: {self.provider}")

            if self.initialized:
                print(f"✅ LLM {self.provider} initialisé")
            else:
                print(f"⚠️  Échec initialisation LLM {self.provider}")

        except Exception as e:
            print(f"⚠️  Erreur LLM: {e}")
            self.initialized = False

    async def health_check(self) -> bool:
        """Vérifie que le LLM est accessible"""
        if not self.initialized:
            await self.initialize()
        return self.initialized

    async def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
        """Génère une réponse"""
        if not self.initialized:
            await self.initialize()

        if not self.initialized:
            raise LLMError("LLM non initialisé")

        try:
            if self.provider == "ollama":
                # Interface Ollama existante
                context = {"input": prompt, "emotion": "neutral", "memories": []}
                result = await self.client.generate(context)
                return result.get("response", "Erreur de génération")

            else:  # apertus
                # Client Apertus existant
                system_prompt = "Tu es Jeffrey, un assistant IA créé en Suisse."
                response, metadata = await self.client.chat(system_prompt, prompt)
                return response

        except Exception as e:
            raise LLMError(f"Erreur {self.provider}: {e}") from e

    async def chat_simple(self, user_input: str) -> str:
        """Chat simple pour tests rapides"""
        return await self.generate(user_input, max_tokens=500)


# Instance globale
_llm_instance: LLMProvider | None = None


def get_llm() -> LLMProvider:
    """Récupère l'instance globale"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMProvider()
    return _llm_instance
