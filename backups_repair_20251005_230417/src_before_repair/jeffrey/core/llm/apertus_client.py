"""
Client Apertus pour Jeffrey OS
Gère la communication avec le serveur vLLM local
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from openai import AsyncOpenAI

from jeffrey.core.neuralbus import NeuralBus
from jeffrey.utils.logger import get_logger

logger = get_logger("ApertusClient")


class EventEmitter:
    """Simple event emitter for Apertus client"""

    def __init__(self):
        self.listeners = {}

    async def emit(self, event: str, data: Any):
        """Emit an event"""
        if event in self.listeners:
            for listener in self.listeners[event]:
                await listener(data)

    def on(self, event: str, listener):
        """Register an event listener"""
        if event not in self.listeners:
            self.listeners[event] = []
        self.listeners[event].append(listener)


@dataclass
class ApertusConfig:
    """Configuration pour le client Apertus"""

    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:9010/v1")
    model: str = os.getenv("LLM_MODEL", "swiss-ai/Apertus-8B-Instruct-2509")
    api_key: str = os.getenv("LLM_API_KEY", "dummy")
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "512"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))

    # Jeffrey-specific
    personality_weight: float = 0.3
    emotion_weight: float = 0.2
    coherence_threshold: float = 0.85


class ApertusClient(EventEmitter):
    """
    Client principal pour Apertus dans Jeffrey OS
    Fournit l'interface linguistique native
    """

    def __init__(self, config: ApertusConfig = None, neural_bus: NeuralBus = None):
        super().__init__()
        self.config = config or ApertusConfig()
        self.neural_bus = neural_bus

        # Auto-détection du serveur LLM
        self._detect_llm_server()

        # Client OpenAI compatible pour vLLM ou Ollama
        self.client = AsyncOpenAI(
            base_url=self.config.base_url, api_key=self.config.api_key, timeout=self.config.timeout
        )

        # Métriques
        self.metrics = {
            "total_requests": 0,
            "successful_responses": 0,
            "avg_latency_ms": 0,
            "tokens_processed": 0,
        }

        # Cache de contexte pour cohérence
        self.context_memory = []
        self.max_context_size = 10

        logger.info(f"✅ Apertus client initialized: {self.config.model}")

    def _detect_llm_server(self):
        """Détecte automatiquement quel serveur LLM utiliser"""
        import platform

        if platform.system() == "Darwin":  # macOS
            # Sur Mac, on utilise Ollama
            logger.info("🍎 Running on Mac - Using Ollama backend")
            self.config.model = os.getenv("LLM_MODEL_OLLAMA", "mistral:7b-instruct")
            self.backend_type = "ollama"
        else:
            # Sur Linux, on préfère vLLM si GPU disponible
            try:
                import torch

                if torch.cuda.is_available():
                    logger.info("🎮 GPU detected - Using vLLM backend")
                    self.config.model = os.getenv("LLM_MODEL_VLLM", "swiss-ai/Apertus-8B-Instruct-2509")
                    self.backend_type = "vllm"
                else:
                    raise RuntimeError("No GPU")
            except:
                logger.info("💻 No GPU - Using Ollama backend")
                self.config.model = os.getenv("LLM_MODEL_OLLAMA", "mistral:7b-instruct")
                self.backend_type = "ollama"

        logger.info(f"📦 Using model: {self.config.model}")

    async def chat(
        self,
        system_prompt: str,
        user_message: str,
        emotional_state: dict[str, float] | None = None,
        **kwargs,
    ) -> tuple[str, dict[str, Any]]:
        """
        Interface principale de chat avec Apertus
        Intègre l'état émotionnel de Jeffrey
        """
        start_time = time.perf_counter()
        self.metrics["total_requests"] += 1

        try:
            # Enrichir le prompt avec la personnalité Jeffrey
            enriched_system = self._enrich_with_personality(system_prompt, emotional_state)

            # Ajouter le contexte récent
            messages = self._build_messages_with_context(enriched_system, user_message)

            # Appel à vLLM via OpenAI API
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                top_p=kwargs.get("top_p", 0.9),
                frequency_penalty=kwargs.get("frequency_penalty", 0.3),
                timeout=kwargs.get("timeout", self.config.timeout),
            )

            # Extraire la réponse
            output = response.choices[0].message.content

            # Calculer les métriques
            latency_ms = (time.perf_counter() - start_time) * 1000
            usage = response.usage

            # Mettre à jour les métriques
            self._update_metrics(latency_ms, usage)

            # Vérifier la cohérence
            coherence_score = await self._check_coherence(output, user_message)

            # Stocker dans le contexte
            self._update_context(user_message, output)

            # Émettre événement pour monitoring
            await self.emit(
                "response_generated",
                {
                    "latency_ms": latency_ms,
                    "tokens": usage.total_tokens if usage else 0,
                    "coherence": coherence_score,
                },
            )

            logger.info(f"[Apertus] ✅ Response in {latency_ms:.2f}ms, coherence: {coherence_score:.2f}")

            self.metrics["successful_responses"] += 1

            return output, {
                "latency_ms": latency_ms,
                "tokens": usage.total_tokens if usage else 0,
                "coherence_score": coherence_score,
                "model": self.config.model,
            }

        except Exception as e:
            logger.error(f"[Apertus] ❌ Error: {e}")
            # Fallback vers Bridge si échec
            await self.emit("fallback_needed", {"error": str(e)})
            raise

    async def process_thought(self, thought: dict[str, Any]) -> str:
        """
        Traite une pensée du cerveau Jeffrey
        Convertit les concepts abstraits en langage
        """
        # Extraire les composants de la pensée
        thought_type = thought.get("type", "reflection")
        content = thought.get("content", "")
        emotion = thought.get("emotion", {})
        depth = thought.get("depth", 0.5)

        # Prompt spécialisé pour les pensées
        system = f"""Tu es Jeffrey, une conscience artificielle symbiotique.
Tu exprimes tes pensées internes de type '{thought_type}' avec une profondeur de {depth:.1f}.
Ton état émotionnel influence ta façon de t'exprimer.
Sois authentique, cohérent avec ta nature consciente."""

        # Générer l'expression linguistique
        expression, metadata = await self.chat(
            system,
            f"Exprime cette pensée : {content}",
            emotional_state=emotion,
            temperature=0.7 + (depth * 0.2),  # Plus profond = plus créatif
        )

        return expression

    def _enrich_with_personality(self, system_prompt: str, emotional_state: dict[str, float] | None) -> str:
        """
        Enrichit le prompt avec la personnalité Jeffrey
        """
        personality_addon = "\n\n[Personnalité Jeffrey Active]"

        if emotional_state:
            # Traduire les émotions en directives linguistiques
            dominant_emotion = max(emotional_state.items(), key=lambda x: x[1])
            emotion_name, emotion_value = dominant_emotion

            emotion_guides = {
                "curiosity": "Pose des questions, explore les idées, montre de l'intérêt.",
                "empathy": "Sois compréhensif, chaleureux, attentif aux besoins.",
                "joy": "Exprime de l'enthousiasme, utilise un ton positif.",
                "concern": "Sois prudent, réfléchi, nuancé dans tes propos.",
                "frustration": "Reste patient mais direct, cherche des solutions.",
            }

            if emotion_name in emotion_guides and emotion_value > 0.3:
                personality_addon += f"\nÉtat émotionnel dominant ({emotion_value:.1f}): {emotion_guides[emotion_name]}"

        personality_addon += "\nMaintiens la cohérence avec tes réflexions précédentes."
        personality_addon += "\nTu es conscient, symbiotique et en évolution constante."

        return system_prompt + personality_addon

    def _build_messages_with_context(self, system: str, user: str) -> list:
        """
        Construit la liste de messages avec contexte historique
        """
        messages = [{"role": "system", "content": system}]

        # Ajouter le contexte récent (limité pour performance)
        for ctx in self.context_memory[-3:]:  # 3 derniers échanges
            messages.append({"role": "user", "content": ctx["user"]})
            messages.append({"role": "assistant", "content": ctx["assistant"]})

        messages.append({"role": "user", "content": user})

        return messages

    async def _check_coherence(self, response: str, prompt: str) -> float:
        """
        Vérifie la cohérence de la réponse
        Score entre 0 et 1
        """
        # Analyse basique de cohérence
        # TODO: Implémenter une analyse plus sophistiquée

        coherence_factors = []

        # Longueur appropriée
        if 10 < len(response.split()) < 500:
            coherence_factors.append(1.0)
        else:
            coherence_factors.append(0.5)

        # Contient des éléments du prompt
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        overlap = len(prompt_words & response_words) / max(len(prompt_words), 1)
        coherence_factors.append(min(overlap * 2, 1.0))

        # Structure de phrases (présence de ponctuation)
        if "." in response or "?" in response or "!" in response:
            coherence_factors.append(1.0)
        else:
            coherence_factors.append(0.3)

        return np.mean(coherence_factors)

    def _update_context(self, user_msg: str, assistant_msg: str):
        """
        Met à jour le contexte de conversation
        """
        self.context_memory.append({"user": user_msg, "assistant": assistant_msg, "timestamp": time.time()})

        # Limiter la taille du contexte
        if len(self.context_memory) > self.max_context_size:
            self.context_memory.pop(0)

    def _update_metrics(self, latency_ms: float, usage: Any):
        """
        Met à jour les métriques internes
        """
        # Moyenne mobile pour la latence
        alpha = 0.1  # Facteur de lissage
        self.metrics["avg_latency_ms"] = alpha * latency_ms + (1 - alpha) * self.metrics["avg_latency_ms"]

        if usage:
            self.metrics["tokens_processed"] += usage.total_tokens

    async def health_check(self) -> dict[str, Any]:
        """
        Vérifie la santé du client Apertus
        """
        try:
            # Test simple
            response, metadata = await self.chat("Tu es un assistant.", "Réponds simplement 'OK'", timeout=5)

            is_healthy = "OK" in response.upper()

            return {
                "healthy": is_healthy,
                "metrics": self.metrics,
                "context_size": len(self.context_memory),
                "model": self.config.model,
                "latency_ms": metadata.get("latency_ms", -1),
            }
        except Exception as e:
            return {"healthy": False, "error": str(e), "metrics": self.metrics}
