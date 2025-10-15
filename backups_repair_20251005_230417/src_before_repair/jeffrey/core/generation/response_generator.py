"""
Générateur de réponses simple pour Bundle 1
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Générateur de réponses basique"""

    def __init__(self):
        self.templates = {
            "greeting": [
                "Bonjour! Comment puis-je t'aider aujourd'hui?",
                "Salut! Ravi de te voir!",
                "Hello! Qu'est-ce qui t'amène?",
            ],
            "farewell": ["Au revoir! À bientôt!", "Bonne journée!", "À la prochaine!"],
            "question": [
                "Intéressante question, laisse-moi réfléchir...",
                "Hmm, voici ce que je pense:",
                "D'après ce que je sais:",
            ],
            "default": ["Je suis là pour t'aider.", "Dis-moi en plus.", "Continue, je t'écoute."],
        }

    def initialize(self, config: dict[str, Any]):
        """Initialise le générateur"""
        logger.info("✅ Response generator initialized")

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Traite le contexte et génère une réponse si nécessaire"""

        # Si une réponse existe déjà (de Ollama), la conserver
        if context.get("response"):
            return context

        input_text = context.get("input", "").lower()
        emotion = context.get("emotion", "neutral")

        # Détection simple du type de message
        if any(word in input_text for word in ["bonjour", "salut", "hello", "coucou"]):
            template = "greeting"
        elif any(word in input_text for word in ["revoir", "bye", "ciao", "bonne"]):
            template = "farewell"
        elif "?" in input_text:
            template = "question"
        else:
            template = "default"

        # Sélectionner une réponse
        import random

        response = random.choice(self.templates[template])

        # Ajouter contexte émotionnel
        if emotion == "happy":
            response = "😊 " + response
        elif emotion == "sad":
            response = "😔 " + response
        elif emotion == "angry":
            response = "😠 " + response

        context["response"] = response
        context["response_type"] = "template"

        return context

    def shutdown(self):
        """Arrêt propre"""
        logger.info("Response generator shutdown")
