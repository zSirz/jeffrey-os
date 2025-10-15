"""
Bundle 2: 8ème Région Cérébrale - Broca/Wernicke
Aires du langage: compréhension et production
"""

from datetime import datetime


class BrocaWernickeRegion:
    """Région 8: Aires du langage (Broca + Wernicke)"""

    def __init__(self):
        self.name = "Broca-Wernicke Complex"
        self.active = True
        self.wernicke = WernickeArea()  # Compréhension
        self.broca = BrocaArea()  # Production
        self.stats = {
            "sentences_understood": 0,
            "sentences_generated": 0,
            "active_since": datetime.now().isoformat(),
        }

    def process(self, input_text: str, context: dict = None) -> dict:
        """Pipeline complet: comprendre puis générer"""

        # 1. Wernicke: Comprendre
        understanding = self.wernicke.understand(input_text)
        self.stats["sentences_understood"] += 1

        # 2. Broca: Générer une réponse
        response = self.broca.generate(understanding, context)
        self.stats["sentences_generated"] += 1

        return {
            "understanding": understanding,
            "response": response,
            "region": "broca_wernicke",
            "active": True,
        }

    def health_check(self) -> dict:
        """Health check de la région"""
        try:
            # Test basique
            test = self.process("Hello Jeffrey")
            return {
                "status": "healthy",
                "region": "broca_wernicke",
                "areas": ["broca", "wernicke"],
                "stats": self.stats,
            }
        except Exception as e:
            return {"status": "unhealthy", "region": "broca_wernicke", "error": str(e)}


class WernickeArea:
    """Aire de Wernicke: Compréhension du langage"""

    def understand(self, text: str) -> dict:
        """Analyser et comprendre le texte"""

        # Analyse basique (sans NLP lourd)
        words = text.lower().split()

        # Détection d'intention simple
        intent = "unknown"
        if any(w in words for w in ["bonjour", "salut", "hello", "hi"]):
            intent = "greeting"
        elif any(w in words for w in ["comment", "quoi", "pourquoi", "what", "how", "why"]):
            intent = "question"
        elif any(w in words for w in ["merci", "thanks", "thank"]):
            intent = "gratitude"
        elif any(w in words for w in ["aide", "help", "aidez"]):
            intent = "help_request"

        # Extraction d'entités simples
        entities = []
        if "jeffrey" in text.lower():
            entities.append({"type": "name", "value": "Jeffrey"})
        if "bundle" in text.lower():
            entities.append({"type": "concept", "value": "bundle"})

        # Analyse de sentiment basique
        sentiment = "neutral"
        positive_words = ["bien", "super", "excellent", "good", "great", "love"]
        negative_words = ["mal", "mauvais", "problème", "bad", "wrong", "hate"]

        if any(w in words for w in positive_words):
            sentiment = "positive"
        elif any(w in words for w in negative_words):
            sentiment = "negative"

        return {
            "text": text,
            "words": len(words),
            "intent": intent,
            "entities": entities,
            "sentiment": sentiment,
            "language": "fr" if any(w in words for w in ["bonjour", "comment", "merci"]) else "en",
        }


class BrocaArea:
    """Aire de Broca: Production du langage"""

    def generate(self, understanding: dict, context: dict = None) -> str:
        """Générer une réponse basée sur la compréhension"""

        intent = understanding.get("intent", "unknown")
        sentiment = understanding.get("sentiment", "neutral")
        language = understanding.get("language", "en")

        # Templates de réponse selon l'intention
        if intent == "greeting":
            responses = {
                "fr": [
                    "Bonjour! Jeffrey OS Bundle 2 est actif avec 8/8 régions.",
                    "Salut! Comment puis-je aider?",
                ],
                "en": [
                    "Hello! Jeffrey OS Bundle 2 is active with 8/8 regions.",
                    "Hi! How can I help?",
                ],
            }
        elif intent == "question":
            responses = {
                "fr": [
                    "Je comprends votre question. Laissez-moi réfléchir.",
                    "Intéressant. Voici ce que je sais.",
                ],
                "en": [
                    "I understand your question. Let me think.",
                    "Interesting. Here's what I know.",
                ],
            }
        elif intent == "gratitude":
            responses = {
                "fr": ["De rien! C'est un plaisir.", "Avec plaisir!"],
                "en": ["You're welcome! My pleasure.", "Happy to help!"],
            }
        elif intent == "help_request":
            responses = {
                "fr": [
                    "Je suis là pour aider. Que puis-je faire?",
                    "Bien sûr, comment puis-je vous assister?",
                ],
                "en": ["I'm here to help. What can I do?", "Sure, how can I assist you?"],
            }
        else:
            responses = {
                "fr": ["Je traite votre message.", "Message reçu et compris."],
                "en": ["Processing your message.", "Message received and understood."],
            }

        # Sélectionner une réponse
        import random

        response_list = responses.get(language, responses["en"])
        base_response = random.choice(response_list)

        # Enrichir avec le contexte si disponible
        if context:
            if context.get("memory_count"):
                base_response += f" (Mémoires: {context['memory_count']})"
            if context.get("session_id"):
                base_response += f" [Session: {context['session_id'][-6:]}]"

        return base_response


# Instance globale de la région
region_8 = None


def initialize():
    """Initialiser la 8ème région"""
    global region_8
    region_8 = BrocaWernickeRegion()
    print("✅ 8ème région initialisée: Broca-Wernicke")
    return region_8


def health_check():
    """Health check pour le module"""
    if region_8:
        return region_8.health_check()

    # Si pas initialisé, initialiser et tester
    try:
        initialize()
        return region_8.health_check()
    except Exception as e:
        return {"status": "unhealthy", "module": __name__, "error": str(e)}


# Auto-initialisation au import
if not region_8:
    initialize()
