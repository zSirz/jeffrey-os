"""Dialogue engine module - Moteur de dialogue pour Jeffrey AGI"""

__jeffrey_meta__ = {
    "version": "1.0.0",
    "stability": "stable",
    "brain_regions": ["broca_wernicke"],
}

import random
from typing import Any


class DialogueEngine:
    """Moteur de dialogue avec patterns émotionnels"""

    def __init__(self):
        """Initialise le moteur de dialogue"""
        self.conversations = 0
        self.memory = None  # Sera injecté par l'orchestrateur si disponible

        # ✅ CRITIQUE : Initialiser les patterns émotionnels
        self.emotional_patterns = {
            "joie": {
                "templates": ["Génial ! {input}", "Super ! 😊 {input}", "C'est fantastique ! ✨"],
                "default_response": "Je partage ta joie ! 😊",
            },
            "tristesse": {
                "templates": ["Je comprends que ce soit difficile...", "Je suis là pour toi 💙", "Ça doit être dur..."],
                "default_response": "Je t'écoute avec empathie...",
            },
            "peur": {
                "templates": ["C'est normal d'avoir peur...", "Tu es en sécurité ici.", "Prends ton temps..."],
                "default_response": "Je suis là, tu peux me parler.",
            },
            "colère": {
                "templates": ["Je comprends ta frustration...", "C'est ok d'être en colère.", "Parlons-en..."],
                "default_response": "Je t'écoute sans jugement.",
            },
            "curiosité": {
                "templates": ["Excellente question !", "Explorons ça ensemble... 🤔", "C'est fascinant comme sujet !"],
                "default_response": "Je suis curieux aussi ! 🤔",
            },
        }

    async def process(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Traite une requête de dialogue (méthode async legacy)"""
        self.conversations += 1
        return {"status": "ok", "response": "dialogue processed", "count": self.conversations}

    def health_check(self) -> dict[str, Any]:
        """Vérifie la santé du module"""
        return {"status": "healthy", "conversations": self.conversations}

    def generate_response(
        self, user_input: str, user_id: str = "default", conversation_id: str = None, override_emotion: str = None
    ) -> str:
        """
        Génère une réponse de dialogue adaptée

        Args:
            user_input: Message de l'utilisateur
            user_id: Identifiant de l'utilisateur
            conversation_id: ID de conversation optionnel
            override_emotion: Émotion à utiliser pour adapter la réponse

        Returns:
            str: Réponse générée
        """
        # Incrémenter le compteur
        self.conversations += 1

        # ✅ INDENTATION CORRECTE ICI
        # Si on a une émotion spécifique, adapter le ton
        if override_emotion and override_emotion in self.emotional_patterns:
            pattern = self.emotional_patterns[override_emotion]
            response = self._generate_from_pattern(pattern, user_input)
        else:
            # Génération standard
            response = self._generate_standard_response(user_input)

        # Appliquer la mémoire si disponible
        if hasattr(self, 'memory') and self.memory:
            response = self._enrich_with_memory(response, user_id)

        return response

    def _generate_from_pattern(self, pattern: dict, user_input: str) -> str:
        """Génère une réponse depuis un pattern émotionnel"""
        # Si le pattern a des templates
        if "templates" in pattern:
            template = random.choice(pattern["templates"])
            # Formater avec l'input si {input} est présent
            if "{input}" in template:
                return template.format(input=user_input[:50])
            return template

        # Sinon, réponse basique adaptée à l'émotion
        return pattern.get("default_response", "Je t'écoute...")

    def _generate_standard_response(self, user_input: str) -> str:
        """Génère une réponse standard"""
        # Patterns de base
        responses = [
            "C'est intéressant ce que tu dis...",
            "Je comprends, continue...",
            "Dis-m'en plus...",
            "Je t'écoute attentivement.",
            "Hmm, je vois ce que tu veux dire...",
        ]

        # Si c'est une question
        if "?" in user_input:
            responses.extend(["C'est une excellente question...", "Laisse-moi y réfléchir...", "Bonne question ! 🤔"])

        # Si c'est très court
        if len(user_input.strip()) < 10:
            responses.extend(["Je t'écoute... 👂", "Je suis là...", "Continue..."])

        return random.choice(responses)

    def _enrich_with_memory(self, response: str, user_id: str) -> str:
        """Enrichit la réponse avec le contexte mémoire"""
        try:
            # Récupérer contexte
            if hasattr(self.memory, 'get_context_summary'):
                context = self.memory.get_context_summary()

                # Si on a du contexte, on peut l'utiliser
                if context and len(context) > 10:
                    # Ajouter une référence subtile (30% du temps)
                    if random.random() < 0.3:
                        response += " (Ça me rappelle notre dernière conversation...)"

        except Exception:
            pass  # ✅ OK : ignore l'erreur et retourne réponse sans enrichissement

        return response


# Fonction standalone pour health check
def health_check():
    """Health check standalone du module"""
    _ = sum(range(1000))
    return {"status": "healthy", "module": "dialogue_engine", "work": _}
