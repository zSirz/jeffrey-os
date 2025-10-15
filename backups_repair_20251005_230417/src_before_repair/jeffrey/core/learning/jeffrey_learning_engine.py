"""
Jeffrey Learning Engine - Le vrai système d'apprentissage de Jeffrey
Utilise GPT pour COMPRENDRE et APPRENDRE, pas pour répondre
"""

import json
import logging
import os
from datetime import datetime
from typing import Any

from .gpt_understanding_helper import GPTUnderstandingHelper
from .memory_manager import MemoryManager

logger = logging.getLogger(__name__)


class JeffreyLearningEngine:
    """
    Moteur d'apprentissage qui permet à Jeffrey d'apprendre vraiment
    au lieu de simplement transmettre les réponses GPT
    """

    def __init__(self, memory_manager: MemoryManager | None = None):
        self.memory = memory_manager or MemoryManager()
        self.gpt_helper = GPTUnderstandingHelper()

        # Mémoire d'apprentissage de Jeffrey
        self.learned_concepts = {}  # Concepts compris par Jeffrey
        self.conversation_patterns = {}  # Patterns de conversation appris
        self.user_preferences = {}  # Préférences des utilisateurs

        # Charger la mémoire d'apprentissage
        self._load_learning_memory()

    def _load_learning_memory(self):
        """Charge la mémoire d'apprentissage depuis un fichier"""
        try:
            learning_file = os.path.join("data", "jeffrey_learning.json")
            if os.path.exists(learning_file):
                with open(learning_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.learned_concepts = data.get("concepts", {})
                    self.conversation_patterns = data.get("patterns", {})
                    self.user_preferences = data.get("preferences", {})
                logger.info(f"✅ Mémoire d'apprentissage chargée: {len(self.learned_concepts)} concepts")
        except Exception as e:
            logger.error(f"Erreur chargement mémoire apprentissage: {e}")

    def _save_learning_memory(self):
        """Sauvegarde la mémoire d'apprentissage"""
        try:
            learning_file = os.path.join("data", "jeffrey_learning.json")
            os.makedirs("data", exist_ok=True)

            data = {
                "concepts": self.learned_concepts,
                "patterns": self.conversation_patterns,
                "preferences": self.user_preferences,
                "last_update": datetime.now().isoformat(),
            }

            with open(learning_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            logger.error(f"Erreur sauvegarde mémoire apprentissage: {e}")

    def process_learning_opportunity(self, user_input: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Traite une opportunité d'apprentissage:
        1. Vérifie si Jeffrey connaît déjà la réponse
        2. Si non, demande à GPT d'EXPLIQUER (pas de répondre)
        3. Jeffrey comprend et mémorise
        4. Jeffrey formule SA PROPRE réponse
        """

        # 1. Vérifier la mémoire existante
        existing_knowledge = self._check_existing_knowledge(user_input)
        if existing_knowledge:
            logger.info("✅ Jeffrey connaît déjà ce sujet")
            return {
                "source": "memory",
                "knowledge": existing_knowledge,
                "confidence": existing_knowledge.get("confidence", 0.8),
            }

        # 2. Jeffrey ne sait pas - demander à GPT d'EXPLIQUER
        if self.gpt_helper.enabled:
            understanding = self.gpt_helper.understand_intent(user_input, context)

            # Si c'est une question technique/conceptuelle
            if understanding.get("intention") == "question" and understanding.get("topics"):
                # Demander une explication à GPT, pas une réponse
                explanation = self._request_explanation_from_gpt(user_input, understanding["topics"])

                if explanation:
                    # 3. Jeffrey comprend et mémorise
                    self._learn_from_explanation(understanding["topics"], explanation)

                    # 4. Retourner la compréhension de Jeffrey
                    return {
                        "source": "learning",
                        "understanding": understanding,
                        "explanation": explanation,
                        "confidence": 0.7,  # Nouvelle connaissance, confiance modérée
                    }

        # Pas d'apprentissage possible
        return {"source": "unknown", "confidence": 0.3}

    def _check_existing_knowledge(self, user_input: str) -> dict[str, Any] | None:
        """Vérifie si Jeffrey a déjà des connaissances sur le sujet"""
        input_lower = user_input.lower()

        # Vérifier les concepts appris
        for concept, knowledge in self.learned_concepts.items():
            if concept.lower() in input_lower:
                return knowledge

        # Vérifier les patterns de conversation
        for pattern, response_data in self.conversation_patterns.items():
            if pattern in input_lower:
                return response_data

        return None

    def _request_explanation_from_gpt(self, user_input: str, topics: list[str]) -> dict[str, Any] | None:
        """Demande à GPT d'EXPLIQUER un concept, pas de répondre"""
        if not self.gpt_helper.enabled:
            return None

        try:
            # Utiliser l'API OpenAI directement pour une demande d'explication
            from openai import OpenAI

            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

            prompt = f"""
            L'utilisateur demande: "{user_input}"
            Sujets identifiés: {', '.join(topics)}

            EXPLIQUE ces concepts de manière simple pour que Jeffrey puisse comprendre et apprendre.
            NE RÉPONDS PAS à la place de Jeffrey.

            Format JSON attendu:
            {{
                "concept_principal": "le concept clé",
                "explication_simple": "explication en termes simples",
                "exemples": ["exemple 1", "exemple 2"],
                "mots_cles": ["mot1", "mot2"],
                "type_interaction": "technique/emotionnel/social",
                "elements_reponse": ["élément que Jeffrey pourrait mentionner"]
            }}
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un tuteur qui explique des concepts à Jeffrey pour qu'il apprenne. Réponds TOUJOURS en JSON valide.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
            )

            # Parser la réponse de manière sécurisée
            try:
                content = response.choices[0].message.content.strip()
                # Essayer de parser le JSON
                if content.startswith("{"):
                    explanation = json.loads(content)
                else:
                    # Si pas JSON, créer une structure basique
                    logger.warning("GPT n'a pas retourné du JSON valide, extraction du contenu...")
                    explanation = {
                        "concept_principal": topics[0] if topics else "concept",
                        "explication_simple": content,
                        "exemples": [],
                        "mots_cles": topics,
                        "type_interaction": "general",
                        "elements_reponse": [content[:100]],
                    }
            except json.JSONDecodeError as e:
                logger.error(f"Erreur parsing JSON: {e}")
                # Fallback - créer une structure minimale
                explanation = {
                    "concept_principal": topics[0] if topics else "concept",
                    "explication_simple": response.choices[0].message.content,
                    "exemples": [],
                    "mots_cles": topics,
                    "type_interaction": "general",
                    "elements_reponse": ["Je comprends mieux maintenant"],
                }

            logger.info(f"💡 GPT a expliqué: {explanation.get('concept_principal')}")
            return explanation

        except Exception as e:
            logger.error(f"Erreur demande explication GPT: {e}")
            return None

    def _learn_from_explanation(self, topics: list[str], explanation: dict[str, Any]):
        """Jeffrey apprend de l'explication et l'intègre dans sa mémoire"""
        concept = explanation.get("concept_principal", topics[0] if topics else "unknown")

        # Mémoriser le concept
        self.learned_concepts[concept] = {
            "explanation": explanation.get("explication_simple"),
            "examples": explanation.get("exemples", []),
            "keywords": explanation.get("mots_cles", []),
            "learned_at": datetime.now().isoformat(),
            "confidence": 0.7,
            "usage_count": 0,
        }

        # Mémoriser des patterns de réponse
        for element in explanation.get("elements_reponse", []):
            pattern_key = f"{concept}_response_{len(self.conversation_patterns)}"
            self.conversation_patterns[pattern_key] = {
                "trigger": concept,
                "response_element": element,
                "type": explanation.get("type_interaction", "general"),
            }

        # Sauvegarder
        self._save_learning_memory()
        logger.info(f"✨ Jeffrey a appris le concept: {concept}")

    def generate_jeffrey_response(
        self, user_input: str, learning_data: dict[str, Any], emotional_state: dict[str, Any]
    ) -> str:
        """
        Génère la PROPRE réponse de Jeffrey basée sur:
        - Ce qu'il a appris
        - Son état émotionnel
        - Sa personnalité
        """

        if learning_data["source"] == "memory":
            # Jeffrey utilise ses connaissances existantes
            knowledge = learning_data["knowledge"]
            base_response = knowledge.get("explanation", "")

            # Augmenter la confiance car il a déjà utilisé cette connaissance
            if "usage_count" in knowledge:
                knowledge["usage_count"] += 1
                knowledge["confidence"] = min(1.0, knowledge["confidence"] + 0.05)

        elif learning_data["source"] == "learning":
            # Jeffrey vient d'apprendre quelque chose
            explanation = learning_data.get("explanation", {})
            base_response = explanation.get("explication_simple", "")

            # Ajouter de la personnalité Jeffrey
            if explanation.get("type_interaction") == "technique":
                prefixes = [
                    "Oh, c'est fascinant! ",
                    "J'ai compris! ",
                    "Alors si je comprends bien, ",
                    "C'est intéressant, ",
                ]
            else:
                prefixes = [
                    "Je pense que ",
                    "D'après ce que je comprends, ",
                    "Il me semble que ",
                    "J'ai l'impression que ",
                ]

            import random

            base_response = random.choice(prefixes) + base_response

        else:
            # Jeffrey ne sait pas - réponse honnête
            responses = [
                "Hmm, c'est une excellente question! Je ne suis pas sûr de bien comprendre. Peux-tu m'en dire plus?",
                "J'aimerais t'aider mais je ne connais pas encore bien ce sujet. Tu peux m'expliquer?",
                "C'est intéressant! Je n'ai pas encore appris ça. Qu'est-ce que tu en penses toi?",
                "Oh, je découvre! Raconte-moi ce que tu sais là-dessus?",
            ]
            import random

            return random.choice(responses)

        # Adapter selon l'émotion
        emotion = emotional_state.get("dominant", "neutral")
        if emotion == "joy":
            base_response += " 😊"
        elif emotion == "curiosity":
            base_response += " 🤔 D'ailleurs, ça me fait penser..."
        elif emotion == "empathy":
            base_response += " 💝 J'espère que ça t'aide!"

        return base_response

    def update_from_feedback(self, user_input: str, jeffrey_response: str, user_reaction: str | None = None):
        """Met à jour l'apprentissage basé sur le feedback"""
        # Extraire les points d'apprentissage si GPT est disponible
        if self.gpt_helper.enabled and user_reaction:
            understanding = self.gpt_helper.understand_intent(user_input, {})
            learning_points = self.gpt_helper.extract_learning_points(user_input, jeffrey_response, understanding)

            # Mettre à jour les patterns si l'interaction était réussie
            if learning_points.get("successful_understanding"):
                for pattern in learning_points.get("patterns", []):
                    self.conversation_patterns[pattern] = {
                        "example": user_input,
                        "successful_response": jeffrey_response,
                        "timestamp": datetime.now().isoformat(),
                    }

                self._save_learning_memory()
                logger.info("📈 Jeffrey a amélioré ses patterns de conversation")
