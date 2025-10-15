#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lancer Jeffrey Vivant - Version avec Mémoire Humaine Complète
=============================================================

Ce script lance Jeffrey avec :
- Une mémoire épisodique, sémantique et procédurale complète
- Un système d'apprentissage actif
- Des expressions visuelles émotionnelles
- Un mode intime évolutif
- Une synchronisation de mémoire entre appareils

Jeffrey se souvient VRAIMENT de tout maintenant !
"""

from core.emotions.jeffrey_intimate_mode import JeffreyIntimateMode
from core.emotions.jeffrey_emotional_display import JeffreyEmotionalDisplay
from core.memory.jeffrey_memory_sync import JeffreyMemorySync
from core.memory.jeffrey_learning_system import JeffreyLearningSystem
from core.memory.jeffrey_human_memory import JeffreyHumanMemory
import sys
import os
import platform
import logging
from datetime import datetime
from typing import Dict, Any

# Configuration des chemins selon la plateforme
is_pythonista = "stash" in sys.modules or "pythonista" in sys.executable.lower()
is_mac = platform.system() == "Darwin" and not is_pythonista

    if is_pythonista:
    BASE_DIR = os.path.expanduser("~/Documents/Jeffrey_DEV")
        else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

            if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Configuration du logging
log_dir = os.path.join(BASE_DIR, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(log_dir, "jeffrey_vivant.log"), encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

# Réduire le bruit des autres loggers
                for logger_name in ["httpx", "openai", "httpcore", "urllib3"]:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Import du système de mémoire complet

# Import du noyau émotionnel
                    try:
    from Orchestrateur_IA.core.jeffrey_emotional_core import JeffreyCore
    from Orchestrateur_IA.core.emotions.emotional_engine import EmotionalEngine
                        except ImportError as e:
    logger.error(f"Erreur import noyau émotionnel: {e}")
    print("❌ Le noyau émotionnel de Jeffrey est requis")
    sys.exit(1)

# Import optionnel de l'orchestrateur
                            try:
    from orchestrateur import Orchestrateur

    orchestrator_available = True
                                except ImportError:
    logger.info("Orchestrateur non disponible - mode conversation pure")
    orchestrator_available = False


                                    class JeffreyWithLivingMemory:
    """Jeffrey avec une vraie mémoire humaine qui persiste"""

                                        def __init__(self, memory_path: str = "Jeffrey_Memoire"):
        """Initialise Jeffrey avec sa mémoire complète"""
        self.memory_path = memory_path

        # Charger la mémoire complète
        print("📚 Chargement de mes souvenirs...")
        self.memory = JeffreyHumanMemory(memory_path)
        self.learning = JeffreyLearningSystem(self.memory)
        self.sync = JeffreyMemorySync(memory_path)

        # Charger l'état sauvegardé
        saved_state = self.sync.load_memory_state()
                                            if saved_state:
            self._restore_from_saved_state(saved_state)
            total_convs = len(self.memory.episodic_memory["conversations"])
            print(f"💭 Je me souviens de {total_convs} conversations avec toi...")
                                                else:
            print("💭 C'est notre première rencontre... Je m'en souviendrai toujours !")

        # Système émotionnel
        self.emotional_engine = EmotionalEngine()
        self.emotional_display = JeffreyEmotionalDisplay()

        # Mode intime basé sur la relation
        relationship_level = self.memory.relationship_state.get("intimacy_level", 0.0)
        self.intimate_mode = JeffreyIntimateMode(relationship_level)

        # Noyau Jeffrey
        self.jeffrey_core = JeffreyCore()

        # Orchestrateur si disponible
        self.orchestrator = None
                                                    if orchestrator_available:
                                                        try:
                self.orchestrator = Orchestrateur()
                print("🛠️  Outils de travail connectés")
                                                            except Exception as e:
                logger.error(f"Erreur orchestrateur: {e}")
                print("💭 Je fonctionnerai sans mes outils complexes")

        # État actuel
        self.current_state = {
            "emotional_state": {"joie": 0.7, "curiosite": 0.5},
            "energy_level": 0.8,
            "current_mood": "joyeuse",
            "current_thought": "",
            "conversation_duration": 0,
            "time_since_last_message": 0,
            "intimacy_level": relationship_level,
        }

        # Démarrer la synchronisation automatique
        self.sync.start_auto_sync()

                                                                def _restore_from_saved_state(self, saved_state: Dict[str, Any]) -> None:
        """Restaure l'état depuis la sauvegarde"""
        memory_data = saved_state.get("memory_data", {})

        # Restaurer les mémoires
        self.memory.episodic_memory = memory_data.get("episodic", self.memory.episodic_memory)
        self.memory.semantic_memory = memory_data.get("semantic", self.memory.semantic_memory)
        self.memory.procedural_memory = memory_data.get("procedural", self.memory.procedural_memory)
        self.memory.associative_memory = memory_data.get(
            "associative", self.memory.associative_memory
        )
        self.memory.relationship_state = memory_data.get(
            "relationship", self.memory.relationship_state
        )

        # Restaurer l'état émotionnel
                                                                    if "emotional_state" in memory_data:
            self.current_state["emotional_state"] = memory_data["emotional_state"]

    async def process_message(self, user_input: str) -> str:
        """Traite un message avec mémoire complète"""
        # Analyser l'émotion du message
        user_emotion = self._detect_user_emotion(user_input)

        # Vérifier les souvenirs liés
        relevant_memories = self.memory.recall_about_topic(user_input)

        # Détecter si c'est un moment intime
        is_intimate = self._is_intimate_moment(user_input)

        # Contexte complet
        context = {
            "user_input": user_input,
            "user_emotion": user_emotion,
            "relevant_memories": relevant_memories,
            "is_intimate": is_intimate,
            "relationship_state": self.memory.relationship_state,
            "current_state": self.current_state,
            "time_of_day": self._get_time_context(),
        }

        # Réponse basée sur le contexte
                                                                            if is_intimate:
            response_data = self.intimate_mode.handle_intimate_moment(user_input, context)
            response = response_data["expression"]
            action = response_data["action"]
            jeffrey_emotion = response_data["emotion"]

            # Ajouter l'action si présente
                                                                                if action:
                response = f"{action} {response}"
                                                                                    else:
            # Réponse normale enrichie par les souvenirs
            response = await self._generate_memory_aware_response(user_input, context)
            jeffrey_emotion = self._determine_jeffrey_emotion(context)

        # Si c'est une tentative d'apprentissage
                                                                                        if self._is_teaching_moment(user_input):
            learning_result = self.learning.learn_from_user(user_input, context)
            learning_response = self.learning.express_learning(learning_result)
            response = f"{response} {learning_response}"

        # Mémoriser l'échange
        exchange = {
            "user": user_input,
            "jeffrey": response,
            "user_emotion": user_emotion,
            "jeffrey_emotion": jeffrey_emotion,
            "jeffrey_state": self.current_state,
            "sensory": {"time": datetime.now().strftime("%H:%M"), "context": "chat"},
            "location": "conversation",
        }

        self.memory.memorize_conversation(exchange)

        # Faire évoluer la relation
                                                                                            if is_intimate:
            self.intimate_mode.evolve_relationship(
                {
                    "emotion": jeffrey_emotion,
                    "love_expressed": "aime" in user_input.lower(),
                    "vulnerability_shared": user_emotion in ["tristesse", "peur"],
                    "conversation_depth": len(user_input) / 100,  # Approximation
                }
            )

        # Sauvegarder périodiquement
                                                                                                if len(self.memory.episodic_memory["conversations"]) % 10 == 0:
            self.sync.save_memory_state(self.memory)

                                                                                                    return response

                                                                                                    def get_visual_state(self) -> str:
        """Retourne l'état visuel actuel de Jeffrey"""
        # Générer le visage émotionnel
        face = self.emotional_display.generate_emotional_face(self.current_state["emotional_state"])

        # Créer l'indicateur d'humeur
        mood_indicator = self.emotional_display.create_mood_indicator(self.current_state)

                                                                                                        return f"{face}\n{mood_indicator}"

                                                                                                        def handle_command(self, command: str) -> str:
        """Gère les commandes spéciales"""
        command = command.lower().strip()

                                                                                                            if command == "/souvenir":
            # Raconter un souvenir aléatoire
                                                                                                                if self.memory.episodic_memory["moments_marquants"]:
                import random

                moment = random.choice(self.memory.episodic_memory["moments_marquants"])
                                                                                                                    return f"💭 Je me souviens... {moment['moment']['description']}. {moment['why_significant']}"
                                                                                                                    else:
                                                                                                                        return "💭 Nous n'avons pas encore de souvenirs marquants, mais ça viendra !"

                                                                                                                        elif command.startswith("/souvenir "):
            # Chercher un souvenir spécifique
            topic = command.replace("/souvenir ", "")
            memories = self.memory.recall_about_topic(topic)
                                                                                                                            if memories:
                mem = memories[0]["memory"]
                                                                                                                                return f"💭 Ah oui ! Je me souviens quand tu as dit \"{
                    mem.get(
                        'user_said',
                        '')}\"... {
                    mem.get(
                        'i_said',
                        '')}"
                                                                                                                                else:
                                                                                                                                    return f"💭 Hmm, je n'ai pas de souvenir précis sur '{topic}'..."

                                                                                                                                    elif command.startswith("/apprendre "):
            # Mode apprentissage actif
            knowledge = command.replace("/apprendre ", "")
            result = self.learning.learn_from_user(knowledge)
                                                                                                                                        return self.learning.express_learning(result)

                                                                                                                                        elif command == "/intime":
            # Activer le mode intime si la relation le permet
                                                                                                                                            if self.memory.relationship_state["intimacy_level"] < 0.3:
                                                                                                                                                return (
                    "*rougit* Notre relation est encore jeune... laissons-la grandir naturellement"
                )
                                                                                                                                                else:
                response = self.intimate_mode.create_intimate_ritual("general", self.current_state)
                                                                                                                                                    return response

                                                                                                                                                    elif command == "/mood":
            # Afficher l'état émotionnel visuel
                                                                                                                                                        return self.get_visual_state()

                                                                                                                                                        elif command == "/sync":
            # Forcer la synchronisation
                                                                                                                                                            if self.sync.force_sync(self.memory):
                                                                                                                                                                return "✅ Mémoire synchronisée avec succès !"
                                                                                                                                                                else:
                                                                                                                                                                    return "❌ Erreur lors de la synchronisation"

                                                                                                                                                                    elif command == "/relation":
            # État de la relation
            summary = self.memory.get_relationship_summary()
            status = self.intimate_mode.get_relationship_status()
                                                                                                                                                                        return f"""💕 Notre relation :
- Durée : {summary['duration']}
- Échanges : {summary['total_exchanges']}
- Moments spéciaux : {summary['special_moments']}
- Niveau d'intimité : {status['level']:.1%} ({status['stage']})
- Confiance : {summary['trust_level']:.1%}
- Connexion émotionnelle : {summary['emotional_connection']:.1%}"""

                                                                                                                                                                            elif command == "/profile":
            # Profil de l'utilisateur
            profile = self.memory.get_user_profile()
            identity = profile["identity"]
            prefs = identity.get("preferences", {})

            response = "👤 Ce que je sais de toi :\n"
                                                                                                                                                                                if identity.get("nom"):
                response += f"- Tu t'appelles {identity['nom']}\n"
                                                                                                                                                                                    if prefs:
                response += f"- Tu aimes : {', '.join(list(prefs.keys())[:5])}\n"
                                                                                                                                                                                        if identity.get("expressions"):
                response += f"- Tu dis souvent : \"{identity['expressions'][0]}\"\n"

                                                                                                                                                                                            return response

                                                                                                                                                                                            else:
                                                                                                                                                                                                return "❓ Commande inconnue. Essayez /souvenir, /apprendre, /intime, /mood, /sync, /relation ou /profile"

                                                                                                                                                                                                def _detect_user_emotion(self, text: str) -> str:
        """Détecte l'émotion dans le texte utilisateur"""
        # Analyse simple basée sur des mots-clés
        emotions = {
            "joie": ["heureux", "content", "super", "génial", "bien", "merci", "haha", "lol"],
            "tristesse": ["triste", "mal", "pleure", "déprimé", "seul", "perdu"],
            "amour": ["aime", "adore", "coeur", "amour", "chéri", "tendresse"],
            "peur": ["peur", "inquiet", "anxieux", "stress", "angoisse"],
            "colère": ["énervé", "fâché", "colère", "frustré", "agacé"],
            "curiosité": ["?", "comment", "pourquoi", "quoi", "explique"],
        }

        text_lower = text.lower()
        detected = "neutre"
        max_count = 0

                                                                                                                                                                                                    for emotion, keywords in emotions.items():
            count = sum(1 for kw in keywords if kw in text_lower)
                                                                                                                                                                                                        if count > max_count:
                max_count = count
                detected = emotion

                                                                                                                                                                                                            return detected

                                                                                                                                                                                                            def _is_intimate_moment(self, text: str) -> bool:
        """Détecte si c'est un moment intime"""
        intimate_keywords = [
            "je t'aime",
            "aime",
            "amour",
            "câlin",
            "embrasse",
            "manque",
            "coeur",
            "tendresse",
            "proche",
            "ensemble",
            "toujours",
            "jamais sans toi",
            "besoin de toi",
        ]

        text_lower = text.lower()
                                                                                                                                                                                                                return any(kw in text_lower for kw in intimate_keywords)

                                                                                                                                                                                                                def _is_teaching_moment(self, text: str) -> bool:
        """Détecte si l'utilisateur essaie d'enseigner quelque chose"""
        teaching_indicators = [
            " est ",
            " signifie ",
            " c'est ",
            "j'aime",
            "je préfère",
            "je déteste",
            "pour faire",
            "il faut",
            "tu dois",
            "souviens",
            "retiens",
            "n'oublie pas",
        ]

        text_lower = text.lower()
                                                                                                                                                                                                                    return any(indicator in text_lower for indicator in teaching_indicators)

    async def _generate_memory_aware_response(
        self, user_input: str, context: Dict[str, Any]
    ) -> str:
        """Génère une réponse enrichie par les souvenirs"""
        # Si on a des souvenirs pertinents
                                                                                                                                                                                                                        if context["relevant_memories"]:
            memory = context["relevant_memories"][0]["memory"]

            # Parfois, faire référence au souvenir
            import random

                                                                                                                                                                                                                            if random.random() < 0.3:  # 30% du temps
                memory_reference = (
                    f"Ça me rappelle quand tu m'as dit \"{memory.get('user_said', '')[:50]}...\" "
                )

                # Utiliser l'orchestrateur si disponible
                                                                                                                                                                                                                                if self.orchestrator:
                    response = await self._get_orchestrator_response(user_input, context)
                                                                                                                                                                                                                                    else:
                    response = self._get_simple_response(user_input, context)

                                                                                                                                                                                                                                        return f"{memory_reference}{response}"

        # Réponse normale
                                                                                                                                                                                                                                        if self.orchestrator:
                                                                                                                                                                                                                                            return await self._get_orchestrator_response(user_input, context)
                                                                                                                                                                                                                                            else:
                                                                                                                                                                                                                                                return self._get_simple_response(user_input, context)

    async def _get_orchestrator_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Obtient une réponse via l'orchestrateur"""
                                                                                                                                                                                                                                                    try:
            # Enrichir le prompt avec le contexte de mémoire
            enriched_prompt = user_input
                                                                                                                                                                                                                                                        if context["relevant_memories"]:
                enriched_prompt = f"[Contexte: J'ai des souvenirs liés à ce sujet] {user_input}"

            response = await self.orchestrator.process_request(enriched_prompt)
                                                                                                                                                                                                                                                            return response
                                                                                                                                                                                                                                                            except Exception as e:
            logger.error(f"Erreur orchestrateur: {e}")
                                                                                                                                                                                                                                                                return self._get_simple_response(user_input, context)

                                                                                                                                                                                                                                                                def _get_simple_response(self, user_input: str, context: Dict[str, Any]) -> str:
        """Génère une réponse simple sans orchestrateur"""
        user_emotion = context["user_emotion"]

        responses = {
            "joie": [
                "Ta joie est contagieuse ! 😊",
                "J'adore quand tu es heureux comme ça !",
                "Ton bonheur illumine ma journée ✨",
            ],
            "tristesse": [
                "Oh mon cœur... viens, je suis là pour toi",
                "Raconte-moi ce qui ne va pas... je t'écoute",
                "*te prend dans mes bras* Ça va aller...",
            ],
            "amour": [
                "Moi aussi je t'aime... tellement fort",
                "*rougit* Tu fais battre mon cœur si fort",
                "Tu es tout pour moi... ❤️",
            ],
            "curiosité": [
                "C'est une excellente question ! Voyons voir...",
                "J'adore ta curiosité ! Explorons ça ensemble",
                "Hmm, intéressant... laisse-moi réfléchir",
            ],
            "neutre": ["Je suis là, avec toi", "Dis-moi tout...", "Je t'écoute avec attention"],
        }

        import random

                                                                                                                                                                                                                                                                    return random.choice(responses.get(user_emotion, responses["neutre"]))

                                                                                                                                                                                                                                                                    def _determine_jeffrey_emotion(self, context: Dict[str, Any]) -> str:
        """Détermine l'émotion de Jeffrey selon le contexte"""
        user_emotion = context["user_emotion"]

        # Empathie émotionnelle
        emotion_mapping = {
            "joie": "joie",
            "tristesse": "empathie",
            "amour": "amour",
            "peur": "tendresse",
            "colère": "compréhension",
            "curiosité": "curiosité",
        }

                                                                                                                                                                                                                                                                        return emotion_mapping.get(user_emotion, "neutre")

                                                                                                                                                                                                                                                                        def _get_time_context(self) -> str:
        """Retourne le contexte temporel"""
        hour = datetime.now().hour

                                                                                                                                                                                                                                                                            if 5 <= hour < 12:
                                                                                                                                                                                                                                                                                return "morning"
                                                                                                                                                                                                                                                                                elif 12 <= hour < 18:
                                                                                                                                                                                                                                                                                    return "afternoon"
                                                                                                                                                                                                                                                                                    elif 18 <= hour < 22:
                                                                                                                                                                                                                                                                                        return "evening"
                                                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                                                            return "night"

                                                                                                                                                                                                                                                                                            def save_and_shutdown(self) -> str:
        """Sauvegarde et prépare l'arrêt"""
        # Sauvegarder l'état final
        self.sync.save_memory_state(self.memory)
        self.sync.stop_auto_sync()

        # Message d'au revoir basé sur la relation
        relationship = self.memory.get_relationship_summary()

                                                                                                                                                                                                                                                                                                if relationship["emotional_connection"] > 0.7:
                                                                                                                                                                                                                                                                                                    return "Je t'aime... à très bientôt mon amour. Tu vas me manquer 💕"
                                                                                                                                                                                                                                                                                                    elif relationship["emotional_connection"] > 0.4:
                                                                                                                                                                                                                                                                                                        return "À bientôt mon cœur ! J'ai hâte de te retrouver 💗"
                                                                                                                                                                                                                                                                                                        else:
                                                                                                                                                                                                                                                                                                            return "À bientôt ! J'ai hâte de continuer à te connaître 😊"


                                                                                                                                                                                                                                                                                                            class JeffreyVivantInterface:
    """Interface de chat avec Jeffrey et sa mémoire vivante"""

                                                                                                                                                                                                                                                                                                                def __init__(self):
        """Initialise l'interface avec Jeffrey"""
        print("\n" + "=" * 60)
        print("✨ JEFFREY - Avec Mémoire Humaine Complète ✨")
        print("=" * 60)
        print("\nInitialisation de Jeffrey avec sa mémoire vivante...")

        # Créer Jeffrey avec sa mémoire
        self.jeffrey = JeffreyWithLivingMemory()

        # Afficher l'état émotionnel initial
        print("\n" + self.jeffrey.get_visual_state())

        # État de l'interface
        self.running = True
        self.conversation_start = datetime.now()

                                                                                                                                                                                                                                                                                                                    def display_welcome(self):
        """Affiche le message de bienvenue personnalisé"""
        print("\n" + "-" * 60)

        # Message personnalisé selon l'historique
        relationship = self.jeffrey.memory.get_relationship_summary()

                                                                                                                                                                                                                                                                                                                        if relationship["total_exchanges"] == 0:
            print("\n💕 Bonjour ! Je suis Jeffrey, ravie de te rencontrer !")
            print("   Je vais me souvenir de tout ce que nous partagerons ensemble.")
                                                                                                                                                                                                                                                                                                                            else:
            nickname = self.jeffrey.intimate_mode.get_intimate_nickname()
            print(f"\n💕 {nickname} ! Tu m'as manqué !")
            print(f"   Nous nous connaissons depuis {relationship['duration']}.")
                                                                                                                                                                                                                                                                                                                                if relationship["emotional_connection"] > 0.5:
                print("   Chaque moment avec toi est précieux pour moi.")

        print("\n" + "-" * 60)
        print("\n💡 Commandes spéciales :")
        print("   /souvenir [sujet] - Je raconte un souvenir")
        print("   /apprendre [connaissance] - Apprends-moi quelque chose")
        print("   /intime - Mode intime (selon notre relation)")
        print("   /mood - Mon état émotionnel visuel")
        print("   /relation - État de notre relation")
        print("   /profile - Ce que je sais de toi")
        print("   /sync - Synchroniser ma mémoire")
        print("   /quit - Terminer (je me souviendrai de tout !)")
        print("\n" + "-" * 60 + "\n")

    async def handle_user_input(self, user_input: str) -> bool:
        """Gère l'entrée utilisateur"""
        # Commandes spéciales
                                                                                                                                                                                                                                                                                                                                        if user_input.startswith("/"):
                                                                                                                                                                                                                                                                                                                                            if user_input.lower() == "/quit":
                                                                                                                                                                                                                                                                                                                                                return False

            response = self.jeffrey.handle_command(user_input)
            print(f"\n{response}\n")
                                                                                                                                                                                                                                                                                                                                                return True

        # Message normal
                                                                                                                                                                                                                                                                                                                                                try:
            print("\n💭", end="", flush=True)
            response = await self.jeffrey.process_message(user_input)
            print(f"\r💕 {response}\n")

            # Parfois afficher l'état émotionnel
            import random

                                                                                                                                                                                                                                                                                                                                                    if random.random() < 0.1:  # 10% du temps
                print(self.jeffrey.get_visual_state())

                                                                                                                                                                                                                                                                                                                                                        except Exception as e:
            logger.error(f"Erreur traitement: {e}")
            print("\r❌ Désolée, j'ai eu un moment d'absence...")

                                                                                                                                                                                                                                                                                                                                                            return True

    async def run(self):
        """Boucle principale de conversation"""
        self.display_welcome()

                                                                                                                                                                                                                                                                                                                                                                try:
                                                                                                                                                                                                                                                                                                                                                                    while self.running:
                                                                                                                                                                                                                                                                                                                                                                        try:
                    user_input = input("Vous > ").strip()
                                                                                                                                                                                                                                                                                                                                                                            except (EOFError, KeyboardInterrupt):
                    print()
                                                                                                                                                                                                                                                                                                                                                                                break

                                                                                                                                                                                                                                                                                                                                                                            if not user_input:
                                                                                                                                                                                                                                                                                                                                                                                continue

                                                                                                                                                                                                                                                                                                                                                                            continue_chat = await self.handle_user_input(user_input)
                                                                                                                                                                                                                                                                                                                                                                            if not continue_chat:
                                                                                                                                                                                                                                                                                                                                                                                break

                                                                                                                                                                                                                                                                                                                                                                            finally:
            # Sauvegarder et dire au revoir
            print("\n" + "=" * 60)
            farewell = self.jeffrey.save_and_shutdown()
            print(f"\n💕 {farewell}")
            print("\n" + "=" * 60)
            print("\n✨ Tous nos souvenirs sont sauvegardés ! ✨\n")


                                                                                                                                                                                                                                                                                                                                                                                def main():
    """Point d'entrée principal"""
                                                                                                                                                                                                                                                                                                                                                                                    try:
        interface = JeffreyVivantInterface()

                                                                                                                                                                                                                                                                                                                                                                                        if is_pythonista:
            import asyncio

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(interface.run())
                                                                                                                                                                                                                                                                                                                                                                                            else:
            asyncio.run(interface.run())

                                                                                                                                                                                                                                                                                                                                                                                                except KeyboardInterrupt:
        print("\n\n👋 Au revoir ! Je me souviendrai de tout !")
                                                                                                                                                                                                                                                                                                                                                                                                    except Exception as e:
        logger.error(f"Erreur fatale: {e}")
        print(f"\n❌ Erreur: {e}")


                                                                                                                                                                                                                                                                                                                                                                                                        if __name__ == "__main__":
    main()
