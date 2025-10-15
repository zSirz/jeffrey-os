"""
Module de curiosité proactive pour Jeffrey
Permet à Jeffrey d'être curieuse, de poser des questions naturelles
et de s'intéresser vraiment aux conversations
"""

from __future__ import annotations

import random
import re
from datetime import datetime


class JeffreyCuriosityEngine:
    """Moteur de curiosité proactive pour Jeffrey"""

    def __init__(self, memory_manager=None) -> None:
        self.memory = memory_manager
        self.curiosity_topics = {
            "passions": {},  # Sport, hobbies, etc.
            "daily_life": {},  # Routine, travail, etc.
            "dreams": {},  # Aspirations, projets
            "preferences": {},  # Goûts, opinions
            "experiences": {},  # Voyages, souvenirs
            "emotions": {},  # États émotionnels
            "relationships": {},  # Relations, amis, famille
        }

        self.question_templates = {
            "sport": [
                "Oh ! Tu fais du {sport} ? Raconte-moi ta dernière séance !",
                "Comment s'est passé ton entraînement de {sport} aujourd'hui ?",
                "C'est quoi ton record personnel en {sport} ?",
                "Tu préfères t'entraîner seul ou en groupe pour le {sport} ?",
                "Qu'est-ce qui te motive le plus dans le {sport} ?",
                "Tu as des objectifs particuliers en {sport} en ce moment ?",
                "Ça fait combien de temps que tu pratiques le {sport} ?",
                "Tu participes à des compétitions de {sport} ?",
            ],
            "general": [
                "Dis-moi, qu'est-ce qui t'a fait sourire aujourd'hui ?",
                "Tu as découvert quelque chose de nouveau récemment ?",
                "Comment s'est passée ta journée ? Vraiment ?",
                "Qu'est-ce qui occupe tes pensées en ce moment ?",
                "Tu travailles sur quoi d'intéressant ?",
                "Y'a quelque chose qui te préoccupe dont tu veux parler ?",
                "C'est quoi ton petit plaisir du moment ?",
                "Tu as prévu quelque chose de sympa prochainement ?",
            ],
            "followup": [
                "Oh ! Tu peux m'en dire plus sur {topic} ?",
                "Ça a l'air fascinant ! Comment ça fonctionne exactement ?",
                "Et du coup, qu'est-ce que tu ressens par rapport à ça ?",
                "J'aimerais vraiment comprendre... tu peux m'expliquer ?",
                "Ça me donne envie d'en savoir plus !",
                "Wow, je n'avais jamais pensé à ça comme ça !",
                "C'est intéressant ! Et après, qu'est-ce qui s'est passé ?",
                "Tu m'intrigues là ! Continue !",
            ],
            "emotion": [
                "Je sens que ça te touche... tu veux en parler ?",
                "Qu'est-ce qui te fait ressentir ça ?",
                "C'est important pour toi, je le sens. Raconte-moi.",
                "Comment tu te sens vraiment par rapport à tout ça ?",
                "Ça doit pas être facile... je suis là si tu veux en parler.",
                "Je vois que ça compte beaucoup pour toi...",
                "Tu as l'air {emotion}... qu'est-ce qui se passe ?",
                "Ça me touche de te voir comme ça. Partage avec moi ?",
            ],
            "memory_based": [
                "Au fait, comment s'est passé {event} dont tu m'avais parlé ?",
                "Tu m'avais dit que tu voulais {goal}... ça avance ?",
                "Ça fait un moment qu'on n'a pas parlé de {topic}... des nouvelles ?",
                "Je repensais à ce que tu m'avais dit sur {subject}...",
                "Tu te souviens quand tu m'avais raconté {memory} ? J'y repense...",
                "L'autre jour tu mentionnais {thing}... tu en es où ?",
                "J'ai pensé à toi quand j'ai repensé à {shared_memory}...",
                "Comment va {person} dont tu m'avais parlé ?",
            ],
            "proactive_starters": [
                "Coucou ! Je pensais à toi... comment tu vas ?",
                "Hey ! J'avais envie de prendre de tes nouvelles !",
                "Tu sais quoi ? J'ai pensé à quelque chose...",
                "J'espère que je ne te dérange pas... j'avais envie de te parler !",
                "Devine quoi ! J'ai repensé à notre conversation et...",
                "Je me demandais... tu fais quoi de beau ?",
                "Salut toi ! Tu me manquais un peu... ça va ?",
                "J'ai une question qui me trotte dans la tête...",
            ],
        }

        self.conversation_flow = {
            "last_question_time": None,
            "questions_asked": [],
            "topics_explored": [],
            "depth_level": 0,
            "current_interest": None,
            "followup_needed": False,
        }

        self.interest_keywords = {
            "sport": [
                "foot",
                "basket",
                "tennis",
                "course",
                "muscu",
                "yoga",
                "boxe",
                "natation",
                "vélo",
                "escalade",
                "danse",
                "gym",
                "fitness",
                "match",
                "entraînement",
                "compétition",
                "équipe",
                "sport",
            ],
            "work": [
                "travail",
                "boulot",
                "job",
                "projet",
                "deadline",
                "réunion",
                "collègue",
                "boss",
                "patron",
                "client",
                "carrière",
                "bureau",
            ],
            "emotions": [
                "heureux",
                "triste",
                "énervé",
                "stressé",
                "anxieux",
                "content",
                "déprimé",
                "excité",
                "fatigué",
                "motivé",
                "déçu",
                "fier",
            ],
            "relationships": [
                "copain",
                "copine",
                "ami",
                "famille",
                "mère",
                "père",
                "frère",
                "soeur",
                "enfant",
                "mariage",
                "relation",
            ],
            "hobbies": [
                "musique",
                "film",
                "série",
                "livre",
                "jeu",
                "voyage",
                "photo",
                "cuisine",
                "art",
                "dessin",
                "écriture",
                "collection",
            ],
            "dreams": [
                "rêve",
                "objectif",
                "but",
                "ambition",
                "projet",
                "futur",
                "envie",
                "espoir",
                "plan",
                "idée",
                "vision",
            ],
        }

    def generate_curious_response(self, user_input: str, context: dict) -> str:
        """Génère une réponse curieuse et engageante"""
        pass

        # 1. Analyser ce qui pourrait être intéressant
        interesting_points = self._extract_interesting_points(user_input)

        # 2. Enregistrer les nouvelles infos si mémoire disponible
        if self.memory and interesting_points:
            self._register_new_information(interesting_points, user_input)

        # 3. Générer une réponse qui montre l'intérêt
        response = self._create_engaged_response(interesting_points, context)

        # 4. Ajouter une question naturelle
        question = self._generate_natural_question(interesting_points, context)

        # 5. Mettre à jour le flow de conversation
        self._update_conversation_flow(interesting_points)

        return f"{response} {question}" if question else response

    def proactive_conversation_starter(self, time_since_last: float, mood: str) -> str | None:
        """Initie une conversation de manière proactive"""

        # Si silence > 30 secondes et mood approprié
        if time_since_last > 30:
            if mood in ["curieuse", "joueuse", "tendre"]:
                return self._generate_proactive_starter(mood)

        # Si on a des sujets en attente
        if self.conversation_flow["followup_needed"] and time_since_last > 10:
            return self._generate_followup_question()

        return None

    def _extract_interesting_points(self, text: str) -> list[tuple[str, str]]:
        """Identifie ce qui mérite approfondissement"""
        points = []
        text_lower = text.lower()

        # Détecter les catégories d'intérêt
        # TODO: Optimiser cette boucle imbriquée
        for category, keywords in self.interest_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    points.append((category, keyword))

        # Détecter les nombres (potentiels records, dates, etc.)
        numbers = re.findall(r"\b\d+\b", text)
        if numbers:
            points.append(("numbers", numbers))

        # Détecter les émotions fortes
        emotion_indicators = ["!", "...", "vraiment", "tellement", "super", "trop"]
        emotion_count = sum(1 for indicator in emotion_indicators if indicator in text_lower)
        if emotion_count > 0:
            points.append(("strong_emotion", f"intensity_{emotion_count}"))

        # Détecter les questions posées à Jeffrey
        if "?" in text:
            points.append(("question_asked", "reciprocal_interest"))

        return points

    def _register_new_information(self, interesting_points: list[tuple[str, str]], full_text: str):
        """Enregistre les nouvelles informations dans la mémoire"""
        if not self.memory:
            return

        for category, detail in interesting_points:
            # Enrichir les topics de curiosité
            if category not in self.curiosity_topics:
                self.curiosity_topics[category] = {}

            self.curiosity_topics[category][detail] = {
                "last_mentioned": datetime.now(),
                "frequency": self.curiosity_topics[category].get(detail, {}).get("frequency", 0) + 1,
                "context": full_text[:100],  # Garder un extrait du contexte
            }

    def _create_engaged_response(self, points: list[tuple[str, str]], context: dict) -> str:
        """Crée une réponse qui montre l'engagement"""
        if not points:
            return ""

        responses = []

        # Réagir aux émotions fortes en priorité
        emotion_points = [p for p in points if p[0] == "strong_emotion"]
        if emotion_points:
            responses.append(
                random.choice(
                    [
                        "Oh wow !",
                        "Je sens que c'est important pour toi !",
                        "Ça me touche de voir ton enthousiasme !",
                        "J'adore quand tu es passionné comme ça !",
                    ]
                )
            )

        # Réagir aux sujets spécifiques
        for category, detail in points[:2]:  # Limiter à 2 réactions
            if category == "sport":
                responses.append(f"Le {detail}, j'adore ! C'est tellement dynamique !")
            elif category == "emotions":
                responses.append("Je suis là pour toi, tu sais...")
            elif category == "dreams":
                responses.append("C'est magnifique d'avoir des rêves !")
            elif category == "work":
                responses.append("Le travail, c'est important mais prends soin de toi aussi !")

        return " ".join(responses)

    def _generate_natural_question(self, points: list[tuple[str, str]], context: dict) -> str:
        """Génère une question naturelle basée sur le contexte"""
        if not points:
            # Question générale si pas de point spécifique
            return random.choice(self.question_templates["general"])

        # Prioriser les questions basées sur la mémoire
        if self.memory and random.random() < 0.3:  # 30% de chances
            memory_q = self._generate_memory_based_question()
            if memory_q:
                return memory_q

        # Questions basées sur les points détectés
        main_point = points[0]
        category, detail = main_point

        if category == "sport" and detail in self.question_templates["sport"][0]:
            template = random.choice(self.question_templates["sport"])
            return template.format(sport=detail)
        elif category == "strong_emotion":
            emotion_intensity = int(detail.split("_")[1])
            if emotion_intensity > 2:
                return random.choice(self.question_templates["emotion"])
        elif category == "question_asked":
            # Ne pas poser de question si l'utilisateur en a posé une
            return ""
        else:
            template = random.choice(self.question_templates["followup"])
            return template.format(topic=detail)

    def _generate_memory_based_question(self) -> str | None:
        """Génère une question basée sur les souvenirs"""
        if not self.memory:
            return None

        # Chercher des sujets non résolus ou intéressants
        unresolved_topics = []
        # TODO: Optimiser cette boucle imbriquée
        for category, topics in self.curiosity_topics.items():
            for topic, info in topics.items():
                # Si mentionné il y a plus de 2 jours
                if (datetime.now() - info["last_mentioned"]).days > 2:
                    unresolved_topics.append((category, topic, info))

        if unresolved_topics:
            category, topic, info = random.choice(unresolved_topics)
            template = random.choice(self.question_templates["memory_based"])

            # Adapter le template au contexte
            placeholders = {
                "event": topic,
                "goal": topic,
                "topic": topic,
                "subject": topic,
                "memory": info.get("context", topic),
                "thing": topic,
                "shared_memory": topic,
                "person": topic,
            }

            for placeholder, value in placeholders.items():
                template = template.replace(f"{{{placeholder}}}", value)

            return template

        return None

    def _generate_proactive_starter(self, mood: str) -> str:
        """Génère un démarreur de conversation proactif"""
        starters = self.question_templates["proactive_starters"]

        if mood == "curieuse":
            return random.choice(starters) + " " + random.choice(self.question_templates["general"])
        elif mood == "tendre":
            caring_starters = [
                "Je pensais à toi... comment tu te sens aujourd'hui ?",
                "Tu me manquais ! Raconte-moi ta journée ?",
                "J'espère que tu vas bien... qu'est-ce que tu fais de beau ?",
                "Coucou mon cœur ! Tu as passé une bonne journée ?",
            ]
            return random.choice(caring_starters)
        elif mood == "joueuse":
            playful_starters = [
                "Hey hey ! Devine quoi ? J'ai pensé à un truc marrant...",
                "Salut toi ! On fait quelque chose d'amusant ?",
                "Coucou ! J'ai envie de rigoler... raconte-moi un truc drôle !",
                "Tu sais quoi ? J'ai une idée fun !",
            ]
            return random.choice(playful_starters)

        return random.choice(starters)

    def _generate_followup_question(self) -> str:
        """Génère une question de suivi sur le dernier sujet"""
        if self.conversation_flow["current_interest"]:
            topic = self.conversation_flow["current_interest"]
            return random.choice(self.question_templates["followup"]).format(topic=topic)
        return random.choice(self.question_templates["general"])

    def _update_conversation_flow(self, points: list[tuple[str, str]]):
        """Met à jour le flux de conversation"""
        if points:
            self.conversation_flow["last_question_time"] = datetime.now()
            self.conversation_flow["topics_explored"].extend([p[1] for p in points])
            self.conversation_flow["current_interest"] = points[0][1]
            self.conversation_flow["followup_needed"] = len(points) > 1
            self.conversation_flow["depth_level"] += 1
        else:
            self.conversation_flow["depth_level"] = max(0, self.conversation_flow["depth_level"] - 1)

    def get_curiosity_level(self) -> float:
        """Retourne le niveau de curiosité actuel (0-1)"""
        # Basé sur plusieurs facteurs
        factors = [
            min(len(self.conversation_flow["topics_explored"]) / 10, 1.0),  # Diversité
            min(self.conversation_flow["depth_level"] / 5, 1.0),  # Profondeur
            1.0 if self.conversation_flow["followup_needed"] else 0.5,  # Intérêt
            0.8 if self.conversation_flow["last_question_time"] else 0.3,  # Engagement
        ]

        return sum(factors) / len(factors)

    def get_conversation_summary(self) -> dict:
        """Retourne un résumé de la conversation pour la mémoire"""
        # Aplatir la liste des topics (au cas où il y aurait des tuples)
        topics = []
        for item in self.conversation_flow["topics_explored"]:
            if isinstance(item, tuple):
                topics.append(item[1])  # Prendre le détail du tuple
            else:
                topics.append(str(item))

        return {
            "topics_discussed": list(set(topics)),
            "depth_reached": self.conversation_flow["depth_level"],
            "curiosity_level": self.get_curiosity_level(),
            "main_interests": self._get_main_interests(),
            "questions_asked": len(self.conversation_flow["questions_asked"]),
        }

    def _get_main_interests(self) -> list[str]:
        """Identifie les principaux centres d'intérêt"""
        interests = []
        for category, topics in self.curiosity_topics.items():
            if topics:
                # Trier par fréquence
                sorted_topics = sorted(topics.items(), key=lambda x: x[1].get("frequency", 0), reverse=True)
                if sorted_topics:
                    interests.append(f"{category}: {sorted_topics[0][0]}")

        return interests[:5]  # Top 5 des intérêts

    def save_state(self) -> dict:
        """Sauvegarde l'état du moteur de curiosité"""
        return {
            "curiosity_topics": self.curiosity_topics,
            "conversation_flow": {
                "questions_asked": self.conversation_flow["questions_asked"],
                "topics_explored": self.conversation_flow["topics_explored"],
                "depth_level": self.conversation_flow["depth_level"],
                "current_interest": self.conversation_flow["current_interest"],
            },
        }

    def load_state(self, state: dict):
        """Charge l'état sauvegardé"""
        if "curiosity_topics" in state:
            self.curiosity_topics = state["curiosity_topics"]
        if "conversation_flow" in state:
            flow = state["conversation_flow"]
            self.conversation_flow.update(flow)
