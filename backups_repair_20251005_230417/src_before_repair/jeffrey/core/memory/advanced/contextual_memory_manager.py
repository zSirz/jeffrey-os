"""
Module de gestion de la mémoire contextuelle pour Jeffrey.

Ce module gère l'historique des conversations et le contexte émotionnel
pour permettre à Jeffrey de maintenir une cohérence dans ses interactions.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


class ContextualMemoryManager:
    """
    Classe ContextualMemoryManager pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    def __init__(self, storage_path: str = "data/conversation.json") -> None:
        """
        Initialise le gestionnaire de mémoire contextuelle.

        Args:
            storage_path: Chemin vers le fichier de stockage JSON
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.conversation_history: list[dict] = []
        self._load_history()

    def _load_history(self) -> None:
        """Charge l'historique depuis le fichier JSON."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, encoding="utf-8") as f:
                    self.conversation_history = json.load(f)
            except json.JSONDecodeError:
                # Fichier vide ou corrompu -> on repart sur un historique vide
                self.conversation_history = []
        else:
            # Crée le dossier si besoin et initialise l'historique
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            self.conversation_history = []

    def _save_history(self) -> None:
        """Sauvegarde l'historique dans le fichier JSON."""
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)

    def add_interaction(
        self,
        user_message: str,
        jeffrey_response: str,
        emotion: str,
        metadata: dict | None = None,
    ):
        """
        Ajoute une interaction à l'historique.

        Args:
            user_message: Message de l'utilisateur
            jeffrey_response: Réponse de Jeffrey
            emotion: Émotion détectée/exprimée
            metadata: Métadonnées supplémentaires (optionnel)
        """
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "jeffrey_response": jeffrey_response,
            "emotion": emotion,
            "metadata": metadata or {},
        }

        self.conversation_history.append(interaction)
        self._save_history()

    def get_recent_context(self, n_interactions: int = 5) -> list[dict]:
        """
        Récupère le contexte récent des n dernières interactions.

        Args:
            n_interactions: Nombre d'interactions à récupérer

        Returns:
            Liste des n dernières interactions
        """
        return self.conversation_history[-n_interactions:]

    def get_personality_summary(self) -> str:
        """
        Génère un résumé de la personnalité perçue de l'utilisateur.

        Returns:
            Résumé textuel de la personnalité
        """
        if not self.conversation_history:
            return "Pas encore assez d'interactions pour établir un profil."

        # Analyse des émotions dominantes
        emotions = {}
        topics = {}

        for interaction in self.conversation_history[-50:]:  # Derniers 50 échanges
            emotion = interaction.get("emotion", "neutre")
            emotions[emotion] = emotions.get(emotion, 0) + 1

            # Analyse simple des sujets (à améliorer)
            message = interaction["user_message"].lower()
            if "arkadia" in message:
                topics["arkadia"] = topics.get("arkadia", 0) + 1
            if "projet" in message:
                topics["projets"] = topics.get("projets", 0) + 1

        # Génération du résumé
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
        dominant_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:3]

        summary = f"Profil émotionnel : {dominant_emotion}\n"
        summary += "Centres d'intérêt principaux :\n"
        for topic, count in dominant_topics:
            summary += f"- {topic} ({count} mentions)\n"

        return summary

    def clear_history(self):
        """Efface tout l'historique des conversations."""
        self.conversation_history = []
        self._save_history()

    def generate_followup_question(self) -> str:
        """
        Génère une question de suivi basée sur l'historique des interactions.

        Returns:
            str: Une question de suivi contextuelle
        """
        if not self.conversation_history:
            return "Comment puis-je t'aider aujourd'hui ?"

        # Récupérer les dernières interactions
        recent_interactions = self.get_recent_context(n_interactions=3)

        # Analyser les émotions récentes
        recent_emotions = [interaction["emotion"] for interaction in recent_interactions]
        dominant_emotion = max(set(recent_emotions), key=recent_emotions.count)

        # Questions de suivi selon l'émotion dominante
        followup_questions = {
            "joyeux": [
                "Est-ce que tu veux partager ce qui te rend si heureux ?",
                "C'est agréable de te voir si joyeux ! Qu'est-ce qui t'a mis dans cet état d'esprit ?",
                "Ta bonne humeur est contagieuse ! Raconte-moi plus...",
            ],
            "triste": [
                "Je sens que quelque chose te tracasse... Veux-tu en parler ?",
                "Je suis là pour t'écouter si tu as besoin de te confier...",
                "Comment puis-je t'aider à te sentir mieux ?",
            ],
            "énervé": [
                "Je sens que quelque chose t'énerve... Veux-tu m'en parler ?",
                "Qu'est-ce qui t'a mis dans cet état ?",
                "Je suis là pour t'écouter si tu as besoin de te défouler...",
            ],
            "anxieux": [
                "Je sens que tu es un peu tendu... Est-ce que tout va bien ?",
                "Veux-tu parler de ce qui t'inquiète ?",
                "Comment puis-je t'aider à te sentir plus serein ?",
            ],
            "neutre": [
                "Comment puis-je t'aider à avancer ?",
                "Sur quoi aimerais-tu qu'on se concentre ?",
                "Qu'est-ce qui t'intéresse en ce moment ?",
            ],
            "surpris": [
                "Wow ! Qu'est-ce qui t'a tant surpris ?",
                "C'est intéressant ! Peux-tu m'en dire plus ?",
                "Je suis curieux d'en savoir plus sur ce qui t'a étonné...",
            ],
            "fatigué": [
                "Tu as l'air un peu fatigué... Est-ce que tu veux qu'on fasse une pause ?",
                "Comment puis-je t'aider à te reposer ?",
                "Veux-tu qu'on parle de quelque chose de plus léger ?",
            ],
        }

        # Sélectionner une question appropriée
        if dominant_emotion in followup_questions:
            return random.choice(followup_questions[dominant_emotion])

        # Question par défaut si l'émotion n'est pas reconnue
        default_questions = [
            "Comment puis-je t'aider à avancer ?",
            "Sur quoi aimerais-tu qu'on se concentre ?",
            "Qu'est-ce qui t'intéresse en ce moment ?",
            "Comment te sens-tu ?",
            "Que souhaites-tu explorer ?",
        ]

        return random.choice(default_questions)

    def extract_topics(self, message: str) -> list[str]:
        """
        Extrait des mots-clés simples du message pour les utiliser comme sujets.

        Args:
            message: Message texte de l'utilisateur

        Returns:
            Liste de topics (mots-clés potentiels)
        """
        # Nettoyage du message
        message = re.sub(r"[^\w\s]", "", message.lower())
        mots = message.split()

        # Liste de mots peu informatifs à ignorer
        stopwords = set(
            [
                "je",
                "tu",
                "il",
                "elle",
                "nous",
                "vous",
                "ils",
                "elles",
                "le",
                "la",
                "les",
                "un",
                "une",
                "des",
                "ce",
                "ces",
                "à",
                "de",
                "du",
                "dans",
                "sur",
                "avec",
                "et",
                "ou",
                "mais",
                "pour",
                "par",
                "mon",
                "ton",
                "ma",
                "ta",
                "mes",
                "tes",
                "se",
                "sa",
                "ses",
                "est",
                "suis",
                "était",
                "es",
                "était",
                "avoir",
                "être",
                "faire",
                "dis",
                "dit",
            ]
        )

        mots_significatifs = [mot for mot in mots if mot not in stopwords and len(mot) > 3]
        plus_frequents = [mot for mot, _ in Counter(mots_significatifs).most_common(3)]

        return plus_frequents
